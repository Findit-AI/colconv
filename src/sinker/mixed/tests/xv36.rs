//! Tier 5 XV36 sinker tests — Ship 12b.
//!
//! Coverage matrix:
//! - Single-output paths (rgb, rgba, rgb_u16, rgba_u16, luma u8,
//!   luma u16) on solid-gray frames.
//! - Strategy A invariant (`with_rgb` + `with_rgba` byte-identical;
//!   same for the u16 variants with BITS=12).
//! - SIMD-vs-scalar parity across multiple widths covering the main
//!   loop + scalar tail of every backend block size.
//! - Three error-path tests: row index out of range, short packed
//!   slice, and short rgba_u16 buffer.
//!
//! XV36 specifics vs V410:
//! - Source buffer is `&[u16]` (4 x u16 per pixel), not `&[u32]`.
//! - Channels are 12-bit MSB-aligned (low 4 bits zero per sample).
//! - `with_luma_u16` is supported natively (Y >> 4 → low-bit-packed
//!   12-bit value).
//! - u16 alpha max = `0x0FFF` (12-bit max, not 0x3FF / 0xFFFF).

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- Solid-color XV36 builder -----------------------------------------

/// Builds a solid-color XV36 plane with one (U, Y, V, A) quadruple
/// repeated. Each pixel is stored as four u16 words:
///   word 0: U  (12-bit, MSB-aligned: `u_val << 4`)
///   word 1: Y  (12-bit, MSB-aligned: `y_val << 4`)
///   word 2: V  (12-bit, MSB-aligned: `v_val << 4`)
///   word 3: A  (12-bit, MSB-aligned: `a_val << 4`, padding)
///
/// The parameters `u`, `y`, `v`, `a` are raw 12-bit values (`[0, 4095]`);
/// they are left-shifted by 4 internally to produce the MSB-aligned
/// representation stored in the buffer.
///
/// Row stride equals `width x 4` u16 elements (no padding between rows).
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_xv36_frame(
  width: u32,
  height: u32,
  u: u16,
  y: u16,
  v: u16,
  a: u16,
) -> Vec<u16> {
  debug_assert!(u <= 0xFFF && y <= 0xFFF && v <= 0xFFF && a <= 0xFFF);
  let quad = [u << 4, y << 4, v << 4, a << 4];
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
fn xv36_rgb_only_converts_gray_to_gray() {
  // Y=U=V=0x800 (12-bit midpoint ≈ 0.5x4095). After YUV→RGB at 12-bit
  // depth the per-channel u8 value should be near 128 ± tolerance.
  let buf = solid_xv36_frame(12, 4, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 12, 4, 48);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<Xv36>::new(12, 4).with_rgb(&mut rgb).unwrap();
  xv36_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
fn xv36_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  // Alpha must be forced to 0xFF (XV36 A slot is padding).
  let buf = solid_xv36_frame(12, 4, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 12, 4, 48);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<Xv36>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  xv36_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF, "alpha must be 0xFF for XV36");
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_rgb_u16_only_converts_gray_to_gray_native_depth() {
  // Y=U=V=0x800 (12-bit midpoint = 2048). After YUV→RGB at 12-bit depth
  // the per-channel u16 value should be near 2048; allow ±0x10 for Q15
  // rounding.
  let buf = solid_xv36_frame(12, 4, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 12, 4, 48);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<Xv36>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  xv36_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(
      px[0].abs_diff(0x800) <= 0x10,
      "expected ~0x800, got {:#X}",
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
fn xv36_rgba_u16_alpha_is_max() {
  // 12-bit alpha max = 0x0FFF = 4095 (NOT 0x3FF or 0xFFFF).
  let buf = solid_xv36_frame(12, 4, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 12, 4, 48);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<Xv36>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  xv36_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0x0FFF, "u16 alpha must be 0x0FFF for XV36");
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_luma_only_extracts_y_bytes_downshifted() {
  // Y=0xF00 (12-bit value MSB-aligned: 0xF00 << 4 = 0xF000 in u16).
  // u8 luma: MSB-aligned u16 >> 8 = 0xF000 >> 8 = 0xF0 = 240.
  let buf = solid_xv36_frame(6, 8, 0x800, 0xF00, 0x800, 0);
  let src = Xv36Frame::new(&buf, 6, 8, 24);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<Xv36>::new(6, 8).with_luma(&mut luma).unwrap();
  xv36_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  // 0xF00 << 4 = 0xF000; 0xF000 >> 8 = 0xF0 = 240.
  assert!(luma.iter().all(|&y| y == 0xF0), "luma {luma:?}");
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_luma_u16_only_extracts_y_native_depth() {
  // Y=0xABC (12-bit value). MSB-aligned u16 value: 0xABC << 4 = 0xABC0.
  // u16 luma_u16: MSB-aligned >> 4 = 0xABC0 >> 4 = 0xABC = 2748.
  let buf = solid_xv36_frame(6, 8, 0x800, 0xABC, 0x800, 0);
  let src = Xv36Frame::new(&buf, 6, 8, 24);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<Xv36>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  xv36_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  // 0xABC0 >> 4 = 0xABC.
  assert!(
    luma.iter().all(|&y| y == 0xABC),
    "luma_u16 expected 0xABC, got {:?}",
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
fn xv36_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_xv36_frame(w, h, 0x300, 0x700, 0x500, 0);
  let src = Xv36Frame::new(&buf, w, h, w * 4);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Xv36>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  xv36_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
fn xv36_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path — alpha must be 0x0FFF (12-bit
  // max, low-bit-packed).
  let w = 12u32;
  let h = 4u32;
  let buf = solid_xv36_frame(w, h, 0x300, 0x700, 0x500, 0);
  let src = Xv36Frame::new(&buf, w, h, w * 4);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Xv36>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  xv36_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
    assert_eq!(rgba[i * 4 + 3], 0x0FFF, "alpha must be 0x0FFF at pixel {i}");
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_with_simd_false_matches_with_simd_true() {
  // Pseudo-random XV36 across multiple widths covering the main loop
  // + scalar tail of every backend block size. XV36 samples are 12-bit
  // MSB-aligned in u16 (low 4 bits zero); pseudo-random fill uses the
  // low 12 bits of each channel slot, then shifts left by 4.
  for w in [1usize, 2, 4, 7, 8, 15, 16, 17, 31, 32, 33, 1920, 1922] {
    let h = 2usize;
    let mut buf = std::vec![0u16; w * h * 4];
    // Fill each u16 slot with a pseudo-random 12-bit value, MSB-aligned.
    let mut state = 0xC0FFEE_u32;
    for slot in &mut buf {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let val = ((state >> 4) as u16) & 0xFFF0; // 12-bit, MSB-aligned
      *slot = val;
    }
    let src = Xv36Frame::new(&buf, w as u32, h as u32, (w * 4) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Xv36>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Xv36>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    xv36_to(&src, false, KernelMatrix::Bt709, &mut sink_simd).unwrap();
    xv36_to(&src, false, KernelMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "XV36 SIMD≠scalar at width {w}");
  }
}

// ---- Planar parity oracle ---------------------------------------------------

/// Pack three 12-bit planes (Y / U / V at 4:4:4, low-bit-packed u16) into
/// XV36 quadruple layout: `[u << 4, y << 4, v << 4, 0]` per pixel
/// (MSB-aligned at 12-bit, low 4 bits zero).
/// Yuv444p12 stores 12-bit values low-bit-packed (high 4 bits zero).
/// XV36 stores them MSB-aligned (low 4 bits zero). Convert via `<< 4`.
///
/// Gated on `yuv-planar`: the cross-format oracle below feeds the planar
/// `Yuv444p12` source, absent in a `yuv-444-packed`-solo build.
#[cfg(feature = "yuv-planar")]
fn pack_yuv444p12_to_xv36(
  y_plane: &[u16],
  u_plane: &[u16],
  v_plane: &[u16],
  width: usize,
  height: usize,
) -> Vec<u16> {
  let mut packed = Vec::with_capacity(width * 4 * height);
  for r in 0..height {
    for c in 0..width {
      let y = (y_plane[r * width + c] & 0x0FFF) << 4;
      let u = (u_plane[r * width + c] & 0x0FFF) << 4;
      let v = (v_plane[r * width + c] & 0x0FFF) << 4;
      packed.push(u);
      packed.push(y);
      packed.push(v);
      packed.push(0); // padding A
    }
  }
  packed
}

// Cross-format oracle vs the planar `Yuv444p12` source — gated on
// `yuv-planar` so a `yuv-444-packed`-solo `--tests` build (which lacks the
// planar frame / walker) still compiles.
#[test]
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_planar_parity_with_yuv444p12() {
  // Oracle: Yuv444p12 (separate planes) and XV36 (packed u16 quadruples)
  // carry identical logical 12-bit samples — both paths MUST produce
  // byte-identical RGB output (u8 and u16).
  let width = 16usize;
  let height = 4usize;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; width * height];
  let mut vp = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC0FFEE, 12);
  pseudo_random_u16_low_n_bits(&mut up, 0xBADF00D, 12);
  pseudo_random_u16_low_n_bits(&mut vp, 0xFEEDFACE, 12);

  let planar = Yuv444p12Frame::new(
    &yp,
    &up,
    &vp,
    width as u32,
    height as u32,
    width as u32,
    width as u32,
    width as u32,
  );
  let packed = pack_yuv444p12_to_xv36(&yp, &up, &vp, width, height);
  let xv36 = Xv36Frame::new(&packed, width as u32, height as u32, (width * 4) as u32);

  // u8 RGB parity
  let mut p_rgb = std::vec![0u8; width * height * 3];
  let mut x_rgb = std::vec![0u8; width * height * 3];
  let mut p_sink = MixedSinker::<Yuv444p12>::new(width, height)
    .with_rgb(&mut p_rgb)
    .unwrap();
  let mut x_sink = MixedSinker::<Xv36>::new(width, height)
    .with_rgb(&mut x_rgb)
    .unwrap();
  yuv444p12_to(&planar, false, KernelMatrix::Bt709, &mut p_sink).unwrap();
  xv36_to(&xv36, false, KernelMatrix::Bt709, &mut x_sink).unwrap();
  assert_eq!(p_rgb, x_rgb, "XV36 ↔ Yuv444p12 u8 RGB diverges");

  // u16 RGB parity (validates the BITS=12 path against the established
  // planar reference)
  let mut p_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut x_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut p_sink2 = MixedSinker::<Yuv444p12>::new(width, height)
    .with_rgb_u16(&mut p_rgb_u16)
    .unwrap();
  let mut x_sink2 = MixedSinker::<Xv36>::new(width, height)
    .with_rgb_u16(&mut x_rgb_u16)
    .unwrap();
  yuv444p12_to(&planar, false, KernelMatrix::Bt709, &mut p_sink2).unwrap();
  xv36_to(&xv36, false, KernelMatrix::Bt709, &mut x_sink2).unwrap();
  assert_eq!(p_rgb_u16, x_rgb_u16, "XV36 ↔ Yuv444p12 u16 RGB diverges");
}

// ---- Error-path tests --------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn xv36_buffer_too_short_for_rgba_u16_returns_err() {
  // Buffer holds 6x7 = 42 elements x 4 channels = 168 u16 elements;
  // a 6x8 frame needs 6x8x4 = 192.
  let mut rgba = std::vec![0u16; 6 * 7 * 4];
  let result = MixedSinker::<Xv36>::new(6, 8).with_rgba_u16(&mut rgba);
  let Err(err) = result else {
    panic!("expected InsufficientRgbaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbaU16Buffer(InsufficientBuffer::new(192, 168))
  );
}
// Phase 4 Tier 5 — Frame BE flag, XV36 LE+BE round-trip parity test.
//
// XV36 packs each pixel as four u16 channels (`U, Y, V, A`), 12-bit MSB-aligned.
// The BE wire variant byte-swaps each u16 channel before extraction. Encoding
// the same logical samples as LE bytes vs BE bytes and feeding through
// `MixedSinker<Xv36<false>>` vs `MixedSinker<Xv36<true>>` must produce
// byte-identical output.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_le_be_roundtrip_byte_identical() {
  // Build a logical (U, Y, V, A) MSB-aligned u16 quadruple stream.
  let logical: std::vec::Vec<u16> = (0..8 * 4 * 4)
    .map(|i| match i % 4 {
      0 => 0x0800u16, // U = 0x080 << 4
      1 => 0x4000u16, // Y = 0x400 << 4
      2 => 0xA000u16, // V = 0xA00 << 4
      _ => 0x0000u16, // A = padding
    })
    .collect();
  let pix_le: std::vec::Vec<u16> = logical.iter().map(|&v| as_le_u16(v)).collect();
  let pix_be: std::vec::Vec<u16> = logical.iter().map(|&v| as_be_u16(v)).collect();

  let frame_le = Xv36LeFrame::try_new(&pix_le, 8, 4, 8 * 4).unwrap();
  let mut out_le = std::vec![0u8; 8 * 4 * 4];
  let mut sink_le = MixedSinker::<Xv36>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_le)
    .unwrap();
  xv36_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Xv36BeFrame::try_new(&pix_be, 8, 4, 8 * 4).unwrap();
  let mut out_be = std::vec![0u8; 8 * 4 * 4];
  let mut sink_be = MixedSinker::<Xv36<true>>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_be)
    .unwrap();
  xv36_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le, out_be,
    "Xv36 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

// ---- Atomicity (#308): RGB-scratch preflight before output writes -------
//
// Xv36's identity `process` hoists the RGB-scratch reservation above every
// output write (luma, luma_u16, then the u16 RGB / RGBA fan-out), so an
// allocator refusal of the scratch grow surfaces as a recoverable
// `AllocationFailed` BEFORE any output plane is touched. The allocating
// (rgb = None) arm of `rgb_row_buf_or_scratch` is reached only at
// `want_hsv && want_rgba && !want_rgb` (HSV-only routes through the direct
// `xv36_to_hsv_row` kernel and never allocates): attach luma + luma_u16 + rgba
// + hsv (no rgb) and arm the shared failpoint. `yuva`-gated (shares the crate's
// RGB-scratch failpoint).
#[cfg(all(test, feature = "std", feature = "yuva"))]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let buf = solid_xv36_frame(16, 8, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 16, 8, 64);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut luma_u16 = std::vec![0xABCDu16; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Xv36>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = xv36_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
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
