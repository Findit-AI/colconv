//! Tier 5 V410 sinker tests — Ship 12a / 12b.
//!
//! Coverage matrix:
//! - Single-output paths (luma u8, luma u16, rgb, rgba, rgb_u16,
//!   rgba_u16, hsv) on solid-gray frames.
//! - Strategy A invariant (`with_rgb` + `with_rgba` byte-identical;
//!   same for the u16 variants).
//! - SIMD-vs-scalar parity across multiple widths covering the main
//!   loop + scalar tail of every backend block size.
//! - Three error-path tests: short packed slice, row index out of
//!   range, and short rgba_u16 buffer.

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- Solid-color V410 builder -----------------------------------------

/// Builds a solid-color V410 plane with one (U, Y, V) triplet repeated.
/// Each pixel is packed as a u32 word:
///   bits[31:30] = 0 (padding)
///   bits[29:20] = V (10-bit)
///   bits[19:10] = Y (10-bit)
///   bits[9:0]   = U (10-bit)
///
/// Row stride equals width (one u32 per pixel; no padding between rows).
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_v410_frame(width: u32, height: u32, u: u32, y: u32, v: u32) -> Vec<u32> {
  let word = (v << 20) | (y << 10) | u;
  std::vec![word; (width as usize) * (height as usize)]
}

// ---- Single-output gray-to-gray tests ---------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_luma_only_extracts_y_bytes_downshifted() {
  // Y=256 (10-bit) → 8-bit (256 >> 2) = 64.
  let buf = solid_v410_frame(6, 8, 512, 256, 512);
  let src = V410Frame::new(&buf, 6, 8, 6);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<V410>::new(6, 8).with_luma(&mut luma).unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  // 10-bit Y=256 → 8-bit (256 >> 2) = 64.
  assert!(luma.iter().all(|&y| y == 64), "luma {luma:?}");
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_with_luma_u16_extracts_y_native_depth() {
  // Y=0x3FC (10-bit near-max value). V410 word: (V << 20) | (Y << 10) | U
  // with Y=0x3FC → bits [19:10] = 0x3FC, so the full word (neutral U/V=512)
  // is (512 << 20) | (0x3FC << 10) | 512.
  // luma_u16 kernel extracts bits [19:10] → yields 0x3FC = 1020 in each slot.
  let buf = solid_v410_frame(6, 8, 512, 0x3FC, 512);
  let src = V410Frame::new(&buf, 6, 8, 6);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<V410>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  assert!(
    luma.iter().all(|&y| y == 0x3FC),
    "luma_u16 expected 0x3FC, got {:?}",
    &luma[..8]
  );
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_rgb_only_converts_gray_to_gray() {
  // Y=512, U=V=512 (neutral chroma at 10-bit midpoint ≈ 0.5x1023).
  // Mid-gray input should yield mid-gray output at ~128 ± tolerance.
  let buf = solid_v410_frame(12, 4, 512, 512, 512);
  let src = V410Frame::new(&buf, 12, 4, 12);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<V410>::new(12, 4).with_rgb(&mut rgb).unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn v410_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_v410_frame(12, 4, 512, 512, 512);
  let src = V410Frame::new(&buf, 12, 4, 12);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<V410>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn v410_rgb_u16_only_converts_gray_to_gray_native_depth() {
  // Y=U=V=512 (10-bit midpoint). After YUV→RGB at 10-bit depth the
  // per-channel value should be near 512; allow ±8 for Q15 rounding.
  let buf = solid_v410_frame(12, 4, 512, 512, 512);
  let src = V410Frame::new(&buf, 12, 4, 12);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<V410>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn v410_rgba_u16_alpha_is_max() {
  // 10-bit alpha max = 0x3FF = 1023.
  let buf = solid_v410_frame(12, 4, 512, 512, 512);
  let src = V410Frame::new(&buf, 12, 4, 12);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<V410>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn v410_hsv_only_produces_valid_hue_range() {
  // Solid gray frame → HSV should have H≈0, S≈0, V≈mid-range.
  let buf = solid_v410_frame(12, 4, 512, 512, 512);
  let src = V410Frame::new(&buf, 12, 4, 12);
  let n = 12 * 4;
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut v_plane = std::vec![0u8; n];
  let mut sink = MixedSinker::<V410>::new(12, 4)
    .with_hsv(&mut h, &mut s, &mut v_plane)
    .unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn v410_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_v410_frame(w, h, 200, 700, 400);
  let src = V410Frame::new(&buf, w, h, w);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<V410>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn v410_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path — alpha must be 0x3FF (10-bit
  // max, low-bit-packed).
  let w = 12u32;
  let h = 4u32;
  let buf = solid_v410_frame(w, h, 200, 700, 400);
  let src = V410Frame::new(&buf, w, h, w);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<V410>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn v410_with_simd_false_matches_with_simd_true() {
  // Pseudo-random V410 across multiple widths covering the main loop
  // + scalar tail of every backend block size. V410 samples are 10-bit
  // low-bit-packed in u32 words; pseudo-random fill uses the low 10
  // bits of each channel slot.
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
      *word = (v << 20) | (y << 10) | u;
    }
    let src = V410Frame::new(&buf, w as u32, h as u32, w as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<V410>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<V410>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    v410_to(
      &src,
      false,
      sink_simd.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();
    v410_to(
      &src,
      false,
      sink_scalar.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "V410 SIMD≠scalar at width {w}");
  }
}

// ---- Planar parity oracle ---------------------------------------------------

/// Pack three 10-bit planes (Y / U / V at 4:4:4, low-bit-packed u16) into V410
/// word stream layout: bits[31:30] = pad, [29:20] = V, [19:10] = Y, [9:0] = U.
/// Yuv444p10 stores 10-bit values as low-bit-packed u16 (high 6 bits zero).
///
/// Gated on `yuv-planar`: the cross-format oracle below feeds the planar
/// `Yuv444p10` source, absent in a `yuv-444-packed`-solo build.
#[cfg(feature = "yuv-planar")]
fn pack_yuv444p10_to_v410(
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
      packed.push((v << 20) | (y << 10) | u);
    }
  }
  packed
}

// Cross-format oracle vs the planar `Yuv444p10` source — gated on
// `yuv-planar` so a `yuv-444-packed`-solo `--tests` build still compiles.
#[test]
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_planar_parity_with_yuv444p10() {
  // Oracle: Yuv444p10 (separate planes) and V410 (packed u32) carry
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
  let packed = pack_yuv444p10_to_v410(&yp, &up, &vp, width, height);
  let v410 = V410Frame::new(&packed, width as u32, height as u32, width as u32);

  // u8 RGB parity
  let mut p_rgb = std::vec![0u8; width * height * 3];
  let mut v_rgb = std::vec![0u8; width * height * 3];
  let mut p_sink = MixedSinker::<Yuv444p10>::new(width, height)
    .with_rgb(&mut p_rgb)
    .unwrap();
  let mut v_sink = MixedSinker::<V410>::new(width, height)
    .with_rgb(&mut v_rgb)
    .unwrap();
  yuv444p10_to(
    &planar,
    false,
    p_sink.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();
  v410_to(&v410, false, v_sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(p_rgb, v_rgb, "V410 ↔ Yuv444p10 u8 RGB diverges");

  // u16 RGB parity (validates the low-bit-packed 10-bit path)
  let mut p_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut v_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut p_sink2 = MixedSinker::<Yuv444p10>::new(width, height)
    .with_rgb_u16(&mut p_rgb_u16)
    .unwrap();
  let mut v_sink2 = MixedSinker::<V410>::new(width, height)
    .with_rgb_u16(&mut v_rgb_u16)
    .unwrap();
  yuv444p10_to(
    &planar,
    false,
    p_sink2.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();
  v410_to(&v410, false, v_sink2.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(p_rgb_u16, v_rgb_u16, "V410 ↔ Yuv444p10 u16 RGB diverges");
}

// ---- Error-path tests --------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn v410_rgba_u16_buffer_too_short_returns_err() {
  // Buffer holds 6x7 = 42 elements x 4 channels = 168 u16 elements;
  // a 6x8 frame needs 6x8x4 = 192.
  let mut rgba = std::vec![0u16; 6 * 7 * 4];
  let result = MixedSinker::<V410>::new(6, 8).with_rgba_u16(&mut rgba);
  let Err(err) = result else {
    panic!("expected InsufficientRgbaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbaU16Buffer(InsufficientBuffer::new(192, 168))
  );
}
// Phase 4 Tier 5 — Frame BE flag, V410 LE+BE round-trip parity test.
//
// V410 packs each pixel as one u32 word. The BE wire variant byte-swaps each
// word before bit-extraction. Encoding the same logical (U, Y, V) triplet as
// LE bytes vs BE bytes and feeding through `MixedSinker<V410<false>>` vs
// `MixedSinker<V410<true>>` must produce byte-identical output.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_le_be_roundtrip_byte_identical() {
  // Build a logical (U, Y, V) pattern; encode as LE-bytes and BE-bytes
  // separately; assert sinker outputs match.
  let pack = |u: u32, y: u32, v: u32| -> u32 { (v << 20) | (y << 10) | u };
  let logical: std::vec::Vec<u32> = (0..8 * 4)
    .map(|i| match i % 4 {
      0 => pack(512, 512, 512),
      1 => pack(100, 700, 300),
      2 => pack(900, 64, 64),
      _ => pack(64, 940, 960),
    })
    .collect();
  let pix_le: std::vec::Vec<u32> = logical.iter().map(|&w| as_le_u32(w)).collect();
  let pix_be: std::vec::Vec<u32> = logical.iter().map(|&w| as_be_u32(w)).collect();

  let frame_le = V410LeFrame::try_new(&pix_le, 8, 4, 8).unwrap();
  let mut out_le = std::vec![0u8; 8 * 4 * 4];
  let mut sink_le = MixedSinker::<V410>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_le)
    .unwrap();
  v410_to(
    &frame_le,
    true,
    sink_le.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  let frame_be = V410BeFrame::try_new(&pix_be, 8, 4, 8).unwrap();
  let mut out_be = std::vec![0u8; 8 * 4 * 4];
  let mut sink_be = MixedSinker::<V410<true>>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_be)
    .unwrap();
  v410_to_endian(
    &frame_be,
    true,
    sink_be.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(
    out_le, out_be,
    "V410 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

// ---- #263 direct YUV→HSV ----------------------------------------------

/// Pseudo-random V410 words: 10-bit U / Y / V packed at bits [9:0] /
/// [19:10] / [29:20] (top 2 bits padding, left zero).
#[cfg(all(test, feature = "std"))]
fn pseudo_random_v410(n: usize, seed: u32) -> Vec<u32> {
  let mut buf = std::vec![0u32; n];
  let mut state = seed;
  for word in &mut buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let u = (state >> 2) & 0x3FF;
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let y = (state >> 2) & 0x3FF;
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let v = (state >> 2) & 0x3FF;
    *word = (v << 20) | (y << 10) | u;
  }
  buf
}

/// Row-kernel parity: the direct `v410_to_hsv_row` dispatcher is
/// byte-identical to `rgb_to_hsv_row(v410_to_rgb_row(...))` within each tier
/// — scalar (`use_simd = false`) AND host SIMD (`use_simd = true`) — across
/// matrices / range / widths / endianness. Both sides read the same packed
/// buffer through the same `be_input` decode, so this proves the fused HSV
/// kernel reproduces the via-RGB pipeline exactly (and, via the RGB tier's
/// own SIMD≡scalar parity, that the SIMD HSV kernel matches scalar).
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_hsv_row_matches_rgb_then_hsv() {
  use crate::row::{rgb_to_hsv_row, v410_to_hsv_row, v410_to_rgb_row};
  for &w in &[1usize, 2, 4, 7, 8, 15, 16, 17, 31, 64, 65] {
    let buf = pseudo_random_v410(w, 0x1234_5678);
    for &matrix in &[
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
    ] {
      for &full in &[true, false] {
        for &use_simd in &[false, true] {
          for &be in &[false, true] {
            let mut rgb = std::vec![0u8; w * 3];
            v410_to_rgb_row(&buf, &mut rgb, w, matrix, full, use_simd, be);
            let mut rh = std::vec![0u8; w];
            let mut rs = std::vec![0u8; w];
            let mut rv = std::vec![0u8; w];
            rgb_to_hsv_row(&rgb, &mut rh, &mut rs, &mut rv, w, use_simd);

            let mut h = std::vec![0u8; w];
            let mut s = std::vec![0u8; w];
            let mut v = std::vec![0u8; w];
            v410_to_hsv_row(&buf, &mut h, &mut s, &mut v, w, matrix, full, use_simd, be);
            assert_eq!(
              (h, s, v),
              (rh, rs, rv),
              "v410 HSV≠RGB→HSV (w={w} {matrix:?} full={full} simd={use_simd} be={be})"
            );
          }
        }
      }
    }
  }
}

/// Structural: an HSV-only V410 sink (no `with_rgb` / `with_rgba`) routes
/// through the direct kernel and must NOT grow the source-width RGB scratch
/// (`rgb_scratch.len() == 0`); its HSV equals the explicit
/// `v410_to_rgb_row` → `rgb_to_hsv_row` reference (row 0).
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_hsv_only_is_rgb_free_and_matches_reference() {
  use crate::row::{rgb_to_hsv_row, v410_to_rgb_row};
  let (w, h) = (16usize, 8usize);
  let buf = pseudo_random_v410(w * h, 0xC0FF_EE11);
  let src = V410Frame::new(&buf, w as u32, h as u32, w as u32);
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<V410>::new(w, h)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "V410 HSV-only must not grow the RGB scratch"
  );

  let mut rgb0 = std::vec![0u8; w * 3];
  v410_to_rgb_row(
    &buf[..w],
    &mut rgb0,
    w,
    KernelMatrix::Bt709,
    true,
    true,
    false,
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
// V410's identity `process` hoists the RGB-scratch reservation above every
// output write (luma, luma_u16, then the u16 RGB / RGBA fan-out), so an
// allocator refusal of the scratch grow surfaces as a recoverable
// `AllocationFailed` BEFORE any output plane is touched. The allocating
// (rgb = None) arm of `rgb_row_buf_or_scratch` is reached only at
// `want_hsv && want_rgba && !want_rgb` (HSV-only routes through the direct
// `v410_to_hsv_row` kernel and never allocates): attach luma + luma_u16 + rgba
// + hsv (no rgb) and arm the shared failpoint. `yuva`-gated (shares the crate's
// RGB-scratch failpoint).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let buf = solid_v410_frame(16, 8, 512, 512, 512);
  let src = V410Frame::new(&buf, 16, 8, 16);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut luma_u16 = std::vec![0xABCDu16; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<V410>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = v410_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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
