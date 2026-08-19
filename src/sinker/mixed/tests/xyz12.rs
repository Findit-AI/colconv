use super::*;
use crate::KernelGamut;

// ---- Tier 12 — Xyz12 packed-CIE-XYZ-12-bit source family --------------

/// Tolerance for f32 comparisons. The OETF + matmul + f32 narrow chain
/// adds about 4e-7 of noise; 4e-6 is comfortably above any platform
/// drift. Matches the spec's `EPSILON_F32 = 4e-6` constant.
const EPSILON_F32: f32 = 4e-6;

/// Encodes a 12-bit code into the high-bit-packed LE wire `u16`
/// (`code << 4`, then LE-bytes reinterpreted as host-native `u16`).
/// Per FFmpeg `AV_PIX_FMT_XYZ12LE`: active 12 bits in `[15:4]`, low 4
/// bits zero. Host-independent: same logical wire value on LE and BE
/// hosts.
#[cfg_attr(not(tarpaulin), inline(always))]
fn pack12_le(code: u16) -> u16 {
  u16::from_ne_bytes((code << 4).to_le_bytes())
}

/// Encodes a 12-bit code into the high-bit-packed BE wire `u16` —
/// `(code << 4).to_be_bytes()` reinterpreted as host-native. Same
/// logical wire value on LE and BE hosts; the kernel's `from_be`
/// recovers the LE-encoded form internally.
#[cfg_attr(not(tarpaulin), inline(always))]
fn pack12_be(code: u16) -> u16 {
  u16::from_ne_bytes((code << 4).to_be_bytes())
}

/// Bit-pattern equality for `f32` slices — elementwise `to_bits()`
/// comparison via `iter().zip()` (no temporary `Vec` allocations).
#[cfg_attr(not(tarpaulin), inline(always))]
fn bits_eq_f32(a: &[f32], b: &[f32]) -> bool {
  a.len() == b.len()
    && a
      .iter()
      .zip(b.iter())
      .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Bit-pattern equality for `half::f16` slices — same allocation-free
/// pattern as [`bits_eq_f32`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn bits_eq_f16(a: &[half::f16], b: &[half::f16]) -> bool {
  a.len() == b.len()
    && a
      .iter()
      .zip(b.iter())
      .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Builds a tightly packed Xyz12LE frame with a constant 12-bit
/// `(X, Y, Z)` triple. Inputs are 12-bit codes; the helper applies the
/// high-bit packing. Allocation is exactly `w * h * 3` `u16` elements
/// and the frame uses stride `3 * width`, so no per-row padding bytes
/// are present (row-stride coverage lives in other tests).
fn solid_xyz12_frame_le(width: u32, height: u32, x: u16, y: u16, z: u16) -> Vec<u16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u16; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = pack12_le(x);
    px[1] = pack12_le(y);
    px[2] = pack12_le(z);
  }
  buf
}

/// Builds the BE-wire variant of `solid_xyz12_frame_le` for the same
/// 12-bit codes (host-independent via `pack12_be`). Same tightly packed
/// allocation: `w * h * 3` `u16` elements with stride `3 * width`.
fn solid_xyz12_frame_be(width: u32, height: u32, x: u16, y: u16, z: u16) -> Vec<u16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u16; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = pack12_be(x);
    px[1] = pack12_be(y);
    px[2] = pack12_be(z);
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_rgb_zero_input_zero_output() {
  let pix = solid_xyz12_frame_le(8, 4, 0, 0, 0);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 8 * 4 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [0, 0, 0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_rgb_max_input_clamps_to_255() {
  // (4095, 4095, 4095) under DCI-P3 (DCI white) →
  // linear ~(1.383, 1.001, 1.151) → after clamp [0,1] + OETF + x255
  // → all channels saturate to 255.
  let pix = solid_xyz12_frame_le(8, 4, 0xFFF, 0xFFF, 0xFFF);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 8 * 4 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 255]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_rgba_fills_alpha_max_u8() {
  let pix = solid_xyz12_frame_le(8, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgba_out = std::vec![0u8; 8 * 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFF, "alpha");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_rgb_u16_full_range_at_max() {
  let pix = solid_xyz12_frame_le(8, 4, 0xFFF, 0xFFF, 0xFFF);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgb_out = std::vec![0u16; 8 * 4 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgb_u16(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [65535, 65535, 65535]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_rgba_u16_fills_alpha_max() {
  let pix = solid_xyz12_frame_le(8, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgba_out = std::vec![0u16; 8 * 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgba_u16(&mut rgba_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::Rec709)).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFFFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_rgb_f32_preserves_negatives() {
  // y_only_max under Rec.709 → R = -1.677, G = +2.05, B = -0.222.
  // f32 path skips clamp, so negatives + > 1 values are preserved.
  let pix = solid_xyz12_frame_le(4, 2, 0, 0xFFF, 0);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgb_out = std::vec![0.0f32; 4 * 2 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_rgb_f32(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::Rec709)).unwrap();

  for px in rgb_out.chunks(3) {
    assert!(px[0] < 0.0, "R should be negative, got {}", px[0]);
    assert!(px[1] > 1.0, "G should be > 1, got {}", px[1]);
    assert!(px[2] < 0.0, "B should be negative, got {}", px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_xyz_f32_lossless_passthrough() {
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut xyz_out = std::vec![0.0f32; 4 * 2 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_xyz_f32(&mut xyz_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();

  // Linear XYZ pass-through: only step-1 inverse OETF applied. All
  // three channels should have identical values for an input of
  // (0x800, 0x800, 0x800).
  for px in xyz_out.chunks(3) {
    assert!((px[0] - px[1]).abs() < EPSILON_F32);
    assert!((px[1] - px[2]).abs() < EPSILON_F32);
    // Should be > 0 and bounded by 1 / 0.91653 ≈ 1.091.
    assert!(px[0] > 0.0);
    assert!(px[0] < 1.1);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_rgb_f16_rec709_mid_gray() {
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgb_out = std::vec![half::f16::ZERO; 4 * 2 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_rgb_f16(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::Rec709)).unwrap();

  for px in rgb_out.chunks(3) {
    // Each channel should be a positive value in (0, 1) since OETF +
    // clamp [0, 1] applies.
    for c in px {
      let v = c.to_f32();
      assert!((0.0..=1.0).contains(&v), "f16 channel out of [0,1]: {v}");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_rgba_f16_alpha_one() {
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgba_out = std::vec![half::f16::ZERO; 4 * 2 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_rgba_f16(&mut rgba_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px[3].to_f32(), 1.0);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_luma_via_staging() {
  let pix = solid_xyz12_frame_le(4, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 4, 4 * 3).unwrap();

  let mut luma_out = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 4)
    .with_luma(&mut luma_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::Rec709)).unwrap();

  // Mid-gray under Rec.709 → grayish output (around 113-128 u8).
  for &y in &luma_out {
    assert!(y > 0, "expected non-zero luma");
    assert!(y < 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_luma_u16_zero_extends() {
  let pix = solid_xyz12_frame_le(4, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 4, 4 * 3).unwrap();

  let mut luma_out = std::vec![0u16; 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 4)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::Rec709)).unwrap();

  for &y in &luma_out {
    // Same value as `with_luma` (u8) but zero-extended into u16.
    assert!(y > 0);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_with_hsv_via_staging() {
  let pix = solid_xyz12_frame_le(4, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 4, 4 * 3).unwrap();

  let mut h_out = std::vec![0u8; 4 * 4];
  let mut s_out = std::vec![0u8; 4 * 4];
  let mut v_out = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 4)
    .with_hsv(&mut h_out, &mut s_out, &mut v_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();

  // Mid-gray → low saturation. Hue undefined for grayscale but should
  // not panic.
  for &v in &v_out {
    assert!(v > 0);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_target_gamut_changes_output() {
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut out_p3 = std::vec![0u8; 4 * 2 * 3];
  let mut out_709 = std::vec![0u8; 4 * 2 * 3];
  let mut out_2020 = std::vec![0u8; 4 * 2 * 3];
  {
    let mut sink_p3 = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_p3)
      .unwrap();
    xyz12_to(&src, sink_p3.set_kernel_gamut(KernelGamut::DciP3)).unwrap();
  }
  {
    let mut sink_709 = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_709)
      .unwrap();
    xyz12_to(&src, sink_709.set_kernel_gamut(KernelGamut::Rec709)).unwrap();
  }
  {
    let mut sink_2020 = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_2020)
      .unwrap();
    xyz12_to(&src, sink_2020.set_kernel_gamut(KernelGamut::Rec2020)).unwrap();
  }

  // The three gamut matrices differ at the second-decimal level, so
  // the resulting u8 outputs must differ on at least one channel.
  assert_ne!(out_p3, out_709, "DCI-P3 vs Rec.709");
  assert_ne!(out_p3, out_2020, "DCI-P3 vs Rec.2020");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_be_byte_swap_matches_le() {
  // BE input that is the byte-swapped LE encoding of the same
  // logical sample value should produce identical output.
  let pix_le = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let pix_be = solid_xyz12_frame_be(4, 2, 0x800, 0x800, 0x800);
  let src_le = Xyz12LeFrame::try_new(&pix_le, 4, 2, 4 * 3).unwrap();
  let src_be = Xyz12BeFrame::try_new(&pix_be, 4, 2, 4 * 3).unwrap();

  let mut out_le = std::vec![0u8; 4 * 2 * 3];
  let mut out_be = std::vec![0u8; 4 * 2 * 3];
  {
    let mut sink_le = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_le)
      .unwrap();
    xyz12_to(&src_le, sink_le.set_kernel_gamut(KernelGamut::DciP3)).unwrap();
  }
  {
    let mut sink_be = MixedSinker::<Xyz12Be>::new(4, 2)
      .with_rgb(&mut out_be)
      .unwrap();
    xyz12_to(&src_be, sink_be.set_kernel_gamut(KernelGamut::DciP3)).unwrap();
  }
  assert_eq!(out_le, out_be);
}

/// Builds a tightly packed Xyz12 frame with a varying-pattern of 12-bit
/// `(X, Y, Z)` codes — exercises every byte-position so any byte-swap
/// regression in the BE path surfaces. Allocation is exactly `w * h * 3`
/// `u16` elements with stride `3 * width` (no row padding). `pack`
/// chooses the wire-format (high-bit-packed LE or BE).
fn pattern_xyz12_frame(width: u32, height: u32, pack: fn(u16) -> u16) -> std::vec::Vec<u16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u16; w * h * 3];
  // 12-bit codes spread across the active range, including extremes.
  // The 6-code cycle has length coprime with 3 (the X/Y/Z stride), so
  // the mapping of (code, channel) advances by one step per pixel —
  // every (channel, code) pair appears within the first six pixels.
  // That suffices to surface an X/Y/Z swap regression even though each
  // channel does not get its own private sequence.
  let codes: [u16; 6] = [0x000, 0x123, 0x456, 0x789, 0xABC, 0xFFF];
  for (i, c) in buf.iter_mut().enumerate() {
    *c = pack(codes[i % codes.len()]);
  }
  buf
}

/// Runs the full LE/BE round-trip across every Xyz12 sinker output
/// path under the given `simd` flag and `target_gamut`. All paths
/// must be byte-identical between the LE and BE encodings of the
/// same logical 12-bit codes — proves `<const BE>` propagates from
/// frame → walker → row marker → sinker → row kernel for every
/// output kernel (rgb, rgba, rgb_u16, rgba_u16, rgb_f32, rgb_f16,
/// rgba_f16, luma, luma_u16, hsv, xyz_f32 — staging-shared paths
/// included).
#[allow(clippy::too_many_lines)]
fn assert_xyz12_le_be_roundtrip_all_outputs(
  width: u32,
  height: u32,
  target_gamut: KernelGamut,
  simd: bool,
) {
  let w = width as usize;
  let h = height as usize;
  let pix_le = pattern_xyz12_frame(width, height, pack12_le);
  let pix_be = pattern_xyz12_frame(width, height, pack12_be);
  let src_le = Xyz12LeFrame::try_new(&pix_le, width, height, 3 * width).unwrap();
  let src_be = Xyz12BeFrame::try_new(&pix_be, width, height, 3 * width).unwrap();

  let mut le_rgb = std::vec![0u8; w * h * 3];
  let mut le_rgba = std::vec![0u8; w * h * 4];
  let mut le_rgb_u16 = std::vec![0u16; w * h * 3];
  let mut le_rgba_u16 = std::vec![0u16; w * h * 4];
  let mut le_rgb_f32 = std::vec![0.0f32; w * h * 3];
  let mut le_xyz_f32 = std::vec![0.0f32; w * h * 3];
  let mut le_rgb_f16 = std::vec![half::f16::ZERO; w * h * 3];
  let mut le_rgba_f16 = std::vec![half::f16::ZERO; w * h * 4];
  let mut le_luma = std::vec![0u8; w * h];
  let mut le_luma_u16 = std::vec![0u16; w * h];
  let mut le_h = std::vec![0u8; w * h];
  let mut le_s = std::vec![0u8; w * h];
  let mut le_v = std::vec![0u8; w * h];

  let mut be_rgb = std::vec![0u8; w * h * 3];
  let mut be_rgba = std::vec![0u8; w * h * 4];
  let mut be_rgb_u16 = std::vec![0u16; w * h * 3];
  let mut be_rgba_u16 = std::vec![0u16; w * h * 4];
  let mut be_rgb_f32 = std::vec![0.0f32; w * h * 3];
  let mut be_xyz_f32 = std::vec![0.0f32; w * h * 3];
  let mut be_rgb_f16 = std::vec![half::f16::ZERO; w * h * 3];
  let mut be_rgba_f16 = std::vec![half::f16::ZERO; w * h * 4];
  let mut be_luma = std::vec![0u8; w * h];
  let mut be_luma_u16 = std::vec![0u16; w * h];
  let mut be_h = std::vec![0u8; w * h];
  let mut be_s = std::vec![0u8; w * h];
  let mut be_v = std::vec![0u8; w * h];

  {
    let mut sink_le = MixedSinker::<Xyz12Le>::new(w, h)
      .with_simd(simd)
      .with_rgb(&mut le_rgb)
      .unwrap()
      .with_rgba(&mut le_rgba)
      .unwrap()
      .with_rgb_u16(&mut le_rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut le_rgba_u16)
      .unwrap()
      .with_rgb_f32(&mut le_rgb_f32)
      .unwrap()
      .with_xyz_f32(&mut le_xyz_f32)
      .unwrap()
      .with_rgb_f16(&mut le_rgb_f16)
      .unwrap()
      .with_rgba_f16(&mut le_rgba_f16)
      .unwrap()
      .with_luma(&mut le_luma)
      .unwrap()
      .with_luma_u16(&mut le_luma_u16)
      .unwrap()
      .with_hsv(&mut le_h, &mut le_s, &mut le_v)
      .unwrap();
    xyz12_to(&src_le, sink_le.set_kernel_gamut(target_gamut)).unwrap();
  }
  {
    let mut sink_be = MixedSinker::<Xyz12Be>::new(w, h)
      .with_simd(simd)
      .with_rgb(&mut be_rgb)
      .unwrap()
      .with_rgba(&mut be_rgba)
      .unwrap()
      .with_rgb_u16(&mut be_rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut be_rgba_u16)
      .unwrap()
      .with_rgb_f32(&mut be_rgb_f32)
      .unwrap()
      .with_xyz_f32(&mut be_xyz_f32)
      .unwrap()
      .with_rgb_f16(&mut be_rgb_f16)
      .unwrap()
      .with_rgba_f16(&mut be_rgba_f16)
      .unwrap()
      .with_luma(&mut be_luma)
      .unwrap()
      .with_luma_u16(&mut be_luma_u16)
      .unwrap()
      .with_hsv(&mut be_h, &mut be_s, &mut be_v)
      .unwrap();
    xyz12_to(&src_be, sink_be.set_kernel_gamut(target_gamut)).unwrap();
  }

  // Every output path must be bit-identical (or bit-identical f32/f16
  // patterns — the kernels are deterministic per `<BE>` since the
  // post-load value is the same logical 12-bit code).
  assert_eq!(le_rgb, be_rgb, "rgb (simd={simd}, gamut={target_gamut:?})");
  assert_eq!(
    le_rgba, be_rgba,
    "rgba (simd={simd}, gamut={target_gamut:?})"
  );
  assert_eq!(
    le_rgb_u16, be_rgb_u16,
    "rgb_u16 (simd={simd}, gamut={target_gamut:?})"
  );
  assert_eq!(
    le_rgba_u16, be_rgba_u16,
    "rgba_u16 (simd={simd}, gamut={target_gamut:?})"
  );
  assert!(
    bits_eq_f32(&le_rgb_f32, &be_rgb_f32),
    "rgb_f32 (simd={simd}, gamut={target_gamut:?})"
  );
  assert!(
    bits_eq_f32(&le_xyz_f32, &be_xyz_f32),
    "xyz_f32 (simd={simd}, gamut={target_gamut:?})"
  );
  assert!(
    bits_eq_f16(&le_rgb_f16, &be_rgb_f16),
    "rgb_f16 (simd={simd}, gamut={target_gamut:?})"
  );
  assert!(
    bits_eq_f16(&le_rgba_f16, &be_rgba_f16),
    "rgba_f16 (simd={simd}, gamut={target_gamut:?})"
  );
  assert_eq!(
    le_luma, be_luma,
    "luma (simd={simd}, gamut={target_gamut:?})"
  );
  assert_eq!(
    le_luma_u16, be_luma_u16,
    "luma_u16 (simd={simd}, gamut={target_gamut:?})"
  );
  assert_eq!(le_h, be_h, "hsv.h (simd={simd}, gamut={target_gamut:?})");
  assert_eq!(le_s, be_s, "hsv.s (simd={simd}, gamut={target_gamut:?})");
  assert_eq!(le_v, be_v, "hsv.v (simd={simd}, gamut={target_gamut:?})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn xyz12_le_be_roundtrip_all_outputs_scalar_dcip3() {
  assert_xyz12_le_be_roundtrip_all_outputs(16, 4, KernelGamut::DciP3, false);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn xyz12_le_be_roundtrip_all_outputs_scalar_rec709() {
  assert_xyz12_le_be_roundtrip_all_outputs(16, 4, KernelGamut::Rec709, false);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn xyz12_le_be_roundtrip_all_outputs_scalar_rec2020() {
  assert_xyz12_le_be_roundtrip_all_outputs(16, 4, KernelGamut::Rec2020, false);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_le_be_roundtrip_all_outputs_simd_dcip3() {
  assert_xyz12_le_be_roundtrip_all_outputs(16, 4, KernelGamut::DciP3, true);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_combined_rgb_rgba_consistent() {
  // Combined `with_rgb + with_rgba` must produce the same RGB on both
  // outputs (alpha = 0xFF on the rgba path).
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x900, 0x700);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 4 * 2 * 3];
  let mut rgba_out = std::vec![0u8; 4 * 2 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_rgb(&mut rgb_out)
    .unwrap()
    .with_rgba(&mut rgba_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();

  for x in 0..(4 * 2) {
    let rgb_idx = x * 3;
    let rgba_idx = x * 4;
    assert_eq!(rgb_out[rgb_idx], rgba_out[rgba_idx]);
    assert_eq!(rgb_out[rgb_idx + 1], rgba_out[rgba_idx + 1]);
    assert_eq!(rgb_out[rgb_idx + 2], rgba_out[rgba_idx + 2]);
    assert_eq!(rgba_out[rgba_idx + 3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_dirty_low_bits_discarded_by_kernel() {
  // FFmpeg `AV_PIX_FMT_XYZ12LE` reserves bits `[3:0]` as zero. A
  // non-spec-compliant producer that sets them anyway must produce
  // identical output to the clean input — every kernel applies `>> 4`
  // after the endian-aware load.
  let mut clean = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let mut dirty = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  for px in dirty.chunks_mut(3) {
    // Set the reserved low 4 bits with arbitrary garbage on the LE
    // wire — `from_le_bytes` interprets them as the low byte.
    let lo_dirt = [0x000F_u16, 0x000A_u16, 0x0007_u16];
    for (i, c) in px.iter_mut().enumerate() {
      let bytes = c.to_le_bytes();
      let mut dirty_u16 = u16::from_le_bytes(bytes);
      dirty_u16 |= u16::from_le_bytes(lo_dirt[i].to_le_bytes());
      *c = dirty_u16;
    }
  }
  let src_clean = Xyz12LeFrame::try_new(&clean, 4, 2, 4 * 3).unwrap();
  let src_dirty = Xyz12LeFrame::try_new(&dirty, 4, 2, 4 * 3).unwrap();

  let mut out_clean = std::vec![0u8; 4 * 2 * 3];
  let mut out_dirty = std::vec![0u8; 4 * 2 * 3];
  {
    let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_clean)
      .unwrap();
    xyz12_to(&src_clean, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();
  }
  {
    let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_dirty)
      .unwrap();
    xyz12_to(&src_dirty, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();
  }
  assert_eq!(out_clean, out_dirty);
  // Mark `clean` as touched to silence the lint for symmetry; all real
  // mutation happens on `dirty`.
  let _ = &mut clean;
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_mid_gray_sample_is_nonzero() {
  // Pre-fix regression: mid-gray sample (`0x8000` on the wire = code
  // `0x800`) was decoded as `0x000`, producing all-zero output.
  // Post-fix the `>> 4` shift extracts the active code and the kernel
  // produces a positive linear value.
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut xyz_out = std::vec![0.0f32; 4 * 2 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_xyz_f32(&mut xyz_out)
    .unwrap();
  xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3)).unwrap();
  for px in xyz_out.chunks(3) {
    assert!(px[0] > 0.1, "expected mid-gray X > 0.1, got {}", px[0]);
    assert!(px[1] > 0.1, "expected mid-gray Y > 0.1, got {}", px[1]);
    assert!(px[2] > 0.1, "expected mid-gray Z > 0.1, got {}", px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_short_buffer_rejected_at_attach() {
  let pix = solid_xyz12_frame_le(8, 4, 0, 0, 0);
  let _src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 8 * 4 * 3 - 1]; // one byte short
  let res = MixedSinker::<Xyz12Le>::new(8, 4).with_rgb(&mut rgb_out);
  assert!(matches!(
    res,
    Err(MixedSinkerError::InsufficientRgbBuffer(_))
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_short_xyz_f32_buffer_rejected_at_attach() {
  let mut xyz_out = std::vec![0.0f32; 8 * 4 * 3 - 1];
  let res = MixedSinker::<Xyz12Le>::new(8, 4).with_xyz_f32(&mut xyz_out);
  assert!(matches!(
    res,
    Err(MixedSinkerError::InsufficientXyzF32Buffer(_))
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_dimension_mismatch_rejected_at_begin_frame() {
  let pix = solid_xyz12_frame_le(4, 2, 0, 0, 0);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 8 * 4 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  let res = xyz12_to(&src, sink.set_kernel_gamut(KernelGamut::DciP3));
  assert!(matches!(res, Err(MixedSinkerError::DimensionMismatch(_))));
}
