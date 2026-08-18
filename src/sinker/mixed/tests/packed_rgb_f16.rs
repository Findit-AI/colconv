use super::*;

// ---- Tier 9 — Rgbf16 packed-half-float-RGB source family ----------------

/// Builds a tightly-packed Rgbf16 row buffer (`width * height * 3` `f16`
/// elements, no row stride padding) filled with a constant `(R, G, B)` triple.
fn solid_rgbf16_frame(
  width: u32,
  height: u32,
  r: half::f16,
  g: half::f16,
  b: half::f16,
) -> std::vec::Vec<half::f16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![half::f16::ZERO; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = r;
    px[1] = g;
    px[2] = b;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_clamps_to_u8() {
  // 1.0 → 255, 2.0 → 255 (HDR clamp), -0.5 → 0 (negative clamp).
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(2.0),
    half::f16::from_f32(-0.5),
  );
  let src = Rgbf16LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_u16_clamps_to_u16() {
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(0.5),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.5),
  );
  let src = Rgbf16LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_u16(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  // 0.5 * 65535 ≈ 32767 or 32768 (half-precision rounds 0.5 to exact 0.5,
  // so downstream is the same as Rgbf32); 1.0 → 65535; 1.5 → 65535 (clamp).
  for px in rgb_out.chunks(3) {
    assert!(
      px[0] >= 32767 && px[0] <= 32768,
      "unexpected mid: {}",
      px[0]
    );
    assert_eq!(px[1], 65535);
    assert_eq!(px[2], 65535);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_f16_is_lossless() {
  // Include HDR, negatives, and in-range values to confirm bit-exact
  // pass-through.
  let vals_f32 = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0];
  let n_pixels = 16 * 4;
  let pix: std::vec::Vec<half::f16> = (0..n_pixels * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  let src = Rgbf16LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_f16(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  // Bit-exact equality (no rounding, no clamping in the f16 path).
  assert_eq!(rgb_out, pix);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_f32_widens_losslessly() {
  // Includes HDR (> 1.0), negatives, and exact values.
  let vals_f32 = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0];
  let n_pixels = 16 * 4;
  let pix: std::vec::Vec<half::f16> = (0..n_pixels * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  let src = Rgbf16LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_f32(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  // Each widened f32 must equal the f16 widened via to_f32.
  let expected: std::vec::Vec<f32> = pix.iter().map(|h| h.to_f32()).collect();
  assert_eq!(rgb_out, expected, "rgb_f32 widen is not lossless");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_luma_u8() {
  // Constant white → BT.709 full-range luma 255.
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
  );
  let src = Rgbf16LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_luma(&mut luma_out)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_luma_u16() {
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
  );
  let src = Rgbf16LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u16; 16 * 4];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  // u8 luma 255 → u16 255 (zero-extended).
  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_hsv() {
  // Pure red → H=0, S=255, V=255 in the OpenCV 8-bit HSV encoding.
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::ZERO,
    half::f16::ZERO,
  );
  let src = Rgbf16LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let n = 16 * 4;
  let mut h_out = std::vec![0u8; n];
  let mut s_out = std::vec![0u8; n];
  let mut v_out = std::vec![0u8; n];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_hsv(&mut h_out, &mut s_out, &mut v_out)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for i in 0..n {
    assert_eq!(h_out[i], 0);
    assert_eq!(s_out[i], 255);
    assert_eq!(v_out[i], 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop and scalar tail across
  // every backend block size.
  let w = 1921usize;
  let h = 4usize;
  let n_lanes = w * h * 3;
  let mut pix = std::vec![half::f16::ZERO; n_lanes];
  let mut state: u32 = 0xDEAD_BEEF;
  for (i, v) in pix.iter_mut().enumerate() {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let f = match (state >> 28) & 0b11 {
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
    *v = half::f16::from_f32(f);
  }
  let src = Rgbf16LeFrame::try_new(&pix, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];
  let mut rgba_u16_simd = std::vec![0u16; w * h * 4];
  let mut rgba_u16_scalar = std::vec![0u16; w * h * 4];
  let mut rgb_f16_simd = std::vec![half::f16::ZERO; w * h * 3];
  let mut rgb_f16_scalar = std::vec![half::f16::ZERO; w * h * 3];
  let mut rgb_f32_simd = std::vec![0.0f32; w * h * 3];
  let mut rgb_f32_scalar = std::vec![0.0f32; w * h * 3];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];
  let mut luma_u16_simd = std::vec![0u16; w * h];
  let mut luma_u16_scalar = std::vec![0u16; w * h];

  let mut s_simd = MixedSinker::<Rgbf16>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_simd)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16_simd)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap()
    .with_luma_u16(&mut luma_u16_simd)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Rgbf16>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_scalar)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16_scalar)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap()
    .with_luma_u16(&mut luma_u16_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "RGB u16 output diverges");
  assert_eq!(rgba_u16_simd, rgba_u16_scalar, "RGBA u16 output diverges");
  assert_eq!(rgb_f16_simd, rgb_f16_scalar, "RGB f16 output diverges");
  assert_eq!(rgb_f32_simd, rgb_f32_scalar, "RGB f32 output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
  assert_eq!(luma_u16_simd, luma_u16_scalar, "Luma u16 output diverges");
  assert_eq!(rgb_f16_simd, pix, "RGB f16 output is not lossless");
}

/// LE-encoded byte contract regression: builds an [`Rgbf16Frame`] from a
/// `&[half::f16]` plane explicitly encoded as LE bytes (per the FFmpeg
/// `AV_PIX_FMT_*LE` convention documented on `Rgbf16Frame`), runs it
/// through the sinker's `with_rgb_f16` lossless pass-through, and asserts
/// the output equals the host-native intended values.
///
/// Vacuous on LE hosts (where `to_le` on a `u16` is a no-op so the LE-
/// encoded plane *is* host-native), but on a BE host this would fail fast
/// for any regression that drops the `::<false>` kernel routing — the
/// kernel must apply `u16::from_le` to recover host-native f16 bit-patterns
/// from the LE-encoded bytes.
///
/// Mirrors the `Grayf32` LE-encoded byte contract regression.
///
/// Forces `with_simd(false)` so this test runs purely scalar — no SIMD
/// intrinsics — which lets it execute under `cargo miri test`. BE CI is
/// driven by miri on s390x / powerpc64; gating this test out of miri
/// would skip exactly the host where BE corruption would surface.
///
/// Re-gated on miri because the fixture builder calls `half::f16::from_f32`,
/// which on aarch64 / x86 / x86_64 with `target_feature = "fp16"` (or F16C)
/// expands to inline `asm!` that miri rejects. BE-host miri (s390x /
/// powerpc64) covers the byte-swap correctness via the f32 LE-encoded
/// regression tests in this module instead.
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_sinker_le_encoded_frame_decodes_correctly() {
  let vals_f32 = [0.5f32, 1.5, -0.25, 100.0];
  let intended: Vec<half::f16> = (0..16 * 4 * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  // Encode the plane as LE bytes reinterpreted as f16 (the documented
  // `*LE` Frame contract). On LE host: identity. On BE host: byte-swapped
  // bit-patterns the kernel must `from_le` back to host-native.
  let pix: Vec<half::f16> = intended
    .iter()
    .map(|&v| half::f16::from_bits(v.to_bits().to_le()))
    .collect();
  let src = Rgbf16LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_f16_out = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_simd(false)
    .with_rgb_f16(&mut rgb_f16_out)
    .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  assert_eq!(
    rgb_f16_out, intended,
    "Rgbf16 sinker LE-encoded plane decoded incorrectly"
  );
}
// Phase 4 — Rgbf16 LE/BE round-trip
//
// Build host-independent fixtures for the same logical samples in BOTH plane
// orderings (LE-encoded bytes and BE-encoded bytes) and run them through the
// matching `Rgbf16<BE>` sinker monomorphizations. Output A and B must be
// byte-identical because the kernel byte-swaps under the hood — the same
// logical samples should yield the same f16 outputs regardless of input byte
// order. This catches:
//   - missing `<BE>` propagation in the rgbf16 sinker call sites,
//   - regressions in the `f16::from_bits(u16::from_le/be(...))` swap path,
//   - mismatches between `MixedSinker<Rgbf16<true>>` and the BE row kernels.
//
// Gated on miri because `half::f16::from_f32` (used to build the fixture)
// expands to inline `asm!` on platforms with hardware f16 support, which miri
// rejects. The plain LE-decode regression above already covers BE-host miri
// (via s390x / powerpc64) using f32 fixtures that don't need hardware f16.
/// Re-encode a host-native f16 slice as **LE-encoded** byte storage.
fn as_le_rgbf16(host: &[half::f16]) -> Vec<half::f16> {
  host
    .iter()
    .map(|&v| half::f16::from_bits(u16::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Re-encode a host-native f16 slice as **BE-encoded** byte storage.
fn as_be_rgbf16(host: &[half::f16]) -> Vec<half::f16> {
  host
    .iter()
    .map(|&v| half::f16::from_bits(u16::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_le_be_roundtrip_byte_identical() {
  let vals_f32 = [0.5f32, 1.5, -0.25, 100.0];
  let intended: Vec<half::f16> = (0..16 * 4 * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  let pix_le = as_le_rgbf16(&intended);
  let pix_be = as_be_rgbf16(&intended);

  // LE path — default `Rgbf16` marker.
  let frame_le = Rgbf16LeFrame::try_new(&pix_le, 16, 4, 16 * 3).unwrap();
  let mut out_le = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let mut sink_le = MixedSinker::<Rgbf16>::new(16, 4)
    .with_simd(false)
    .with_rgb_f16(&mut out_le)
    .unwrap();
  rgbf16_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  // BE path — explicit `Rgbf16<true>` monomorphization.
  let frame_be = Rgbf16BeFrame::try_new(&pix_be, 16, 4, 16 * 3).unwrap();
  let mut out_be = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let mut sink_be = MixedSinker::<Rgbf16<true>>::new(16, 4)
    .with_simd(false)
    .with_rgb_f16(&mut out_be)
    .unwrap();
  rgbf16_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  // Both outputs must equal the intended host-native values bit-for-bit.
  assert_eq!(
    out_le, intended,
    "Rgbf16 LE plane decoded to wrong host-native values"
  );
  assert_eq!(
    out_be, intended,
    "Rgbf16 BE plane decoded to wrong host-native values — `<const BE>` propagation broken"
  );
  assert_eq!(
    out_le, out_be,
    "Rgbf16 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}
