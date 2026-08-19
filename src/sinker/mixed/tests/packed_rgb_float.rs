use super::*;

// ---- Tier 9 — Rgbf32 packed-float-RGB source family ---------------------

/// Builds a row-padded Rgbf32 frame with a constant `(R, G, B)` triple.
fn solid_rgbf32_frame(width: u32, height: u32, r: f32, g: f32, b: f32) -> Vec<f32> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0.0f32; w * h * 3];
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
fn rgbf32_with_rgb_clamps_to_u8() {
  // 0.5 → 128 (round-half-even at 127.5), 1.0 → 255, 2.0 → 255 (HDR
  // clamp), -0.5 → 0 (negative clamp).
  let pix = solid_rgbf32_frame(16, 4, 1.0, 2.0, -0.5);
  let src = Rgbf32LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgbf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_rgb_u16_clamps_to_u16() {
  let pix = solid_rgbf32_frame(16, 4, 0.5, 1.0, 1.5);
  let src = Rgbf32LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_rgb_u16(&mut rgb_out)
    .unwrap();
  rgbf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  // 0.5 * 65535 = 32767.5 → 32768 (round-half-even); 1.0 → 65535;
  // 1.5 → 65535 (clamp).
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [32768, 65535, 65535]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_rgb_f32_passes_through_lossless() {
  // Includes HDR (> 1.0), negatives, and exact integer values to
  // confirm bit-exact pass-through.
  let mut pix = std::vec![0.0f32; 16 * 4 * 3];
  for (i, v) in pix.iter_mut().enumerate() {
    // Build a deterministic mix: HDR, negative, in-range.
    *v = match i % 6 {
      0 => 0.0,
      1 => 1.0,
      2 => -0.25,
      3 => 1.5,
      4 => 0.5,
      _ => 100.0,
    };
  }
  let src = Rgbf32LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_rgb_f32(&mut rgb_out)
    .unwrap();
  rgbf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  // Bit-exact equality (no rounding, no clamping in the f32 path).
  assert_eq!(rgb_out, pix);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_luma_u8() {
  // Constant white → BT.709 luma 235 (limited range) or 255 (full range).
  let pix = solid_rgbf32_frame(16, 4, 1.0, 1.0, 1.0);
  let src = Rgbf32LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_luma(&mut luma_out)
    .unwrap();
  rgbf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  // Full-range BT.709: white maps to 255.
  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_luma_u16() {
  let pix = solid_rgbf32_frame(16, 4, 1.0, 1.0, 1.0);
  let src = Rgbf32LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u16; 16 * 4];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  rgbf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  // u8 luma 255 → u16 255 (zero-extended, matching the packed-YUV
  // luma_u16 convention).
  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_hsv() {
  // Pure red → H=0, S=255, V=255 in the OpenCV 8-bit HSV encoding.
  let pix = solid_rgbf32_frame(16, 4, 1.0, 0.0, 0.0);
  let src = Rgbf32LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let n = 16 * 4;
  let mut h_out = std::vec![0u8; n];
  let mut s_out = std::vec![0u8; n];
  let mut v_out = std::vec![0u8; n];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_hsv(&mut h_out, &mut s_out, &mut v_out)
    .unwrap();
  rgbf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn rgbf32_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop and scalar tail across
  // every backend block size.
  let w = 1921usize;
  let h = 4usize;
  let n_lanes = w * h * 3;
  // Mix of in-range, HDR, and negative values to exercise all branches.
  let mut pix = std::vec![0.0f32; n_lanes];
  let mut state: u32 = 0xDEAD_BEEF;
  for (i, v) in pix.iter_mut().enumerate() {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *v = match (state >> 28) & 0b11 {
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
  }
  let src = Rgbf32LeFrame::try_new(&pix, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];
  let mut rgba_u16_simd = std::vec![0u16; w * h * 4];
  let mut rgba_u16_scalar = std::vec![0u16; w * h * 4];
  let mut rgb_f32_simd = std::vec![0.0f32; w * h * 3];
  let mut rgb_f32_scalar = std::vec![0.0f32; w * h * 3];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];
  let mut luma_u16_simd = std::vec![0u16; w * h];
  let mut luma_u16_scalar = std::vec![0u16; w * h];

  let mut s_simd = MixedSinker::<Rgbf32>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_simd)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap()
    .with_luma_u16(&mut luma_u16_simd)
    .unwrap();
  rgbf32_to(&src, true, s_simd.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut s_scalar = MixedSinker::<Rgbf32>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_scalar)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap()
    .with_luma_u16(&mut luma_u16_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  rgbf32_to(&src, true, s_scalar.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "RGB u16 output diverges");
  assert_eq!(rgba_u16_simd, rgba_u16_scalar, "RGBA u16 output diverges");
  assert_eq!(rgb_f32_simd, rgb_f32_scalar, "RGB f32 output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
  assert_eq!(luma_u16_simd, luma_u16_scalar, "Luma u16 output diverges");
  assert_eq!(rgb_f32_simd, pix, "RGB f32 output is not lossless");
}

/// LE-encoded byte contract regression: builds an [`Rgbf32Frame`] from a
/// `&[f32]` plane explicitly encoded as LE bytes (per the FFmpeg
/// `AV_PIX_FMT_*LE` convention documented on `Rgbf32Frame`), runs it
/// through the sinker's `with_rgb_f32` lossless pass-through, and asserts
/// the output equals the host-native intended values.
///
/// Vacuous on LE hosts (where `f32::to_le_bytes` is a no-op so the LE-
/// encoded plane *is* host-native), but on a BE host this would fail fast
/// for any regression that drops the `::<false>` kernel routing — the
/// kernel must apply `u32::from_le` to recover host-native f32 from the
/// LE-encoded bytes; if it skipped the swap (e.g. `::<HOST_NATIVE_BE>` on
/// BE), the output would be byte-swapped relative to `intended`.
///
/// Mirrors the `Grayf32` LE-encoded byte contract regression.
///
/// Forces `with_simd(false)` so this test runs purely scalar — no SIMD
/// intrinsics — which lets it execute under `cargo miri test`. BE CI is
/// driven by miri on s390x / powerpc64; gating this test out of miri
/// would skip exactly the host where BE corruption would surface.
#[test]
fn rgbf32_sinker_le_encoded_frame_decodes_correctly() {
  // Mix HDR, in-range, and negative values — the f32 lossless path must
  // round-trip them bit-exact on every host.
  let intended: Vec<f32> = (0..16 * 4 * 3)
    .map(|i| match i % 4 {
      0 => 0.5,
      1 => 1.5,
      2 => -0.25,
      _ => 100.0,
    })
    .collect();
  // Construct the plane as LE-encoded bytes reinterpreted as f32 (the
  // documented `*LE` Frame contract). On LE host this is identity; on BE
  // host the bit-pattern is byte-swapped so the kernel must `from_le` it
  // back to host-native.
  let pix: Vec<f32> = intended
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect();
  let src = Rgbf32LeFrame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_f32_out = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_simd(false)
    .with_rgb_f32(&mut rgb_f32_out)
    .unwrap();
  rgbf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  // Output must be host-native intended values. On a BE host with a
  // regressed `::<HOST_NATIVE_BE>` routing this would be byte-swapped.
  assert_eq!(
    rgb_f32_out, intended,
    "Rgbf32 sinker LE-encoded plane decoded incorrectly"
  );
}
// Phase 4 — Rgbf32 LE/BE round-trip
//
// Build host-independent fixtures for the same logical samples in BOTH plane
// orderings (LE-encoded bytes and BE-encoded bytes) and run them through the
// matching `Rgbf32<BE>` sinker monomorphizations. Output A and B must be
// byte-identical because the kernel byte-swaps under the hood — the same
// logical samples should yield the same f32 outputs regardless of input byte
// order. This catches:
//   - missing `<BE>` propagation in the rgbf32 sinker call sites,
//   - regressions in the `f32::from_bits(u32::from_le/be(...))` swap path,
//   - mismatches between `MixedSinker<Rgbf32<true>>` and the BE row kernels.
/// Re-encode a host-native f32 slice as **LE-encoded** byte storage. Used to
/// build `Rgbf32LeFrame` planes whose bytes are little-endian; the kernel
/// recovers host-native via `from_le` on the bit pattern.
fn as_le_rgbf32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Re-encode a host-native f32 slice as **BE-encoded** byte storage. Used to
/// build `Rgbf32BeFrame` planes whose bytes are big-endian; the kernel
/// recovers host-native via `from_be` on the bit pattern.
fn as_be_rgbf32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect()
}

#[test]
fn rgbf32_le_be_roundtrip_byte_identical() {
  // Mix HDR, in-range, and negative values to surface any swap regression.
  let intended: Vec<f32> = (0..16 * 4 * 3)
    .map(|i| match i % 4 {
      0 => 0.5,
      1 => 1.5,
      2 => -0.25,
      _ => 100.0,
    })
    .collect();
  let pix_le = as_le_rgbf32(&intended);
  let pix_be = as_be_rgbf32(&intended);

  // LE path — default `Rgbf32` marker.
  let frame_le = Rgbf32LeFrame::try_new(&pix_le, 16, 4, 16 * 3).unwrap();
  let mut out_le = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink_le = MixedSinker::<Rgbf32>::new(16, 4)
    .with_simd(false)
    .with_rgb_f32(&mut out_le)
    .unwrap();
  rgbf32_to(
    &frame_le,
    true,
    sink_le.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  // BE path — explicit `Rgbf32<true>` monomorphization.
  let frame_be = Rgbf32BeFrame::try_new(&pix_be, 16, 4, 16 * 3).unwrap();
  let mut out_be = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink_be = MixedSinker::<Rgbf32<true>>::new(16, 4)
    .with_simd(false)
    .with_rgb_f32(&mut out_be)
    .unwrap();
  rgbf32_to_endian(
    &frame_be,
    true,
    sink_be.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  // Both outputs must equal the intended host-native values bit-for-bit.
  assert_eq!(
    out_le, intended,
    "Rgbf32 LE plane decoded to wrong host-native values"
  );
  assert_eq!(
    out_be, intended,
    "Rgbf32 BE plane decoded to wrong host-native values — `<const BE>` propagation broken"
  );
  assert_eq!(
    out_le, out_be,
    "Rgbf32 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}
