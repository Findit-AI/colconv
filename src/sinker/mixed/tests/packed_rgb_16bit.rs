use super::*;
use crate::{
  frame::{
    Bgr48BeFrame, Bgr48Frame, Bgra64BeFrame, Bgra64Frame, Rgb48BeFrame, Rgb48Frame, Rgba64BeFrame,
    Rgba64Frame,
  },
  sinker::mixed::MixedSinker,
  source::{Bgr48, Bgra64, Rgb48, Rgba64, bgr48_to, bgra64_to, rgb48_to, rgba64_to},
};

/// Re-encode a host-native u16 slice as LE-encoded byte storage. Sink kernels
/// recover the intended logical values via `u16::from_le` on both LE (no-op)
/// and BE (byte-swap) hosts.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

// ---- Rgb48 -----------------------------------------------------------------

#[test]
fn rgb48_with_rgb_u16_identity() {
  // Native passthrough: each channel copied verbatim (no shift).
  let src: Vec<u16> = as_le_u16(&[0x1234, 0x5678, 0x9ABC]);
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u16; 3];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1)
    .with_rgb_u16(&mut out)
    .unwrap();
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out, vec![0x1234u16, 0x5678, 0x9ABC]);
}

#[test]
fn rgb48_with_rgb_narrows_correctly() {
  // Each 16-bit channel narrowed >> 8: 0xFF00 → 0xFF, 0x8000 → 0x80, 0x0100 → 0x01.
  let src: Vec<u16> = as_le_u16(&[0xFF00, 0x8000, 0x0100]);
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u8; 3];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1).with_rgb(&mut out).unwrap();
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out, vec![0xFFu8, 0x80, 0x01]);
}

#[test]
fn rgb48_with_rgba_forces_0xff() {
  let src: Vec<u16> = as_le_u16(&[0xFFFF, 0x8000, 0x0000]);
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1).with_rgba(&mut out).unwrap();
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xFF); // alpha forced to opaque
}

#[test]
fn rgb48_with_rgba_u16_forces_0xffff() {
  let src: Vec<u16> = as_le_u16(&[0x1000, 0x2000, 0x3000]);
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xFFFF);
}

#[test]
fn rgb48_with_rgb_u16_and_rgba_u16_both_correct() {
  // Two pixels: verify identity copy for RGB and forced alpha for RGBA.
  // `intended` holds host-native logical samples; `src` is the LE-encoded
  // byte storage the kernel decodes via `from_le`. The decoded `rgb_u16`
  // output is host-native, so it must be compared against `intended`, not
  // against the byte-swapped `src` (which would only match on LE hosts).
  let intended: [u16; 6] = [0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x1357, 0x2468];
  let src: Vec<u16> = as_le_u16(&intended);
  let frame = Rgb48Frame::new(&src, 2, 1, 6);
  let mut rgb_u16 = vec![0u16; 6];
  let mut rgba_u16 = vec![0u16; 8];
  let mut sinker = MixedSinker::<Rgb48>::new(2, 1)
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  // RGB u16: identity copy of source decoded to host-native intended values.
  assert_eq!(rgb_u16[..6], intended[..6]);
  // RGBA u16: alpha slots forced to 0xFFFF
  assert_eq!(rgba_u16[3], 0xFFFF);
  assert_eq!(rgba_u16[7], 0xFFFF);
}

#[test]
fn rgb48_with_luma_non_zero_for_nonzero_input() {
  // Any non-zero equal R/G/B → non-zero luma.
  let src: Vec<u16> = as_le_u16(&[0x8000, 0x8000, 0x8000]);
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut luma = vec![0u8; 1];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1)
    .with_luma(&mut luma)
    .unwrap();
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_ne!(luma[0], 0);
}

#[test]
fn rgb48_with_luma_u16_non_zero_for_nonzero_input() {
  let src: Vec<u16> = as_le_u16(&[0x8000, 0x8000, 0x8000]);
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut luma_u16 = vec![0u16; 1];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1)
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_ne!(luma_u16[0], 0);
}

#[test]
fn rgb48_simd_matches_scalar() {
  // width=8: exercises SIMD path (if available) vs forced scalar path.
  let w: usize = 8;
  let src: Vec<u16> = (0..w * 3).map(|i| (i * 0x1001) as u16).collect();
  let frame = Rgb48Frame::new(&src, w as u32, 1, w as u32 * 3);
  let mut rgb_simd = vec![0u8; w * 3];
  let mut rgb_scalar = vec![0u8; w * 3];
  let mut s1 = MixedSinker::<Rgb48>::new(w, 1)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut s1).unwrap();
  let mut s2 = MixedSinker::<Rgb48>::new(w, 1)
    .with_rgb(&mut rgb_scalar)
    .unwrap();
  s2.set_simd(false);
  rgb48_to(&frame, true, KernelMatrix::Bt709, &mut s2).unwrap();
  assert_eq!(rgb_simd, rgb_scalar);
}

// ---- Bgr48 -----------------------------------------------------------------

#[test]
fn bgr48_channel_order_swapped_vs_rgb48() {
  // B=0x1000, G=0x2000, R=0x3000 stored in BGR input order.
  // After sinker's B↔R swap on output: R=0x30, G=0x20, B=0x10 in u8 RGB.
  let src: Vec<u16> = as_le_u16(&[0x1000, 0x2000, 0x3000]);
  let rgb_frame = Rgb48Frame::new(&src, 1, 1, 3);
  let bgr_frame = Bgr48Frame::new(&src, 1, 1, 3);
  let mut rgb_from_rgb48 = vec![0u8; 3];
  let mut rgb_from_bgr48 = vec![0u8; 3];
  let mut s1 = MixedSinker::<Rgb48>::new(1, 1)
    .with_rgb(&mut rgb_from_rgb48)
    .unwrap();
  rgb48_to(&rgb_frame, false, KernelMatrix::Bt709, &mut s1).unwrap();
  let mut s2 = MixedSinker::<Bgr48>::new(1, 1)
    .with_rgb(&mut rgb_from_bgr48)
    .unwrap();
  bgr48_to(&bgr_frame, false, KernelMatrix::Bt709, &mut s2).unwrap();
  // Bgr48 with_rgb: R=0x3000>>8=0x30, G=0x2000>>8=0x20, B=0x1000>>8=0x10.
  assert_eq!(rgb_from_bgr48[0], 0x30); // R
  assert_eq!(rgb_from_bgr48[1], 0x20); // G
  assert_eq!(rgb_from_bgr48[2], 0x10); // B
  // Rgb48 on same bytes treats first element as R, which differs.
  assert_ne!(rgb_from_rgb48[0], rgb_from_bgr48[0]);
}

#[test]
fn bgr48_with_rgba_forces_0xff() {
  let src: Vec<u16> = as_le_u16(&[0xAAAA, 0xBBBB, 0xCCCC]);
  let frame = Bgr48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Bgr48>::new(1, 1).with_rgba(&mut out).unwrap();
  bgr48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xFF);
}

#[test]
fn bgr48_with_rgba_u16_forces_0xffff() {
  let src: Vec<u16> = as_le_u16(&[0x1234, 0x5678, 0x9ABC]);
  let frame = Bgr48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Bgr48>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  bgr48_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xFFFF);
}

// ---- Rgba64 ----------------------------------------------------------------

#[test]
fn rgba64_with_rgba_passes_source_alpha_u8() {
  // R=0xFFFF, G=0x8000, B=0x0000, A=0xABFF → alpha byte = 0xABFF >> 8 = 0xAB.
  let src: Vec<u16> = as_le_u16(&[0xFFFF, 0x8000, 0x0000, 0xABFF]);
  let frame = Rgba64Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Rgba64>::new(1, 1)
    .with_rgba(&mut out)
    .unwrap();
  rgba64_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xAB); // 0xABFF >> 8 = 0xAB
}

#[test]
fn rgba64_with_rgba_u16_passes_source_alpha_native() {
  // Source α = 0xABCD must be preserved verbatim (no shift).
  let src: Vec<u16> = as_le_u16(&[0xFFFF, 0x8000, 0x0000, 0xABCD]);
  let frame = Rgba64Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Rgba64>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  rgba64_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xABCD);
}

#[test]
fn rgba64_strategy_a_plus_rgb_and_rgba_byte_identical() {
  // Strategy A+: running with_rgb + with_rgba together must produce the
  // same RGBA bytes as running with_rgba alone (chroma kernel once).
  let w: usize = 4;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1111) as u16).collect();
  let frame = Rgba64Frame::new(&src, w as u32, 1, w as u32 * 4);

  let mut rgb_combo = vec![0u8; w * 3];
  let mut rgba_combo = vec![0u8; w * 4];
  let mut sinker = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgb(&mut rgb_combo)
    .unwrap()
    .with_rgba(&mut rgba_combo)
    .unwrap();
  rgba64_to(&frame, false, KernelMatrix::Bt709, &mut sinker).unwrap();

  let mut rgba_only = vec![0u8; w * 4];
  let mut sinker2 = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgba(&mut rgba_only)
    .unwrap();
  rgba64_to(&frame, false, KernelMatrix::Bt709, &mut sinker2).unwrap();

  assert_eq!(rgba_combo, rgba_only);
}

#[test]
fn rgba64_strategy_a_plus_u16_path_byte_identical() {
  // Strategy A+ on u16 path: with_rgb_u16 + with_rgba_u16 combined must
  // match with_rgba_u16 alone (single deinterleave pass + alpha scatter).
  let w: usize = 4;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1000) as u16).collect();
  let frame = Rgba64Frame::new(&src, w as u32, 1, w as u32 * 4);

  let mut rgb_u16_combo = vec![0u16; w * 3];
  let mut rgba_u16_combo = vec![0u16; w * 4];
  let mut sinker = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgb_u16(&mut rgb_u16_combo)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_combo)
    .unwrap();
  rgba64_to(&frame, false, KernelMatrix::Bt709, &mut sinker).unwrap();

  let mut rgba_u16_only = vec![0u16; w * 4];
  let mut sinker2 = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgba_u16(&mut rgba_u16_only)
    .unwrap();
  rgba64_to(&frame, false, KernelMatrix::Bt709, &mut sinker2).unwrap();

  assert_eq!(rgba_u16_combo, rgba_u16_only);
}

#[test]
fn rgba64_simd_matches_scalar() {
  // SIMD path (if available) must match the forced scalar path byte-for-byte.
  let w: usize = 8;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1001) as u16).collect();
  let frame = Rgba64Frame::new(&src, w as u32, 1, w as u32 * 4);
  let mut rgba_simd = vec![0u8; w * 4];
  let mut rgba_scalar = vec![0u8; w * 4];
  let mut s1 = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgba(&mut rgba_simd)
    .unwrap();
  rgba64_to(&frame, true, KernelMatrix::Bt709, &mut s1).unwrap();
  let mut s2 = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgba(&mut rgba_scalar)
    .unwrap();
  s2.set_simd(false);
  rgba64_to(&frame, true, KernelMatrix::Bt709, &mut s2).unwrap();
  assert_eq!(rgba_simd, rgba_scalar);
}

#[test]
fn rgba64_with_rgb_u16_drops_alpha() {
  // with_rgb_u16 must output exactly 3 elements per pixel (alpha slot dropped).
  let src: Vec<u16> = as_le_u16(&[0x1111, 0x2222, 0x3333, 0xAAAA]);
  let frame = Rgba64Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u16; 3];
  let mut sinker = MixedSinker::<Rgba64>::new(1, 1)
    .with_rgb_u16(&mut out)
    .unwrap();
  rgba64_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[0], 0x1111);
  assert_eq!(out[1], 0x2222);
  assert_eq!(out[2], 0x3333);
}

// ---- Bgra64 ----------------------------------------------------------------

#[test]
fn bgra64_channel_order_and_alpha_preserved() {
  // B=0x1000, G=0x2000, R=0x3000, A=0xAAAA in BGRA source order.
  // After B↔R swap on output: R=0x30, G=0x20, B=0x10. Alpha 0xAAAA>>8=0xAA.
  let src: Vec<u16> = as_le_u16(&[0x1000, 0x2000, 0x3000, 0xAAAA]);
  let frame = Bgra64Frame::new(&src, 1, 1, 4);
  let mut rgb_out = vec![0u8; 3];
  let mut rgba_out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Bgra64>::new(1, 1)
    .with_rgb(&mut rgb_out)
    .unwrap()
    .with_rgba(&mut rgba_out)
    .unwrap();
  bgra64_to(&frame, false, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(rgb_out[0], 0x30); // R
  assert_eq!(rgb_out[1], 0x20); // G
  assert_eq!(rgb_out[2], 0x10); // B
  assert_eq!(rgba_out[3], 0xAA); // A = 0xAAAA >> 8
}

#[test]
fn bgra64_strategy_a_plus_rgb_and_rgba_byte_identical() {
  // Strategy A+ must produce same RGBA as standalone with_rgba path.
  let w: usize = 4;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1111) as u16).collect();
  let frame = Bgra64Frame::new(&src, w as u32, 1, w as u32 * 4);

  let mut rgb_combo = vec![0u8; w * 3];
  let mut rgba_combo = vec![0u8; w * 4];
  let mut sinker = MixedSinker::<Bgra64>::new(w, 1)
    .with_rgb(&mut rgb_combo)
    .unwrap()
    .with_rgba(&mut rgba_combo)
    .unwrap();
  bgra64_to(&frame, false, KernelMatrix::Bt709, &mut sinker).unwrap();

  let mut rgba_only = vec![0u8; w * 4];
  let mut sinker2 = MixedSinker::<Bgra64>::new(w, 1)
    .with_rgba(&mut rgba_only)
    .unwrap();
  bgra64_to(&frame, false, KernelMatrix::Bt709, &mut sinker2).unwrap();

  assert_eq!(rgba_combo, rgba_only);
}

#[test]
fn bgra64_strategy_a_plus_u16_path_byte_identical() {
  let w: usize = 4;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1000) as u16).collect();
  let frame = Bgra64Frame::new(&src, w as u32, 1, w as u32 * 4);

  let mut rgb_u16_combo = vec![0u16; w * 3];
  let mut rgba_u16_combo = vec![0u16; w * 4];
  let mut sinker = MixedSinker::<Bgra64>::new(w, 1)
    .with_rgb_u16(&mut rgb_u16_combo)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_combo)
    .unwrap();
  bgra64_to(&frame, false, KernelMatrix::Bt709, &mut sinker).unwrap();

  let mut rgba_u16_only = vec![0u16; w * 4];
  let mut sinker2 = MixedSinker::<Bgra64>::new(w, 1)
    .with_rgba_u16(&mut rgba_u16_only)
    .unwrap();
  bgra64_to(&frame, false, KernelMatrix::Bt709, &mut sinker2).unwrap();

  assert_eq!(rgba_u16_combo, rgba_u16_only);
}

#[test]
fn bgra64_with_rgba_u16_passes_source_alpha_native() {
  let src: Vec<u16> = as_le_u16(&[0x1000, 0x2000, 0x3000, 0xBEEF]);
  let frame = Bgra64Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Bgra64>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  bgra64_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xBEEF); // native u16 alpha verbatim
}

// ---- Edge cases / error paths ----------------------------------------------

#[test]
fn rgb48_multi_row_frame() {
  // 2x2 frame: verify correct row-by-row accumulation.
  let src: Vec<u16> = as_le_u16(&[
    0xFF00, 0x0000, 0x0000, // row 0, px 0: R=0xFF, G=0x00, B=0x00
    0x0000, 0xFF00, 0x0000, // row 0, px 1: R=0x00, G=0xFF, B=0x00
    0x0000, 0x0000, 0xFF00, // row 1, px 0: R=0x00, G=0x00, B=0xFF
    0xFF00, 0xFF00, 0xFF00, // row 1, px 1: R=0xFF, G=0xFF, B=0xFF
  ]);
  let frame = Rgb48Frame::new(&src, 2, 2, 6);
  let mut out = vec![0u8; 12];
  let mut sinker = MixedSinker::<Rgb48>::new(2, 2).with_rgb(&mut out).unwrap();
  rgb48_to(&frame, false, KernelMatrix::Bt709, &mut sinker).unwrap();
  // Row 0, px 0: only R set
  assert_eq!(out[0], 0xFF); // R
  assert_eq!(out[1], 0x00); // G
  assert_eq!(out[2], 0x00); // B
  // Row 1, px 1: all channels max
  assert_eq!(out[9], 0xFF);
  assert_eq!(out[10], 0xFF);
  assert_eq!(out[11], 0xFF);
}

// ---- BE-contract regression -----------------------------------------------

/// Rgb48 sinker LE-encoded plane decodes correctly on every host.
///
/// The frame doc-comment contract (see `src/frame/packed_rgb_16bit.rs`) says
/// the `&[u16]` plane is the **LE-encoded byte layout** reinterpreted as
/// `u16` (matching FFmpeg's `*LE` suffix). On a little-endian host LE bytes
/// are host-native — identity. On a big-endian host the bytes are swapped
/// relative to host-native, so the kernel must apply `u16::from_le` (kernel
/// generic `BE = false`) to recover the host-native sample before arithmetic.
///
/// This test builds the plane from LE-encoded u16 patterns
/// (`intended.to_le()` on each sample) and asserts the sinker output matches
/// the host-native `intended` values bit-exact via the `with_rgb_u16`
/// (identity) path. On a BE host with a regressed pre-swap (caller swaps,
/// kernel swaps again → double swap) this would corrupt every sample.
///
/// Forces `with_simd(false)` so this test runs purely scalar — no SIMD
/// intrinsics — which lets it execute under `cargo miri test`. BE CI is
/// driven by miri on s390x / powerpc64; gating it out of miri would skip
/// exactly the host where BE corruption would surface.
///
/// Mirrors the `rgbf32_sinker_le_encoded_frame_decodes_correctly` pattern.
#[test]
fn rgb48_sinker_le_encoded_frame_decodes_correctly() {
  // Mix high / mid / low / asymmetric byte patterns so any byte-swap regression
  // shows up as a non-trivial mismatch (not just a no-op pattern).
  let intended: Vec<u16> = (0..16 * 4 * 3)
    .map(|i| match i % 4 {
      0 => 0x1234,
      1 => 0xABCD,
      2 => 0x00FF,
      _ => 0xFF00,
    })
    .collect();
  // Construct the plane as LE-encoded u16 (the documented `*LE` Frame
  // contract). On LE host this is identity; on BE host the bit-pattern is
  // byte-swapped so the kernel must `from_le` it back to host-native.
  let pix: Vec<u16> = intended.iter().map(|&v| v.to_le()).collect();
  let src = Rgb48Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  // `with_rgb_u16` is the identity passthrough — the cleanest probe of the
  // endian contract because no narrowing or arithmetic obscures the bit
  // pattern. A single mismatched sample byte-swap would be unmissable.
  let mut rgb_u16_out = vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgb48>::new(16, 4)
    .with_simd(false)
    .with_rgb_u16(&mut rgb_u16_out)
    .unwrap();
  rgb48_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  // Output must be host-native intended values. On a BE host with a
  // regressed pre-swap (caller swaps, kernel swaps again) this would be
  // byte-swapped relative to `intended`.
  assert_eq!(
    rgb_u16_out, intended,
    "Rgb48 sinker LE-encoded plane decoded incorrectly (BE-contract regression)"
  );
}
// Phase 4 — Frame BE flag, Tier 8 trial. LE+BE round-trip parity tests.
//
// Pattern (per format):
//   1. Build a host-native `intended` u16 plane.
//   2. Encode the plane as LE bytes (`to_le_bytes`) → `pix_le`. Build
//      `MarkerLeFrame` + `MixedSinker<Marker<false>>`. Walk; collect output A.
//   3. Encode the same plane as BE bytes (`to_be_bytes`) → `pix_be`. Build
//      `MarkerBeFrame` + `MixedSinker<Marker<true>>`. Walk; collect output B.
//   4. Assert `A == B` byte-identical.
//
// Output A and B must be byte-identical because the kernel byte-swaps under
// the hood — the same logical samples should yield the same RGBA bytes
// regardless of input byte order. This catches:
//   - missing `<BE>` propagation in sinker call sites,
//   - regressions in the `load_endian_u16::<BE>` byte-swap path,
//   - mismatches between `MixedSinker<Rgb48<true>>` and the BE row kernels.
/// Re-encode a host-native u16 slice as **BE-encoded** byte storage. Used to
/// build `*BeFrame` planes whose bytes are big-endian; the kernel swaps them
/// back to host-native via `from_be`.
fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

#[test]
fn rgb48_le_be_roundtrip_byte_identical() {
  // Mix of patterns to surface any byte-swap regression.
  let intended: std::vec::Vec<u16> = (0..16 * 4 * 3)
    .map(|i| match i % 4 {
      0 => 0x1234,
      1 => 0xABCD,
      2 => 0x00FF,
      _ => 0xFF00,
    })
    .collect();
  let pix_le = as_le_u16(&intended);
  let pix_be = as_be_u16(&intended);

  let frame_le = Rgb48Frame::try_new(&pix_le, 16, 4, 16 * 3).unwrap();
  let mut out_le = vec![0u8; 16 * 4 * 4];
  let mut sink_le = MixedSinker::<Rgb48>::new(16, 4)
    .with_simd(false)
    .with_rgba(&mut out_le)
    .unwrap();
  rgb48_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Rgb48BeFrame::try_new(&pix_be, 16, 4, 16 * 3).unwrap();
  let mut out_be = vec![0u8; 16 * 4 * 4];
  let mut sink_be = MixedSinker::<Rgb48<true>>::new(16, 4)
    .with_simd(false)
    .with_rgba(&mut out_be)
    .unwrap();
  rgb48_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le, out_be,
    "Rgb48 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

#[test]
fn bgr48_le_be_roundtrip_byte_identical() {
  let intended: std::vec::Vec<u16> = (0..16 * 4 * 3)
    .map(|i| match i % 4 {
      0 => 0x1234,
      1 => 0xABCD,
      2 => 0x00FF,
      _ => 0xFF00,
    })
    .collect();
  let pix_le = as_le_u16(&intended);
  let pix_be = as_be_u16(&intended);

  let frame_le = Bgr48Frame::try_new(&pix_le, 16, 4, 16 * 3).unwrap();
  let mut out_le = vec![0u8; 16 * 4 * 4];
  let mut sink_le = MixedSinker::<Bgr48>::new(16, 4)
    .with_simd(false)
    .with_rgba(&mut out_le)
    .unwrap();
  bgr48_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Bgr48BeFrame::try_new(&pix_be, 16, 4, 16 * 3).unwrap();
  let mut out_be = vec![0u8; 16 * 4 * 4];
  let mut sink_be = MixedSinker::<Bgr48<true>>::new(16, 4)
    .with_simd(false)
    .with_rgba(&mut out_be)
    .unwrap();
  bgr48_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le, out_be,
    "Bgr48 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

#[test]
fn rgba64_le_be_roundtrip_byte_identical() {
  let intended: std::vec::Vec<u16> = (0..16 * 4 * 4)
    .map(|i| match i % 5 {
      0 => 0x1234,
      1 => 0xABCD,
      2 => 0x00FF,
      3 => 0xFF00,
      _ => 0x7FFF,
    })
    .collect();
  let pix_le = as_le_u16(&intended);
  let pix_be = as_be_u16(&intended);

  // Exercise both u8 and u16 RGBA paths via `with_rgba` + `with_rgba_u16`
  // (Strategy A+ standalone), which cover all four `rgba64_to_*` kernels.
  let frame_le = Rgba64Frame::try_new(&pix_le, 16, 4, 16 * 4).unwrap();
  let mut out_le_rgba = vec![0u8; 16 * 4 * 4];
  let mut out_le_rgba_u16 = vec![0u16; 16 * 4 * 4];
  let mut sink_le = MixedSinker::<Rgba64>::new(16, 4)
    .with_simd(false)
    .with_rgba(&mut out_le_rgba)
    .unwrap()
    .with_rgba_u16(&mut out_le_rgba_u16)
    .unwrap();
  rgba64_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Rgba64BeFrame::try_new(&pix_be, 16, 4, 16 * 4).unwrap();
  let mut out_be_rgba = vec![0u8; 16 * 4 * 4];
  let mut out_be_rgba_u16 = vec![0u16; 16 * 4 * 4];
  let mut sink_be = MixedSinker::<Rgba64<true>>::new(16, 4)
    .with_simd(false)
    .with_rgba(&mut out_be_rgba)
    .unwrap()
    .with_rgba_u16(&mut out_be_rgba_u16)
    .unwrap();
  rgba64_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le_rgba, out_be_rgba,
    "Rgba64 RGBA u8 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
  assert_eq!(
    out_le_rgba_u16, out_be_rgba_u16,
    "Rgba64 RGBA u16 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

#[test]
fn bgra64_le_be_roundtrip_byte_identical() {
  let intended: std::vec::Vec<u16> = (0..16 * 4 * 4)
    .map(|i| match i % 5 {
      0 => 0x1234,
      1 => 0xABCD,
      2 => 0x00FF,
      3 => 0xFF00,
      _ => 0x7FFF,
    })
    .collect();
  let pix_le = as_le_u16(&intended);
  let pix_be = as_be_u16(&intended);

  let frame_le = Bgra64Frame::try_new(&pix_le, 16, 4, 16 * 4).unwrap();
  let mut out_le_rgba = vec![0u8; 16 * 4 * 4];
  let mut out_le_rgba_u16 = vec![0u16; 16 * 4 * 4];
  let mut sink_le = MixedSinker::<Bgra64>::new(16, 4)
    .with_simd(false)
    .with_rgba(&mut out_le_rgba)
    .unwrap()
    .with_rgba_u16(&mut out_le_rgba_u16)
    .unwrap();
  bgra64_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Bgra64BeFrame::try_new(&pix_be, 16, 4, 16 * 4).unwrap();
  let mut out_be_rgba = vec![0u8; 16 * 4 * 4];
  let mut out_be_rgba_u16 = vec![0u16; 16 * 4 * 4];
  let mut sink_be = MixedSinker::<Bgra64<true>>::new(16, 4)
    .with_simd(false)
    .with_rgba(&mut out_be_rgba)
    .unwrap()
    .with_rgba_u16(&mut out_be_rgba_u16)
    .unwrap();
  bgra64_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le_rgba, out_be_rgba,
    "Bgra64 RGBA u8 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
  assert_eq!(
    out_le_rgba_u16, out_be_rgba_u16,
    "Bgra64 RGBA u16 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}
