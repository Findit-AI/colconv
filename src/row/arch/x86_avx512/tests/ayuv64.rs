use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Build a deterministic pseudo-random AYUV64 packed stream.
/// Returns `width * 4` u16 elements. Channels vary across the full
/// u16 range (0..65535).
fn pseudo_random_ayuv64(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 4)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed.wrapping_mul(3));
      (s & 0xFFFF) as u16
    })
    .collect()
}

fn check_rgb<const ALPHA: bool, const ALPHA_SRC: bool>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let p = pseudo_random_ayuv64(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut s, width, matrix, full_range);
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX-512 ayuv64<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool, const ALPHA_SRC: bool>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let p = pseudo_random_ayuv64(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>(
    &p, &mut s, width, matrix, full_range,
  );
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX-512 ayuv64<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_ayuv64(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::ayuv64_to_luma_row(&p, &mut s, width);
  unsafe {
    ayuv64_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX-512 ayuv64→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_ayuv64(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::ayuv64_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    ayuv64_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX-512 ayuv64→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_ayuv64_rgb_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      // u8 path: SIMD main loop has block size 64 — width 64 = one main-loop
      // iteration with no scalar tail, so this exercises the AVX-512 u8 SIMD
      // code under EVERY matrix × range combo (not just BT.709 from the
      // width-sweep test). Required to catch matrix-specific coefficient /
      // sign / lane bugs in the 64-pixel u8 SIMD path.
      check_rgb::<false, false>(64, m, full); // u8 RGB (one main-loop iter)
      check_rgb::<true, true>(64, m, full); // u8 RGBA + source α (one main-loop iter)
      // u16 path: SIMD main loop has block size 32 — width 32 = one main-loop
      // iteration with no scalar tail, exercises the i64-chroma u16 SIMD code.
      check_rgb_u16::<false, false>(32, m, full); // u16 RGB (one main-loop iter)
      check_rgb_u16::<true, true>(32, m, full); // u16 RGBA + source α (one main-loop iter)
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_ayuv64_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  // Width sweep covers:
  //   - tail-only widths < 32 (no SIMD main loop in u16 path)
  //   - u16 SIMD block boundary at 32 (one main-loop iteration of u16 path)
  //   - u8 SIMD block-boundary 64 (one main-loop iteration of u8 path, no tail)
  //   - partial-block-plus-tail 95/96/97 (one u8 main-loop + 31/32/33-px tail)
  //   - production 1920p widths and odd tails (1921, 1923).
  for w in [
    1usize, 2, 3, 31, 32, 33, 63, 64, 65, 95, 96, 97, 1920, 1921, 1923,
  ] {
    check_rgb::<false, false>(w, ColorMatrix::Bt709, false);
    check_rgb::<true, true>(w, ColorMatrix::Bt709, true);
    check_rgb_u16::<false, false>(w, ColorMatrix::Bt2020Ncl, true);
    check_rgb_u16::<true, true>(w, ColorMatrix::Bt601, false);
    check_luma(w);
    check_luma_u16(w);
  }
}

/// Multi-channel Y+A lane-order regression test.
///
/// Encodes Y[n] = n + 1 (range 1..=32, fits u16) AND A[n] = 2n + 1
/// (range 1..=63, fits u16) for n in 0..32.
///
/// Uses 32 pixels to match the AVX-512 u16 SIMD block size (one full
/// main-loop iteration, no tail). Also exercises the u16 luma kernel
/// (32 px/iter) at exactly one block.
///
/// Uses neutral chroma (U = V = 32768) so chroma contribution is zero,
/// making the conversion purely Y-dependent.
///
/// Asserts:
/// - `luma_u16_row` output = `[1, 2, 3, …, 32]` (Y values direct)
/// - α u16 channel (slot 3 of each RGBA u16 quadruple) = `[1, 3, 5, …, 63]`
///
/// This catches:
/// - Cross-vector permute-index bugs (`A_FROM_PAIR_IDX` /
///   `Y_FROM_PAIR_IDX` gathering from the wrong channel slot).
/// - `COMBINE_IDX` swap / scramble bugs at the round-2 step.
/// - Lane-fixup bugs in the u16 pack path
///   (`_mm512_packus_epi32` + `pack_fixup`).
/// - Channel-tuple destructuring confusion (e.g., A and Y slots swapped).
///
/// It confirms that A and Y lanes are deinterleaved into the correct
/// AVX-512 channel vectors (slot 0 = A, slot 1 = Y) and that neither
/// bleeds into the other across the entire 32-pixel block.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_ayuv64_lane_order_per_pixel_y_and_a() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }

  const W: usize = 32;
  // Build packed AYUV64: A[n]=2n+1, Y[n]=n+1, U=32768, V=32768.
  let mut packed = std::vec::Vec::with_capacity(W * 4);
  for n in 0..W {
    packed.push((2 * n + 1) as u16); // slot 0 = A
    packed.push((n + 1) as u16); // slot 1 = Y
    packed.push(32768u16); // slot 2 = U (neutral — bias-subtracted = 0)
    packed.push(32768u16); // slot 3 = V (neutral)
  }

  // --- luma_u16 path: Y values should be direct (no conversion). ---
  let mut luma_out = std::vec![0u16; W];
  unsafe {
    ayuv64_to_luma_u16_row(&packed, &mut luma_out, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(
    luma_out, expected_luma,
    "luma_u16: Y lane order incorrect — expected Y[n]=n+1, got {luma_out:?}"
  );

  // --- RGBA u16 path: α channel (slot 3 of output) = A values direct. ---
  // Use full_range=true so neutral chroma (bias-subtracted = 0) gives
  // a well-defined Y output. Matrix choice does not affect neutral chroma.
  let mut rgba_out = std::vec![0u16; W * 4];
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<true, true>(
      &packed,
      &mut rgba_out,
      W,
      ColorMatrix::Bt709,
      true, // full_range
    );
  }
  // α is at slot 3 (index 3) of each RGBA quadruple in the output.
  let alpha_out: std::vec::Vec<u16> = (0..W).map(|n| rgba_out[n * 4 + 3]).collect();
  let expected_alpha: std::vec::Vec<u16> = (0..W as u16).map(|n| 2 * n + 1).collect();
  assert_eq!(
    alpha_out, expected_alpha,
    "rgba_u16: A lane order incorrect — expected A[n]=2n+1, got {alpha_out:?}"
  );
}
