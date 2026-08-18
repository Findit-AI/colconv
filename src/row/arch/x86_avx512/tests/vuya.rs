use super::super::*;
use crate::{KernelMatrix, row::scalar};

/// Build a deterministic pseudo-random VUYA packed stream.
/// Returns `width * 4` bytes with channels varying across all 8-bit values.
fn pseudo_random_vuya(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 4)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed.wrapping_mul(3));
      (s & 0xFF) as u8
    })
    .collect()
}

fn check_rgb<const ALPHA: bool, const ALPHA_SRC: bool>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let p = pseudo_random_vuya(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut s, width, matrix, full_range);
  unsafe {
    vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX-512 vuya<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_vuya(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::vuya_to_luma_row(&p, &mut s, width);
  unsafe {
    vuya_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX-512 vuya→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_vuya(width, 0xBEEF);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::vuya_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    vuya_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX-512 vuya→luma_u16 diverges (width={width})");
}

/// Build a VUYA packed stream with Y[n] = n+1, A[n] = 2n+1, V=U=128.
///
/// VUYA layout per pixel: `[V(8), U(8), Y(8), A(8)]`. Source α is real
/// (not padding). Encoding:
/// - V = 128 (neutral 8-bit midpoint)
/// - U = 128 (neutral)
/// - Y[n] = n + 1
/// - A[n] = 2n + 1  (source α — distinct values per pixel)
fn build_vuya_packed_y_n_plus_1_a_2n_plus_1_u_v_neutral(width: usize) -> std::vec::Vec<u8> {
  let mut packed = std::vec![0u8; width * 4];
  for n in 0..width {
    packed[n * 4] = 128; // V
    packed[n * 4 + 1] = 128; // U
    packed[n * 4 + 2] = (n as u8) + 1; // Y = n+1
    packed[n * 4 + 3] = (n as u8) * 2 + 1; // A = 2n+1
  }
  packed
}

/// Multi-channel lane-order regression — encodes pixel index in
/// BOTH Y AND A so we catch per-channel asymmetric mask bugs that
/// the previous Y-only test would miss. Pattern from Ship 12d
/// AYUV64 backport. VUYA has source α — assert the α slot directly.
///
/// AVX-512 SIMD threshold: 64 px/iter. W=128 covers exactly 2 full
/// SIMD iterations. For n in 0..127: Y[n]=n+1 ≤ 128 and A[n]=2n+1 ≤ 255
/// — both fit in u8.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_vuya_lane_order_per_pixel_y_and_a() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  const W: usize = 128;
  let packed = build_vuya_packed_y_n_plus_1_a_2n_plus_1_u_v_neutral(W);

  // Part 1: Luma natural-order (u8 path, Y is direct).
  let mut luma = std::vec![0u8; W];
  unsafe {
    vuya_to_luma_row(&packed, &mut luma, W);
  }
  let expected_luma: std::vec::Vec<u8> = (1..=W as u8).collect();
  assert_eq!(luma, expected_luma, "avx512 vuya luma reorder bug");

  // Part 2: u8 RGBA — α slot (every 4th byte) directly verifies
  // A-channel deinterleave. neutral U/V → chroma contribution is zero.
  let mut rgba = std::vec![0u8; W * 4];
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, true>(&packed, &mut rgba, W, KernelMatrix::Bt709, false);
  }
  let alpha_out: std::vec::Vec<u8> = (0..W).map(|n| rgba[n * 4 + 3]).collect();
  let expected_alpha: std::vec::Vec<u8> = (0..W).map(|n| (n as u8) * 2 + 1).collect();
  assert_eq!(alpha_out, expected_alpha, "avx512 vuya rgba α reorder bug");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_vuya_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  // Width sweep covers:
  //   - tail-only widths < 64 (no SIMD main loop)
  //   - the SIMD block-boundary 64 (one main-loop iteration, no tail)
  //   - partial-block widths 95/96/97 (one main-loop + 31/32/33-px tail)
  //   - production 1920p widths and odd tails (1921, 1923).
  for w in [
    1usize, 2, 3, 31, 32, 33, 63, 64, 65, 95, 96, 97, 1920, 1921, 1923,
  ] {
    check_rgb::<false, false>(w, KernelMatrix::Bt709, false);
    check_rgb::<true, true>(w, KernelMatrix::Bt709, true);
    check_rgb::<true, false>(w, KernelMatrix::Bt2020Ncl, true);
    check_luma(w);
    check_luma_u16(w);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_vuya_rgb_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      // Width 64 = one main-loop iteration (no tail).
      check_rgb::<false, false>(64, m, full);
      check_rgb::<true, true>(64, m, full);
      check_rgb::<true, false>(64, m, full);
    }
  }
}

// ===== VYU444 (3-byte, no alpha) AVX-512 parity ========================
//
// SDE-gated (`is_x86_feature_detected!`) — compile-checked on every x86
// host, run under Intel SDE in CI. The AVX-512 VYU444 path dispatches to
// the SSE4.1 16-px 3-byte de-interleave kernel (the 3-byte stride does not
// tile onto 512-bit lanes), so this exercises that delegation end-to-end.

/// Deterministic pseudo-random VYU444 packed stream (`width * 3` bytes,
/// 3 bytes per pixel: `V ‖ Y ‖ U`).
fn pseudo_random_vyu444(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 3)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed.wrapping_mul(3));
      (s & 0xFF) as u8
    })
    .collect()
}

/// Same-tier HSV reference: stage RGB via the AVX-512 VYU444 RGB kernel,
/// then quantize with the AVX-512 `rgb_to_hsv_row`. Per-tier identity —
/// not the fused scalar (the SIMD quantizer can differ ±1 LSB).
fn avx512_vyu444_hsv_ref(
  packed: &[u8],
  w: usize,
  m: KernelMatrix,
  fr: bool,
) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let mut rgb = std::vec![0u8; w * 3];
  unsafe {
    vyu444_to_rgb_row(packed, &mut rgb, w, m, fr);
  }
  let (mut h, mut s, mut v) = (std::vec![0u8; w], std::vec![0u8; w], std::vec![0u8; w]);
  unsafe {
    rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, w);
  }
  (h, s, v)
}

/// VYU444 AVX-512 parity against scalar: RGB / RGBA (α forced `0xFF`) /
/// luma u8 / luma u16 over every matrix + range, plus the per-tier HSV
/// identity.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_vyu444_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  for w in [
    1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 1920, 1921, 1923,
  ] {
    let p = pseudo_random_vyu444(w, 0xC0DE);
    for m in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      for fr in [false, true] {
        let mut s = std::vec![0u8; w * 3];
        let mut k = std::vec![0u8; w * 3];
        scalar::vyu444_to_rgb_row(&p, &mut s, w, m, fr);
        unsafe { vyu444_to_rgb_row(&p, &mut k, w, m, fr) };
        assert_eq!(s, k, "AVX-512 VYU444 RGB (w={w}, m={m:?}, fr={fr})");

        let mut s4 = std::vec![0u8; w * 4];
        let mut k4 = std::vec![0u8; w * 4];
        scalar::vyu444_to_rgba_row(&p, &mut s4, w, m, fr);
        unsafe { vyu444_to_rgba_row(&p, &mut k4, w, m, fr) };
        assert_eq!(s4, k4, "AVX-512 VYU444 RGBA (w={w}, m={m:?}, fr={fr})");
      }
    }
    let mut sl = std::vec![0u8; w];
    let mut kl = std::vec![0u8; w];
    scalar::vyu444_to_luma_row(&p, &mut sl, w);
    unsafe { vyu444_to_luma_row(&p, &mut kl, w) };
    assert_eq!(sl, kl, "AVX-512 VYU444 luma (w={w})");

    let mut su = std::vec![0u16; w];
    let mut ku = std::vec![0u16; w];
    scalar::vyu444_to_luma_u16_row(&p, &mut su, w);
    unsafe { vyu444_to_luma_u16_row(&p, &mut ku, w) };
    assert_eq!(su, ku, "AVX-512 VYU444 luma_u16 (w={w})");

    let want = avx512_vyu444_hsv_ref(&p, w, KernelMatrix::Bt601, false);
    let (mut kh, mut ks, mut kv) = (std::vec![0u8; w], std::vec![0u8; w], std::vec![0u8; w]);
    unsafe { vyu444_to_hsv_row(&p, &mut kh, &mut ks, &mut kv, w, KernelMatrix::Bt601, false) };
    assert_eq!(want, (kh, ks, kv), "AVX-512 VYU444 hsv (w={w})");
  }
}

/// Lane-order regression for the 3-byte VYU444 de-interleave: encode the
/// pixel index in Y so a per-channel swizzle bug surfaces. Neutral V/U
/// keeps the colour at grey; the luma path reads Y directly.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_vyu444_lane_order_y() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  const W: usize = 32;
  let mut packed = std::vec![0u8; W * 3];
  for n in 0..W {
    packed[n * 3] = 128; // V
    packed[n * 3 + 1] = (n as u8) + 1; // Y = n+1
    packed[n * 3 + 2] = 128; // U
  }
  let mut luma = std::vec![0u8; W];
  unsafe { vyu444_to_luma_row(&packed, &mut luma, W) };
  let expected: std::vec::Vec<u8> = (1..=W as u8).collect();
  assert_eq!(luma, expected, "avx512 vyu444 luma reorder bug");
}
