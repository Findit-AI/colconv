use super::super::*;
use crate::{KernelMatrix, row::scalar};

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
    "NEON vuya<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
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
  assert_eq!(s, k, "NEON vuya→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_vuya(width, 0xBEEF);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::vuya_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    vuya_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON vuya→luma_u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_vuya_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      // All 3 valid (ALPHA, ALPHA_SRC) combinations.
      check_rgb::<false, false>(16, m, full); // RGB
      check_rgb::<true, true>(16, m, full); // RGBA + source alpha (VUYA)
      check_rgb::<true, false>(16, m, full); // RGBA + forced 0xFF (VUYX)
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_vuya_matches_scalar_widths() {
  for w in [
    1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921, 1923,
  ] {
    check_rgb::<false, false>(w, KernelMatrix::Bt709, false);
    check_rgb::<true, true>(w, KernelMatrix::Bt709, true);
    check_rgb::<true, false>(w, KernelMatrix::Bt2020Ncl, true);
    check_luma(w);
    check_luma_u16(w);
  }
}

/// Multi-channel lane-order regression — encodes pixel index in
/// BOTH Y AND A so we catch per-channel asymmetric mask bugs that
/// a Y-only test would miss. Pattern from Ship 12d AYUV64 backport.
/// VUYA has source α — assert the α slot directly.
///
/// NEON SIMD threshold: 16 px/iter (`vld4q_u8`). W=32 covers exactly
/// 2 full SIMD iterations.
///
/// NEON has no runtime CPU detection guard on aarch64 targets where
/// NEON is mandatory — no `is_*_feature_detected!` needed.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_vuya_lane_order_per_pixel_y_and_a() {
  const W: usize = 32;
  let packed = build_vuya_packed_y_n_plus_1_a_2n_plus_1_u_v_neutral(W);

  // Part 1: Luma natural-order (u8 path, Y is direct).
  let mut luma = std::vec![0u8; W];
  unsafe {
    vuya_to_luma_row(&packed, &mut luma, W);
  }
  let expected_luma: std::vec::Vec<u8> = (1..=W as u8).collect();
  assert_eq!(luma, expected_luma, "neon vuya luma reorder bug");

  // Part 2: u8 RGBA — α slot (every 4th byte) directly verifies
  // A-channel deinterleave. neutral U/V → chroma contribution is zero.
  let mut rgba = std::vec![0u8; W * 4];
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, true>(&packed, &mut rgba, W, KernelMatrix::Bt709, false);
  }
  let alpha_out: std::vec::Vec<u8> = (0..W).map(|n| rgba[n * 4 + 3]).collect();
  let expected_alpha: std::vec::Vec<u8> = (0..W).map(|n| (n as u8) * 2 + 1).collect();
  assert_eq!(alpha_out, expected_alpha, "neon vuya rgba α reorder bug");
}

// ---- AYUV / UYVA / VYU444 NEON-vs-scalar parity -------------------------
//
// The new formats are byte-for-byte channel re-orderings of VUYA / VUYX
// (AYUV / UYVA: 4 bytes; VYU444: 3 bytes, no alpha). Each NEON kernel is
// checked against its scalar reference over a spread of widths (covering
// the SIMD block boundary at 16) and all colour matrices.

const NEON_MATRICES: [KernelMatrix; 6] = [
  KernelMatrix::Bt601,
  KernelMatrix::Bt709,
  KernelMatrix::Bt2020Ncl,
  KernelMatrix::Smpte240m,
  KernelMatrix::Fcc,
  KernelMatrix::YCgCo,
];
const NEON_WIDTHS: [usize; 13] = [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 64, 1921];

/// Pseudo-random packed stream of `width * bytes_per_pixel` bytes.
fn pseudo_random(width: usize, bpp: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * bpp)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed.wrapping_mul(3));
      (s & 0xFF) as u8
    })
    .collect()
}

/// Same-tier HSV reference: stage RGB via `fill_rgb` (the NEON RGB kernel),
/// then quantize with the NEON [`rgb_to_hsv_row`]. This is the crate's HSV
/// contract — a fused `{fmt}_to_hsv_row` is byte-identical to
/// `rgb_to_hsv_row({fmt}_to_rgb_row(...))` *within a tier*, not necessarily
/// to the fused scalar (the NEON quantizer can differ ±1 LSB from scalar).
fn neon_hsv_ref(
  w: usize,
  _matrix: KernelMatrix,
  _full_range: bool,
  fill_rgb: impl FnOnce(&mut [u8]),
) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let mut rgb = std::vec![0u8; w * 3];
  fill_rgb(&mut rgb);
  let (mut h, mut s, mut v) = (std::vec![0u8; w], std::vec![0u8; w], std::vec![0u8; w]);
  unsafe {
    rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, w);
  }
  (h, s, v)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_ayuv_matches_scalar() {
  for &w in &NEON_WIDTHS {
    let p = pseudo_random(w, 4, 0x1234);
    for &m in &NEON_MATRICES {
      for &fr in &[false, true] {
        let mut s = std::vec![0u8; w * 3];
        let mut k = std::vec![0u8; w * 3];
        scalar::ayuv_to_rgb_row(&p, &mut s, w, m, fr);
        unsafe { ayuv_to_rgb_row(&p, &mut k, w, m, fr) };
        assert_eq!(s, k, "AYUV NEON RGB (w={w}, m={m:?}, fr={fr})");
        let mut s4 = std::vec![0u8; w * 4];
        let mut k4 = std::vec![0u8; w * 4];
        scalar::ayuv_to_rgba_row(&p, &mut s4, w, m, fr);
        unsafe { ayuv_to_rgba_row(&p, &mut k4, w, m, fr) };
        assert_eq!(s4, k4, "AYUV NEON RGBA (w={w}, m={m:?}, fr={fr})");
      }
    }
    let mut sl = std::vec![0u8; w];
    let mut kl = std::vec![0u8; w];
    scalar::ayuv_to_luma_row(&p, &mut sl, w);
    unsafe { ayuv_to_luma_row(&p, &mut kl, w) };
    assert_eq!(sl, kl, "AYUV NEON luma (w={w})");
    let mut su = std::vec![0u16; w];
    let mut ku = std::vec![0u16; w];
    scalar::ayuv_to_luma_u16_row(&p, &mut su, w);
    unsafe { ayuv_to_luma_u16_row(&p, &mut ku, w) };
    assert_eq!(su, ku, "AYUV NEON luma_u16 (w={w})");
    // HSV contract: NEON `{fmt}_to_hsv_row` must equal the same-tier
    // two-step `rgb_to_hsv_row(NEON {fmt}_to_rgb_row(...))` (NOT the fused
    // scalar, which can differ by ±1 LSB from the NEON HSV quantizer).
    let want = neon_hsv_ref(w, KernelMatrix::Bt709, false, |rgb| unsafe {
      ayuv_to_rgb_row(&p, rgb, w, KernelMatrix::Bt709, false)
    });
    let (mut kh, mut ks, mut kv) = (std::vec![0u8; w], std::vec![0u8; w], std::vec![0u8; w]);
    unsafe { ayuv_to_hsv_row(&p, &mut kh, &mut ks, &mut kv, w, KernelMatrix::Bt709, false) };
    assert_eq!(want, (kh, ks, kv), "AYUV NEON hsv (w={w})");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_uyva_matches_scalar() {
  for &w in &NEON_WIDTHS {
    let p = pseudo_random(w, 4, 0x55AA);
    for &m in &NEON_MATRICES {
      for &fr in &[false, true] {
        let mut s = std::vec![0u8; w * 3];
        let mut k = std::vec![0u8; w * 3];
        scalar::uyva_to_rgb_row(&p, &mut s, w, m, fr);
        unsafe { uyva_to_rgb_row(&p, &mut k, w, m, fr) };
        assert_eq!(s, k, "UYVA NEON RGB (w={w}, m={m:?}, fr={fr})");
        let mut s4 = std::vec![0u8; w * 4];
        let mut k4 = std::vec![0u8; w * 4];
        scalar::uyva_to_rgba_row(&p, &mut s4, w, m, fr);
        unsafe { uyva_to_rgba_row(&p, &mut k4, w, m, fr) };
        assert_eq!(s4, k4, "UYVA NEON RGBA (w={w}, m={m:?}, fr={fr})");
      }
    }
    let mut sl = std::vec![0u8; w];
    let mut kl = std::vec![0u8; w];
    scalar::uyva_to_luma_row(&p, &mut sl, w);
    unsafe { uyva_to_luma_row(&p, &mut kl, w) };
    assert_eq!(sl, kl, "UYVA NEON luma (w={w})");
    let mut su = std::vec![0u16; w];
    let mut ku = std::vec![0u16; w];
    scalar::uyva_to_luma_u16_row(&p, &mut su, w);
    unsafe { uyva_to_luma_u16_row(&p, &mut ku, w) };
    assert_eq!(su, ku, "UYVA NEON luma_u16 (w={w})");
    let want = neon_hsv_ref(w, KernelMatrix::Bt2020Ncl, true, |rgb| unsafe {
      uyva_to_rgb_row(&p, rgb, w, KernelMatrix::Bt2020Ncl, true)
    });
    let (mut kh, mut ks, mut kv) = (std::vec![0u8; w], std::vec![0u8; w], std::vec![0u8; w]);
    unsafe {
      uyva_to_hsv_row(
        &p,
        &mut kh,
        &mut ks,
        &mut kv,
        w,
        KernelMatrix::Bt2020Ncl,
        true,
      )
    };
    assert_eq!(want, (kh, ks, kv), "UYVA NEON hsv (w={w})");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_vyu444_matches_scalar() {
  for &w in &NEON_WIDTHS {
    let p = pseudo_random(w, 3, 0xC0DE); // 3 bytes per pixel
    for &m in &NEON_MATRICES {
      for &fr in &[false, true] {
        let mut s = std::vec![0u8; w * 3];
        let mut k = std::vec![0u8; w * 3];
        scalar::vyu444_to_rgb_row(&p, &mut s, w, m, fr);
        unsafe { vyu444_to_rgb_row(&p, &mut k, w, m, fr) };
        assert_eq!(s, k, "VYU444 NEON RGB (w={w}, m={m:?}, fr={fr})");
        let mut s4 = std::vec![0u8; w * 4];
        let mut k4 = std::vec![0u8; w * 4];
        scalar::vyu444_to_rgba_row(&p, &mut s4, w, m, fr);
        unsafe { vyu444_to_rgba_row(&p, &mut k4, w, m, fr) };
        assert_eq!(s4, k4, "VYU444 NEON RGBA (w={w}, m={m:?}, fr={fr})");
      }
    }
    let mut sl = std::vec![0u8; w];
    let mut kl = std::vec![0u8; w];
    scalar::vyu444_to_luma_row(&p, &mut sl, w);
    unsafe { vyu444_to_luma_row(&p, &mut kl, w) };
    assert_eq!(sl, kl, "VYU444 NEON luma (w={w})");
    let mut su = std::vec![0u16; w];
    let mut ku = std::vec![0u16; w];
    scalar::vyu444_to_luma_u16_row(&p, &mut su, w);
    unsafe { vyu444_to_luma_u16_row(&p, &mut ku, w) };
    assert_eq!(su, ku, "VYU444 NEON luma_u16 (w={w})");
    let want = neon_hsv_ref(w, KernelMatrix::Bt601, false, |rgb| unsafe {
      vyu444_to_rgb_row(&p, rgb, w, KernelMatrix::Bt601, false)
    });
    let (mut kh, mut ks, mut kv) = (std::vec![0u8; w], std::vec![0u8; w], std::vec![0u8; w]);
    unsafe { vyu444_to_hsv_row(&p, &mut kh, &mut ks, &mut kv, w, KernelMatrix::Bt601, false) };
    assert_eq!(want, (kh, ks, kv), "VYU444 NEON hsv (w={w})");
  }
}

/// Lane-order regression for the 3-byte VYU444 deinterleave: encode the
/// pixel index in Y so a per-channel swizzle bug surfaces. Neutral V/U
/// keeps the colour at grey; the luma path reads Y directly.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_vyu444_lane_order_y() {
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
  assert_eq!(luma, expected, "neon vyu444 luma reorder bug");
}
