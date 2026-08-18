use super::super::*;

// All tests in this file are for x86_64 AVX2. Each test must call
// `is_x86_feature_detected!("avx2")` and early-return when not available,
// so CI sanitizer / Miri on non-x86 hosts does not trip.
//
// F16C-gated tests additionally call `is_x86_feature_detected!("f16c")`.
//
// Lane-order regression tests use asymmetric R/G/B/A patterns
// (`R[n] = n+1`, `G[n] = 2n+1`, `B[n] = 3n+1`, `A[n] = 4n+1`) — uniform-
// input tests miss per-channel mask bugs (see the AYUV64 lane-order
// failure mode). The per-pixel asymmetry distinguishes channels after
// interleave so a swapped R/G/B mask trips the assertion.

const WIDTHS: &[usize] = &[1, 4, 5, 7, 8, 9, 15, 16, 17, 24, 32, 33, 64, 65, 128, 130];

/// Pseudo-random f32 values in [-0.1, 1.3] — includes HDR > 1.0 and
/// slightly negative (to exercise clamping code paths).
fn prng_f32(out: &mut [f32], seed: u32) {
  let mut s = seed;
  for v in out.iter_mut() {
    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *v = ((s >> 8) as f32) / (u32::MAX as f32) * 1.4 - 0.1;
  }
}

/// Pseudo-random f16 values (derived from f32 PRNG, then narrowed to f16).
fn prng_f16(out: &mut [half::f16], seed: u32) {
  let mut buf = std::vec![0.0f32; out.len()];
  prng_f32(&mut buf, seed);
  for (o, v) in out.iter_mut().zip(buf.iter()) {
    *o = half::f16::from_f32(*v);
  }
}

/// Asymmetric per-channel ramp f32 input. Used by lane-order regression
/// tests so each output channel is uniquely identifiable. Values stay in
/// [0, 1) by capping at width = 60 (60/255 ≈ 0.235).
fn asym_ramp_f32(g: &mut [f32], b: &mut [f32], r: &mut [f32]) {
  for n in 0..g.len() {
    g[n] = ((n + 1) * 2) as f32 / 255.0; // 2, 4, 6, ...
    b[n] = ((n + 1) * 3) as f32 / 255.0; // 3, 6, 9, ...
    r[n] = (n + 1) as f32 / 255.0; // 1, 2, 3, ...
  }
}

/// Asymmetric per-channel ramp f32 input including alpha.
fn asym_ramp_f32_a(g: &mut [f32], b: &mut [f32], r: &mut [f32], a: &mut [f32]) {
  for n in 0..g.len() {
    g[n] = ((n + 1) * 2) as f32 / 255.0;
    b[n] = ((n + 1) * 3) as f32 / 255.0;
    r[n] = (n + 1) as f32 / 255.0;
    a[n] = ((n + 1) * 4) as f32 / 255.0;
  }
}

/// Asymmetric per-channel ramp f16 input.
fn asym_ramp_f16(g: &mut [half::f16], b: &mut [half::f16], r: &mut [half::f16]) {
  for n in 0..g.len() {
    g[n] = half::f16::from_f32(((n + 1) * 2) as f32 / 255.0);
    b[n] = half::f16::from_f32(((n + 1) * 3) as f32 / 255.0);
    r[n] = half::f16::from_f32((n + 1) as f32 / 255.0);
  }
}

/// Asymmetric per-channel ramp f16 input including alpha.
fn asym_ramp_f16_a(
  g: &mut [half::f16],
  b: &mut [half::f16],
  r: &mut [half::f16],
  a: &mut [half::f16],
) {
  for n in 0..g.len() {
    g[n] = half::f16::from_f32(((n + 1) * 2) as f32 / 255.0);
    b[n] = half::f16::from_f32(((n + 1) * 3) as f32 / 255.0);
    r[n] = half::f16::from_f32((n + 1) as f32 / 255.0);
    a[n] = half::f16::from_f32(((n + 1) * 4) as f32 / 255.0);
  }
}

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA001_0001);
    prng_f32(&mut b, 0xA001_0002);
    prng_f32(&mut r, 0xA001_0003);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgb_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    asym_ramp_f32(&mut g, &mut b, &mut r);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb lane-order width={w}");
  }
}

// ---- Gbrpf32 → u8 RGBA ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA002_0001);
    prng_f32(&mut b, 0xA002_0002);
    prng_f32(&mut r, 0xA002_0003);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgba_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    asym_ramp_f32(&mut g, &mut b, &mut r);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba lane-order width={w}");
  }
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA003_0001);
    prng_f32(&mut b, 0xA003_0002);
    prng_f32(&mut r, 0xA003_0003);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgb_u16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    asym_ramp_f32(&mut g, &mut b, &mut r);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_u16 lane-order width={w}");
  }
}

// ---- Gbrpf32 → u16 RGBA -----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA004_0001);
    prng_f32(&mut b, 0xA004_0002);
    prng_f32(&mut r, 0xA004_0003);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgba_u16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    asym_ramp_f32(&mut g, &mut b, &mut r);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_u16 lane-order width={w}");
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgb_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA005_0001);
    prng_f32(&mut b, 0xA005_0002);
    prng_f32(&mut r, 0xA005_0003);
    let mut simd = std::vec![0.0f32; w * 3];
    let mut scal = std::vec![0.0f32; w * 3];
    unsafe { gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_f32 width={w}");
  }
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgba_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA006_0001);
    prng_f32(&mut b, 0xA006_0002);
    prng_f32(&mut r, 0xA006_0003);
    let mut simd = std::vec![0.0f32; w * 4];
    let mut scal = std::vec![0.0f32; w * 4];
    unsafe { gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_f32 width={w}");
  }
}

// ---- Gbrpf32 → f16 RGB (F16C narrow) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgb_f16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA007_0001);
    prng_f32(&mut b, 0xA007_0002);
    prng_f32(&mut r, 0xA007_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 3];
    let mut scal = std::vec![half::f16::ZERO; w * 3];
    unsafe { gbrpf32_to_rgb_f16_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_f16 (F16C) width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgb_f16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    asym_ramp_f32(&mut g, &mut b, &mut r);
    let mut simd = std::vec![half::f16::ZERO; w * 3];
    let mut scal = std::vec![half::f16::ZERO; w * 3];
    unsafe { gbrpf32_to_rgb_f16_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_f16 lane-order width={w}");
  }
}

// ---- Gbrpf32 → f16 RGBA (F16C narrow, α = f16(1.0)) -------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgba_f16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA008_0001);
    prng_f32(&mut b, 0xA008_0002);
    prng_f32(&mut r, 0xA008_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrpf32_to_rgba_f16_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_f16 (F16C) width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgba_f16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    asym_ramp_f32(&mut g, &mut b, &mut r);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrpf32_to_rgba_f16_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_f16 lane-order width={w}");
  }
}

// ---- Gbrpf32 → u8 luma ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_luma_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA009_0001);
    prng_f32(&mut b, 0xA009_0002);
    prng_f32(&mut r, 0xA009_0003);
    let mut simd = std::vec![0u8; w];
    let mut scal = std::vec![0u8; w];
    unsafe { gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut simd, w, KernelMatrix::Bt709, true) };
    scalar::planar_gbr_float::gbrpf32_to_luma_row::<false>(
      &g,
      &b,
      &r,
      &mut scal,
      w,
      KernelMatrix::Bt709,
      true,
    );
    assert_eq!(simd, scal, "gbrpf32_to_luma width={w}");
  }
}

// ---- Gbrpf32 → u16 luma -----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_luma_u16_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA00A_0001);
    prng_f32(&mut b, 0xA00A_0002);
    prng_f32(&mut r, 0xA00A_0003);
    let mut simd = std::vec![0u16; w];
    let mut scal = std::vec![0u16; w];
    unsafe {
      gbrpf32_to_luma_u16_row::<false>(&g, &b, &r, &mut simd, w, KernelMatrix::Bt709, true)
    };
    scalar::planar_gbr_float::gbrpf32_to_luma_u16_row::<false>(
      &g,
      &b,
      &r,
      &mut scal,
      w,
      KernelMatrix::Bt709,
      true,
    );
    assert_eq!(simd, scal, "gbrpf32_to_luma_u16 width={w}");
  }
}

// ---- Gbrpf32 → HSV ----------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_hsv_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA00B_0001);
    prng_f32(&mut b, 0xA00B_0002);
    prng_f32(&mut r, 0xA00B_0003);
    let mut simd_h = std::vec![0u8; w];
    let mut simd_s = std::vec![0u8; w];
    let mut simd_v = std::vec![0u8; w];
    let mut scal_h = std::vec![0u8; w];
    let mut scal_s = std::vec![0u8; w];
    let mut scal_v = std::vec![0u8; w];
    unsafe { gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut simd_h, &mut simd_s, &mut simd_v, w) };
    scalar::planar_gbr_float::gbrpf32_to_hsv_row::<false>(
      &g,
      &b,
      &r,
      &mut scal_h,
      &mut scal_s,
      &mut scal_v,
      w,
    );
    assert_eq!(simd_h, scal_h, "gbrpf32 hsv H width={w}");
    assert_eq!(simd_s, scal_s, "gbrpf32 hsv S width={w}");
    assert_eq!(simd_v, scal_v, "gbrpf32 hsv V width={w}");
  }
}

// ---- Gbrapf32 → u8 RGBA (source α) -----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf32_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA00C_0001);
    prng_f32(&mut b, 0xA00C_0002);
    prng_f32(&mut r, 0xA00C_0003);
    prng_f32(&mut a, 0xA00C_0004);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf32_to_rgba_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    asym_ramp_f32_a(&mut g, &mut b, &mut r, &mut a);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba lane-order width={w}");
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf32_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA00D_0001);
    prng_f32(&mut b, 0xA00D_0002);
    prng_f32(&mut r, 0xA00D_0003);
    prng_f32(&mut a, 0xA00D_0004);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf32_to_rgba_u16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    asym_ramp_f32_a(&mut g, &mut b, &mut r, &mut a);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_u16 lane-order width={w}");
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless, source α) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf32_to_rgba_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA00E_0001);
    prng_f32(&mut b, 0xA00E_0002);
    prng_f32(&mut r, 0xA00E_0003);
    prng_f32(&mut a, 0xA00E_0004);
    let mut simd = std::vec![0.0f32; w * 4];
    let mut scal = std::vec![0.0f32; w * 4];
    unsafe { gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_f32 width={w}");
  }
}

// ---- Gbrapf32 → f16 RGBA (F16C narrow, source α) ----------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf32_to_rgba_f16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xA00F_0001);
    prng_f32(&mut b, 0xA00F_0002);
    prng_f32(&mut r, 0xA00F_0003);
    prng_f32(&mut a, 0xA00F_0004);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrapf32_to_rgba_f16_row_f16c::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_f16 (F16C) width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf32_to_rgba_f16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    asym_ramp_f32_a(&mut g, &mut b, &mut r, &mut a);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrapf32_to_rgba_f16_row_f16c::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_f16 lane-order width={w}");
  }
}

// ---- Gbrpf16 → u8 RGB (F16C widen) -----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB001_0001);
    prng_f16(&mut b, 0xB001_0002);
    prng_f16(&mut r, 0xB001_0003);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gbrpf16_to_rgb_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgb_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb (F16C widen) width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    asym_ramp_f16(&mut g, &mut b, &mut r);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gbrpf16_to_rgb_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgb_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb lane-order width={w}");
  }
}

// ---- Gbrpf16 → u8 RGBA (F16C widen) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB002_0001);
    prng_f16(&mut b, 0xB002_0002);
    prng_f16(&mut r, 0xB002_0003);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrpf16_to_rgba_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgba_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba (F16C widen) width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    asym_ramp_f16(&mut g, &mut b, &mut r);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrpf16_to_rgba_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgba_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba lane-order width={w}");
  }
}

// ---- Gbrpf16 → u16 RGB (F16C widen) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_u16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB003_0001);
    prng_f16(&mut b, 0xB003_0002);
    prng_f16(&mut r, 0xB003_0003);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { gbrpf16_to_rgb_u16_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgb_u16_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb_u16 (F16C widen) width={w}");
  }
}

// ---- Gbrpf16 → u16 RGBA (F16C widen) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_u16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB004_0001);
    prng_f16(&mut b, 0xB004_0002);
    prng_f16(&mut r, 0xB004_0003);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { gbrpf16_to_rgba_u16_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgba_u16_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba_u16 (F16C widen) width={w}");
  }
}

// ---- Gbrpf16 → f32 RGB (F16C widen, lossless) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_f32_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB005_0001);
    prng_f16(&mut b, 0xB005_0002);
    prng_f16(&mut r, 0xB005_0003);
    let mut simd = std::vec![0.0f32; w * 3];
    let mut scal = std::vec![0.0f32; w * 3];
    unsafe { gbrpf16_to_rgb_f32_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgb_f32_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb_f32 (F16C widen) width={w}");
  }
}

// ---- Gbrpf16 → f32 RGBA (F16C widen, lossless, α = 1.0) --------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_f32_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB006_0001);
    prng_f16(&mut b, 0xB006_0002);
    prng_f16(&mut r, 0xB006_0003);
    let mut simd = std::vec![0.0f32; w * 4];
    let mut scal = std::vec![0.0f32; w * 4];
    unsafe { gbrpf16_to_rgba_f32_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgba_f32_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba_f32 (F16C widen) width={w}");
  }
}

// ---- Gbrpf16 → f16 RGB (lossless, no F16C needed) ---------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_f16_lossless_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB007_0001);
    prng_f16(&mut b, 0xB007_0002);
    prng_f16(&mut r, 0xB007_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 3];
    let mut scal = std::vec![half::f16::ZERO; w * 3];
    unsafe { gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_f16::gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb_f16 lossless width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_f16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    asym_ramp_f16(&mut g, &mut b, &mut r);
    let mut simd = std::vec![half::f16::ZERO; w * 3];
    let mut scal = std::vec![half::f16::ZERO; w * 3];
    unsafe { gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_f16::gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb_f16 lane-order width={w}");
  }
}

// ---- Gbrpf16 → f16 RGBA (lossless, no F16C needed) -------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_f16_lossless_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB008_0001);
    prng_f16(&mut b, 0xB008_0002);
    prng_f16(&mut r, 0xB008_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_f16::gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba_f16 lossless width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_f16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    asym_ramp_f16(&mut g, &mut b, &mut r);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_f16::gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba_f16 lane-order width={w}");
  }
}

// ---- Gbrpf16 → u8 luma (F16C widen) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_luma_f16c_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB009_0001);
    prng_f16(&mut b, 0xB009_0002);
    prng_f16(&mut r, 0xB009_0003);
    let mut simd = std::vec![0u8; w];
    let mut scal = std::vec![0u8; w];
    unsafe {
      gbrpf16_to_luma_row_f16c::<false>(&g, &b, &r, &mut simd, w, KernelMatrix::Bt709, true)
    };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_luma_row::<false>(
      &gf,
      &bf,
      &rf,
      &mut scal,
      w,
      KernelMatrix::Bt709,
      true,
    );
    assert_eq!(simd, scal, "gbrpf16_to_luma (F16C) width={w}");
  }
}

// ---- Gbrpf16 → u16 luma (F16C widen) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_luma_u16_f16c_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB00A_0001);
    prng_f16(&mut b, 0xB00A_0002);
    prng_f16(&mut r, 0xB00A_0003);
    let mut simd = std::vec![0u16; w];
    let mut scal = std::vec![0u16; w];
    unsafe {
      gbrpf16_to_luma_u16_row_f16c::<false>(&g, &b, &r, &mut simd, w, KernelMatrix::Bt709, true)
    };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_luma_u16_row::<false>(
      &gf,
      &bf,
      &rf,
      &mut scal,
      w,
      KernelMatrix::Bt709,
      true,
    );
    assert_eq!(simd, scal, "gbrpf16_to_luma_u16 (F16C) width={w}");
  }
}

// ---- Gbrpf16 → HSV (F16C widen) --------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_hsv_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB00B_0001);
    prng_f16(&mut b, 0xB00B_0002);
    prng_f16(&mut r, 0xB00B_0003);
    let mut simd_h = std::vec![0u8; w];
    let mut simd_s = std::vec![0u8; w];
    let mut simd_v = std::vec![0u8; w];
    let mut scal_h = std::vec![0u8; w];
    let mut scal_s = std::vec![0u8; w];
    let mut scal_v = std::vec![0u8; w];
    unsafe {
      gbrpf16_to_hsv_row_f16c::<false>(&g, &b, &r, &mut simd_h, &mut simd_s, &mut simd_v, w)
    };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_hsv_row::<false>(
      &gf,
      &bf,
      &rf,
      &mut scal_h,
      &mut scal_s,
      &mut scal_v,
      w,
    );
    assert_eq!(simd_h, scal_h, "gbrpf16 hsv H (F16C) width={w}");
    assert_eq!(simd_s, scal_s, "gbrpf16 hsv S (F16C) width={w}");
    assert_eq!(simd_v, scal_v, "gbrpf16 hsv V (F16C) width={w}");
  }
}

// ---- Gbrapf16 → u8 RGBA (F16C widen, source α) ------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB00C_0001);
    prng_f16(&mut b, 0xB00C_0002);
    prng_f16(&mut r, 0xB00C_0003);
    prng_f16(&mut a, 0xB00C_0004);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrapf16_to_rgba_row_f16c::<false>(&g, &b, &r, &a, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let af: std::vec::Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrapf32_to_rgba_row::<false>(&gf, &bf, &rf, &af, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba (F16C widen) width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    asym_ramp_f16_a(&mut g, &mut b, &mut r, &mut a);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrapf16_to_rgba_row_f16c::<false>(&g, &b, &r, &a, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let af: std::vec::Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrapf32_to_rgba_row::<false>(&gf, &bf, &rf, &af, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba lane-order width={w}");
  }
}

// ---- Gbrapf16 → u16 RGBA (F16C widen, source α) -----------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_u16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB00D_0001);
    prng_f16(&mut b, 0xB00D_0002);
    prng_f16(&mut r, 0xB00D_0003);
    prng_f16(&mut a, 0xB00D_0004);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { gbrapf16_to_rgba_u16_row_f16c::<false>(&g, &b, &r, &a, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let af: std::vec::Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrapf32_to_rgba_u16_row::<false>(&gf, &bf, &rf, &af, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba_u16 (F16C widen) width={w}");
  }
}

// ---- Gbrapf16 → f32 RGBA (F16C widen, lossless, source α) ------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 + F16C SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_f32_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB00E_0001);
    prng_f16(&mut b, 0xB00E_0002);
    prng_f16(&mut r, 0xB00E_0003);
    prng_f16(&mut a, 0xB00E_0004);
    let mut simd = std::vec![0.0f32; w * 4];
    let mut scal = std::vec![0.0f32; w * 4];
    unsafe { gbrapf16_to_rgba_f32_row_f16c::<false>(&g, &b, &r, &a, &mut simd, w) };
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let af: std::vec::Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrapf32_to_rgba_f32_row::<false>(&gf, &bf, &rf, &af, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba_f32 (F16C widen) width={w}");
  }
}

// ---- Gbrapf16 → f16 RGBA (lossless, no F16C needed, source α) ---------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_f16_lossless_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xB00F_0001);
    prng_f16(&mut b, 0xB00F_0002);
    prng_f16(&mut r, 0xB00F_0003);
    prng_f16(&mut a, 0xB00F_0004);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_f16::gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba_f16 lossless width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_f16_lane_order() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    if w > 60 {
      continue;
    }
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    asym_ramp_f16_a(&mut g, &mut b, &mut r, &mut a);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_f16::gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba_f16 lane-order width={w}");
  }
}

// ---- BE parity helpers -------------------------------------------------------

fn be_encode_f32(src: &[f32]) -> std::vec::Vec<f32> {
  src
    .iter()
    .map(|v| f32::from_bits(v.to_bits().swap_bytes()))
    .collect()
}

fn be_encode_f16(src: &[half::f16]) -> std::vec::Vec<half::f16> {
  src
    .iter()
    .map(|v| half::f16::from_bits(v.to_bits().swap_bytes()))
    .collect()
}

// ---- BE parity: Gbrpf32 → u8 RGB -------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgb_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xBE01_0001);
    prng_f32(&mut b, 0xBE01_0002);
    prng_f32(&mut r, 0xBE01_0003);
    let mut le_out = std::vec![0u8; w * 3];
    let mut be_out = std::vec![0u8; w * 3];
    unsafe {
      gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f32(&g);
    let b_be = be_encode_f32(&b);
    let r_be = be_encode_f32(&r);
    unsafe {
      gbrpf32_to_rgb_row::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(le_out, be_out, "gbrpf32_to_rgb BE parity width={w}");
  }
}

// ---- BE parity: Gbrpf32 → u8 RGBA ------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf32_to_rgba_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xBE02_0001);
    prng_f32(&mut b, 0xBE02_0002);
    prng_f32(&mut r, 0xBE02_0003);
    let mut le_out = std::vec![0u8; w * 4];
    let mut be_out = std::vec![0u8; w * 4];
    unsafe {
      gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f32(&g);
    let b_be = be_encode_f32(&b);
    let r_be = be_encode_f32(&r);
    unsafe {
      gbrpf32_to_rgba_row::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(le_out, be_out, "gbrpf32_to_rgba BE parity width={w}");
  }
}

// ---- BE parity: Gbrpf16 → f16 RGB (lossless) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_f16_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE07_0001);
    prng_f16(&mut b, 0xBE07_0002);
    prng_f16(&mut r, 0xBE07_0003);
    let mut le_out = std::vec![half::f16::ZERO; w * 3];
    let mut be_out = std::vec![half::f16::ZERO; w * 3];
    unsafe {
      gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgb_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(le_out, be_out, "gbrpf16_to_rgb_f16 BE parity width={w}");
  }
}

// ---- BE parity: Gbrapf16 → f16 RGBA (lossless) ------------------------------

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_f16_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE0F_0001);
    prng_f16(&mut b, 0xBE0F_0002);
    prng_f16(&mut r, 0xBE0F_0003);
    prng_f16(&mut a, 0xBE0F_0004);
    let mut le_out = std::vec![half::f16::ZERO; w * 4];
    let mut be_out = std::vec![half::f16::ZERO; w * 4];
    unsafe {
      gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    unsafe {
      gbrapf16_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, w);
    }
    assert_eq!(le_out, be_out, "gbrapf16_to_rgba_f16 BE parity width={w}");
  }
}

// BE parity: SIMD-tail Gbrpf16 → u8/u16 at 8 px lane × non-multiple widths.
//
// The SIMD scalar tail in `gbrpf16_to_*_row_f16c` widens f16 → f32 then
// routes the scalar f32 kernel. Without normalizing the f16 bits first via
// `from_be` / `from_le`, the BE-source-on-LE-host path double-byte-swaps.
// AVX2 lane = 8, so widths 5 / 7 / 33 = 8·0 + 5, 8·0 + 7, 8·4 + 1 — each
// exercises a different non-multiple tail length.
const SIMD_TAIL_WIDTHS: &[usize] = &[5, 7, 33];

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE10_0001);
    prng_f16(&mut b, 0xBE10_0002);
    prng_f16(&mut r, 0xBE10_0003);
    let mut le_out = std::vec![0u8; w * 3];
    let mut be_out = std::vec![0u8; w * 3];
    unsafe {
      gbrpf16_to_rgb_row_f16c::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgb_row_f16c::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrpf16_to_rgb SIMD-tail BE parity width={w}"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE11_0001);
    prng_f16(&mut b, 0xBE11_0002);
    prng_f16(&mut r, 0xBE11_0003);
    let mut le_out = std::vec![0u8; w * 4];
    let mut be_out = std::vec![0u8; w * 4];
    unsafe {
      gbrpf16_to_rgba_row_f16c::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgba_row_f16c::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrpf16_to_rgba SIMD-tail BE parity width={w}"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_u16_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE12_0001);
    prng_f16(&mut b, 0xBE12_0002);
    prng_f16(&mut r, 0xBE12_0003);
    let mut le_out = std::vec![0u16; w * 3];
    let mut be_out = std::vec![0u16; w * 3];
    unsafe {
      gbrpf16_to_rgb_u16_row_f16c::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgb_u16_row_f16c::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrpf16_to_rgb_u16 SIMD-tail BE parity width={w}"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_u16_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE13_0001);
    prng_f16(&mut b, 0xBE13_0002);
    prng_f16(&mut r, 0xBE13_0003);
    let mut le_out = std::vec![0u16; w * 4];
    let mut be_out = std::vec![0u16; w * 4];
    unsafe {
      gbrpf16_to_rgba_u16_row_f16c::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgba_u16_row_f16c::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrpf16_to_rgba_u16 SIMD-tail BE parity width={w}"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE14_0001);
    prng_f16(&mut b, 0xBE14_0002);
    prng_f16(&mut r, 0xBE14_0003);
    prng_f16(&mut a, 0xBE14_0004);
    let mut le_out = std::vec![0u8; w * 4];
    let mut be_out = std::vec![0u8; w * 4];
    unsafe {
      gbrapf16_to_rgba_row_f16c::<false>(&g, &b, &r, &a, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    unsafe {
      gbrapf16_to_rgba_row_f16c::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrapf16_to_rgba SIMD-tail BE parity width={w}"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_u16_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE15_0001);
    prng_f16(&mut b, 0xBE15_0002);
    prng_f16(&mut r, 0xBE15_0003);
    prng_f16(&mut a, 0xBE15_0004);
    let mut le_out = std::vec![0u16; w * 4];
    let mut be_out = std::vec![0u16; w * 4];
    unsafe {
      gbrapf16_to_rgba_u16_row_f16c::<false>(&g, &b, &r, &a, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    unsafe {
      gbrapf16_to_rgba_u16_row_f16c::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrapf16_to_rgba_u16 SIMD-tail BE parity width={w}"
    );
  }
}

// BE parity: SIMD-tail Gbrpf16/Gbrapf16 → f32.
//
// Same bug class as the integer-output tails but for the f16 → f32 lossless
// row kernels: a tail that widens BE-encoded f16 bits via host-native
// `to_f32` and routes scratch through `scalar::gbrpf32_to_*_f32_row::<BE>`
// byte-swaps the (already-wrong) f32 again. AVX2 lane = 8, so widths
// 5 / 7 / 33 each exercise a different non-multiple tail length.

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgb_f32_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE16_0001);
    prng_f16(&mut b, 0xBE16_0002);
    prng_f16(&mut r, 0xBE16_0003);
    let mut le_out = std::vec![0.0f32; w * 3];
    let mut be_out = std::vec![0.0f32; w * 3];
    unsafe {
      gbrpf16_to_rgb_f32_row_f16c::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgb_f32_row_f16c::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::vec::Vec<_>>(),
      be_out
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::vec::Vec<_>>(),
      "gbrpf16_to_rgb_f32 SIMD-tail BE parity width={w}"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrpf16_to_rgba_f32_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE17_0001);
    prng_f16(&mut b, 0xBE17_0002);
    prng_f16(&mut r, 0xBE17_0003);
    let mut le_out = std::vec![0.0f32; w * 4];
    let mut be_out = std::vec![0.0f32; w * 4];
    unsafe {
      gbrpf16_to_rgba_f32_row_f16c::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgba_f32_row_f16c::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::vec::Vec<_>>(),
      be_out
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::vec::Vec<_>>(),
      "gbrpf16_to_rgba_f32 SIMD-tail BE parity width={w}"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_gbrapf16_to_rgba_f32_simd_tail_be_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
    return;
  }
  for &w in SIMD_TAIL_WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE18_0001);
    prng_f16(&mut b, 0xBE18_0002);
    prng_f16(&mut r, 0xBE18_0003);
    prng_f16(&mut a, 0xBE18_0004);
    let mut le_out = std::vec![0.0f32; w * 4];
    let mut be_out = std::vec![0.0f32; w * 4];
    unsafe {
      gbrapf16_to_rgba_f32_row_f16c::<false>(&g, &b, &r, &a, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    unsafe {
      gbrapf16_to_rgba_f32_row_f16c::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, w);
    }
    assert_eq!(
      le_out
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::vec::Vec<_>>(),
      be_out
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::vec::Vec<_>>(),
      "gbrapf16_to_rgba_f32 SIMD-tail BE parity width={w}"
    );
  }
}
