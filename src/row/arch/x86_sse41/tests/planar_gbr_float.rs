use super::super::*;

// All tests in this file are for x86_64 SSE4.1. Each test must call
// `is_x86_feature_detected!("sse4.1")` and early-return when not available,
// so CI sanitizer / Miri on non-x86 hosts does not trip.
//
// F16C-gated tests additionally call `is_x86_feature_detected!("f16c")`.

const WIDTHS: &[usize] = &[1, 4, 5, 7, 8, 16, 17, 32, 33, 128, 130];

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

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF001_0001);
    prng_f32(&mut b, 0xF001_0002);
    prng_f32(&mut r, 0xF001_0003);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb width={w}");
  }
}

// ---- Gbrpf32 → u8 RGBA ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF002_0001);
    prng_f32(&mut b, 0xF002_0002);
    prng_f32(&mut r, 0xF002_0003);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba width={w}");
  }
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF003_0001);
    prng_f32(&mut b, 0xF003_0002);
    prng_f32(&mut r, 0xF003_0003);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_u16 width={w}");
  }
}

// ---- Gbrpf32 → u16 RGBA -----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF004_0001);
    prng_f32(&mut b, 0xF004_0002);
    prng_f32(&mut r, 0xF004_0003);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_u16 width={w}");
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgb_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF005_0001);
    prng_f32(&mut b, 0xF005_0002);
    prng_f32(&mut r, 0xF005_0003);
    let mut simd = std::vec![0.0f32; w * 3];
    let mut scal = std::vec![0.0f32; w * 3];
    unsafe { gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_f32 width={w}");
  }
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgba_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF006_0001);
    prng_f32(&mut b, 0xF006_0002);
    prng_f32(&mut r, 0xF006_0003);
    let mut simd = std::vec![0.0f32; w * 4];
    let mut scal = std::vec![0.0f32; w * 4];
    unsafe { gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_f32 width={w}");
  }
}

// ---- Gbrpf32 → f16 RGB (F16C narrow) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgb_f16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF007_0001);
    prng_f32(&mut b, 0xF007_0002);
    prng_f32(&mut r, 0xF007_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 3];
    let mut scal = std::vec![half::f16::ZERO; w * 3];
    unsafe { gbrpf32_to_rgb_f16_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_f16 (F16C) width={w}");
  }
}

// ---- Gbrpf32 → f16 RGBA (F16C narrow, α = f16(1.0)) -------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgba_f16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF008_0001);
    prng_f32(&mut b, 0xF008_0002);
    prng_f32(&mut r, 0xF008_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrpf32_to_rgba_f16_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_float::gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_f16 (F16C) width={w}");
  }
}

// ---- Gbrpf32 → u8 luma ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_luma_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF009_0001);
    prng_f32(&mut b, 0xF009_0002);
    prng_f32(&mut r, 0xF009_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_luma_u16_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF00A_0001);
    prng_f32(&mut b, 0xF00A_0002);
    prng_f32(&mut r, 0xF00A_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_hsv_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF00B_0001);
    prng_f32(&mut b, 0xF00B_0002);
    prng_f32(&mut r, 0xF00B_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf32_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF00C_0001);
    prng_f32(&mut b, 0xF00C_0002);
    prng_f32(&mut r, 0xF00C_0003);
    prng_f32(&mut a, 0xF00C_0004);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba width={w}");
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf32_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF00D_0001);
    prng_f32(&mut b, 0xF00D_0002);
    prng_f32(&mut r, 0xF00D_0003);
    prng_f32(&mut a, 0xF00D_0004);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_u16 width={w}");
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless, source α) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf32_to_rgba_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF00E_0001);
    prng_f32(&mut b, 0xF00E_0002);
    prng_f32(&mut r, 0xF00E_0003);
    prng_f32(&mut a, 0xF00E_0004);
    let mut simd = std::vec![0.0f32; w * 4];
    let mut scal = std::vec![0.0f32; w * 4];
    unsafe { gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_f32 width={w}");
  }
}

// ---- Gbrapf32 → f16 RGBA (F16C narrow, source α) ----------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf32_to_rgba_f16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    let mut a = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF00F_0001);
    prng_f32(&mut b, 0xF00F_0002);
    prng_f32(&mut r, 0xF00F_0003);
    prng_f32(&mut a, 0xF00F_0004);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrapf32_to_rgba_f16_row_f16c::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_float::gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_f16 (F16C) width={w}");
  }
}

// ---- Gbrpf16 → u8 RGB (F16C widen) -----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgb_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE001_0001);
    prng_f16(&mut b, 0xE001_0002);
    prng_f16(&mut r, 0xE001_0003);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gbrpf16_to_rgb_row_f16c::<false>(&g, &b, &r, &mut simd, w) };
    // Scalar reference: widen f16→f32, then scalar gbrpf32_to_rgb_row.
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgb_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb (F16C widen) width={w}");
  }
}

// ---- Gbrpf16 → u8 RGBA (F16C widen) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgba_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE002_0001);
    prng_f16(&mut b, 0xE002_0002);
    prng_f16(&mut r, 0xE002_0003);
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

// ---- Gbrpf16 → u16 RGB (F16C widen) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgb_u16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE003_0001);
    prng_f16(&mut b, 0xE003_0002);
    prng_f16(&mut r, 0xE003_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgba_u16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE004_0001);
    prng_f16(&mut b, 0xE004_0002);
    prng_f16(&mut r, 0xE004_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgb_f32_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE005_0001);
    prng_f16(&mut b, 0xE005_0002);
    prng_f16(&mut r, 0xE005_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgba_f32_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE006_0001);
    prng_f16(&mut b, 0xE006_0002);
    prng_f16(&mut r, 0xE006_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgb_f16_lossless_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE007_0001);
    prng_f16(&mut b, 0xE007_0002);
    prng_f16(&mut r, 0xE007_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 3];
    let mut scal = std::vec![half::f16::ZERO; w * 3];
    unsafe { gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_f16::gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb_f16 lossless width={w}");
  }
}

// ---- Gbrpf16 → f16 RGBA (lossless, no F16C needed) -------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgba_f16_lossless_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE008_0001);
    prng_f16(&mut b, 0xE008_0002);
    prng_f16(&mut r, 0xE008_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut simd, w) };
    scalar::planar_gbr_f16::gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba_f16 lossless width={w}");
  }
}

// ---- Gbrpf16 → u8 luma (F16C widen) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_luma_f16c_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE009_0001);
    prng_f16(&mut b, 0xE009_0002);
    prng_f16(&mut r, 0xE009_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_luma_u16_f16c_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE00A_0001);
    prng_f16(&mut b, 0xE00A_0002);
    prng_f16(&mut r, 0xE00A_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_hsv_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE00B_0001);
    prng_f16(&mut b, 0xE00B_0002);
    prng_f16(&mut r, 0xE00B_0003);
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
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf16_to_rgba_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE00C_0001);
    prng_f16(&mut b, 0xE00C_0002);
    prng_f16(&mut r, 0xE00C_0003);
    prng_f16(&mut a, 0xE00C_0004);
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

// ---- Gbrapf16 → u16 RGBA (F16C widen, source α) -----------------------------

#[test]
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf16_to_rgba_u16_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE00D_0001);
    prng_f16(&mut b, 0xE00D_0002);
    prng_f16(&mut r, 0xE00D_0003);
    prng_f16(&mut a, 0xE00D_0004);
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
#[cfg_attr(miri, ignore = "SSE4.1 + F16C SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf16_to_rgba_f32_f16c_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") || !std::arch::is_x86_feature_detected!("f16c")
  {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE00E_0001);
    prng_f16(&mut b, 0xE00E_0002);
    prng_f16(&mut r, 0xE00E_0003);
    prng_f16(&mut a, 0xE00E_0004);
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf16_to_rgba_f16_lossless_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    let mut a = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE00F_0001);
    prng_f16(&mut b, 0xE00F_0002);
    prng_f16(&mut r, 0xE00F_0003);
    prng_f16(&mut a, 0xE00F_0004);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe { gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut simd, w) };
    scalar::planar_gbr_f16::gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba_f16 lossless width={w}");
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgb_be_parity() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf32_to_rgba_be_parity() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrpf16_to_rgb_f16_be_parity() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
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
#[cfg_attr(miri, ignore = "SSE4.1 SIMD intrinsics unsupported by Miri")]
fn sse41_gbrapf16_to_rgba_f16_be_parity() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
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
