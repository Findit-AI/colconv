use super::*;

// All tests in this file are for aarch64 NEON; `std` feature is required for Vec.
// f16-widening/narrowing tests additionally gate on `is_aarch64_feature_detected!("fp16")`.

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
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgb_matches_scalar() {
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF001_0001);
    prng_f32(&mut b, 0xF001_0002);
    prng_f32(&mut r, 0xF001_0003);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe {
      gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb width={w}");
  }
}

// ---- Gbrpf32 → u8 RGBA ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgba_matches_scalar() {
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF002_0001);
    prng_f32(&mut b, 0xF002_0002);
    prng_f32(&mut r, 0xF002_0003);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe {
      gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba width={w}");
  }
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgb_u16_matches_scalar() {
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF003_0001);
    prng_f32(&mut b, 0xF003_0002);
    prng_f32(&mut r, 0xF003_0003);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe {
      gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_u16 width={w}");
  }
}

// ---- Gbrpf32 → u16 RGBA -----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgba_u16_matches_scalar() {
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF004_0001);
    prng_f32(&mut b, 0xF004_0002);
    prng_f32(&mut r, 0xF004_0003);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe {
      gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_u16 width={w}");
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgb_f32_matches_scalar() {
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF005_0001);
    prng_f32(&mut b, 0xF005_0002);
    prng_f32(&mut r, 0xF005_0003);
    let mut simd = std::vec![0.0f32; w * 3];
    let mut scal = std::vec![0.0f32; w * 3];
    unsafe {
      gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_f32 width={w}");
  }
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgba_f32_matches_scalar() {
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF006_0001);
    prng_f32(&mut b, 0xF006_0002);
    prng_f32(&mut r, 0xF006_0003);
    let mut simd = std::vec![0.0f32; w * 4];
    let mut scal = std::vec![0.0f32; w * 4];
    unsafe {
      gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_f32 width={w}");
  }
}

// ---- Gbrpf32 → f16 RGB (fp16-gated) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgb_f16_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrpf32_to_rgb_f16_row_fp16::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgb_f16 width={w}");
  }
}

// ---- Gbrpf32 → f16 RGBA (fp16-gated) ---------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgba_f16_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrpf32_to_rgba_f16_row_fp16::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf32_to_rgba_f16 width={w}");
  }
}

// ---- Gbrpf32 → u8 luma ------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_luma_matches_scalar() {
  use crate::KernelMatrix;
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xF009_0001);
    prng_f32(&mut b, 0xF009_0002);
    prng_f32(&mut r, 0xF009_0003);
    let mut simd = std::vec![0u8; w];
    let mut scal = std::vec![0u8; w];
    unsafe {
      gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut simd, w, KernelMatrix::Bt709, true);
    }
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
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_luma_u16_matches_scalar() {
  use crate::KernelMatrix;
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
      gbrpf32_to_luma_u16_row::<false>(&g, &b, &r, &mut simd, w, KernelMatrix::Bt709, true);
    }
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
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_hsv_matches_scalar() {
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
    unsafe {
      gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut simd_h, &mut simd_s, &mut simd_v, w);
    }
    scalar::planar_gbr_float::gbrpf32_to_hsv_row::<false>(
      &g,
      &b,
      &r,
      &mut scal_h,
      &mut scal_s,
      &mut scal_v,
      w,
    );
    assert_eq!(simd_h, scal_h, "hsv H width={w}");
    assert_eq!(simd_s, scal_s, "hsv S width={w}");
    assert_eq!(simd_v, scal_v, "hsv V width={w}");
  }
}

// ---- Gbrapf32 → u8 RGBA (source α) -----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf32_to_rgba_matches_scalar() {
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
    unsafe {
      gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba width={w}");
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf32_to_rgba_u16_matches_scalar() {
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
    unsafe {
      gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_u16 width={w}");
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless, source α) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf32_to_rgba_f32_matches_scalar() {
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
    unsafe {
      gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_f32 width={w}");
  }
}

// ---- Gbrapf32 → f16 RGBA (fp16-gated, source α) -----------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf32_to_rgba_f16_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrapf32_to_rgba_f16_row_fp16::<false>(&g, &b, &r, &a, &mut simd, w);
    }
    scalar::planar_gbr_float::gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf32_to_rgba_f16 width={w}");
  }
}

// ---- Gbrpf16 → u8 RGB (fp16-gated widening) ---------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgb_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrpf16_to_rgb_row_fp16::<false>(&g, &b, &r, &mut simd, w);
    }
    // Scalar reference: widen f16→f32, then scalar gbrpf32_to_rgb_row.
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgb_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb width={w}");
  }
}

// ---- Gbrpf16 → u8 RGBA (fp16-gated widening) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgba_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrpf16_to_rgba_row_fp16::<false>(&g, &b, &r, &mut simd, w);
    }
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgba_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba width={w}");
  }
}

// ---- Gbrpf16 → u16 RGB (fp16-gated widening) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgb_u16_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrpf16_to_rgb_u16_row_fp16::<false>(&g, &b, &r, &mut simd, w);
    }
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgb_u16_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb_u16 width={w}");
  }
}

// ---- Gbrpf16 → u16 RGBA (fp16-gated widening) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgba_u16_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrpf16_to_rgba_u16_row_fp16::<false>(&g, &b, &r, &mut simd, w);
    }
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgba_u16_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba_u16 width={w}");
  }
}

// ---- Gbrpf16 → f32 RGB (fp16-gated, lossless widen) -------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgb_f32_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrpf16_to_rgb_f32_row_fp16::<false>(&g, &b, &r, &mut simd, w);
    }
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgb_f32_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb_f32 width={w}");
  }
}

// ---- Gbrpf16 → f32 RGBA (fp16-gated, lossless widen, α = 1.0) --------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgba_f32_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrpf16_to_rgba_f32_row_fp16::<false>(&g, &b, &r, &mut simd, w);
    }
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrpf32_to_rgba_f32_row::<false>(&gf, &bf, &rf, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba_f32 width={w}");
  }
}

// ---- Gbrpf16 → f16 RGB (lossless, no fp16 feature needed) ------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgb_f16_lossless_matches_scalar() {
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE007_0001);
    prng_f16(&mut b, 0xE007_0002);
    prng_f16(&mut r, 0xE007_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 3];
    let mut scal = std::vec![half::f16::ZERO; w * 3];
    unsafe {
      gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_f16::gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgb_f16 width={w}");
  }
}

// ---- Gbrpf16 → f16 RGBA (lossless, no fp16 feature needed) -----------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgba_f16_lossless_matches_scalar() {
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xE008_0001);
    prng_f16(&mut b, 0xE008_0002);
    prng_f16(&mut r, 0xE008_0003);
    let mut simd = std::vec![half::f16::ZERO; w * 4];
    let mut scal = std::vec![half::f16::ZERO; w * 4];
    unsafe {
      gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut simd, w);
    }
    scalar::planar_gbr_f16::gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut scal, w);
    assert_eq!(simd, scal, "gbrpf16_to_rgba_f16 width={w}");
  }
}

// ---- Gbrpf16 → u8 luma (fp16-gated) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_luma_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_luma_row_fp16::<false>(&g, &b, &r, &mut simd, w, KernelMatrix::Bt709, true);
    }
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
    assert_eq!(simd, scal, "gbrpf16_to_luma width={w}");
  }
}

// ---- Gbrpf16 → u16 luma (fp16-gated) ----------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_luma_u16_matches_scalar() {
  use crate::KernelMatrix;
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_luma_u16_row_fp16::<false>(&g, &b, &r, &mut simd, w, KernelMatrix::Bt709, true);
    }
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
    assert_eq!(simd, scal, "gbrpf16_to_luma_u16 width={w}");
  }
}

// ---- Gbrpf16 → HSV (fp16-gated) --------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_hsv_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_hsv_row_fp16::<false>(&g, &b, &r, &mut simd_h, &mut simd_s, &mut simd_v, w);
    }
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
    assert_eq!(simd_h, scal_h, "gbrpf16 hsv H width={w}");
    assert_eq!(simd_s, scal_s, "gbrpf16 hsv S width={w}");
    assert_eq!(simd_v, scal_v, "gbrpf16 hsv V width={w}");
  }
}

// ---- Gbrapf16 → u8 RGBA (fp16-gated, source α) ------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf16_to_rgba_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrapf16_to_rgba_row_fp16::<false>(&g, &b, &r, &a, &mut simd, w);
    }
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let af: std::vec::Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrapf32_to_rgba_row::<false>(&gf, &bf, &rf, &af, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba width={w}");
  }
}

// ---- Gbrapf16 → u16 RGBA (fp16-gated, source α) -----------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf16_to_rgba_u16_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrapf16_to_rgba_u16_row_fp16::<false>(&g, &b, &r, &a, &mut simd, w);
    }
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let af: std::vec::Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrapf32_to_rgba_u16_row::<false>(&gf, &bf, &rf, &af, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba_u16 width={w}");
  }
}

// ---- Gbrapf16 → f32 RGBA (fp16-gated, lossless widen, source α) -------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf16_to_rgba_f32_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
    unsafe {
      gbrapf16_to_rgba_f32_row_fp16::<false>(&g, &b, &r, &a, &mut simd, w);
    }
    let gf: std::vec::Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bf: std::vec::Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let rf: std::vec::Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let af: std::vec::Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
    scalar::planar_gbr_float::gbrapf32_to_rgba_f32_row::<false>(&gf, &bf, &rf, &af, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba_f32 width={w}");
  }
}

// ---- Gbrapf16 → f16 RGBA (lossless, source α, no fp16 feature needed) -------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf16_to_rgba_f16_lossless_matches_scalar() {
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
    unsafe {
      gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut simd, w);
    }
    scalar::planar_gbr_f16::gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut scal, w);
    assert_eq!(simd, scal, "gbrapf16_to_rgba_f16 width={w}");
  }
}

// ---- BE parity helpers ------------------------------------------------------

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
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgb_be_parity() {
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
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgba_be_parity() {
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

// ---- BE parity: Gbrpf32 → f32 RGB (lossless) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf32_to_rgb_f32_be_parity() {
  for &w in WIDTHS {
    let mut g = std::vec![0.0f32; w];
    let mut b = std::vec![0.0f32; w];
    let mut r = std::vec![0.0f32; w];
    prng_f32(&mut g, 0xBE05_0001);
    prng_f32(&mut b, 0xBE05_0002);
    prng_f32(&mut r, 0xBE05_0003);
    let mut le_out = std::vec![0.0f32; w * 3];
    let mut be_out = std::vec![0.0f32; w * 3];
    unsafe {
      gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f32(&g);
    let b_be = be_encode_f32(&b);
    let r_be = be_encode_f32(&r);
    unsafe {
      gbrpf32_to_rgb_f32_row::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(le_out, be_out, "gbrpf32_to_rgb_f32 BE parity width={w}");
  }
}

// ---- BE parity: Gbrpf16 → f16 RGB (lossless) --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgb_f16_be_parity() {
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

// ---- BE parity: Gbrpf16 → f16 RGBA (lossless) -------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgba_f16_be_parity() {
  for &w in WIDTHS {
    let mut g = std::vec![half::f16::ZERO; w];
    let mut b = std::vec![half::f16::ZERO; w];
    let mut r = std::vec![half::f16::ZERO; w];
    prng_f16(&mut g, 0xBE08_0001);
    prng_f16(&mut b, 0xBE08_0002);
    prng_f16(&mut r, 0xBE08_0003);
    let mut le_out = std::vec![half::f16::ZERO; w * 4];
    let mut be_out = std::vec![half::f16::ZERO; w * 4];
    unsafe {
      gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(le_out, be_out, "gbrpf16_to_rgba_f16 BE parity width={w}");
  }
}

// ---- BE parity: Gbrapf16 → f16 RGBA (lossless) ------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf16_to_rgba_f16_be_parity() {
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

// BE parity: SIMD-tail Gbrpf16 → u8 RGB at 4 px lane × non-multiple widths.
//
// The SIMD scalar tail in `gbrpf16_to_rgb_row_fp16` widens f16 → f32 then
// routes the scalar f32 kernel. Without normalizing the f16 bits first, the
// BE-source-on-LE-host path double-byte-swaps (raw `to_f32` widens BE bits
// as if host-native, then `scalar::gbrpf32_to_rgb_row::<true>` byte-swaps
// the f32). At widths that are not a multiple of the SIMD lane count
// (4 for NEON), the tail is reached and the bug manifests. Widths
// 5 / 7 / 33 = 4·1 + 1, 4·1 + 3, 4·8 + 1 — each exercises a different tail
// length.
const SIMD_TAIL_WIDTHS: &[usize] = &[5, 7, 33];

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgb_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_rgb_row_fp16::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgb_row_fp16::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrpf16_to_rgb SIMD-tail BE parity width={w}"
    );
  }
}

// ---- BE parity: SIMD-tail Gbrpf16 → u8 RGBA --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgba_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_rgba_row_fp16::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgba_row_fp16::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrpf16_to_rgba SIMD-tail BE parity width={w}"
    );
  }
}

// ---- BE parity: SIMD-tail Gbrpf16 → u16 RGB --------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgb_u16_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_rgb_u16_row_fp16::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgb_u16_row_fp16::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrpf16_to_rgb_u16 SIMD-tail BE parity width={w}"
    );
  }
}

// ---- BE parity: SIMD-tail Gbrpf16 → u16 RGBA -------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgba_u16_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_rgba_u16_row_fp16::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgba_u16_row_fp16::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrpf16_to_rgba_u16 SIMD-tail BE parity width={w}"
    );
  }
}

// ---- BE parity: SIMD-tail Gbrapf16 → u8 RGBA -------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf16_to_rgba_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrapf16_to_rgba_row_fp16::<false>(&g, &b, &r, &a, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    unsafe {
      gbrapf16_to_rgba_row_fp16::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrapf16_to_rgba SIMD-tail BE parity width={w}"
    );
  }
}

// ---- BE parity: SIMD-tail Gbrapf16 → u16 RGBA ------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf16_to_rgba_u16_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrapf16_to_rgba_u16_row_fp16::<false>(&g, &b, &r, &a, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    unsafe {
      gbrapf16_to_rgba_u16_row_fp16::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "gbrapf16_to_rgba_u16 SIMD-tail BE parity width={w}"
    );
  }
}

// BE parity: SIMD-tail Gbrpf16 → f32 RGB.
//
// The f16 → f32 lossless tail paths share the double-byte-swap bug class
// of the integer-output tails. At widths that are not a multiple of 4
// (NEON SIMD lane count), a scalar tail that widens BE-encoded f16 bits
// as host-native via `to_f32` and routes scratch through
// `scalar::gbrpf32_to_*::<BE>` byte-swaps the (already-wrong) f32 again.
// Widths 5 / 7 / 33 each exercise a different tail length.

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgb_f32_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_rgb_f32_row_fp16::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgb_f32_row_fp16::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
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

// ---- BE parity: SIMD-tail Gbrpf16 → f32 RGBA -------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrpf16_to_rgba_f32_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrpf16_to_rgba_f32_row_fp16::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgba_f32_row_fp16::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
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

// ---- BE parity: SIMD-tail Gbrapf16 → f32 RGBA ------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbrapf16_to_rgba_f32_simd_tail_be_parity() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
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
      gbrapf16_to_rgba_f32_row_fp16::<false>(&g, &b, &r, &a, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    unsafe {
      gbrapf16_to_rgba_f32_row_fp16::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, w);
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
