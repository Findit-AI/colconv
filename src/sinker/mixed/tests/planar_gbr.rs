//! Tier 10 sinker integration tests — planar GBR (`Gbrp` / `Gbrap`).

use super::*;
use crate::sinker::MixedSinker;

// ---- shared helpers ----------------------------------------------------

/// Build three planar G/B/R planes from a packed-RGB seed buffer (the
/// inverse of `gbr_to_rgb_row`). Returns `(g, b, r)` plane buffers, all
/// of length `width * height`.
fn planes_from_packed_rgb(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let n = width * height;
  let mut g = std::vec![0u8; n];
  let mut b = std::vec![0u8; n];
  let mut r = std::vec![0u8; n];
  for i in 0..n {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  (g, b, r)
}

/// Random packed RGB seed (3 bytes per pixel).
fn random_rgb(width: usize, height: usize, seed: u32) -> Vec<u8> {
  let mut buf = std::vec![0u8; width * height * 3];
  pseudo_random_u8(&mut buf, seed);
  buf
}

/// Random alpha plane.
fn random_alpha(width: usize, height: usize, seed: u32) -> Vec<u8> {
  let mut buf = std::vec![0u8; width * height];
  pseudo_random_u8(&mut buf, seed);
  buf
}

// ---- Gbrp tests --------------------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_with_rgb_reorders_planes_to_packed_rgb() {
  let w = 16usize;
  let h = 4usize;
  // Solid colour: R=200, G=100, B=50.
  let mut g = std::vec![100u8; w * h];
  let mut b = std::vec![50u8; w * h];
  let mut r = std::vec![200u8; w * h];
  // Touch one pixel to make sure we're not just memcpying a constant.
  g[7] = 33;
  b[7] = 44;
  r[7] = 55;

  let src =
    GbrpFrame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Gbrp>::new(w, h)
    .with_rgb(&mut rgb_out)
    .unwrap();
  gbrp_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for i in 0..w * h {
    assert_eq!(rgb_out[i * 3], r[i], "R px {i}");
    assert_eq!(rgb_out[i * 3 + 1], g[i], "G px {i}");
    assert_eq!(rgb_out[i * 3 + 2], b[i], "B px {i}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_with_rgba_appends_opaque_alpha() {
  let w = 16usize;
  let h = 4usize;
  let g = std::vec![100u8; w * h];
  let b = std::vec![50u8; w * h];
  let r = std::vec![200u8; w * h];
  let src =
    GbrpFrame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();

  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Gbrp>::new(w, h).with_rgba(&mut rgba).unwrap();
  gbrp_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0xFF]);
  }
}

// Cross-references the `rgb`-only `Rgb24` source for a parity oracle, so
// it only compiles when `rgb` is also enabled (a `gbr`-solo build has no
// `Rgb24Frame` / `rgb24_to`).
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_planar_parity_with_rgb24() {
  // Convert the same pixel data through both the Rgb24 and Gbrp source
  // paths; outputs (RGB, RGBA, luma, HSV) must be byte-identical.
  let w = 16usize;
  let h = 8usize;
  let rgb_seed = random_rgb(w, h, 0x1234_5678);
  let (g, b, r) = planes_from_packed_rgb(&rgb_seed, w, h);

  // ---- Rgb24 reference outputs ----
  let rgb24 = Rgb24Frame::try_new(&rgb_seed, w as u32, h as u32, (w * 3) as u32).unwrap();
  let mut rgb_ref = std::vec![0u8; w * h * 3];
  let mut rgba_ref = std::vec![0u8; w * h * 4];
  let mut luma_ref = std::vec![0u8; w * h];
  let mut h_ref = std::vec![0u8; w * h];
  let mut s_ref = std::vec![0u8; w * h];
  let mut v_ref = std::vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Rgb24>::new(w, h)
      .with_rgb(&mut rgb_ref)
      .unwrap()
      .with_rgba(&mut rgba_ref)
      .unwrap()
      .with_luma(&mut luma_ref)
      .unwrap()
      .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
      .unwrap();
    rgb24_to(&rgb24, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // ---- Gbrp outputs from the same pixel data ----
  let gbrp =
    GbrpFrame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();
  let mut rgb_g = std::vec![0u8; w * h * 3];
  let mut rgba_g = std::vec![0u8; w * h * 4];
  let mut luma_g = std::vec![0u8; w * h];
  let mut h_g = std::vec![0u8; w * h];
  let mut s_g = std::vec![0u8; w * h];
  let mut v_g = std::vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Gbrp>::new(w, h)
      .with_rgb(&mut rgb_g)
      .unwrap()
      .with_rgba(&mut rgba_g)
      .unwrap()
      .with_luma(&mut luma_g)
      .unwrap()
      .with_hsv(&mut h_g, &mut s_g, &mut v_g)
      .unwrap();
    gbrp_to(&gbrp, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(rgb_g, rgb_ref, "RGB mismatch Gbrp vs Rgb24");
  assert_eq!(rgba_g, rgba_ref, "RGBA mismatch Gbrp vs Rgb24");
  assert_eq!(luma_g, luma_ref, "luma mismatch Gbrp vs Rgb24");
  assert_eq!(h_g, h_ref, "H mismatch Gbrp vs Rgb24");
  assert_eq!(s_g, s_ref, "S mismatch Gbrp vs Rgb24");
  assert_eq!(v_g, v_ref, "V mismatch Gbrp vs Rgb24");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_with_luma_u16_zero_extends_u8_luma() {
  let w = 16usize;
  let h = 4usize;
  let rgb_seed = random_rgb(w, h, 0xABCD_EF01);
  let (g, b, r) = planes_from_packed_rgb(&rgb_seed, w, h);
  let src =
    GbrpFrame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();

  let mut luma_u8 = std::vec![0u8; w * h];
  let mut luma_u16 = std::vec![0u16; w * h];
  let mut sink = MixedSinker::<Gbrp>::new(w, h)
    .with_luma(&mut luma_u8)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  gbrp_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for i in 0..w * h {
    assert_eq!(luma_u16[i], luma_u8[i] as u16, "px {i}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_with_luma_u16_wide_row_no_alloc_regression() {
  // Regression for the prior `STACK_CAP = 8192` heap-fallback path:
  // verify a wider row produces the same byte-zero-extended luma as a
  // narrower row would, exercising the direct `rgb_to_luma_u16_row`
  // path that replaced the per-row `Vec<u8>` allocation.
  let w = 9000usize;
  let h = 1usize;
  let rgb_seed = random_rgb(w, h, 0x1234_5678);
  let (g, b, r) = planes_from_packed_rgb(&rgb_seed, w, h);
  let src =
    GbrpFrame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();

  let mut luma_u8 = std::vec![0u8; w * h];
  let mut luma_u16 = std::vec![0u16; w * h];
  let mut sink = MixedSinker::<Gbrp>::new(w, h)
    .with_luma(&mut luma_u8)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  gbrp_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  // u16 luma == u8 luma zero-extended (same byte values, native u16).
  for i in 0..w * h {
    assert_eq!(luma_u16[i], luma_u8[i] as u16, "px {i}");
  }
}

// ---- Gbrap tests -------------------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap_with_rgba_passes_source_alpha() {
  let w = 16usize;
  let h = 4usize;
  let g = std::vec![100u8; w * h];
  let b = std::vec![50u8; w * h];
  let r = std::vec![200u8; w * h];
  // Random alpha — make sure each pixel keeps its own α.
  let a = random_alpha(w, h, 0xDEAD_BEEF);
  let src = GbrapFrame::try_new(
    &g, &b, &r, &a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Gbrap>::new(w, h)
    .with_rgba(&mut rgba)
    .unwrap();
  gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for i in 0..w * h {
    assert_eq!(rgba[i * 4], r[i], "R px {i}");
    assert_eq!(rgba[i * 4 + 1], g[i], "G px {i}");
    assert_eq!(rgba[i * 4 + 2], b[i], "B px {i}");
    assert_eq!(rgba[i * 4 + 3], a[i], "A px {i}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap_with_rgb_drops_alpha() {
  let w = 16usize;
  let h = 4usize;
  let rgb_seed = random_rgb(w, h, 0xBEEF_F00D);
  let (g, b, r) = planes_from_packed_rgb(&rgb_seed, w, h);
  let a = random_alpha(w, h, 0x1357_9BDF);
  let src = GbrapFrame::try_new(
    &g, &b, &r, &a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgb_out = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Gbrap>::new(w, h)
    .with_rgb(&mut rgb_out)
    .unwrap();
  gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  // Output must equal the rgb_seed reconstructed from the planes.
  let (rg, rb, rr) = planes_from_packed_rgb(&rgb_out, w, h);
  assert_eq!(rg, g, "G");
  assert_eq!(rb, b, "B");
  assert_eq!(rr, r, "R");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap_with_rgb_and_with_rgba_strategy_a_plus_matches_independent() {
  // Strategy A+ correctness: the combo path (with_rgb + with_rgba)
  // should produce byte-identical output to running each request
  // through its own dedicated sinker.
  let w = 32usize;
  let h = 8usize;
  let rgb_seed = random_rgb(w, h, 0xABBA_BABE);
  let (g, b, r) = planes_from_packed_rgb(&rgb_seed, w, h);
  let a = random_alpha(w, h, 0xCAFE_BABE);
  let src = GbrapFrame::try_new(
    &g, &b, &r, &a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // ---- Reference: two independent sinkers ----
  let mut rgb_ref = std::vec![0u8; w * h * 3];
  let mut rgba_ref = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrap>::new(w, h)
      .with_rgb(&mut rgb_ref)
      .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Gbrap>::new(w, h)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // ---- Strategy A+ combo: one sinker writes both ----
  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrap>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba(&mut rgba_combo)
      .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(rgb_combo, rgb_ref, "RGB mismatch combo vs independent");
  assert_eq!(rgba_combo, rgba_ref, "RGBA mismatch combo vs independent");
}

// Cross-references the `rgb`-only `Rgb24` source for a parity oracle —
// `rgb`-gated for the same reason as `gbrp_planar_parity_with_rgb24`.
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap_planar_parity_with_rgb24_when_alpha_drop() {
  // Same pixel data fed through Rgb24 and Gbrap (with α discarded
  // via with_rgb): outputs must be byte-identical.
  let w = 16usize;
  let h = 8usize;
  let rgb_seed = random_rgb(w, h, 0xC001_F00D);
  let (g, b, r) = planes_from_packed_rgb(&rgb_seed, w, h);
  let a = random_alpha(w, h, 0xDEAD_FA11);

  let rgb24 = Rgb24Frame::try_new(&rgb_seed, w as u32, h as u32, (w * 3) as u32).unwrap();
  let gbrap = GbrapFrame::try_new(
    &g, &b, &r, &a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Rgb24 path.
  let mut rgb_ref = std::vec![0u8; w * h * 3];
  let mut luma_ref = std::vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Rgb24>::new(w, h)
      .with_rgb(&mut rgb_ref)
      .unwrap()
      .with_luma(&mut luma_ref)
      .unwrap();
    rgb24_to(&rgb24, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // Gbrap with α-drop path.
  let mut rgb_g = std::vec![0u8; w * h * 3];
  let mut luma_g = std::vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Gbrap>::new(w, h)
      .with_rgb(&mut rgb_g)
      .unwrap()
      .with_luma(&mut luma_g)
      .unwrap();
    gbrap_to(&gbrap, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(rgb_g, rgb_ref, "RGB mismatch Gbrap vs Rgb24");
  assert_eq!(luma_g, luma_ref, "luma mismatch Gbrap vs Rgb24");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_simd_matches_scalar() {
  // Differential test: SIMD vs scalar fan-out across {RGB, RGBA, luma,
  // luma_u16, HSV} on random GBR data.
  let w = 64usize;
  let h = 8usize;
  let rgb_seed = random_rgb(w, h, 0xFEED_FACE);
  let (g, b, r) = planes_from_packed_rgb(&rgb_seed, w, h);
  let src =
    GbrpFrame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();

  let make_buffers = || {
    (
      std::vec![0u8; w * h * 3], // rgb
      std::vec![0u8; w * h * 4], // rgba
      std::vec![0u8; w * h],     // luma
      std::vec![0u16; w * h],    // luma_u16
      std::vec![0u8; w * h],     // h
      std::vec![0u8; w * h],     // s
      std::vec![0u8; w * h],     // v
    )
  };

  let (mut r1, mut a1, mut l1, mut lu1, mut h1, mut s1, mut v1) = make_buffers();
  let (mut r2, mut a2, mut l2, mut lu2, mut h2, mut s2, mut v2) = make_buffers();

  {
    let mut sink = MixedSinker::<Gbrp>::new(w, h)
      .with_rgb(&mut r1)
      .unwrap()
      .with_rgba(&mut a1)
      .unwrap()
      .with_luma(&mut l1)
      .unwrap()
      .with_luma_u16(&mut lu1)
      .unwrap()
      .with_hsv(&mut h1, &mut s1, &mut v1)
      .unwrap()
      .with_simd(true);
    gbrp_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Gbrp>::new(w, h)
      .with_rgb(&mut r2)
      .unwrap()
      .with_rgba(&mut a2)
      .unwrap()
      .with_luma(&mut l2)
      .unwrap()
      .with_luma_u16(&mut lu2)
      .unwrap()
      .with_hsv(&mut h2, &mut s2, &mut v2)
      .unwrap()
      .with_simd(false);
    gbrp_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(r1, r2, "rgb SIMD vs scalar");
  assert_eq!(a1, a2, "rgba SIMD vs scalar");
  assert_eq!(l1, l2, "luma SIMD vs scalar");
  assert_eq!(lu1, lu2, "luma_u16 SIMD vs scalar");
  // HSV SIMD vs scalar can drift by ±1 LSB (OpenCV-style; H is
  // circular). See the existing per-arch HSV tests for the tolerance
  // rationale.
  assert_hsv_within_one_lsb(&h1, &s1, &v1, &h2, &s2, &v2);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap_simd_matches_scalar() {
  let w = 64usize;
  let h = 8usize;
  let rgb_seed = random_rgb(w, h, 0xFACE_F00D);
  let (g, b, r) = planes_from_packed_rgb(&rgb_seed, w, h);
  let a = random_alpha(w, h, 0xC0DE_BABE);
  let src = GbrapFrame::try_new(
    &g, &b, &r, &a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let make_buffers = || {
    (
      std::vec![0u8; w * h * 3],
      std::vec![0u8; w * h * 4],
      std::vec![0u8; w * h],
      std::vec![0u16; w * h],
      std::vec![0u8; w * h],
      std::vec![0u8; w * h],
      std::vec![0u8; w * h],
    )
  };

  let (mut r1, mut a1, mut l1, mut lu1, mut h1, mut s1, mut v1) = make_buffers();
  let (mut r2, mut a2, mut l2, mut lu2, mut h2, mut s2, mut v2) = make_buffers();

  {
    let mut sink = MixedSinker::<Gbrap>::new(w, h)
      .with_rgb(&mut r1)
      .unwrap()
      .with_rgba(&mut a1)
      .unwrap()
      .with_luma(&mut l1)
      .unwrap()
      .with_luma_u16(&mut lu1)
      .unwrap()
      .with_hsv(&mut h1, &mut s1, &mut v1)
      .unwrap()
      .with_simd(true);
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Gbrap>::new(w, h)
      .with_rgb(&mut r2)
      .unwrap()
      .with_rgba(&mut a2)
      .unwrap()
      .with_luma(&mut l2)
      .unwrap()
      .with_luma_u16(&mut lu2)
      .unwrap()
      .with_hsv(&mut h2, &mut s2, &mut v2)
      .unwrap()
      .with_simd(false);
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(r1, r2, "rgb SIMD vs scalar");
  assert_eq!(a1, a2, "rgba SIMD vs scalar");
  assert_eq!(l1, l2, "luma SIMD vs scalar");
  assert_eq!(lu1, lu2, "luma_u16 SIMD vs scalar");
  assert_hsv_within_one_lsb(&h1, &s1, &v1, &h2, &s2, &v2);
}

/// HSV SIMD vs scalar can disagree by ±1 LSB at boundary pixels (the
/// scalar path uses an integer LUT, SIMD uses true f32 division). H is
/// circular: distance between 0 and 179 is 1, not 179.
fn assert_hsv_within_one_lsb(h1: &[u8], s1: &[u8], v1: &[u8], h2: &[u8], s2: &[u8], v2: &[u8]) {
  for (i, (&a, &b)) in h1.iter().zip(h2.iter()).enumerate() {
    let d = a.abs_diff(b);
    let circ = d.min(180 - d);
    assert!(circ <= 1, "H divergence at pixel {i}: simd={a} scalar={b}");
  }
  for (i, (&a, &b)) in s1.iter().zip(s2.iter()).enumerate() {
    assert!(
      a.abs_diff(b) <= 1,
      "S divergence at pixel {i}: simd={a} scalar={b}"
    );
  }
  for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
    assert!(
      a.abs_diff(b) <= 1,
      "V divergence at pixel {i}: simd={a} scalar={b}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_with_luma_bt709_pure_red() {
  // Pure red full-range BT.709: Y = 0.2126 * 255 ≈ 54.21 → 54.
  let w = 16usize;
  let h = 4usize;
  let g = std::vec![0u8; w * h];
  let b = std::vec![0u8; w * h];
  let r = std::vec![255u8; w * h];
  let src =
    GbrpFrame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();

  let mut luma = std::vec![0u8; w * h];
  let mut sink = MixedSinker::<Gbrp>::new(w, h).with_luma(&mut luma).unwrap();
  gbrp_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for &y in &luma {
    assert!(y.abs_diff(54) <= 1, "got Y={y}");
  }
}
