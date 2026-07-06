//! Task 9 — cross-format / HDR / α / round-half-up integration tests for
//! the Tier 10 float planar GBR source family.
//!
//! Covers:
//! - Cross-format planar parity (Gbrpf32 vs Rgbf32; Gbrpf16 vs Rgbf16).
//! - Gbrapf32 Strategy A+ byte-equivalence against independent-kernel path.
//! - HDR pass-through (NaN, Inf, values > 1.0 preserved on lossless paths).
//! - f32 α plane reaches RGBA output slot 3 untouched (memcpy semantics).
//! - f16-narrowing saturation: values > 65504 → +Inf; values < -65504 → -Inf.
//! - Round-half-up regression: {0.5/255, 2.5/255, 4.5/255} → {1, 3, 5} on
//!   every backend (scalar + NEON + SSE4.1 + AVX2 + AVX-512).
//! - Gbrapf32 Strategy A+ for rgba_u8, rgba_u16, rgba_f32, rgba_f16 outputs.
//! - 32-bit overflow guard (only on 32-bit targets).

use super::*;
use crate::sinker::MixedSinker;

// ---- Helpers ---------------------------------------------------------------

/// Build a Gbrpf32 frame with constant colour `(r, g, b)` across
/// all `width x height` pixels. Returns `(g_plane, b_plane, r_plane)`.
fn solid_gbrpf32_planes(
  width: usize,
  height: usize,
  r: f32,
  g: f32,
  b: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let n = width * height;
  (std::vec![g; n], std::vec![b; n], std::vec![r; n])
}

/// Build a Gbrpf32 frame from random-ish per-pixel f32 values.
/// Values cycle through `vals` (repeated across G, B, R planes).
fn patterned_gbrpf32_planes(
  width: usize,
  height: usize,
  vals: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let n = width * height;
  let g: Vec<f32> = (0..n).map(|i| vals[i % vals.len()]).collect();
  let b: Vec<f32> = (0..n).map(|i| vals[(i + 1) % vals.len()]).collect();
  let r: Vec<f32> = (0..n).map(|i| vals[(i + 2) % vals.len()]).collect();
  (g, b, r)
}

/// Build an α plane from `vals` (cycled), then build a Gbrapf32 frame.
fn patterned_alpha_f32(width: usize, height: usize, seed: u8) -> Vec<f32> {
  let n = width * height;
  let mut buf = std::vec![0u8; n];
  pseudo_random_u8(&mut buf, seed as u32);
  buf.iter().map(|&b| b as f32 / 255.0).collect()
}

/// Build packed Rgbf32 data from the same (r, g, b) values as
/// `patterned_gbrpf32_planes`, so the two frames contain identical RGB data.
/// Consumed only by the `rgb-float`-gated `Rgbf32` parity test.
#[cfg(feature = "rgb-float")]
fn patterned_rgbf32_packed(width: usize, height: usize, vals: &[f32]) -> Vec<f32> {
  let n = width * height;
  let mut buf = std::vec![0.0f32; n * 3];
  for i in 0..n {
    buf[i * 3] = vals[(i + 2) % vals.len()]; // R
    buf[i * 3 + 1] = vals[i % vals.len()]; // G
    buf[i * 3 + 2] = vals[(i + 1) % vals.len()]; // B
  }
  buf
}

// ---- Cross-format planar parity (6 tests) ----------------------------------

/// Gbrpf32 and Rgbf32 with the same per-pixel RGB values produce byte-identical
/// u8 RGB output.
// Cross-references the `rgb-float`-only `Rgbf32` source for a parity
// oracle, so it only compiles when `rgb-float` is also enabled (a
// `gbr`-solo build has no `Rgbf32Frame` / `rgbf32_to`).
#[cfg(feature = "rgb-float")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgb_matches_rgbf32_rgb() {
  let w = 32usize;
  let h = 8usize;
  // Use a set that spans in-range, HDR-clamped, and negative-clamped.
  let vals = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5];
  let (gp, bp, rp) = patterned_gbrpf32_planes(w, h, &vals);
  let packed = patterned_rgbf32_packed(w, h, &vals);

  let gbrp_src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();
  let rgbf32_src = Rgbf32Frame::try_new(&packed, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_gbrp = std::vec![0u8; w * h * 3];
  let mut rgb_rgbf32 = std::vec![0u8; w * h * 3];

  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb(&mut rgb_gbrp)
      .unwrap();
    gbrpf32_to(&gbrp_src, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Rgbf32>::new(w, h)
      .with_rgb(&mut rgb_rgbf32)
      .unwrap();
    rgbf32_to(&rgbf32_src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(
    rgb_gbrp, rgb_rgbf32,
    "Gbrpf32 vs Rgbf32 RGB u8 output must be byte-identical"
  );
}

/// Gbrpf16 and Rgbf16 with the same per-pixel values produce byte-identical
/// u8 RGB output.
// Cross-references the `rgb-float`-only `Rgbf16` source for a parity
// oracle — `rgb-float`-gated for the same reason as
// `gbrpf32_rgb_matches_rgbf32_rgb`.
#[cfg(feature = "rgb-float")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_rgb_matches_rgbf16_rgb() {
  let w = 32usize;
  let h = 8usize;
  let vals_f32 = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5];
  let vals_f16: Vec<half::f16> = vals_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
  let n = w * h;

  // G, B, R planes (planar; same cycling as patterned_gbrpf32_planes)
  let gp: Vec<half::f16> = (0..n).map(|i| vals_f16[i % vals_f16.len()]).collect();
  let bp: Vec<half::f16> = (0..n).map(|i| vals_f16[(i + 1) % vals_f16.len()]).collect();
  let rp: Vec<half::f16> = (0..n).map(|i| vals_f16[(i + 2) % vals_f16.len()]).collect();

  // Packed Rgbf16: R=vals[(i+2)%len], G=vals[i%len], B=vals[(i+1)%len]
  let packed: Vec<half::f16> = (0..n * 3)
    .map(|j| {
      let i = j / 3;
      let ch = j % 3;
      match ch {
        0 => vals_f16[(i + 2) % vals_f16.len()], // R
        1 => vals_f16[i % vals_f16.len()],       // G
        _ => vals_f16[(i + 1) % vals_f16.len()], // B
      }
    })
    .collect();

  let gbrp_src = Gbrpf16Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();
  let rgbf16_src = Rgbf16Frame::try_new(&packed, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_gbrp = std::vec![0u8; w * h * 3];
  let mut rgb_rgbf16 = std::vec![0u8; w * h * 3];

  {
    let mut sink = MixedSinker::<Gbrpf16>::new(w, h)
      .with_rgb(&mut rgb_gbrp)
      .unwrap();
    gbrpf16_to(&gbrp_src, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Rgbf16>::new(w, h)
      .with_rgb(&mut rgb_rgbf16)
      .unwrap();
    rgbf16_to(&rgbf16_src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(
    rgb_gbrp, rgb_rgbf16,
    "Gbrpf16 vs Rgbf16 RGB u8 output must be byte-identical"
  );
}

/// Gbrapf32 Strategy A+ (with_rgb + with_rgba) produces byte-identical output
/// to two independent-kernel runs.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let vals = [0.1f32, 0.3, 0.5, 0.7, 0.9, 1.1, -0.1];
  let (gp, bp, rp) = patterned_gbrpf32_planes(w, h, &vals);
  let ap = patterned_alpha_f32(w, h, 0xAB);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Independent: two separate sinks.
  let mut rgb_ref = std::vec![0u8; w * h * 3];
  let mut rgba_ref = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Strategy A+ combo.
  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgb_combo, rgb_ref,
    "Strategy A+ RGB must match independent RGB"
  );
  assert_eq!(
    rgba_combo, rgba_ref,
    "Strategy A+ RGBA must match independent RGBA"
  );
}

/// Gbrapf32 with_rgba_f32: α plane passes through lossless (memcpy semantics
/// for the α slot; no scaling, no clamping on the f32-output path).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_f32_lossless_alpha() {
  let w = 16usize;
  let h = 4usize;
  // Use a mix including out-of-[0,1] α values to confirm lossless pass-through.
  let ap: Vec<f32> = (0..w * h)
    .map(|i| match i % 4 {
      0 => 0.0,
      1 => 0.5,
      2 => 1.0,
      _ => 1.5, // HDR α
    })
    .collect();
  let g = std::vec![0.5f32; w * h];
  let b = std::vec![0.5f32; w * h];
  let r = std::vec![0.5f32; w * h];
  let src = Gbrapf32Frame::try_new(
    &g, &b, &r, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_out = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_f32(&mut rgba_out)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Slot 3 of each pixel must be the source α value, bit-exact.
  for i in 0..w * h {
    let got = rgba_out[i * 4 + 3];
    let want = ap[i];
    assert_eq!(
      got.to_bits(),
      want.to_bits(),
      "α at pixel {i}: got {got} want {want}"
    );
  }
}

/// Gbrpf32 with_rgb_f16 is byte-equivalent to with_rgb_f32 followed by
/// per-element `half::f16::from_f32`. Locks in the fused narrow correctness.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgb_f16_matches_caller_side_narrow() {
  let w = 64usize;
  let h = 4usize;
  let vals = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, -0.5];
  let (gp, bp, rp) = patterned_gbrpf32_planes(w, h, &vals);
  let src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Fused path.
  let mut rgb_f16_fused = std::vec![half::f16::ZERO; w * h * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb_f16(&mut rgb_f16_fused)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // Caller-side: f32 first, then narrow.
  let mut rgb_f32 = std::vec![0.0f32; w * h * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  let rgb_f16_caller: Vec<half::f16> = rgb_f32.iter().map(|&v| half::f16::from_f32(v)).collect();

  assert_eq!(
    rgb_f16_fused, rgb_f16_caller,
    "fused with_rgb_f16 must equal caller-side narrow of with_rgb_f32"
  );
}

/// Gbrpf32 with_rgb_f16 saturates HDR values > 65504 to f16::INFINITY /
/// f16::NEG_INFINITY (regression guard for the narrowing path).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgb_f16_saturates_hdr_to_f16_inf() {
  let w = 4usize;
  let h = 1usize;
  // 70000 > 65504 (f16 max representable finite) — must saturate to +Inf.
  // -70000 < -65504 — must saturate to -Inf.
  let r = std::vec![70000.0f32, -70000.0, 0.5, 1.0];
  let g = std::vec![70000.0f32, -70000.0, 0.5, 1.0];
  let b = std::vec![70000.0f32, -70000.0, 0.5, 1.0];
  let src =
    Gbrpf32Frame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();

  let mut out = std::vec![half::f16::ZERO; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb_f16(&mut out)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // Pixel 0: R/G/B all +Inf.
  assert_eq!(out[0], half::f16::INFINITY, "px0 R must be +Inf");
  assert_eq!(out[1], half::f16::INFINITY, "px0 G must be +Inf");
  assert_eq!(out[2], half::f16::INFINITY, "px0 B must be +Inf");
  // Pixel 1: R/G/B all -Inf.
  assert_eq!(out[3], half::f16::NEG_INFINITY, "px1 R must be -Inf");
  assert_eq!(out[4], half::f16::NEG_INFINITY, "px1 G must be -Inf");
  assert_eq!(out[5], half::f16::NEG_INFINITY, "px1 B must be -Inf");
}

// ---- HDR pass-through (4 tests) --------------------------------------------

/// Gbrpf32 with_rgb_f32 preserves HDR > 1.0, NaN, and Inf bit-exact.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgb_f32_preserves_hdr_nan_inf() {
  let w = 8usize;
  let h = 1usize;
  let special: [f32; 8] = [
    1.5,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    100.0,
    -0.5,
    0.0,
    1.0,
  ];
  let gp: Vec<f32> = special.to_vec();
  let bp: Vec<f32> = special.to_vec();
  let rp: Vec<f32> = special.to_vec();
  let src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut out = std::vec![0.0f32; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb_f32(&mut out)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // Output R channel per pixel = rp[i] bit-exact (including NaN/Inf).
  for i in 0..w {
    let r_out = out[i * 3];
    let r_src = rp[i];
    // NaN != NaN, so compare bit patterns.
    assert_eq!(
      r_out.to_bits(),
      r_src.to_bits(),
      "R at pixel {i}: expected {r_src} (bits {:08x}) got {r_out} (bits {:08x})",
      r_src.to_bits(),
      r_out.to_bits()
    );
  }
}

/// Gbrapf32 with_rgba_f32 preserves α plane bit-exact through Strategy A+.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_f32_preserves_hdr_alpha() {
  let w = 8usize;
  let h = 2usize;
  let ap: Vec<f32> = (0..w * h)
    .map(|i| match i % 4 {
      0 => 0.0,
      1 => f32::INFINITY,
      2 => 1.5,
      _ => 0.75,
    })
    .collect();
  let g = std::vec![0.5f32; w * h];
  let b = std::vec![0.5f32; w * h];
  let r = std::vec![0.5f32; w * h];
  let src = Gbrapf32Frame::try_new(
    &g, &b, &r, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_out = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_f32(&mut rgba_out)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  for i in 0..w * h {
    let got = rgba_out[i * 4 + 3];
    let want = ap[i];
    assert_eq!(
      got.to_bits(),
      want.to_bits(),
      "α at pixel {i}: got {got} (bits {:08x}) want {want} (bits {:08x})",
      got.to_bits(),
      want.to_bits()
    );
  }
}

/// Gbrpf16 with_rgb_f16 is lossless — no conversion, bit-exact interleave.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_rgb_f16_lossless_passthrough() {
  let w = 16usize;
  let h = 4usize;
  let vals_f32 = [0.0f32, 0.25, 0.5, 1.0, 1.5, 100.0, -0.5, f32::INFINITY];
  let vals_f16: Vec<half::f16> = vals_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
  let n = w * h;
  let gp: Vec<half::f16> = (0..n).map(|i| vals_f16[i % vals_f16.len()]).collect();
  let bp: Vec<half::f16> = (0..n).map(|i| vals_f16[(i + 1) % vals_f16.len()]).collect();
  let rp: Vec<half::f16> = (0..n).map(|i| vals_f16[(i + 2) % vals_f16.len()]).collect();
  let src = Gbrpf16Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut out = std::vec![half::f16::ZERO; w * h * 3];
  {
    let mut sink = MixedSinker::<Gbrpf16>::new(w, h)
      .with_rgb_f16(&mut out)
      .unwrap();
    gbrpf16_to(&src, &mut sink).unwrap();
  }

  // Packed output order is R, G, B.
  for i in 0..n {
    assert_eq!(out[i * 3].to_bits(), rp[i].to_bits(), "R at pixel {i}");
    assert_eq!(out[i * 3 + 1].to_bits(), gp[i].to_bits(), "G at pixel {i}");
    assert_eq!(out[i * 3 + 2].to_bits(), bp[i].to_bits(), "B at pixel {i}");
  }
}

/// Gbrpf32 (no α) with_rgba_f32 fills slot 3 with α = 1.0 exactly.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgba_f32_alpha_max_filled_correctly() {
  let w = 16usize;
  let h = 4usize;
  let (gp, bp, rp) = solid_gbrpf32_planes(w, h, 0.5, 0.25, 0.75);
  let src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_out = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgba_f32(&mut rgba_out)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  for i in 0..w * h {
    let alpha = rgba_out[i * 4 + 3];
    assert_eq!(
      alpha.to_bits(),
      1.0f32.to_bits(),
      "α slot at pixel {i}: expected 1.0 got {alpha}"
    );
  }
}

// ---- Round-half-up regression (5 tests, one per backend) -------------------
//
// Feed inputs where banker's rounding would diverge from round-half-up:
//   0.5/255 → 1 (not 0)   [banker's rounds 0.5 to even = 0]
//   2.5/255 → 3 (not 2)   [banker's rounds 2.5 to even = 2 on some impls]
//   4.5/255 → 5 (not 4)
// All backends must produce {1, 3, 5}.

fn round_half_up_check(rgb_out: &[u8], prefix: &str) {
  // The three input pixels are replicated across R, G, B identically.
  // rgb_out[0..3] = pixel 0 (R,G,B) from 0.5/255; rgb_out[3..6] = pixel 1 (2.5/255); etc.
  assert_eq!(
    rgb_out[0], 1,
    "{prefix}: R[0] from 0.5/255 must be 1 (round-half-up)"
  );
  assert_eq!(rgb_out[1], 1, "{prefix}: G[0] from 0.5/255 must be 1");
  assert_eq!(rgb_out[2], 1, "{prefix}: B[0] from 0.5/255 must be 1");
  assert_eq!(rgb_out[3], 3, "{prefix}: R[1] from 2.5/255 must be 3");
  assert_eq!(rgb_out[4], 3, "{prefix}: G[1] from 2.5/255 must be 3");
  assert_eq!(rgb_out[5], 3, "{prefix}: B[1] from 2.5/255 must be 3");
  assert_eq!(rgb_out[6], 5, "{prefix}: R[2] from 4.5/255 must be 5");
  assert_eq!(rgb_out[7], 5, "{prefix}: G[2] from 4.5/255 must be 5");
  assert_eq!(rgb_out[8], 5, "{prefix}: B[2] from 4.5/255 must be 5");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_scalar() {
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(false); // scalar path
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "scalar");
}

#[test]
#[cfg(target_arch = "aarch64")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_neon() {
  if !crate::row::neon_available() {
    return;
  }
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(true);
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "neon");
}

#[test]
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_sse41() {
  if !is_x86_feature_detected!("sse4.1") {
    return;
  }
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(true);
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "sse41");
}

#[test]
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_avx2() {
  if !is_x86_feature_detected!("avx2") {
    return;
  }
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(true);
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "avx2");
}

#[test]
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_avx512() {
  if !is_x86_feature_detected!("avx512f") {
    return;
  }
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(true);
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "avx512");
}

// ---- Strategy A+ byte-equivalence tests (5 tests) --------------------------
//
// For Gbrapf32: rgba_u8, rgba_u16, rgba_f32, rgba_f16 — verify Strategy A+
// (when both with_rgb AND the rgba accessor are attached) produces byte-identical
// output to the independent-kernel path (only the rgba accessor attached).

fn make_gbrapf32_src(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
  let vals = [0.1f32, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8];
  let (gp, bp, rp) = patterned_gbrpf32_planes(w, h, &vals);
  let ap = patterned_alpha_f32(w, h, 0xC3);
  (gp, bp, rp, ap)
}

/// Strategy A+ for rgba (u8): combo path = independent path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_u8_strategy_a_plus_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let (gp, bp, rp, ap) = make_gbrapf32_src(w, h);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_ref = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combo, rgba_ref,
    "Strategy A+ rgba_u8 must match independent kernel"
  );
}

/// Strategy A+ for rgba_u16: combo path = independent path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_u16_strategy_a_plus_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let (gp, bp, rp, ap) = make_gbrapf32_src(w, h);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_ref = std::vec![0u16; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_u16(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u16; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba_u16(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combo, rgba_ref,
    "Strategy A+ rgba_u16 must match independent kernel"
  );
}

/// Strategy A+ for rgba_f32: combo path = independent path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_f32_strategy_a_plus_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let (gp, bp, rp, ap) = make_gbrapf32_src(w, h);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_ref = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_f32(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba_f32(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // f32: compare bit-patterns to catch NaN equality issues.
  let bits_ref: Vec<u32> = rgba_ref.iter().map(|v| v.to_bits()).collect();
  let bits_combo: Vec<u32> = rgba_combo.iter().map(|v| v.to_bits()).collect();
  assert_eq!(
    bits_combo, bits_ref,
    "Strategy A+ rgba_f32 must match independent kernel"
  );
}

/// Strategy A+ for rgba_f16: combo path = independent path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_f16_strategy_a_plus_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let (gp, bp, rp, ap) = make_gbrapf32_src(w, h);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_ref = std::vec![half::f16::ZERO; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_f16(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![half::f16::ZERO; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba_f16(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let bits_ref: Vec<u16> = rgba_ref.iter().map(|v| v.to_bits()).collect();
  let bits_combo: Vec<u16> = rgba_combo.iter().map(|v| v.to_bits()).collect();
  assert_eq!(
    bits_combo, bits_ref,
    "Strategy A+ rgba_f16 must match independent kernel"
  );
}

// LE-encoded byte contract regressions.
//
// Each of the four float planar GBR Frame types is documented as
// LE-encoded bytes reinterpreted as `f32` / `half::f16` (FFmpeg `*LE`
// pixel-format convention). The sinker row-kernel dispatch must apply
// `u32::from_le` / `u16::from_le` (kernel `BE = false`) to recover host-
// native arithmetic from those bytes. These tests build a plane explicitly
// from LE-encoded bit patterns (`f32::from_bits(intended.to_bits().to_le())`
// and the f16 analogue) and assert the lossless pass-through output equals
// the host-native intended values.
//
// Vacuous on LE host (where `to_le` is identity so the LE-encoded plane is
// host-native already), but on a BE host any regression that drops the
// `::<false>` routing would be caught here — a kernel without `from_le`
// would emit byte-swapped bit-patterns, failing the bit-exact assertion
// below.

/// LE-encoded byte contract regression for [`Gbrpf32`].
///
/// Forces `with_simd(false)` so the test runs purely scalar — no SIMD
/// intrinsics — which lets it execute under `cargo miri test`. BE CI is
/// driven by miri on s390x / powerpc64; gating this test out of miri
/// would skip exactly the host where BE corruption would surface.
#[test]
fn gbrpf32_sinker_le_encoded_frame_decodes_correctly() {
  let w = 16usize;
  let h = 4usize;
  // Mix HDR, in-range, and negative values — the f32 lossless path must
  // round-trip them bit-exact on every host.
  let intended_g: Vec<f32> = (0..w * h)
    .map(|i| match i % 4 {
      0 => 0.5,
      1 => 1.5,
      2 => -0.25,
      _ => 100.0,
    })
    .collect();
  let intended_b: Vec<f32> = (0..w * h)
    .map(|i| match i % 4 {
      0 => 0.0,
      1 => 0.25,
      2 => 1.0,
      _ => f32::INFINITY,
    })
    .collect();
  let intended_r: Vec<f32> = (0..w * h)
    .map(|i| match i % 4 {
      0 => 1.0,
      1 => -1.0,
      2 => 65505.0,
      _ => 0.5,
    })
    .collect();
  // LE-encode each plane (per the documented `*LE` Frame contract).
  let gp: Vec<f32> = intended_g
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect();
  let bp: Vec<f32> = intended_b
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect();
  let rp: Vec<f32> = intended_r
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect();
  let src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgb_f32 = std::vec![0.0f32; w * h * 3];
  let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
    .with_simd(false)
    .with_rgb_f32(&mut rgb_f32)
    .unwrap();
  gbrpf32_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(
      rgb_f32[i * 3].to_bits(),
      intended_r[i].to_bits(),
      "R idx {i}"
    );
    assert_eq!(
      rgb_f32[i * 3 + 1].to_bits(),
      intended_g[i].to_bits(),
      "G idx {i}"
    );
    assert_eq!(
      rgb_f32[i * 3 + 2].to_bits(),
      intended_b[i].to_bits(),
      "B idx {i}"
    );
  }
}

/// LE-encoded byte contract regression for [`Gbrapf32`] (lossless RGBA
/// pass-through, including the α plane).
///
/// Forces `with_simd(false)` so the test is miri-safe and runs on BE-host
/// miri CI. See the `gbrpf32_sinker_le_encoded_frame_decodes_correctly`
/// docstring for the rationale.
#[test]
fn gbrapf32_sinker_le_encoded_frame_decodes_correctly() {
  let w = 16usize;
  let h = 4usize;
  let intended_g: Vec<f32> = (0..w * h).map(|i| 0.1 + (i as f32) * 0.001).collect();
  let intended_b: Vec<f32> = (0..w * h).map(|i| 0.2 + (i as f32) * 0.002).collect();
  let intended_r: Vec<f32> = (0..w * h).map(|i| 0.3 + (i as f32) * 0.003).collect();
  let intended_a: Vec<f32> = (0..w * h).map(|i| 0.5 + (i as f32) * 0.0005).collect();

  let le = |v: &Vec<f32>| -> Vec<f32> {
    v.iter()
      .map(|&x| f32::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le(&intended_g);
  let bp = le(&intended_b);
  let rp = le(&intended_r);
  let ap = le(&intended_a);

  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_f32 = std::vec![0.0f32; w * h * 4];
  let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
    .with_simd(false)
    .with_rgba_f32(&mut rgba_f32)
    .unwrap();
  gbrapf32_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(
      rgba_f32[i * 4].to_bits(),
      intended_r[i].to_bits(),
      "R idx {i}"
    );
    assert_eq!(
      rgba_f32[i * 4 + 1].to_bits(),
      intended_g[i].to_bits(),
      "G idx {i}"
    );
    assert_eq!(
      rgba_f32[i * 4 + 2].to_bits(),
      intended_b[i].to_bits(),
      "B idx {i}"
    );
    assert_eq!(
      rgba_f32[i * 4 + 3].to_bits(),
      intended_a[i].to_bits(),
      "A idx {i}"
    );
  }
}

/// LE-encoded byte contract regression for [`Gbrpf16`].
///
/// Forces `with_simd(false)` so the kernel runs purely scalar — no SIMD
/// intrinsics — but the fixture builder calls `half::f16::from_f32`, which
/// on aarch64 / x86 / x86_64 with `target_feature = "fp16"` (or F16C)
/// expands to inline `asm!` unsupported by miri. Miri-gated on every
/// target — BE-host miri (s390x / powerpc64) covers the byte-swap
/// correctness via the f32 LE-encoded regressions in this module.
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn gbrpf16_sinker_le_encoded_frame_decodes_correctly() {
  let w = 16usize;
  let h = 4usize;
  let intended_g: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.5,
        1 => 1.5,
        2 => -0.25,
        _ => 100.0,
      })
    })
    .collect();
  let intended_b: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.0,
        1 => 0.25,
        2 => 1.0,
        _ => 65000.0,
      })
    })
    .collect();
  let intended_r: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 1.0,
        1 => -1.0,
        2 => 0.125,
        _ => 0.5,
      })
    })
    .collect();
  let le_f16 = |v: &Vec<half::f16>| -> Vec<half::f16> {
    v.iter()
      .map(|&x| half::f16::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le_f16(&intended_g);
  let bp = le_f16(&intended_b);
  let rp = le_f16(&intended_r);

  let src = Gbrpf16Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgb_f16 = std::vec![half::f16::ZERO; w * h * 3];
  let mut sink = MixedSinker::<Gbrpf16>::new(w, h)
    .with_simd(false)
    .with_rgb_f16(&mut rgb_f16)
    .unwrap();
  gbrpf16_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(
      rgb_f16[i * 3].to_bits(),
      intended_r[i].to_bits(),
      "R idx {i}"
    );
    assert_eq!(
      rgb_f16[i * 3 + 1].to_bits(),
      intended_g[i].to_bits(),
      "G idx {i}"
    );
    assert_eq!(
      rgb_f16[i * 3 + 2].to_bits(),
      intended_b[i].to_bits(),
      "B idx {i}"
    );
  }
}

/// LE-encoded byte contract regression for [`Gbrapf16`] (lossless RGBA
/// pass-through, including the α plane).
///
/// Forces `with_simd(false)` so the kernel runs purely scalar — no SIMD
/// intrinsics — but the fixture builder calls `half::f16::from_f32`, which
/// on aarch64 / x86 / x86_64 with `target_feature = "fp16"` (or F16C)
/// expands to inline `asm!` unsupported by miri. Miri-gated on every
/// target — BE-host miri (s390x / powerpc64) covers the byte-swap
/// correctness via the f32 LE-encoded regressions in this module.
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn gbrapf16_sinker_le_encoded_frame_decodes_correctly() {
  let w = 16usize;
  let h = 4usize;
  let intended_g: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.1 + (i as f32) * 0.001))
    .collect();
  let intended_b: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.2 + (i as f32) * 0.002))
    .collect();
  let intended_r: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.3 + (i as f32) * 0.003))
    .collect();
  let intended_a: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.5 + (i as f32) * 0.001))
    .collect();
  let le_f16 = |v: &Vec<half::f16>| -> Vec<half::f16> {
    v.iter()
      .map(|&x| half::f16::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le_f16(&intended_g);
  let bp = le_f16(&intended_b);
  let rp = le_f16(&intended_r);
  let ap = le_f16(&intended_a);

  let src = Gbrapf16Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_f16 = std::vec![half::f16::ZERO; w * h * 4];
  let mut sink = MixedSinker::<Gbrapf16>::new(w, h)
    .with_simd(false)
    .with_rgba_f16(&mut rgba_f16)
    .unwrap();
  gbrapf16_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(
      rgba_f16[i * 4].to_bits(),
      intended_r[i].to_bits(),
      "R idx {i}"
    );
    assert_eq!(
      rgba_f16[i * 4 + 1].to_bits(),
      intended_g[i].to_bits(),
      "G idx {i}"
    );
    assert_eq!(
      rgba_f16[i * 4 + 2].to_bits(),
      intended_b[i].to_bits(),
      "B idx {i}"
    );
    assert_eq!(
      rgba_f16[i * 4 + 3].to_bits(),
      intended_a[i].to_bits(),
      "A idx {i}"
    );
  }
}

/// LE-encoded byte contract regression for [`Gbrpf16`] **widening path**
/// (`with_rgb_f32`). Exercises the f16 → f32 widen step in the sinker — which
/// must bit-normalise LE-encoded f16 plane bits before converting to f32.
///
/// Vacuous on LE hosts (where `to_le` is identity); on a BE host any
/// regression that drops the bit-normalize-first step in
/// `widen_f16_be_to_host_f32::<false>` would interpret byte-swapped bits as
/// host-native f16 and decode to wildly wrong f32 values.
///
/// Forces `with_simd(false)` so the kernel runs purely scalar — no SIMD
/// intrinsics — but the fixture builder calls `half::f16::from_f32`, which
/// on aarch64 / x86 / x86_64 with `target_feature = "fp16"` (or F16C)
/// expands to inline `asm!` unsupported by miri. Miri-gated on every
/// target — BE-host miri (s390x / powerpc64) covers the byte-swap
/// correctness via the f32 LE-encoded regressions in this module.
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn gbrpf16_sinker_widen_path_le_encoded_frame_decodes_correctly() {
  let w = 16usize;
  let h = 4usize;
  let intended_g: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.5,
        1 => 0.25,
        2 => 0.0,
        _ => 1.0,
      })
    })
    .collect();
  let intended_b: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.125,
        1 => 0.75,
        2 => 0.0625,
        _ => 0.875,
      })
    })
    .collect();
  let intended_r: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.375,
        1 => 0.625,
        2 => 0.9375,
        _ => 0.03125,
      })
    })
    .collect();
  let le_f16 = |v: &Vec<half::f16>| -> Vec<half::f16> {
    v.iter()
      .map(|&x| half::f16::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le_f16(&intended_g);
  let bp = le_f16(&intended_b);
  let rp = le_f16(&intended_r);

  let src = Gbrpf16Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgb_f32 = std::vec![0.0f32; w * h * 3];
  let mut sink = MixedSinker::<Gbrpf16>::new(w, h)
    .with_simd(false)
    .with_rgb_f32(&mut rgb_f32)
    .unwrap();
  gbrpf16_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgb_f32[i * 3], intended_r[i].to_f32(), "R idx {i}");
    assert_eq!(rgb_f32[i * 3 + 1], intended_g[i].to_f32(), "G idx {i}");
    assert_eq!(rgb_f32[i * 3 + 2], intended_b[i].to_f32(), "B idx {i}");
  }
}

/// LE-encoded byte contract regression for [`Gbrapf16`] **widening path**
/// (`with_rgba_f32`, including the α plane). Exercises the four-plane f16 →
/// f32 widen step — same bit-normalise-first contract as the no-α variant.
///
/// Forces `with_simd(false)` so the kernel runs purely scalar — no SIMD
/// intrinsics — but the fixture builder calls `half::f16::from_f32`, which
/// on aarch64 / x86 / x86_64 with `target_feature = "fp16"` (or F16C)
/// expands to inline `asm!` unsupported by miri. Miri-gated on every
/// target — BE-host miri (s390x / powerpc64) covers the byte-swap
/// correctness via the f32 LE-encoded regressions in this module.
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn gbrapf16_sinker_widen_path_le_encoded_frame_decodes_correctly() {
  let w = 16usize;
  let h = 4usize;
  let intended_g: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.1 + (i as f32) * 0.001))
    .collect();
  let intended_b: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.2 + (i as f32) * 0.002))
    .collect();
  let intended_r: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.3 + (i as f32) * 0.003))
    .collect();
  let intended_a: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.5 + (i as f32) * 0.001))
    .collect();
  let le_f16 = |v: &Vec<half::f16>| -> Vec<half::f16> {
    v.iter()
      .map(|&x| half::f16::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le_f16(&intended_g);
  let bp = le_f16(&intended_b);
  let rp = le_f16(&intended_r);
  let ap = le_f16(&intended_a);

  let src = Gbrapf16Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_f32 = std::vec![0.0f32; w * h * 4];
  let mut sink = MixedSinker::<Gbrapf16>::new(w, h)
    .with_simd(false)
    .with_rgba_f32(&mut rgba_f32)
    .unwrap();
  gbrapf16_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgba_f32[i * 4], intended_r[i].to_f32(), "R idx {i}");
    assert_eq!(rgba_f32[i * 4 + 1], intended_g[i].to_f32(), "G idx {i}");
    assert_eq!(rgba_f32[i * 4 + 2], intended_b[i].to_f32(), "B idx {i}");
    assert_eq!(rgba_f32[i * 4 + 3], intended_a[i].to_f32(), "A idx {i}");
  }
}

/// LE-encoded byte contract regression for [`Gbrpf16`] **widening → narrow
/// chain** (`with_rgb_u16` and `with_rgba`). Covers the post-widen routing
/// where `gbrpf32_to_rgb_u16_row` / `gbrpf32_to_rgba_u16_row` /
/// `gbrpf32_to_rgb_row` are invoked on **host-native f32 scratch** produced
/// by `widen_f16_be_to_host_f32::<false>`.
///
/// On a BE host this would have been corrupted under the prior
/// `gbrpf32_to_*::<false>` post-widen routing — that kernel applied
/// `from_le` to scratch that was already host-native, byte-swapping the
/// f32 representation before scaling. Fixed by routing post-widen calls
/// through `::<HOST_NATIVE_BE>` (`true` on BE, `false` on LE), which makes
/// the kernel byte-swap a no-op on every host. Vacuous on LE; would catch
/// the double-swap on BE.
///
/// Forces `with_simd(false)` so the kernel runs purely scalar — no SIMD
/// intrinsics — but the fixture builder calls `half::f16::from_f32`, which
/// on aarch64 / x86 / x86_64 with `target_feature = "fp16"` (or F16C)
/// expands to inline `asm!` unsupported by miri. Miri-gated on every
/// target — BE-host miri (s390x / powerpc64) covers the byte-swap
/// correctness via the f32 LE-encoded regressions in this module.
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn gbrpf16_sinker_widen_path_u16_and_u8_le_encoded_frame_decodes_correctly() {
  let w = 16usize;
  let h = 4usize;
  let intended_g: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.5,
        1 => 0.25,
        2 => 0.0,
        _ => 1.0,
      })
    })
    .collect();
  let intended_b: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.125,
        1 => 0.75,
        2 => 0.0625,
        _ => 0.875,
      })
    })
    .collect();
  let intended_r: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.375,
        1 => 0.625,
        2 => 0.9375,
        _ => 0.03125,
      })
    })
    .collect();
  let le_f16 = |v: &Vec<half::f16>| -> Vec<half::f16> {
    v.iter()
      .map(|&x| half::f16::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le_f16(&intended_g);
  let bp = le_f16(&intended_b);
  let rp = le_f16(&intended_r);

  let src = Gbrpf16Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Exercise the u16 narrow path (post-widen → gbrpf32_to_rgb_u16_row).
  let mut rgb_u16 = std::vec![0u16; w * h * 3];
  // Exercise the u8 narrow path via with_rgba (Strategy A: post-widen
  // is unused for u8 since rgba=opaque-α; we trigger the SAME post-widen
  // path by also attaching luma_u16 alongside u16).
  let mut luma_u16 = std::vec![0u16; w * h];
  {
    let mut sink = MixedSinker::<Gbrpf16>::new(w, h)
      .with_simd(false)
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
    gbrpf16_to(&src, &mut sink).unwrap();
  }

  // Assert RGB u16 output matches the intended (clamp+scale x 65535) values.
  let to_u16 = |v: f32| -> u16 { (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16 };
  for i in 0..(w * h) {
    assert_eq!(
      rgb_u16[i * 3],
      to_u16(intended_r[i].to_f32()),
      "RGB u16 R idx {i}"
    );
    assert_eq!(
      rgb_u16[i * 3 + 1],
      to_u16(intended_g[i].to_f32()),
      "RGB u16 G idx {i}"
    );
    assert_eq!(
      rgb_u16[i * 3 + 2],
      to_u16(intended_b[i].to_f32()),
      "RGB u16 B idx {i}"
    );
  }
  // Sanity: luma_u16 (post-widen narrow) is non-zero — locks down that
  // the post-widen luma kernel also sees host-native f32 scratch.
  assert!(
    luma_u16.iter().any(|&v| v > 0),
    "luma_u16 must contain non-zero samples — \
     a corrupted byte-swap would still emit non-zero output but the rgb_u16 \
     assertion above is the primary guard"
  );
}

// LE-encoded Strategy A+ alpha-patch regressions.
//
// The `copy_alpha_plane_f32_to_u8` (and `copy_alpha_plane_f32_to_u16`,
// `copy_alpha_plane_f32`) helpers must read each f32 α sample with the
// same LE/host-endian gate as the rest of the float planar pipeline. A
// host-native read silently corrupts the α slot on BE hosts processing
// the LE-encoded `Gbrapf32Frame` α plane (the byte-swapped bits clamp to
// near-zero or near-one, producing α = 0 or 255 regardless of intent).
//
// These regressions trigger the **Strategy A+ combo path** (`with_rgb` +
// `with_rgba`, `with_rgb_u16` + `with_rgba_u16`) on a Frame whose α plane
// is built from explicit LE-encoded f32 bit-patterns. On a LE host the
// `to_le` on f32 bits is identity so the test reduces to the original
// semantics; on a BE host the kernel without `from_le` would clamp
// byte-swapped garbage and the assertion would fail. The non-multiple-of-
// SIMD widths (15, 17) exercise scalar-tail correctness in addition to
// any vectorized body.

/// Gbrapf32 Strategy A+ (`with_rgb` + `with_rgba`) on a LE-encoded f32 α
/// plane must reproduce standalone `with_rgba` output byte-for-byte. The
/// standalone path uses `gbrapf32_to_rgba_row::<false>` (already endian-
/// aware), so any deviation indicates the Strategy A+ alpha-patch path
/// corrupted the α plane.
///
/// Forces `with_simd(false)` so the test is miri-safe and runs on BE-host
/// miri CI. See the `gbrpf32_sinker_le_encoded_frame_decodes_correctly`
/// docstring for the rationale.
#[test]
fn gbrapf32_strategy_a_plus_le_encoded_frame_alpha_decodes_correctly() {
  // 15 is non-multiple-of-{4,8,16} — exercises scalar tail in every backend.
  let w = 15usize;
  let h = 3usize;
  let intended_g: Vec<f32> = (0..w * h).map(|i| 0.10 + (i as f32) * 0.001).collect();
  let intended_b: Vec<f32> = (0..w * h).map(|i| 0.20 + (i as f32) * 0.002).collect();
  let intended_r: Vec<f32> = (0..w * h).map(|i| 0.30 + (i as f32) * 0.003).collect();
  // Deliberately mix in-range, boundary, > 1, and negative α to stress
  // clamp/scale correctness *after* the bit-normalize step.
  let intended_a: Vec<f32> = (0..w * h)
    .map(|i| match i % 7 {
      0 => 0.0,
      1 => 0.5,
      2 => 1.0,
      3 => 1.5,
      4 => -0.1,
      5 => 0.123,
      _ => 0.876,
    })
    .collect();

  // LE-encode every plane (per the documented `*LE` Frame contract).
  let le = |v: &Vec<f32>| -> Vec<f32> {
    v.iter()
      .map(|&x| f32::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le(&intended_g);
  let bp = le(&intended_b);
  let rp = le(&intended_r);
  let ap = le(&intended_a);

  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Reference: standalone `with_rgba`.
  let mut rgba_ref = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_simd(false)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Strategy A+: `with_rgb` + `with_rgba` combo (alpha-patch path).
  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_simd(false)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combo, rgba_ref,
    "Gbrapf32 Strategy A+ alpha-patch must equal standalone `with_rgba`"
  );
}

/// Gbrapf32 Strategy A+ (`with_rgb_u16` + `with_rgba_u16`) on a
/// LE-encoded f32 α plane. Defense-in-depth: the
/// current sinker calls `gbrapf32_to_rgba_u16_row::<false>` directly here
/// (no alpha-patch helper invocation), but any future routing change that
/// switches to the alpha-patch helper must keep BE-host correctness.
///
/// Forces `with_simd(false)` so the test is miri-safe and runs on BE-host
/// miri CI. See the `gbrpf32_sinker_le_encoded_frame_decodes_correctly`
/// docstring for the rationale.
#[test]
fn gbrapf32_strategy_a_plus_le_encoded_u16_alpha_decodes_correctly() {
  // 17 is non-multiple-of-{4,8,16}.
  let w = 17usize;
  let h = 3usize;
  let intended_g: Vec<f32> = (0..w * h).map(|i| 0.11 + (i as f32) * 0.0011).collect();
  let intended_b: Vec<f32> = (0..w * h).map(|i| 0.22 + (i as f32) * 0.0022).collect();
  let intended_r: Vec<f32> = (0..w * h).map(|i| 0.33 + (i as f32) * 0.0033).collect();
  let intended_a: Vec<f32> = (0..w * h)
    .map(|i| match i % 5 {
      0 => 0.0,
      1 => 0.25,
      2 => 1.0,
      3 => 0.5,
      _ => 0.75,
    })
    .collect();

  let le = |v: &Vec<f32>| -> Vec<f32> {
    v.iter()
      .map(|&x| f32::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le(&intended_g);
  let bp = le(&intended_b);
  let rp = le(&intended_r);
  let ap = le(&intended_a);

  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Reference: standalone `with_rgba_u16`.
  let mut rgba_ref = std::vec![0u16; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_simd(false)
      .with_rgba_u16(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Combo: `with_rgb_u16` + `with_rgba_u16`.
  let mut rgb_combo = std::vec![0u16; w * h * 3];
  let mut rgba_combo = std::vec![0u16; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_simd(false)
      .with_rgb_u16(&mut rgb_combo)
      .unwrap()
      .with_rgba_u16(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combo, rgba_ref,
    "Gbrapf32 Strategy A+ rgba_u16 must equal standalone `with_rgba_u16`"
  );

  // Independently assert the α slot reflects the intended values
  // (clamp x 65535 + 0.5). This catches a hypothetical regression where
  // both code paths share the same bug.
  let to_u16 = |v: f32| -> u16 { (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16 };
  for i in 0..(w * h) {
    assert_eq!(
      rgba_combo[i * 4 + 3],
      to_u16(intended_a[i]),
      "α slot idx {i}"
    );
  }
}

/// Gbrapf16 Strategy A+ (`with_rgb` + `with_rgba`) on a LE-encoded f16
/// α plane. This exercises the **post-widen** routing
/// pattern in `widen_and_scatter_f16_alpha_to_u8`: the f16 α plane is
/// widened to host-native f32 scratch via `widen_f16_be_to_host_f32::<false>`,
/// then the alpha-patch helper must consume that scratch with
/// `BE = HOST_NATIVE_BE` (no double byte-swap). The test compares the
/// Strategy A+ combo output against the standalone `with_rgba` path, which
/// uses the `gbrpf16_to_rgba_row::<false>` direct kernel + the same
/// `widen_and_scatter_f16_alpha_to_u8` helper (both paths share the
/// `widen_and_scatter` helper, so this test guards against the
/// post-widen routing flag being wrong).
///
/// Forces `with_simd(false)` so the kernel runs purely scalar — no SIMD
/// intrinsics — but the fixture builder calls `half::f16::from_f32`, which
/// on aarch64 / x86 / x86_64 with `target_feature = "fp16"` (or F16C)
/// expands to inline `asm!` unsupported by miri. Miri-gated on every
/// target — BE-host miri (s390x / powerpc64) covers the byte-swap
/// correctness via the f32 LE-encoded regressions in this module.
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn gbrapf16_strategy_a_plus_post_widen_alpha_decodes_correctly() {
  let w = 15usize;
  let h = 3usize;
  let intended_g: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.10 + (i as f32) * 0.001))
    .collect();
  let intended_b: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.20 + (i as f32) * 0.002))
    .collect();
  let intended_r: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.30 + (i as f32) * 0.003))
    .collect();
  let intended_a: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 5 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 0.25,
        _ => 0.75,
      })
    })
    .collect();

  let le_f16 = |v: &Vec<half::f16>| -> Vec<half::f16> {
    v.iter()
      .map(|&x| half::f16::from_bits(x.to_bits().to_le()))
      .collect()
  };
  let gp = le_f16(&intended_g);
  let bp = le_f16(&intended_b);
  let rp = le_f16(&intended_r);
  let ap = le_f16(&intended_a);

  let src = Gbrapf16Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Reference: standalone `with_rgba` (uses `gbrpf16_to_rgba_row` then
  // `widen_and_scatter_f16_alpha_to_u8` → exercises the post-widen helper
  // as well, with the same routing as the combo path).
  let mut rgba_ref = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf16>::new(w, h)
      .with_simd(false)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }

  // Strategy A+: `with_rgb` + `with_rgba`.
  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf16>::new(w, h)
      .with_simd(false)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba(&mut rgba_combo)
      .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combo, rgba_ref,
    "Gbrapf16 Strategy A+ post-widen alpha-patch must equal standalone `with_rgba`"
  );

  // Independently assert α slot reflects the intended values
  // (widen → clamp x 255 + 0.5).
  let to_u8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8 };
  for i in 0..(w * h) {
    assert_eq!(
      rgba_combo[i * 4 + 3],
      to_u8(intended_a[i].to_f32()),
      "α slot idx {i}"
    );
  }
}

// ---- 32-bit overflow guards ------------------------------------------------
//
// Feeding width = usize::MAX / 2 + 1 to a dispatcher must panic with a message
// containing "overflows usize". These tests are only meaningful on 32-bit
// targets where usize = u32. On 64-bit targets the width would be ~2 GiB which
// won't trigger the overflow guard, so the tests are compiled-out.

#[cfg(target_pointer_width = "32")]
#[test]
#[should_panic(expected = "overflows usize")]
fn gbr_float_dispatch_panics_on_width_overflow_gbrpf32_rgb() {
  let bad_width = usize::MAX / 2 + 1;
  // Allocate 1-element planes — the overflow panic fires before the plane-len
  // check so the short planes won't be the cause of the panic.
  let g = [0.0f32; 1];
  let b = [0.0f32; 1];
  let r = [0.0f32; 1];
  let mut out = [0u8; 3];
  crate::row::gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out, bad_width, false);
}

#[cfg(target_pointer_width = "32")]
#[test]
#[should_panic(expected = "overflows usize")]
fn gbr_float_dispatch_panics_on_width_overflow_gbrpf32_rgba() {
  let bad_width = usize::MAX / 2 + 1;
  let g = [0.0f32; 1];
  let b = [0.0f32; 1];
  let r = [0.0f32; 1];
  let mut out = [0u8; 4];
  crate::row::gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut out, bad_width, false);
}

#[cfg(target_pointer_width = "32")]
#[test]
#[should_panic(expected = "overflows usize")]
fn gbr_float_dispatch_panics_on_width_overflow_gbrpf32_rgb_u16() {
  let bad_width = usize::MAX / 2 + 1;
  let g = [0.0f32; 1];
  let b = [0.0f32; 1];
  let r = [0.0f32; 1];
  let mut out = [0u16; 3];
  crate::row::gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut out, bad_width, false);
}

#[cfg(target_pointer_width = "32")]
#[test]
#[should_panic(expected = "overflows usize")]
fn gbr_float_dispatch_panics_on_width_overflow_gbrpf32_rgba_u16() {
  let bad_width = usize::MAX / 2 + 1;
  let g = [0.0f32; 1];
  let b = [0.0f32; 1];
  let r = [0.0f32; 1];
  let mut out = [0u16; 4];
  crate::row::gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut out, bad_width, false);
}

// ====================================================================================
// Phase 4 — Frame BE flag, Tier 10 float. LE+BE round-trip parity tests.
//
// Pattern: build a host-native plane, encode bits as LE (`to_le_bytes`) and
// BE (`to_be_bytes`) byte storage reinterpreted via `from_ne_bytes` /
// `from_le_bytes`, drive each through its `MixedSinker<MarkerN<BE>>`
// monomorphization, and assert the outputs are byte-identical. The kernel
// performs the byte-swap (or no-op) of the float bit pattern under the hood.
// ====================================================================================

fn as_le_f32_buf(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

fn as_be_f32_buf(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

fn as_le_f16_buf(host: &[half::f16]) -> std::vec::Vec<half::f16> {
  host
    .iter()
    .map(|v| {
      let bits = v.to_bits();
      let native_bits = u16::from_ne_bytes(bits.to_le_bytes());
      half::f16::from_bits(native_bits)
    })
    .collect()
}

fn as_be_f16_buf(host: &[half::f16]) -> std::vec::Vec<half::f16> {
  // Encode bits as BE, then reinterpret those bytes as a host-native f16 so
  // the resulting `&[half::f16]` slice has the BE byte pattern in memory.
  host
    .iter()
    .map(|v| {
      let bits = v.to_bits();
      let native_bits = u16::from_ne_bytes(bits.to_be_bytes());
      half::f16::from_bits(native_bits)
    })
    .collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_le_be_roundtrip_byte_identical() {
  let w = 16usize;
  let h = 4usize;
  let intended: std::vec::Vec<f32> = (0..w * h)
    .map(|i| {
      let v = (i as f32) / (w * h) as f32;
      match i % 4 {
        0 => v,
        1 => 1.0 - v,
        2 => 0.5,
        _ => 0.25,
      }
    })
    .collect();
  let g_le = as_le_f32_buf(&intended);
  let b_le = as_le_f32_buf(&intended);
  let r_le = as_le_f32_buf(&intended);
  let g_be = as_be_f32_buf(&intended);
  let b_be = as_be_f32_buf(&intended);
  let r_be = as_be_f32_buf(&intended);

  // Cover both scalar and SIMD dispatch — the SIMD path catches missing
  // `<BE>` propagation in the SIMD-aware row kernels that scalar misses.
  for use_simd in [false, true] {
    let stride = w as u32;
    let frame_le = Gbrpf32LeFrame::try_new(
      &g_le, &b_le, &r_le, w as u32, h as u32, stride, stride, stride,
    )
    .unwrap();
    let mut out_le = std::vec![0u8; w * h * 4];
    let mut sink_le = MixedSinker::<Gbrpf32>::new(w, h)
      .with_simd(use_simd)
      .with_rgba(&mut out_le)
      .unwrap();
    gbrpf32_to(&frame_le, &mut sink_le).unwrap();

    let frame_be = Gbrpf32BeFrame::try_new(
      &g_be, &b_be, &r_be, w as u32, h as u32, stride, stride, stride,
    )
    .unwrap();
    let mut out_be = std::vec![0u8; w * h * 4];
    let mut sink_be = MixedSinker::<Gbrpf32<true>>::new(w, h)
      .with_simd(use_simd)
      .with_rgba(&mut out_be)
      .unwrap();
    // BE-frame call must use the `_endian` helper.
    gbrpf32_to_endian(&frame_be, &mut sink_be).unwrap();

    assert_eq!(
      out_le, out_be,
      "Gbrpf32 LE/BE outputs diverge — `<const BE>` propagation broken (use_simd={use_simd})",
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_le_be_roundtrip_byte_identical() {
  let w = 16usize;
  let h = 4usize;
  let intended: std::vec::Vec<f32> = (0..w * h)
    .map(|i| {
      let v = (i as f32) / (w * h) as f32;
      match i % 5 {
        0 => v,
        1 => 1.0 - v,
        2 => 0.5,
        3 => 0.25,
        _ => 0.75,
      }
    })
    .collect();
  let g_le = as_le_f32_buf(&intended);
  let b_le = as_le_f32_buf(&intended);
  let r_le = as_le_f32_buf(&intended);
  let a_le = as_le_f32_buf(&intended);
  let g_be = as_be_f32_buf(&intended);
  let b_be = as_be_f32_buf(&intended);
  let r_be = as_be_f32_buf(&intended);
  let a_be = as_be_f32_buf(&intended);

  for use_simd in [false, true] {
    let stride = w as u32;
    let frame_le = Gbrapf32LeFrame::try_new(
      &g_le, &b_le, &r_le, &a_le, w as u32, h as u32, stride, stride, stride, stride,
    )
    .unwrap();
    let mut out_le = std::vec![0u8; w * h * 4];
    let mut sink_le = MixedSinker::<Gbrapf32>::new(w, h)
      .with_simd(use_simd)
      .with_rgba(&mut out_le)
      .unwrap();
    gbrapf32_to(&frame_le, &mut sink_le).unwrap();

    let frame_be = Gbrapf32BeFrame::try_new(
      &g_be, &b_be, &r_be, &a_be, w as u32, h as u32, stride, stride, stride, stride,
    )
    .unwrap();
    let mut out_be = std::vec![0u8; w * h * 4];
    let mut sink_be = MixedSinker::<Gbrapf32<true>>::new(w, h)
      .with_simd(use_simd)
      .with_rgba(&mut out_be)
      .unwrap();
    gbrapf32_to_endian(&frame_be, &mut sink_be).unwrap();

    assert_eq!(
      out_le, out_be,
      "Gbrapf32 LE/BE outputs diverge — `<const BE>` propagation broken (use_simd={use_simd})",
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_le_be_roundtrip_byte_identical() {
  let w = 16usize;
  let h = 4usize;
  let intended: std::vec::Vec<half::f16> = (0..w * h)
    .map(|i| {
      let v = (i as f32) / (w * h) as f32;
      half::f16::from_f32(match i % 4 {
        0 => v,
        1 => 1.0 - v,
        2 => 0.5,
        _ => 0.25,
      })
    })
    .collect();
  let g_le = as_le_f16_buf(&intended);
  let b_le = as_le_f16_buf(&intended);
  let r_le = as_le_f16_buf(&intended);
  let g_be = as_be_f16_buf(&intended);
  let b_be = as_be_f16_buf(&intended);
  let r_be = as_be_f16_buf(&intended);

  for use_simd in [false, true] {
    let stride = w as u32;
    let frame_le = Gbrpf16LeFrame::try_new(
      &g_le, &b_le, &r_le, w as u32, h as u32, stride, stride, stride,
    )
    .unwrap();
    let mut out_le = std::vec![0u8; w * h * 4];
    let mut sink_le = MixedSinker::<Gbrpf16>::new(w, h)
      .with_simd(use_simd)
      .with_rgba(&mut out_le)
      .unwrap();
    gbrpf16_to(&frame_le, &mut sink_le).unwrap();

    let frame_be = Gbrpf16BeFrame::try_new(
      &g_be, &b_be, &r_be, w as u32, h as u32, stride, stride, stride,
    )
    .unwrap();
    let mut out_be = std::vec![0u8; w * h * 4];
    let mut sink_be = MixedSinker::<Gbrpf16<true>>::new(w, h)
      .with_simd(use_simd)
      .with_rgba(&mut out_be)
      .unwrap();
    gbrpf16_to_endian(&frame_be, &mut sink_be).unwrap();

    assert_eq!(
      out_le, out_be,
      "Gbrpf16 LE/BE outputs diverge — `<const BE>` propagation broken (use_simd={use_simd})",
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_le_be_roundtrip_byte_identical() {
  let w = 16usize;
  let h = 4usize;
  let intended: std::vec::Vec<half::f16> = (0..w * h)
    .map(|i| {
      let v = (i as f32) / (w * h) as f32;
      half::f16::from_f32(match i % 5 {
        0 => v,
        1 => 1.0 - v,
        2 => 0.5,
        3 => 0.25,
        _ => 0.75,
      })
    })
    .collect();
  let g_le = as_le_f16_buf(&intended);
  let b_le = as_le_f16_buf(&intended);
  let r_le = as_le_f16_buf(&intended);
  let a_le = as_le_f16_buf(&intended);
  let g_be = as_be_f16_buf(&intended);
  let b_be = as_be_f16_buf(&intended);
  let r_be = as_be_f16_buf(&intended);
  let a_be = as_be_f16_buf(&intended);

  for use_simd in [false, true] {
    let stride = w as u32;
    let frame_le = Gbrapf16LeFrame::try_new(
      &g_le, &b_le, &r_le, &a_le, w as u32, h as u32, stride, stride, stride, stride,
    )
    .unwrap();
    let mut out_le = std::vec![0u8; w * h * 4];
    let mut sink_le = MixedSinker::<Gbrapf16>::new(w, h)
      .with_simd(use_simd)
      .with_rgba(&mut out_le)
      .unwrap();
    gbrapf16_to(&frame_le, &mut sink_le).unwrap();

    let frame_be = Gbrapf16BeFrame::try_new(
      &g_be, &b_be, &r_be, &a_be, w as u32, h as u32, stride, stride, stride, stride,
    )
    .unwrap();
    let mut out_be = std::vec![0u8; w * h * 4];
    let mut sink_be = MixedSinker::<Gbrapf16<true>>::new(w, h)
      .with_simd(use_simd)
      .with_rgba(&mut out_be)
      .unwrap();
    gbrapf16_to_endian(&frame_be, &mut sink_be).unwrap();

    assert_eq!(
      out_le, out_be,
      "Gbrapf16 LE/BE outputs diverge — `<const BE>` propagation broken (use_simd={use_simd})",
    );
  }
}
