//! Integration tests for the 32-bit planar GBR + alpha (`Gbrap32`) sinker.
//!
//! Mirrors the `Gbrap16` sinker tests at the `u32` element:
//! - Channel reorder (G/B/R/A → R, G, B[, A]) and saturation.
//! - u32 → u8 (`>> 24`) / u32 → u16 (`>> 16`) narrow + low-bit drop.
//! - Real per-pixel alpha pass-through.
//! - Strategy A+ combo (with_rgb + with_rgba == standalone with_rgba).
//! - SIMD vs scalar equivalence (SIMD main loop + scalar tail widths).
//! - Frame LE/BE byte-order parity.
//! - Native-precision Q15 `luma_u16`.

use super::*;
use crate::sinker::MixedSinker;

/// Build a solid-colour Gbrap32 LE frame from four `u32` planes.
fn gbrap32_frame<'a>(
  g: &'a [u32],
  b: &'a [u32],
  r: &'a [u32],
  a: &'a [u32],
  w: u32,
  h: u32,
) -> crate::frame::Gbrap32LeFrame<'a> {
  crate::frame::Gbrap32LeFrame::try_new(g, b, r, a, w, h, w, w, w, w).unwrap()
}

/// Deterministic pseudo-random `u32` plane (full 32-bit range, nonzero low
/// bits so the narrow is genuinely lossy).
fn pseudo_random_u32(out: &mut [u32], seed: u32) {
  let mut s = seed.wrapping_mul(0x9E37_79B9).wrapping_add(1);
  for v in out.iter_mut() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *v = s;
  }
}

// ---- Channel reorder + saturation ------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap32_channel_reorder_u8() {
  let w = 16usize;
  let h = 4usize;
  let g = std::vec![100u32 << 24; w * h];
  let b = std::vec![50u32 << 24; w * h];
  let r = std::vec![200u32 << 24; w * h];
  let a = std::vec![0xFFFF_FFFFu32; w * h];
  let src = gbrap32_frame(&g, &b, &r, &a, w as u32, h as u32);
  let mut out = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_rgb(&mut out)
    .unwrap();
  crate::source::gbrap32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  for i in 0..w * h {
    assert_eq!(out[i * 3], 200, "R[{i}]");
    assert_eq!(out[i * 3 + 1], 100, "G[{i}]");
    assert_eq!(out[i * 3 + 2], 50, "B[{i}]");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap32_all_max_saturates_u8_and_u16() {
  let w = 12usize;
  let h = 3usize;
  let n = w * h;
  let p = std::vec![0xFFFF_FFFFu32; n];
  let src = gbrap32_frame(&p, &p, &p, &p, w as u32, h as u32);
  let mut u8out = std::vec![0u8; n * 4];
  let mut u16out = std::vec![0u16; n * 4];
  let mut sink = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_rgba(&mut u8out)
    .unwrap()
    .with_rgba_u16(&mut u16out)
    .unwrap();
  crate::source::gbrap32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert!(u8out.iter().all(|&v| v == 0xFF), "u8 must saturate to 0xFF");
  assert!(
    u16out.iter().all(|&v| v == 0xFFFF),
    "u16 must saturate to 0xFFFF"
  );
}

/// The u32 high-bit narrow: u8 takes bits [31:24], u16 takes bits [31:16].
/// Low bits below the narrow point are dropped on the direct path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap32_narrow_drops_low_bits() {
  let w = 8usize;
  let h = 1usize;
  let n = w * h;
  // R plane carries a distinct high byte/halfword with nonzero low 16 bits.
  let g = std::vec![0x1234_5678u32; n];
  let b = std::vec![0x9ABC_DEF0u32; n];
  let r = std::vec![0xCAFE_BEEFu32; n];
  let a = std::vec![0x8000_FFFFu32; n];
  let src = gbrap32_frame(&g, &b, &r, &a, w as u32, h as u32);
  let mut u8out = std::vec![0u8; n * 4];
  let mut u16out = std::vec![0u16; n * 4];
  let mut sink = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_rgba(&mut u8out)
    .unwrap()
    .with_rgba_u16(&mut u16out)
    .unwrap();
  crate::source::gbrap32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  // u8: high byte (>> 24).
  assert_eq!(u8out[0], 0xCA, "R u8 = R>>24");
  assert_eq!(u8out[1], 0x12, "G u8 = G>>24");
  assert_eq!(u8out[2], 0x9A, "B u8 = B>>24");
  assert_eq!(u8out[3], 0x80, "A u8 = A>>24");
  // u16: high halfword (>> 16).
  assert_eq!(u16out[0], 0xCAFE, "R u16 = R>>16");
  assert_eq!(u16out[1], 0x1234, "G u16 = G>>16");
  assert_eq!(u16out[2], 0x9ABC, "B u16 = B>>16");
  assert_eq!(u16out[3], 0x8000, "A u16 = A>>16");
}

// ---- Strategy A+ combo ------------------------------------------------------

/// with_rgb + with_rgba (Strategy A+) produces the same RGBA as standalone
/// with_rgba.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap32_strategy_a_plus_u8_matches_standalone() {
  let w = 32usize;
  let h = 8usize;
  let n = w * h;
  let mut g = std::vec![0u32; n];
  let mut b = std::vec![0u32; n];
  let mut r = std::vec![0u32; n];
  let mut a = std::vec![0u32; n];
  pseudo_random_u32(&mut g, 0x11);
  pseudo_random_u32(&mut b, 0x22);
  pseudo_random_u32(&mut r, 0x33);
  pseudo_random_u32(&mut a, 0x44);

  let src = gbrap32_frame(&g, &b, &r, &a, w as u32, h as u32);
  let mut rgba_ref = std::vec![0u8; n * 4];
  let mut sink_ref = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_rgba(&mut rgba_ref)
    .unwrap();
  crate::source::gbrap32_to(&src, false, sink_ref.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let src2 = gbrap32_frame(&g, &b, &r, &a, w as u32, h as u32);
  let mut rgb_combo = std::vec![0u8; n * 3];
  let mut rgba_combo = std::vec![0u8; n * 4];
  let mut sink_combo = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_rgb(&mut rgb_combo)
    .unwrap()
    .with_rgba(&mut rgba_combo)
    .unwrap();
  crate::source::gbrap32_to(
    &src2,
    false,
    sink_combo.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(rgba_ref, rgba_combo, "Strategy A+ u8 RGBA mismatch");
}

/// with_rgb_u16 + with_rgba_u16 (Strategy A+) matches standalone with_rgba_u16.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap32_strategy_a_plus_u16_matches_standalone() {
  let w = 32usize;
  let h = 8usize;
  let n = w * h;
  let mut g = std::vec![0u32; n];
  let mut b = std::vec![0u32; n];
  let mut r = std::vec![0u32; n];
  let mut a = std::vec![0u32; n];
  pseudo_random_u32(&mut g, 0x55);
  pseudo_random_u32(&mut b, 0x66);
  pseudo_random_u32(&mut r, 0x77);
  pseudo_random_u32(&mut a, 0x88);

  let src = gbrap32_frame(&g, &b, &r, &a, w as u32, h as u32);
  let mut ref_u16 = std::vec![0u16; n * 4];
  let mut sink_ref = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_rgba_u16(&mut ref_u16)
    .unwrap();
  crate::source::gbrap32_to(&src, true, sink_ref.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let src2 = gbrap32_frame(&g, &b, &r, &a, w as u32, h as u32);
  let mut rgb_u16 = std::vec![0u16; n * 3];
  let mut combo_u16 = std::vec![0u16; n * 4];
  let mut sink_combo = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut combo_u16)
    .unwrap();
  crate::source::gbrap32_to(
    &src2,
    true,
    sink_combo.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(ref_u16, combo_u16, "Strategy A+ u16 RGBA mismatch");
}

// ---- SIMD vs scalar equivalence (main loop + tail widths) ------------------

macro_rules! gbrap32_simd_matches_scalar {
  ($name:ident, $w:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = $w as usize;
      let h = 8usize;
      let n = w * h;
      let mut g = std::vec![0u32; n];
      let mut b = std::vec![0u32; n];
      let mut r = std::vec![0u32; n];
      let mut a = std::vec![0u32; n];
      pseudo_random_u32(&mut g, 0xA1);
      pseudo_random_u32(&mut b, 0xB1);
      pseudo_random_u32(&mut r, 0xC1);
      pseudo_random_u32(&mut a, 0xD1);

      let src = gbrap32_frame(&g, &b, &r, &a, w as u32, h as u32);
      let mut rgb_simd = std::vec![0u8; n * 3];
      let mut rgba_simd = std::vec![0u8; n * 4];
      let mut rgb_u16_simd = std::vec![0u16; n * 3];
      let mut rgba_u16_simd = std::vec![0u16; n * 4];
      let mut luma_u16_simd = std::vec![0u16; n];
      {
        let mut sink = MixedSinker::<crate::source::Gbrap32>::new(w, h)
          .with_simd(true)
          .with_rgb(&mut rgb_simd)
          .unwrap()
          .with_rgba(&mut rgba_simd)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16_simd)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16_simd)
          .unwrap()
          .with_luma_u16(&mut luma_u16_simd)
          .unwrap();
        crate::source::gbrap32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      }

      let mut rgb_scal = std::vec![0u8; n * 3];
      let mut rgba_scal = std::vec![0u8; n * 4];
      let mut rgb_u16_scal = std::vec![0u16; n * 3];
      let mut rgba_u16_scal = std::vec![0u16; n * 4];
      let mut luma_u16_scal = std::vec![0u16; n];
      {
        let mut sink = MixedSinker::<crate::source::Gbrap32>::new(w, h)
          .with_simd(false)
          .with_rgb(&mut rgb_scal)
          .unwrap()
          .with_rgba(&mut rgba_scal)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16_scal)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16_scal)
          .unwrap()
          .with_luma_u16(&mut luma_u16_scal)
          .unwrap();
        crate::source::gbrap32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      }

      assert_eq!(rgb_simd, rgb_scal, "rgb SIMD≠scalar w={}", $w);
      assert_eq!(rgba_simd, rgba_scal, "rgba SIMD≠scalar w={}", $w);
      assert_eq!(rgb_u16_simd, rgb_u16_scal, "rgb_u16 SIMD≠scalar w={}", $w);
      assert_eq!(rgba_u16_simd, rgba_u16_scal, "rgba_u16 SIMD≠scalar w={}", $w);
      assert_eq!(luma_u16_simd, luma_u16_scal, "luma_u16 differs w={}", $w);
    }
  };
}

// Widths exercise the SIMD main loop (multiples of 8/16) and scalar tails.
gbrap32_simd_matches_scalar!(gbrap32_w128_simd_matches_scalar, 128);
gbrap32_simd_matches_scalar!(gbrap32_w130_simd_matches_scalar, 130);
gbrap32_simd_matches_scalar!(gbrap32_w131_simd_matches_scalar, 131);

// ---- Frame LE/BE byte-order parity -----------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap32_le_be_roundtrip() {
  let w = 16usize;
  let h = 4usize;
  let n = w * h;
  let intended: std::vec::Vec<u32> = (0..n)
    .map(|i| {
      (i as u32)
        .wrapping_mul(0x0123_4567)
        .wrapping_add(0xCAFE_0001)
    })
    .collect();
  let g_le: std::vec::Vec<u32> = intended
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect();
  let g_be: std::vec::Vec<u32> = intended
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect();

  for use_simd in [false, true] {
    let stride = w as u32;
    let frame_le = crate::frame::Gbrap32LeFrame::try_new(
      &g_le, &g_le, &g_le, &g_le, w as u32, h as u32, stride, stride, stride, stride,
    )
    .unwrap();
    let mut out_le = std::vec![0u16; n * 4];
    let mut sink_le = MixedSinker::<crate::source::Gbrap32>::new(w, h)
      .with_simd(use_simd)
      .with_rgba_u16(&mut out_le)
      .unwrap();
    crate::source::gbrap32_to(
      &frame_le,
      true,
      sink_le.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();

    let frame_be = crate::frame::Gbrap32BeFrame::try_new(
      &g_be, &g_be, &g_be, &g_be, w as u32, h as u32, stride, stride, stride, stride,
    )
    .unwrap();
    let mut out_be = std::vec![0u16; n * 4];
    let mut sink_be = MixedSinker::<crate::source::Gbrap32<true>>::new(w, h)
      .with_simd(use_simd)
      .with_rgba_u16(&mut out_be)
      .unwrap();
    crate::source::gbrap32_to_endian::<_, true>(
      &frame_be,
      true,
      sink_be.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();

    assert_eq!(
      out_le, out_be,
      "Gbrap32 LE/BE diverge — `<const BE>` propagation broken (use_simd={use_simd})"
    );
  }
}

// ---- luma_u16 native precision ---------------------------------------------

/// Neutral grey: G = B = R produces Y' ≈ the narrowed grey value (full-range
/// Q15 coefficients sum to 1).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap32_luma_u16_neutral_grey() {
  let w = 8usize;
  let h = 2usize;
  let n = w * h;
  let grey = 0x6789_ABCDu32; // narrows >> 16 to 0x6789
  let g = std::vec![grey; n];
  let a = std::vec![0xFFFF_FFFFu32; n];
  let src = gbrap32_frame(&g, &g, &g, &a, w as u32, h as u32);
  let mut luma = std::vec![0u16; n];
  let mut sink = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_luma_u16(&mut luma)
    .unwrap();
  crate::source::gbrap32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  for &y in &luma {
    assert!(
      (y as i32 - 0x6789).abs() <= 1,
      "neutral-grey Y' must equal narrowed grey ±1, got {y:#06x}"
    );
  }
}

// ---- Atomicity (#308) -------------------------------------------------------
//
// The `Gbrap32` identity-path `process` must run the RGB-staging preflight
// BEFORE any output row is written. The native-precision `luma_u16` row
// (`gbr32_to_luma_u16_row`, no RGB staging) is written ahead of the RGB-staging
// scratch whose allocating (`None`) arm is reachable via `need_rgb_staging &&
// rgb.is_none()`, so an allocator refusal there must leave the output frame
// untouched rather than partially mutated. This test attaches
// `luma_u16 + HSV` (no RGB): HSV forces `need_rgb_staging`, the scratch has no
// caller RGB buffer to borrow (so it takes its allocating arm), and luma_u16 is
// the before-scratch sentinel. Reuses the crate's RGB-scratch failpoint
// (`yuva`-gated, so this test is too).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap32_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (w, h) = (16usize, 8usize);
  let n = w * h;
  let g = std::vec![0x1234_5678u32; n];
  let b = std::vec![0x2345_6789u32; n];
  let r = std::vec![0x3456_789Au32; n];
  let a = std::vec![0xFFFF_FFFFu32; n];
  let src = gbrap32_frame(&g, &b, &r, &a, w as u32, h as u32);
  let mut luma_u16 = std::vec![0xABABu16; n];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<crate::source::Gbrap32>::new(w, h)
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err =
    crate::source::gbrap32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-staging-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma_u16.iter().all(|&v| v == 0xABAB),
    "native luma_u16 (written before the rgb-staging scratch) must be untouched on the alloc-failure path"
  );
}
