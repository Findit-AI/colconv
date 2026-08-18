//! Integration tests for high-bit-depth planar GBR sinker impls.
//!
//! Covers:
//! - Bit-depth saturation: max-value inputs produce max-value output (both
//!   u16 native and u8 downshifted).
//! - Channel reorder: G=100, B=50, R=200 → packed output (R, G, B) = (200, 100, 50).
//! - Strategy A+ correctness (Gbrap): with_rgb + with_rgba combo produces the
//!   same RGBA as standalone with_rgba using the direct 4-channel kernel.
//! - SIMD vs scalar equivalence for widths that exercise SIMD main loops and
//!   scalar tails (widths 128 and 130).

use super::*;
use crate::sinker::MixedSinker;

// ---- helpers ----------------------------------------------------------------

/// Build a solid-colour GbrpN frame with all planes set to `val`.
fn solid_gbrp_frame<'a, const BITS: u32>(
  g: &'a [u16],
  b: &'a [u16],
  r: &'a [u16],
  w: u32,
  h: u32,
) -> crate::frame::GbrpHighBitFrame<'a, BITS> {
  crate::frame::GbrpHighBitFrame::try_new(g, b, r, w, h, w, w, w).unwrap()
}

/// Build a solid-colour GbrapN frame.
fn solid_gbrap_frame<'a, const BITS: u32>(
  g: &'a [u16],
  b: &'a [u16],
  r: &'a [u16],
  a: &'a [u16],
  w: u32,
  h: u32,
) -> crate::frame::GbrapHighBitFrame<'a, BITS> {
  crate::frame::GbrapHighBitFrame::try_new(g, b, r, a, w, h, w, w, w, w).unwrap()
}

// ---- Bit-depth saturation: u16 output stays at max -------------------------

macro_rules! test_gbrp_saturation_u16 {
  ($name:ident, $marker:ident, $walker:ident, $bits:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = 4usize;
      let h = 1usize;
      let max = ((1u32 << $bits) - 1) as u16;
      let g = std::vec![max; w * h];
      let b = std::vec![max; w * h];
      let r = std::vec![max; w * h];
      let src = solid_gbrp_frame::<$bits>(&g, &b, &r, w as u32, h as u32);
      let mut out = std::vec![0u16; w * h * 3];
      let mut sink = MixedSinker::<crate::source::$marker>::new(w, h)
        .with_rgb_u16(&mut out)
        .unwrap();
      crate::source::$walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
      for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, max, "u16 output[{i}] must be {max} (max for BITS={}) but got {v}", $bits);
      }
    }
  };
}

test_gbrp_saturation_u16!(gbrp9_all_max_u16_saturates, Gbrp9, gbrp9_to, 9);
test_gbrp_saturation_u16!(gbrp10_all_max_u16_saturates, Gbrp10, gbrp10_to, 10);
test_gbrp_saturation_u16!(gbrp12_all_max_u16_saturates, Gbrp12, gbrp12_to, 12);
test_gbrp_saturation_u16!(gbrp14_all_max_u16_saturates, Gbrp14, gbrp14_to, 14);
test_gbrp_saturation_u16!(gbrp16_all_max_u16_saturates, Gbrp16, gbrp16_to, 16);

// ---- Bit-depth saturation: u8 output downshifted to 0xFF -------------------

macro_rules! test_gbrp_saturation_u8 {
  ($name:ident, $marker:ident, $walker:ident, $bits:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = 4usize;
      let h = 1usize;
      let max = ((1u32 << $bits) - 1) as u16;
      let g = std::vec![max; w * h];
      let b = std::vec![max; w * h];
      let r = std::vec![max; w * h];
      let src = solid_gbrp_frame::<$bits>(&g, &b, &r, w as u32, h as u32);
      let mut out = std::vec![0u8; w * h * 3];
      let mut sink = MixedSinker::<crate::source::$marker>::new(w, h)
        .with_rgb(&mut out)
        .unwrap();
      crate::source::$walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
      for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0xFF, "u8 output[{i}] must be 0xFF for max BITS={} input but got {v}", $bits);
      }
    }
  };
}

test_gbrp_saturation_u8!(gbrp9_all_max_u8_is_0xff, Gbrp9, gbrp9_to, 9);
test_gbrp_saturation_u8!(gbrp10_all_max_u8_is_0xff, Gbrp10, gbrp10_to, 10);
test_gbrp_saturation_u8!(gbrp12_all_max_u8_is_0xff, Gbrp12, gbrp12_to, 12);
test_gbrp_saturation_u8!(gbrp14_all_max_u8_is_0xff, Gbrp14, gbrp14_to, 14);
test_gbrp_saturation_u8!(gbrp16_all_max_u8_is_0xff, Gbrp16, gbrp16_to, 16);

// ---- Channel reorder: G=100, B=50, R=200 → (R=200, G=100, B=50) -----------

macro_rules! test_gbrp_channel_reorder {
  ($name:ident, $marker:ident, $walker:ident, $bits:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = 16usize;
      let h = 4usize;
      // Upshift 8-bit seed values to native BITS depth.
      let shift = $bits - 8;
      let g = std::vec![100u16 << shift; w * h];
      let b = std::vec![50u16 << shift; w * h];
      let r = std::vec![200u16 << shift; w * h];
      let src = solid_gbrp_frame::<$bits>(&g, &b, &r, w as u32, h as u32);
      let mut out_u8 = std::vec![0u8; w * h * 3];
      let mut sink = MixedSinker::<crate::source::$marker>::new(w, h)
        .with_rgb(&mut out_u8)
        .unwrap();
      crate::source::$walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
      for i in 0..w * h {
        assert_eq!(out_u8[i * 3],     200, "R[{i}] mismatch");
        assert_eq!(out_u8[i * 3 + 1], 100, "G[{i}] mismatch");
        assert_eq!(out_u8[i * 3 + 2],  50, "B[{i}] mismatch");
      }
    }
  };
}

test_gbrp_channel_reorder!(gbrp9_channel_reorder, Gbrp9, gbrp9_to, 9);
test_gbrp_channel_reorder!(gbrp10_channel_reorder, Gbrp10, gbrp10_to, 10);
test_gbrp_channel_reorder!(gbrp12_channel_reorder, Gbrp12, gbrp12_to, 12);
test_gbrp_channel_reorder!(gbrp14_channel_reorder, Gbrp14, gbrp14_to, 14);
test_gbrp_channel_reorder!(gbrp16_channel_reorder, Gbrp16, gbrp16_to, 16);

// ---- Strategy A+: Gbrap combo RGB+RGBA matches standalone RGBA --------------

macro_rules! test_gbrap_strategy_a_plus {
  ($name:ident, $marker:ident, $walker:ident, $bits:literal) => {
    test_gbrap_strategy_a_plus!($name, $marker, $walker, $bits, 32);
  };
  ($name:ident, $marker:ident, $walker:ident, $bits:literal, $w:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = $w as usize;
      let h = 8usize;
      let n = w * h;
      let mut g = std::vec![0u16; n];
      let mut b = std::vec![0u16; n];
      let mut r = std::vec![0u16; n];
      let mut a = std::vec![0u16; n];
      pseudo_random_u16_low_n_bits(&mut g, 0x11_u32.wrapping_add($bits), $bits);
      pseudo_random_u16_low_n_bits(&mut b, 0x22_u32.wrapping_add($bits), $bits);
      pseudo_random_u16_low_n_bits(&mut r, 0x33_u32.wrapping_add($bits), $bits);
      pseudo_random_u16_low_n_bits(&mut a, 0x44_u32.wrapping_add($bits), $bits);

      // Reference: standalone with_rgba (direct 4-channel kernel).
      let src_ref = solid_gbrap_frame::<$bits>(&g, &b, &r, &a, w as u32, h as u32);
      let mut rgba_ref = std::vec![0u8; n * 4];
      let mut sink_ref = MixedSinker::<crate::source::$marker>::new(w, h)
        .with_rgba(&mut rgba_ref)
        .unwrap();
      crate::source::$walker(&src_ref, false, KernelMatrix::Bt709, &mut sink_ref).unwrap();

      // Combo: with_rgb + with_rgba (Strategy A+).
      let src_combo = solid_gbrap_frame::<$bits>(&g, &b, &r, &a, w as u32, h as u32);
      let mut rgb_combo = std::vec![0u8; n * 3];
      let mut rgba_combo = std::vec![0u8; n * 4];
      let mut sink_combo = MixedSinker::<crate::source::$marker>::new(w, h)
        .with_rgb(&mut rgb_combo)
        .unwrap()
        .with_rgba(&mut rgba_combo)
        .unwrap();
      crate::source::$walker(&src_combo, false, KernelMatrix::Bt709, &mut sink_combo).unwrap();

      // RGBA bytes must be identical between standalone and combo paths.
      assert_eq!(
        rgba_ref, rgba_combo,
        "Strategy A+ RGBA mismatch for BITS={} w={}", $bits, $w,
      );
    }
  };
}

test_gbrap_strategy_a_plus!(
  gbrap10_strategy_a_plus_matches_standalone,
  Gbrap10,
  gbrap10_to,
  10
);
test_gbrap_strategy_a_plus!(
  gbrap12_strategy_a_plus_matches_standalone,
  Gbrap12,
  gbrap12_to,
  12
);
test_gbrap_strategy_a_plus!(
  gbrap14_strategy_a_plus_matches_standalone,
  Gbrap14,
  gbrap14_to,
  14
);
test_gbrap_strategy_a_plus!(
  gbrap16_strategy_a_plus_matches_standalone,
  Gbrap16,
  gbrap16_to,
  16
);

// ---- Strategy A+: Gbrap combo RGB_u16+RGBA_u16 matches standalone RGBA_u16 -
//
// Mirrors the u8 Strategy A+ test above, but covers the native-depth combo
// path (`with_rgb_u16` + `with_rgba_u16`) that routes through
// `copy_alpha_plane_u16` rather than `copy_alpha_plane_u16_to_u8`. Without
// this, a regression in the `BE != cfg!(target_endian)` dispatcher routing
// or in the scalar α-extract helper would not be caught for the native-depth
// path.
//
// Source planes are filled with full-range u16 values (`bits=16` argument
// to `pseudo_random_u16_low_n_bits`) so the upper bits beyond BITS are
// "dirty" — both paths must mask via `(1 << BITS) - 1`, so any drift between
// them surfaces here.
macro_rules! test_gbrap_strategy_a_plus_u16 {
  ($name:ident, $marker:ident, $walker:ident, $bits:literal) => {
    test_gbrap_strategy_a_plus_u16!($name, $marker, $walker, $bits, 32);
  };
  ($name:ident, $marker:ident, $walker:ident, $bits:literal, $w:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = $w as usize;
      let h = 8usize;
      let n = w * h;
      let mut g = std::vec![0u16; n];
      let mut b = std::vec![0u16; n];
      let mut r = std::vec![0u16; n];
      let mut a = std::vec![0u16; n];
      // Use full-range u16 (bits=16) so upper bits beyond BITS are dirty,
      // exercising the mask in both the direct kernel and α-extract paths.
      pseudo_random_u16_low_n_bits(&mut g, 0x55_u32.wrapping_add($bits), 16);
      pseudo_random_u16_low_n_bits(&mut b, 0x66_u32.wrapping_add($bits), 16);
      pseudo_random_u16_low_n_bits(&mut r, 0x77_u32.wrapping_add($bits), 16);
      pseudo_random_u16_low_n_bits(&mut a, 0x88_u32.wrapping_add($bits), 16);

      // Reference: standalone with_rgba_u16 (direct 4-channel kernel).
      let src_ref = solid_gbrap_frame::<$bits>(&g, &b, &r, &a, w as u32, h as u32);
      let mut rgba_u16_ref = std::vec![0u16; n * 4];
      let mut sink_ref = MixedSinker::<crate::source::$marker>::new(w, h)
        .with_rgba_u16(&mut rgba_u16_ref)
        .unwrap();
      crate::source::$walker(&src_ref, false, KernelMatrix::Bt709, &mut sink_ref).unwrap();

      // Combo: with_rgb_u16 + with_rgba_u16 (Strategy A+ native-depth).
      let src_combo = solid_gbrap_frame::<$bits>(&g, &b, &r, &a, w as u32, h as u32);
      let mut rgb_u16_combo = std::vec![0u16; n * 3];
      let mut rgba_u16_combo = std::vec![0u16; n * 4];
      let mut sink_combo = MixedSinker::<crate::source::$marker>::new(w, h)
        .with_rgb_u16(&mut rgb_u16_combo)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16_combo)
        .unwrap();
      crate::source::$walker(&src_combo, false, KernelMatrix::Bt709, &mut sink_combo).unwrap();

      // RGBA u16 elements must be byte-exact between standalone and combo paths.
      assert_eq!(
        rgba_u16_ref, rgba_u16_combo,
        "Strategy A+ native-depth RGBA u16 mismatch for BITS={} w={}", $bits, $w,
      );
    }
  };
}

test_gbrap_strategy_a_plus_u16!(
  gbrap10_strategy_a_plus_u16_matches_standalone,
  Gbrap10,
  gbrap10_to,
  10
);
test_gbrap_strategy_a_plus_u16!(
  gbrap12_strategy_a_plus_u16_matches_standalone,
  Gbrap12,
  gbrap12_to,
  12
);
test_gbrap_strategy_a_plus_u16!(
  gbrap14_strategy_a_plus_u16_matches_standalone,
  Gbrap14,
  gbrap14_to,
  14
);
test_gbrap_strategy_a_plus_u16!(
  gbrap16_strategy_a_plus_u16_matches_standalone,
  Gbrap16,
  gbrap16_to,
  16
);

// Strategy A+ at non-multiple width (31) exercises the SIMD scalar tail.
//
// The SIMD α-extract backends (`copy_alpha_plane_u16{_to_u8}`) hardcode
// `scalar::<BITS, false>` for the tail (e.g. NEON block size 8 + width 31
// leaves 7 px in the tail; AVX2/AVX-512 likewise). The prior dispatcher
// routing (`need_swap = BE != cfg!(target_endian = "big")`) admitted SIMD
// on BE-host/BE-data: the vector body's host-native loads are correct
// there, but the LE-only scalar tail then byte-swaps already-native u16
// samples, silently corrupting α at non-multiple widths. SIMD is now
// gated to the LE-host/LE-data quadrant only; these tests at width 31
// exercise the SIMD tail path on supported (LE) hosts and pin the parity
// guarantee for that quadrant. (The LE/BE, BE/LE, BE/BE quadrants are
// exercised at the scalar level by the `target_endian`-aware scalar
// helper; the dispatcher routes them to scalar always.)

test_gbrap_strategy_a_plus_u16!(
  gbrap10_strategy_a_plus_u16_matches_standalone_w31,
  Gbrap10,
  gbrap10_to,
  10,
  31
);
test_gbrap_strategy_a_plus_u16!(
  gbrap12_strategy_a_plus_u16_matches_standalone_w31,
  Gbrap12,
  gbrap12_to,
  12,
  31
);
test_gbrap_strategy_a_plus_u16!(
  gbrap14_strategy_a_plus_u16_matches_standalone_w31,
  Gbrap14,
  gbrap14_to,
  14,
  31
);
test_gbrap_strategy_a_plus_u16!(
  gbrap16_strategy_a_plus_u16_matches_standalone_w31,
  Gbrap16,
  gbrap16_to,
  16,
  31
);

// u8-path Strategy A+ at width 31 — exercises the SIMD tail of
// `copy_alpha_plane_u16_to_u8` (depth-conv `>> (BITS - 8)`). One BITS value
// is sufficient to cover the same dispatcher path as the u16 set above;
// Gbrap10 chosen for parity with the existing u8 Strategy A+ coverage.
test_gbrap_strategy_a_plus!(
  gbrap10_strategy_a_plus_matches_standalone_w31,
  Gbrap10,
  gbrap10_to,
  10,
  31
);

// ---- Gbrap alpha downshift correctness -------------------------------------

macro_rules! test_gbrap_alpha_downshift {
  ($name:ident, $marker:ident, $walker:ident, $bits:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = 16usize;
      let h = 4usize;
      let n = w * h;
      let max = ((1u32 << $bits) - 1) as u16;
      let mask32 = (1u32 << $bits) - 1;
      let g = std::vec![0u16; n];
      let b = std::vec![0u16; n];
      let r = std::vec![max; n];
      // Varied alpha values, bounded to BITS range.
      let a: std::vec::Vec<u16> = (0..n)
        .map(|i| ((i as u32 * 7 + 13) & mask32) as u16)
        .collect();

      let src = solid_gbrap_frame::<$bits>(&g, &b, &r, &a, w as u32, h as u32);
      let mut rgba = std::vec![0u8; n * 4];
      let mut sink = MixedSinker::<crate::source::$marker>::new(w, h)
        .with_rgba(&mut rgba)
        .unwrap();
      crate::source::$walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

      let shift = $bits - 8;
      for i in 0..n {
        let expected_alpha = (a[i] >> shift) as u8;
        assert_eq!(
          rgba[i * 4 + 3], expected_alpha,
          "alpha at px {i}: expected {} (source {} >> {}), got {}",
          expected_alpha, a[i], shift, rgba[i * 4 + 3],
        );
      }
    }
  };
}

test_gbrap_alpha_downshift!(gbrap10_alpha_downshift_correct, Gbrap10, gbrap10_to, 10);
test_gbrap_alpha_downshift!(gbrap12_alpha_downshift_correct, Gbrap12, gbrap12_to, 12);
test_gbrap_alpha_downshift!(gbrap14_alpha_downshift_correct, Gbrap14, gbrap14_to, 14);
test_gbrap_alpha_downshift!(gbrap16_alpha_downshift_correct, Gbrap16, gbrap16_to, 16);

// ---- SIMD vs scalar equivalence (width 128 + tail width 130) ---------------

macro_rules! test_gbrp_simd_matches_scalar {
  ($name:ident, $marker:ident, $walker:ident, $bits:literal, $w:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = $w;
      let h = 8usize;
      let n = w * h;
      let mut g = std::vec![0u16; n];
      let mut b = std::vec![0u16; n];
      let mut r = std::vec![0u16; n];
      pseudo_random_u16_low_n_bits(&mut g, 0xA0, $bits);
      pseudo_random_u16_low_n_bits(&mut b, 0xB0, $bits);
      pseudo_random_u16_low_n_bits(&mut r, 0xC0, $bits);

      let src_simd = solid_gbrp_frame::<$bits>(&g, &b, &r, w as u32, h as u32);
      let src_scal = solid_gbrp_frame::<$bits>(&g, &b, &r, w as u32, h as u32);

      let mut rgb_simd = std::vec![0u8; n * 3];
      let mut rgb_scal = std::vec![0u8; n * 3];
      let mut rgba_simd = std::vec![0u8; n * 4];
      let mut rgba_scal = std::vec![0u8; n * 4];

      {
        let mut sink = MixedSinker::<crate::source::$marker>::new(w, h)
          .with_rgb(&mut rgb_simd).unwrap()
          .with_rgba(&mut rgba_simd).unwrap();
        crate::source::$walker(&src_simd, true, KernelMatrix::Bt709, &mut sink).unwrap();
      }
      {
        let mut sink = MixedSinker::<crate::source::$marker>::new(w, h)
          .with_rgb(&mut rgb_scal).unwrap()
          .with_rgba(&mut rgba_scal).unwrap();
        crate::source::$walker(&src_scal, false, KernelMatrix::Bt709, &mut sink).unwrap();
      }

      assert_eq!(rgb_simd, rgb_scal, "rgb SIMD≠scalar for BITS={} w={}", $bits, $w);
      assert_eq!(rgba_simd, rgba_scal, "rgba SIMD≠scalar for BITS={} w={}", $bits, $w);
    }
  };
}

test_gbrp_simd_matches_scalar!(gbrp9_w128_simd_matches_scalar, Gbrp9, gbrp9_to, 9, 128);
test_gbrp_simd_matches_scalar!(gbrp9_w130_simd_matches_scalar, Gbrp9, gbrp9_to, 9, 130);
test_gbrp_simd_matches_scalar!(gbrp10_w128_simd_matches_scalar, Gbrp10, gbrp10_to, 10, 128);
test_gbrp_simd_matches_scalar!(gbrp10_w130_simd_matches_scalar, Gbrp10, gbrp10_to, 10, 130);
test_gbrp_simd_matches_scalar!(gbrp12_w128_simd_matches_scalar, Gbrp12, gbrp12_to, 12, 128);
test_gbrp_simd_matches_scalar!(gbrp12_w130_simd_matches_scalar, Gbrp12, gbrp12_to, 12, 130);
test_gbrp_simd_matches_scalar!(gbrp14_w128_simd_matches_scalar, Gbrp14, gbrp14_to, 14, 128);
test_gbrp_simd_matches_scalar!(gbrp14_w130_simd_matches_scalar, Gbrp14, gbrp14_to, 14, 130);
test_gbrp_simd_matches_scalar!(gbrp16_w128_simd_matches_scalar, Gbrp16, gbrp16_to, 16, 128);
test_gbrp_simd_matches_scalar!(gbrp16_w130_simd_matches_scalar, Gbrp16, gbrp16_to, 16, 130);

macro_rules! test_gbrap_simd_matches_scalar {
  ($name:ident, $marker:ident, $walker:ident, $bits:literal, $w:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = $w;
      let h = 8usize;
      let n = w * h;
      let mut g = std::vec![0u16; n];
      let mut b = std::vec![0u16; n];
      let mut r = std::vec![0u16; n];
      let mut a = std::vec![0u16; n];
      pseudo_random_u16_low_n_bits(&mut g, 0xA1, $bits);
      pseudo_random_u16_low_n_bits(&mut b, 0xB1, $bits);
      pseudo_random_u16_low_n_bits(&mut r, 0xC1, $bits);
      pseudo_random_u16_low_n_bits(&mut a, 0xD1, $bits);

      let src_simd = solid_gbrap_frame::<$bits>(&g, &b, &r, &a, w as u32, h as u32);
      let src_scal = solid_gbrap_frame::<$bits>(&g, &b, &r, &a, w as u32, h as u32);

      let mut rgba_simd = std::vec![0u8; n * 4];
      let mut rgba_scal = std::vec![0u8; n * 4];

      {
        let mut sink = MixedSinker::<crate::source::$marker>::new(w, h)
          .with_rgba(&mut rgba_simd).unwrap();
        crate::source::$walker(&src_simd, true, KernelMatrix::Bt709, &mut sink).unwrap();
      }
      {
        let mut sink = MixedSinker::<crate::source::$marker>::new(w, h)
          .with_rgba(&mut rgba_scal).unwrap();
        crate::source::$walker(&src_scal, false, KernelMatrix::Bt709, &mut sink).unwrap();
      }

      assert_eq!(rgba_simd, rgba_scal, "rgba SIMD≠scalar for BITS={} w={}", $bits, $w);
    }
  };
}

test_gbrap_simd_matches_scalar!(
  gbrap10_w128_simd_matches_scalar,
  Gbrap10,
  gbrap10_to,
  10,
  128
);
test_gbrap_simd_matches_scalar!(
  gbrap10_w130_simd_matches_scalar,
  Gbrap10,
  gbrap10_to,
  10,
  130
);
test_gbrap_simd_matches_scalar!(
  gbrap12_w128_simd_matches_scalar,
  Gbrap12,
  gbrap12_to,
  12,
  128
);
test_gbrap_simd_matches_scalar!(
  gbrap12_w130_simd_matches_scalar,
  Gbrap12,
  gbrap12_to,
  12,
  130
);
test_gbrap_simd_matches_scalar!(
  gbrap14_w128_simd_matches_scalar,
  Gbrap14,
  gbrap14_to,
  14,
  128
);
test_gbrap_simd_matches_scalar!(
  gbrap14_w130_simd_matches_scalar,
  Gbrap14,
  gbrap14_to,
  14,
  130
);
test_gbrap_simd_matches_scalar!(
  gbrap16_w128_simd_matches_scalar,
  Gbrap16,
  gbrap16_to,
  16,
  128
);
test_gbrap_simd_matches_scalar!(
  gbrap16_w130_simd_matches_scalar,
  Gbrap16,
  gbrap16_to,
  16,
  130
);

// Frame BE flag — LE+BE round-trip parity tests.
//
// Per-format pattern: build a host-native u16 plane, encode once as LE bytes
// and once as BE bytes (via `to_le_bytes` / `to_be_bytes`), drive each
// through its `MixedSinker<MarkerN<BE>>` monomorphization, and assert the
// outputs are byte-identical. The kernel byte-swaps under the hood, so the
// same logical samples must yield the same RGBA output regardless of plane
// byte order. This catches missing `<BE>` propagation in sinker call sites
// or in the `gbr_to_*_high_bit_row::<BITS, BE>` dispatch.

/// Re-encode a host-native u16 slice as **BE-encoded** byte storage. Used to
/// build `*BeFrame` planes whose bytes are big-endian; the kernel swaps them
/// back to host-native via `from_be`.
fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as **LE-encoded** byte storage.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

macro_rules! gbrp_le_be_roundtrip {
  ($name:ident, $marker:ident, $le_alias:ident, $be_alias:ident, $walker:ident, $walker_endian:ident, $bits:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = 16usize;
      let h = 4usize;
      let mask: u16 = ((1u32 << $bits) - 1) as u16;
      // Mix of patterns within the bit-width range.
      let intended: std::vec::Vec<u16> = (0..w * h)
        .map(|i| {
          let raw: u16 = match i % 4 {
            0 => 0x1234,
            1 => 0xABCD,
            2 => 0x00FF,
            _ => 0x7FFF,
          };
          raw & mask
        })
        .collect();
      let g_le = as_le_u16(&intended);
      let b_le = as_le_u16(&intended);
      let r_le = as_le_u16(&intended);
      let g_be = as_be_u16(&intended);
      let b_be = as_be_u16(&intended);
      let r_be = as_be_u16(&intended);

      // Cover both scalar and SIMD dispatch — the SIMD path catches missing
      // `<BE>` propagation in the SIMD-aware row kernels that scalar misses.
      for use_simd in [false, true] {
        let stride = w as u32;
        let frame_le = crate::frame::$le_alias::try_new(
          &g_le, &b_le, &r_le, w as u32, h as u32, stride, stride, stride,
        )
        .unwrap();
        let mut out_le = std::vec![0u8; w * h * 4];
        let mut sink_le = MixedSinker::<$marker>::new(w, h)
          .with_simd(use_simd)
          .with_rgba(&mut out_le)
          .unwrap();
        $walker(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

        let frame_be = crate::frame::$be_alias::try_new(
          &g_be, &b_be, &r_be, w as u32, h as u32, stride, stride, stride,
        )
        .unwrap();
        let mut out_be = std::vec![0u8; w * h * 4];
        let mut sink_be = MixedSinker::<$marker<true>>::new(w, h)
          .with_simd(use_simd)
          .with_rgba(&mut out_be)
          .unwrap();
        // BE-frame call must use the `_endian` helper — the LE-only wrapper
        // is signature-bound to `Frame<'_, false>`.
        $walker_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

        assert_eq!(
          out_le,
          out_be,
          concat!(
            stringify!($marker),
            " LE/BE outputs diverge — `<const BE>` propagation broken (use_simd={})",
          ),
          use_simd,
        );
      }
    }
  };
}

gbrp_le_be_roundtrip!(
  gbrp9_le_be_roundtrip,
  Gbrp9,
  Gbrp9LeFrame,
  Gbrp9BeFrame,
  gbrp9_to,
  gbrp9_to_endian,
  9
);
gbrp_le_be_roundtrip!(
  gbrp10_le_be_roundtrip,
  Gbrp10,
  Gbrp10LeFrame,
  Gbrp10BeFrame,
  gbrp10_to,
  gbrp10_to_endian,
  10
);
gbrp_le_be_roundtrip!(
  gbrp12_le_be_roundtrip,
  Gbrp12,
  Gbrp12LeFrame,
  Gbrp12BeFrame,
  gbrp12_to,
  gbrp12_to_endian,
  12
);
gbrp_le_be_roundtrip!(
  gbrp14_le_be_roundtrip,
  Gbrp14,
  Gbrp14LeFrame,
  Gbrp14BeFrame,
  gbrp14_to,
  gbrp14_to_endian,
  14
);
gbrp_le_be_roundtrip!(
  gbrp16_le_be_roundtrip,
  Gbrp16,
  Gbrp16LeFrame,
  Gbrp16BeFrame,
  gbrp16_to,
  gbrp16_to_endian,
  16
);

macro_rules! gbrap_le_be_roundtrip {
  ($name:ident, $marker:ident, $le_alias:ident, $be_alias:ident, $walker:ident, $walker_endian:ident, $bits:literal) => {
    #[test]
    #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
    fn $name() {
      let w = 16usize;
      let h = 4usize;
      let mask: u16 = ((1u32 << $bits) - 1) as u16;
      let intended: std::vec::Vec<u16> = (0..w * h)
        .map(|i| {
          let raw: u16 = match i % 5 {
            0 => 0x1234,
            1 => 0xABCD,
            2 => 0x00FF,
            3 => 0x7FFF,
            _ => 0x5555,
          };
          raw & mask
        })
        .collect();
      let g_le = as_le_u16(&intended);
      let b_le = as_le_u16(&intended);
      let r_le = as_le_u16(&intended);
      let a_le = as_le_u16(&intended);
      let g_be = as_be_u16(&intended);
      let b_be = as_be_u16(&intended);
      let r_be = as_be_u16(&intended);
      let a_be = as_be_u16(&intended);

      // Cover both scalar and SIMD dispatch.
      for use_simd in [false, true] {
        let stride = w as u32;
        let frame_le = crate::frame::$le_alias::try_new(
          &g_le, &b_le, &r_le, &a_le, w as u32, h as u32, stride, stride, stride, stride,
        )
        .unwrap();
        // Exercise both u8 and u16 RGBA paths to cover gbra_to_rgba_*_row plus the
        // alpha_extract::copy_alpha_plane_u16 / u16_to_u8 propagation.
        let mut out_le_rgba = std::vec![0u8; w * h * 4];
        let mut out_le_rgba_u16 = std::vec![0u16; w * h * 4];
        let mut sink_le = MixedSinker::<$marker>::new(w, h)
          .with_simd(use_simd)
          .with_rgba(&mut out_le_rgba)
          .unwrap()
          .with_rgba_u16(&mut out_le_rgba_u16)
          .unwrap();
        $walker(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

        let frame_be = crate::frame::$be_alias::try_new(
          &g_be, &b_be, &r_be, &a_be, w as u32, h as u32, stride, stride, stride, stride,
        )
        .unwrap();
        let mut out_be_rgba = std::vec![0u8; w * h * 4];
        let mut out_be_rgba_u16 = std::vec![0u16; w * h * 4];
        let mut sink_be = MixedSinker::<$marker<true>>::new(w, h)
          .with_simd(use_simd)
          .with_rgba(&mut out_be_rgba)
          .unwrap()
          .with_rgba_u16(&mut out_be_rgba_u16)
          .unwrap();
        // BE-frame call must use the `_endian` helper.
        $walker_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

        assert_eq!(
          out_le_rgba,
          out_be_rgba,
          concat!(
            stringify!($marker),
            " RGBA u8 LE/BE outputs diverge (use_simd={})",
          ),
          use_simd,
        );
        assert_eq!(
          out_le_rgba_u16,
          out_be_rgba_u16,
          concat!(
            stringify!($marker),
            " RGBA u16 LE/BE outputs diverge (use_simd={})",
          ),
          use_simd,
        );
      }
    }
  };
}

gbrap_le_be_roundtrip!(
  gbrap10_le_be_roundtrip,
  Gbrap10,
  Gbrap10LeFrame,
  Gbrap10BeFrame,
  gbrap10_to,
  gbrap10_to_endian,
  10
);
gbrap_le_be_roundtrip!(
  gbrap12_le_be_roundtrip,
  Gbrap12,
  Gbrap12LeFrame,
  Gbrap12BeFrame,
  gbrap12_to,
  gbrap12_to_endian,
  12
);
gbrap_le_be_roundtrip!(
  gbrap14_le_be_roundtrip,
  Gbrap14,
  Gbrap14LeFrame,
  Gbrap14BeFrame,
  gbrap14_to,
  gbrap14_to_endian,
  14
);
gbrap_le_be_roundtrip!(
  gbrap16_le_be_roundtrip,
  Gbrap16,
  Gbrap16LeFrame,
  Gbrap16BeFrame,
  gbrap16_to,
  gbrap16_to_endian,
  16
);
