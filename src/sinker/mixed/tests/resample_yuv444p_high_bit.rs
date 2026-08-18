//! Fused-downscale coverage for the high-bit **planar** 4:4:4 YUV family —
//! `Yuv444p10` / `Yuv444p12` / `Yuv444p14` / `Yuv444p16` (LE + BE wire).
//! Low-packed `u16` Y / U / V planes, full-width chroma (no upsampling).
//!
//! These route through [`packed_yuv444_triple_resample`] (the same tail the
//! packed high-bit 4:4:4 family uses), with **three** independent
//! native-precision binnings because the direct path's per-output kernels
//! round and scale *independently* and luma is native Y:
//! - **u8 colour (rgb / rgba / hsv)** bins a converted source-width u8 RGB
//!   row (`yuv444pN_to_rgb_row_endian`).
//! - **u16 colour (rgb_u16 / rgba_u16)** bins an *independent* converted
//!   source-width native u16 RGB row (`yuv444pN_to_rgb_u16_row_endian`) —
//!   never a narrowing of the u8 bin.
//! - **luma** bins the de-interleaved native Y; `luma = binned_Y >>
//!   (BITS - 8)`. (Yuv444p exposes no `luma_u16` output.)
//!
//! Each output is byte-identical to the area-bin of the **direct**
//! full-resolution conversion (convert-then-bin), so the colour oracles
//! drive a direct identity sink at source resolution and 2x2-block-mean its
//! output, and the luma oracle area-bins the native Y plane then narrows.
//! The uniform-gray + saturated-chroma counterexamples pin the real parity
//! bugs: deriving u8 colour by narrowing the u16 bin would change a
//! uniform-gray downscale's colour, and deriving luma from RGB would clamp
//! away from the Y plane under saturated chroma.

use crate::{
  KernelMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, FilteredResampler, ResampleError, Triangle},
  sinker::{MixedSinker, MixedSinkerError},
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Re-encode a host-native u16 slice as host-independent LE-wire byte
/// storage (the `*LeFrame` plane contract), recovered via `from_le`.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as host-independent BE-wire byte
/// storage (the `*BeFrame` plane contract), recovered via `from_be`.
fn as_be_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid 3-channel
/// u8 RGB plane.
fn block_mean_2x2_rgb_u8(rgb: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 3 + c] = ((s + 2) / 4) as u8;
      }
    }
  }
  out
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid u16 plane
/// (host-native logical values).
fn block_mean_2x2_u16(plane: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u16;
    }
  }
  out
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid 3-channel
/// u16 RGB plane.
fn block_mean_2x2_rgb_u16(rgb: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 3 + c] = ((s + 2) / 4) as u16;
      }
    }
  }
  out
}

// A per-depth macro keeps the four near-identical suites in lockstep while
// naming each test after its bit depth. `$frame_le` / `$frame_be` are the
// LE / BE frame types, `$marker` the source marker, `$row` the row type,
// `$walker` the LE walker, `$walker_be` the `_endian` walker, `$bits` the
// active depth.
macro_rules! yuv444p_high_bit_resample_suite {
  (
    $mod:ident, $frame_le:ident, $frame_be:ident, $marker:ident, $row:ident,
    $walker:ident, $walker_be:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $row, $walker, $walker_be};

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = (1u16 << ($bits - 1));

      /// #334: a geometry-only `*Frame::try_new` accepts an out-of-range Y
      /// sample — here a uniform plane carrying the first bit ABOVE the native
      /// depth (`MASK + 1`; a no-op at 16-bit, where no such bit exists). The
      /// luma resample feeds must publish the DEPTH-MASKED Y (`& MASK`), like
      /// the colour kernels, across all three tiers (native fast / row-stage /
      /// filter) and both wire endiannesses — NOT bin the dirty Y and clamp the
      /// overrange bin to the native max (the pre-fix path narrows that to
      /// `0xFF`). A constant survives every resampler, so the area-mean /
      /// filtered output of the uniform plane is that same in-range value.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_masks_dirty_high_bit_y() {
        let clean: u16 = (MASK / 3) & MASK;
        let dirty: u16 = clean | MASK.wrapping_add(1);
        let want = std::vec![(clean >> ($bits - 8)) as u8; OUT * OUT];
        // Chroma is unread by a luma-only sink; reuse the neutral-gray planes
        // for their correct per-format dimensions and overwrite Y.
        let (_, u, v) = uniform_gray(clean);
        let yp = std::vec![dirty; SRC * SRC];
        let (y_le, u_le, v_le) = (as_le_u16(&yp), as_le_u16(&u), as_le_u16(&v));
        let (y_be, u_be, v_be) = (as_be_u16(&yp), as_be_u16(&u), as_be_u16(&v));

        for native in [true, false] {
          let mut luma = std::vec![0u8; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(native)
            .with_luma(&mut luma)
            .unwrap();
            $walker(&frame(&y_le, &u_le, &v_le), FR, M, &mut sink).unwrap();
          }
          assert_eq!(luma, want, "LE area luma not depth-masked (native={native})");

          let mut be_luma = std::vec![0u8; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(native)
            .with_luma(&mut be_luma)
            .unwrap();
            $walker_be::<_, true>(&frame_be(&y_be, &u_be, &v_be), FR, M, &mut sink).unwrap();
          }
          assert_eq!(be_luma, want, "BE area luma not depth-masked (native={native})");
        }

        let mut luma = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&y_le, &u_le, &v_le), FR, M, &mut sink).unwrap();
        }
        assert_eq!(luma, want, "LE filter luma not depth-masked");

        let mut be_luma = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_luma(&mut be_luma)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&y_be, &u_be, &v_be), FR, M, &mut sink).unwrap();
        }
        assert_eq!(be_luma, want, "BE filter luma not depth-masked");
      }

      /// Per-pixel `(Y, U, V)` ramp into full-width `SRC`-grid planes
      /// (4:4:4, low-packed native codes so every kernel sees real math).
      fn ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; SRC * SRC];
        let mut v = vec![0u16; SRC * SRC];
        for i in 0..SRC * SRC {
          y[i] = ((40u32 + i as u32 * 37) & MASK as u32) as u16;
          u[i] = ((300u32 + i as u32 * 53) & MASK as u32) as u16;
          v[i] = (MASK as u32).wrapping_sub(i as u32 * 41) as u16 & MASK;
        }
        (y, u, v)
      }

      /// Uniform-gray planes: constant Y, neutral chroma (U = V = mid).
      /// Binning a uniform frame is identity, so every resampled colour
      /// output must equal the direct full-res conversion.
      fn uniform_gray(y: u16) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        (
          vec![y & MASK; SRC * SRC],
          vec![MID & MASK; SRC * SRC],
          vec![MID & MASK; SRC * SRC],
        )
      }

      /// Saturated-chroma planes: constant Y, extreme U/V — the case where
      /// RGB-derived luma would clamp away from the Y plane.
      fn saturated(y: u16) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        (
          vec![y & MASK; SRC * SRC],
          vec![MASK; SRC * SRC],
          vec![0u16; SRC * SRC],
        )
      }

      fn frame<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16]) -> $frame_le<'a> {
        $frame_le::try_new(
          y, u, v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
        )
        .unwrap()
      }
      fn frame_be<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16]) -> $frame_be<'a> {
        $frame_be::try_new(
          y, u, v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
        )
        .unwrap()
      }

      /// Direct full-resolution u8 RGB of the planar frame.
      fn direct_rgb_u8(y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
        let mut rgb = vec![0u8; SRC * SRC * 3];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgb(&mut rgb)
          .unwrap();
        $walker(&frame(y, u, v), FR, M, &mut sink).unwrap();
        rgb
      }
      /// Direct full-resolution native u16 RGB of the planar frame.
      fn direct_rgb_u16(y: &[u16], u: &[u16], v: &[u16]) -> Vec<u16> {
        let mut rgb = vec![0u16; SRC * SRC * 3];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgb_u16(&mut rgb)
          .unwrap();
        $walker(&frame(y, u, v), FR, M, &mut sink).unwrap();
        rgb
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rgb_u8_matches_area_bin_of_direct() {
        let (y, u, v) = ramp();
        let mut rgb = vec![0u8; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb(&mut rgb)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        assert_eq!(rgb, block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &u, &v)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rgb_u16_is_exact_native_area_mean() {
        let (y, u, v) = ramp();
        let mut rgb = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb_u16(&mut rgb)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        assert_eq!(rgb, block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &u, &v)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn u8_color_is_not_a_narrowing_of_u16() {
        // Independence proof: over a varying ramp the u8 and u16 YUV→RGB
        // kernels (`range_params_n::<BITS, 8>` vs `::<BITS, BITS>`) round
        // and scale differently, so narrowing the binned u16 colour to u8
        // (`>> (BITS - 8)`) diverges from the genuine u8 bin — each binning
        // must match its OWN native-depth oracle, never the other narrowed.
        let (y, u, v) = ramp();
        let mut rgb_u8 = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb(&mut rgb_u8)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        // Each binning must match its own direct oracle.
        assert_eq!(rgb_u8, block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &u, &v)));
        assert_eq!(rgb_u16, block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &u, &v)));
        // And the u8 bin is NOT the narrowing of the u16 bin (the bug).
        let narrowed: Vec<u8> = rgb_u16.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_ne!(
          rgb_u8, narrowed,
          "u8 colour must be an independent bin, not a narrowed u16 bin"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_is_native_y_area_mean() {
        let (y, u, v) = ramp();
        // The Y plane is already native host-native u16, so the native-Y
        // oracle is the area-binned Y plane narrowed `>> (BITS - 8)` — and
        // it is range-INDEPENDENT (luma is the Y plane, no matrix / range
        // applied), so full-range and limited-range must both produce it.
        let y_ref = block_mean_2x2_u16(&y);
        let luma_ref: Vec<u8> = y_ref.iter().map(|&v| (v >> ($bits - 8)) as u8).collect();
        for full_range in [true, false] {
          let mut luma = vec![0u8; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(false)
            .with_luma(&mut luma)
            .unwrap();
            $walker(&frame(&y, &u, &v), full_range, M, &mut sink).unwrap();
          }
          assert_eq!(
            luma, luma_ref,
            "luma = binned native Y >> (BITS - 8) (full_range={full_range})"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_from_native_y_under_saturated_chroma() {
        // Constant Y, extreme U/V: the area-downscaled Y is constant, so
        // luma-from-Y stays exactly `Y >> (BITS - 8)`. RGB-derived luma
        // would clamp away.
        let yc: u16 = (MASK / 4) & MASK;
        let (y, u, v) = saturated(yc);
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        let expect = (yc >> ($bits - 8)) as u8;
        assert!(
          luma.iter().all(|&l| l == expect),
          "luma must be native Y >> shift ({expect}), not RGB-derived; got {luma:?}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn uniform_gray_color_unchanged_counterexample() {
        // Binning a uniform-gray frame is identity, so every colour output
        // must equal the direct full-res conversion (also uniform) — not a
        // narrowed-u16 approximation.
        let (y, u, v) = uniform_gray(MID);
        let direct_u8 = direct_rgb_u8(&y, &u, &v);
        let direct_u16 = direct_rgb_u16(&y, &u, &v);

        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut hh = vec![0u8; OUT * OUT];
        let mut ss = vec![0u8; OUT * OUT];
        let mut vv = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        let g_u8 = &direct_u8[..3];
        for px in rgb.chunks_exact(3) {
          assert_eq!(px, g_u8, "uniform-gray rgb must equal the direct gray");
        }
        for px in rgba.chunks_exact(4) {
          assert_eq!(&px[..3], g_u8, "uniform-gray rgba colour");
          assert_eq!(px[3], 0xFF, "uniform-gray rgba alpha");
        }
        let g_u16 = &direct_u16[..3];
        for px in rgb_u16.chunks_exact(3) {
          assert_eq!(px, g_u16, "uniform-gray rgb_u16 must equal the direct gray");
        }
        assert!(hh.iter().all(|&h| h == 0), "uniform-gray hsv H");
        assert!(ss.iter().all(|&s| s == 0), "uniform-gray hsv S");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn all_outputs_combo() {
        // Every output attached: each must match its own oracle, proving the
        // three binnings (u8 colour, native u16 colour, native Y) coexist.
        let (y, u, v) = ramp();
        let rgb_u8_ref = block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &u, &v));
        let rgb_u16_ref = block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &u, &v));
        let y_ref = block_mean_2x2_u16(&y);

        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
        let mut luma = vec![0u8; OUT * OUT];
        let mut hh = vec![0u8; OUT * OUT];
        let mut ss = vec![0u8; OUT * OUT];
        let mut vv = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        assert_eq!(rgb, rgb_u8_ref, "all-outputs rgb");
        for (px, rgb_px) in rgba.chunks_exact(4).zip(rgb_u8_ref.chunks_exact(3)) {
          assert_eq!(&px[..3], rgb_px, "all-outputs rgba colour");
          assert_eq!(px[3], 0xFF, "all-outputs rgba alpha");
        }
        assert_eq!(rgb_u16, rgb_u16_ref, "all-outputs rgb_u16");
        for (px, rgb_px) in rgba_u16.chunks_exact(4).zip(rgb_u16_ref.chunks_exact(3)) {
          assert_eq!(&px[..3], rgb_px, "all-outputs rgba_u16 colour");
          assert_eq!(px[3], MASK, "all-outputs rgba_u16 alpha");
        }
        let luma_ref: Vec<u8> = y_ref.iter().map(|&v| (v >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "all-outputs luma");
        let mut hh_ref = vec![0u8; OUT * OUT];
        let mut ss_ref = vec![0u8; OUT * OUT];
        let mut vv_ref = vec![0u8; OUT * OUT];
        crate::row::rgb_to_hsv_row(
          &rgb_u8_ref,
          &mut hh_ref,
          &mut ss_ref,
          &mut vv_ref,
          OUT * OUT,
          false,
        );
        assert_eq!(hh, hh_ref, "all-outputs hsv H");
        assert_eq!(ss, ss_ref, "all-outputs hsv S");
        assert_eq!(vv, vv_ref, "all-outputs hsv V");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn le_be_outputs_identical() {
        // LE and BE wire encodings of the same logical planes must produce
        // identical outputs: the binned row is host-native and the decode
        // recovers it with the source wire const, so a wrong const on either
        // host shows up as an LE/BE divergence.
        let (y, u, v) = ramp();
        let (y_le, u_le, v_le) = (as_le_u16(&y), as_le_u16(&u), as_le_u16(&v));
        let (y_be, u_be, v_be) = (as_be_u16(&y), as_be_u16(&u), as_be_u16(&v));

        let mut le_rgb = vec![0u8; OUT * OUT * 3];
        let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut le_luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb(&mut le_rgb)
          .unwrap()
          .with_rgb_u16(&mut le_rgb_u16)
          .unwrap()
          .with_luma(&mut le_luma)
          .unwrap();
          $walker(&frame(&y_le, &u_le, &v_le), FR, M, &mut sink).unwrap();
        }

        let mut be_rgb = vec![0u8; OUT * OUT * 3];
        let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut be_luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb(&mut be_rgb)
          .unwrap()
          .with_rgb_u16(&mut be_rgb_u16)
          .unwrap()
          .with_luma(&mut be_luma)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&y_be, &u_be, &v_be), FR, M, &mut sink).unwrap();
        }

        assert_eq!(le_rgb, be_rgb, "rgb LE/BE diverge");
        assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
        assert_eq!(le_luma, be_luma, "luma LE/BE diverge");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn simd_matches_scalar() {
        // The routed binnings must be SIMD/scalar bit-identical for every
        // output (the scalar and dispatched kernels share the contract).
        let (y, u, v) = ramp();
        let run = |simd: bool| {
          let mut rgb = vec![0u8; OUT * OUT * 3];
          let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
          let mut luma = vec![0u8; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(false)
            .with_simd(simd)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb_u16)
            .unwrap()
            .with_luma(&mut luma)
            .unwrap();
            $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
          }
          (rgb, rgb_u16, luma)
        };
        assert_eq!(run(true), run(false), "SIMD vs scalar diverge");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn fractional_ratio_simd_matches_scalar() {
        // A non-integer ratio (8 -> 3) exercises fractional area coverage;
        // the routed binnings must stay SIMD/scalar bit-identical (the
        // fractional weights are the same in both kernels).
        const OW: usize = 3;
        let (y, u, v) = ramp();
        let run = |simd: bool| {
          let mut rgb = vec![0u8; OW * OW * 3];
          let mut rgb_u16 = vec![0u16; OW * OW * 3];
          let mut luma = vec![0u8; OW * OW];
          {
            let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OW, OW),
            )
            .unwrap()
            .with_simd(simd)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb_u16)
            .unwrap()
            .with_luma(&mut luma)
            .unwrap();
            $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
          }
          (rgb, rgb_u16, luma)
        };
        assert_eq!(run(true), run(false), "fractional SIMD vs scalar diverge");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_new_sink() {
        let (y, u, v) = ramp();
        let mut direct = vec![0u8; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut direct)
            .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        let mut via_area = vec![0u8; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgb(&mut via_area)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        assert_eq!(direct, via_area, "identity plan must match the direct sink");
      }

      #[test]
      fn no_outputs_is_a_no_op() {
        let (y, u, v) = ramp();
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap();
        $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        assert!(
          !sink.luma_stream_u16_allocated(),
          "no-output sink allocated a luma stream"
        );
        assert!(
          !sink.rgb_stream_allocated(),
          "no-output sink allocated an rgb stream"
        );
        assert!(
          !sink.rgb_stream_u16_allocated(),
          "no-output sink allocated a u16 rgb stream"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resets_streams_across_frames() {
        // A reused sink must reset all three streams each frame; without the
        // reset, frame 2's row 0 is rejected as out-of-sequence.
        let (y1, u1, v1) = ramp();
        let invert =
          |p: &[u16]| -> Vec<u16> { p.iter().map(|&x| MASK.wrapping_sub(x) & MASK).collect() };
        let (y2, u2, v2) = (invert(&y1), invert(&u1), invert(&v1));
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&y1, &u1, &v1), FR, M, &mut sink).unwrap();
          $walker(&frame(&y2, &u2, &v2), FR, M, &mut sink).unwrap();
        }
        assert_eq!(
          rgb_u16,
          block_mean_2x2_rgb_u16(&direct_rgb_u16(&y2, &u2, &v2))
        );
        let y_ref = block_mean_2x2_u16(&y2);
        let luma_ref: Vec<u8> = y_ref.iter().map(|&v| (v >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref);
      }

      #[test]
      fn out_of_sequence_first_row_rejected_before_allocation() {
        let (y, u, v) = ramp();
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let r = 3 * SRC;
        let err = sink
          .process($row::new(
            &y[r..r + SRC],
            &u[r..r + SRC],
            &v[r..r + SRC],
            3,
            M,
            FR,
          ))
          .unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "expected OutOfSequenceRow, got {err:?}"
        );
        assert!(
          !sink.luma_stream_u16_allocated()
            && !sink.rgb_stream_allocated()
            && !sink.rgb_stream_u16_allocated(),
          "stream allocated for a rejected row"
        );
        assert!(
          rgb.iter().all(|&b| b == 0)
            && rgb_u16.iter().all(|&b| b == 0)
            && luma.iter().all(|&b| b == 0),
          "rejected row mutated output"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_mid_frame_output_change() {
        let (y, u, v) = ramp();
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&y[..SRC], &u[..SRC], &v[..SRC], 0, M, FR))
          .unwrap();
        sink.set_luma(&mut luma).unwrap();
        let err = sink
          .process($row::new(
            &y[SRC..2 * SRC],
            &u[SRC..2 * SRC],
            &v[SRC..2 * SRC],
            1,
            M,
            FR,
          ))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "expected ResampleOutputsChanged, got {err:?}"
        );
        assert!(
          luma.iter().all(|&b| b == 0),
          "rejected row mutated the new output"
        );
      }
    }
  };
}

yuv444p_high_bit_resample_suite!(
  yuv444p9,
  Yuv444p9LeFrame,
  Yuv444p9BeFrame,
  Yuv444p9,
  Yuv444p9Row,
  yuv444p9_to,
  yuv444p9_to_endian,
  9,
);
yuv444p_high_bit_resample_suite!(
  yuv444p10,
  Yuv444p10LeFrame,
  Yuv444p10BeFrame,
  Yuv444p10,
  Yuv444p10Row,
  yuv444p10_to,
  yuv444p10_to_endian,
  10,
);
yuv444p_high_bit_resample_suite!(
  yuv444p12,
  Yuv444p12LeFrame,
  Yuv444p12BeFrame,
  Yuv444p12,
  Yuv444p12Row,
  yuv444p12_to,
  yuv444p12_to_endian,
  12,
);
yuv444p_high_bit_resample_suite!(
  yuv444p14,
  Yuv444p14LeFrame,
  Yuv444p14BeFrame,
  Yuv444p14,
  Yuv444p14Row,
  yuv444p14_to,
  yuv444p14_to_endian,
  14,
);
yuv444p_high_bit_resample_suite!(
  yuv444p16,
  Yuv444p16LeFrame,
  Yuv444p16BeFrame,
  Yuv444p16,
  Yuv444p16Row,
  yuv444p16_to,
  yuv444p16_to_endian,
  16,
);
