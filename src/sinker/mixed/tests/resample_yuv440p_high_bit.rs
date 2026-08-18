//! Fused-downscale coverage for the high-bit **planar** 4:4:0 YUV family —
//! `Yuv440p10` / `Yuv440p12` (LE + BE wire). Low-packed `u16` Y plane +
//! **full-width**, half-height U / V planes (vertical-only subsampling — no
//! horizontal chroma upsampling; the vertical chroma sharing is resolved by
//! the walker, which hands each luma row its vertically-shared full-width
//! `u` / `v`).
//!
//! These route through [`packed_yuv444_triple_resample`] (the same tail the
//! high-bit planar 4:4:4 family uses — 4:4:0 reuses the `yuv444pN_to_rgb_*`
//! kernels at full-width chroma), with **three** independent native-precision
//! binnings:
//! - **u8 colour (rgb / rgba / hsv)** bins a converted source-width u8 RGB
//!   row (`yuv444pN_to_rgb_row_endian`).
//! - **u16 colour (rgb_u16 / rgba_u16)** bins an *independent* converted
//!   source-width native u16 RGB row — never a narrowing of the u8 bin.
//! - **luma** bins the de-interleaved native Y; `luma = binned_Y >>
//!   (BITS - 8)`. (Yuv440p exposes no `luma_u16` output.)
//!
//! Colour oracles drive a direct identity sink at source resolution (the
//! walker upsamples chroma vertically to full-res before the sink converts)
//! and 2x2-block-mean its output; the luma oracle area-bins the native Y
//! plane then narrows. Uniform-gray + saturated-chroma counterexamples pin
//! the independent-u8/u16 and native-Y-luma contracts.

use crate::{
  KernelMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
};

const SRC: usize = 8;
const CW: usize = SRC;
const CH: usize = SRC / 2;
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

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid u16 plane.
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

macro_rules! yuv440p_high_bit_resample_suite {
  (
    $mod:ident, $frame_le:ident, $frame_be:ident, $marker:ident, $row:ident,
    $walker:ident, $walker_be:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $row, $walker, $walker_be};

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = (1u16 << ($bits - 1));

      /// Per-pixel `(Y)` ramp + per-chroma-sample `(U, V)` ramp into an
      /// `SRC`-grid Y plane and full-width / half-height U / V planes
      /// (4:4:0, low-packed native codes so every kernel sees real math).
      fn ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; CW * CH];
        let mut v = vec![0u16; CW * CH];
        for i in 0..SRC * SRC {
          y[i] = ((40u32 + i as u32 * 37) & MASK as u32) as u16;
        }
        for i in 0..CW * CH {
          u[i] = ((300u32 + i as u32 * 53) & MASK as u32) as u16;
          v[i] = (MASK as u32).wrapping_sub(i as u32 * 41) as u16 & MASK;
        }
        (y, u, v)
      }

      /// Uniform-gray planes: constant Y, neutral chroma (U = V = mid).
      fn uniform_gray(y: u16) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        (
          vec![y & MASK; SRC * SRC],
          vec![MID & MASK; CW * CH],
          vec![MID & MASK; CW * CH],
        )
      }

      /// Saturated-chroma planes: constant Y, extreme U/V.
      fn saturated(y: u16) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        (
          vec![y & MASK; SRC * SRC],
          vec![MASK; CW * CH],
          vec![0u16; CW * CH],
        )
      }

      fn frame<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16]) -> $frame_le<'a> {
        $frame_le::try_new(
          y, u, v, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
        )
        .unwrap()
      }
      fn frame_be<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16]) -> $frame_be<'a> {
        $frame_be::try_new(
          y, u, v, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
        )
        .unwrap()
      }

      fn direct_rgb_u8(y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
        let mut rgb = vec![0u8; SRC * SRC * 3];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgb(&mut rgb)
          .unwrap();
        $walker(&frame(y, u, v), FR, M, &mut sink).unwrap();
        rgb
      }
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
        // kernels round and scale differently, so narrowing the binned u16
        // colour to u8 (`>> (BITS - 8)`) diverges from the genuine u8 bin —
        // each binning must match its OWN native-depth oracle.
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
        assert_eq!(rgb_u8, block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &u, &v)));
        assert_eq!(rgb_u16, block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &u, &v)));
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
        // Native Y is range-INDEPENDENT (luma is the Y plane area-mean, no
        // matrix / range applied), so full-range and limited-range must
        // both produce the area-binned Y plane narrowed `>> (BITS - 8)`.
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
      fn chroma_upsample_matches_direct() {
        // 4:4:0 is vertical-only chroma subsampling: the walker shares each
        // chroma row across the two luma rows above/below it (no horizontal
        // upsampling — chroma is full-width). A per-pixel-varying chroma
        // frame must still match the direct convert-then-bin (which shares
        // the same chroma row vertically at full resolution).
        let (y, u, v) = ramp();
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
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
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        assert_eq!(
          rgb,
          block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &u, &v)),
          "u8 chroma-upsampled"
        );
        assert_eq!(
          rgb_u16,
          block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &u, &v)),
          "u16 chroma-upsampled"
        );
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
        // Row 3's vertically-shared chroma row is `3 / 2 == 1` (full-width).
        let (yr, cr) = (3 * SRC, 1 * CW);
        let err = sink
          .process($row::new(
            &y[yr..yr + SRC],
            &u[cr..cr + CW],
            &v[cr..cr + CW],
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
        // Rows 0 and 1 both read chroma row 0 (`r / 2`), full-width.
        sink
          .process($row::new(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
          .unwrap();
        sink.set_luma(&mut luma).unwrap();
        let err = sink
          .process($row::new(&y[SRC..2 * SRC], &u[..CW], &v[..CW], 1, M, FR))
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

yuv440p_high_bit_resample_suite!(
  yuv440p10,
  Yuv440p10LeFrame,
  Yuv440p10BeFrame,
  Yuv440p10,
  Yuv440p10Row,
  yuv440p10_to,
  yuv440p10_to_endian,
  10,
);
yuv440p_high_bit_resample_suite!(
  yuv440p12,
  Yuv440p12LeFrame,
  Yuv440p12BeFrame,
  Yuv440p12,
  Yuv440p12Row,
  yuv440p12_to,
  yuv440p12_to_endian,
  12,
);
