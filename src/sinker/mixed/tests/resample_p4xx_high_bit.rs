//! Fused-downscale coverage for the high-bit **semi-planar** 4:4:4 YUV
//! family — `P410` / `P412` / `P416` (LE + BE wire). A full-width
//! high-bit-packed `u16` Y plane plus one **interleaved full-width /
//! full-height** `U,V,U,V…` plane (one chroma pair per pixel); the row
//! kernels de-interleave the UV plane per-pixel — there is **no** chroma
//! upsampling.
//!
//! These route through [`packed_yuv444_triple_resample`] with **three**
//! independent native-precision binnings (see `resample_p0xx_high_bit`
//! for the full contract — the only difference is full-resolution chroma):
//! - **u8 colour (rgb / rgba / hsv)** bins a converted source-width u8
//!   RGB row (`pNNN_to_rgb_row_endian`, UV de-interleaved internally).
//! - **u16 colour (rgb_u16 / rgba_u16)** bins an *independent* converted
//!   source-width native u16 RGB row (low-bit-packed) — never a narrowing.
//! - **luma** bins the **de-packed** native Y; `luma = binned_Y >>
//!   (BITS - 8)`. (P-formats expose no `luma_u16` output.)
//!
//! Colour oracles drive a direct identity P-format sink at source
//! resolution and 2x2-block-mean its output; the luma oracle area-bins
//! the de-packed logical Y plane then narrows.

use crate::{
  KernelMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, ResampleError},
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

macro_rules! p4xx_high_bit_resample_suite {
  (
    $mod:ident, $frame_le:ident, $frame_be:ident, $marker:ident, $row:ident,
    $walker:ident, $walker_be:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $row, $walker, $walker_be};

      const SHIFT: u32 = 16 - $bits;
      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = (1u16 << ($bits - 1));

      /// Per-pixel logical `(Y)` ramp + per-pixel logical `(U, V)` ramp,
      /// high-bit-packed into a full-width Y plane and a **full-width /
      /// full-height** interleaved `U,V,U,V…` plane (4:4:4, one pair per
      /// pixel).
      fn ramp() -> (Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut uv = vec![0u16; SRC * SRC * 2];
        for i in 0..SRC * SRC {
          y[i] = (((40u32 + i as u32 * 37) & MASK as u32) as u16) << SHIFT;
          let u = ((300u32 + i as u32 * 53) & MASK as u32) as u16;
          let v = ((MASK as u32).wrapping_sub(i as u32 * 41) as u16) & MASK;
          uv[2 * i] = u << SHIFT;
          uv[2 * i + 1] = v << SHIFT;
        }
        (y, uv)
      }

      /// Uniform-gray planes: constant logical Y, neutral chroma
      /// (U = V = mid), high-bit-packed.
      fn uniform_gray(y: u16) -> (Vec<u16>, Vec<u16>) {
        (
          vec![(y & MASK) << SHIFT; SRC * SRC],
          vec![(MID & MASK) << SHIFT; SRC * SRC * 2],
        )
      }

      /// Saturated-chroma planes: constant logical Y, extreme U/V,
      /// high-bit-packed.
      fn saturated(y: u16) -> (Vec<u16>, Vec<u16>) {
        let mut uv = vec![0u16; SRC * SRC * 2];
        for i in 0..SRC * SRC {
          uv[2 * i] = MASK << SHIFT;
          uv[2 * i + 1] = 0u16;
        }
        (vec![(y & MASK) << SHIFT; SRC * SRC], uv)
      }

      fn frame<'a>(y: &'a [u16], uv: &'a [u16]) -> $frame_le<'a> {
        // 4:4:4 interleaved UV stride = `2 * SRC` u16 (one pair per pixel).
        $frame_le::new(y, uv, SRC as u32, SRC as u32, SRC as u32, 2 * SRC as u32)
      }
      fn frame_be<'a>(y: &'a [u16], uv: &'a [u16]) -> $frame_be<'a> {
        $frame_be::new(y, uv, SRC as u32, SRC as u32, SRC as u32, 2 * SRC as u32)
      }

      /// Logical (de-packed) Y plane for the luma oracle.
      fn logical_y(y: &[u16]) -> Vec<u16> {
        y.iter().map(|&s| s >> SHIFT).collect()
      }

      fn direct_rgb_u8(y: &[u16], uv: &[u16]) -> Vec<u8> {
        let mut rgb = vec![0u8; SRC * SRC * 3];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgb(&mut rgb)
          .unwrap();
        $walker(&frame(y, uv), FR, sink.set_kernel_matrix(M)).unwrap();
        rgb
      }
      fn direct_rgb_u16(y: &[u16], uv: &[u16]) -> Vec<u16> {
        let mut rgb = vec![0u16; SRC * SRC * 3];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgb_u16(&mut rgb)
          .unwrap();
        $walker(&frame(y, uv), FR, sink.set_kernel_matrix(M)).unwrap();
        rgb
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rgb_u8_matches_area_bin_of_direct() {
        let (y, uv) = ramp();
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
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(rgb, block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &uv)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rgb_u16_is_exact_native_area_mean() {
        let (y, uv) = ramp();
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
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(rgb, block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &uv)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn u8_color_is_not_a_narrowing_of_u16() {
        // Independence proof: over a varying ramp the u8 and u16 YUV→RGB
        // kernels round and scale differently, so narrowing the binned u16
        // colour to u8 (`>> (BITS - 8)`) diverges from the genuine u8 bin.
        let (y, uv) = ramp();
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
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(rgb_u8, block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &uv)));
        assert_eq!(rgb_u16, block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &uv)));
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
        let (y, uv) = ramp();
        let y_ref = block_mean_2x2_u16(&logical_y(&y));
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
            $walker(&frame(&y, &uv), full_range, sink.set_kernel_matrix(M)).unwrap();
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
        let (y, uv) = saturated(yc);
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
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
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
        let (y, uv) = uniform_gray(MID);
        let direct_u8 = direct_rgb_u8(&y, &uv);
        let direct_u16 = direct_rgb_u16(&y, &uv);

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
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
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
        let (y, uv) = ramp();
        let rgb_u8_ref = block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &uv));
        let rgb_u16_ref = block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &uv));
        let y_ref = block_mean_2x2_u16(&logical_y(&y));

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
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
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
        let (y, uv) = ramp();
        let (y_le, uv_le) = (as_le_u16(&y), as_le_u16(&uv));
        let (y_be, uv_be) = (as_be_u16(&y), as_be_u16(&uv));

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
          $walker(&frame(&y_le, &uv_le), FR, sink.set_kernel_matrix(M)).unwrap();
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
          $walker_be::<_, true>(&frame_be(&y_be, &uv_be), FR, sink.set_kernel_matrix(M)).unwrap();
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
        let (y, uv) = ramp();
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
            $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
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
      fn chroma_per_pixel_matches_direct() {
        // 4:4:4 has one chroma pair per pixel — no upsampling. The UV
        // de-interleave happens in-register inside the conversion closures.
        // A per-pixel-varying chroma frame must still match the direct
        // convert-then-bin.
        let (y, uv) = ramp();
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
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(
          rgb,
          block_mean_2x2_rgb_u8(&direct_rgb_u8(&y, &uv)),
          "u8 per-pixel chroma"
        );
        assert_eq!(
          rgb_u16,
          block_mean_2x2_rgb_u16(&direct_rgb_u16(&y, &uv)),
          "u16 per-pixel chroma"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_new_sink() {
        let (y, uv) = ramp();
        let mut direct = vec![0u8; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut direct)
            .unwrap();
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let mut via_area = vec![0u8; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_native(false)
          .with_rgb(&mut via_area)
          .unwrap();
          $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(direct, via_area, "identity plan must match the direct sink");
      }

      #[test]
      fn no_outputs_is_a_no_op() {
        let (y, uv) = ramp();
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap();
        $walker(&frame(&y, &uv), FR, sink.set_kernel_matrix(M)).unwrap();
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
        let (y1, uv1) = ramp();
        let invert = |p: &[u16]| -> Vec<u16> {
          p.iter()
            .map(|&x| (MASK.wrapping_sub(x >> SHIFT) & MASK) << SHIFT)
            .collect()
        };
        let (y2, uv2) = (invert(&y1), invert(&uv1));
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
          $walker(&frame(&y1, &uv1), FR, sink.set_kernel_matrix(M)).unwrap();
          $walker(&frame(&y2, &uv2), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(rgb_u16, block_mean_2x2_rgb_u16(&direct_rgb_u16(&y2, &uv2)));
        let y_ref = block_mean_2x2_u16(&logical_y(&y2));
        let luma_ref: Vec<u8> = y_ref.iter().map(|&v| (v >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref);
      }

      #[test]
      fn out_of_sequence_first_row_rejected_before_allocation() {
        let (y, uv) = ramp();
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
        // 4:4:4 chroma row == luma row; the interleaved chroma row is
        // `2 * SRC` u16 wide (one pair per pixel).
        let (yr, cr) = (3 * SRC, 3 * 2 * SRC);
        let err = sink
          .process($row::for_tests(
            &y[yr..yr + SRC],
            &uv[cr..cr + 2 * SRC],
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
        let (y, uv) = ramp();
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
        // 4:4:4: each luma row reads its own (same-index) chroma row,
        // `2 * SRC` u16 wide.
        sink
          .process($row::for_tests(&y[..SRC], &uv[..2 * SRC], 0, M, FR))
          .unwrap();
        sink.set_luma(&mut luma).unwrap();
        let err = sink
          .process($row::for_tests(
            &y[SRC..2 * SRC],
            &uv[2 * SRC..4 * SRC],
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

p4xx_high_bit_resample_suite!(
  p410,
  P410LeFrame,
  P410BeFrame,
  P410,
  P410Row,
  p410_to,
  p410_to_endian,
  10,
);
p4xx_high_bit_resample_suite!(
  p412,
  P412LeFrame,
  P412BeFrame,
  P412,
  P412Row,
  p412_to,
  p412_to_endian,
  12,
);
p4xx_high_bit_resample_suite!(
  p416,
  P416LeFrame,
  P416BeFrame,
  P416,
  P416Row,
  p416_to,
  p416_to_endian,
  16,
);
