//! Fused-downscale coverage for the high-bit packed 4:2:2 YUV family —
//! `Y210` (10-bit), `Y212` (12-bit), `Y216` (16-bit). Packed YUYV-order
//! u16 quadruples, MSB-aligned.
//!
//! These route through [`packed_yuv422_triple_resample`], the 4:2:2
//! analogue of the high-bit 4:4:4 route, with **three** independent
//! native-precision binnings — the u8 and u16 YUV→RGB kernels round and
//! scale *independently*, and luma is native Y:
//! - **u8 colour (rgb / rgba / hsv)** bins a converted source-width u8
//!   RGB row (the format's `*_to_rgb_row` kernel).
//! - **u16 colour (rgb_u16 / rgba_u16)** bins a converted source-width
//!   native u16 RGB row at native depth.
//! - **luma / luma_u16** bin the de-interleaved native Y; luma_u16 is the
//!   binned native Y, luma is `binned_Y >> (BITS - 8)`.
//!
//! Each output is byte-identical to the area-bin of the **direct**
//! full-resolution conversion (convert-then-bin), so the oracles below
//! drive a direct identity sink at source resolution and 2x2-block-mean
//! its output. The uniform-gray counterexample pins the real parity bug:
//! deriving u8 colour by narrowing the u16 bin would change a
//! uniform-gray downscale's colour, so the u8 group must bin its own u8
//! conversion.

use super::force_row_stage;
use crate::{
  KernelMatrix, PixelSink,
  frame::Y2xxFrame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Re-encode a host-native u16 slice as BE-encoded byte storage (the
/// `Y2xxBE` plane contract), recovered via `u16::from_be` in the kernel.
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

// A per-format macro keeps the three near-identical suites in lockstep
// while naming each test after its format (so a failure points at the
// exact bit depth). `$marker` is the source marker, `$walker` the LE
// walker, `$walker_be` the `_endian` walker, `$bits` the active depth.
macro_rules! y2xx_resample_suite {
  (
    $mod:ident, $marker:ident, $row:ident, $walker:ident, $walker_be:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $row, $walker, $walker_be};

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = (1u16 << ($bits - 1));
      const SHIFT: u32 = 16 - $bits;

      /// Per-pixel `(Y, U, V)` ramp packed into an `SRC`-grid Y2xx plane
      /// (`Y₀, U, Y₁, V` quadruples, MSB-aligned). Chroma is sampled at
      /// the even column of each 2-pixel pair (4:2:2). Interior native
      /// codes so every kernel sees real math.
      fn ramp_packed() -> Vec<u16> {
        let mut buf = vec![0u16; SRC * 2 * SRC];
        for row in 0..SRC {
          for cx in 0..SRC / 2 {
            let y0 = ((40u32 + (row * SRC + cx * 2) as u32 * 37) & MASK as u32) as u16;
            let y1 = ((40u32 + (row * SRC + cx * 2 + 1) as u32 * 37) & MASK as u32) as u16;
            let u = ((300u32 + (cx as u32) * 53 + row as u32 * 11) & MASK as u32) as u16;
            let v = (MASK as u32).wrapping_sub((cx as u32) * 41 + row as u32 * 7) as u16 & MASK;
            let base = row * 2 * SRC + cx * 4;
            buf[base] = y0 << SHIFT;
            buf[base + 1] = u << SHIFT;
            buf[base + 2] = y1 << SHIFT;
            buf[base + 3] = v << SHIFT;
          }
        }
        buf
      }

      /// Uniform-gray plane: constant Y, neutral chroma (U = V = mid).
      /// Binning a uniform frame is identity, so every resampled colour
      /// output must equal the direct full-res conversion.
      fn uniform_gray_packed(y: u16) -> Vec<u16> {
        let mut buf = vec![0u16; SRC * 2 * SRC];
        for q in 0..(SRC * SRC / 2) {
          let base = q * 4;
          buf[base] = (y & MASK) << SHIFT;
          buf[base + 1] = (MID & MASK) << SHIFT;
          buf[base + 2] = (y & MASK) << SHIFT;
          buf[base + 3] = (MID & MASK) << SHIFT;
        }
        buf
      }

      /// Saturated-chroma plane: constant Y, extreme U/V — the case where
      /// RGB-derived luma would clamp away from the Y plane.
      fn saturated_packed(y: u16) -> Vec<u16> {
        let mut buf = vec![0u16; SRC * 2 * SRC];
        for q in 0..(SRC * SRC / 2) {
          let base = q * 4;
          buf[base] = (y & MASK) << SHIFT;
          buf[base + 1] = MASK << SHIFT;
          buf[base + 2] = (y & MASK) << SHIFT;
          buf[base + 3] = 0;
        }
        buf
      }

      fn frame(buf: &[u16]) -> Y2xxFrame<'_, $bits, false> {
        Y2xxFrame::try_new(buf, SRC as u32, SRC as u32, (2 * SRC) as u32).unwrap()
      }
      fn frame_be(buf: &[u16]) -> Y2xxFrame<'_, $bits, true> {
        Y2xxFrame::try_new(buf, SRC as u32, SRC as u32, (2 * SRC) as u32).unwrap()
      }

      /// Direct full-resolution u8 RGB of the packed frame.
      fn direct_rgb_u8(packed: &[u16]) -> Vec<u8> {
        let mut rgb = vec![0u8; SRC * SRC * 3];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgb(&mut rgb)
          .unwrap();
        $walker(&frame(packed), FR, M, &mut sink).unwrap();
        rgb
      }
      /// Direct full-resolution native u16 RGB of the packed frame.
      fn direct_rgb_u16(packed: &[u16]) -> Vec<u16> {
        let mut rgb = vec![0u16; SRC * SRC * 3];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgb_u16(&mut rgb)
          .unwrap();
        $walker(&frame(packed), FR, M, &mut sink).unwrap();
        rgb
      }
      /// Direct full-resolution native Y (u16) of the packed frame.
      fn direct_luma_u16(packed: &[u16]) -> Vec<u16> {
        let mut y = vec![0u16; SRC * SRC];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_luma_u16(&mut y)
          .unwrap();
        $walker(&frame(packed), FR, M, &mut sink).unwrap();
        y
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rgb_u8_matches_area_bin_of_direct() {
        let packed = ramp_packed();
        let mut rgb = vec![0u8; OUT * OUT * 3];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
          .with_rgb(&mut rgb)
          .unwrap();
          $walker(&frame(&packed), FR, M, &mut sink).unwrap();
        }
        assert_eq!(rgb, block_mean_2x2_rgb_u8(&direct_rgb_u8(&packed)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rgb_u16_is_exact_native_area_mean() {
        let packed = ramp_packed();
        let mut rgb = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
          .with_rgb_u16(&mut rgb)
          .unwrap();
          $walker(&frame(&packed), FR, M, &mut sink).unwrap();
        }
        assert_eq!(rgb, block_mean_2x2_rgb_u16(&direct_rgb_u16(&packed)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_is_native_y_area_mean() {
        let packed = ramp_packed();
        let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walker(&frame(&packed), FR, M, &mut sink).unwrap();
        }
        let y_ref = block_mean_2x2_u16(&direct_luma_u16(&packed));
        assert_eq!(luma_u16, y_ref, "luma_u16 = native-Y area mean");
        // luma is the binned native Y narrowed `>> (BITS - 8)`, matching
        // a direct conversion of the area-downscaled native frame.
        let luma_ref: Vec<u8> = y_ref.iter().map(|&y| (y >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "luma = binned native Y >> (BITS - 8)");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn uniform_gray_color_unchanged_counterexample() {
        // The high-bit-YUV parity bug: deriving u8 colour by narrowing
        // the u16 bin changes a uniform-gray downscale's colour. With a
        // uniform-gray frame, binning is identity, so every colour output
        // must equal the direct full-res conversion (also uniform).
        let packed = uniform_gray_packed(MID);
        let direct_u8 = direct_rgb_u8(&packed);
        let direct_u16 = direct_rgb_u16(&packed);

        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut hh = vec![0u8; OUT * OUT];
        let mut ss = vec![0u8; OUT * OUT];
        let mut vv = vec![0u8; OUT * OUT];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap();
          $walker(&frame(&packed), FR, M, &mut sink).unwrap();
        }
        // Every direct full-res pixel is the same gray; the resampled
        // pixels must match it exactly (not a narrowed-u16 approximation).
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
        // HSV of a uniform-gray frame: achromatic (H = 0, S = 0).
        assert!(hh.iter().all(|&h| h == 0), "uniform-gray hsv H");
        assert!(ss.iter().all(|&s| s == 0), "uniform-gray hsv S");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_from_native_y_under_saturated_chroma() {
        // Constant Y, extreme U/V: the area-downscaled Y is constant, so
        // luma-from-Y stays exactly Y. RGB-derived luma would clamp away.
        let y: u16 = (MASK / 4) & MASK;
        let packed = saturated_packed(y);
        let mut luma_u16 = vec![0u16; OUT * OUT];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walker(&frame(&packed), FR, M, &mut sink).unwrap();
        }
        assert!(
          luma_u16.iter().all(|&v| v == y),
          "luma_u16 must be native Y ({y}), not RGB-derived; got {luma_u16:?}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn all_outputs_combo() {
        // Every output attached: each must match its own oracle, proving
        // the three binnings (u8 colour, native u16 colour, native Y)
        // coexist.
        let packed = ramp_packed();
        let rgb_u8_ref = block_mean_2x2_rgb_u8(&direct_rgb_u8(&packed));
        let rgb_u16_ref = block_mean_2x2_rgb_u16(&direct_rgb_u16(&packed));
        let y_ref = block_mean_2x2_u16(&direct_luma_u16(&packed));

        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
        let mut luma = vec![0u8; OUT * OUT];
        let mut luma_u16 = vec![0u16; OUT * OUT];
        let mut hh = vec![0u8; OUT * OUT];
        let mut ss = vec![0u8; OUT * OUT];
        let mut vv = vec![0u8; OUT * OUT];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
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
          .with_luma_u16(&mut luma_u16)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap();
          $walker(&frame(&packed), FR, M, &mut sink).unwrap();
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
        assert_eq!(luma_u16, y_ref, "all-outputs luma_u16");
        let luma_ref: Vec<u8> = y_ref.iter().map(|&y| (y >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "all-outputs luma");
        // HSV from the binned u8 RGB.
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
        // LE and BE wire encodings of the same logical plane must produce
        // identical outputs: the binned row is host-native and the derive
        // kernels recover it with `HOST_NATIVE_BE`, so a wrong wire const
        // on either host shows up as an LE/BE divergence.
        let packed_le = ramp_packed();
        let packed_be = as_be_u16(&packed_le);

        let mut le_rgb = vec![0u8; OUT * OUT * 3];
        let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut le_luma_u16 = vec![0u16; OUT * OUT];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
          .with_rgb(&mut le_rgb)
          .unwrap()
          .with_rgb_u16(&mut le_rgb_u16)
          .unwrap()
          .with_luma_u16(&mut le_luma_u16)
          .unwrap();
          $walker(&frame(&packed_le), FR, M, &mut sink).unwrap();
        }

        let mut be_rgb = vec![0u8; OUT * OUT * 3];
        let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut be_luma_u16 = vec![0u16; OUT * OUT];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
          .with_rgb(&mut be_rgb)
          .unwrap()
          .with_rgb_u16(&mut be_rgb_u16)
          .unwrap()
          .with_luma_u16(&mut be_luma_u16)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&packed_be), FR, M, &mut sink).unwrap();
        }

        assert_eq!(le_rgb, be_rgb, "rgb LE/BE diverge");
        assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
        assert_eq!(le_luma_u16, be_luma_u16, "luma_u16 LE/BE diverge");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_new_sink() {
        let packed = ramp_packed();
        let mut direct = vec![0u8; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut direct)
            .unwrap();
          $walker(&frame(&packed), FR, M, &mut sink).unwrap();
        }
        let mut via_area = vec![0u8; SRC * SRC * 3];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(SRC, SRC),
            )
            .unwrap(),
          )
          .with_rgb(&mut via_area)
          .unwrap();
          $walker(&frame(&packed), FR, M, &mut sink).unwrap();
        }
        assert_eq!(direct, via_area, "identity plan must match the direct sink");
      }

      #[test]
      fn no_outputs_is_a_no_op() {
        let packed = ramp_packed();
        let mut sink = force_row_stage(
          MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap(),
        );
        $walker(&frame(&packed), FR, M, &mut sink).unwrap();
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
        // A reused sink must reset all three streams each frame; without
        // the reset, frame 2's row 0 is rejected as out-of-sequence.
        let p1 = ramp_packed();
        let mut p2 = p1.clone();
        for v in p2.iter_mut() {
          *v = (MASK << SHIFT).wrapping_sub(*v) & (MASK << SHIFT);
        }
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma_u16 = vec![0u16; OUT * OUT];
        {
          let mut sink = force_row_stage(
            MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap(),
          )
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walker(&frame(&p1), FR, M, &mut sink).unwrap();
          $walker(&frame(&p2), FR, M, &mut sink).unwrap();
        }
        // Frame 2's outputs must reflect frame 2 (reset succeeded).
        assert_eq!(luma_u16, block_mean_2x2_u16(&direct_luma_u16(&p2)));
        assert_eq!(rgb_u16, block_mean_2x2_rgb_u16(&direct_rgb_u16(&p2)));
      }

      #[test]
      fn out_of_sequence_first_row_rejected_before_allocation() {
        let packed = ramp_packed();
        let row3 = &packed[3 * 2 * SRC..4 * 2 * SRC];
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma_u16 = vec![0u16; OUT * OUT];
        let mut sink = force_row_stage(
          MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap(),
        )
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let err = sink.process($row::new(row3, 3, M, FR)).unwrap_err();
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
            && luma_u16.iter().all(|&b| b == 0),
          "rejected row mutated output"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_mid_frame_out_of_sequence() {
        let packed = ramp_packed();
        let mut luma_u16 = vec![0u16; OUT * OUT];
        let mut sink = force_row_stage(
          MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap(),
        )
        .with_luma_u16(&mut luma_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&packed[..2 * SRC], 0, M, FR))
          .unwrap();
        let err = sink
          .process($row::new(&packed[2 * 2 * SRC..3 * 2 * SRC], 2, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "expected OutOfSequenceRow, got {err:?}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_mid_frame_output_change() {
        let packed = ramp_packed();
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut luma_u16 = vec![0u16; OUT * OUT];
        let mut sink = force_row_stage(
          MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap(),
        )
        .with_rgb(&mut rgb)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&packed[..2 * SRC], 0, M, FR))
          .unwrap();
        sink.set_luma_u16(&mut luma_u16).unwrap();
        let err = sink
          .process($row::new(&packed[2 * SRC..2 * 2 * SRC], 1, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "expected ResampleOutputsChanged, got {err:?}"
        );
        assert!(
          luma_u16.iter().all(|&b| b == 0),
          "rejected row mutated the new output"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejected_first_row_does_not_poison_output_retry() {
        // A rejected out-of-sequence FIRST row must store no frozen-output
        // snapshot, so retrying row 0 after reconfiguring the output set
        // succeeds instead of tripping ResampleOutputsChanged against a
        // snapshot the rejected row should never have committed.
        let packed = ramp_packed();
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut sink = force_row_stage(
          MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap(),
        )
        .with_rgb(&mut rgb)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let row3 = &packed[3 * 2 * SRC..4 * 2 * SRC];
        let err = sink.process($row::new(row3, 3, M, FR)).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "expected OutOfSequenceRow, got {err:?}"
        );
        let mut luma_u16 = vec![0u16; OUT * OUT];
        sink.set_luma_u16(&mut luma_u16).unwrap();
        sink
          .process($row::new(&packed[..2 * SRC], 0, M, FR))
          .expect("row 0 must succeed after a rejected out-of-sequence first row");
      }
    }
  };
}

y2xx_resample_suite!(y210, Y210, Y210Row, y210_to, y210_to_endian, 10,);
y2xx_resample_suite!(y212, Y212, Y212Row, y212_to, y212_to_endian, 12,);
y2xx_resample_suite!(y216, Y216, Y216Row, y216_to, y216_to_endian, 16,);
