//! Fused-downscale coverage for the high-bit **semi-planar** 4:2:0 YUV
//! NATIVE fast tier — `P010` / `P012` / `P016` (LE + BE wire), the `u16`
//! semi-planar twin of the 8-bit
//! [`semi_planar_process_native`](crate::sinker::mixed::semi_planar_8bit)
//! that reuses the high-bit PLANAR join
//! ([`yuv420p16_process_native`](crate::sinker::mixed::subsampled_4_2_0_high_bit::yuv420p16_process_native)).
//!
//! The native tier de-interleaves + DE-PACKS (`raw >> (16 - BITS)`) the
//! high-bit-packed Y and interleaved UV wire planes into wrapper-owned
//! host-native LOGICAL u16 scratch, then bins those straight to the output
//! grid and converts ONCE per output row at output width — vs the
//! row-stage tier
//! ([`packed_yuv422_triple_resample`](crate::sinker::mixed::packed_yuv422_triple_resample)),
//! which converts each source row at source width then bins. The tiers
//! differ in colour SEMANTICS (native averages in YUV then converts;
//! row-stage converts then averages in RGB), so native is NOT byte-
//! identical to row-stage — only within a small tolerance in-gamut. Luma
//! is bit-identical (both bin the same de-packed native Y then narrow
//! `>> (BITS - 8)`).
//!
//! Per format (LE + BE): native-vs-rowstage tolerance (luma bit-identical,
//! colour within ±tol), `native_be_within_tolerance_of_rowstage_be` (the
//! `BE = HOST_NATIVE_BE` handoff proof), uniform-gray + independent-kernel
//! guards (#37), the inter-area luma oracle (area-bin of the DE-PACKED
//! logical Y), and the four atomicity regressions on
//! [`arm_p0xx_alloc_failure`].

use crate::{
  KernelMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
};

const SRC: usize = 8;
const CW: usize = SRC / 2;
const CH: usize = SRC / 2;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// In-gamut per-channel tolerance between the native and row-stage tiers.
/// The two average in different domains (YUV vs RGB) and round
/// independently per output pixel; this pins the observed in-gamut max
/// plus a 1-LSB cross-platform SIMD-vs-scalar margin. Mirrors the planar
/// high-bit native suite.
const TOL_U8: u8 = 2;

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

macro_rules! p0xx_high_bit_native_suite {
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

      /// Per-pixel logical Y ramp + per-chroma-sample logical U / V ramp,
      /// every code in-gamut-ish, HIGH-BIT-PACKED (`logical << (16 - BITS)`)
      /// into a full-width Y plane and a half-width / half-height
      /// interleaved `U,V,U,V…` plane (4:2:0).
      fn ramp() -> (Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut uv = vec![0u16; CW * CH * 2];
        for i in 0..SRC * SRC {
          // Keep Y near the legal-range middle so converted RGB stays in
          // gamut and the native-vs-rowstage delta is per-pixel rounding.
          let logical = (MID as u32 + ((i as u32 * 37) % (MASK as u32 / 4))) as u16 & MASK;
          y[i] = logical << SHIFT;
        }
        for i in 0..CW * CH {
          // Chroma in a modest band around neutral so the result stays in
          // gamut.
          let u =
            (MID as u32 + ((i as u32 * 53) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16 & MASK;
          let v =
            (MID as u32 + ((i as u32 * 41) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16 & MASK;
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
          vec![(MID & MASK) << SHIFT; CW * CH * 2],
        )
      }

      /// Crafted VARYING illegal-chroma fixture: extreme alternating chroma
      /// over a super-black→super-white Y ramp, high-bit-packed — many 2x2
      /// blocks straddle the RGB clamp, where native (average-in-YUV) and
      /// row-stage (convert-then-average) genuinely diverge.
      fn out_of_gamut() -> (Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut uv = vec![0u16; CW * CH * 2];
        for i in 0..SRC * SRC {
          let logical = ((i as u32 * MASK as u32) / (SRC * SRC) as u32) as u16 & MASK;
          y[i] = logical << SHIFT;
        }
        for i in 0..CW * CH {
          let hi = i % 2 == 0;
          uv[2 * i] = if hi { MASK } else { 0 } << SHIFT;
          uv[2 * i + 1] = if hi { 0 } else { MASK } << SHIFT;
        }
        (y, uv)
      }

      fn frame<'a>(y: &'a [u16], uv: &'a [u16]) -> $frame_le<'a> {
        // 4:2:0 interleaved UV stride = `2 * (SRC / 2)` = `SRC` u16.
        $frame_le::new(y, uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32)
      }
      fn frame_be<'a>(y: &'a [u16], uv: &'a [u16]) -> $frame_be<'a> {
        $frame_be::new(y, uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32)
      }

      /// Logical (de-packed) Y plane for the luma oracle.
      fn logical_y(y: &[u16]) -> Vec<u16> {
        y.iter().map(|&s| s >> SHIFT).collect()
      }

      /// Re-encode a host-native u16 slice as host-independent LE-wire byte
      /// storage (the `*LeFrame` plane contract); a no-op on LE, a byte
      /// swap on BE.
      fn as_le(host: &[u16]) -> Vec<u16> {
        host
          .iter()
          .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
          .collect()
      }
      /// Re-encode a host-native u16 slice as host-independent BE-wire byte
      /// storage (the `*BeFrame` plane contract).
      fn as_be(host: &[u16]) -> Vec<u16> {
        host
          .iter()
          .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
          .collect()
      }

      /// Drive the native tier (LE) for rgb / rgb_u16 / luma. Fixtures are
      /// re-encoded to LE-wire storage so the LeFrame plane contract holds
      /// on any host endianness.
      fn native_run(y: &[u16], uv: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let (yl, uvl) = (as_le(y), as_le(uv));
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
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&yl, &uvl), FR, M, &mut sink).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the row-stage tier (LE) — `.with_native(false)`.
      fn rowstage_run(y: &[u16], uv: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let (yl, uvl) = (as_le(y), as_le(uv));
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
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&yl, &uvl), FR, M, &mut sink).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the row-stage tier (BE) — the correct host-independent
      /// reference (it de-interleaves BE-wire bytes to host-native before
      /// converting). On a big-endian host the unfixed native tier would
      /// diverge from it; this is the BE-handoff regression's oracle.
      fn rowstage_be_run(y: &[u16], uv: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let (yb, uvb) = (as_be(y), as_be(uv));
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
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
          $walker_be::<_, true>(&frame_be(&yb, &uvb), FR, M, &mut sink).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_within_tolerance_of_rowstage_and_luma_bit_identical() {
        let (y, uv) = ramp();
        let (n_rgb, n_rgb16, n_luma) = native_run(&y, &uv);
        let (r_rgb, r_rgb16, r_luma) = rowstage_run(&y, &uv);

        // Luma: both tiers bin the SAME de-packed native Y and narrow
        // `>> (BITS - 8)`, so it is bit-identical.
        assert_eq!(n_luma, r_luma, "luma must be bit-identical across tiers");

        let max_u8 = n_rgb
          .iter()
          .zip(&r_rgb)
          .map(|(&a, &b)| a.abs_diff(b))
          .max()
          .unwrap_or(0);
        assert!(
          max_u8 <= TOL_U8,
          "u8 native-vs-rowstage max delta {max_u8} exceeds tolerance {TOL_U8}"
        );

        let tol_u16: u16 = (TOL_U8 as u16) << ($bits - 8);
        let max_u16 = n_rgb16
          .iter()
          .zip(&r_rgb16)
          .map(|(&a, &b)| a.abs_diff(b))
          .max()
          .unwrap_or(0);
        assert!(
          max_u16 <= tol_u16,
          "u16 native-vs-rowstage max delta {max_u16} exceeds tolerance {tol_u16}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_be_matches_native_le() {
        // The native tier de-interleaves + de-packs the wire planes to
        // host-native LOGICAL u16 BEFORE binning, so BE and LE sources
        // produce identical output.
        let (y, uv) = ramp();
        let (n_rgb_le, n_rgb16_le, n_luma_le) = native_run(&y, &uv);

        let (yb, uvb) = (as_be(&y), as_be(&uv));
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&yb, &uvb), FR, M, &mut sink).unwrap();
        }
        assert_eq!(rgb, n_rgb_le, "BE u8 colour must match LE");
        assert_eq!(rgb_u16, n_rgb16_le, "BE u16 colour must match LE");
        assert_eq!(luma, n_luma_le, "BE luma must match LE");
      }

      /// The host-native-endian regression: BE native vs the correct BE
      /// row-stage reference on the SAME ramp, within the SAME tolerances.
      /// This is what proves the `BE = HOST_NATIVE_BE` handoff: on a
      /// big-endian host a wrapper that forwarded the source `BE` to the
      /// delegate would byte-swap the already-native scratch and diverge
      /// from the BE row-stage beyond tolerance. (On this little-endian
      /// host it exercises the BE WIRE path and passes within tolerance.)
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_be_within_tolerance_of_rowstage_be() {
        let (y, uv) = ramp();
        let (yb, uvb) = (as_be(&y), as_be(&uv));
        let mut n_rgb = vec![0u8; OUT * OUT * 3];
        let mut n_rgb16 = vec![0u16; OUT * OUT * 3];
        let mut n_luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(true)
          .with_rgb(&mut n_rgb)
          .unwrap()
          .with_rgb_u16(&mut n_rgb16)
          .unwrap()
          .with_luma(&mut n_luma)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&yb, &uvb), FR, M, &mut sink).unwrap();
        }
        let (r_rgb, r_rgb16, r_luma) = rowstage_be_run(&y, &uv);

        assert_eq!(n_luma, r_luma, "BE luma must be bit-identical across tiers");

        let max_u8 = n_rgb
          .iter()
          .zip(&r_rgb)
          .map(|(&a, &b)| a.abs_diff(b))
          .max()
          .unwrap_or(0);
        assert!(
          max_u8 <= TOL_U8,
          "BE u8 native-vs-rowstage max delta {max_u8} exceeds tolerance {TOL_U8}"
        );

        let tol_u16: u16 = (TOL_U8 as u16) << ($bits - 8);
        let max_u16 = n_rgb16
          .iter()
          .zip(&r_rgb16)
          .map(|(&a, &b)| a.abs_diff(b))
          .max()
          .unwrap_or(0);
        assert!(
          max_u16 <= tol_u16,
          "BE u16 native-vs-rowstage max delta {max_u16} exceeds tolerance {tol_u16}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_matches_inter_area_oracle() {
        // cv2 INTER_AREA parity for luma: the area-bin of the DE-PACKED
        // logical Y plane, narrowed. This is the guard for the Y de-pack
        // (`raw >> (16 - BITS)`) — a missing shift would bin the packed
        // codes and the narrowing would diverge.
        let (y, uv) = ramp();
        let (_, _, n_luma) = native_run(&y, &uv);
        let y_ref = block_mean_2x2_u16(&logical_y(&y));
        let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_eq!(
          n_luma, luma_ref,
          "native luma must equal the INTER_AREA bin of the de-packed Y"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_gamut_native_vs_rowstage_pinned() {
        let (y, uv) = out_of_gamut();
        let (n_rgb, _, n_luma) = native_run(&y, &uv);
        let (r_rgb, _, r_luma) = rowstage_run(&y, &uv);
        assert_eq!(n_luma, r_luma, "luma stays bit-identical out of gamut");
        let max_u8 = n_rgb
          .iter()
          .zip(&r_rgb)
          .map(|(&a, &b)| a.abs_diff(b))
          .max()
          .unwrap_or(0);
        assert!(
          max_u8 > TOL_U8,
          "crafted out-of-gamut case must diverge beyond the in-gamut \
           tolerance {TOL_U8}, got {max_u8}"
        );
        assert!(
          max_u8 < u8::MAX,
          "out-of-gamut delta stays bounded, got {max_u8}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn uniform_gray_leaves_color_unchanged() {
        // Independent-kernel guard (#37): a uniform-gray downscale must
        // leave every colour output equal to the direct conversion of a
        // single gray pixel.
        let (y, uv) = uniform_gray((MID as u32 + (MASK as u32 / 8)) as u16 & MASK);
        let (n_rgb, n_rgb16, _) = native_run(&y, &uv);

        let (yl, uvl) = (as_le(&y), as_le(&uv));
        let mut ref_rgb = vec![0u8; SRC * SRC * 3];
        let mut ref_rgb16 = vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut ref_rgb)
            .unwrap()
            .with_rgb_u16(&mut ref_rgb16)
            .unwrap();
          $walker(&frame(&yl, &uvl), FR, M, &mut sink).unwrap();
        }
        for px in n_rgb.chunks_exact(3) {
          assert_eq!(px, &ref_rgb[..3], "uniform-gray u8 colour drifted");
        }
        for px in n_rgb16.chunks_exact(3) {
          assert_eq!(px, &ref_rgb16[..3], "uniform-gray u16 colour drifted");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn u8_and_u16_color_are_independent_bins() {
        // Independent-kernel guard (#37): narrowing the binned u16 colour
        // to u8 (`>> (BITS - 8)`) diverges from the genuine u8 bin over a
        // varying ramp.
        let (y, uv) = ramp();
        let (n_rgb, n_rgb16, _) = native_run(&y, &uv);
        let narrowed: Vec<u8> = n_rgb16.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_ne!(
          n_rgb, narrowed,
          "u8 colour must be an independent bin, not a narrowed u16 bin"
        );
      }

      #[test]
      fn no_outputs_is_a_no_op() {
        let (y, uv) = ramp();
        let (yl, uvl) = (as_le(&y), as_le(&uv));
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true);
        $walker(&frame(&yl, &uvl), FR, M, &mut sink).unwrap();
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resets_join_across_frames() {
        let (y1, uv1) = ramp();
        let invert = |p: &[u16]| -> Vec<u16> {
          p.iter()
            .map(|&x| (MASK.wrapping_sub(x >> SHIFT) & MASK) << SHIFT)
            .collect()
        };
        let (y2, uv2) = (invert(&y1), invert(&uv1));
        let (y1l, uv1l) = (as_le(&y1), as_le(&uv1));
        let (y2l, uv2l) = (as_le(&y2), as_le(&uv2));
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(true)
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&y1l, &uv1l), FR, M, &mut sink).unwrap();
          $walker(&frame(&y2l, &uv2l), FR, M, &mut sink).unwrap();
        }
        let y_ref = block_mean_2x2_u16(&logical_y(&y2));
        let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "join did not reset between frames");
      }

      // ---- atomicity ----------------------------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_sequence_first_row_rejected_and_does_not_poison_retry() {
        let (y, uv) = ramp();
        let mut luma = vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Row 3's vertically-shared chroma row is `3 / 2 == 1`; the
        // interleaved chroma row is `SRC` u16 wide.
        let (yr, cr) = (3 * SRC, 1 * SRC);
        let err = sink
          .process($row::new(&y[yr..yr + SRC], &uv[cr..cr + SRC], 3, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "expected OutOfSequenceRow, got {err:?}"
        );
        // The rejected first row stored NO frozen-output snapshot, so
        // attaching a NEW output and retrying row 0 must succeed.
        let mut rgb = vec![0u8; OUT * OUT * 3];
        sink.set_rgb(&mut rgb).unwrap();
        sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect("row 0 must succeed after a rejected out-of-sequence first row");
      }

      /// A mid-frame output-set change on a chroma-bearing even row must be
      /// rejected by the join's frozen-output preflight BEFORE the wrapper
      /// de-pack scratch alloc — `ResampleOutputsChanged`, never
      /// `AllocationFailed`. The failpoint is armed on the reserve the
      /// changed row WOULD reach: with the preflight first the frozen check
      /// fires and the failpoint is never consumed; with the alloc ordered
      /// ahead of the frozen check (the bug) the armed reserve refuses as
      /// AllocationFailed.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn frozen_mid_frame_change_rejected_before_scratch_alloc() {
        let (y, uv) = ramp();
        let mut luma = vec![0u8; OUT * OUT];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Luma-only rows 0 and 1 freeze a luma-only output set (Y scratch
        // grows, no colour).
        for r in 0..2 {
          let cr = (r / 2) * SRC;
          sink
            .process($row::new(
              &y[r * SRC..(r + 1) * SRC],
              &uv[cr..cr + SRC],
              r,
              M,
              FR,
            ))
            .expect("luma-only rows freeze a luma-only output set");
        }
        // Attach u16 colour mid-frame, changing the output set, and arm the
        // wrapper scratch failpoint on the reserve the changed row reaches.
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_p0xx_alloc_failure();
        let err = sink
          .process($row::new(&y[2 * SRC..3 * SRC], &uv[SRC..2 * SRC], 2, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "mid-frame output change must reject as ResampleOutputsChanged \
           before the scratch alloc, got {err:?}"
        );
        assert!(
          rgb_u16.iter().all(|&b| b == 0),
          "rejected mid-frame-change row touched the new colour output"
        );
        // The failpoint is single-shot. It must NOT have been consumed:
        // prove it via a fresh in-sequence colour row that DOES fire it.
        let mut rgb_u16b = vec![0u16; OUT * OUT * 3];
        let mut sink2 = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_rgb_u16(&mut rgb_u16b)
        .unwrap();
        let err2 = sink2
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err2,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "armed failpoint must still be live (unconsumed by the rejected \
           mid-frame-change row) and fire on the first in-sequence colour \
           reserve, got {err2:?}"
        );
      }

      /// Sequence rejection after a RECOVERABLE wrapper-scratch allocation
      /// failure: because the wrapper now runs the COMPARE-ONLY preflight (the
      /// commit is the delegate's), a de-pack failure on an in-sequence colour
      /// row 0 leaves `resample_outputs` UNFROZEN and the join's Y stream still
      /// at row 0, so a later OUT-OF-SEQUENCE row is caught by the pre-compare
      /// first-row sequence check and must reject as the deterministic
      /// `OutOfSequenceRow`, never `AllocationFailed`.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn oos_after_recoverable_alloc_failure_rejected_before_scratch_alloc() {
        let (y, uv) = ramp();
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
        // Step 1 — a RECOVERABLE wrapper-scratch failure on the in-sequence
        // colour row 0. The compare-only preflight clears WITHOUT freezing, then
        // the armed (Y) scratch reserve refuses: AllocationFailed. `resample_outputs`
        // stays None (the delegate owns the commit, never reached) and the join's
        // Y stream has not been fed (the grow precedes it), so it still expects
        // row 0.
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_p0xx_alloc_failure();
        let err0 = sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err0,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the recoverable scratch failure on row 0 must surface \
           AllocationFailed, got {err0:?}"
        );
        // Step 2 — RE-ARM, then feed an OUT-OF-SEQUENCE row (idx 2; the
        // join's Y stream still expects 0). Outputs are unfrozen (step 1 never
        // committed), so the preflight's pre-compare first-row sequence check is
        // the gate; it must reject as OutOfSequenceRow BEFORE the re-armed
        // scratch reserve.
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_p0xx_alloc_failure();
        let err2 = sink
          .process($row::new(&y[2 * SRC..3 * SRC], &uv[SRC..2 * SRC], 2, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err2,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "an out-of-sequence row after a recoverable scratch failure must \
           reject as OutOfSequenceRow (the pre-compare sequence check), \
           never AllocationFailed, got {err2:?}"
        );
        assert!(
          rgb_u16.iter().all(|&b| b == 0),
          "neither the recoverable-failure nor the out-of-sequence row \
           touched the colour output"
        );
        // Step 3 — the failpoint re-armed in step 2 must NOT have been
        // consumed: prove it via a fresh in-sequence colour row 0.
        let mut rgb_u16b = vec![0u16; OUT * OUT * 3];
        let mut sink2 = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_rgb_u16(&mut rgb_u16b)
        .unwrap();
        let err3 = sink2
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err3,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the failpoint re-armed in step 2 must still be live and fire on \
           the first in-sequence colour reserve, got {err3:?}"
        );
      }

      // ---- colour-capability rebuild transactionality (RFC #238) --------

      /// Per-source-row `$row` over the suite's `ramp()` planes (interleaved
      /// chroma row `r / 2`), the shared fixture the rebuild-atomicity tests
      /// below feed row by row. The delegate is the shared
      /// [`yuv420p16_process_native`] core, so these prove the P-format
      /// (semi-planar) caller reaches the same transactional commit.
      fn rebuild_row<'a>(y: &'a [u16], uv: &'a [u16], r: usize) -> $row<'a> {
        let cr = (r / 2) * SRC;
        $row::new(&y[r * SRC..(r + 1) * SRC], &uv[cr..cr + SRC], r, M, FR)
      }

      /// RFC #238 — the 4:2:0 high-bit native delegate is preflight-transactional
      /// for the semi-planar (P-format) caller too. A colour-capability REBUILD
      /// whose CHROMA-PLAN build fails leaves the cached luma-only join AND the
      /// output-set freeze INTACT (uncommitted), so a retry with a changed output
      /// set is accepted — a committed `{luma, hsv}` freeze would reject it.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_colour_capability_rebuild_chroma_oom_is_transactional() {
        let (y, uv) = ramp();
        let (mut hh, mut ss, mut vv) = (
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
        );
        let mut luma = vec![0u8; OUT * OUT];
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        for r in 0..SRC {
          sink
            .process(rebuild_row(&y, &uv, r))
            .expect("luma-only frame builds the native join");
        }
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_chroma_failure();
        let err = sink.process(rebuild_row(&y, &uv, 0)).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the colour-capability rebuild chroma-plan OOM must surface \
           AllocationFailed, got {err:?}"
        );
        assert!(
          sink.native_420_u16.is_some(),
          "the cached luma-only join must survive a failed colour rebuild"
        );
        assert!(
          !sink.resample_outputs_frozen(),
          "a failed rebuild must leave resample_outputs uncommitted"
        );
        sink.set_rgb(&mut rgb).unwrap();
        sink
          .process(rebuild_row(&y, &uv, 0))
          .expect("row 0 with a changed output set must succeed after a rebuild chroma OOM");
        for r in 1..SRC {
          sink
            .process(rebuild_row(&y, &uv, r))
            .expect("the recovered join must process the rest of the frame");
        }
        assert!(
          rgb.iter().any(|&b| b != 0),
          "the recovered join must produce real colour output"
        );
      }

      /// RFC #238 — the semi-planar u8 colour-scratch failpoint companion: the
      /// same rebuild, but the u8 `rgb_scratch` grow (attaching `rgb`) fails.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_colour_capability_rebuild_rgb_scratch_oom_is_transactional() {
        let (y, uv) = ramp();
        let (mut hh, mut ss, mut vv) = (
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
        );
        let mut luma = vec![0u8; OUT * OUT];
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        for r in 0..SRC {
          sink
            .process(rebuild_row(&y, &uv, r))
            .expect("luma-only frame builds the native join");
        }
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink.set_rgb(&mut rgb).unwrap();
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_rgb_scratch_failure();
        let err = sink.process(rebuild_row(&y, &uv, 0)).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the colour-capability rebuild u8-scratch OOM must surface \
           AllocationFailed, got {err:?}"
        );
        assert!(
          sink.native_420_u16.is_some(),
          "the cached luma-only join must survive a failed u8-scratch rebuild"
        );
        assert!(
          !sink.resample_outputs_frozen(),
          "a failed u8-scratch rebuild must leave resample_outputs uncommitted"
        );
        sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
        sink
          .process(rebuild_row(&y, &uv, 0))
          .expect("row 0 with a changed output set must succeed after a u8-scratch OOM");
        for r in 1..SRC {
          sink
            .process(rebuild_row(&y, &uv, r))
            .expect("the recovered join must process the rest of the frame");
        }
        assert!(
          rgb.iter().any(|&b| b != 0),
          "the recovered join must produce real colour output"
        );
      }

      /// RFC #238 — a LATER (u16 colour-scratch) pre-feed grow failing is atomic
      /// too for the semi-planar caller: with BOTH `rgb` and `rgb_u16` attached
      /// the u8 scratch grows first, then the u16 grow refuses. The cached join +
      /// freeze + output stay untouched (the already-grown u8 scratch keeps its
      /// capacity), so a changed-output retry is accepted and drives real output.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_colour_capability_rebuild_rgb_u16_scratch_oom_is_transactional() {
        let (y, uv) = ramp();
        let (mut hh, mut ss, mut vv) = (
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
        );
        let mut luma = vec![0u8; OUT * OUT];
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        for r in 0..SRC {
          sink
            .process(rebuild_row(&y, &uv, r))
            .expect("luma-only frame builds the native join");
        }
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink.set_rgb(&mut rgb).unwrap();
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_rgb_u16_scratch_failure();
        let err = sink.process(rebuild_row(&y, &uv, 0)).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the u16 colour-scratch OOM (after the u8 scratch grew) must surface \
           AllocationFailed, got {err:?}"
        );
        assert!(
          sink.native_420_u16.is_some(),
          "the cached luma-only join must survive a failed u16-scratch rebuild"
        );
        assert!(
          !sink.resample_outputs_frozen(),
          "a failed u16-scratch rebuild must leave resample_outputs uncommitted"
        );
        sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
        sink
          .process(rebuild_row(&y, &uv, 0))
          .expect("row 0 with a changed output set must succeed after a u16-scratch OOM");
        for r in 1..SRC {
          sink
            .process(rebuild_row(&y, &uv, r))
            .expect("the recovered join must process the rest of the frame");
        }
        assert!(
          rgb.iter().any(|&b| b != 0) && rgb_u16.iter().any(|&b| b != 0),
          "the recovered join must produce real u8 and u16 colour output"
        );
      }

      // ---- frozen native-vs-row-stage route -----------------------------

      /// Flipping `set_native(true) -> false` mid-frame must reject as the
      /// deterministic `NativeRouteChanged` BEFORE either tier consumes the
      /// row — NOT a silently mixed frame, and NOT the wrong error
      /// (`OutOfSequenceRow` from the fresh row-stage stream, which is what
      /// surfaces with the route guard removed).
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_to_rowstage_route_flip_mid_frame_rejected() {
        let (y, uv) = ramp();
        let mut luma = vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Row 0 freezes the route = native (chroma row `0 / 2 == 0`).
        sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect("native row 0 freezes the route and succeeds");
        // Flip to the row-stage tier and feed the next in-sequence row.
        sink.set_native(false);
        let err = sink
          .process($row::new(&y[SRC..2 * SRC], &uv[..SRC], 1, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "a native -> row-stage mid-frame route flip must reject as \
           NativeRouteChanged, got {err:?}"
        );
      }

      /// The reverse flip `set_native(false) -> true` mid-frame must reject
      /// identically — the guard catches BOTH directions.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rowstage_to_native_route_flip_mid_frame_rejected() {
        let (y, uv) = ramp();
        let mut luma = vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(false)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Row 0 freezes the route = row-stage.
        sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect("row-stage row 0 freezes the route and succeeds");
        // Flip to the native tier and feed the next in-sequence row.
        sink.set_native(true);
        let err = sink
          .process($row::new(&y[SRC..2 * SRC], &uv[..SRC], 1, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "a row-stage -> native mid-frame route flip must reject as \
           NativeRouteChanged, got {err:?}"
        );
      }

      /// No false rejection: a frame whose route stays constant runs to
      /// completion, and the per-frame reset lets the NEXT frame pick the
      /// OTHER tier — frame 1 native, `begin_frame`, frame 2 row-stage, both
      /// succeed (the route guard is reset, not sticky across frames).
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn route_constant_succeeds_and_resets_across_frames() {
        let (y, uv) = ramp();
        let (yl, uvl) = (as_le(&y), as_le(&uv));
        let mut luma = vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        // Frame 1: native, route constant across every row — no false
        // rejection.
        $walker(&frame(&yl, &uvl), FR, M, &mut sink).unwrap();
        // Frame 2: flip to row-stage for the WHOLE frame after begin_frame.
        // The reset cleared the frozen route, so this is allowed.
        sink.set_native(false);
        $walker(&frame(&yl, &uvl), FR, M, &mut sink)
          .expect("a new frame may pick the other tier; the route reset per frame");
      }

      /// A NO-OUTPUT resampled row under native consumes no stream state and
      /// must NOT freeze the route: a later first OUTPUT-bearing row under
      /// the OTHER tier (`set_native(false)`) must SUCCEED. The route only
      /// freezes on an output-bearing row a tier accepts. With the old
      /// snapshot-before-preflight code the no-output call would have frozen
      /// the route to native and this would wrongly reject as
      /// `NativeRouteChanged`.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn no_output_call_then_output_bearing_flip_succeeds() {
        let (y, uv) = ramp();
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true);
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Row 0 with NO outputs attached: a no-op under native. It must not
        // freeze the route.
        sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect("a no-output native row is a no-op");
        // Attach an output and flip to the row-stage tier, then feed the
        // first OUTPUT-bearing row. Because the no-output call consumed no
        // state, the row-stage tier is free to take the frame's first real
        // row.
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        sink.set_native(false);
        sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect(
            "the first output-bearing row may pick the other tier after a \
             no-output call left the route unfrozen",
          );
      }

      /// An output-bearing OUT-OF-SEQUENCE first row under native is rejected
      /// by the join preflight (`OutOfSequenceRow`) and must leave the route
      /// UNFROZEN — so a valid in-sequence first row under the OTHER tier
      /// (`set_native(false)`) then SUCCEEDS. With the old
      /// snapshot-before-preflight code the rejected OOS call would have
      /// frozen the route to native and the other-tier row would wrongly
      /// reject as `NativeRouteChanged` (the recoverable/no-state-on-reject
      /// contract).
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn oos_first_row_reject_does_not_freeze_route() {
        let (y, uv) = ramp();
        let mut luma = vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Output-bearing OUT-OF-SEQUENCE first row (idx 3; the join expects
        // 0). Its shared chroma row is `3 / 2 == 1`, `SRC` u16 wide.
        let (yr, cr) = (3 * SRC, 1 * SRC);
        let err = sink
          .process($row::new(&y[yr..yr + SRC], &uv[cr..cr + SRC], 3, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "expected OutOfSequenceRow on the OOS first row, got {err:?}"
        );
        // The rejected OOS call left the route unfrozen, so a valid
        // in-sequence first row under the OTHER tier must succeed.
        sink.set_native(false);
        sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect(
            "a valid in-sequence first row under the other tier must succeed \
             after a rejected out-of-sequence row left the route unfrozen",
          );
      }

      /// A NO-OUTPUT call arriving AFTER an output-bearing row froze the
      /// route must be a TRUE no-op — route-invisible — even when its
      /// `set_native` is FLIPPED relative to the frozen route: it must
      /// return `Ok` (not `NativeRouteChanged`) and leave the frozen route
      /// untouched. The route CHECK and the SET both gate on `need_output`,
      /// so a no-output call neither checks nor freezes; this preserves the
      /// invariant "a no-output call is a true no-op regardless of row
      /// index". With the CHECK's `need_output` gate removed the ungated
      /// CHECK sees `frozen(native) != row-stage` and wrongly rejects.
      ///
      /// There is no public API to DETACH an output from a sink that has
      /// already frozen its route (output buffers only transition
      /// `None -> Some`), so a "no-output call after a frozen route" cannot
      /// be staged on one sink through the builder alone. We build the
      /// faithful equivalent by setting the per-sink `frozen_native_route`
      /// field directly to the value an accepted output-bearing native first
      /// row stores (`Some(true)` = route frozen to native — see the SET
      /// site in `p0xx.rs`), then issuing the flipped no-output call. This is
      /// the same white-box reach the atomicity tests use for
      /// `arm_p0xx_alloc_failure`.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn no_output_call_after_frozen_route_is_a_noop() {
        let (y, uv) = ramp();
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true);
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Freeze the route to native exactly as an accepted output-bearing
        // native first row does (the SET site stores `Some(true)`).
        sink.frozen_native_route = Some(true);
        // A NO-OUTPUT row (no outputs attached -> `need_output` false) with
        // the route FLIPPED to row-stage. Gated on `need_output`, the CHECK
        // is skipped, so this is a true no-op (Ok); a no-output call is
        // route-invisible. Both tiers short-circuit a no-output call before
        // touching any stream state, so it consumes nothing.
        sink.set_native(false);
        sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect(
            "a no-output call after a frozen route must be a true no-op, not \
             NativeRouteChanged",
          );
        // The no-output call must not have moved the frozen route.
        assert_eq!(
          sink.frozen_native_route,
          Some(true),
          "a no-output call must leave the frozen route unchanged",
        );
        // Proof the route is STILL frozen to native and the no-output call
        // consumed no stream state: an OUTPUT-bearing row 0 under the
        // ORIGINAL (native) route is in-sequence and succeeds...
        let mut luma = vec![0u8; OUT * OUT];
        sink.set_native(true);
        sink.set_luma(&mut luma).unwrap();
        sink
          .process($row::new(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect("an output-bearing row under the original native route succeeds");
        // ...while an OUTPUT-bearing flip to the OTHER route now rejects,
        // confirming the frozen route is native (the no-output call did not
        // change it to row-stage).
        sink.set_native(false);
        let err = sink
          .process($row::new(&y[SRC..2 * SRC], &uv[..SRC], 1, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "an output-bearing flip after the frozen route stayed native must \
           reject as NativeRouteChanged, got {err:?}"
        );
      }
    }
  };
}

p0xx_high_bit_native_suite!(
  p010,
  P010LeFrame,
  P010BeFrame,
  P010,
  P010Row,
  p010_to,
  p010_to_endian,
  10,
);
p0xx_high_bit_native_suite!(
  p012,
  P012LeFrame,
  P012BeFrame,
  P012,
  P012Row,
  p012_to,
  p012_to_endian,
  12,
);
p0xx_high_bit_native_suite!(
  p016,
  P016LeFrame,
  P016BeFrame,
  P016,
  P016Row,
  p016_to,
  p016_to_endian,
  16,
);
