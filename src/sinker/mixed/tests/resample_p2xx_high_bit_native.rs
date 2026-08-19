//! Fused-downscale coverage for the high-bit **semi-planar 4:2:2** YUV NATIVE
//! fast tier — `P210` / `P212` / `P216` (LE + BE wire), the `u16` 4:2:2
//! semi-planar twin of the 8-bit
//! [`semi_planar_process_native_non420`](crate::sinker::mixed::semi_planar_8bit)
//! (Nv16) and the non-4:2:0 sibling of the high-bit 4:2:0
//! [`p0xx_process_native`](crate::sinker::mixed::subsampled_4_2_0_high_bit).
//! Reuses the high-bit non-4:2:0 PLANAR join
//! ([`yuv_planar16_process_native`](crate::sinker::mixed::planar_high_bit_native))
//! after de-interleaving + DE-PACKING (`raw >> (16 - BITS)`) the high-bit-packed
//! Y and interleaved UV wire planes into wrapper-owned host-native LOGICAL u16
//! scratch.
//!
//! The native tier bins those planes straight to the output grid and converts
//! ONCE per output row at output width (4:4:4 kernels) — vs the row-stage tier
//! ([`packed_yuv422_triple_resample`](crate::sinker::mixed::packed_yuv422_triple_resample)),
//! which converts each source row at source width then bins. The tiers differ
//! in colour SEMANTICS (native averages in YUV then converts; row-stage
//! converts then averages in RGB), so native is NOT byte-identical to row-stage
//! — only within a small tolerance in-gamut. Luma is bit-identical (both bin
//! the same de-packed native Y then narrow `>> (BITS - 8)`).
//!
//! Per format + depth (LE + BE):
//! - `native_equals_bin_then_convert_oracle`: the GROUND-TRUTH check — native
//!   output EXACTLY equals an independent bin-then-convert oracle that
//!   de-interleaves + de-packs the P-format, area-bins each plane to OUTPUT
//!   resolution (Y 2:1 x 2:1, chroma 2:1 horizontal only — 4:2:2), then converts
//!   ONCE through an identity-resolution high-bit `Yuv422pN` sink (the SAME
//!   native-depth kernels + `(1 << BITS) - 1` clamp the native tier finalizes
//!   with). The luma oracle clamps INDEPENDENTLY of the sink.
//! - `native_equals_planar_twin`: native P21x == native `Yuv422pN` on the
//!   de-interleaved + de-packed planes — the strong cross-check that the
//!   semi-planar wrapper is a faithful de-interleave in front of the planar join.
//! - `native_within_tolerance_of_rowstage` / `_be_*`: the cv2 INTER_AREA parity
//!   bound + the `BE = HOST_NATIVE_BE` handoff proof.
//! - `native_luma_clamps_overrange_y` / `rowstage_luma_clamps_overrange_y`: the
//!   native-depth luma clamp (`min((1 << BITS) - 1) >> (BITS - 8)`), BOTH tiers.
//! - `luma_only_native_skips_chroma_planning`: a luma-only sink plans no chroma.
//! - the four atomicity regressions on [`arm_p2xx_alloc_failure`] + the route
//!   freeze guard.

use crate::{
  KernelMatrix,
  PixelSink,
  frame::*,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  // Bin-then-convert oracle markers (chroma binned to full output width → 4:4:4
  // sink) + twin-parity markers (de-interleaved + de-packed planes → 4:2:2
  // native sink at source resolution). Both run the SAME native-depth kernels +
  // `(1 << BITS) - 1` clamp the semi-planar native tier delegates to.
  source::{
    Yuv422p10, Yuv422p12, Yuv422p16, Yuv444p10, Yuv444p12, Yuv444p16, yuv422p10_to, yuv422p12_to,
    yuv422p16_to, yuv444p10_to, yuv444p12_to, yuv444p16_to,
  },
};

const SRC: usize = 8;
const CW: usize = SRC / 2;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// In-gamut per-channel u8 tolerance between the native and row-stage tiers.
/// The two average in different domains (YUV vs RGB) and round independently
/// per output pixel; native correctness itself is pinned EXACTLY by
/// `native_equals_bin_then_convert_oracle`, this bound only documents the
/// row-stage semantic gap. Matches the planar high-bit non-4:2:0 suite's bound.
const TOL_U8: u8 = 5;

/// Exact integer-ratio area mean (round-half-up) of an `in_w x in_h` u16 plane
/// down to `OUT x OUT`, binning each axis by its own ratio.
fn bin_to_out(plane: &[u16], in_w: usize, in_h: usize) -> Vec<u16> {
  let (rx, ry) = (in_w / OUT, in_h / OUT);
  let denom = (rx * ry) as u32;
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..ry {
        for dx in 0..rx {
          s += plane[(oy * ry + dy) * in_w + ox * rx + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + denom / 2) / denom) as u16;
    }
  }
  out
}

/// Re-encode a host-native u16 slice as host-independent LE-wire byte storage
/// (the `*LeFrame` plane contract); a no-op on LE, a byte swap on BE.
fn as_le(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}
/// Re-encode a host-native u16 slice as host-independent BE-wire byte storage
/// (the `*BeFrame` plane contract).
fn as_be(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

macro_rules! p2xx_high_bit_native_suite {
  (
    $mod:ident, $frame_le:ident, $frame_be:ident, $marker:ident, $row:ident,
    $walker:ident, $walker_be:ident,
    // The bin-then-convert oracle bins chroma to FULL output resolution, so it
    // feeds a 4:4:4 sink (`Yuv444pN`). The twin-parity check feeds the
    // de-interleaved + de-packed planes at SOURCE resolution to the matching
    // 4:2:2 native sink (`Yuv422pN`).
    $oracle_marker:ident, $oracle_frame:ident, $oracle_walker:ident,
    $twin_marker:ident, $twin_frame:ident, $twin_walker:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $row, $walker, $walker_be};

      const SHIFT: u32 = 16 - $bits;
      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = (1u16 << ($bits - 1));

      /// Per-pixel logical Y ramp + per-chroma-sample logical U / V ramp, kept
      /// near the legal-range middle so the converted RGB stays in gamut and
      /// the native-vs-rowstage delta is per-pixel rounding. HIGH-BIT-PACKED
      /// (`logical << (16 - BITS)`) into a full-width Y plane and a
      /// half-width / **full-height** interleaved `U,V,U,V…` plane (4:2:2).
      fn ramp() -> (Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut uv = vec![0u16; CW * SRC * 2];
        for i in 0..SRC * SRC {
          let logical = (MID as u32 + ((i as u32 * 37) % (MASK as u32 / 4))) as u16 & MASK;
          y[i] = logical << SHIFT;
        }
        for i in 0..CW * SRC {
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
          vec![(MID & MASK) << SHIFT; CW * SRC * 2],
        )
      }

      /// Crafted VARYING illegal-chroma fixture: extreme alternating chroma over
      /// a super-black->super-white Y ramp, high-bit-packed — many 2x2 blocks
      /// straddle the RGB clamp, where native (average-in-YUV) and row-stage
      /// (convert-then-average) genuinely diverge.
      fn out_of_gamut() -> (Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut uv = vec![0u16; CW * SRC * 2];
        for i in 0..SRC * SRC {
          let logical = ((i as u32 * MASK as u32) / (SRC * SRC) as u32) as u16 & MASK;
          y[i] = logical << SHIFT;
        }
        for i in 0..CW * SRC {
          let hi = i % 2 == 0;
          uv[2 * i] = if hi { MASK } else { 0 } << SHIFT;
          uv[2 * i + 1] = if hi { 0 } else { MASK } << SHIFT;
        }
        (y, uv)
      }

      /// Full-scale-Y fixture: every Y at the native max `MASK`, high-bit-packed
      /// (`MASK << SHIFT`). Unlike the LOW-packed planar family, an MSB-packed
      /// P-format sample CANNOT exceed `MASK` (the de-pack `wire >> (16 - BITS)`
      /// caps at `MASK`), and an area mean of `<= MASK` stays `<= MASK` — so the
      /// achievable boundary is the legal max. Exercises the native-depth luma
      /// clamp at that boundary: `min(MASK) >> (BITS - 8)` must saturate, never
      /// wrap. (Genuine beyond-max overrange is impossible to encode here; the
      /// reused planar join's clamp for overrange is covered by the planar
      /// high-bit native suite's low-packed fixture.) Chroma stays legal.
      fn full_scale_luma() -> (Vec<u16>, Vec<u16>) {
        let (_, uv) = ramp();
        (vec![MASK << SHIFT; SRC * SRC], uv)
      }

      fn frame<'a>(y: &'a [u16], uv: &'a [u16]) -> $frame_le<'a> {
        // 4:2:2 interleaved UV stride = `2 * (SRC / 2)` = `SRC` u16.
        $frame_le::new(y, uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32)
      }
      fn frame_be<'a>(y: &'a [u16], uv: &'a [u16]) -> $frame_be<'a> {
        $frame_be::new(y, uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32)
      }

      /// De-pack the high-bit-packed Y plane into the logical native Y.
      fn logical_y(y: &[u16]) -> Vec<u16> {
        y.iter().map(|&s| s >> SHIFT).collect()
      }
      /// De-interleave + de-pack the packed UV plane into separate logical
      /// `U` / `V` planes (`CW x SRC` each) — the planes the native wrapper
      /// hands the planar join.
      fn deinterleave_depack(uv: &[u16]) -> (Vec<u16>, Vec<u16>) {
        let mut u = vec![0u16; CW * SRC];
        let mut v = vec![0u16; CW * SRC];
        for (i, pair) in uv.chunks_exact(2).enumerate() {
          u[i] = pair[0] >> SHIFT;
          v[i] = pair[1] >> SHIFT;
        }
        (u, v)
      }

      /// Drive the LE source through a tier for the full output set (u8 RGB +
      /// u16 RGB + luma). `native` toggles the bin-then-convert native fast tier
      /// vs the convert-then-bin row-stage tier.
      fn run(y: &[u16], uv: &[u16], native: bool) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
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
          .with_native(native)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&yl, &uvl), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the BE source through the native tier (the host-native-endian
      /// guard reference).
      fn native_be_run(y: &[u16], uv: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
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
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&yb, &uvb), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the row-stage tier (BE) — the correct host-independent reference
      /// (it de-interleaves BE-wire bytes to host-native before converting).
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
          $walker_be::<_, true>(&frame_be(&yb, &uvb), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// The bin-then-convert oracle: de-interleave + de-pack the P-format,
      /// area-bin every plane to OUTPUT resolution (Y from `SRC x SRC`, chroma
      /// from `CW x SRC` — horizontal-only subsample), then convert the
      /// full-output-width host-native planes ONCE through an identity-resolution
      /// high-bit `Yuv422pN` sink. The luma oracle clamps INDEPENDENTLY of the
      /// sink's luma path.
      fn oracle(y: &[u16], uv: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let yl = logical_y(y);
        let (u, v) = deinterleave_depack(uv);
        let yb = bin_to_out(&yl, SRC, SRC);
        let ub = bin_to_out(&u, CW, SRC);
        let vb = bin_to_out(&v, CW, SRC);
        let (ye, ue, ve) = (as_le(&yb), as_le(&ub), as_le(&vb));
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        {
          // Chroma is binned to FULL output width, so feed a 4:4:4 sink.
          let mut sink = MixedSinker::<$oracle_marker>::new(OUT, OUT)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb_u16)
            .unwrap();
          let f = $oracle_frame::try_new(
            &ye, &ue, &ve, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
          )
          .unwrap();
          $oracle_walker(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let luma: Vec<u8> = yb
          .iter()
          .map(|&by| (by.min(MASK) >> ($bits - 8)) as u8)
          .collect();
        (rgb, rgb_u16, luma)
      }

      /// Native `Yuv422pN` reference on the de-interleaved + de-packed planes —
      /// the twin-parity cross-check (same join, fed planar instead of through
      /// the semi-planar de-interleave).
      fn planar_twin_native(y: &[u16], uv: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let yl = logical_y(y);
        let (u, v) = deinterleave_depack(uv);
        let (ye, ue, ve) = (as_le(&yl), as_le(&u), as_le(&v));
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$twin_marker, AreaResampler>::with_resampler(
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
          let f = $twin_frame::try_new(
            &ye, &ue, &ve, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
          )
          .unwrap();
          $twin_walker(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      fn max_delta_u8(a: &[u8], b: &[u8]) -> u8 {
        a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y)).max().unwrap_or(0)
      }
      fn max_delta_u16(a: &[u16], b: &[u16]) -> u16 {
        a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y)).max().unwrap_or(0)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_equals_bin_then_convert_oracle() {
        let (y, uv) = ramp();
        let (n_rgb, n_rgb16, n_luma) = run(&y, &uv, true);
        let (o_rgb, o_rgb16, o_luma) = oracle(&y, &uv);
        assert_eq!(n_rgb, o_rgb, "u8 rgb must equal the bin-then-convert oracle");
        assert_eq!(
          n_rgb16, o_rgb16,
          "u16 rgb must equal the bin-then-convert oracle (clamp-for-clamp)"
        );
        assert_eq!(n_luma, o_luma, "luma must equal the binned-then-narrowed Y");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_equals_planar_twin() {
        // The semi-planar wrapper IS a de-interleave + de-pack in front of the
        // planar join, so its output must be bit-identical to feeding the
        // de-interleaved + de-packed planes straight to the native Yuv422pN.
        let (y, uv) = ramp();
        let (n_rgb, n_rgb16, n_luma) = run(&y, &uv, true);
        let (t_rgb, t_rgb16, t_luma) = planar_twin_native(&y, &uv);
        assert_eq!(n_rgb, t_rgb, "u8 rgb must match the planar twin");
        assert_eq!(n_rgb16, t_rgb16, "u16 rgb must match the planar twin");
        assert_eq!(n_luma, t_luma, "luma must match the planar twin");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_within_tolerance_of_rowstage() {
        let (y, uv) = ramp();
        let (n_rgb, n_rgb16, n_luma) = run(&y, &uv, true);
        let (r_rgb, r_rgb16, r_luma) = run(&y, &uv, false);
        assert_eq!(n_luma, r_luma, "luma must be bit-identical across tiers");
        let d_u8 = max_delta_u8(&n_rgb, &r_rgb);
        assert!(
          d_u8 <= TOL_U8,
          "u8 native-vs-rowstage max delta {d_u8} exceeds tolerance {TOL_U8}"
        );
        let tol_u16: u16 = (TOL_U8 as u16) << ($bits - 8);
        let d_u16 = max_delta_u16(&n_rgb16, &r_rgb16);
        assert!(
          d_u16 <= tol_u16,
          "u16 native-vs-rowstage max delta {d_u16} exceeds tolerance {tol_u16}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_be_matches_native_le() {
        // The native tier de-interleaves + de-packs the wire planes to
        // host-native LOGICAL u16 BEFORE binning, so BE and LE sources produce
        // identical output.
        let (y, uv) = ramp();
        let le = run(&y, &uv, true);
        let be = native_be_run(&y, &uv);
        assert_eq!(be.0, le.0, "BE u8 colour must match LE");
        assert_eq!(be.1, le.1, "BE u16 colour must match LE");
        assert_eq!(be.2, le.2, "BE luma must match LE");
      }

      /// The host-native-endian regression: BE native vs the correct BE
      /// row-stage reference, within the same tolerances + luma bit-identical.
      /// Proves the `BE = HOST_NATIVE_BE` handoff — a wrapper forwarding the
      /// source `BE` to the delegate would byte-swap the already-native scratch
      /// on a big-endian host.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_be_within_tolerance_of_rowstage_be() {
        let (y, uv) = ramp();
        let (n_rgb, n_rgb16, n_luma) = native_be_run(&y, &uv);
        let (r_rgb, r_rgb16, r_luma) = rowstage_be_run(&y, &uv);
        assert_eq!(n_luma, r_luma, "BE luma must be bit-identical across tiers");
        let d_u8 = max_delta_u8(&n_rgb, &r_rgb);
        assert!(
          d_u8 <= TOL_U8,
          "BE u8 native-vs-rowstage max delta {d_u8} exceeds tolerance {TOL_U8}"
        );
        let tol_u16: u16 = (TOL_U8 as u16) << ($bits - 8);
        let d_u16 = max_delta_u16(&n_rgb16, &r_rgb16);
        assert!(
          d_u16 <= tol_u16,
          "BE u16 native-vs-rowstage max delta {d_u16} exceeds tolerance {tol_u16}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_matches_inter_area_oracle() {
        // cv2 INTER_AREA parity for luma: the area-bin of the DE-PACKED logical
        // Y, narrowed. Guards the Y de-pack (`raw >> (16 - BITS)`).
        let (y, uv) = ramp();
        let (_, _, n_luma) = run(&y, &uv, true);
        let y_ref = bin_to_out(&logical_y(&y), SRC, SRC);
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
      fn native_luma_clamps_full_scale_y() {
        // A full-scale binned Y must SATURATE through the native-depth clamp +
        // `>> (BITS - 8)` narrowing, never wrap. The oracle clamps independently
        // of the sink.
        let (y, uv) = full_scale_luma();
        let (_, _, n_luma) = run(&y, &uv, true);
        let yb = bin_to_out(&logical_y(&y), SRC, SRC);
        let expect: Vec<u8> = yb
          .iter()
          .map(|&by| (by.min(MASK) >> ($bits - 8)) as u8)
          .collect();
        assert_eq!(
          n_luma, expect,
          "full-scale binned Y must clamp to native-max before narrowing, not wrap"
        );
        let sat = (MASK >> ($bits - 8)) as u8;
        assert!(
          n_luma.iter().all(|&l| l == sat),
          "all full-scale luma must saturate to {sat}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rowstage_luma_clamps_full_scale_y() {
        // Same clamp on the ROW-STAGE (with_native(false)) path.
        let (y, uv) = full_scale_luma();
        let (_, _, r_luma) = run(&y, &uv, false);
        let yb = bin_to_out(&logical_y(&y), SRC, SRC);
        let expect: Vec<u8> = yb
          .iter()
          .map(|&by| (by.min(MASK) >> ($bits - 8)) as u8)
          .collect();
        assert_eq!(
          r_luma, expect,
          "row-stage full-scale luma must clamp to native-max before narrowing, not wrap"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_gamut_native_vs_rowstage_pinned() {
        let (y, uv) = out_of_gamut();
        let (n_rgb, _, n_luma) = run(&y, &uv, true);
        let (r_rgb, _, r_luma) = run(&y, &uv, false);
        assert_eq!(n_luma, r_luma, "luma stays bit-identical out of gamut");
        let d = max_delta_u8(&n_rgb, &r_rgb);
        assert!(
          d > TOL_U8,
          "crafted out-of-gamut case must diverge beyond the in-gamut tolerance {TOL_U8}, got {d}"
        );
        assert!(d < u8::MAX, "out-of-gamut delta stays bounded, got {d}");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn uniform_gray_leaves_color_unchanged() {
        // Independent-kernel guard (#37): a uniform-gray downscale must leave
        // every colour output equal to the direct conversion of a single pixel.
        let (y, uv) = uniform_gray((MID as u32 + (MASK as u32 / 8)) as u16 & MASK);
        let (n_rgb, n_rgb16, _) = run(&y, &uv, true);
        let (yl, uvl) = (as_le(&y), as_le(&uv));
        let mut ref_rgb = vec![0u8; SRC * SRC * 3];
        let mut ref_rgb16 = vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut ref_rgb)
            .unwrap()
            .with_rgb_u16(&mut ref_rgb16)
            .unwrap();
          $walker(&frame(&yl, &uvl), FR, sink.set_kernel_matrix(M)).unwrap();
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
        // Independent-kernel guard (#37): narrowing the binned u16 colour to u8
        // diverges from the genuine u8 bin over a varying ramp.
        let (y, uv) = ramp();
        let (n_rgb, n_rgb16, _) = run(&y, &uv, true);
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
        $walker(&frame(&yl, &uvl), FR, sink.set_kernel_matrix(M)).unwrap();
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
          $walker(&frame(&y1l, &uv1l), FR, sink.set_kernel_matrix(M)).unwrap();
          $walker(&frame(&y2l, &uv2l), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let y_ref = bin_to_out(&logical_y(&y2), SRC, SRC);
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
        // 4:2:2: chroma row `r` per Y row `r`; the interleaved chroma row is
        // `SRC` u16 wide.
        let (yr, cr) = (3 * SRC, 3 * SRC);
        let err = sink
          .process($row::for_tests(&y[yr..yr + SRC], &uv[cr..cr + SRC], 3, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "expected OutOfSequenceRow, got {err:?}"
        );
        let mut rgb = vec![0u8; OUT * OUT * 3];
        sink.set_rgb(&mut rgb).unwrap();
        sink
          .process($row::for_tests(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect("row 0 must succeed after a rejected out-of-sequence first row");
      }

      /// A mid-frame output-set change must be rejected by the join's
      /// frozen-output preflight BEFORE the wrapper de-pack scratch alloc —
      /// `ResampleOutputsChanged`, never `AllocationFailed`.
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
        // Luma-only rows 0 and 1 freeze a luma-only output set.
        for r in 0..2 {
          let cr = r * SRC;
          sink
            .process($row::for_tests(
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
        crate::sinker::mixed::subsampled_4_2_2_high_bit::arm_p2xx_alloc_failure();
        let err = sink
          .process($row::for_tests(&y[2 * SRC..3 * SRC], &uv[2 * SRC..3 * SRC], 2, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "mid-frame output change must reject as ResampleOutputsChanged before \
           the scratch alloc, got {err:?}"
        );
        assert!(
          rgb_u16.iter().all(|&b| b == 0),
          "rejected mid-frame-change row touched the new colour output"
        );
        // The failpoint is single-shot; prove it was NOT consumed via a fresh
        // in-sequence colour row that DOES fire it.
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
          .process($row::for_tests(&y[..SRC], &uv[..SRC], 0, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err2,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "armed failpoint must still be live and fire on the first in-sequence \
           colour reserve, got {err2:?}"
        );
      }

      /// Sequence rejection after a RECOVERABLE wrapper-scratch allocation
      /// failure: because the wrapper now runs the COMPARE-ONLY preflight (the
      /// commit is the delegate's), a de-pack failure on an in-sequence colour
      /// row 0 leaves the output set UNFROZEN and the join unbuilt, so a later
      /// OUT-OF-SEQUENCE row is caught by the pre-compare first-row sequence
      /// check and must reject as `OutOfSequenceRow`, never `AllocationFailed`.
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
        crate::sinker::mixed::subsampled_4_2_2_high_bit::arm_p2xx_alloc_failure();
        let err0 = sink
          .process($row::for_tests(&y[..SRC], &uv[..SRC], 0, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err0,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the recoverable scratch failure on row 0 must surface AllocationFailed, got {err0:?}"
        );
        crate::sinker::mixed::subsampled_4_2_2_high_bit::arm_p2xx_alloc_failure();
        let err2 = sink
          .process($row::for_tests(&y[2 * SRC..3 * SRC], &uv[2 * SRC..3 * SRC], 2, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err2,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "an out-of-sequence row after a recoverable scratch failure must reject \
           as OutOfSequenceRow, never AllocationFailed, got {err2:?}"
        );
        assert!(
          rgb_u16.iter().all(|&b| b == 0),
          "neither the recoverable-failure nor the out-of-sequence row touched the colour output"
        );
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
          .process($row::for_tests(&y[..SRC], &uv[..SRC], 0, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err3,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the failpoint re-armed in step 2 must still be live and fire on the \
           first in-sequence colour reserve, got {err3:?}"
        );
      }

      // ---- frozen native-vs-row-stage route -----------------------------

      /// Flipping `set_native(true) -> false` mid-frame must reject as the
      /// deterministic `NativeRouteChanged` BEFORE either tier consumes the row.
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
        sink
          .process($row::for_tests(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect("native row 0 freezes the route and succeeds");
        sink.set_native(false);
        let err = sink
          .process($row::for_tests(&y[SRC..2 * SRC], &uv[SRC..2 * SRC], 1, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "a native -> row-stage mid-frame route flip must reject as NativeRouteChanged, got {err:?}"
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
        sink
          .process($row::for_tests(&y[..SRC], &uv[..SRC], 0, M, FR))
          .expect("row-stage row 0 freezes the route and succeeds");
        sink.set_native(true);
        let err = sink
          .process($row::for_tests(&y[SRC..2 * SRC], &uv[SRC..2 * SRC], 1, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "a row-stage -> native mid-frame route flip must reject as NativeRouteChanged, got {err:?}"
        );
      }
    }
  };
}

p2xx_high_bit_native_suite!(
  p210,
  P210LeFrame,
  P210BeFrame,
  P210,
  P210Row,
  p210_to,
  p210_to_endian,
  Yuv444p10,
  Yuv444p10LeFrame,
  yuv444p10_to,
  Yuv422p10,
  Yuv422p10LeFrame,
  yuv422p10_to,
  10,
);
p2xx_high_bit_native_suite!(
  p212,
  P212LeFrame,
  P212BeFrame,
  P212,
  P212Row,
  p212_to,
  p212_to_endian,
  Yuv444p12,
  Yuv444p12LeFrame,
  yuv444p12_to,
  Yuv422p12,
  Yuv422p12LeFrame,
  yuv422p12_to,
  12,
);
p2xx_high_bit_native_suite!(
  p216,
  P216LeFrame,
  P216BeFrame,
  P216,
  P216Row,
  p216_to,
  p216_to_endian,
  Yuv444p16,
  Yuv444p16LeFrame,
  yuv444p16_to,
  Yuv422p16,
  Yuv422p16LeFrame,
  yuv422p16_to,
  16,
);

/// A luma-only high-bit semi-planar 4:2:2 native sink must NOT plan or allocate
/// any chroma state. Armed with the planar join's chroma-planning failpoint at
/// the standard integer-ratio geometry: a luma-only row leaves it unconsumed
/// (so the run succeeds), while a colour row reaches chroma planning and fires.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_native_skips_chroma_planning() {
  use crate::source::{P210, p210_to};
  let y = vec![(1u16 << 9) << (16 - 10); SRC * SRC];
  let uv = vec![(1u16 << 9) << (16 - 10); CW * SRC * 2];
  let (yl, uvl) = (as_le(&y), as_le(&uv));
  let frame = P210LeFrame::new(&yl, &uvl, SRC as u32, SRC as u32, SRC as u32, SRC as u32);

  crate::sinker::mixed::arm_planar_hb_native_chroma_failure();

  // Luma-only: the chroma failpoint is armed but never reached -> Ok.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<P210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    p210_to(&frame, FR, sink.set_kernel_matrix(M)).expect("luma-only native must not plan chroma");
  }

  // Colour: the still-armed failpoint fires at chroma planning -> Err. This both
  // proves the failpoint is wired to chroma planning and consumes the arm.
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<P210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
  assert!(
    p210_to(&frame, FR, sink.set_kernel_matrix(M)).is_err(),
    "colour native must reach chroma planning (the armed failpoint fires)"
  );
}

/// RFC #238 — the high-bit non-4:2:0 native delegate is preflight-transactional
/// through the SEMI-PLANAR 4:2:2 wrapper too. A colour-capability REBUILD (frame
/// 2 attaches colour to a frame-1 luma-only join) whose CHROMA-PLAN build fails
/// (the delegate's failpoint, reached via the P210 wrapper) must surface the
/// typed `AllocationFailed` while leaving the cached luma-only join AND the
/// output-set freeze INTACT (uncommitted), so retrying row 0 with YET ANOTHER
/// output attached is accepted — a committed `{luma, hsv}` freeze would reject it
/// as `ResampleOutputsChanged`.
#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_colour_capability_rebuild_chroma_oom_is_transactional_semi_planar() {
  use crate::source::{P210, P210Row};
  // P210 is 10-bit high-bit-packed (`logical << (16 - 10)`); a uniform mid-gray.
  let y = vec![(1u16 << 9) << (16 - 10); SRC * SRC];
  // Interleaved UV row stride = `2 * CW == SRC` u16.
  let uv = vec![(1u16 << 9) << (16 - 10); SRC * SRC];
  let (yl, uvl) = (as_le(&y), as_le(&uv));
  let row = |r: usize| {
    P210Row::for_tests(
      &yl[r * SRC..(r + 1) * SRC],
      &uvl[r * SRC..(r + 1) * SRC],
      r,
      M,
      FR,
    )
  };

  let mut luma = vec![0u8; OUT * OUT];
  let (mut hh, mut ss, mut vv) = (
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
  );
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<P210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();

  // Frame 1: a luma-only native join (chroma absent), frozen as `{luma}`.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  for r in 0..SRC {
    sink
      .process(row(r))
      .expect("luma-only frame builds the native join");
  }

  // Frame 2: attach HSV -> the row-0 rebuild MUST build the chroma half; arm the
  // planar join's chroma-plan failpoint so that rebuild fails.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  crate::sinker::mixed::arm_planar_hb_native_chroma_failure();
  let err = sink.process(row(0)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the P210 colour-capability rebuild chroma-plan OOM must surface AllocationFailed, got {err:?}"
  );

  // The freeze was never committed and the cached join stayed intact: attaching
  // ANOTHER output (rgb) and retrying row 0 is ACCEPTED.
  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(row(0))
    .expect("row 0 with a changed output set must succeed after a rebuild chroma OOM");
  // Drive the rest of the frame through the recovered join: a fully-recovered
  // join emits real colour output (mid-gray input -> non-zero RGB).
  for r in 1..SRC {
    sink
      .process(row(r))
      .expect("the recovered join must process the rest of the frame");
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered join must produce real colour output"
  );
}

/// The source-scratch failpoint companion of
/// [`native_colour_capability_rebuild_chroma_oom_is_transactional_semi_planar`]:
/// the delegate's SOURCE-SCRATCH grow (on the rebuilt local join, reached via the
/// P210 wrapper) fails on the colour-capability rebuild row and must likewise
/// leave the cached join + freeze uncommitted so a changed-output retry succeeds.
#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_colour_capability_rebuild_src_scratch_oom_is_transactional_semi_planar() {
  use crate::source::{P210, P210Row};
  let y = vec![(1u16 << 9) << (16 - 10); SRC * SRC];
  let uv = vec![(1u16 << 9) << (16 - 10); SRC * SRC];
  let (yl, uvl) = (as_le(&y), as_le(&uv));
  let row = |r: usize| {
    P210Row::for_tests(
      &yl[r * SRC..(r + 1) * SRC],
      &uvl[r * SRC..(r + 1) * SRC],
      r,
      M,
      FR,
    )
  };

  let mut luma = vec![0u8; OUT * OUT];
  let (mut hh, mut ss, mut vv) = (
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
  );
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<P210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();

  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  for r in 0..SRC {
    sink
      .process(row(r))
      .expect("luma-only frame builds the native join");
  }

  // Frame 2: attach HSV -> rebuild WITH chroma; arm the delegate's source-scratch
  // failpoint so the rebuilt join's Y de-interleave scratch grow refuses.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  crate::sinker::mixed::arm_planar_hb_native_alloc_failure();
  let err = sink.process(row(0)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the P210 colour-capability rebuild source-scratch OOM must surface AllocationFailed, got {err:?}"
  );

  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(row(0))
    .expect("row 0 with a changed output set must succeed after a rebuild source-scratch OOM");
  for r in 1..SRC {
    sink
      .process(row(r))
      .expect("the recovered join must process the rest of the frame");
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered join must produce real colour output"
  );
}
