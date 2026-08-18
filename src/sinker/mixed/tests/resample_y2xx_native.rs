//! Fused-downscale coverage for the high-bit **packed 4:2:2** YUV NATIVE fast
//! tier — `Y210` (10-bit) / `Y212` (12-bit) / `Y216` (16-bit), LE + BE wire.
//! The packed, MSB-aligned twin of the high-bit **semi-planar** 4:2:2
//! [`p2xx_process_native`](crate::sinker::mixed::subsampled_4_2_2_high_bit): both
//! reuse the high-bit non-4:2:0 PLANAR join
//! ([`yuv_planar16_process_native`](crate::sinker::mixed::planar_high_bit_native))
//! after de-interleaving + DE-PACKING (`wire >> (16 - BITS)`) the wire planes
//! into wrapper-owned host-native LOGICAL u16 scratch. Y2xx packs four u16 words
//! per 2-pixel group in YUYV order (`Y₀ U Y₁ V`), so the de-pack reads Y from
//! words 0 / 2 and U / V from words 1 / 3.
//!
//! The native tier bins those planes straight to the output grid and converts
//! ONCE per output row at output width (4:4:4 kernels) — vs the row-stage tier
//! ([`packed_yuv422_triple_resample`](crate::sinker::mixed::packed_yuv422_triple_resample)),
//! which converts each source row at source width then bins. The tiers differ in
//! colour SEMANTICS (native averages in YUV then converts; row-stage converts
//! then averages in RGB), so native is NOT byte-identical to row-stage — only
//! within a small tolerance in-gamut. Luma is bit-identical (both bin the same
//! de-packed native Y then narrow `>> (BITS - 8)`).
//!
//! The high-bit planar join now emits BOTH u8 `luma` and the native-depth
//! `luma_u16` (the clamped binned Y), so the Y2xx sink routes to native for
//! EVERY output set it exposes — attaching `luma_u16` no longer falls the
//! pipeline back to row-stage (which would silently change the rgb colour
//! semantics). `native_luma_u16_equals_clamped_binned_y` pins the native
//! `luma_u16` against an independent clamped-binned-Y oracle and asserts a
//! rgb + `luma_u16` sink uses native for BOTH;
//! `luma_u16_attach_mid_frame_rejected_as_outputs_changed` pins the
//! frozen-output precedence (a mid-frame `luma_u16` attach is
//! `ResampleOutputsChanged`, not `NativeRouteChanged`).
//!
//! Per format + depth (LE + BE):
//! - `native_equals_bin_then_convert_oracle`: the GROUND-TRUTH check — native
//!   output EXACTLY equals an independent bin-then-convert oracle that de-packs
//!   the Y2xx quadruples, area-bins each plane to OUTPUT resolution (Y 2:1 x 2:1,
//!   chroma 2:1 horizontal only — 4:2:2), then converts ONCE through an
//!   identity-resolution high-bit `Yuv444pN` sink (chroma binned to full output
//!   width) with the SAME native-depth kernels + `(1 << BITS) - 1` clamp the
//!   native tier finalizes with. The luma oracle clamps INDEPENDENTLY.
//! - `native_equals_planar_twin`: native Y2xx == native `Yuv422pN` on the
//!   de-packed planes — the strong cross-check that the packed wrapper is a
//!   faithful de-pack in front of the planar join.
//! - `native_within_tolerance_of_rowstage` / `_be_*`: the cv2 INTER_AREA parity
//!   bound + the `BE = HOST_NATIVE_BE` handoff proof.
//! - `native_luma_clamps_full_scale_y` / `rowstage_luma_clamps_full_scale_y`: the
//!   native-depth luma clamp at the achievable full-scale boundary, BOTH tiers.
//! - `luma_only_native_skips_chroma_planning`: a luma-only sink plans no chroma.
//! - the four atomicity regressions on [`arm_y2xx_alloc_failure`] + the route
//!   freeze guard.

use crate::{
  KernelMatrix,
  PixelSink,
  // `Y2xxFrame` is the packed source frame; the `Yuv4*pNLeFrame` types are the
  // planar oracle / twin-parity frames the macro references by name.
  frame::{
    Y2xxFrame, Yuv422p10LeFrame, Yuv422p12LeFrame, Yuv422p16LeFrame, Yuv444p10LeFrame,
    Yuv444p12LeFrame, Yuv444p16LeFrame,
  },
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  // Bin-then-convert oracle markers (chroma binned to full output width → 4:4:4
  // sink) + twin-parity markers (de-packed planes → 4:2:2 native sink at source
  // resolution). Both run the SAME native-depth kernels + `(1 << BITS) - 1` clamp
  // the packed native tier delegates to.
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
/// The two average in different domains (YUV vs RGB) and round independently per
/// output pixel; native correctness itself is pinned EXACTLY by
/// `native_equals_bin_then_convert_oracle`, this bound only documents the
/// row-stage semantic gap. Matches the high-bit semi-planar 4:2:2 suite's bound.
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

/// Re-encode a host-native u16 slice as host-independent LE-wire byte storage (a
/// no-op on LE, a byte swap on BE).
fn as_le(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}
/// Re-encode a host-native u16 slice as host-independent BE-wire byte storage.
fn as_be(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

macro_rules! y2xx_native_suite {
  (
    $mod:ident, $marker:ident, $row:ident, $walker:ident, $walker_be:ident,
    // The bin-then-convert oracle bins chroma to FULL output resolution, so it
    // feeds a 4:4:4 sink (`Yuv444pN`). The twin-parity check feeds the de-packed
    // planes at SOURCE resolution to the matching 4:2:2 native sink (`Yuv422pN`).
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
      /// near the legal-range middle so the converted RGB stays in gamut and the
      /// native-vs-rowstage delta is per-pixel rounding. Packed into an `SRC`-grid
      /// Y2xx plane of MSB-aligned (`logical << (16 - BITS)`) `Y₀ U Y₁ V`
      /// quadruples — chroma sampled at the even column of each 2-pixel pair
      /// (4:2:2).
      fn ramp_packed() -> Vec<u16> {
        let mut buf = vec![0u16; SRC * 2 * SRC];
        for row in 0..SRC {
          for cx in 0..CW {
            let i0 = row * SRC + cx * 2;
            let i1 = i0 + 1;
            let ci = row * CW + cx;
            let y0 = (MID as u32 + ((i0 as u32 * 37) % (MASK as u32 / 4))) as u16 & MASK;
            let y1 = (MID as u32 + ((i1 as u32 * 37) % (MASK as u32 / 4))) as u16 & MASK;
            let u =
              (MID as u32 + ((ci as u32 * 53) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16
                & MASK;
            let v =
              (MID as u32 + ((ci as u32 * 41) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16
                & MASK;
            let base = row * 2 * SRC + cx * 4;
            buf[base] = y0 << SHIFT;
            buf[base + 1] = u << SHIFT;
            buf[base + 2] = y1 << SHIFT;
            buf[base + 3] = v << SHIFT;
          }
        }
        buf
      }

      /// Uniform-gray plane: constant logical Y, neutral chroma (U = V = mid),
      /// MSB-aligned.
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

      /// Crafted VARYING illegal-chroma fixture: extreme alternating chroma over
      /// a super-black->super-white Y ramp, MSB-aligned — many 2x2 blocks
      /// straddle the RGB clamp, where native (average-in-YUV) and row-stage
      /// (convert-then-average) genuinely diverge.
      fn out_of_gamut_packed() -> Vec<u16> {
        let mut buf = vec![0u16; SRC * 2 * SRC];
        for row in 0..SRC {
          for cx in 0..CW {
            let i0 = row * SRC + cx * 2;
            let i1 = i0 + 1;
            let ci = row * CW + cx;
            let y0 = ((i0 as u32 * MASK as u32) / (SRC * SRC) as u32) as u16 & MASK;
            let y1 = ((i1 as u32 * MASK as u32) / (SRC * SRC) as u32) as u16 & MASK;
            let hi = ci % 2 == 0;
            let base = row * 2 * SRC + cx * 4;
            buf[base] = y0 << SHIFT;
            buf[base + 1] = if hi { MASK } else { 0 } << SHIFT;
            buf[base + 2] = y1 << SHIFT;
            buf[base + 3] = if hi { 0 } else { MASK } << SHIFT;
          }
        }
        buf
      }

      /// Full-scale-Y fixture: every Y at the native max `MASK`, MSB-aligned
      /// (`MASK << SHIFT`). An MSB-packed Y2xx sample CANNOT exceed `MASK` (the
      /// de-pack `wire >> (16 - BITS)` caps at `MASK`), and an area mean of
      /// `<= MASK` stays `<= MASK` — so the achievable boundary is the legal max.
      /// Exercises the native-depth luma clamp at that boundary:
      /// `min(MASK) >> (BITS - 8)` must saturate, never wrap. (Genuine
      /// beyond-max overrange is impossible to encode here; the reused planar
      /// join's clamp for overrange is covered by the planar high-bit native
      /// suite's low-packed fixture.) Chroma stays legal.
      fn full_scale_luma_packed() -> Vec<u16> {
        let mut buf = ramp_packed();
        for q in 0..(SRC * SRC / 2) {
          buf[q * 4] = MASK << SHIFT;
          buf[q * 4 + 2] = MASK << SHIFT;
        }
        buf
      }

      fn frame(buf: &[u16]) -> Y2xxFrame<'_, $bits, false> {
        Y2xxFrame::try_new(buf, SRC as u32, SRC as u32, (2 * SRC) as u32).unwrap()
      }
      fn frame_be(buf: &[u16]) -> Y2xxFrame<'_, $bits, true> {
        Y2xxFrame::try_new(buf, SRC as u32, SRC as u32, (2 * SRC) as u32).unwrap()
      }

      /// De-pack the MSB-aligned Y2xx quadruples into the logical native Y plane
      /// (`SRC x SRC`): Y from words 0 / 2 of each group.
      fn logical_y(packed: &[u16]) -> Vec<u16> {
        let mut y = vec![0u16; SRC * SRC];
        for (i, group) in packed.chunks_exact(4).enumerate() {
          y[i * 2] = group[0] >> SHIFT;
          y[i * 2 + 1] = group[2] >> SHIFT;
        }
        y
      }
      /// De-pack + de-interleave the Y2xx quadruples into separate logical
      /// `U` / `V` planes (`CW x SRC` each): U from word 1, V from word 3.
      fn depack_uv(packed: &[u16]) -> (Vec<u16>, Vec<u16>) {
        let mut u = vec![0u16; CW * SRC];
        let mut v = vec![0u16; CW * SRC];
        for (i, group) in packed.chunks_exact(4).enumerate() {
          u[i] = group[1] >> SHIFT;
          v[i] = group[3] >> SHIFT;
        }
        (u, v)
      }

      /// Drive the LE source through a tier for the native output set (u8 RGB +
      /// u16 RGB + u8 luma). `native` toggles the bin-then-convert native fast
      /// tier vs the convert-then-bin row-stage tier. NB: u8 `luma` (not
      /// `luma_u16`) — the native tier serves only the planar join's u8 luma.
      fn run(packed: &[u16], native: bool) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let p = as_le(packed);
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
          $walker(&frame(&p), FR, M, &mut sink).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the BE source through the native tier (the host-native-endian
      /// guard reference).
      fn native_be_run(packed: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let p = as_be(packed);
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
          $walker_be::<_, true>(&frame_be(&p), FR, M, &mut sink).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the row-stage tier (BE) — the correct host-independent reference
      /// (it de-packs BE-wire bytes to host-native before converting).
      fn rowstage_be_run(packed: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let p = as_be(packed);
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
          $walker_be::<_, true>(&frame_be(&p), FR, M, &mut sink).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// The bin-then-convert oracle: de-pack the Y2xx quadruples, area-bin every
      /// plane to OUTPUT resolution (Y from `SRC x SRC`, chroma from `CW x SRC` —
      /// horizontal-only subsample), then convert the full-output-width
      /// host-native planes ONCE through an identity-resolution high-bit
      /// `Yuv444pN` sink. The luma oracle clamps INDEPENDENTLY of the sink.
      fn oracle(packed: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let yl = logical_y(packed);
        let (u, v) = depack_uv(packed);
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
          $oracle_walker(&f, FR, M, &mut sink).unwrap();
        }
        let luma: Vec<u8> = yb
          .iter()
          .map(|&by| (by.min(MASK) >> ($bits - 8)) as u8)
          .collect();
        (rgb, rgb_u16, luma)
      }

      /// Native `Yuv422pN` reference on the de-packed planes — the twin-parity
      /// cross-check (same join, fed planar instead of through the packed
      /// de-pack).
      fn planar_twin_native(packed: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let yl = logical_y(packed);
        let (u, v) = depack_uv(packed);
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
          $twin_walker(&f, FR, M, &mut sink).unwrap();
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
        let packed = ramp_packed();
        let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
        let (o_rgb, o_rgb16, o_luma) = oracle(&packed);
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
        // The packed wrapper IS a de-pack + de-interleave in front of the planar
        // join, so its output must be bit-identical to feeding the de-packed
        // planes straight to the native Yuv422pN.
        let packed = ramp_packed();
        let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
        let (t_rgb, t_rgb16, t_luma) = planar_twin_native(&packed);
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
        let packed = ramp_packed();
        let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
        let (r_rgb, r_rgb16, r_luma) = run(&packed, false);
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
        // The native tier de-packs the wire quadruples to host-native LOGICAL u16
        // BEFORE binning, so BE and LE sources produce identical output.
        let packed = ramp_packed();
        let le = run(&packed, true);
        let be = native_be_run(&packed);
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
        let packed = ramp_packed();
        let (n_rgb, n_rgb16, n_luma) = native_be_run(&packed);
        let (r_rgb, r_rgb16, r_luma) = rowstage_be_run(&packed);
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
        // Y, narrowed. Guards the Y de-pack (`raw >> (16 - BITS)`) off words 0 / 2.
        let packed = ramp_packed();
        let (_, _, n_luma) = run(&packed, true);
        let y_ref = bin_to_out(&logical_y(&packed), SRC, SRC);
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
        // `>> (BITS - 8)` narrowing, never wrap. The oracle clamps independently.
        let packed = full_scale_luma_packed();
        let (_, _, n_luma) = run(&packed, true);
        let yb = bin_to_out(&logical_y(&packed), SRC, SRC);
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
        let packed = full_scale_luma_packed();
        let (_, _, r_luma) = run(&packed, false);
        let yb = bin_to_out(&logical_y(&packed), SRC, SRC);
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
        let packed = out_of_gamut_packed();
        let (n_rgb, _, n_luma) = run(&packed, true);
        let (r_rgb, _, r_luma) = run(&packed, false);
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
        let packed = uniform_gray_packed((MID as u32 + (MASK as u32 / 8)) as u16 & MASK);
        let (n_rgb, n_rgb16, _) = run(&packed, true);
        let p = as_le(&packed);
        let mut ref_rgb = vec![0u8; SRC * SRC * 3];
        let mut ref_rgb16 = vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut ref_rgb)
            .unwrap()
            .with_rgb_u16(&mut ref_rgb16)
            .unwrap();
          $walker(&frame(&p), FR, M, &mut sink).unwrap();
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
        let packed = ramp_packed();
        let (n_rgb, n_rgb16, _) = run(&packed, true);
        let narrowed: Vec<u8> = n_rgb16.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_ne!(
          n_rgb, narrowed,
          "u8 colour must be an independent bin, not a narrowed u16 bin"
        );
      }

      /// The native fast tier emits the native-depth `luma_u16` directly (the
      /// clamped binned Y, host-native u16 — NOT narrowed), so it equals an
      /// INDEPENDENT clamped-binned-Y oracle (`bin_to_out(logical Y).min(MASK)`)
      /// bit-for-bit, LE + BE. AND a `rgb` + `luma_u16` sink takes the NATIVE
      /// route for BOTH: the rgb matches the native bin-then-convert oracle (not
      /// the row-stage tier), proving that attaching `luma_u16` no longer changes
      /// the rgb colour semantics.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_u16_equals_clamped_binned_y() {
        let packed = ramp_packed();
        // Independent oracle: area-bin the de-packed native Y to OUTPUT res, then
        // clamp to the native max (the native-depth contract).
        let luma_u16_oracle: Vec<u16> = bin_to_out(&logical_y(&packed), SRC, SRC)
          .iter()
          .map(|&by| by.min(MASK))
          .collect();

        // LE: a `luma_u16`-only native sink.
        {
          let p = as_le(&packed);
          let mut luma_u16 = vec![0u16; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(true)
            .with_luma_u16(&mut luma_u16)
            .unwrap();
            $walker(&frame(&p), FR, M, &mut sink).unwrap();
          }
          assert_eq!(
            luma_u16, luma_u16_oracle,
            "native luma_u16 (LE) must equal the clamped-binned-Y oracle"
          );
        }

        // BE: same, through the BE wire — the native tier de-packs to host-native
        // LOGICAL u16 before binning, so it is bit-identical to LE.
        {
          let p = as_be(&packed);
          let mut luma_u16 = vec![0u16; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(true)
            .with_luma_u16(&mut luma_u16)
            .unwrap();
            $walker_be::<_, true>(&frame_be(&p), FR, M, &mut sink).unwrap();
          }
          assert_eq!(
            luma_u16, luma_u16_oracle,
            "native luma_u16 (BE) must equal the clamped-binned-Y oracle"
          );
        }

        // A `rgb` + `luma_u16` sink uses the NATIVE route for BOTH: rgb equals the
        // native bin-then-convert oracle (NOT row-stage), and luma_u16 equals the
        // clamped-binned-Y oracle — attaching luma_u16 no longer changes rgb.
        {
          let p = as_le(&packed);
          let mut rgb = vec![0u8; OUT * OUT * 3];
          let mut luma_u16 = vec![0u16; OUT * OUT];
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
            .with_luma_u16(&mut luma_u16)
            .unwrap();
            $walker(&frame(&p), FR, M, &mut sink).unwrap();
          }
          let (o_rgb, _, _) = oracle(&packed);
          assert_eq!(
            rgb, o_rgb,
            "rgb + luma_u16 must take the NATIVE route — rgb equals the native \
             oracle, not row-stage"
          );
          assert_eq!(
            luma_u16, luma_u16_oracle,
            "rgb + luma_u16: luma_u16 must equal the clamped-binned-Y oracle"
          );
        }
      }

      #[test]
      fn no_outputs_is_a_no_op() {
        let packed = ramp_packed();
        let p = as_le(&packed);
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true);
        $walker(&frame(&p), FR, M, &mut sink).unwrap();
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resets_join_across_frames() {
        let packed1 = ramp_packed();
        let invert = |p: &[u16]| -> Vec<u16> {
          p.iter()
            .map(|&x| (MASK.wrapping_sub(x >> SHIFT) & MASK) << SHIFT)
            .collect()
        };
        let packed2 = invert(&packed1);
        let p1 = as_le(&packed1);
        let p2 = as_le(&packed2);
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
          $walker(&frame(&p1), FR, M, &mut sink).unwrap();
          $walker(&frame(&p2), FR, M, &mut sink).unwrap();
        }
        let y_ref = bin_to_out(&logical_y(&packed2), SRC, SRC);
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
        let packed = ramp_packed();
        let p = as_le(&packed);
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
        // The Y2xx row is `width x 2` u16 (one packed plane).
        let row3 = &p[3 * 2 * SRC..4 * 2 * SRC];
        let err = sink.process($row::new(row3, 3, M, FR)).unwrap_err();
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
          .process($row::new(&p[..2 * SRC], 0, M, FR))
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
        let packed = ramp_packed();
        let p = as_le(&packed);
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
          sink
            .process($row::new(&p[r * 2 * SRC..(r + 1) * 2 * SRC], r, M, FR))
            .expect("luma-only rows freeze a luma-only output set");
        }
        // Attach u16 colour mid-frame, changing the output set, and arm the
        // wrapper scratch failpoint on the reserve the changed row reaches.
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        crate::sinker::mixed::arm_y2xx_alloc_failure();
        let err = sink
          .process($row::new(&p[2 * 2 * SRC..3 * 2 * SRC], 2, M, FR))
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
        let err2 = sink2.process($row::new(&p[..2 * SRC], 0, M, FR)).unwrap_err();
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
      /// delegate owns the single commit), a de-pack failure on an in-sequence
      /// colour row 0 leaves the output set UNFROZEN and the join unbuilt, so a
      /// later OUT-OF-SEQUENCE row is caught by the pre-compare first-row sequence
      /// check and must reject as `OutOfSequenceRow`, never `AllocationFailed`.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn oos_after_recoverable_alloc_failure_rejected_before_scratch_alloc() {
        let packed = ramp_packed();
        let p = as_le(&packed);
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
        crate::sinker::mixed::arm_y2xx_alloc_failure();
        let err0 = sink.process($row::new(&p[..2 * SRC], 0, M, FR)).unwrap_err();
        assert!(
          matches!(
            err0,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the recoverable scratch failure on row 0 must surface AllocationFailed, got {err0:?}"
        );
        crate::sinker::mixed::arm_y2xx_alloc_failure();
        let err2 = sink
          .process($row::new(&p[2 * 2 * SRC..3 * 2 * SRC], 2, M, FR))
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
        let err3 = sink2.process($row::new(&p[..2 * SRC], 0, M, FR)).unwrap_err();
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
        let packed = ramp_packed();
        let p = as_le(&packed);
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
          .process($row::new(&p[..2 * SRC], 0, M, FR))
          .expect("native row 0 freezes the route and succeeds");
        sink.set_native(false);
        let err = sink
          .process($row::new(&p[2 * SRC..2 * 2 * SRC], 1, M, FR))
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
        let packed = ramp_packed();
        let p = as_le(&packed);
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
          .process($row::new(&p[..2 * SRC], 0, M, FR))
          .expect("row-stage row 0 freezes the route and succeeds");
        sink.set_native(true);
        let err = sink
          .process($row::new(&p[2 * SRC..2 * 2 * SRC], 1, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "a row-stage -> native mid-frame route flip must reject as NativeRouteChanged, got {err:?}"
        );
      }

      /// Attaching `luma_u16` MID-FRAME (after a native u8-luma row froze the
      /// output set) must be classified by the FROZEN-OUTPUT check as
      /// `ResampleOutputsChanged`, NOT by the route guard as `NativeRouteChanged`.
      /// A `take_native = native && luma_u16.is_none()` predicate would flip the
      /// route on the attach (`true` -> `false`) and spuriously trip
      /// `NativeRouteChanged` before the frozen-output check could see it; because
      /// `take_native = native` is invariant to `luma_u16`, the row enters the
      /// native delegate and the frozen-output preflight reports the genuine
      /// output-set change — the correct, more specific error.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_u16_attach_mid_frame_rejected_as_outputs_changed() {
        let packed = ramp_packed();
        let p = as_le(&packed);
        let mut luma = vec![0u8; OUT * OUT];
        let mut luma_u16 = vec![0u16; OUT * OUT];
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
        // Native u8-luma row 0 freezes a {luma} output set on the native route.
        sink
          .process($row::new(&p[..2 * SRC], 0, M, FR))
          .expect("native luma row 0 freezes the output set and the route");
        // Attach native-depth luma_u16 mid-frame: the route stays native
        // (`take_native = *native`), so the frozen-output check owns the verdict.
        sink.set_luma_u16(&mut luma_u16).unwrap();
        let err = sink
          .process($row::new(&p[2 * SRC..2 * 2 * SRC], 1, M, FR))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "a mid-frame luma_u16 attach must reject as ResampleOutputsChanged (the \
           frozen-output check), not NativeRouteChanged, got {err:?}"
        );
        assert!(
          luma_u16.iter().all(|&b| b == 0),
          "the rejected mid-frame-change row must not touch the new luma_u16 output"
        );
      }
    }
  };
}

y2xx_native_suite!(
  y210,
  Y210,
  Y210Row,
  y210_to,
  y210_to_endian,
  Yuv444p10,
  Yuv444p10LeFrame,
  yuv444p10_to,
  Yuv422p10,
  Yuv422p10LeFrame,
  yuv422p10_to,
  10,
);
y2xx_native_suite!(
  y212,
  Y212,
  Y212Row,
  y212_to,
  y212_to_endian,
  Yuv444p12,
  Yuv444p12LeFrame,
  yuv444p12_to,
  Yuv422p12,
  Yuv422p12LeFrame,
  yuv422p12_to,
  12,
);
y2xx_native_suite!(
  y216,
  Y216,
  Y216Row,
  y216_to,
  y216_to_endian,
  Yuv444p16,
  Yuv444p16LeFrame,
  yuv444p16_to,
  Yuv422p16,
  Yuv422p16LeFrame,
  yuv422p16_to,
  16,
);

/// A luma-only high-bit packed 4:2:2 native sink must NOT plan or allocate any
/// chroma state. Armed with the planar join's chroma-planning failpoint at the
/// standard integer-ratio geometry: a luma-only row leaves it unconsumed (so the
/// run succeeds), while a colour row reaches chroma planning and fires.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_native_skips_chroma_planning() {
  use crate::source::{Y210, y210_to};
  // Uniform MSB-aligned 10-bit packed plane (`Y₀ U Y₁ V` quadruples).
  let mut packed = vec![0u16; SRC * 2 * SRC];
  for q in 0..(SRC * SRC / 2) {
    let base = q * 4;
    packed[base] = (1u16 << 9) << (16 - 10);
    packed[base + 1] = (1u16 << 9) << (16 - 10);
    packed[base + 2] = (1u16 << 9) << (16 - 10);
    packed[base + 3] = (1u16 << 9) << (16 - 10);
  }
  let p = as_le(&packed);
  let frame =
    Y2xxFrame::<10, false>::try_new(&p, SRC as u32, SRC as u32, (2 * SRC) as u32).unwrap();

  crate::sinker::mixed::arm_planar_hb_native_chroma_failure();

  // Luma-only: the chroma failpoint is armed but never reached -> Ok.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Y210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    y210_to(&frame, FR, M, &mut sink).expect("luma-only native must not plan chroma");
  }

  // Colour: the still-armed failpoint fires at chroma planning -> Err. This both
  // proves the failpoint is wired to chroma planning and consumes the arm.
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Y210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
  assert!(
    y210_to(&frame, FR, M, &mut sink).is_err(),
    "colour native must reach chroma planning (the armed failpoint fires)"
  );
}

/// RFC #238 — a fallible grow INSIDE the delegate, reached through a PACKED
/// wrapper (Y210) AFTER the wrapper's own de-pack scratch work has succeeded,
/// must still leave the output-set freeze untouched. Because the packed wrappers
/// now run the COMPARE-ONLY preflight (the delegate owns the single commit), the
/// wrapper commits no freeze; a colour-capability rebuild whose delegate
/// `rgb_scratch_u16` grow refuses returns AllocationFailed with `resample_outputs`
/// unfrozen, so a retry of row 0 with yet another output attached is accepted
/// (the pre-fix committing wrapper would have frozen `{luma, rgb_u16}` before the
/// delegate's grow and mis-rejected the retry as ResampleOutputsChanged).
#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_delegate_grow_failure_through_packed_wrapper_is_transactional() {
  use crate::source::{Y210, Y210Row};
  // Uniform MSB-aligned 10-bit packed plane (`Y₀ U Y₁ V` quads); 2*SRC u16 / row.
  let mut packed = vec![0u16; SRC * 2 * SRC];
  for q in 0..(SRC * SRC / 2) {
    let base = q * 4;
    for k in 0..4 {
      packed[base + k] = (1u16 << 9) << (16 - 10);
    }
  }
  let p = as_le(&packed);
  let row = |r: usize| Y210Row::new(&p[r * 2 * SRC..(r + 1) * 2 * SRC], r, M, FR);

  let mut luma = vec![0u8; OUT * OUT];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Y210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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

  // Frame 2: attach u16 colour -> the Y210 wrapper de-packs (its OWN scratch work
  // succeeds), then the delegate rebuilds and grows `rgb_scratch_u16`; arm that
  // delegate grow to refuse, so the failure lands INSIDE the delegate.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_rgb_u16(&mut rgb_u16).unwrap();
  crate::sinker::mixed::arm_planar_hb_native_rgb_u16_scratch_failure();
  let err = sink.process(row(0)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "a delegate u16-scratch OOM (after the packed wrapper's de-pack) must surface \
     AllocationFailed, got {err:?}"
  );

  // The wrapper committed NO freeze: attaching ANOTHER output (rgb) and retrying
  // row 0 is accepted — proof the freeze was left untouched by the failed
  // delegate grow.
  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(row(0))
    .expect("row 0 with a changed output set must succeed after a delegate u16-scratch OOM");
  for r in 1..SRC {
    sink
      .process(row(r))
      .expect("the recovered join must process the rest of the frame");
  }
  assert!(
    rgb.iter().any(|&b| b != 0) && rgb_u16.iter().any(|&b| b != 0),
    "the recovered join must produce real u8 and u16 colour output"
  );
}
