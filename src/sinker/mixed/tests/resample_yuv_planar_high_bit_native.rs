//! Fused-downscale coverage for the high-bit **planar non-4:2:0** YUV NATIVE
//! fast tier — `Yuv422p10/12/14/16` (4:2:2), `Yuv444p10/12/14/16` (4:4:4),
//! `Yuv440p10/12` (4:4:0), LE + BE wire — the `u16` twin of the 8-bit
//! [`yuv_planar_process_native`](crate::sinker::mixed::planar_8bit::yuv_planar_process_native)
//! and the non-4:2:0 sibling of the high-bit 4:2:0
//! [`yuv420p16_process_native`](crate::sinker::mixed::subsampled_4_2_0_high_bit::yuv420p16_process_native).
//!
//! The native tier bins the host-native Y / U / V planes straight to the
//! output grid and converts ONCE per output row at output width (4:4:4
//! kernels), vs the row-stage tier
//! ([`packed_yuv422_triple_resample`](crate::sinker::mixed::packed_yuv422_triple_resample)
//! / [`packed_yuv444_triple_resample`](crate::sinker::mixed::packed_yuv444_triple_resample)),
//! which converts each source row at source width then bins. The tiers differ
//! in colour SEMANTICS (native averages in YUV then converts; row-stage
//! converts then averages in RGB), so native is NOT byte-identical to
//! row-stage — only within a small tolerance in-gamut, with documented
//! out-of-gamut divergence. Luma is bit-identical (both bin the same native Y
//! stream then narrow `>> (BITS - 8)`).
//!
//! Per format + depth (LE + BE):
//! - `native_equals_bin_then_convert_oracle`: the GROUND-TRUTH check — native
//!   output is EXACTLY the bin-then-convert reference. Area-bin every source
//!   plane to OUTPUT resolution by its own subsample ratio (Y always
//!   2:1 x 2:1; chroma 2:1 only on its subsampled axis: horizontal for 4:2:2,
//!   vertical for 4:4:0, both for 4:4:4), giving full-output-width chroma, then
//!   convert ONCE through an identity-resolution high-bit `Yuv444pN` sink — the
//!   SAME native-depth 4:4:4 kernels (and their `(1 << BITS) - 1` clamp) the
//!   native tier finalizes with, so the comparison is clamp-for-clamp exact,
//!   never against an unclamped value (`rgb_u16` is the native-depth u16
//!   colour, `rgb` the independent u8 colour, `luma` the binned-then-narrowed
//!   Y). This pins the per-plane binning geometry + the once-per-output-pixel
//!   convert independently of the row-stage tier.
//! - `native_within_tolerance_of_rowstage`: same source through
//!   `with_native(true)` and `with_native(false)`, per-channel
//!   `|native - rowstage| <= TOL` for u8 / u16 colour in gamut, LUMA
//!   bit-identical. The row-stage tier IS the cv2 INTER_AREA oracle, so this is
//!   the INTER_AREA parity check.
//! - `native_be_matches_native_le`: the native tier de-interleaves the wire to
//!   host-native BEFORE binning, so BE and LE sources produce identical output
//!   (the host-native-endian guard).
//! - `native_luma_matches_inter_area_oracle`: native luma equals the direct
//!   per-axis area mean of the native Y plane, narrowed.
//! - `out_of_gamut_native_vs_rowstage_pinned`: on a crafted illegal-chroma
//!   fixture the tiers diverge by MORE than the in-gamut tolerance, luma still
//!   bit-identical.
//! - `native_default_matches_explicit_true`: `with_native` defaults to native.

use crate::{
  KernelMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Yuv444p9, Yuv444p10, Yuv444p12, Yuv444p14, Yuv444p16, yuv444p9_to, yuv444p10_to, yuv444p12_to,
    yuv444p14_to, yuv444p16_to,
  },
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// In-gamut per-channel u8 tolerance between the native and row-stage tiers.
/// The two average in different domains (YUV vs RGB) and round independently
/// per output pixel; the empirical in-gamut maximum on the mid-range ramp
/// fixtures here is small, pinned with a margin for cross-platform
/// SIMD-vs-scalar rounding. Native correctness itself is pinned EXACTLY by
/// `native_equals_bin_then_convert_oracle`; this bound only documents the
/// row-stage semantic gap. (The non-4:2:0 chroma carries more spatial detail
/// into each 2x2 bin than 4:2:0, so the convert-order gap is a touch wider than
/// the 4:2:0 high-bit suite's bound of 2.) The u16 colour uses
/// `TOL_U8 << (BITS - 8)`.
const TOL_U8: u8 = 5;

/// Re-encode a host-native u16 slice as host-independent LE-wire byte storage
/// (the `*LeFrame` plane contract): a no-op on a little-endian host, a byte
/// swap on big-endian.
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

macro_rules! yuv_planar_hb_native_suite {
  (
    $mod:ident,
    $marker:ident, $frame_le:ident, $frame_be:ident, $row:ident,
    $walker:ident, $walker_be:ident,
    $oracle_marker:ident, $oracle_frame:ident, $oracle_walker:ident,
    $cw:expr, $ch:expr, $cvsub:expr, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $row, $walker, $walker_be};

      const CW: usize = $cw;
      const CH: usize = $ch;
      /// Vertical chroma cadence: a chroma source row per `CVSUB` luma rows (1
      /// for 4:2:2 / 4:4:4, 2 for 4:4:0). The chroma source row feeding luma
      /// row `idx` is `(idx / CVSUB) * CW`.
      const CVSUB: usize = $cvsub;
      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = 1u16 << ($bits - 1);

      /// Per-pixel Y ramp + per-chroma-sample U / V ramp — low-packed native
      /// codes kept near the legal-range middle so the converted RGB stays in
      /// gamut and the native-vs-rowstage delta is the per-pixel rounding
      /// difference, not a clamp divergence.
      fn ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; CW * CH];
        let mut v = vec![0u16; CW * CH];
        for i in 0..SRC * SRC {
          y[i] = (MID as u32 + ((i as u32 * 37) % (MASK as u32 / 4))) as u16 & MASK;
        }
        for i in 0..CW * CH {
          u[i] =
            (MID as u32 + ((i as u32 * 53) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16 & MASK;
          v[i] =
            (MID as u32 + ((i as u32 * 41) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16 & MASK;
        }
        (y, u, v)
      }

      /// Crafted VARYING illegal-chroma fixture: extreme alternating chroma
      /// (full-scale vs zero) over a super-black->super-white Y ramp, so many
      /// 2x2 blocks straddle the RGB clamp. Here native (average-in-YUV,
      /// convert once) and row-stage (convert-then-average) genuinely diverge.
      fn out_of_gamut() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; CW * CH];
        let mut v = vec![0u16; CW * CH];
        for i in 0..SRC * SRC {
          y[i] = ((i as u32 * MASK as u32) / (SRC * SRC) as u32) as u16 & MASK;
        }
        for i in 0..CW * CH {
          let hi = i % 2 == 0;
          u[i] = if hi { MASK } else { 0 };
          v[i] = if hi { 0 } else { MASK };
        }
        (y, u, v)
      }

      /// Overrange-Y fixture: a legal payload OR-ed with the first bit ABOVE
      /// the native depth (`MASK + 1`; a no-op at 16-bit, where no such bit
      /// exists, leaving the legal full-scale payload). #334: the native-Y luma
      /// feed must DEPTH-MASK each sample (`& MASK`) like the colour kernels, so
      /// the dirty bit is dropped to the legal payload BEFORE binning — not
      /// binned dirty then clamped to `native_max`. Chroma stays legal.
      fn overrange_luma() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let (_, u, v) = ramp();
        let ovr = ((MASK / 3) & MASK) | MASK.wrapping_add(1);
        (vec![ovr; SRC * SRC], u, v)
      }

      /// Exact integer-ratio area mean (round-half-up) of an `in_w x in_h` u16
      /// plane down to `OUT x OUT`, binning each axis by its own ratio
      /// (`in_w / OUT` horizontally, `in_h / OUT` vertically). Reproduces the
      /// native tier's per-plane binning to full output resolution for any
      /// subsample direction.
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

      fn frame_le<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16]) -> $frame_le<'a> {
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

      /// Drive the LE source through a tier for the full output set (u8 RGB +
      /// u16 RGB + luma). `native` toggles the bin-then-convert native fast
      /// tier vs the convert-then-bin row-stage tier.
      fn run(y: &[u16], u: &[u16], v: &[u16], native: bool) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let (yl, ul, vl) = (as_le(y), as_le(u), as_le(v));
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
          $walker(&frame_le(&yl, &ul, &vl), FR, M, &mut sink).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the BE source through the native tier (the host-native-endian
      /// guard reference).
      fn native_be_run(y: &[u16], u: &[u16], v: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let (yb, ub, vb) = (as_be(y), as_be(u), as_be(v));
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
          $walker_be(&frame_be(&yb, &ub, &vb), FR, M, &mut sink).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// The bin-then-convert oracle: area-bin every source plane to OUTPUT
      /// resolution by its own subsample ratio (Y from `SRC x SRC`, chroma from
      /// `CW x CH`), then convert the full-output-width host-native planes ONCE
      /// through an identity-resolution high-bit `Yuv444pN` sink. That sink
      /// runs the SAME native-depth 4:4:4 kernels (and `(1 << BITS) - 1` clamp)
      /// the native tier finalizes with, so this is the exact ground truth the
      /// native tier must reproduce byte-for-byte — clamp included.
      fn oracle(y: &[u16], u: &[u16], v: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let yb = bin_to_out(y, SRC, SRC);
        let ub = bin_to_out(u, CW, CH);
        let vb = bin_to_out(v, CW, CH);
        let (yl, ul, vl) = (as_le(&yb), as_le(&ub), as_le(&vb));
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$oracle_marker>::new(OUT, OUT)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb_u16)
            .unwrap();
          let f = $oracle_frame::try_new(
            &yl, &ul, &vl, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
          )
          .unwrap();
          $oracle_walker(&f, FR, M, &mut sink).unwrap();
        }
        // Luma oracle computed INDEPENDENTLY of the sinker luma path: clamp the
        // binned Y to the native max, then narrow. Routing it through the sink
        // would mirror a clamp bug in that path instead of catching it.
        let luma: Vec<u8> = yb
          .iter()
          .map(|&by| (by.min(MASK) >> ($bits - 8)) as u8)
          .collect();
        (rgb, rgb_u16, luma)
      }

      fn max_delta_u8(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
          .zip(b)
          .map(|(&x, &y)| x.abs_diff(y))
          .max()
          .unwrap_or(0)
      }
      fn max_delta_u16(a: &[u16], b: &[u16]) -> u16 {
        a.iter()
          .zip(b)
          .map(|(&x, &y)| x.abs_diff(y))
          .max()
          .unwrap_or(0)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_equals_bin_then_convert_oracle() {
        // Ground truth: the native tier IS bin-then-convert. Every output must
        // match the independent (same-clamp) oracle EXACTLY.
        let (y, u, v) = ramp();
        let (n_rgb, n_rgb16, n_luma) = run(&y, &u, &v, true);
        let (o_rgb, o_rgb16, o_luma) = oracle(&y, &u, &v);
        assert_eq!(
          n_rgb, o_rgb,
          "u8 rgb must equal the bin-then-convert oracle"
        );
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
      fn native_within_tolerance_of_rowstage() {
        let (y, u, v) = ramp();
        let (n_rgb, n_rgb16, n_luma) = run(&y, &u, &v, true);
        let (r_rgb, r_rgb16, r_luma) = run(&y, &u, &v, false);

        // Luma: both tiers bin the SAME native Y stream and narrow, so it is
        // bit-identical.
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
        // The native tier de-interleaves the wire planes to host-native BEFORE
        // binning, so BE and LE sources produce identical output.
        let (y, u, v) = ramp();
        let le = run(&y, &u, &v, true);
        let be = native_be_run(&y, &u, &v);
        assert_eq!(be.0, le.0, "BE u8 colour must match LE");
        assert_eq!(be.1, le.1, "BE u16 colour must match LE");
        assert_eq!(be.2, le.2, "BE luma must match LE");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_be_equals_oracle() {
        // The BE native output equals the (host-native) bin-then-convert
        // oracle exactly — the host-native-endian regression in oracle terms.
        let (y, u, v) = ramp();
        let be = native_be_run(&y, &u, &v);
        let o = oracle(&y, &u, &v);
        assert_eq!(be.0, o.0, "BE u8 colour must equal the oracle");
        assert_eq!(be.1, o.1, "BE u16 colour must equal the oracle");
        assert_eq!(be.2, o.2, "BE luma must equal the oracle");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_masks_overrange_y() {
        // #334: an out-of-range Y must be DEPTH-MASKED (`& MASK`) like the
        // colour kernels — never binned dirty then CLAMPED to native-max (the
        // pre-fix path, which saturated every overrange luma to 255). The oracle
        // masks each source sample, then bins, then narrows.
        let (y, u, v) = overrange_luma();
        let (_, _, n_luma) = run(&y, &u, &v, true);
        let masked: Vec<u16> = y.iter().map(|&c| c & MASK).collect();
        let yb = bin_to_out(&masked, SRC, SRC);
        let expect: Vec<u8> = yb.iter().map(|&by| (by >> ($bits - 8)) as u8).collect();
        assert_eq!(
          n_luma, expect,
          "overrange binned Y must be depth-masked before narrowing, not clamped"
        );
        // The masked payload sits well below native-max, so luma is NOT the
        // saturated `MASK >> (BITS - 8)` the pre-fix clamp produced (at <16-bit).
        let masked_narrow = (((MASK / 3) & MASK) >> ($bits - 8)) as u8;
        assert!(
          n_luma.iter().all(|&l| l == masked_narrow),
          "all overrange luma must narrow the masked payload to {masked_narrow}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rowstage_luma_masks_overrange_y() {
        // Same depth-mask on the ROW-STAGE (with_native(false)) path: the shared
        // triple_resample luma emitter masks the de-interleaved Y to `& MASK`
        // before the `>> (BITS - 8)` narrow (#334), not clamp. Same oracle.
        let (y, u, v) = overrange_luma();
        let (_, _, r_luma) = run(&y, &u, &v, false);
        let masked: Vec<u16> = y.iter().map(|&c| c & MASK).collect();
        let yb = bin_to_out(&masked, SRC, SRC);
        let expect: Vec<u8> = yb.iter().map(|&by| (by >> ($bits - 8)) as u8).collect();
        assert_eq!(
          r_luma, expect,
          "row-stage overrange luma must be depth-masked before narrowing, not clamped"
        );
        let masked_narrow = (((MASK / 3) & MASK) >> ($bits - 8)) as u8;
        assert!(
          r_luma.iter().all(|&l| l == masked_narrow),
          "all row-stage overrange luma must narrow the masked payload to {masked_narrow}"
        );
      }

      /// A `CW x CH` chroma plane of a `0` / `MASK + 1` checkerboard. Every
      /// chroma area-bin (1x2 for 4:2:2, 2x1 for 4:4:0, 2x2 for 4:4:4) then
      /// holds an EQUAL split of `0` and `MASK + 1`, so the bin AVERAGE is
      /// `(MASK + 1) / 2 == MID` — a LEGAL code that SURVIVES the depth mask —
      /// whereas masking each source sample FIRST gives all-`0`. The two orders
      /// disagree, so the binned colour reveals whether the tier masks before or
      /// after binning (#334). (`MASK + 1` wraps to `0` at 16-bit, where no
      /// over-range bit exists; the checkerboard collapses to all-`0` and the
      /// equality below degenerates to a legal-input no-op.)
      fn dirty_chroma_checkerboard() -> Vec<u16> {
        let mut c = vec![0u16; CW * CH];
        for yy in 0..CH {
          for xx in 0..CW {
            c[yy * CW + xx] = if (xx + yy) % 2 == 0 {
              0
            } else {
              MASK.wrapping_add(1)
            };
          }
        }
        c
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_chroma_u_masks_overrange_before_bin() {
        // #334: a dirty (over-range) U sample must be DEPTH-MASKED per SOURCE
        // sample BEFORE the AreaStream bins it. With a 0 / MASK+1 chroma
        // checkerboard each chroma bin averages to MID (a legal, neutral code
        // that survives the mask), while masking each sample FIRST gives 0. The
        // native tier must reproduce the mask-then-bin-then-convert oracle
        // (chroma 0) on BOTH the LE and BE wire — never the avg-then-mask (MID)
        // value the pre-fix unmasked de-interleave produced.
        let (y, _, _) = ramp();
        let u = dirty_chroma_checkerboard();
        let v = vec![MID & MASK; CW * CH]; // legal neutral V (isolate U)
        let u_masked: Vec<u16> = u.iter().map(|&c| c & MASK).collect();

        // mask-then-bin-then-convert ground truth — the SAME-clamp oracle the
        // native tier reproduces exactly for legal input
        // (`native_equals_bin_then_convert_oracle`).
        let (o_rgb, o_rgb16, _) = oracle(&y, &u_masked, &v);

        let (n_rgb, n_rgb16, _) = run(&y, &u, &v, true);
        assert_eq!(
          n_rgb, o_rgb,
          "LE U: native colour must equal the per-sample-masked oracle"
        );
        assert_eq!(
          n_rgb16, o_rgb16,
          "LE U: native u16 colour must equal the per-sample-masked oracle"
        );

        let (be_rgb, be_rgb16, _) = native_be_run(&y, &u, &v);
        assert_eq!(
          be_rgb, o_rgb,
          "BE U: native colour must equal the per-sample-masked oracle"
        );
        assert_eq!(
          be_rgb16, o_rgb16,
          "BE U: native u16 colour must equal the per-sample-masked oracle"
        );

        // Non-vacuous below 16-bit: the avg-then-mask (pre-fix) oracle bins the
        // DIRTY U first (-> MID) then converts; it must DIFFER from the masked-0
        // result, proving the dirty bits move the binned code.
        if MASK != u16::MAX {
          let (bug_rgb, _, _) = oracle(&y, &u, &v);
          assert_ne!(
            n_rgb, bug_rgb,
            "LE U: fixed colour must NOT equal the avg-then-mask oracle"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_chroma_v_masks_overrange_before_bin() {
        // The V twin of `native_chroma_u_masks_overrange_before_bin` — the fix
        // masks the V de-interleave feed independently of U.
        let (y, _, _) = ramp();
        let u = vec![MID & MASK; CW * CH]; // legal neutral U (isolate V)
        let v = dirty_chroma_checkerboard();
        let v_masked: Vec<u16> = v.iter().map(|&c| c & MASK).collect();

        let (o_rgb, o_rgb16, _) = oracle(&y, &u, &v_masked);

        let (n_rgb, n_rgb16, _) = run(&y, &u, &v, true);
        assert_eq!(
          n_rgb, o_rgb,
          "LE V: native colour must equal the per-sample-masked oracle"
        );
        assert_eq!(
          n_rgb16, o_rgb16,
          "LE V: native u16 colour must equal the per-sample-masked oracle"
        );

        let (be_rgb, be_rgb16, _) = native_be_run(&y, &u, &v);
        assert_eq!(
          be_rgb, o_rgb,
          "BE V: native colour must equal the per-sample-masked oracle"
        );
        assert_eq!(
          be_rgb16, o_rgb16,
          "BE V: native u16 colour must equal the per-sample-masked oracle"
        );

        if MASK != u16::MAX {
          let (bug_rgb, _, _) = oracle(&y, &u, &v);
          assert_ne!(
            n_rgb, bug_rgb,
            "LE V: fixed colour must NOT equal the avg-then-mask oracle"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_matches_inter_area_oracle() {
        // cv2 INTER_AREA parity for luma: the per-axis area bin of the native Y
        // plane, narrowed (luma derives from Y, never from RGB).
        let (y, u, v) = ramp();
        let (_, _, n_luma) = run(&y, &u, &v, true);
        let y_ref = bin_to_out(&y, SRC, SRC);
        let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_eq!(
          n_luma, luma_ref,
          "native luma must equal the INTER_AREA Y bin"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_gamut_native_vs_rowstage_pinned() {
        let (y, u, v) = out_of_gamut();
        let (n_rgb, _, n_luma) = run(&y, &u, &v, true);
        let (r_rgb, _, r_luma) = run(&y, &u, &v, false);
        // Luma stays bit-identical even out of gamut (native Y bin, unaffected
        // by the colour clamp).
        assert_eq!(n_luma, r_luma, "luma stays bit-identical out of gamut");
        let d = max_delta_u8(&n_rgb, &r_rgb);
        assert!(
          d > TOL_U8,
          "crafted out-of-gamut case must diverge beyond the in-gamut \
           tolerance {TOL_U8}, got {d}"
        );
        assert!(d < u8::MAX, "out-of-gamut delta stays bounded, got {d}");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_default_matches_explicit_true() {
        // `with_native` defaults to true, so the default sink and an explicit
        // `with_native(true)` sink must agree byte-for-byte.
        let (y, u, v) = ramp();
        let e = run(&y, &u, &v, true);
        let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
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
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame_le(&yl, &ul, &vl), FR, M, &mut sink).unwrap();
        }
        assert_eq!(rgb, e.0, "default u8 rgb must match explicit native");
        assert_eq!(rgb_u16, e.1, "default u16 rgb must match explicit native");
        assert_eq!(luma, e.2, "default luma must match explicit native");
      }

      // ---- atomicity ----------------------------------------------------

      /// The chroma source-row index feeding luma row `idx` (cadence `CVSUB`).
      fn chroma_row(idx: usize) -> usize {
        (idx / CVSUB) * CW
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_sequence_first_row_rejected_and_does_not_poison_retry() {
        let (y, u, v) = ramp();
        let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
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
        let (yr, cr) = (3 * SRC, chroma_row(3));
        let err = sink
          .process($row::new(
            &yl[yr..yr + SRC],
            &ul[cr..cr + CW],
            &vl[cr..cr + CW],
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
        // The rejected first row stored NO frozen-output snapshot (the
        // pre-freeze first-row check fires BEFORE the freeze), so attaching a
        // NEW output and retrying row 0 must succeed.
        let mut rgb = vec![0u8; OUT * OUT * 3];
        sink.set_rgb(&mut rgb).unwrap();
        sink
          .process($row::new(&yl[..SRC], &ul[..CW], &vl[..CW], 0, M, FR))
          .expect("row 0 must succeed after a rejected out-of-sequence first row");
      }

      /// A mid-frame output-set change on a chroma-bearing row must be rejected
      /// by the native preflight's frozen-output check BEFORE the source-scratch
      /// alloc — `ResampleOutputsChanged`, never `AllocationFailed`. The
      /// source-scratch failpoint is armed on the reserve the changed row WOULD
      /// reach: with the preflight first the frozen check fires and the
      /// failpoint is never consumed.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn frozen_mid_frame_change_rejected_before_scratch_alloc() {
        let (y, u, v) = ramp();
        let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
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
          let cr = chroma_row(r);
          sink
            .process($row::new(
              &yl[r * SRC..(r + 1) * SRC],
              &ul[cr..cr + CW],
              &vl[cr..cr + CW],
              r,
              M,
              FR,
            ))
            .expect("luma-only rows freeze a luma-only output set");
        }
        // Attach u16 colour mid-frame, changing the output set, and arm the
        // source-scratch failpoint on the reserve the changed row reaches.
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        crate::sinker::mixed::arm_planar_hb_native_alloc_failure();
        let cr = chroma_row(2);
        let err = sink
          .process($row::new(
            &yl[2 * SRC..3 * SRC],
            &ul[cr..cr + CW],
            &vl[cr..cr + CW],
            2,
            M,
            FR,
          ))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "mid-frame output change must reject as ResampleOutputsChanged \
           before the source-scratch alloc, got {err:?}"
        );
        assert!(
          rgb_u16.iter().all(|&b| b == 0),
          "rejected mid-frame-change row touched the new colour output"
        );
        // The failpoint is single-shot and must NOT have been consumed: prove
        // it via a fresh in-sequence colour row that DOES fire it.
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
          .process($row::new(&yl[..SRC], &ul[..CW], &vl[..CW], 0, M, FR))
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

      /// Flipping `set_native(true) -> false` mid-frame must reject as the
      /// deterministic `NativeRouteChanged` BEFORE either tier consumes the
      /// row.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_to_rowstage_route_flip_mid_frame_rejected() {
        let (y, u, v) = ramp();
        let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
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
          .process($row::new(&yl[..SRC], &ul[..CW], &vl[..CW], 0, M, FR))
          .expect("native row 0 freezes the route and succeeds");
        sink.set_native(false);
        let cr = chroma_row(1);
        let err = sink
          .process($row::new(
            &yl[SRC..2 * SRC],
            &ul[cr..cr + CW],
            &vl[cr..cr + CW],
            1,
            M,
            FR,
          ))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "a native -> row-stage mid-frame route flip must reject as \
           NativeRouteChanged, got {err:?}"
        );
      }
    }
  };
}

// 4:2:2: chroma `w/2 x h` — half width, full height.
yuv_planar_hb_native_suite!(
  yuv422p9,
  Yuv422p9,
  Yuv422p9LeFrame,
  Yuv422p9BeFrame,
  Yuv422p9Row,
  yuv422p9_to,
  yuv422p9_to_endian,
  Yuv444p9,
  Yuv444p9LeFrame,
  yuv444p9_to,
  SRC / 2,
  SRC,
  1,
  9,
);
yuv_planar_hb_native_suite!(
  yuv422p10,
  Yuv422p10,
  Yuv422p10LeFrame,
  Yuv422p10BeFrame,
  Yuv422p10Row,
  yuv422p10_to,
  yuv422p10_to_endian,
  Yuv444p10,
  Yuv444p10LeFrame,
  yuv444p10_to,
  SRC / 2,
  SRC,
  1,
  10,
);
yuv_planar_hb_native_suite!(
  yuv422p12,
  Yuv422p12,
  Yuv422p12LeFrame,
  Yuv422p12BeFrame,
  Yuv422p12Row,
  yuv422p12_to,
  yuv422p12_to_endian,
  Yuv444p12,
  Yuv444p12LeFrame,
  yuv444p12_to,
  SRC / 2,
  SRC,
  1,
  12,
);
yuv_planar_hb_native_suite!(
  yuv422p14,
  Yuv422p14,
  Yuv422p14LeFrame,
  Yuv422p14BeFrame,
  Yuv422p14Row,
  yuv422p14_to,
  yuv422p14_to_endian,
  Yuv444p14,
  Yuv444p14LeFrame,
  yuv444p14_to,
  SRC / 2,
  SRC,
  1,
  14,
);
yuv_planar_hb_native_suite!(
  yuv422p16,
  Yuv422p16,
  Yuv422p16LeFrame,
  Yuv422p16BeFrame,
  Yuv422p16Row,
  yuv422p16_to,
  yuv422p16_to_endian,
  Yuv444p16,
  Yuv444p16LeFrame,
  yuv444p16_to,
  SRC / 2,
  SRC,
  1,
  16,
);

// 4:4:4: chroma `w x h` — identical to Y.
yuv_planar_hb_native_suite!(
  yuv444p9,
  Yuv444p9,
  Yuv444p9LeFrame,
  Yuv444p9BeFrame,
  Yuv444p9Row,
  yuv444p9_to,
  yuv444p9_to_endian,
  Yuv444p9,
  Yuv444p9LeFrame,
  yuv444p9_to,
  SRC,
  SRC,
  1,
  9,
);
yuv_planar_hb_native_suite!(
  yuv444p10,
  Yuv444p10,
  Yuv444p10LeFrame,
  Yuv444p10BeFrame,
  Yuv444p10Row,
  yuv444p10_to,
  yuv444p10_to_endian,
  Yuv444p10,
  Yuv444p10LeFrame,
  yuv444p10_to,
  SRC,
  SRC,
  1,
  10,
);
yuv_planar_hb_native_suite!(
  yuv444p12,
  Yuv444p12,
  Yuv444p12LeFrame,
  Yuv444p12BeFrame,
  Yuv444p12Row,
  yuv444p12_to,
  yuv444p12_to_endian,
  Yuv444p12,
  Yuv444p12LeFrame,
  yuv444p12_to,
  SRC,
  SRC,
  1,
  12,
);
yuv_planar_hb_native_suite!(
  yuv444p14,
  Yuv444p14,
  Yuv444p14LeFrame,
  Yuv444p14BeFrame,
  Yuv444p14Row,
  yuv444p14_to,
  yuv444p14_to_endian,
  Yuv444p14,
  Yuv444p14LeFrame,
  yuv444p14_to,
  SRC,
  SRC,
  1,
  14,
);
yuv_planar_hb_native_suite!(
  yuv444p16,
  Yuv444p16,
  Yuv444p16LeFrame,
  Yuv444p16BeFrame,
  Yuv444p16Row,
  yuv444p16_to,
  yuv444p16_to_endian,
  Yuv444p16,
  Yuv444p16LeFrame,
  yuv444p16_to,
  SRC,
  SRC,
  1,
  16,
);

// 4:4:0: chroma `w x h/2` — full width, half height.
yuv_planar_hb_native_suite!(
  yuv440p10,
  Yuv440p10,
  Yuv440p10LeFrame,
  Yuv440p10BeFrame,
  Yuv440p10Row,
  yuv440p10_to,
  yuv440p10_to_endian,
  Yuv444p10,
  Yuv444p10LeFrame,
  yuv444p10_to,
  SRC,
  SRC / 2,
  2,
  10,
);
yuv_planar_hb_native_suite!(
  yuv440p12,
  Yuv440p12,
  Yuv440p12LeFrame,
  Yuv440p12BeFrame,
  Yuv440p12Row,
  yuv440p12_to,
  yuv440p12_to_endian,
  Yuv444p12,
  Yuv444p12LeFrame,
  yuv444p12_to,
  SRC,
  SRC / 2,
  2,
  12,
);

/// A luma-only high-bit non-4:2:0 native sink must NOT plan or allocate any
/// chroma state (else luma-only resampling depends on an unused chroma
/// allocation and can fail under memory pressure before producing luma).
/// Armed with the chroma-planning failpoint at the standard integer-ratio
/// geometry: a luma-only row leaves it unconsumed (so the run succeeds),
/// while a colour row reaches chroma planning and the failpoint fires.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_native_skips_chroma_planning() {
  use crate::source::{Yuv422p10, yuv422p10_to};
  let y = vec![1u16 << 9; SRC * SRC];
  let u = vec![1u16 << 9; (SRC / 2) * SRC];
  let v = vec![1u16 << 9; (SRC / 2) * SRC];
  let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
  let frame = Yuv422p10LeFrame::try_new(
    &yl,
    &ul,
    &vl,
    SRC as u32,
    SRC as u32,
    SRC as u32,
    (SRC / 2) as u32,
    (SRC / 2) as u32,
  )
  .unwrap();

  crate::sinker::mixed::arm_planar_hb_native_chroma_failure();

  // Luma-only: the chroma failpoint is armed but never reached -> Ok.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink = MixedSinker::<Yuv422p10, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_native(true)
    .with_luma(&mut luma)
    .unwrap();
    yuv422p10_to(&frame, FR, M, &mut sink).expect("luma-only native must not plan chroma");
  }

  // Colour: the still-armed failpoint fires at chroma planning -> Err. This
  // both proves the failpoint is wired to chroma planning and consumes the arm
  // so it cannot leak to another test on this thread.
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Yuv422p10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
  assert!(
    yuv422p10_to(&frame, FR, M, &mut sink).is_err(),
    "colour native must reach chroma planning (the armed failpoint fires)"
  );
}

/// RFC #238 — the high-bit non-4:2:0 native delegate is preflight-transactional.
/// A colour-capability REBUILD (frame 2 attaches colour to a frame-1 luma-only
/// join) whose CHROMA-PLAN build fails must surface the typed `AllocationFailed`
/// while leaving the cached luma-only join AND the output-set freeze INTACT
/// (uncommitted), so retrying row 0 with YET ANOTHER output attached is accepted
/// — a committed `{luma, hsv}` freeze would reject it as `ResampleOutputsChanged`.
/// The pre-fix delegate froze the output set and eager-cleared the join BEFORE
/// this fallible rebuild, corrupting the retry.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_colour_capability_rebuild_chroma_oom_is_transactional() {
  use crate::source::Yuv444p16Row;
  // 4:4:4: chroma is full width (CW == SRC), a chroma row per Y row.
  let y = vec![1u16 << 15; SRC * SRC];
  let u = vec![1u16 << 15; SRC * SRC];
  let v = vec![1u16 << 15; SRC * SRC];
  let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
  let row = |r: usize| {
    Yuv444p16Row::new(
      &yl[r * SRC..(r + 1) * SRC],
      &ul[r * SRC..(r + 1) * SRC],
      &vl[r * SRC..(r + 1) * SRC],
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
    MixedSinker::<Yuv444p16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
  // chroma-plan failpoint so that rebuild fails.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  crate::sinker::mixed::arm_planar_hb_native_chroma_failure();
  let err = sink.process(row(0)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the colour-capability rebuild chroma-plan OOM must surface AllocationFailed, got {err:?}"
  );

  // The freeze was never committed and the cached join stayed intact: attaching
  // ANOTHER output (rgb) and retrying row 0 is ACCEPTED — the changed output set
  // proves the freeze was rolled back (a committed `{luma, hsv}` would reject
  // this as ResampleOutputsChanged).
  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(row(0))
    .expect("row 0 with a changed output set must succeed after a rebuild chroma OOM");
  // Drive the rest of the frame through the recovered join: a fully-recovered
  // join emits real colour output (mid-gray input -> non-zero RGB), proving the
  // rebuild produced a coherent join, not merely that row 0 was accepted.
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

/// RFC #238 — the source-scratch failpoint companion of
/// [`native_colour_capability_rebuild_chroma_oom_is_transactional`]: the same
/// colour-capability rebuild, but the fallible step that fails is the
/// SOURCE-SCRATCH grow on the rebuilt (local) join. It too must leave the cached
/// join + output-set freeze uncommitted so a retry with a changed output set is
/// accepted.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_colour_capability_rebuild_src_scratch_oom_is_transactional() {
  use crate::source::Yuv444p16Row;
  let y = vec![1u16 << 15; SRC * SRC];
  let u = vec![1u16 << 15; SRC * SRC];
  let v = vec![1u16 << 15; SRC * SRC];
  let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
  let row = |r: usize| {
    Yuv444p16Row::new(
      &yl[r * SRC..(r + 1) * SRC],
      &ul[r * SRC..(r + 1) * SRC],
      &vl[r * SRC..(r + 1) * SRC],
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
    MixedSinker::<Yuv444p16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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

  // Frame 2: attach HSV -> rebuild WITH chroma; arm the source-scratch failpoint
  // so the rebuilt join's Y de-interleave scratch grow refuses.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  crate::sinker::mixed::arm_planar_hb_native_alloc_failure();
  let err = sink.process(row(0)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the colour-capability rebuild source-scratch OOM must surface AllocationFailed, got {err:?}"
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

/// RFC #238 — a LATER pre-feed grow failing is atomic too. Yuv444p16 carries BOTH
/// a u8 colour scratch (`rgb`) and a native-depth u16 colour scratch
/// (`rgb_u16`): on the colour-capability rebuild row the u8 `rgb_scratch` grows
/// FIRST and then the u16 `rgb_scratch_u16` grow refuses. The already-grown
/// `rgb_scratch` keeps its capacity (join / freeze / output-atomic, not
/// memory-exact), but the cached join + `resample_outputs` freeze + output frame
/// stay untouched, so a retry with yet another output attached is accepted and
/// drives real output.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_colour_capability_rebuild_rgb_u16_scratch_oom_is_transactional() {
  use crate::source::Yuv444p16Row;
  let y = vec![1u16 << 15; SRC * SRC];
  let u = vec![1u16 << 15; SRC * SRC];
  let v = vec![1u16 << 15; SRC * SRC];
  let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
  let row = |r: usize| {
    Yuv444p16Row::new(
      &yl[r * SRC..(r + 1) * SRC],
      &ul[r * SRC..(r + 1) * SRC],
      &vl[r * SRC..(r + 1) * SRC],
      r,
      M,
      FR,
    )
  };

  let mut luma = vec![0u8; OUT * OUT];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let (mut hh, mut ss, mut vv) = (
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
  );
  let mut sink =
    MixedSinker::<Yuv444p16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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

  // Frame 2: attach BOTH u8 (rgb) and u16 (rgb_u16) colour -> the rebuild grows
  // `rgb_scratch` then `rgb_scratch_u16`; arm the LATER (u16) grow to refuse.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_rgb(&mut rgb).unwrap();
  sink.set_rgb_u16(&mut rgb_u16).unwrap();
  crate::sinker::mixed::arm_planar_hb_native_rgb_u16_scratch_failure();
  let err = sink.process(row(0)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the u16 colour-scratch OOM (after the u8 scratch grew) must surface AllocationFailed, got {err:?}"
  );

  // The freeze was never committed: attaching hsv and retrying row 0 is accepted.
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  sink
    .process(row(0))
    .expect("row 0 with a changed output set must succeed after a u16-scratch OOM");
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

/// RFC #238 — a LATER SOURCE-plane grow failing is atomic too. On the 4:4:4
/// colour-capability rebuild row all of `y_src` / `u_src` / `v_src` grow; the
/// failpoint's skip counter refuses the SECOND source grow (`u_src`) after
/// `y_src` already grew. The already-grown `y_src` keeps its capacity, but the
/// cached join + freeze + output stay untouched, so a changed-output retry is
/// accepted and drives real output.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_colour_capability_rebuild_later_source_scratch_oom_is_transactional() {
  use crate::source::Yuv444p16Row;
  let y = vec![1u16 << 15; SRC * SRC];
  let u = vec![1u16 << 15; SRC * SRC];
  let v = vec![1u16 << 15; SRC * SRC];
  let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
  let row = |r: usize| {
    Yuv444p16Row::new(
      &yl[r * SRC..(r + 1) * SRC],
      &ul[r * SRC..(r + 1) * SRC],
      &vl[r * SRC..(r + 1) * SRC],
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
    MixedSinker::<Yuv444p16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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

  // Frame 2: attach HSV -> rebuild WITH chroma (y_src, u_src, v_src all grow on
  // row 0); arm the SECOND source grow (`u_src`, after `y_src`) to refuse.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  crate::sinker::mixed::arm_planar_hb_native_alloc_failure_after(1);
  let err = sink.process(row(0)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the later (u_src) source-scratch OOM must surface AllocationFailed, got {err:?}"
  );

  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(row(0))
    .expect("row 0 with a changed output set must succeed after a later source-scratch OOM");
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
