//! Fused-downscale coverage for the high-bit **planar** 4:2:0 YUV NATIVE
//! fast tier — `Yuv420p10` / `Yuv420p12` / `Yuv420p14` / `Yuv420p16`
//! (LE + BE wire), the `u16` twin of the 8-bit
//! [`yuv420p_process_native`](crate::sinker::mixed::planar_8bit::yuv420p_process_native).
//!
//! The native tier bins the host-native Y / U / V planes straight to the
//! output grid and converts ONCE per output row at output width (4:4:4
//! kernels), vs the row-stage tier
//! ([`packed_yuv422_triple_resample`](crate::sinker::mixed::packed_yuv422_triple_resample)),
//! which converts each source row at source width then bins. The tiers
//! differ in colour SEMANTICS (native averages in YUV then converts;
//! row-stage converts then averages in RGB), so native is NOT byte-
//! identical to row-stage — only within a small tolerance in-gamut, with
//! documented out-of-gamut divergence. Luma is bit-identical (both bin the
//! same native Y stream then narrow `>> (BITS - 8)`).
//!
//! Per format (LE + BE):
//! - tolerance/parity: same source through `with_native(true)` and
//!   `with_native(false)`, asserting per-channel `|native - rowstage| <= N`
//!   in-gamut and LUMA bit-identical. The row-stage tier IS the cv2
//!   INTER_AREA oracle (convert-then-area-bin), so the within-tolerance
//!   comparison to it is the INTER_AREA parity check; luma additionally
//!   matches the direct 2x2-block area mean of the native Y. The
//!   out-of-gamut delta is pinned on a crafted illegal-chroma case.
//! - independent-kernel guards (#37): a uniform-gray downscale leaves every
//!   colour output unchanged; a saturated-chroma case shows the u16 and u8
//!   outputs differing as expected.
//! - atomicity: OOS-first-row, frozen-mid-frame (armed alloc failpoint),
//!   and OOS-retry-after-a-recoverable-alloc-failure — each the
//!   deterministic typed error, never AllocationFailed.

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
/// independently per output pixel. The empirical in-gamut maximum on the
/// mid-range ramp fixture here is 1 (u8) / 0 (u16); this bound pins that
/// observed max plus a 1-LSB margin for cross-platform SIMD-vs-scalar
/// rounding. Out-of-gamut content diverges further (observed max 7 on the
/// crafted illegal-chroma fixture) and is pinned separately by
/// `out_of_gamut_native_vs_rowstage_pinned`, which also asserts the
/// out-of-gamut delta exceeds this in-gamut bound.
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

macro_rules! yuv420p_high_bit_native_suite {
  (
    $mod:ident, $frame_le:ident, $frame_be:ident, $marker:ident, $row:ident,
    $walker:ident, $walker_be:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $row, $walker, $walker_be};

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = (1u16 << ($bits - 1));

      /// Per-pixel Y ramp + per-chroma-sample U / V ramp — low-packed
      /// native codes, every code in-gamut-ish (real chroma but not the
      /// crafted illegal extremes).
      fn ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; CW * CH];
        let mut v = vec![0u16; CW * CH];
        for i in 0..SRC * SRC {
          // Keep Y near the legal-range middle so the converted RGB stays
          // in gamut and the native-vs-rowstage delta is the per-pixel
          // rounding difference, not a clamp divergence.
          y[i] = (MID as u32 + ((i as u32 * 37) % (MASK as u32 / 4))) as u16 & MASK;
        }
        for i in 0..CW * CH {
          // Chroma kept within a modest band around neutral (mid) so the
          // result stays in gamut.
          u[i] =
            (MID as u32 + ((i as u32 * 53) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16 & MASK;
          v[i] =
            (MID as u32 + ((i as u32 * 41) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16 & MASK;
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

      /// Re-encode a host-native u16 slice as host-independent LE-wire
      /// byte storage (the `*LeFrame` plane contract). Raw host-native u16
      /// is only valid LE-wire on a little-endian host; this makes the LE
      /// fixtures host-endian-independent (a no-op on LE, a byte swap on BE).
      fn as_le(host: &[u16]) -> Vec<u16> {
        host
          .iter()
          .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
          .collect()
      }

      /// Re-encode a host-native u16 slice as host-independent BE-wire
      /// byte storage (the `*BeFrame` plane contract).
      fn as_be(host: &[u16]) -> Vec<u16> {
        host
          .iter()
          .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
          .collect()
      }

      /// Drive the native tier (LE) for the given output set. The host-
      /// native fixtures are re-encoded to LE-wire storage so the LeFrame
      /// plane contract holds on any host endianness.
      fn native_run(y: &[u16], u: &[u16], v: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
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
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&yl, &ul, &vl), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the row-stage tier (LE) for the given output set. The host-
      /// native fixtures are re-encoded to LE-wire storage so the LeFrame
      /// plane contract holds on any host endianness.
      fn rowstage_run(y: &[u16], u: &[u16], v: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
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
          .with_native(false)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&yl, &ul, &vl), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the row-stage tier (BE) for the given output set — the BE
      /// twin of `rowstage_run`, with `.with_native(false)`. This is the
      /// correct host-independent reference (it de-interleaves BE-wire bytes
      /// to host-native before converting), so on a big-endian host the
      /// unfixed native tier diverges from it; on a little-endian host this
      /// exercises the BE wire path and stays within tolerance.
      fn rowstage_be_run(y: &[u16], u: &[u16], v: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
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
          .with_native(false)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker_be(&frame_be(&yb, &ub, &vb), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_within_tolerance_of_rowstage_and_luma_bit_identical() {
        let (y, u, v) = ramp();
        let (n_rgb, n_rgb16, n_luma) = native_run(&y, &u, &v);
        let (r_rgb, r_rgb16, r_luma) = rowstage_run(&y, &u, &v);

        // Luma: both tiers bin the SAME native Y stream and narrow
        // `>> (BITS - 8)`, so it is bit-identical.
        assert_eq!(n_luma, r_luma, "luma must be bit-identical across tiers");

        // u8 colour: within tolerance in gamut.
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

        // u16 colour: within the same relative tolerance, scaled to the
        // bit depth (`TOL_U8 << (BITS - 8)`).
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
        // The native tier de-interleaves the wire planes to host-native
        // BEFORE binning, so BE and LE sources produce identical output.
        let (y, u, v) = ramp();
        let (n_rgb_le, n_rgb16_le, n_luma_le) = native_run(&y, &u, &v);

        let (yb, ub, vb) = (as_be(&y), as_be(&u), as_be(&v));
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
          $walker_be(&frame_be(&yb, &ub, &vb), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(rgb, n_rgb_le, "BE u8 colour must match LE");
        assert_eq!(rgb_u16, n_rgb16_le, "BE u16 colour must match LE");
        assert_eq!(luma, n_luma_le, "BE luma must match LE");
      }

      /// The host-native-endian regression: BE native vs the correct BE
      /// row-stage reference on the SAME ramp, within the SAME tolerances as
      /// the LE parity test. Comparing BE native to the BE ROW-STAGE (not to
      /// LE native) is what catches the R1/R2 host-native-endian bug: on a
      /// big-endian host the unfixed native tier diverges from the correct
      /// BE row-stage beyond tolerance, while the fixed tier stays within it.
      /// (On this little-endian host this exercises the BE WIRE path and
      /// passes within tolerance — it only TRIPS the host-native bug on a
      /// big-endian target/CI, which is the point of the regression.)
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_be_within_tolerance_of_rowstage_be() {
        let (y, u, v) = ramp();
        let (yb, ub, vb) = (as_be(&y), as_be(&u), as_be(&v));
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
          $walker_be(&frame_be(&yb, &ub, &vb), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let (r_rgb, r_rgb16, r_luma) = rowstage_be_run(&y, &u, &v);

        // Luma: both tiers bin the SAME native Y stream and narrow
        // `>> (BITS - 8)`, so it is bit-identical.
        assert_eq!(n_luma, r_luma, "BE luma must be bit-identical across tiers");

        // u8 colour: within tolerance in gamut.
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

        // u16 colour: within the same relative tolerance, scaled to the
        // bit depth (`TOL_U8 << (BITS - 8)`).
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
        // cv2 INTER_AREA parity for luma: the area-bin of the native Y
        // plane, narrowed.
        let (y, u, v) = ramp();
        let (_, _, n_luma) = native_run(&y, &u, &v);
        let y_ref = block_mean_2x2_u16(&y);
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
      fn native_luma_masks_overrange_y() {
        // #334: an out-of-range Y (a legal payload plus the first bit ABOVE the
        // native depth; a no-op at 16-bit) must be DEPTH-MASKED (`& MASK`) like
        // the colour kernels, NOT binned dirty then CLAMPED to native-max (the
        // pre-fix path saturated every overrange luma to 255). Chroma stays legal.
        let (_, u, v) = ramp();
        let ovr = ((MASK / 3) & MASK) | MASK.wrapping_add(1);
        let y = vec![ovr; SRC * SRC];
        let (_, _, n_luma) = native_run(&y, &u, &v);
        let masked: Vec<u16> = y.iter().map(|&c| c & MASK).collect();
        let y_ref = block_mean_2x2_u16(&masked);
        let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_eq!(
          n_luma, luma_ref,
          "overrange binned Y must be depth-masked before narrowing, not clamped"
        );
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
        // packed_yuv422_triple_resample luma emitter masks the de-interleaved Y
        // to `& MASK` before the `>> (BITS - 8)` narrow (#334), not clamp.
        let (_, u, v) = ramp();
        let ovr = ((MASK / 3) & MASK) | MASK.wrapping_add(1);
        let y = vec![ovr; SRC * SRC];
        let (_, _, r_luma) = rowstage_run(&y, &u, &v);
        let masked: Vec<u16> = y.iter().map(|&c| c & MASK).collect();
        let y_ref = block_mean_2x2_u16(&masked);
        let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_eq!(
          r_luma, luma_ref,
          "row-stage overrange luma must be depth-masked before narrowing, not clamped"
        );
        let masked_narrow = (((MASK / 3) & MASK) >> ($bits - 8)) as u8;
        assert!(
          r_luma.iter().all(|&l| l == masked_narrow),
          "all row-stage overrange luma must narrow the masked payload to {masked_narrow}"
        );
      }

      /// Drive the native tier (LE) at a 2x-further downscale to `OUT2 x OUT2`,
      /// where the 4:2:0 chroma plane (`CW x CH`) is GENUINELY area-binned 2x2
      /// -> 1. At the suite's default `OUT` the 4:2:0 chroma is a 1:1
      /// passthrough (chroma `CW == OUT`) and never averaged, so it cannot
      /// exercise the dirty-chroma fix; only once the AreaStream averages
      /// DISTINCT chroma samples does `avg(dirty) & mask` diverge from
      /// `avg(dirty & mask)` (#334).
      fn native_run_2x2(y: &[u16], u: &[u16], v: &[u16]) -> (Vec<u8>, Vec<u16>) {
        const OUT2: usize = 2;
        let (yl, ul, vl) = (as_le(y), as_le(u), as_le(v));
        let mut rgb = vec![0u8; OUT2 * OUT2 * 3];
        let mut rgb_u16 = vec![0u16; OUT2 * OUT2 * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT2, OUT2),
          )
          .unwrap()
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walker(&frame(&yl, &ul, &vl), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16)
      }

      /// BE twin of [`native_run_2x2`]. The native tier de-interleaves AND (with
      /// the fix) masks the wire to host-native BEFORE binning, so a correct fix
      /// makes this byte-identical to the LE run.
      fn native_be_run_2x2(y: &[u16], u: &[u16], v: &[u16]) -> (Vec<u8>, Vec<u16>) {
        const OUT2: usize = 2;
        let (yb, ub, vb) = (as_be(y), as_be(u), as_be(v));
        let mut rgb = vec![0u8; OUT2 * OUT2 * 3];
        let mut rgb_u16 = vec![0u16; OUT2 * OUT2 * 3];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT2, OUT2),
          )
          .unwrap()
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walker_be(&frame_be(&yb, &ub, &vb), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16)
      }

      /// A `CW x CH` chroma plane of a `0` / `MASK + 1` checkerboard. Each 2x2
      /// area-bin (at `OUT2`) then holds two `0`s and two `MASK + 1`s, so the
      /// bin AVERAGE is `(MASK + 1) / 2 == MID` — a LEGAL code that SURVIVES the
      /// depth mask — whereas masking each source sample FIRST gives all-`0`.
      /// The two orders disagree, so the binned colour reveals whether the tier
      /// masks before or after binning. (`MASK + 1` wraps to `0` at 16-bit,
      /// where no over-range bit exists; the checkerboard collapses to all-`0`
      /// and the equality below degenerates to a legal-input no-op.)
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
        // checkerboard, bin-then-mask (the pre-fix order) survives as MID
        // (neutral), while mask-then-bin is 0 — the native tier must decode the
        // mask-then-bin (chroma 0) colour, on BOTH the LE and BE wire.
        let (y, _, _) = ramp();
        let u = dirty_chroma_checkerboard();
        let v = vec![MID & MASK; CW * CH]; // legal neutral V (isolate U)
        let u_masked: Vec<u16> = u.iter().map(|&c| c & MASK).collect();

        // Per-sample-masked reference: feed the already-masked (all-0) U. With
        // the fix the native tier decodes the dirty U identically — it masks
        // each sample to 0 before binning.
        let (ref_rgb, ref_rgb16) = native_run_2x2(&y, &u_masked, &v);

        let (le_rgb, le_rgb16) = native_run_2x2(&y, &u, &v);
        assert_eq!(
          le_rgb, ref_rgb,
          "LE U: native colour must equal the per-sample-masked (chroma 0) \
           reference, not the avg-then-mask (chroma MID) value"
        );
        assert_eq!(
          le_rgb16, ref_rgb16,
          "LE U: u16 colour must equal the per-sample-masked reference"
        );

        let (be_rgb, be_rgb16) = native_be_run_2x2(&y, &u, &v);
        assert_eq!(
          be_rgb, ref_rgb,
          "BE U: native colour must equal the per-sample-masked reference"
        );
        assert_eq!(
          be_rgb16, ref_rgb16,
          "BE U: u16 colour must equal the per-sample-masked reference"
        );

        // Non-vacuous below 16-bit: the avg-then-mask (pre-fix) colour decodes
        // the neutral MID a 0/(MASK+1) bin averages to; it must DIFFER from the
        // masked-0 result, proving the dirty bits actually move the binned code.
        if MASK != u16::MAX {
          let u_avgmask = vec![MID & MASK; CW * CH];
          let (bug_rgb, _) = native_run_2x2(&y, &u_avgmask, &v);
          assert_ne!(
            le_rgb, bug_rgb,
            "LE U: the fixed (masked-0) colour must NOT equal the avg-then-mask \
             (chroma MID) colour — else the test could not distinguish the bug"
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

        let (ref_rgb, ref_rgb16) = native_run_2x2(&y, &u, &v_masked);

        let (le_rgb, le_rgb16) = native_run_2x2(&y, &u, &v);
        assert_eq!(
          le_rgb, ref_rgb,
          "LE V: native colour must equal the per-sample-masked (chroma 0) \
           reference, not the avg-then-mask (chroma MID) value"
        );
        assert_eq!(
          le_rgb16, ref_rgb16,
          "LE V: u16 colour must equal the per-sample-masked reference"
        );

        let (be_rgb, be_rgb16) = native_be_run_2x2(&y, &u, &v);
        assert_eq!(
          be_rgb, ref_rgb,
          "BE V: native colour must equal the per-sample-masked reference"
        );
        assert_eq!(
          be_rgb16, ref_rgb16,
          "BE V: u16 colour must equal the per-sample-masked reference"
        );

        if MASK != u16::MAX {
          let v_avgmask = vec![MID & MASK; CW * CH];
          let (bug_rgb, _) = native_run_2x2(&y, &u, &v_avgmask);
          assert_ne!(
            le_rgb, bug_rgb,
            "LE V: the fixed (masked-0) colour must NOT equal the avg-then-mask \
             (chroma MID) colour — else the test could not distinguish the bug"
          );
        }
      }

      /// Crafted VARYING illegal-chroma fixture: extreme alternating chroma
      /// (full-scale vs zero) over a super-black→super-white Y ramp, so
      /// many 2x2 blocks straddle the RGB clamp. Here native
      /// (average-in-YUV, convert once) and row-stage (convert-then-average)
      /// genuinely diverge — convert-then-clamp-then-average is not
      /// average-then-convert-then-clamp at the boundary.
      fn out_of_gamut() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; CW * CH];
        let mut v = vec![0u16; CW * CH];
        for i in 0..SRC * SRC {
          // Sweep the full code range including the super-black / -white
          // excursions that drive R/G/B past [0, max].
          y[i] = ((i as u32 * MASK as u32) / (SRC * SRC) as u32) as u16 & MASK;
        }
        for i in 0..CW * CH {
          // Alternate the chroma extremes per sample so adjacent columns
          // pull opposite directions — maximal clamp activity.
          let hi = i % 2 == 0;
          u[i] = if hi { MASK } else { 0 };
          v[i] = if hi { 0 } else { MASK };
        }
        (y, u, v)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_gamut_native_vs_rowstage_pinned() {
        let (y, u, v) = out_of_gamut();
        let (n_rgb, _, n_luma) = native_run(&y, &u, &v);
        let (r_rgb, _, r_luma) = rowstage_run(&y, &u, &v);
        // Luma is still bit-identical even out of gamut (native Y bin,
        // unaffected by the colour clamp).
        assert_eq!(n_luma, r_luma, "luma stays bit-identical out of gamut");
        let max_u8 = n_rgb
          .iter()
          .zip(&r_rgb)
          .map(|(&a, &b)| a.abs_diff(b))
          .max()
          .unwrap_or(0);
        // The documented divergence: out of gamut the tiers differ by MORE
        // than the in-gamut tolerance (a real per-pixel delta, not noise),
        // pinned to bound regression. The observed max on this fixture is
        // recorded as the lower bound; it must stay above the in-gamut
        // tolerance and below the full u8 range.
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
        // single gray pixel — a single narrowed binning would silently
        // break this. Native bins neutral chroma + flat Y, so each output
        // pixel equals the converted gray.
        let (y, u, v) = uniform_gray((MID as u32 + (MASK as u32 / 8)) as u16 & MASK);
        let (n_rgb, n_rgb16, _) = native_run(&y, &u, &v);

        // Direct single-pixel conversion (identity sink, 1x1 of the same
        // codes) gives the reference RGB the whole flat field must match.
        let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
        let mut ref_rgb = vec![0u8; SRC * SRC * 3];
        let mut ref_rgb16 = vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut ref_rgb)
            .unwrap()
            .with_rgb_u16(&mut ref_rgb16)
            .unwrap();
          $walker(&frame(&yl, &ul, &vl), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        // Every output pixel equals the (uniform) direct pixel 0.
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
        // Independent-kernel guard (#37): the u8 and u16 YUV→RGB kernels
        // round and scale independently, so narrowing the binned u16 colour
        // to u8 (`>> (BITS - 8)`) diverges from the genuine u8 bin over a
        // varying ramp. (A flat saturated field clamps identically in both,
        // so it does NOT exhibit the divergence — the ramp does.)
        let (y, u, v) = ramp();
        let (n_rgb, n_rgb16, _) = native_run(&y, &u, &v);
        let narrowed: Vec<u8> = n_rgb16.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_ne!(
          n_rgb, narrowed,
          "u8 colour must be an independent bin, not a narrowed u16 bin"
        );
      }

      #[test]
      fn no_outputs_is_a_no_op() {
        let (y, u, v) = ramp();
        let (yl, ul, vl) = (as_le(&y), as_le(&u), as_le(&v));
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true);
        $walker(&frame(&yl, &ul, &vl), FR, sink.set_kernel_matrix(M)).unwrap();
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resets_join_across_frames() {
        let (y1, u1, v1) = ramp();
        let invert =
          |p: &[u16]| -> Vec<u16> { p.iter().map(|&x| MASK.wrapping_sub(x) & MASK).collect() };
        let (y2, u2, v2) = (invert(&y1), invert(&u1), invert(&v1));
        let (y1l, u1l, v1l) = (as_le(&y1), as_le(&u1), as_le(&v1));
        let (y2l, u2l, v2l) = (as_le(&y2), as_le(&u2), as_le(&v2));
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
          $walker(&frame(&y1l, &u1l, &v1l), FR, sink.set_kernel_matrix(M)).unwrap();
          $walker(&frame(&y2l, &u2l, &v2l), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        // Second frame's luma is the INTER_AREA bin of its own native Y.
        let y_ref = block_mean_2x2_u16(&y2);
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
        let (y, u, v) = ramp();
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
        // Row 3's vertically-shared chroma row is `3 / 2 == 1`.
        let (yr, cr) = (3 * SRC, 1 * CW);
        let err = sink
          .process($row::for_tests(
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
        // The rejected first row stored NO frozen-output snapshot (the
        // pre-freeze first-row check fires BEFORE the freeze), so attaching
        // a NEW output and retrying row 0 must succeed — not trip
        // ResampleOutputsChanged. This is the guard the pre-freeze check
        // provides; a freeze on the rejected row would poison this retry.
        // (The no-output-mutation property of a rejected row is asserted by
        // the frozen-mid-frame and OOS-retry tests, which read the buffer
        // after the sink releases it.)
        let mut rgb = vec![0u8; OUT * OUT * 3];
        sink.set_rgb(&mut rgb).unwrap();
        sink
          .process($row::for_tests(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
          .expect("row 0 must succeed after a rejected out-of-sequence first row");
      }

      /// A mid-frame output-set change on a chroma-bearing even row must be
      /// rejected by the native preflight's frozen-output check BEFORE the
      /// source-scratch alloc — `ResampleOutputsChanged`, never
      /// `AllocationFailed`. The de-interleave / source-scratch alloc
      /// failpoint is armed on the reserve the changed row WOULD reach: with
      /// the preflight first the frozen check fires and the failpoint is
      /// never consumed; with the alloc ordered ahead of the frozen check
      /// (the bug) the armed reserve refuses as AllocationFailed.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn frozen_mid_frame_change_rejected_before_scratch_alloc() {
        let (y, u, v) = ramp();
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
        // Luma-only rows 0 and 1 freeze a luma-only output set (no colour,
        // so the source scratch is grown but no colour scratch).
        for r in 0..2 {
          let cr = (r / 2) * CW;
          sink
            .process($row::for_tests(
              &y[r * SRC..(r + 1) * SRC],
              &u[cr..cr + CW],
              &v[cr..cr + CW],
              r,
              M,
              FR,
            ))
            .expect("luma-only rows freeze a luma-only output set");
        }
        // Attach u16 colour mid-frame, changing the output set, and arm the
        // source-scratch failpoint on the reserve the changed row reaches.
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_alloc_failure();
        let cr = (2 / 2) * CW;
        let err = sink
          .process($row::for_tests(
            &y[2 * SRC..3 * SRC],
            &u[cr..cr + CW],
            &v[cr..cr + CW],
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
        // The failpoint is single-shot. It must NOT have been consumed:
        // prove it by running a fresh in-sequence colour row that DOES fire
        // it (a fresh sink with the same frozen u16 colour shape).
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
          .process($row::for_tests(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
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

      /// The post-freeze rejection point: after a RECOVERABLE source-scratch
      /// allocation failure on an in-sequence colour row 0 leaves
      /// `resample_outputs` frozen but the join's Y stream still at row 0, a
      /// later OUT-OF-SEQUENCE row must reject as the deterministic
      /// `OutOfSequenceRow`, never `AllocationFailed`. The pre-freeze
      /// first-row branch is skipped (outputs frozen), so only the
      /// post-freeze sequence check stands between the out-of-sequence row
      /// and the re-armed failpoint.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn oos_after_recoverable_alloc_failure_rejected_before_scratch_alloc() {
        let (y, u, v) = ramp();
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
        // Step 1 — a RECOVERABLE source-scratch failure on the in-sequence
        // colour row 0. The full preflight clears (freezing the u16 colour
        // output set), then the armed source-scratch reserve refuses:
        // AllocationFailed. The join is built (the source-scratch grow runs
        // after `new`), but its Y stream has not been fed (the grow precedes
        // the feed), so it still expects row 0.
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_alloc_failure();
        let err0 = sink
          .process($row::for_tests(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
          .unwrap_err();
        assert!(
          matches!(
            err0,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the recoverable source-scratch failure on row 0 must surface \
           AllocationFailed, got {err0:?}"
        );
        // Step 2 — RE-ARM, then feed an OUT-OF-SEQUENCE row (idx 2; the
        // join's Y stream still expects 0). The pre-freeze first-row branch
        // is skipped (frozen in step 1), so the post-freeze sequence check
        // is the sole gate; it must reject as OutOfSequenceRow BEFORE the
        // re-armed source-scratch reserve.
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_alloc_failure();
        let cr = (2 / 2) * CW;
        let err2 = sink
          .process($row::for_tests(
            &y[2 * SRC..3 * SRC],
            &u[cr..cr + CW],
            &v[cr..cr + CW],
            2,
            M,
            FR,
          ))
          .unwrap_err();
        assert!(
          matches!(
            err2,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "an out-of-sequence row after a recoverable source-scratch failure \
           must reject as OutOfSequenceRow (the post-freeze sequence check), \
           never AllocationFailed, got {err2:?}"
        );
        assert!(
          rgb_u16.iter().all(|&b| b == 0),
          "neither the recoverable-failure nor the out-of-sequence row \
           touched the colour output"
        );
        // Step 3 — the failpoint re-armed in step 2 must NOT have been
        // consumed: prove it via a fresh in-sequence colour row 0 that fires
        // it.
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
          .process($row::for_tests(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
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

      /// Per-source-row `$row` over the suite's `ramp()` planes (chroma row
      /// `r / 2`), the shared fixture the rebuild-atomicity tests below feed
      /// row by row.
      fn rebuild_row<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16], r: usize) -> $row<'a> {
        let cr = (r / 2) * CW;
        $row::for_tests(
          &y[r * SRC..(r + 1) * SRC],
          &u[cr..cr + CW],
          &v[cr..cr + CW],
          r,
          M,
          FR,
        )
      }

      /// RFC #238 — the 4:2:0 high-bit native delegate is preflight-transactional.
      /// A colour-capability REBUILD (frame 2 attaches colour to a frame-1
      /// luma-only join) whose CHROMA-PLAN build fails must surface the typed
      /// `AllocationFailed` while leaving the cached luma-only join AND the
      /// output-set freeze INTACT (uncommitted): the join stays installed and
      /// `resample_outputs` stays unfrozen, so retrying row 0 with YET ANOTHER
      /// output attached is accepted — a committed `{luma, hsv}` freeze would
      /// reject it as `ResampleOutputsChanged`. The pre-fix delegate froze the
      /// output set and eager-cleared the join BEFORE this fallible rebuild,
      /// corrupting the retry.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_colour_capability_rebuild_chroma_oom_is_transactional() {
        let (y, u, v) = ramp();
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
        // Frame 1: a luma-only native join (chroma absent), frozen as `{luma}`.
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        for r in 0..SRC {
          sink
            .process(rebuild_row(&y, &u, &v, r))
            .expect("luma-only frame builds the native join");
        }
        // Frame 2: attach HSV -> the row-0 rebuild MUST build the chroma half;
        // arm the chroma-plan failpoint so that rebuild fails.
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_chroma_failure();
        let err = sink.process(rebuild_row(&y, &u, &v, 0)).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the colour-capability rebuild chroma-plan OOM must surface \
           AllocationFailed, got {err:?}"
        );
        // Both halves of the commit-together shape rolled back: the cached
        // luma-only join is still installed and `resample_outputs` is unfrozen.
        assert!(
          sink.native_420_u16.is_some(),
          "the cached luma-only join must survive a failed colour rebuild"
        );
        assert!(
          !sink.resample_outputs_frozen(),
          "a failed rebuild must leave resample_outputs uncommitted"
        );
        // Attaching ANOTHER output (rgb) and retrying row 0 is ACCEPTED — the
        // changed output set proves the freeze rolled back (a committed
        // `{luma, hsv}` would reject this as ResampleOutputsChanged).
        sink.set_rgb(&mut rgb).unwrap();
        sink
          .process(rebuild_row(&y, &u, &v, 0))
          .expect("row 0 with a changed output set must succeed after a rebuild chroma OOM");
        for r in 1..SRC {
          sink
            .process(rebuild_row(&y, &u, &v, r))
            .expect("the recovered join must process the rest of the frame");
        }
        assert!(
          rgb.iter().any(|&b| b != 0),
          "the recovered join must produce real colour output"
        );
      }

      /// RFC #238 — the u8 colour-scratch failpoint companion: the same
      /// colour-capability rebuild, but the fallible step that fails is the u8
      /// `rgb_scratch` grow (attaching `rgb`). It too must leave the cached join
      /// and output-set freeze uncommitted so a retry with a changed output set
      /// is accepted and drives real output.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_colour_capability_rebuild_rgb_scratch_oom_is_transactional() {
        let (y, u, v) = ramp();
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
            .process(rebuild_row(&y, &u, &v, r))
            .expect("luma-only frame builds the native join");
        }
        // Frame 2: attach RGB -> the rebuild builds the chroma half then grows the
        // (now-needed, empty) output RGB scratch; arm that u8 grow to refuse.
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink.set_rgb(&mut rgb).unwrap();
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_rgb_scratch_failure();
        let err = sink.process(rebuild_row(&y, &u, &v, 0)).unwrap_err();
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
          .process(rebuild_row(&y, &u, &v, 0))
          .expect("row 0 with a changed output set must succeed after a u8-scratch OOM");
        for r in 1..SRC {
          sink
            .process(rebuild_row(&y, &u, &v, r))
            .expect("the recovered join must process the rest of the frame");
        }
        assert!(
          rgb.iter().any(|&b| b != 0),
          "the recovered join must produce real colour output"
        );
      }

      /// RFC #238 — a LATER pre-feed grow failing is atomic too. With BOTH a u8
      /// colour scratch (`rgb`) and a native-depth u16 colour scratch
      /// (`rgb_u16`) attached, the u8 `rgb_scratch` grows FIRST and then the u16
      /// `rgb_scratch_u16` grow refuses. The already-grown `rgb_scratch` keeps
      /// its capacity (join / freeze / output-atomic, not memory-exact), but the
      /// cached join + `resample_outputs` freeze + output frame stay untouched, so
      /// a retry with yet another output attached is accepted and drives real
      /// output.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_colour_capability_rebuild_rgb_u16_scratch_oom_is_transactional() {
        let (y, u, v) = ramp();
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
            .process(rebuild_row(&y, &u, &v, r))
            .expect("luma-only frame builds the native join");
        }
        // Frame 2: attach BOTH u8 (rgb) and u16 (rgb_u16) colour -> the rebuild
        // grows `rgb_scratch` then `rgb_scratch_u16`; arm the LATER (u16) grow.
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink.set_rgb(&mut rgb).unwrap();
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        crate::sinker::mixed::subsampled_4_2_0_high_bit::arm_native_u16_rgb_u16_scratch_failure();
        let err = sink.process(rebuild_row(&y, &u, &v, 0)).unwrap_err();
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
        // The freeze was never committed: attaching hsv and retrying row 0 is
        // accepted.
        sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
        sink
          .process(rebuild_row(&y, &u, &v, 0))
          .expect("row 0 with a changed output set must succeed after a u16-scratch OOM");
        for r in 1..SRC {
          sink
            .process(rebuild_row(&y, &u, &v, r))
            .expect("the recovered join must process the rest of the frame");
        }
        assert!(
          rgb.iter().any(|&b| b != 0) && rgb_u16.iter().any(|&b| b != 0),
          "the recovered join must produce real u8 and u16 colour output"
        );
      }

      // ---- frozen native-vs-row-stage route (issue #186) ----------------

      /// Flipping `set_native(true) -> false` mid-frame must reject as the
      /// deterministic `NativeRouteChanged` BEFORE either tier consumes the
      /// row — the high-bit planar 4:2:0 twin of the P0xx guard.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_to_rowstage_route_flip_mid_frame_rejected() {
        let (y, u, v) = ramp();
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
        // Row 0 freezes the route = native.
        sink
          .process($row::for_tests(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
          .expect("native row 0 freezes the route and succeeds");
        // Flip to the row-stage tier and feed the next in-sequence row.
        sink.set_native(false);
        let err = sink
          .process($row::for_tests(
            &y[SRC..2 * SRC],
            &u[..CW],
            &v[..CW],
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

      /// The reverse flip `set_native(false) -> true` mid-frame must reject
      /// identically — the guard catches BOTH directions.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rowstage_to_native_route_flip_mid_frame_rejected() {
        let (y, u, v) = ramp();
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
          .process($row::for_tests(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
          .expect("row-stage row 0 freezes the route and succeeds");
        sink.set_native(true);
        let err = sink
          .process($row::for_tests(
            &y[SRC..2 * SRC],
            &u[..CW],
            &v[..CW],
            1,
            M,
            FR,
          ))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "a row-stage -> native mid-frame route flip must reject as \
           NativeRouteChanged, got {err:?}"
        );
      }

      /// A constant-route frame runs to completion, and the per-frame reset
      /// (via `reset_high_bit_yuv_streams` in `begin_frame`) lets the NEXT
      /// frame pick the OTHER tier — the route guard is reset, not sticky.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn route_constant_succeeds_and_resets_across_frames() {
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
        // Frame 1: native, route constant across every row.
        $walker(&frame(&yl, &ul, &vl), FR, sink.set_kernel_matrix(M)).unwrap();
        // Frame 2: flip to row-stage for the WHOLE frame after begin_frame.
        sink.set_native(false);
        $walker(&frame(&yl, &ul, &vl), FR, sink.set_kernel_matrix(M))
          .expect("a new frame may pick the other tier; the route reset per frame");
      }

      /// A NO-OUTPUT call AFTER an output-bearing row froze the route must be
      /// a TRUE no-op — route-invisible — even with `set_native` FLIPPED: it
      /// returns `Ok` (not `NativeRouteChanged`) and leaves the frozen route
      /// untouched (both the CHECK and the SET gate on `need_output`). No
      /// public API detaches an output, so we set `frozen_native_route`
      /// directly to the value an accepted output-bearing native first row
      /// stores (`Some(true)` = native) — the same white-box reach the
      /// atomicity tests use.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn no_output_call_after_frozen_route_is_a_noop() {
        let (y, u, v) = ramp();
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true);
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink.frozen_native_route = Some(true);
        // No-output row (no outputs -> `need_output` false), route flipped to
        // row-stage. The CHECK is skipped, so this is a true no-op.
        sink.set_native(false);
        sink
          .process($row::for_tests(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
          .expect(
            "a no-output call after a frozen route must be a true no-op, not \
             NativeRouteChanged",
          );
        assert_eq!(
          sink.frozen_native_route,
          Some(true),
          "a no-output call must leave the frozen route unchanged",
        );
        // The route is STILL native and consumed no stream state: an
        // output-bearing native row 0 succeeds...
        let mut luma = vec![0u8; OUT * OUT];
        sink.set_native(true);
        sink.set_luma(&mut luma).unwrap();
        sink
          .process($row::for_tests(&y[..SRC], &u[..CW], &v[..CW], 0, M, FR))
          .expect("an output-bearing row under the original native route succeeds");
        // ...while an output-bearing flip to the OTHER route now rejects.
        sink.set_native(false);
        let err = sink
          .process($row::for_tests(
            &y[SRC..2 * SRC],
            &u[..CW],
            &v[..CW],
            1,
            M,
            FR,
          ))
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

yuv420p_high_bit_native_suite!(
  yuv420p9,
  Yuv420p9LeFrame,
  Yuv420p9BeFrame,
  Yuv420p9,
  Yuv420p9Row,
  yuv420p9_to,
  yuv420p9_to_endian,
  9,
);
yuv420p_high_bit_native_suite!(
  yuv420p10,
  Yuv420p10LeFrame,
  Yuv420p10BeFrame,
  Yuv420p10,
  Yuv420p10Row,
  yuv420p10_to,
  yuv420p10_to_endian,
  10,
);
yuv420p_high_bit_native_suite!(
  yuv420p12,
  Yuv420p12LeFrame,
  Yuv420p12BeFrame,
  Yuv420p12,
  Yuv420p12Row,
  yuv420p12_to,
  yuv420p12_to_endian,
  12,
);
yuv420p_high_bit_native_suite!(
  yuv420p14,
  Yuv420p14LeFrame,
  Yuv420p14BeFrame,
  Yuv420p14,
  Yuv420p14Row,
  yuv420p14_to,
  yuv420p14_to_endian,
  14,
);
yuv420p_high_bit_native_suite!(
  yuv420p16,
  Yuv420p16LeFrame,
  Yuv420p16BeFrame,
  Yuv420p16,
  Yuv420p16Row,
  yuv420p16_to,
  yuv420p16_to_endian,
  16,
);
