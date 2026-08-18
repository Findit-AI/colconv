//! Fused-downscale coverage for the 8-bit **planar** non-4:2:0 YUV NATIVE
//! fast tier — `Yuv422p` (4:2:2), `Yuv444p` (4:4:4), `Yuv440p` (4:4:0) —
//! the siblings of the 4:2:0
//! [`yuv420p_process_native`](crate::sinker::mixed::planar_8bit::yuv420p_process_native)
//! for chroma layouts that are not half-resolution in both axes.
//!
//! The native tier bins the Y / U / V planes straight to the output grid
//! and converts ONCE per output row at output width (4:4:4 kernel), vs the
//! row-stage tier
//! ([`planar_dual_resample`](crate::sinker::mixed::planar_resample::planar_dual_resample)),
//! which converts each source row at source width then bins. The tiers
//! differ in colour SEMANTICS (native averages in YUV then converts;
//! row-stage converts then averages in RGB), so native is NOT byte-
//! identical to row-stage — only within a small tolerance in-gamut, with
//! documented out-of-gamut divergence. Luma is bit-identical (both bin the
//! same native Y stream).
//!
//! Per format the suite asserts:
//! - `native_equals_bin_then_convert_oracle`: native output is EXACTLY the
//!   bin-then-convert reference — area-bin each source plane to OUTPUT
//!   resolution by its own subsample ratio (Y always 2:1x2:1; chroma 2:1
//!   only on its subsampled axis: horizontal for 4:2:2, vertical for 4:4:0,
//!   both for 4:4:4), giving full-output-width chroma, then convert ONCE
//!   through the 4:4:4 kernel. This is the ground-truth correctness check:
//!   it pins the per-plane binning geometry + the once-per-output-pixel
//!   convert independently of the row-stage tier, and is exactly what the
//!   prior attempt's chroma-ratio bug (delta 8) failed. RGBA is the RGB row
//!   fanned with opaque alpha; HSV is that RGB row's HSV; luma / luma_u16
//!   are the binned Y.
//! - `native_within_tolerance_of_rowstage`: same source through
//!   `with_native(true)` and `with_native(false)`, per-channel
//!   `|native - rowstage| <= TOL_U8` for RGB / RGBA in gamut, with LUMA
//!   bit-identical (the row-stage tier IS the cv2 INTER_AREA oracle, so this
//!   is the INTER_AREA parity check). HSV is excluded — saturation is
//!   numerically unstable under a 1-LSB RGB swing, so a tight bound is not
//!   meaningful.
//! - `native_luma_matches_inter_area_oracle`: native luma equals the direct
//!   2x2-block area mean of the native Y plane.
//! - `out_of_gamut_native_vs_rowstage_pinned`: on a crafted illegal-chroma
//!   fixture the tiers diverge by MORE than the in-gamut tolerance (the
//!   documented convert-vs-average-at-the-clamp divergence), with luma still
//!   bit-identical.
//! - `native_default_matches_explicit_true`: the default tier IS the native
//!   tier (`with_native` defaults to `true`).

use crate::{
  KernelMatrix,
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Yuv444p, yuv444p_to},
};
use mediaframe::frame::{Yuv422pFrame, Yuv440pFrame, Yuv444pFrame};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// In-gamut per-channel tolerance between the native and row-stage tiers.
/// The two average in different domains (YUV vs RGB) and round
/// independently per output pixel; the empirical in-gamut maximum on the
/// mid-range ramp fixtures here is 4 (the non-4:2:0 chroma carries more
/// spatial detail into each 2x2 bin than 4:2:0, so the convert-order gap is
/// a touch wider), pinned with a 1-LSB margin for cross-platform
/// SIMD-vs-scalar rounding. Native correctness itself is pinned exactly by
/// `native_equals_bin_then_convert_oracle`; this bound only documents the
/// row-stage semantic gap. Out-of-gamut content diverges further and is
/// pinned separately by `out_of_gamut_native_vs_rowstage_pinned`.
const TOL_U8: u8 = 5;

/// Exact integer-ratio area mean (round-half-up) of an `in_w x in_h` u8
/// plane down to `OUT x OUT`, binning each axis by its own ratio
/// (`in_w / OUT` horizontally, `in_h / OUT` vertically — each 1 or 2 for the
/// fixtures here). With a per-axis ratio of 1 that axis is copied
/// unchanged; with 2 it is a 2:1 box mean. This reproduces the native
/// tier's per-plane binning to full output resolution for any subsample
/// direction (Y bins 2:1x2:1; 4:2:2 chroma horizontal-only-identity +
/// vertical-2:1; 4:4:0 chroma horizontal-2:1 + vertical-identity; 4:4:4
/// chroma 2:1x2:1).
fn bin_to_out(plane: &[u8], in_w: usize, in_h: usize) -> Vec<u8> {
  let (rx, ry) = (in_w / OUT, in_h / OUT);
  let denom = (rx * ry) as u32;
  let mut out = vec![0u8; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..ry {
        for dx in 0..rx {
          s += plane[(oy * ry + dy) * in_w + ox * rx + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + denom / 2) / denom) as u8;
    }
  }
  out
}

macro_rules! yuv_planar_8bit_native_suite {
  (
    $mod:ident, $marker:ident, $frame:ident, $walker:ident,
    $cw:expr, $ch:expr,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $walker};

      const CW: usize = $cw;
      const CH: usize = $ch;

      /// Mid-range Y + chroma ramp — every code in gamut, so the
      /// native-vs-rowstage delta is the per-pixel rounding difference, not
      /// a clamp divergence.
      fn ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; SRC * SRC];
        let mut u = vec![0u8; CW * CH];
        let mut v = vec![0u8; CW * CH];
        for (i, p) in y.iter_mut().enumerate() {
          *p = 40 + ((i as u32 * 3) % 160) as u8;
        }
        for (i, p) in u.iter_mut().enumerate() {
          *p = 100 + ((i as u32 * 7) % 56) as u8;
        }
        for (i, p) in v.iter_mut().enumerate() {
          *p = 150 - ((i as u32 * 5) % 56) as u8;
        }
        (y, u, v)
      }

      /// Crafted VARYING illegal-chroma fixture: extreme alternating chroma
      /// (full-scale vs zero) over a super-black->super-white Y ramp, so many
      /// 2x2 blocks straddle the RGB clamp. Here native (average-in-YUV,
      /// convert once) and row-stage (convert-then-average) genuinely
      /// diverge.
      fn out_of_gamut() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; SRC * SRC];
        let mut u = vec![0u8; CW * CH];
        let mut v = vec![0u8; CW * CH];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((i as u32 * 255) / (SRC * SRC) as u32) as u8;
        }
        for (i, p) in u.iter_mut().enumerate() {
          *p = if i % 2 == 0 { 255 } else { 0 };
        }
        for (i, p) in v.iter_mut().enumerate() {
          *p = if i % 2 == 0 { 0 } else { 255 };
        }
        (y, u, v)
      }

      fn frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> $frame<'a> {
        $frame::new(
          y, u, v, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
        )
      }

      /// Drive a tier for the full output set (RGB + RGBA + HSV + luma +
      /// luma_u16). `native` toggles between the bin-then-convert native
      /// fast tier and the convert-then-bin row-stage tier.
      fn run(
        y: &[u8],
        u: &[u8],
        v: &[u8],
        native: bool,
      ) -> (
        Vec<u8>,
        Vec<u8>,
        (Vec<u8>, Vec<u8>, Vec<u8>),
        Vec<u8>,
        Vec<u16>,
      ) {
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let (mut hh, mut ss, mut vv) = (
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
        );
        let mut luma = vec![0u8; OUT * OUT];
        let mut luma_u16 = vec![0u16; OUT * OUT];
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
          .with_rgba(&mut rgba)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walker(&frame(y, u, v), FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, (hh, ss, vv), luma, luma_u16)
      }

      /// The bin-then-convert oracle: area-bin every source plane to OUTPUT
      /// resolution by its own subsample ratio (Y from `SRC x SRC`, chroma
      /// from `CW x CH` — `bin_to_out` handles the per-axis ratio), then
      /// convert the full-output-width planes ONCE through the 4:4:4 path
      /// (an identity-resolution `Yuv444p` sink). The binned chroma is full
      /// output width on every format, so the convert is always 4:4:4 — the
      /// exact ground truth the native tier must reproduce byte-for-byte.
      fn oracle(
        y: &[u8],
        u: &[u8],
        v: &[u8],
      ) -> (
        Vec<u8>,
        Vec<u8>,
        (Vec<u8>, Vec<u8>, Vec<u8>),
        Vec<u8>,
        Vec<u16>,
      ) {
        let yb = bin_to_out(y, SRC, SRC);
        let ub = bin_to_out(u, CW, CH);
        let vb = bin_to_out(v, CW, CH);
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let (mut hh, mut ss, mut vv) = (
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
        );
        let mut luma = vec![0u8; OUT * OUT];
        let mut luma_u16 = vec![0u16; OUT * OUT];
        {
          let mut sink = MixedSinker::<Yuv444p>::new(OUT, OUT)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgba(&mut rgba)
            .unwrap()
            .with_hsv(&mut hh, &mut ss, &mut vv)
            .unwrap()
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut luma_u16)
            .unwrap();
          let f = Yuv444pFrame::new(
            &yb, &ub, &vb, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
          );
          yuv444p_to(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, (hh, ss, vv), luma, luma_u16)
      }

      fn max_delta(a: &[u8], b: &[u8]) -> u8 {
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
        // Ground truth: the native tier IS bin-then-convert. Every output
        // must match the independent oracle exactly — this is the check the
        // prior attempt's chroma-ratio bug (delta 8) failed.
        let (y, u, v) = ramp();
        let n = run(&y, &u, &v, true);
        let o = oracle(&y, &u, &v);
        assert_eq!(n.0, o.0, "rgb must equal the bin-then-convert oracle");
        assert_eq!(n.1, o.1, "rgba must equal the bin-then-convert oracle");
        assert_eq!(n.2, o.2, "hsv must equal the bin-then-convert oracle");
        assert_eq!(n.3, o.3, "luma must equal the binned Y");
        assert_eq!(n.4, o.4, "luma_u16 must equal the binned Y");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_within_tolerance_of_rowstage() {
        let (y, u, v) = ramp();
        let (n_rgb, n_rgba, _, n_luma, n_luma16) = run(&y, &u, &v, true);
        let (r_rgb, r_rgba, _, r_luma, r_luma16) = run(&y, &u, &v, false);

        // Luma: both tiers bin the SAME native Y stream, so it is
        // bit-identical (u8 and zero-extended u16 alike).
        assert_eq!(n_luma, r_luma, "luma must be bit-identical across tiers");
        assert_eq!(
          n_luma16, r_luma16,
          "luma_u16 must be bit-identical across tiers"
        );

        // RGB / RGBA: within tolerance in gamut (RGBA is the RGB row fanned
        // with opaque alpha, so it tracks the RGB delta).
        let d_rgb = max_delta(&n_rgb, &r_rgb);
        assert!(
          d_rgb <= TOL_U8,
          "rgb native-vs-rowstage max delta {d_rgb} exceeds tolerance {TOL_U8}"
        );
        let d_rgba = max_delta(&n_rgba, &r_rgba);
        assert!(
          d_rgba <= TOL_U8,
          "rgba native-vs-rowstage max delta {d_rgba} exceeds tolerance {TOL_U8}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_matches_inter_area_oracle() {
        // cv2 INTER_AREA parity for luma: the 2x2-block area mean of the
        // native Y plane (luma derives from Y, never from RGB).
        let (y, u, v) = ramp();
        let (_, _, _, n_luma, _) = run(&y, &u, &v, true);
        assert_eq!(
          n_luma,
          bin_to_out(&y, SRC, SRC),
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
        let (n_rgb, _, _, n_luma, _) = run(&y, &u, &v, true);
        let (r_rgb, _, _, r_luma, _) = run(&y, &u, &v, false);
        // Luma stays bit-identical even out of gamut (native Y bin,
        // unaffected by the colour clamp).
        assert_eq!(n_luma, r_luma, "luma stays bit-identical out of gamut");
        // The documented divergence: out of gamut the tiers differ by MORE
        // than the in-gamut tolerance (a real per-pixel delta, not noise).
        let d = max_delta(&n_rgb, &r_rgb);
        assert!(
          d > TOL_U8,
          "out-of-gamut rgb delta {d} must exceed the in-gamut tolerance {TOL_U8}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_default_matches_explicit_true() {
        // `with_native` defaults to true, so the default sink and an
        // explicit `with_native(true)` sink must agree byte-for-byte.
        let (y, u, v) = ramp();
        let e = run(&y, &u, &v, true);

        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let (mut hh, mut ss, mut vv) = (
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
          vec![0u8; OUT * OUT],
        );
        let mut luma = vec![0u8; OUT * OUT];
        let mut luma_u16 = vec![0u16; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walker(&frame(&y, &u, &v), FR, M, &mut sink).unwrap();
        }
        assert_eq!(rgb, e.0, "default rgb must match explicit native");
        assert_eq!(rgba, e.1, "default rgba must match explicit native");
        assert_eq!((hh, ss, vv), e.2, "default hsv must match explicit native");
        assert_eq!(luma, e.3, "default luma must match explicit native");
        assert_eq!(luma_u16, e.4, "default luma_u16 must match explicit native");
      }
    }
  };
}

// 4:2:2: chroma `w/2 x h` — half width, full height.
yuv_planar_8bit_native_suite!(yuv422p, Yuv422p, Yuv422pFrame, yuv422p_to, SRC / 2, SRC,);
// 4:4:4: chroma `w x h` — identical to Y.
yuv_planar_8bit_native_suite!(yuv444p, Yuv444p, Yuv444pFrame, yuv444p_to, SRC, SRC,);
// 4:4:0: chroma `w x h/2` — full width, half height.
yuv_planar_8bit_native_suite!(yuv440p, Yuv440p, Yuv440pFrame, yuv440p_to, SRC, SRC / 2,);
