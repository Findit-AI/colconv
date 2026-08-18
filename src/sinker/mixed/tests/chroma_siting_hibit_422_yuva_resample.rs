//! Chroma-siting-aware fused-downscale coverage for the HIGH-BIT **planar** 4:2:2
//! YUV-with-alpha family — `Yuva422p9` / `Yuva422p10` / `Yuva422p12` /
//! `Yuva422p16` (RFC #238 S7), the `u16` alpha-bearing twin of the 8-bit
//! `chroma_siting_422_yuva_resample` suite and the alpha-bearing sibling of the
//! high-bit planar `chroma_siting_hibit_422_resample` suite.
//!
//! `Yuva422pN` is planar 4:2:2 YUV (half-width, FULL-height U / V planes) PLUS a
//! **full-resolution** straight alpha plane that is never subsampled. 4:2:2 is
//! subsampled horizontally only — no vertical chroma phase — so S7 routes the
//! HORIZONTAL centered siting (`Center` / `Top`,
//! [`chroma_422_center_sited_h`](super::super::chroma_422_center_sited_h)) through
//! the resample. The full-resolution α plane is siting-independent on every path.
//!
//! Unlike the no-alpha planar `Yuv422pN` (which owns a native code-domain
//! plane-binning fast tier), the high-bit `Yuva422pN` 4:2:2 sink has **only the
//! packed-YUVA tail** — every downscale converts each source row to RGBA (chroma
//! reconstructed in-register) then area- / filter-bins the packed RGBA. So the
//! centered area AND filter routes are both RGB-domain reconstruct-then-bin, and
//! the oracle drives a full-resolution `Yuva444pN` frame (U / V reconstructed to
//! full width per source row `r`) through the SAME packed tail via an
//! identity-chroma `Yuva444pN` resample.
//!
//! Assertions per depth (`p10` low-packed, `p12` mid, `p16` full `u16`; BE + LE):
//!  - the co-sited / unspecified group stays byte-identical to the pre-siting
//!    resample (every output, α included);
//!  - centered area / filter RGBA (α carried) == the RGB-domain
//!    reconstruct-then-bin oracle, `u8` AND native `u16`;
//!  - the centered chroma decode (α-drop RGB / HSV / luma) is BIT-IDENTICAL to the
//!    no-alpha `Yuv422pN` row-stage centered resample (α is orthogonal);
//!  - the α channel is IDENTICAL between co-sited and centered (colour differs);
//!  - SIMD == scalar, BE == LE, centered != co-sited on a ramp / == on flat, and a
//!    mid-frame siting flip is rejected across tiers.
//!
//! Every centered assertion is an EXACT match (single rounding for a given tier),
//! never a tolerance.

use crate::{
  ChromaLocation, KernelMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, FilteredResampler, Triangle},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::*,
};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Independent #302 centered horizontal upsample (`1/4`–`3/4`, edge clamp,
/// round-half-up to `u16`) — the RGB-domain oracle's reconstruction step.
fn recon_full_row_u16(c: &[u16], cw: usize) -> Vec<u16> {
  let mut out = vec![0u16; 2 * cw];
  for j in 0..cw {
    let l = u32::from(c[j.saturating_sub(1)]);
    let m = u32::from(c[j]);
    let r = u32::from(c[if j + 1 < cw { j + 1 } else { j }]);
    out[2 * j] = ((l + 3 * m + 2) >> 2) as u16;
    out[2 * j + 1] = ((3 * m + r + 2) >> 2) as u16;
  }
  out
}

/// Re-encode a host-native `u16` slice as host-independent BE-wire storage.
fn as_be(host: &[u16]) -> Vec<u16> {
  host.iter().map(|v| v.to_be()).collect()
}

/// The geometries the oracle-equality tests sweep: clean 2:1 and fractional
/// ratios. Widths are even (`Yuva422p` requires it).
const GEOMS: [(usize, usize, usize, usize); 4] =
  [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)];

/// `(rgb, rgba, rgb_u16, rgba_u16, (h,s,v), luma, luma_u16)` — the full output set.
type Outs = (
  Vec<u8>,
  Vec<u8>,
  Vec<u16>,
  Vec<u16>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);
/// The no-alpha `Yuv422pN` cross-check outputs `(rgb, rgb_u16, (h,s,v), luma)` —
/// the α-drop channels the alpha-bearing decode must match.
type YuvOuts = (Vec<u8>, Vec<u16>, (Vec<u8>, Vec<u8>, Vec<u8>), Vec<u8>);

macro_rules! hibit_422_yuva_resample_siting {
  (
    $mod:ident, $bits:expr,
    $M422:ident, $F422le:ident, $w422:ident,
    $M422be:ty, $F422be:ident, $w422be:ident, $Row:ident,
    $M444:ident, $F444:ident, $w444:ident,
    $MYuv:ident, $FYuv:ident, $wYuv:ident
  ) => {
    mod $mod {
      use super::*;

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = 1u16 << ($bits - 1);

      /// A `Yuva422pN` fixture with a strong HORIZONTAL chroma ramp (so the
      /// centered triangle genuinely differs from the co-sited nearest decode)
      /// plus a per-row tilt (a vertical mistake would show) and a varying,
      /// non-opaque alpha (so the α-preservation assertions are non-vacuous). Chroma
      /// is FULL-height (`cw x sh`). Low-packed native codes so every kernel sees
      /// real math.
      fn ramp(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let cw = sw / 2;
        let step = (MASK as u32 / 16).max(1);
        let mut y = vec![0u16; sw * sh];
        let mut u = vec![0u16; cw * sh];
        let mut v = vec![0u16; cw * sh];
        let mut a = vec![0u16; sw * sh];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 37) & MASK as u32) as u16;
        }
        for r in 0..sh {
          for c in 0..cw {
            u[r * cw + c] = (step * c as u32 + step + r as u32 * 5).min(MASK as u32) as u16;
            v[r * cw + c] = (MASK as u32).saturating_sub(step * c as u32).max(step) as u16;
          }
        }
        for (i, p) in a.iter_mut().enumerate() {
          *p = ((MASK as u32 / 5 + i as u32 * 23) & MASK as u32) as u16;
        }
        (y, u, v, a)
      }

      /// Flat chroma: the centered triangle of a constant is that constant, so
      /// centered must equal co-sited. Luma and α still vary.
      fn flat(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let cw = sw / 2;
        let mut y = vec![0u16; sw * sh];
        let mut a = vec![0u16; sw * sh];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 29) & MASK as u32) as u16;
        }
        for (i, p) in a.iter_mut().enumerate() {
          *p = ((MASK as u32 / 3 + i as u32 * 17) & MASK as u32) as u16;
        }
        (y, vec![MID; cw * sh], vec![MID; cw * sh], a)
      }

      /// Attach the full output set to a fresh `Vec` bundle sized for `ow x oh`.
      fn bufs(ow: usize, oh: usize) -> Outs {
        (
          vec![0u8; ow * oh * 3],
          vec![0u8; ow * oh * 4],
          vec![0u16; ow * oh * 3],
          vec![0u16; ow * oh * 4],
          (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]),
          vec![0u8; ow * oh],
          vec![0u16; ow * oh],
        )
      }

      /// Drive an LE `Yuva422pN` STRAIGHT-alpha AREA resample for the full set.
      #[allow(clippy::too_many_arguments)]
      fn run(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        a: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
        simd: bool,
      ) -> Outs {
        let cw = sw / 2;
        let (
          mut rgb,
          mut rgba,
          mut rgb16,
          mut rgba16,
          (mut hh, mut ss, mut vv),
          mut luma,
          mut luma16,
        ) = bufs(ow, oh);
        {
          let mut sink =
            MixedSinker::<$M422, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_chroma_location(loc.clone())
              .with_simd(simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgba(&mut rgba)
              .unwrap()
              .with_rgb_u16(&mut rgb16)
              .unwrap()
              .with_rgba_u16(&mut rgba16)
              .unwrap()
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap()
              .with_luma(&mut luma)
              .unwrap()
              .with_luma_u16(&mut luma16)
              .unwrap();
          let f = $F422le::try_new(
            y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
          )
          .unwrap();
          $w422(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, rgb16, rgba16, (hh, ss, vv), luma, luma16)
      }

      /// Drive an LE `Yuva422pN` single-kernel filter resample (Triangle).
      #[allow(clippy::too_many_arguments)]
      fn run_filter(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        a: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
        simd: bool,
      ) -> Outs {
        let cw = sw / 2;
        let (
          mut rgb,
          mut rgba,
          mut rgb16,
          mut rgba16,
          (mut hh, mut ss, mut vv),
          mut luma,
          mut luma16,
        ) = bufs(ow, oh);
        {
          let mut sink = MixedSinker::<$M422, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Straight)
          .with_chroma_location(loc.clone())
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_rgba_u16(&mut rgba16)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma16)
          .unwrap();
          let f = $F422le::try_new(
            y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
          )
          .unwrap();
          $w422(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, rgb16, rgba16, (hh, ss, vv), luma, luma16)
      }

      /// Drive a BE `Yuva422pN` AREA resample (planes re-encoded BE-wire).
      #[allow(clippy::too_many_arguments)]
      fn run_be(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        a: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
      ) -> Outs {
        let cw = sw / 2;
        let (yb, ub, vb, ab) = (as_be(y), as_be(u), as_be(v), as_be(a));
        let (
          mut rgb,
          mut rgba,
          mut rgb16,
          mut rgba16,
          (mut hh, mut ss, mut vv),
          mut luma,
          mut luma16,
        ) = bufs(ow, oh);
        {
          let mut sink = MixedSinker::<$M422be, AreaResampler>::with_resampler(
            sw,
            sh,
            AreaResampler::to(ow, oh),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Straight)
          .with_chroma_location(loc.clone())
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_rgba_u16(&mut rgba16)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma16)
          .unwrap();
          let f = $F422be::try_new(
            &yb, &ub, &vb, &ab, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
          )
          .unwrap();
          $w422be(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, rgb16, rgba16, (hh, ss, vv), luma, luma16)
      }

      /// The centered RGB-domain oracle for the packed area / filter tail:
      /// reconstruct U / V to full width (`u16`) with the #302 kernel — each luma
      /// row `r` taking chroma row `r` (4:2:2 chroma is full-height) — then run that
      /// full-resolution `Yuva444pN` frame (with the UNTOUCHED α plane) through the
      /// given resampler with STRAIGHT alpha. Returns `(rgba, rgba_u16)` — the exact
      /// convert-each-row-then-bin the routed path does.
      #[allow(clippy::too_many_arguments)]
      fn rgba_oracle(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        a: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        simd: bool,
        filter: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let cw = sw / 2;
        let mut uf = vec![0u16; sw * sh];
        let mut vf = vec![0u16; sw * sh];
        for r in 0..sh {
          uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_u16(&u[r * cw..r * cw + cw], cw));
          vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_u16(&v[r * cw..r * cw + cw], cw));
        }
        let mut rgba = vec![0u8; ow * oh * 4];
        let mut rgba16 = vec![0u16; ow * oh * 4];
        let f = $F444::try_new(
          y, &uf, &vf, a, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32, sw as u32,
        )
        .unwrap();
        if filter {
          let mut sink = MixedSinker::<$M444, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Straight)
          .with_simd(simd)
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgba_u16(&mut rgba16)
          .unwrap();
          $w444(&f, FR, M, &mut sink).unwrap();
        } else {
          let mut sink =
            MixedSinker::<$M444, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_simd(simd)
              .with_rgba(&mut rgba)
              .unwrap()
              .with_rgba_u16(&mut rgba16)
              .unwrap();
          $w444(&f, FR, M, &mut sink).unwrap();
        }
        (rgba, rgba16)
      }

      /// Drive the no-alpha `Yuv422pN` resample on the SAME Y / U / V through the
      /// ROW-STAGE tier (`with_native(false)`, RGB-domain — the tier the packed
      /// YUVA path mirrors). RGB is attached so HSV derives from the same binned
      /// RGB the α-drop YUVA HSV does, making the HSV comparison valid.
      #[allow(clippy::too_many_arguments)]
      fn run_yuv(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
        simd: bool,
      ) -> YuvOuts {
        let cw = sw / 2;
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
        let mut luma = vec![0u8; ow * oh];
        {
          let mut sink =
            MixedSinker::<$MYuv, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
              .unwrap()
              .with_native(false)
              .with_chroma_location(loc.clone())
              .with_simd(simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgb_u16(&mut rgb16)
              .unwrap()
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap()
              .with_luma(&mut luma)
              .unwrap();
          let f = $FYuv::new(
            y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
          );
          $wYuv(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16, (hh, ss, vv), luma)
      }

      // ---- co-sited byte-identity (the regression contract) ----------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn cosited_group_is_byte_identical_across_tiers() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          let base = run(
            &y,
            &u,
            &v,
            &a,
            sw,
            sh,
            ow,
            oh,
            ChromaLocation::Unspecified,
            true,
          );
          for loc in [
            ChromaLocation::Left,
            ChromaLocation::TopLeft,
            ChromaLocation::BottomLeft,
            ChromaLocation::other("unassigned-7"),
          ] {
            assert_eq!(
              run(&y, &u, &v, &a, sw, sh, ow, oh, loc.clone(), true),
              base,
              "co-sited {loc:?} area must keep the byte-identical decode \
               ({sw}x{sh}->{ow}x{oh})"
            );
          }
          let fbase = run_filter(
            &y,
            &u,
            &v,
            &a,
            sw,
            sh,
            ow,
            oh,
            ChromaLocation::Unspecified,
            true,
          );
          assert_eq!(
            run_filter(&y, &u, &v, &a, sw, sh, ow, oh, ChromaLocation::Left, true),
            fbase,
            "co-sited filter must stay byte-identical ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      // ---- centered area / filter RGBA == RGB-domain reconstruct-then-bin ---

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_area_rgba_equals_rgb_domain_oracle() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          let (want_rgba, want_rgba16) = rgba_oracle(&y, &u, &v, &a, sw, sh, ow, oh, true, false);
          for loc in [ChromaLocation::Center, ChromaLocation::Top] {
            let got = run(&y, &u, &v, &a, sw, sh, ow, oh, loc.clone(), true);
            assert_eq!(
              got.1, want_rgba,
              "centered area rgba {loc:?} ({sw}x{sh}->{ow}x{oh})"
            );
            assert_eq!(
              got.3, want_rgba16,
              "centered area rgba_u16 {loc:?} ({sw}x{sh}->{ow}x{oh})"
            );
          }
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_filter_rgba_equals_rgb_domain_oracle() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          let (want_rgba, want_rgba16) = rgba_oracle(&y, &u, &v, &a, sw, sh, ow, oh, true, true);
          let got = run_filter(&y, &u, &v, &a, sw, sh, ow, oh, ChromaLocation::Center, true);
          assert_eq!(
            got.1, want_rgba,
            "centered filter rgba ({sw}x{sh}->{ow}x{oh})"
          );
          assert_eq!(
            got.3, want_rgba16,
            "centered filter rgba_u16 ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      // ---- STRONG cross-check: centered chroma decode == Yuv422pN centered --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_chroma_matches_yuv422p_centered() {
        // α is orthogonal to chroma siting, so the centered `Yuva422pN` Y/U/V → RGB
        // (and RGB_u16 / HSV / luma) decode must be BIT-IDENTICAL to the no-alpha
        // `Yuv422pN` ROW-STAGE centered resample of the same Y/U/V (the packed YUVA
        // tail is RGB-domain, so it matches the row-stage tier, α-drop).
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          for loc in [ChromaLocation::Center, ChromaLocation::Top] {
            let ya = run(&y, &u, &v, &a, sw, sh, ow, oh, loc.clone(), true);
            let yv = run_yuv(&y, &u, &v, sw, sh, ow, oh, loc.clone(), true);
            assert_eq!(ya.0, yv.0, "rgb {loc:?} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.2, yv.1, "rgb_u16 {loc:?} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.4, yv.2, "hsv {loc:?} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.5, yv.3, "luma {loc:?} ({sw}x{sh}->{ow}x{oh})");
            for px in 0..ow * oh {
              assert_eq!(
                &ya.1[px * 4..px * 4 + 3],
                &ya.0[px * 3..px * 3 + 3],
                "rgba colour vs rgb {loc:?} px {px}"
              );
              assert_eq!(
                &ya.3[px * 4..px * 4 + 3],
                &ya.2[px * 3..px * 3 + 3],
                "rgba_u16 colour vs rgb_u16 {loc:?} px {px}"
              );
            }
          }
        }
      }

      // ---- alpha preservation: siting never touches α ----------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn alpha_is_identical_between_cosited_and_centered() {
        for is_filter in [false, true] {
          let (y, u, v, a) = ramp(8, 8);
          let drive = |loc| {
            if is_filter {
              run_filter(&y, &u, &v, &a, 8, 8, 4, 4, loc, true)
            } else {
              run(&y, &u, &v, &a, 8, 8, 4, 4, loc, true)
            }
          };
          let cos = drive(ChromaLocation::Left);
          let cen = drive(ChromaLocation::Center);
          let cos_a: Vec<u8> = cos.1.iter().skip(3).step_by(4).copied().collect();
          let cen_a: Vec<u8> = cen.1.iter().skip(3).step_by(4).copied().collect();
          assert_eq!(
            cen_a, cos_a,
            "centered α (u8) must equal co-sited (filter={is_filter})"
          );
          let cos_a16: Vec<u16> = cos.3.iter().skip(3).step_by(4).copied().collect();
          let cen_a16: Vec<u16> = cen.3.iter().skip(3).step_by(4).copied().collect();
          assert_eq!(
            cen_a16, cos_a16,
            "centered α (u16) must equal co-sited (filter={is_filter})"
          );
          assert_ne!(
            cen.0, cos.0,
            "centered colour must differ (filter={is_filter})"
          );
        }
      }

      // ---- non-vacuous + flat-chroma sanity --------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_differs_from_cosited_on_a_chroma_ramp() {
        let (y, u, v, a) = ramp(8, 8);
        let cos = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Left, true);
        let cen = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, true);
        assert_ne!(
          cos.0, cen.0,
          "centered rgb must differ from co-sited on a ramp"
        );
        assert_ne!(
          cos.3, cen.3,
          "centered rgba_u16 must differ from co-sited on a ramp"
        );
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_equals_cosited_on_flat_chroma() {
        let (y, u, v, a) = flat(8, 8);
        let cos = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Left, true);
        let cen = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, true);
        assert_eq!(cos.1, cen.1, "flat-chroma rgba");
        assert_eq!(cos.3, cen.3, "flat-chroma rgba_u16");
        assert_eq!(cos.4, cen.4, "flat-chroma hsv");
      }

      // ---- SIMD == scalar --------------------------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_simd_matches_scalar() {
        let (y, u, v, a) = ramp(8, 8);
        assert_eq!(
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, true),
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, false),
          "centered area SIMD vs scalar must agree"
        );
        assert_eq!(
          run_filter(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, true),
          run_filter(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, false),
          "centered filter SIMD vs scalar must agree"
        );
      }

      // ---- wire endianness is siting-independent ---------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn be_centered_matches_le() {
        let (y, u, v, a) = ramp(8, 8);
        assert_eq!(
          run_be(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center),
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, true),
          "BE centered decode must equal LE for the same logical planes"
        );
      }

      // ---- mid-frame siting change is rejected across tiers ----------------

      /// Accept row 0 at `loc1` (freezes the phase), flip to `loc2`, feed the
      /// IN-SEQUENCE row 1 (4:2:2 chroma row 1), and return its result.
      fn flip_row1<R>(
        mut sink: MixedSinker<'_, $M422, R>,
        y: &[u16],
        u: &[u16],
        v: &[u16],
        a: &[u16],
        loc1: ChromaLocation,
        loc2: ChromaLocation,
      ) -> Result<(), MixedSinkerError> {
        sink.set_chroma_location(loc1.clone());
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        let row0 = $Row::new(&y[0..8], &u[0..4], &v[0..4], &a[0..8], 0, M, FR);
        PixelSink::process(&mut sink, row0).unwrap();
        sink.set_chroma_location(loc2.clone());
        let row1 = $Row::new(&y[8..16], &u[4..8], &v[4..8], &a[8..16], 1, M, FR);
        PixelSink::process(&mut sink, row1)
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn mid_frame_siting_change_rejected() {
        let (y, u, v, a) = ramp(8, 8);
        for (loc1, loc2) in [
          (ChromaLocation::Center, ChromaLocation::Left),
          (ChromaLocation::Left, ChromaLocation::Center),
        ] {
          // Area tier.
          let mut rgba = vec![0u8; 4 * 4 * 4];
          let sink =
            MixedSinker::<$M422, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_rgba(&mut rgba)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "area {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // u16-only colour area tier (sequences on the independent u16 stream).
          let mut rgba16 = vec![0u16; 4 * 4 * 4];
          let sink =
            MixedSinker::<$M422, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_rgba_u16(&mut rgba16)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "u16 area {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Filter tier.
          let mut rgba = vec![0u8; 4 * 4 * 4];
          let sink = MixedSinker::<$M422, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Straight)
          .with_rgba(&mut rgba)
          .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
        }
      }
    }
  };
}

hibit_422_yuva_resample_siting!(
  p10,
  10,
  Yuva422p10,
  Yuva422p10LeFrame,
  yuva422p10_to,
  Yuva422p10<true>,
  Yuva422p10BeFrame,
  yuva422p10_to_endian,
  Yuva422p10Row,
  Yuva444p10,
  Yuva444p10Frame,
  yuva444p10_to,
  Yuv422p10,
  Yuv422p10Frame,
  yuv422p10_to
);
hibit_422_yuva_resample_siting!(
  p12,
  12,
  Yuva422p12,
  Yuva422p12LeFrame,
  yuva422p12_to,
  Yuva422p12<true>,
  Yuva422p12BeFrame,
  yuva422p12_to_endian,
  Yuva422p12Row,
  Yuva444p12,
  Yuva444p12Frame,
  yuva444p12_to,
  Yuv422p12,
  Yuv422p12Frame,
  yuv422p12_to
);
hibit_422_yuva_resample_siting!(
  p16,
  16,
  Yuva422p16,
  Yuva422p16LeFrame,
  yuva422p16_to,
  Yuva422p16<true>,
  Yuva422p16BeFrame,
  yuva422p16_to_endian,
  Yuva422p16Row,
  Yuva444p16,
  Yuva444p16Frame,
  yuva444p16_to,
  Yuv422p16,
  Yuv422p16Frame,
  yuv422p16_to
);
