//! Chroma-siting-aware fused-downscale coverage for the HIGH-BIT **planar**
//! 4:2:0 YUV-with-alpha family — `Yuva420p9` / `Yuva420p10` / `Yuva420p12` /
//! `Yuva420p16` (RFC #238 S6c + S6f), the `u16` alpha-bearing twin of the 8-bit
//! `chroma_siting_yuva420p_resample` suite and the alpha-bearing sibling of the
//! high-bit planar `chroma_siting_hibit_420_resample` suite.
//!
//! `Yuva420pN` is planar 4:2:0 YUV (half-width, half-height U / V planes) PLUS a
//! **full-resolution** straight alpha plane that is never subsampled. S6c routes
//! the HORIZONTAL centered siting (`Center` / `Top`,
//! [`chroma_420_center_sited_h`](super::super::chroma_420_center_sited_h))
//! through the resample; RFC #238 S6f adds the VERTICAL `Bottom` (`v = 1`) fold
//! ([`chroma_420_bottom_sited_v`](super::super::chroma_420_bottom_sited_v)) — the
//! even output row box-blends its predecessor chroma row — while `Center` / `Top`
//! keep `v_phase = 0` (co-sited vertical). The full-resolution α plane is
//! siting-independent on every path.
//!
//! Unlike the no-alpha planar `Yuv420pN` (which owns a native code-domain
//! plane-binning fast tier), the high-bit `Yuva420pN` 4:2:0 sink has **only the
//! packed-YUVA tail** — every downscale converts each source row to RGBA (chroma
//! reconstructed in-register) then area- / filter-bins the packed RGBA. So the
//! centered area AND filter routes are both RGB-domain reconstruct-then-bin, and
//! the oracle drives a full-resolution `Yuva444pN` frame (U / V reconstructed to
//! full width per source row, each luma row `r` taking chroma row `r / 2`)
//! through the SAME packed tail via an identity-chroma `Yuva444pN` resample.
//!
//! Assertions per depth (`p10` low-packed, `p12` mid, `p16` full `u16`; BE + LE):
//!  - the co-sited / unspecified group stays byte-identical to the pre-siting
//!    resample (every output, α included);
//!  - centered area / filter RGBA (α carried) == the RGB-domain
//!    reconstruct-then-bin oracle, `u8` AND native `u16`;
//!  - the centered chroma decode (α-drop RGB / HSV / luma) is BIT-IDENTICAL to
//!    the no-alpha `Yuv420pN` row-stage centered resample (α is orthogonal);
//!  - the α channel is IDENTICAL between co-sited and centered (colour differs);
//!  - SIMD == scalar, BE == LE, centered != co-sited on a ramp / == on flat,
//!    and a mid-frame siting flip is rejected on both tiers.
//!
//! The RFC #238 S6f `Bottom` block mirrors each of those for the vertical fold:
//! area / filter RGBA (α carried) == the `Bottom` reconstruct-then-bin oracle;
//! the α-drop decode == the S6d-verified no-alpha `Yuv420pN` `Bottom`; α is
//! byte-identical co-sited vs `Bottom`; `Bottom` != `Center` on a vertical ramp /
//! == co-sited on flat; SIMD == scalar; BE == LE; the identity resample == the
//! direct decode; a mid-frame `Center` ⇆ `Bottom` flip is rejected; an odd-height
//! final-even-row reserve failure retries with the blend (not a clamp); and a
//! direct luma-only-then-late-colour attach box-blends the staged lookback.
//!
//! Every centered / bottom assertion is an EXACT match (single rounding for a
//! given tier), never a tolerance. Source heights are EVEN (except the odd-height
//! atomicity probe) so the plan's luma-domain V axis equals the co-sited box over
//! the `ch = sh / 2` chroma rows.

use crate::{
  ChromaLocation, ColorMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, FilteredResampler, Triangle},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::*,
};

const M: ColorMatrix = ColorMatrix::Bt601;
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

/// Reconstruct half-res `u16` chroma to full resolution (`sw x sh`) for the
/// bottom-sited (`v = 1`, RFC #238 S6f) decode — the identity bottom kernel at
/// source width: even rows box-blend chroma rows `i - 1` (clamped) and `i`
/// (round-half-up), odd rows take chroma row `i`, each then horizontally
/// upsampled with the #302 centered kernel ([`recon_full_row_u16`]). TWO
/// roundings (vertical then horizontal), exactly what the routed kernel
/// `chroma_upsample_420_bottom_even_h_u16` does — the shared reconstruction step
/// the packed YUVA area / filter tail reproduces for `Bottom`.
fn recon_full_bottom_u16(u: &[u16], v: &[u16], sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>) {
  let cw = sw / 2;
  let vblend = |plane: &[u16], cr: usize, prev: usize| -> Vec<u16> {
    (0..cw)
      .map(|c| {
        let a = u32::from(plane[prev * cw + c]);
        let b = u32::from(plane[cr * cw + c]);
        ((a + b + 1) >> 1) as u16
      })
      .collect::<Vec<u16>>()
  };
  let mut uf = vec![0u16; sw * sh];
  let mut vf = vec![0u16; sw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let (uh, vh) = if r & 1 == 0 {
      let prev = cr.saturating_sub(1);
      (vblend(u, cr, prev), vblend(v, cr, prev))
    } else {
      (
        u[cr * cw..cr * cw + cw].to_vec(),
        v[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_u16(&uh, cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_u16(&vh, cw));
  }
  (uf, vf)
}

/// The geometries the oracle-equality tests sweep: clean 2:1 and fractional
/// ratios, EVEN source heights only (so the vertical luma-domain pairing equals
/// the co-sited chroma box). Widths are even (`Yuva420p` requires it).
const GEOMS: [(usize, usize, usize, usize); 4] =
  [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)];

/// `(rgb, rgba, rgb_u16, rgba_u16, (h,s,v), luma, luma_u16)` — the full output
/// set every driver returns so a single call feeds every assertion.
type Outs = (
  Vec<u8>,
  Vec<u8>,
  Vec<u16>,
  Vec<u16>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);
/// The no-alpha `Yuv420pN` cross-check outputs: `(rgb, rgb_u16, (h,s,v), luma)`
/// — the α-drop channels the alpha-bearing decode must match. (`Yuv420pN`
/// exposes no `luma_u16`; the YUVA `luma_u16` is siting-independent native Y and
/// is pinned by the co-sited suite.)
type YuvOuts = (Vec<u8>, Vec<u16>, (Vec<u8>, Vec<u8>, Vec<u8>), Vec<u8>);

macro_rules! hibit_420_yuva_resample_siting {
  (
    $mod:ident, $bits:expr,
    $M420:ident, $F420le:ident, $w420:ident,
    $M420be:ty, $F420be:ident, $w420be:ident, $Row:ident,
    $M444:ident, $F444:ident, $w444:ident,
    $MYuv:ident, $FYuv:ident, $wYuv:ident
  ) => {
    mod $mod {
      use super::*;

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = 1u16 << ($bits - 1);

      /// A `Yuva420pN` fixture with a strong HORIZONTAL chroma ramp (so the
      /// centered triangle genuinely differs from the co-sited nearest decode)
      /// plus a per-row tilt (a vertical mistake would show) and a varying,
      /// non-opaque alpha (so the α-preservation assertions are non-vacuous).
      /// Low-packed native codes so every kernel sees real math.
      fn ramp(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let (cw, ch) = (sw / 2, sh / 2);
        let step = (MASK as u32 / 16).max(1);
        let mut y = vec![0u16; sw * sh];
        let mut u = vec![0u16; cw * ch];
        let mut v = vec![0u16; cw * ch];
        let mut a = vec![0u16; sw * sh];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 37) & MASK as u32) as u16;
        }
        for r in 0..ch {
          for c in 0..cw {
            u[r * cw + c] = (step * c as u32 + step + r as u32 * 5).min(MASK as u32) as u16;
            v[r * cw + c] = (MASK as u32).saturating_sub(step * c as u32).max(step) as u16;
          }
        }
        // Non-opaque, varying α across the full resolution.
        for (i, p) in a.iter_mut().enumerate() {
          *p = ((MASK as u32 / 5 + i as u32 * 23) & MASK as u32) as u16;
        }
        (y, u, v, a)
      }

      /// Flat chroma: the centered triangle of a constant is that constant, so
      /// centered must equal co-sited. Luma and α still vary.
      fn flat(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let (cw, ch) = (sw / 2, sh / 2);
        let mut y = vec![0u16; sw * sh];
        let mut a = vec![0u16; sw * sh];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 29) & MASK as u32) as u16;
        }
        for (i, p) in a.iter_mut().enumerate() {
          *p = ((MASK as u32 / 3 + i as u32 * 17) & MASK as u32) as u16;
        }
        (y, vec![MID; cw * ch], vec![MID; cw * ch], a)
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

      /// Drive an LE `Yuva420pN` STRAIGHT-alpha AREA resample for the full
      /// output set (the high-bit YUVA 4:2:0 sink is packed-only, so `native`
      /// has no fast-tier fork — the packed area tail runs regardless).
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
            MixedSinker::<$M420, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_chroma_location(loc)
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
          let f = $F420le::try_new(
            y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
          )
          .unwrap();
          $w420(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, rgb16, rgba16, (hh, ss, vv), luma, luma16)
      }

      /// Drive an LE `Yuva420pN` single-kernel filter resample (Triangle) for
      /// rgba + rgba_u16 + the α-drop channels.
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
          let mut sink = MixedSinker::<$M420, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Straight)
          .with_chroma_location(loc)
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
          let f = $F420le::try_new(
            y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
          )
          .unwrap();
          $w420(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, rgb16, rgba16, (hh, ss, vv), luma, luma16)
      }

      /// Drive a BE `Yuva420pN` AREA resample (planes re-encoded BE-wire).
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
          let mut sink = MixedSinker::<$M420be, AreaResampler>::with_resampler(
            sw,
            sh,
            AreaResampler::to(ow, oh),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Straight)
          .with_chroma_location(loc)
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
          let f = $F420be::try_new(
            &yb, &ub, &vb, &ab, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
          )
          .unwrap();
          $w420be(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, rgb16, rgba16, (hh, ss, vv), luma, luma16)
      }

      /// The centered RGB-domain oracle for the packed area / filter tail:
      /// reconstruct U / V to full width (`u16`) with the #302 kernel — each luma
      /// row `r` taking chroma row `r / 2` (co-sited vertical replication) — then
      /// run that full-resolution `Yuva444pN` frame (with the UNTOUCHED α plane)
      /// through the given resampler with STRAIGHT alpha. Returns `(rgba,
      /// rgba_u16)` — the exact convert-each-row-then-bin the routed path does.
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
          let cr = r / 2;
          uf[r * sw..r * sw + sw]
            .copy_from_slice(&recon_full_row_u16(&u[cr * cw..cr * cw + cw], cw));
          vf[r * sw..r * sw + sw]
            .copy_from_slice(&recon_full_row_u16(&v[cr * cw..cr * cw + cw], cw));
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

      /// Drive the no-alpha `Yuv420pN` resample on the SAME Y / U / V through the
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
              .with_chroma_location(loc)
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
            ChromaLocation::Unknown(7),
          ] {
            assert_eq!(
              run(&y, &u, &v, &a, sw, sh, ow, oh, loc, true),
              base,
              "co-sited {loc:?} area must keep the byte-identical decode \
               ({sw}x{sh}->{ow}x{oh})"
            );
          }
          // The filter tier too.
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
            let got = run(&y, &u, &v, &a, sw, sh, ow, oh, loc, true);
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

      // ---- STRONG cross-check: centered chroma decode == Yuv420pN centered --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_chroma_matches_yuv420p_centered() {
        // α is orthogonal to chroma siting, so the centered `Yuva420pN` Y/U/V →
        // RGB (and RGB_u16 / HSV / luma) decode must be BIT-IDENTICAL to the
        // no-alpha `Yuv420pN` ROW-STAGE centered resample of the same Y/U/V (the
        // packed YUVA tail is RGB-domain, so it matches the row-stage tier, α-drop).
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          for loc in [ChromaLocation::Center, ChromaLocation::Top] {
            let ya = run(&y, &u, &v, &a, sw, sh, ow, oh, loc, true);
            let yv = run_yuv(&y, &u, &v, sw, sh, ow, oh, loc, true);
            assert_eq!(ya.0, yv.0, "rgb {loc:?} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.2, yv.1, "rgb_u16 {loc:?} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.4, yv.2, "hsv {loc:?} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.5, yv.3, "luma {loc:?} ({sw}x{sh}->{ow}x{oh})");
            // The RGBA / RGBA_u16 colour channels equal the same centered RGB.
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
        // Chroma siting must not touch the full-resolution α plane: the centered
        // RGBA's alpha channel equals the co-sited path's alpha byte-for-byte on
        // BOTH tiers (u8 and u16) — while the colour channels DO differ.
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
      /// IN-SEQUENCE row 1 (chroma row `1 / 2 = 0`), and return its result.
      fn flip_row1<R>(
        mut sink: MixedSinker<'_, $M420, R>,
        y: &[u16],
        u: &[u16],
        v: &[u16],
        a: &[u16],
        loc1: ChromaLocation,
        loc2: ChromaLocation,
      ) -> Result<(), MixedSinkerError> {
        sink.set_chroma_location(loc1);
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        let row0 = $Row::new(&y[0..8], &u[0..4], &v[0..4], &a[0..8], 0, M, FR);
        PixelSink::process(&mut sink, row0).unwrap();
        sink.set_chroma_location(loc2);
        let row1 = $Row::new(&y[8..16], &u[0..4], &v[0..4], &a[8..16], 1, M, FR);
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
            MixedSinker::<$M420, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_rgba(&mut rgba)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "area {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // u16-only colour area tier (sequences on the independent u16 stream).
          let mut rgba16 = vec![0u16; 4 * 4 * 4];
          let sink =
            MixedSinker::<$M420, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_rgba_u16(&mut rgba16)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "u16 area {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Filter tier.
          let mut rgba = vec![0u8; 4 * 4 * 4];
          let sink = MixedSinker::<$M420, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Straight)
          .with_rgba(&mut rgba)
          .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
        }
      }

      // ==== RFC #238 S6f: bottom-sited (v = 1) VERTICAL fold =================
      //
      // The high-bit YUVA 4:2:0 sink is PACKED-ONLY (no native code-domain tier),
      // so `Bottom` folds via the RGB-domain reconstruct-then-bin — the same
      // packed area / filter tail as the centered horizontal path, but the chroma
      // reconstruction ADDS the `v = 1` vertical blend of the previous chroma row.

      /// Flat-luma fixture with a strong per-ROW chroma step (flat across
      /// columns) plus a varying, non-opaque α: the vertical `Bottom` fold is
      /// observable in isolation (a horizontal-only siting leaves it untouched, so
      /// `Bottom` visibly diverges from `Center`), while α stays non-vacuous.
      fn vramp(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let (cw, ch) = (sw / 2, sh / 2);
        let step = (MASK as u32 / 8).max(1);
        let y = vec![MID; sw * sh];
        let mut u = vec![0u16; cw * ch];
        let mut v = vec![0u16; cw * ch];
        let mut a = vec![0u16; sw * sh];
        for r in 0..ch {
          for c in 0..cw {
            u[r * cw + c] = (step + r as u32 * step).min(MASK as u32) as u16;
            v[r * cw + c] = (MASK as u32).saturating_sub(r as u32 * step).max(step) as u16;
          }
        }
        for (i, p) in a.iter_mut().enumerate() {
          *p = ((MASK as u32 / 5 + i as u32 * 23) & MASK as u32) as u16;
        }
        (y, u, v, a)
      }

      /// The bottom-sited RGB-domain oracle for the packed area / filter tail:
      /// reconstruct U / V to full width with the vertical `Bottom` blend
      /// ([`recon_full_bottom_u16`]) — each even luma row box-blending its
      /// predecessor chroma row — then run that full-resolution `Yuva444pN` frame
      /// (α UNTOUCHED) through the given resampler with STRAIGHT alpha. Returns
      /// `(rgba, rgba_u16)`, the exact convert-each-row-then-bin the routed
      /// `Bottom` path does. `filter = true` uses the Triangle tier.
      #[allow(clippy::too_many_arguments)]
      fn rgba_oracle_bottom(
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
        let (uf, vf) = recon_full_bottom_u16(u, v, sw, sh);
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

      /// Direct (non-resample, identity dims) `Yuva420pN` `Bottom` decode to
      /// straight RGBA (u8 + native u16) — the delay-line kernel path, the
      /// already-validated identity reference the resample path must match at
      /// identity dimensions (colour AND α).
      fn direct_bottom_yuva(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        a: &[u16],
        sw: usize,
        sh: usize,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let cw = sw / 2;
        let mut rgba = vec![0u8; sw * sh * 4];
        let mut rgba16 = vec![0u16; sw * sh * 4];
        {
          let mut sink = MixedSinker::<$M420>::new(sw, sh)
            .with_alpha_mode(AlphaMode::Straight)
            .with_chroma_location(ChromaLocation::Bottom)
            .with_simd(simd)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_rgba_u16(&mut rgba16)
            .unwrap();
          let f = $F420le::try_new(
            y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
          )
          .unwrap();
          $w420(&f, FR, M, &mut sink).unwrap();
        }
        (rgba, rgba16)
      }

      // ---- Area / Filter RGBA == RGB-domain reconstruct-then-bin (α carried) --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_area_rgba_equals_rgb_domain_oracle() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          let (want_rgba, want_rgba16) =
            rgba_oracle_bottom(&y, &u, &v, &a, sw, sh, ow, oh, true, false);
          let got = run(&y, &u, &v, &a, sw, sh, ow, oh, ChromaLocation::Bottom, true);
          assert_eq!(got.1, want_rgba, "bottom area rgba ({sw}x{sh}->{ow}x{oh})");
          assert_eq!(
            got.3, want_rgba16,
            "bottom area rgba_u16 ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_filter_rgba_equals_rgb_domain_oracle() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          let (want_rgba, want_rgba16) =
            rgba_oracle_bottom(&y, &u, &v, &a, sw, sh, ow, oh, true, true);
          let got = run_filter(&y, &u, &v, &a, sw, sh, ow, oh, ChromaLocation::Bottom, true);
          assert_eq!(got.1, want_rgba, "bottom filter rgba ({sw}x{sh}->{ow}x{oh})");
          assert_eq!(
            got.3, want_rgba16,
            "bottom filter rgba_u16 ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      // ---- α-drop Bottom decode == the S6d-verified no-alpha Yuv420pN Bottom --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_chroma_matches_yuv420p_bottom() {
        // α is orthogonal to chroma siting, so the Bottom `Yuva420pN` α-drop
        // decode (RGB / RGB_u16 / HSV / luma) must be BIT-IDENTICAL to the no-alpha
        // `Yuv420pN` ROW-STAGE Bottom resample (S6d-verified) of the same Y/U/V.
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          let ya = run(&y, &u, &v, &a, sw, sh, ow, oh, ChromaLocation::Bottom, true);
          let yv = run_yuv(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Bottom, true);
          assert_eq!(ya.0, yv.0, "rgb ({sw}x{sh}->{ow}x{oh})");
          assert_eq!(ya.2, yv.1, "rgb_u16 ({sw}x{sh}->{ow}x{oh})");
          assert_eq!(ya.4, yv.2, "hsv ({sw}x{sh}->{ow}x{oh})");
          assert_eq!(ya.5, yv.3, "luma ({sw}x{sh}->{ow}x{oh})");
        }
      }

      // ---- α preservation: the vertical fold never touches α -----------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_alpha_is_identical_to_cosited() {
        // The Bottom RGBA's alpha channel equals the co-sited path's α byte-for-
        // byte on BOTH tiers (u8 and u16), while the colour channels DO move (a
        // vertical chroma ramp, so Bottom genuinely diverges — the non-vacuous
        // control).
        for is_filter in [false, true] {
          let (y, u, v, a) = vramp(8, 8);
          let drive = |loc| {
            if is_filter {
              run_filter(&y, &u, &v, &a, 8, 8, 4, 4, loc, true)
            } else {
              run(&y, &u, &v, &a, 8, 8, 4, 4, loc, true)
            }
          };
          let cos = drive(ChromaLocation::Left);
          let bot = drive(ChromaLocation::Bottom);
          let cos_a: Vec<u8> = cos.1.iter().skip(3).step_by(4).copied().collect();
          let bot_a: Vec<u8> = bot.1.iter().skip(3).step_by(4).copied().collect();
          assert_eq!(
            bot_a, cos_a,
            "bottom α (u8) must equal co-sited (filter={is_filter})"
          );
          let cos_a16: Vec<u16> = cos.3.iter().skip(3).step_by(4).copied().collect();
          let bot_a16: Vec<u16> = bot.3.iter().skip(3).step_by(4).copied().collect();
          assert_eq!(
            bot_a16, cos_a16,
            "bottom α (u16) must equal co-sited (filter={is_filter})"
          );
          assert_ne!(
            bot.0, cos.0,
            "bottom colour must differ from co-sited (filter={is_filter})"
          );
        }
      }

      // ---- non-vacuous + flat-chroma sanity ----------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_differs_from_center_on_a_vertical_chroma_ramp() {
        // The v=1 fold must actually MOVE the chroma: on a purely-vertical chroma
        // ramp Bottom diverges from the vertically-co-sited Center on both tiers.
        let (y, u, v, a) = vramp(8, 8);
        let cen = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, true);
        let bot = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, true);
        assert_ne!(bot.0, cen.0, "bottom area rgb must differ from center");
        assert_ne!(bot.3, cen.3, "bottom area rgba_u16 must differ from center");
        let cenf = run_filter(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, true);
        let botf = run_filter(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, true);
        assert_ne!(botf.1, cenf.1, "bottom filter rgba must differ from center");
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_equals_cosited_on_flat_chroma() {
        // Constant chroma: the vertical blend and horizontal triangle are both
        // no-ops, so Bottom collapses to the co-sited decode byte-for-byte.
        let (y, u, v, a) = flat(8, 8);
        let cos = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Left, true);
        let bot = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, true);
        assert_eq!(cos.1, bot.1, "flat-chroma bottom rgba");
        assert_eq!(cos.3, bot.3, "flat-chroma bottom rgba_u16");
        assert_eq!(cos.4, bot.4, "flat-chroma bottom hsv");
      }

      // ---- SIMD == scalar, BE == LE ------------------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_simd_matches_scalar() {
        let (y, u, v, a) = ramp(8, 8);
        assert_eq!(
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, true),
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, false),
          "bottom area SIMD vs scalar must agree"
        );
        assert_eq!(
          run_filter(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, true),
          run_filter(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, false),
          "bottom filter SIMD vs scalar must agree"
        );
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_be_matches_le() {
        let (y, u, v, a) = ramp(8, 8);
        assert_eq!(
          run_be(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom),
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, true),
          "BE bottom decode must equal LE for the same logical planes"
        );
      }

      // ---- bottom-LEFT-sited (co-sited h + v = 1) ---------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottomleft_chroma_matches_yuv420p_bottomleft() {
        // α is orthogonal to chroma siting, so the BottomLeft `Yuva420pN` α-drop
        // decode must be BIT-IDENTICAL to the no-alpha `Yuv420pN` BottomLeft
        // resample (oracle-verified) of the same Y/U/V — native + row-stage.
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v, a) = ramp(sw, sh);
          for native in [true, false] {
            let ya = run(&y, &u, &v, &a, sw, sh, ow, oh, ChromaLocation::BottomLeft, native);
            let yv = run_yuv(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::BottomLeft, native);
            assert_eq!(ya.0, yv.0, "rgb native={native} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.2, yv.1, "rgb_u16 native={native} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.4, yv.2, "hsv native={native} ({sw}x{sh}->{ow}x{oh})");
            assert_eq!(ya.5, yv.3, "luma native={native} ({sw}x{sh}->{ow}x{oh})");
          }
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottomleft_alpha_is_identical_to_cosited_and_folds_the_phase() {
        // BottomLeft's α equals the co-sited α byte-for-byte on both tiers, while
        // the colour channels move (vertical ramp), and BottomLeft differs from
        // Bottom on a horizontal ramp (the co-sited-h phase).
        for is_filter in [false, true] {
          let (y, u, v, a) = vramp(8, 8);
          let drive = |loc| {
            if is_filter {
              run_filter(&y, &u, &v, &a, 8, 8, 4, 4, loc, true)
            } else {
              run(&y, &u, &v, &a, 8, 8, 4, 4, loc, true)
            }
          };
          let cos = drive(ChromaLocation::Left);
          let bl = drive(ChromaLocation::BottomLeft);
          let cos_a16: Vec<u16> = cos.3.iter().skip(3).step_by(4).copied().collect();
          let bl_a16: Vec<u16> = bl.3.iter().skip(3).step_by(4).copied().collect();
          assert_eq!(
            bl_a16, cos_a16,
            "bottom-left α (u16) must equal co-sited (filter={is_filter})"
          );
          assert_ne!(
            bl.0, cos.0,
            "bottom-left colour must differ from co-sited (filter={is_filter})"
          );
        }
        let (y, u, v, a) = ramp(8, 8);
        assert_ne!(
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::BottomLeft, true).0,
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Bottom, true).0,
          "bottom-left (h=0) must differ from bottom (h=0.5) on a horizontal ramp"
        );
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottomleft_be_and_simd_agree() {
        let (y, u, v, a) = ramp(8, 8);
        assert_eq!(
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::BottomLeft, true),
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::BottomLeft, false),
          "bottom-left area SIMD vs scalar must agree"
        );
        assert_eq!(
          run_filter(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::BottomLeft, true),
          run_filter(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::BottomLeft, false),
          "bottom-left filter SIMD vs scalar must agree"
        );
        assert_eq!(
          run_be(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::BottomLeft),
          run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::BottomLeft, true),
          "BE bottom-left must equal LE"
        );
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottomleft_equals_cosited_on_flat_chroma() {
        let (y, u, v, a) = flat(8, 8);
        let cos = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Left, true);
        let bl = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::BottomLeft, true);
        assert_eq!(cos.3, bl.3, "flat-chroma bottom-left rgba_u16");
        assert_eq!(cos.4, bl.4, "flat-chroma bottom-left hsv");
      }

      // ---- identity resample == direct decode --------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_identity_resample_matches_direct_decode() {
        // At identity dims (out == src) the resample tier reconstructs chroma with
        // the SAME bottom kernel as the direct (non-resample) decode and bins with
        // pass-through area weights, so routing Bottom through the resample
        // preserves the alpha-aware decode byte-for-byte (colour AND α).
        let (y, u, v, a) = vramp(8, 8);
        let res = run(&y, &u, &v, &a, 8, 8, 8, 8, ChromaLocation::Bottom, true);
        let (drgba, drgba16) = direct_bottom_yuva(&y, &u, &v, &a, 8, 8, true);
        assert_eq!(res.1, drgba, "identity resample bottom rgba == direct decode");
        assert_eq!(
          res.3, drgba16,
          "identity resample bottom rgba_u16 == direct decode"
        );
      }

      // ---- odd-height final-even-row reserve failure retries with the blend --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_final_even_row_reserve_alloc_failure_retries_with_blend() {
        // ODD source height => the FINAL luma row is EVEN, so its Bottom vertical
        // blend reads the PREDECESSOR chroma row via the `chroma_prev_u16`
        // lookback. If the reservation is refused on that final row, the retry must
        // STILL blend with the predecessor: the reserve sits BEHIND the resample
        // preflight (#180) and the arms DEFER the stage until the commit accepts
        // the row, so a rejected row leaves the lookback (and every output)
        // untouched. Without the atomicity a failed attempt would clamp the retry.
        use crate::resample::ResampleError;
        let (sw, sh, ow, oh) = (8usize, 7usize, 4usize, 3usize);
        let cw = sw / 2;
        let ch = sh.div_ceil(2);
        let step = (MASK as u32 / 8).max(1);
        let y = vec![MID; sw * sh];
        let mut u = vec![0u16; cw * ch];
        let mut v = vec![0u16; cw * ch];
        for r in 0..ch {
          for c in 0..cw {
            u[r * cw + c] = (step + r as u32 * step).min(MASK as u32) as u16;
            v[r * cw + c] = (MASK as u32).saturating_sub(r as u32 * step).max(step) as u16;
          }
        }
        let mut a = vec![0u16; sw * sh];
        for (i, p) in a.iter_mut().enumerate() {
          *p = ((MASK as u32 / 5 + i as u32 * 23) & MASK as u32) as u16;
        }
        let run_frame = |arm_fail: bool, loc: ChromaLocation| -> (Vec<u8>, Vec<u16>) {
          let mut rgba = vec![0u8; ow * oh * 4];
          let mut rgba16 = vec![0u16; ow * oh * 4];
          {
            let mut sink = MixedSinker::<$M420, AreaResampler>::with_resampler(
              sw,
              sh,
              AreaResampler::to(ow, oh),
            )
            .unwrap()
            .with_alpha_mode(AlphaMode::Straight)
            .with_chroma_location(loc)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_rgba_u16(&mut rgba16)
            .unwrap();
            sink.begin_frame(sw as u32, sh as u32).unwrap();
            let feed = |sink: &mut MixedSinker<'_, $M420, AreaResampler>, r: usize| {
              let yr = &y[r * sw..(r + 1) * sw];
              let cr = r / 2;
              let ur = &u[cr * cw..(cr + 1) * cw];
              let vr = &v[cr * cw..(cr + 1) * cw];
              let ar = &a[r * sw..(r + 1) * sw];
              sink.process($Row::new(yr, ur, vr, ar, r, M, FR))
            };
            for r in 0..sh - 1 {
              feed(&mut sink, r).unwrap();
            }
            if arm_fail {
              super::super::arm_chroma_prev_alloc_failure();
              let err = feed(&mut sink, sh - 1).unwrap_err();
              assert!(
                matches!(
                  err,
                  MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
                ),
                "armed final-row lookback reserve must surface AllocationFailed, got {err:?}"
              );
            }
            // Retry the SAME final row (the failpoint is one-shot, already taken).
            feed(&mut sink, sh - 1).unwrap();
          }
          (rgba, rgba16)
        };
        let reference = run_frame(false, ChromaLocation::Bottom);
        let retried = run_frame(true, ChromaLocation::Bottom);
        assert_eq!(
          retried.0, reference.0,
          "post-failure retry rgba must blend with the predecessor (not clamp)"
        );
        assert_eq!(
          retried.1, reference.1,
          "post-failure retry rgba_u16 (colour + α) must match the clean run"
        );
        // Non-vacuous: the bottom vertical blend genuinely moves the final even
        // row vs the vertically-co-sited Center decode.
        assert_ne!(
          reference.0,
          run_frame(false, ChromaLocation::Center).0,
          "the bottom vertical blend must move the final even row vs co-sited Center"
        );
      }

      // ---- mid-frame Center <-> Bottom flip is rejected ----------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn mid_frame_center_bottom_flip_rejected() {
        // Center and Bottom are BOTH horizontally centered, so the `center_sited`
        // flag alone cannot tell them apart; the frozen VERTICAL phase rejects the
        // flip on every tier.
        let (y, u, v, a) = vramp(8, 8);
        for (loc1, loc2) in [
          (ChromaLocation::Center, ChromaLocation::Bottom),
          (ChromaLocation::Bottom, ChromaLocation::Center),
        ] {
          // Area tier.
          let mut rgba = vec![0u8; 4 * 4 * 4];
          let sink =
            MixedSinker::<$M420, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_rgba(&mut rgba)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "area {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // u16-only colour area tier (sequences on the independent u16 stream).
          let mut rgba16 = vec![0u16; 4 * 4 * 4];
          let sink =
            MixedSinker::<$M420, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_alpha_mode(AlphaMode::Straight)
              .with_rgba_u16(&mut rgba16)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "u16 area {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Filter tier.
          let mut rgba = vec![0u8; 4 * 4 * 4];
          let sink = MixedSinker::<$M420, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Straight)
          .with_rgba(&mut rgba)
          .unwrap();
          let err = flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
        }
      }

      // ---- direct decode: luma-only rows STAGE the lookback for a late colour --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_direct_luma_only_then_late_colour_attach_blends_not_clamps() {
        // A `Yuva420pN` `Bottom` DIRECT decode (no resampler) where rows 0/1 are
        // luma-only, then the caller attaches RGBA before row 2 (direct sinks allow
        // late-attach; no frozen-output lock). Row 2 is EVEN — its Bottom blend
        // reads the PREDECESSOR chroma row via the lookback — so the luma-only rows
        // 0/1 must have STAGED it. Its colour + α must equal the all-output Bottom
        // reference (box-blend, NOT clamp). Non-vacuous: on a vertical chroma ramp
        // a clamped row 2 (empty lookback) differs from the blended one.
        let (y, u, v, a) = vramp(8, 8);
        let (sw, sh, cw) = (8usize, 8usize, 4usize);
        let row_at = |r: usize| {
          let cr = r / 2;
          $Row::new(
            &y[r * sw..r * sw + sw],
            &u[cr * cw..cr * cw + cw],
            &v[cr * cw..cr * cw + cw],
            &a[r * sw..r * sw + sw],
            r,
            M,
            FR,
          )
        };

        // Reference: full-output Bottom direct decode (every row carries RGBA, so
        // its lookback is staged inside the colour upsample on every row).
        let reference = direct_bottom_yuva(&y, &u, &v, &a, sw, sh, true).0;

        // Late-attach: rows 0/1 luma-only, RGBA attached before row 2.
        let mut luma = vec![0u8; sw * sh];
        let mut rgba = vec![0u8; sw * sh * 4];
        {
          let mut sink = MixedSinker::<$M420>::new(sw, sh)
            .with_alpha_mode(AlphaMode::Straight)
            .with_chroma_location(ChromaLocation::Bottom)
            .with_simd(true)
            .with_luma(&mut luma)
            .unwrap();
          PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
          for r in 0..2 {
            PixelSink::process(&mut sink, row_at(r)).unwrap();
          }
          sink.set_rgba(&mut rgba).unwrap();
          for r in 2..sh {
            PixelSink::process(&mut sink, row_at(r)).unwrap();
          }
        }
        let (row2, row3) = (2 * sw * 4, 3 * sw * 4);
        assert_eq!(
          &rgba[row2..row3],
          &reference[row2..row3],
          "late-colour-attach row 2 must box-blend the luma-only-staged chroma (colour + α), not clamp"
        );

        // Non-vacuous control: feed row 2 as the FIRST row of a fresh frame, so the
        // lookback is empty and the even-row blend CLAMPS to the current chroma row.
        let mut rgba_clamp = vec![0u8; sw * sh * 4];
        {
          let mut sink = MixedSinker::<$M420>::new(sw, sh)
            .with_alpha_mode(AlphaMode::Straight)
            .with_chroma_location(ChromaLocation::Bottom)
            .with_simd(true)
            .with_rgba(&mut rgba_clamp)
            .unwrap();
          PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
          PixelSink::process(&mut sink, row_at(2)).unwrap();
        }
        assert_ne!(
          &rgba_clamp[row2..row3],
          &reference[row2..row3],
          "control: a clamped row 2 (empty lookback) must differ from the blended reference"
        );
      }

      // ---- begin_frame invalidates the lookback (no cross-frame stale blend) --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_two_frames_no_cross_frame_stale_blend() {
        // `begin_frame` must invalidate the bottom-sited lookback tag
        // (`chroma_prev_row`): otherwise a new frame whose first decoded EVEN row
        // (`chroma_row > 0`) reaches the blend before its OWN predecessor is staged
        // would falsely match the PREVIOUS frame's stale tag and box-blend that
        // frame's chroma. After `begin_frame` the lookback is empty, so such a row
        // must CLAMP to the current chroma row alone — byte-identical to a fresh
        // sink. (`reset_high_bit_yuva_streams` clears the tag; a missed reset here
        // silently contaminates frame N+1.)
        let (y, u, v, a) = vramp(8, 8);
        let (sw, sh, cw) = (8usize, 8usize, 4usize);
        let row_at = |r: usize| {
          let cr = r / 2;
          $Row::new(
            &y[r * sw..r * sw + sw],
            &u[cr * cw..cr * cw + cw],
            &v[cr * cw..cr * cw + cw],
            &a[r * sw..r * sw + sw],
            r,
            M,
            FR,
          )
        };
        let (row2, row3) = (2 * sw * 4, 3 * sw * 4);

        // FRESH sink: begin_frame then feed ONLY row 2 (even, chroma_row 1). The
        // empty lookback forces the even-row blend to CLAMP (no predecessor) — the
        // correct, contamination-free decode of this partial frame.
        let mut fresh = std::vec![0u8; sw * sh * 4];
        {
          let mut sink = MixedSinker::<$M420>::new(sw, sh)
            .with_alpha_mode(AlphaMode::Straight)
            .with_chroma_location(ChromaLocation::Bottom)
            .with_simd(true)
            .with_rgba(&mut fresh)
            .unwrap();
          PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
          PixelSink::process(&mut sink, row_at(2)).unwrap();
        }

        // Non-vacuous control: a FULL in-order frame's row 2 DOES blend (its own
        // rows 0/1 stage the predecessor), so it differs from the clamped partial
        // decode — proving the main assertion distinguishes a blend from a clamp.
        let mut full = std::vec![0u8; sw * sh * 4];
        {
          let mut sink = MixedSinker::<$M420>::new(sw, sh)
            .with_alpha_mode(AlphaMode::Straight)
            .with_chroma_location(ChromaLocation::Bottom)
            .with_simd(true)
            .with_rgba(&mut full)
            .unwrap();
          PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
          for r in 0..sh {
            PixelSink::process(&mut sink, row_at(r)).unwrap();
          }
        }
        assert_ne!(
          &full[row2..row3],
          &fresh[row2..row3],
          "control: a full-frame row 2 blends its predecessor (differs from the clamped partial decode)"
        );

        // REUSED sink: frame 1 feeds rows 0/1 (chroma row 0), priming the lookback
        // `chroma_prev_row = Some(0)` + its chroma buffer; then `begin_frame`, then
        // feed ONLY row 2 of frame 2. Row 2 needs predecessor tag `Some(0)` — the
        // stale frame-1 tag would FALSELY match without the reset. `begin_frame`
        // must clear it so row 2 clamps EXACTLY as the fresh sink.
        let mut reused = std::vec![0u8; sw * sh * 4];
        {
          let mut sink = MixedSinker::<$M420>::new(sw, sh)
            .with_alpha_mode(AlphaMode::Straight)
            .with_chroma_location(ChromaLocation::Bottom)
            .with_simd(true)
            .with_rgba(&mut reused)
            .unwrap();
          PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
          PixelSink::process(&mut sink, row_at(0)).unwrap();
          PixelSink::process(&mut sink, row_at(1)).unwrap();
          PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
          PixelSink::process(&mut sink, row_at(2)).unwrap();
        }
        assert_eq!(
          &reused[row2..row3],
          &fresh[row2..row3],
          "begin_frame must invalidate the Bottom lookback: frame 2's first even row \
           must clamp (== fresh sink), never box-blend the previous frame's stale chroma"
        );
      }
    }
  };
}

hibit_420_yuva_resample_siting!(
  p10,
  10,
  Yuva420p10,
  Yuva420p10LeFrame,
  yuva420p10_to,
  Yuva420p10<true>,
  Yuva420p10BeFrame,
  yuva420p10_to_endian,
  Yuva420p10Row,
  Yuva444p10,
  Yuva444p10Frame,
  yuva444p10_to,
  Yuv420p10,
  Yuv420p10Frame,
  yuv420p10_to
);
hibit_420_yuva_resample_siting!(
  p12,
  12,
  Yuva420p12,
  Yuva420p12LeFrame,
  yuva420p12_to,
  Yuva420p12<true>,
  Yuva420p12BeFrame,
  yuva420p12_to_endian,
  Yuva420p12Row,
  Yuva444p12,
  Yuva444p12Frame,
  yuva444p12_to,
  Yuv420p12,
  Yuv420p12Frame,
  yuv420p12_to
);
hibit_420_yuva_resample_siting!(
  p16,
  16,
  Yuva420p16,
  Yuva420p16LeFrame,
  yuva420p16_to,
  Yuva420p16<true>,
  Yuva420p16BeFrame,
  yuva420p16_to_endian,
  Yuva420p16Row,
  Yuva444p16,
  Yuva444p16Frame,
  yuva444p16_to,
  Yuv420p16,
  Yuv420p16Frame,
  yuv420p16_to
);
