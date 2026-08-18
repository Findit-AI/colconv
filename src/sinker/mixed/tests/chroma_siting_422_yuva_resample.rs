//! RFC #238 S7 — chroma-siting-aware 4:2:2 **resample** for the 8-bit `Yuva422p`,
//! the alpha-bearing twin of `chroma_siting_422_resample` (S1, `Yuv422p`) and the
//! 4:2:2 sibling of `chroma_siting_yuva420p_resample` (S3c).
//!
//! `Yuva422p` is planar 4:2:2 YUV (half-width, FULL-height U / V planes) PLUS a
//! **full-resolution** straight alpha plane that is never subsampled. 4:2:2 is
//! subsampled horizontally only — there is no vertical chroma phase — so S7 routes
//! the HORIZONTAL centered siting (`Center` / `Top`,
//! [`chroma_422_center_sited_h`](super::super::chroma_422_center_sited_h)) through
//! the resample. The α plane is orthogonal to chroma siting; it bins on the luma
//! grid unchanged on every path, so the effective siting touches ONLY the Y/U/V
//! colour decode.
//!
//! Unlike the no-alpha `Yuv422p` (which owns a native code-domain fast tier), the
//! 8-bit `Yuva422p` sink is **packed-only** — every downscale converts each source
//! row to RGBA (chroma reconstructed in-register) then area- / filter-bins the
//! packed RGBA. So the centered area AND filter routes are both RGB-domain
//! reconstruct-then-bin, and the RGBA oracle drives a full-resolution `Yuva444p`
//! frame (U / V reconstructed to full width per source row) through the SAME
//! packed tail via an identity-chroma `Yuva444p` resample.
//!
//! Contracts:
//!  - the co-sited / unspecified group stays **byte-identical** to the pre-siting
//!    resample (phase 0), every output (α included);
//!  - centered area / filter RGBA (α carried) == the RGB-domain
//!    reconstruct-then-bin oracle;
//!  - the centered α-drop decode (RGB / HSV / luma) is BIT-IDENTICAL to the
//!    no-alpha `Yuv422p` row-stage centered resample of the same Y/U/V;
//!  - the α channel is IDENTICAL between co-sited and centered (colour differs);
//!  - SIMD == scalar; centered != co-sited on a ramp / == on flat;
//!  - a mid-frame siting flip is rejected with `ChromaSitingChanged` on both tiers;
//!  - the centered reserve sits BEHIND the resample preflight (#180 atomicity).

use crate::{
  ChromaLocation, KernelMatrix, PixelSink,
  resample::{AreaResampler, Bicublin, FilteredResampler, ResampleError, Triangle},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Yuv422p, Yuva422p, Yuva422pRow, Yuva444p, yuv422p_to, yuva422p_to, yuva444p_to},
};
use mediaframe::frame::{Yuv422pFrame, Yuva422pFrame, Yuva444pFrame};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Independent #302 centered horizontal upsample (`1/4`–`3/4`, edge clamp,
/// round-half-up to `u8`) — the RGB-domain oracle's reconstruction step, matching
/// the production `upsample_420_chroma_center_h`.
fn recon_full_row_u8(c: &[u8], cw: usize) -> Vec<u8> {
  let mut out = vec![0u8; 2 * cw];
  for j in 0..cw {
    let l = u32::from(c[j.saturating_sub(1)]);
    let m = u32::from(c[j]);
    let r = u32::from(c[if j + 1 < cw { j + 1 } else { j }]);
    out[2 * j] = ((l + 3 * m + 2) >> 2) as u8;
    out[2 * j + 1] = ((3 * m + r + 2) >> 2) as u8;
  }
  out
}

/// The geometries the oracle-equality tests sweep: clean 2:1 and fractional
/// ratios. Widths are even (`Yuva422p` requires it).
const GEOMS: [(usize, usize, usize, usize); 4] =
  [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)];

/// `(rgb, rgba, (h,s,v), luma, luma_u16)` — the full output set every driver
/// returns so a single call feeds every assertion. The 8-bit `Yuva422p` exposes no
/// u16 colour outputs.
type Outs = (
  Vec<u8>,
  Vec<u8>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);
/// The no-alpha `Yuv422p` cross-check outputs `(rgb, (h,s,v), luma, luma_u16)` —
/// the α-drop channels the alpha-bearing decode must match.
type YuvOuts = (Vec<u8>, (Vec<u8>, Vec<u8>, Vec<u8>), Vec<u8>, Vec<u16>);

/// A `Yuva422p` fixture (`cw = sw / 2`, FULL-height chroma `cw x sh`) with a strong
/// HORIZONTAL chroma ramp (so the centered triangle genuinely differs from the
/// co-sited nearest decode) plus a per-row tilt (a vertical mistake would show)
/// and a varying, non-opaque alpha (so the α-preservation assertions are
/// non-vacuous).
fn ramp(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let mut y = vec![0u8; sw * sh];
  let mut u = vec![0u8; cw * sh];
  let mut v = vec![0u8; cw * sh];
  let mut a = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 3) % 160) as u8;
  }
  for r in 0..sh {
    for c in 0..cw {
      u[r * cw + c] = (30 + c * 44 + r * 4).min(240) as u8;
      v[r * cw + c] = (230u32.saturating_sub((c * 44 + r * 4) as u32)).max(16) as u8;
    }
  }
  for (i, p) in a.iter_mut().enumerate() {
    *p = 20 + ((i as u32 * 11) % 220) as u8;
  }
  (y, u, v, a)
}

/// Flat chroma: the centered triangle of a constant is that constant, so centered
/// must equal co-sited. Luma and α still vary.
fn flat(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let mut y = vec![0u8; sw * sh];
  let mut a = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 7) % 170) as u8;
  }
  for (i, p) in a.iter_mut().enumerate() {
    *p = 30 + ((i as u32 * 13) % 200) as u8;
  }
  (y, vec![110u8; cw * sh], vec![140u8; cw * sh], a)
}

/// Drive a `Yuva422p` STRAIGHT-alpha AREA resample for the full output set.
#[allow(clippy::too_many_arguments)]
fn run(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  simd: bool,
) -> Outs {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuva422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_alpha_mode(AlphaMode::Straight)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma16)
        .unwrap();
    let f = Yuva422pFrame::try_new(
      y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
    )
    .unwrap();
    yuva422p_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma16)
}

/// Drive a `Yuva422p` single-kernel filter resample (Triangle) for the full set.
#[allow(clippy::too_many_arguments)]
fn run_filter(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  simd: bool,
) -> Outs {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma16 = vec![0u16; ow * oh];
  {
    let mut sink = MixedSinker::<Yuva422p, FilteredResampler<Triangle>>::with_resampler(
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
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma16)
    .unwrap();
    let f = Yuva422pFrame::try_new(
      y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
    )
    .unwrap();
    yuva422p_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma16)
}

/// The centered RGB-domain oracle for the packed area / filter tail: reconstruct
/// U / V to full width with the #302 kernel (each luma row `r` taking chroma row
/// `r` — 4:2:2 chroma is full-height) then run that full-resolution `Yuva444p`
/// frame (with the UNTOUCHED α plane) through the given resampler with STRAIGHT
/// alpha. Returns the RGBA — the exact convert-each-row-then-bin the routed path
/// does.
#[allow(clippy::too_many_arguments)]
fn rgba_oracle(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
  filter: bool,
) -> Vec<u8> {
  let cw = sw / 2;
  let mut uf = vec![0u8; sw * sh];
  let mut vf = vec![0u8; sw * sh];
  for r in 0..sh {
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_u8(&u[r * cw..r * cw + cw], cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_u8(&v[r * cw..r * cw + cw], cw));
  }
  let mut rgba = vec![0u8; ow * oh * 4];
  let f = Yuva444pFrame::try_new(
    y, &uf, &vf, a, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32, sw as u32,
  )
  .unwrap();
  if filter {
    let mut sink = MixedSinker::<Yuva444p, FilteredResampler<Triangle>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, Triangle),
    )
    .unwrap()
    .with_alpha_mode(AlphaMode::Straight)
    .with_simd(simd)
    .with_rgba(&mut rgba)
    .unwrap();
    yuva444p_to(&f, FR, M, &mut sink).unwrap();
  } else {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_alpha_mode(AlphaMode::Straight)
        .with_simd(simd)
        .with_rgba(&mut rgba)
        .unwrap();
    yuva444p_to(&f, FR, M, &mut sink).unwrap();
  }
  rgba
}

/// Drive the no-alpha `Yuv422p` resample on the SAME Y / U / V through the
/// ROW-STAGE tier (`with_native(false)`, RGB-domain — the tier the packed YUVA
/// path mirrors). RGB is attached so HSV derives from the same binned RGB the
/// α-drop YUVA HSV does, making the HSV comparison valid.
#[allow(clippy::too_many_arguments)]
fn run_yuv(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  simd: bool,
) -> YuvOuts {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma16)
        .unwrap();
    let f = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv422p_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, (hh, ss, vv), luma, luma16)
}

// ---- co-sited byte-identity (the regression contract) ----------------------

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
        "co-sited {loc:?} area must keep the byte-identical decode ({sw}x{sh}->{ow}x{oh})"
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

// ---- centered area / filter RGBA == RGB-domain reconstruct-then-bin ---------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_area_rgba_equals_rgb_domain_oracle() {
  for (sw, sh, ow, oh) in GEOMS {
    let (y, u, v, a) = ramp(sw, sh);
    let want = rgba_oracle(&y, &u, &v, &a, sw, sh, ow, oh, true, false);
    for loc in [ChromaLocation::Center, ChromaLocation::Top] {
      let got = run(&y, &u, &v, &a, sw, sh, ow, oh, loc.clone(), true);
      assert_eq!(
        got.1, want,
        "centered area rgba {loc:?} ({sw}x{sh}->{ow}x{oh})"
      );
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_filter_rgba_equals_rgb_domain_oracle() {
  for (sw, sh, ow, oh) in GEOMS {
    let (y, u, v, a) = ramp(sw, sh);
    let want = rgba_oracle(&y, &u, &v, &a, sw, sh, ow, oh, true, true);
    let got = run_filter(&y, &u, &v, &a, sw, sh, ow, oh, ChromaLocation::Center, true);
    assert_eq!(got.1, want, "centered filter rgba ({sw}x{sh}->{ow}x{oh})");
  }
}

// ---- STRONG cross-check: centered α-drop decode == Yuv422p centered ---------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_chroma_matches_yuv422p_centered() {
  // α is orthogonal to chroma siting, so the centered `Yuva422p` Y/U/V → RGB (and
  // HSV / luma) decode must be BIT-IDENTICAL to the no-alpha `Yuv422p` ROW-STAGE
  // centered resample of the same Y/U/V (the packed YUVA tail is RGB-domain, so it
  // matches the row-stage tier, α-drop). `luma_u16` is siting-independent native Y
  // and is pinned by the co-sited / SIMD suites.
  for (sw, sh, ow, oh) in GEOMS {
    let (y, u, v, a) = ramp(sw, sh);
    for loc in [ChromaLocation::Center, ChromaLocation::Top] {
      let ya = run(&y, &u, &v, &a, sw, sh, ow, oh, loc.clone(), true);
      let yv = run_yuv(&y, &u, &v, sw, sh, ow, oh, loc.clone(), true);
      assert_eq!(ya.0, yv.0, "rgb {loc:?} ({sw}x{sh}->{ow}x{oh})");
      assert_eq!(ya.2, yv.1, "hsv {loc:?} ({sw}x{sh}->{ow}x{oh})");
      assert_eq!(ya.3, yv.2, "luma {loc:?} ({sw}x{sh}->{ow}x{oh})");
      // The RGBA colour channels equal the same centered RGB.
      for px in 0..ow * oh {
        assert_eq!(
          &ya.1[px * 4..px * 4 + 3],
          &ya.0[px * 3..px * 3 + 3],
          "rgba colour vs rgb {loc:?} px {px}"
        );
      }
    }
  }
}

// ---- alpha preservation: siting never touches α ----------------------------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn alpha_is_identical_between_cosited_and_centered() {
  // Chroma siting must not touch the full-resolution α plane: the centered RGBA's
  // alpha channel equals the co-sited path's alpha byte-for-byte on both tiers —
  // while the colour channels DO differ.
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
      "centered α must equal co-sited (filter={is_filter})"
    );
    assert_ne!(
      cen.0, cos.0,
      "centered colour must differ (filter={is_filter})"
    );
  }
}

// ---- non-vacuous + flat-chroma sanity --------------------------------------

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
    cos.1, cen.1,
    "centered rgba must differ from co-sited on a ramp"
  );
}

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_equals_cosited_on_flat_chroma() {
  let (y, u, v, a) = flat(8, 8);
  let cos = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Left, true);
  let cen = run(&y, &u, &v, &a, 8, 8, 4, 4, ChromaLocation::Center, true);
  assert_eq!(cos.1, cen.1, "flat-chroma rgba");
  assert_eq!(cos.2, cen.2, "flat-chroma hsv");
}

// ---- SIMD == scalar --------------------------------------------------------

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

// ---- mid-frame siting change is rejected across tiers ----------------------

/// Accept row 0 at `loc1` (freezes the phase), flip to `loc2`, feed the
/// IN-SEQUENCE row 1 (4:2:2 chroma row 1), and return its result.
fn flip_row1<R>(
  mut sink: MixedSinker<'_, Yuva422p, R>,
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> Result<(), MixedSinkerError> {
  sink.set_chroma_location(loc1.clone());
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  let row0 = Yuva422pRow::new(&y[0..8], &u[0..4], &v[0..4], &a[0..8], 0, M, FR);
  PixelSink::process(&mut sink, row0).unwrap();
  sink.set_chroma_location(loc2.clone());
  let row1 = Yuva422pRow::new(&y[8..16], &u[4..8], &v[4..8], &a[8..16], 1, M, FR);
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
      MixedSinker::<Yuva422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_alpha_mode(AlphaMode::Straight)
        .with_rgba(&mut rgba)
        .unwrap();
    let err = flip_row1(sink, &y, &u, &v, &a, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "area {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // HSV-only tier (rides the packed-YUVA RGBA path).
    let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
    let sink =
      MixedSinker::<Yuva422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_alpha_mode(AlphaMode::Straight)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let err = flip_row1(sink, &y, &u, &v, &a, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "hsv {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // Filter tier.
    let mut rgba = vec![0u8; 4 * 4 * 4];
    let sink = MixedSinker::<Yuva422p, FilteredResampler<Triangle>>::with_resampler(
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

// ---- atomicity: the centered reserve sits BEHIND the resample preflight -----

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn out_of_sequence_centered_first_row_is_rejected_before_the_chroma_reserve() {
  // The centered chroma reservation must run AFTER the resample preflight, so an
  // out-of-sequence FIRST row is rejected BEFORE any allocation (#180) — a primed
  // allocator refusal is never reached.
  let (y, u, v, a) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuva422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  super::super::arm_chroma_full_alloc_failure();
  // First process call is row 5 — the stream expects row 0.
  let bad = Yuva422pRow::new(
    &y[5 * 8..6 * 8],
    &u[5 * cw..6 * cw],
    &v[5 * cw..6 * cw],
    &a[5 * 8..6 * 8],
    5,
    M,
    FR,
  );
  let err = PixelSink::process(&mut sink, bad).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "out-of-sequence centered first row must be OutOfSequenceRow (reserve unreached), got {err:?}"
  );
  assert_eq!(
    sink.chroma_full.len(),
    0,
    "a rejected row must allocate no chroma scratch"
  );
  // Non-vacuous: the failpoint is still armed, so a VALID first row now REACHES the
  // reserve (proving the guard is ordering, not a disabled reserve).
  let good = Yuva422pRow::new(&y[0..8], &u[0..cw], &v[0..cw], &a[0..8], 0, M, FR);
  let err0 = PixelSink::process(&mut sink, good).unwrap_err();
  assert!(
    matches!(
      err0,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "a valid centered row reaches the reserve (failpoint fires), got {err0:?}"
  );
}

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn luma_only_centered_area_does_not_reserve_chroma() {
  // A luma-only centered area resample never calls the RGBA converter, so it must
  // NOT reserve/reconstruct chroma: with the failpoint armed it still succeeds.
  let (y, u, v, a) = ramp(8, 8);
  let cw = 4usize;
  let mut luma = vec![0u8; 4 * 4];
  {
    let mut sink =
      MixedSinker::<Yuva422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_chroma_location(ChromaLocation::Center)
        .with_luma(&mut luma)
        .unwrap();
    PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
    super::super::arm_chroma_full_alloc_failure();
    for r in 0..8 {
      let row = Yuva422pRow::new(
        &y[r * 8..r * 8 + 8],
        &u[r * cw..r * cw + cw],
        &v[r * cw..r * cw + cw],
        &a[r * 8..r * 8 + 8],
        r,
        M,
        FR,
      );
      PixelSink::process(&mut sink, row).unwrap();
    }
    assert_eq!(
      sink.chroma_full.len(),
      0,
      "luma-only centered resample must never reserve chroma scratch"
    );
  }
  // The luma-only path never reserved, so the failpoint is still armed; consume it
  // via a colour row so it does not leak into the next test.
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuva422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  let f = Yuva422pFrame::try_new(&y, &u, &v, &a, 8, 8, 8, cw as u32, cw as u32, 8).unwrap();
  let _ = yuva422p_to(&f, FR, M, &mut sink);
}

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_filter_bicublin_rejected_before_the_chroma_reserve() {
  // A BICUBLIN (multi-kernel) filter plan on a centered Yuva422p colour output must
  // be rejected as UnsupportedFilter BEFORE the centered chroma reserve —
  // `ensure_single_kernel_filter` is hoisted to the top of the filter arm.
  let (y, u, v, a) = ramp(8, 8);
  let cw = 4usize;
  let mut rgba = vec![0u8; 4 * 4 * 4];
  let mut sink = MixedSinker::<Yuva422p, Bicublin>::with_resampler(8, 8, Bicublin::to(4, 4))
    .unwrap()
    .with_alpha_mode(AlphaMode::Straight)
    .with_chroma_location(ChromaLocation::Center)
    .with_rgba(&mut rgba)
    .unwrap();
  let f = Yuva422pFrame::try_new(&y, &u, &v, &a, 8, 8, 8, cw as u32, cw as u32, 8).unwrap();
  super::super::arm_chroma_full_alloc_failure();
  let err = yuva422p_to(&f, FR, M, &mut sink).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
    ),
    "a BICUBLIN filter plan must be UnsupportedFilter (reserve unreached), got {err:?}"
  );
  assert_eq!(
    sink.chroma_full.len(),
    0,
    "a rejected filter plan allocates no chroma scratch"
  );
  // The Bicublin reject never reached the reserve, so the failpoint is still armed;
  // consume it via a centered colour area row so it does not leak into the next test.
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut consume =
    MixedSinker::<Yuva422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  let cf = Yuva422pFrame::try_new(&y, &u, &v, &a, 8, 8, 8, cw as u32, cw as u32, 8).unwrap();
  let _ = yuva422p_to(&cf, FR, M, &mut consume);
}
