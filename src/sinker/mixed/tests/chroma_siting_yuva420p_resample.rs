//! RFC #238 S3c — chroma-siting-aware 4:2:0 **resample** for `Yuva420p`, the
//! alpha-bearing twin of `chroma_siting_420_resample` (S3a, `Yuv420p`).
//!
//! `Yuva420p` is planar 4:2:0 YUV (half-width, half-height U & V planes) PLUS a
//! **full-resolution** straight alpha plane that is never subsampled. S3c routes
//! the HORIZONTAL centered siting (`Center` / `Top` / `Bottom`,
//! [`chroma_420_center_sited_h`](super::super::chroma_420_center_sited_h))
//! through the resample so a downscale keeps the correct horizontal chroma phase;
//! the VERTICAL chroma pairing stays co-sited (`v_phase = 0`, today's box
//! pairing). The α plane is orthogonal to chroma siting — it bins on the luma
//! grid unchanged on every path, so the effective siting touches ONLY the Y/U/V
//! colour decode.
//!
//! The correctness contracts, per tier (native fast tier + encoded row-stage):
//!  - the co-sited / unspecified group stays **byte-identical** to the pre-siting
//!    resample (phase 0 → the folded plan is never built);
//!  - the centered **Y/U/V → RGB** (and luma / HSV) output is **bit-identical to
//!    the equivalent no-alpha `Yuv420p` centered resample** — the strong oracle,
//!    since α is orthogonal and the chroma decode must match `Yuv420p` exactly;
//!  - the centered native **RGBA** equals the independent bin-then-convert oracle
//!    (Y / A co-sited, U / V through the exact centered chroma oracle, then a
//!    single `Yuva444p` convert with the binned α straight);
//!  - the **alpha channel is IDENTICAL** between the co-sited and centered
//!    decodes (chroma siting must never touch α);
//!  - an in-sequence mid-frame phase change is rejected with
//!    [`ChromaSitingChanged`](super::super::MixedSinkerError::ChromaSitingChanged)
//!    across every tier; the native join rebuilds on a cross-frame phase change;
//!  - the centered reserve sits BEHIND the resample preflight (#180 atomicity).

use crate::{
  ChromaLocation, ColorMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Yuv420p, Yuva420p, Yuva420pRow, Yuva444p, yuv420p_to, yuva420p_to, yuva444p_to},
};
use mediaframe::frame::{Yuv420pFrame, Yuva420pFrame, Yuva444pFrame};

const M: ColorMatrix = ColorMatrix::Bt601;
const FR: bool = true;

/// Round `a / d` half-up — the production `round_div_half_up`, replicated so the
/// oracle is independent.
fn rdhu(a: u64, d: u64) -> u64 {
  let q = a / d;
  let r = a % d;
  q + u64::from(r >= d - d / 2)
}

/// Exact box-overlap area weights for `src -> out`, mirroring
/// `resample::AxisSpans::area`. Returns per output `(first source cell, overlaps)`.
fn area_weights(src: usize, out: usize) -> Vec<(usize, Vec<u64>)> {
  let (src64, out64) = (src as u64, out as u64);
  (0..out)
    .map(|o| {
      let lo = o as u64 * src64;
      let hi = lo + src64;
      let start = (lo / out64) as usize;
      let mut w = Vec::new();
      let mut i = start as u64;
      loop {
        let clo = i * out64;
        if clo >= hi {
          break;
        }
        let chi = clo + out64;
        let ov = chi.min(hi) - clo.max(lo);
        if ov == 0 {
          break;
        }
        w.push(ov);
        if chi >= hi {
          break;
        }
        i += 1;
      }
      (start, w)
    })
    .collect()
}

/// Co-sited box-average of a full-resolution `sw x sh` u8 plane to `ow x oh`
/// (round-half-up) — the reference for a phase-free plane (luma AND the
/// full-resolution alpha plane, both siting-independent).
fn bin_cosited(plane: &[u8], sw: usize, sh: usize, ow: usize, oh: usize) -> Vec<u8> {
  let hw = area_weights(sw, ow);
  let vw = area_weights(sh, oh);
  let denom = (sw * sh) as u64;
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * u64::from(plane[(vs + dy) * sw + hs + dx]);
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// The EXACT centered chroma oracle for the native tier: reconstruct the
/// `cw x ch` chroma to full width with the #302 `1/4`–`3/4` triangle kept
/// UNROUNDED (scaled ×4 to stay integral), then box-average to `ow x oh` —
/// HORIZONTAL over `2·cw`, VERTICAL over the `ch` chroma rows (co-sited) — with a
/// SINGLE round-half-up over `4·(2·cw)·ch`. Identical to the `Yuv420p` S3a
/// oracle (α is orthogonal, so the chroma oracle is unchanged); valid for EVEN
/// `sh`.
fn bin_chroma_centered(c: &[u8], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u8> {
  let full = 2 * cw;
  let mut r4 = vec![0u32; full * ch];
  for r in 0..ch {
    let row = &c[r * cw..r * cw + cw];
    for j in 0..cw {
      let l = u32::from(row[j.saturating_sub(1)]);
      let m = u32::from(row[j]);
      let rt = u32::from(row[if j + 1 < cw { j + 1 } else { j }]);
      r4[r * full + 2 * j] = l + 3 * m;
      r4[r * full + 2 * j + 1] = 3 * m + rt;
    }
  }
  let hw = area_weights(full, ow);
  let vw = area_weights(ch, oh);
  let denom = (4 * full * ch) as u64;
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * u64::from(r4[(vs + dy) * full + hs + dx]);
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// A `Yuva420p` fixture (`cw = sw / 2`, `ch = sh / 2`) with a strong HORIZONTAL
/// chroma ramp (so the centered triangle genuinely differs from the co-sited
/// nearest decode) plus a per-row tilt (a vertical mistake would show) and a
/// varying, non-opaque alpha (so the alpha-preservation assertions are
/// non-vacuous). `sw` / `sh` must be even.
fn ramp(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let ch = sh / 2;
  let mut y = vec![0u8; sw * sh];
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  let mut a = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 3) % 160) as u8;
  }
  for r in 0..ch {
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

/// A flat-chroma fixture: the centered phase is a no-op on constant chroma, so
/// centered must equal co-sited. Luma and alpha still vary.
fn flat_chroma(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let ch = sh / 2;
  let mut y = vec![0u8; sw * sh];
  let mut a = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 7) % 170) as u8;
  }
  for (i, p) in a.iter_mut().enumerate() {
    *p = 30 + ((i as u32 * 13) % 200) as u8;
  }
  (y, vec![110u8; cw * ch], vec![140u8; cw * ch], a)
}

type YuvaOuts = (
  Vec<u8>,
  Vec<u8>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);
type YuvOuts = (Vec<u8>, (Vec<u8>, Vec<u8>, Vec<u8>), Vec<u8>, Vec<u16>);

/// Drive a `Yuva420p` STRAIGHT-alpha area resample (`sw x sh -> ow x oh`) for the
/// full output set, at `loc` siting and `native` tier.
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
  native: bool,
  simd: bool,
) -> YuvaOuts {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(native)
        .with_alpha_mode(AlphaMode::Straight)
        .with_chroma_location(loc)
        .with_simd(simd)
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
    let f = Yuva420pFrame::try_new(
      y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
    )
    .unwrap();
    yuva420p_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// Drive the no-alpha `Yuv420p` resample on the SAME Y/U/V — the cross-check
/// reference for the chroma decode (RGB / HSV / luma). RGB is attached so
/// `Yuv420p`'s HSV derives from the SAME binned RGB the alpha-drop `Yuva420p`
/// HSV does, making the HSV comparison valid on both tiers.
#[allow(clippy::too_many_arguments)]
fn run_yuv420p(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  native: bool,
  simd: bool,
) -> YuvOuts {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(native)
        .with_chroma_location(loc)
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, (hh, ss, vv), luma, luma_u16)
}

/// The centered NATIVE code-domain oracle: bin Y and A co-sited (both
/// full-resolution), bin U / V through the exact centered chroma oracle, then
/// convert ONCE at output width via an identity `Yuva444p` sink with the binned
/// α straight — the exact ground truth the straight-alpha native tier reproduces
/// byte-for-byte (EVEN `sh` only).
#[allow(clippy::too_many_arguments)]
fn native_rgba_oracle(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Vec<u8> {
  let cw = sw / 2;
  let ch = sh / 2;
  let yb = bin_cosited(y, sw, sh, ow, oh);
  let ub = bin_chroma_centered(u, cw, ch, ow, oh);
  let vb = bin_chroma_centered(v, cw, ch, ow, oh);
  let ab = bin_cosited(a, sw, sh, ow, oh);
  let mut rgba = vec![0u8; ow * oh * 4];
  {
    let mut sink = MixedSinker::<Yuva444p>::new(ow, oh)
      .with_alpha_mode(AlphaMode::Straight)
      .with_simd(simd)
      .with_rgba(&mut rgba)
      .unwrap();
    let f = Yuva444pFrame::try_new(
      &yb, &ub, &vb, &ab, ow as u32, oh as u32, ow as u32, ow as u32, ow as u32, ow as u32,
    )
    .unwrap();
    yuva444p_to(&f, FR, M, &mut sink).unwrap();
  }
  rgba
}

// ---- co-sited byte-identity (the regression contract) ----------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn cosited_group_is_byte_identical_across_tiers() {
  // Every co-sited / unspecified siting must produce the byte-identical
  // pre-siting resample (phase 0 → the folded plan is never built), on BOTH
  // tiers. `Unspecified` is the baseline; every output (incl. the α channel).
  let (y, u, v, a) = ramp(8, 8);
  for native in [true, false] {
    let base = run(
      &y,
      &u,
      &v,
      &a,
      8,
      8,
      4,
      4,
      ChromaLocation::Unspecified,
      native,
      true,
    );
    for loc in [
      ChromaLocation::Left,
      ChromaLocation::TopLeft,
      ChromaLocation::BottomLeft,
      ChromaLocation::Unknown(7),
    ] {
      let got = run(&y, &u, &v, &a, 8, 8, 4, 4, loc, native, true);
      assert_eq!(got.0, base.0, "rgb {loc:?} native={native}");
      assert_eq!(got.1, base.1, "rgba {loc:?} native={native}");
      assert_eq!(got.2, base.2, "hsv {loc:?} native={native}");
      assert_eq!(got.3, base.3, "luma {loc:?} native={native}");
      assert_eq!(got.4, base.4, "luma_u16 {loc:?} native={native}");
    }
  }
}

// ---- the STRONG oracle: centered chroma decode == Yuv420p centered ----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_chroma_matches_yuv420p_centered_across_tiers() {
  // α is orthogonal to chroma siting, so the centered `Yuva420p` Y/U/V → RGB
  // (and luma / HSV) decode must be BIT-IDENTICAL to the no-alpha `Yuv420p`
  // centered resample of the same Y/U/V — on BOTH tiers (the native tier embeds
  // the same 4:2:0 join; the row-stage tier bins the same RGB, α-drop).
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
    let (y, u, v, a) = ramp(sw, sh);
    for native in [true, false] {
      // Center / Top keep the vertical pairing co-sited, so Yuva420p == Yuv420p.
      // Bottom is excluded here: Yuv420p now folds its vertical v=1 phase through
      // the resample while Yuva420p still co-sites it, so the two diverge on
      // Bottom until the Yuva420p vertical fold lands.
      for loc in [ChromaLocation::Center, ChromaLocation::Top] {
        let ya = run(&y, &u, &v, &a, sw, sh, ow, oh, loc, native, true);
        let yv = run_yuv420p(&y, &u, &v, sw, sh, ow, oh, loc, native, true);
        assert_eq!(
          ya.0, yv.0,
          "rgb {loc:?} native={native} {sw}x{sh}->{ow}x{oh}"
        );
        assert_eq!(
          ya.2, yv.1,
          "hsv {loc:?} native={native} {sw}x{sh}->{ow}x{oh}"
        );
        assert_eq!(
          ya.3, yv.2,
          "luma {loc:?} native={native} {sw}x{sh}->{ow}x{oh}"
        );
        assert_eq!(
          ya.4, yv.3,
          "luma_u16 {loc:?} native={native} {sw}x{sh}->{ow}x{oh}"
        );
        // The RGBA colour channels equal the same centered RGB.
        for px in 0..ow * oh {
          assert_eq!(
            &ya.1[px * 4..px * 4 + 3],
            &ya.0[px * 3..px * 3 + 3],
            "rgba colour {loc:?} native={native} px {px}"
          );
        }
      }
    }
  }
}

// ---- centered native RGBA == the exact code-domain oracle (α included) ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_native_rgba_equals_code_domain_oracle() {
  // The straight-alpha native tier IS bin-Y/U/V/A-then-convert: Y / A co-sited,
  // U / V through the exact centered chroma oracle, one `Yuva444p` convert.
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
    let (y, u, v, a) = ramp(sw, sh);
    let oracle = native_rgba_oracle(&y, &u, &v, &a, sw, sh, ow, oh, true);
    for loc in [
      ChromaLocation::Center,
      ChromaLocation::Top,
      ChromaLocation::Bottom,
    ] {
      let n = run(&y, &u, &v, &a, sw, sh, ow, oh, loc, true, true);
      assert_eq!(n.1, oracle, "native rgba {loc:?} {sw}x{sh}->{ow}x{oh}");
    }
  }
}

// ---- alpha preservation: siting never touches α ----------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn alpha_is_identical_between_cosited_and_centered() {
  // Chroma siting must not touch the full-resolution α plane: the centered
  // RGBA's alpha channel equals the co-sited path's alpha byte-for-byte, on
  // BOTH tiers — while the colour channels DO differ (the non-vacuous control).
  let (y, u, v, a) = ramp(8, 8);
  for native in [true, false] {
    let cos = run(
      &y,
      &u,
      &v,
      &a,
      8,
      8,
      4,
      4,
      ChromaLocation::Left,
      native,
      true,
    );
    let cen = run(
      &y,
      &u,
      &v,
      &a,
      8,
      8,
      4,
      4,
      ChromaLocation::Center,
      native,
      true,
    );
    let cos_a: Vec<u8> = cos.1.iter().skip(3).step_by(4).copied().collect();
    let cen_a: Vec<u8> = cen.1.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(
      cen_a, cos_a,
      "centered alpha must equal co-sited alpha (native={native})"
    );
    assert_ne!(
      cen.0, cos.0,
      "centered colour must differ from co-sited (native={native})"
    );
  }
}

// ---- non-vacuous + flat-chroma sanity --------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_differs_from_cosited_on_a_chroma_ramp() {
  let (y, u, v, a) = ramp(8, 8);
  for native in [true, false] {
    let cos = run(
      &y,
      &u,
      &v,
      &a,
      8,
      8,
      4,
      4,
      ChromaLocation::Unspecified,
      native,
      true,
    );
    let cen = run(
      &y,
      &u,
      &v,
      &a,
      8,
      8,
      4,
      4,
      ChromaLocation::Center,
      native,
      true,
    );
    assert_ne!(
      cen.0, cos.0,
      "centered rgb must differ from co-sited (native={native})"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_equals_cosited_on_flat_chroma() {
  // On constant chroma the centered triangle is a no-op, so centered and
  // co-sited agree byte-for-byte (the phase machinery corrupts nothing; α is a
  // varying plane, so its byte-identity is a real check too).
  let (y, u, v, a) = flat_chroma(8, 8);
  for native in [true, false] {
    let cos = run(
      &y,
      &u,
      &v,
      &a,
      8,
      8,
      4,
      4,
      ChromaLocation::Left,
      native,
      true,
    );
    let cen = run(
      &y,
      &u,
      &v,
      &a,
      8,
      8,
      4,
      4,
      ChromaLocation::Center,
      native,
      true,
    );
    assert_eq!(cen.0, cos.0, "flat-chroma rgb (native={native})");
    assert_eq!(cen.1, cos.1, "flat-chroma rgba (native={native})");
    assert_eq!(cen.2, cos.2, "flat-chroma hsv (native={native})");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_native_simd_matches_scalar() {
  // Weights are precomputed integers, so the SIMD H/V passes must be 0-ULP.
  let (y, u, v, a) = ramp(8, 8);
  let s = run(
    &y,
    &u,
    &v,
    &a,
    8,
    8,
    4,
    4,
    ChromaLocation::Center,
    true,
    false,
  );
  let d = run(
    &y,
    &u,
    &v,
    &a,
    8,
    8,
    4,
    4,
    ChromaLocation::Center,
    true,
    true,
  );
  assert_eq!(s.0, d.0, "rgb scalar vs simd");
  assert_eq!(s.1, d.1, "rgba scalar vs simd");
  assert_eq!(s.2, d.2, "hsv scalar vs simd");
  assert_eq!(s.3, d.3, "luma scalar vs simd");
}

// ---- cross-frame native join rebuilds on a siting change (RFC #238) ---------

/// Reuse ONE full-output native-tier sink across two frames of the SAME content,
/// siting `loc1` then `loc2`, returning frame 2's outputs.
#[allow(clippy::too_many_arguments)]
fn run_reuse_native(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> YuvaOuts {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(true)
        .with_alpha_mode(AlphaMode::Straight)
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
    let f = Yuva420pFrame::try_new(
      y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
    )
    .unwrap();
    sink.set_chroma_location(loc1);
    yuva420p_to(&f, FR, M, &mut sink).unwrap();
    sink.set_chroma_location(loc2);
    yuva420p_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_join_rebuilds_on_siting_change_across_frames() {
  // Reuse one native-tier sink flipping Left ⇄ Center (both directions): frame 2
  // must match a FRESH sink for frame 2's siting — no stale-phase carryover.
  let (y, u, v, a) = ramp(8, 8);
  for (p, q) in [
    (ChromaLocation::Left, ChromaLocation::Center),
    (ChromaLocation::Center, ChromaLocation::Left),
  ] {
    let reused = run_reuse_native(&y, &u, &v, &a, 8, 8, 4, 4, p, q);
    let fresh = run(&y, &u, &v, &a, 8, 8, 4, 4, q, true, true);
    assert_eq!(reused.0, fresh.0, "native rgb {p:?}->{q:?} stale carryover");
    assert_eq!(reused.1, fresh.1, "native rgba {p:?}->{q:?}");
    assert_eq!(reused.2, fresh.2, "native hsv {p:?}->{q:?}");
    let stale = run(&y, &u, &v, &a, 8, 8, 4, 4, p, true, true);
    assert_ne!(fresh.0, stale.0, "sitings {p:?} vs {q:?} must differ");
  }
}

// ---- IN-SEQUENCE mid-frame phase change is rejected across tiers ------------

/// Drive one `Yuva420p` resample frame: accept row 0 at `loc1` (freezes the
/// phase), flip to `loc2`, then feed the IN-SEQUENCE row 1 and return its
/// `process` result.
fn in_sequence_flip_row1<R>(
  mut sink: MixedSinker<'_, Yuva420p, R>,
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> Result<(), MixedSinkerError> {
  let cw = 4usize;
  sink.set_chroma_location(loc1);
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  let row0 = Yuva420pRow::new(&y[0..8], &u[0..cw], &v[0..cw], &a[0..8], 0, M, FR);
  PixelSink::process(&mut sink, row0).unwrap();
  sink.set_chroma_location(loc2);
  let row1 = Yuva420pRow::new(&y[8..16], &u[0..cw], &v[0..cw], &a[8..16], 1, M, FR);
  PixelSink::process(&mut sink, row1)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn in_sequence_mid_frame_phase_change_rejected_across_tiers() {
  let (y, u, v, a) = ramp(8, 8);
  // Both flip directions: Center->Left (drop the phase) and Left->Center (add
  // it). Each must reject the in-sequence row 1 with ChromaSitingChanged.
  for (loc1, loc2) in [
    (ChromaLocation::Center, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Center),
  ] {
    // Native fast tier.
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "native {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // Encoded row-stage tier (`with_native(false)`).
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "encoded {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // HSV-only row-stage tier (rides the packed-YUVA RGBA path).
    let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
    let sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(false)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "hsv {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // Filter tier (single-kernel Triangle FilteredResampler).
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink = MixedSinker::<
      Yuva420p,
      crate::resample::FilteredResampler<crate::resample::Triangle>,
    >::with_resampler(
      8,
      8,
      crate::resample::FilteredResampler::new(4, 4, crate::resample::Triangle),
    )
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, &a, loc1, loc2).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
  }
}

// ---- atomicity: the centered reserve sits BEHIND the resample preflight -----

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn out_of_sequence_centered_first_row_is_rejected_before_the_chroma_reserve() {
  // The centered chroma reservation must run AFTER the resample preflight, so an
  // out-of-sequence FIRST row is rejected BEFORE any allocation (#180) — a primed
  // allocator refusal is never reached. `with_native(false)` forces the encoded
  // convert path.
  let (y, u, v, a) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  super::super::arm_chroma_full_alloc_failure();
  // First process call is row 5 — the stream expects row 0.
  let bad = Yuva420pRow::new(
    &y[5 * 8..6 * 8],
    &u[2 * cw..3 * cw],
    &v[2 * cw..3 * cw],
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
  // Non-vacuous: the failpoint is still armed, so a VALID first row now REACHES
  // the reserve (proving the guard is ordering, not a disabled reserve).
  let good = Yuva420pRow::new(&y[0..8], &u[0..cw], &v[0..cw], &a[0..8], 0, M, FR);
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
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_centered_area_does_not_reserve_chroma() {
  // A luma-only centered area resample never calls the RGBA converter, so it must
  // NOT reserve/reconstruct chroma: with the failpoint armed it still succeeds
  // (an unfixed path would reserve and mask the luma output).
  let (y, u, v, a) = ramp(8, 8);
  let cw = 4usize;
  let mut luma = vec![0u8; 4 * 4];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(false)
        .with_chroma_location(ChromaLocation::Center)
        .with_luma(&mut luma)
        .unwrap();
    PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
    super::super::arm_chroma_full_alloc_failure();
    for r in 0..8 {
      let cr = r / 2;
      let row = Yuva420pRow::new(
        &y[r * 8..r * 8 + 8],
        &u[cr * cw..cr * cw + cw],
        &v[cr * cw..cr * cw + cw],
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
  // The luma-only path never reserved, so the failpoint is still armed; consume
  // it via a colour row so it does not leak into the next test.
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  let f = Yuva420pFrame::try_new(&y, &u, &v, &a, 8, 8, 8, cw as u32, cw as u32, 8).unwrap();
  let _ = yuva420p_to(&f, FR, M, &mut sink);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_filter_bicublin_rejected_before_the_chroma_reserve() {
  use crate::resample::Bicublin;
  // A BICUBLIN (multi-kernel) filter plan on a centered Yuva420p colour output
  // must be rejected as UnsupportedFilter BEFORE the centered chroma reserve —
  // `ensure_single_kernel_filter` is hoisted to the top of the filter arm. With
  // the chroma-full failpoint armed, an unfixed reserve-first path would surface
  // the wrong AllocationFailed instead, and would mutate `chroma_full`.
  let (y, u, v, a) = ramp(8, 8);
  let cw = 4usize;
  let mut rgba = vec![0u8; 4 * 4 * 4];
  let mut sink = MixedSinker::<Yuva420p, Bicublin>::with_resampler(8, 8, Bicublin::to(4, 4))
    .unwrap()
    .with_alpha_mode(AlphaMode::Straight)
    .with_chroma_location(ChromaLocation::Center)
    .with_rgba(&mut rgba)
    .unwrap();
  let f = Yuva420pFrame::try_new(&y, &u, &v, &a, 8, 8, 8, cw as u32, cw as u32, 8).unwrap();
  super::super::arm_chroma_full_alloc_failure();
  let err = yuva420p_to(&f, FR, M, &mut sink).unwrap_err();
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
  // The Bicublin reject never reached the reserve, so the failpoint is still
  // armed; consume it via a centered colour area row so it does not leak into
  // the next test (it fires there as the expected one-shot AllocationFailed).
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut consume =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  let cf = Yuva420pFrame::try_new(&y, &u, &v, &a, 8, 8, 8, cw as u32, cw as u32, 8).unwrap();
  let _ = yuva420p_to(&cf, FR, M, &mut consume);
}
