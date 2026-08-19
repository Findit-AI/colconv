//! RFC #238 S1 — chroma-siting-aware 4:2:2 **resample** for `Yuv422p`.
//!
//! 4:2:2 subsamples chroma 2:1 horizontally only. Routing the centered siting
//! (`Center` / `Top` / `Bottom`, [`chroma_422_center_sited_h`]) through the
//! resample makes a downscale keep the correct chroma phase:
//!  - the **native fast tier** folds the #302 `1/4`–`3/4` triangle into the
//!    chroma area weights ([`ResamplePlan::area_chroma_422`]) — one SINGLE-
//!    rounding phased box-average on the subsampled grid;
//!  - the **encoded row-stage tier** (`with_native(false)`) reconstructs
//!    full-width chroma then bins in RGB.
//!
//! The co-sited / unspecified group stays phase 0, byte-identical to the
//! pre-siting resample (the folded form at phase 0 = the plain box overlaps).
//!
//! ★ Oracle (native tier): the EXACT code-domain box-average of the UNROUNDED
//! triangle-reconstructed chroma — a SINGLE rounding. This is the
//! user-approved, more-correct form; it differs from the #302
//! reconstruct-to-u8-then-bin (TWO roundings) by ≤ 1 LSB. The native path is
//! pinned against THIS (a YUV-domain oracle), never the RGB-domain oracle
//! (which would prove the wrong averaging domain). The encoded row-stage tier
//! IS the RGB-domain reconstruct-then-bin and is pinned against that.

use crate::{
  ChromaLocation, ColorInfo, ColorSpec, DynamicRange, KernelMatrix, PixelFormat, PixelSink,
  Primaries, Transfer,
  resample::{AreaResampler, AveragingDomain, LinearMode},
  sinker::MixedSinker,
  source::{Yuv422p, Yuv422pRow, Yuv444p, yuv422p_to, yuv444p_to},
};
use mediaframe::frame::{Yuv422pFrame, Yuv444pFrame};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Round `a / d` half-up (ties toward `+∞`) — the production
/// `round_div_half_up`, replicated here so the oracle is independent.
fn rdhu(a: u64, d: u64) -> u64 {
  let q = a / d;
  let r = a % d;
  q + u64::from(r >= d - d / 2)
}

/// Exact box-overlap area weights for `src -> out`, mirroring
/// `resample::AxisSpans::area`: output cell `o` covers `[o·src, (o+1)·src)` in
/// `(src·out)` units, source cell `i` covers `[i·out, (i+1)·out)`; the weight
/// is their overlap. Returns per output `(first source cell, overlaps)`.
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

/// Co-sited box-average of a full-resolution `sw x sh` u8 plane to
/// `ow x oh` (round-half-up) — the reference for a phase-free plane (luma, and
/// the co-sited chroma).
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

/// The EXACT centered chroma oracle: reconstruct the half-width `cw x ch`
/// chroma to full width with the #302 `1/4`–`3/4` triangle kept UNROUNDED
/// (scaled ×4 to stay integral: `r ∈ {1, 3, 4}`), then box-average to
/// `ow x oh` with a SINGLE round-half-up over `4·(2·cw)·ch`. This is the
/// code-domain twin the folded [`ResamplePlan::area_chroma_422`] weights
/// realize.
fn bin_chroma_centered(c: &[u8], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u8> {
  let full = 2 * cw;
  // ×4 reconstruction plane (`full x ch`), independent of the production kernel.
  let mut r4 = vec![0u32; full * ch];
  for r in 0..ch {
    let row = &c[r * cw..r * cw + cw];
    for j in 0..cw {
      let l = u32::from(row[j.saturating_sub(1)]);
      let m = u32::from(row[j]);
      let rt = u32::from(row[if j + 1 < cw { j + 1 } else { j }]);
      r4[r * full + 2 * j] = l + 3 * m; // even col: (c[j-1] + 3·c[j])
      r4[r * full + 2 * j + 1] = 3 * m + rt; // odd col: (3·c[j] + c[j+1])
    }
  }
  let hw = area_weights(full, ow);
  let vw = area_weights(ch, oh);
  let denom = (4 * full * ch) as u64; // ×4 triangle × the box normalization
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

/// Independent #302 centered horizontal upsample (`1/4`–`3/4`, edge clamp,
/// round-half-up to u8) — the RGB-domain oracle's reconstruction step.
fn recon_full_row(c: &[u8], cw: usize) -> Vec<u8> {
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

/// A `Yuv422p` fixture with a strong HORIZONTAL chroma ramp (so the centered
/// triangle, which pulls neighbours, genuinely differs from the co-sited
/// nearest decode) plus a per-row tilt (a vertical mistake would show).
fn ramp(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let mut y = vec![0u8; sw * sh];
  let mut u = vec![0u8; cw * sh];
  let mut v = vec![0u8; cw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 3) % 160) as u8;
  }
  for r in 0..sh {
    for c in 0..cw {
      u[r * cw + c] = (30 + c * 44 + r * 4).min(240) as u8;
      v[r * cw + c] = (230u32.saturating_sub((c * 44 + r * 4) as u32)).max(16) as u8;
    }
  }
  (y, u, v)
}

/// A flat-chroma fixture: the centered phase is a no-op on constant chroma
/// (the triangle of a constant is that constant), so centered must equal
/// co-sited. Luma still varies.
fn flat_chroma(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let mut y = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 7) % 170) as u8;
  }
  (y, vec![110u8; cw * sh], vec![140u8; cw * sh])
}

type Outs = (
  Vec<u8>,
  Vec<u8>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);

/// Drive a `Yuv422p` area resample (`sw x sh -> ow x oh`) for the full output
/// set, at `loc` siting and `native` tier.
#[allow(clippy::too_many_arguments)]
fn run(
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
) -> Outs {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(native)
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
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    let f = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The centered NATIVE oracle: bin Y co-sited and U / V through the exact
/// centered chroma oracle to `ow x oh`, then convert ONCE at output width via
/// an identity `Yuv444p` sink — the exact ground truth the native tier
/// reproduces byte-for-byte.
#[allow(clippy::too_many_arguments)]
fn native_oracle(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Outs {
  let cw = sw / 2;
  let yb = bin_cosited(y, sw, sh, ow, oh);
  let ub = bin_chroma_centered(u, cw, sh, ow, oh);
  let vb = bin_chroma_centered(v, cw, sh, ow, oh);
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink = MixedSinker::<Yuv444p>::new(ow, oh)
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
    let f = Yuv444pFrame::new(
      &yb, &ub, &vb, ow as u32, oh as u32, ow as u32, ow as u32, ow as u32,
    );
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The centered ENCODED row-stage oracle: reconstruct U / V to full width with
/// the #302 kernel (u8), then run that `Yuv444p` frame through a `with_native
/// (false)` RGB-domain resample — i.e. convert-each-row-then-bin-RGB, exactly
/// what the `Yuv422p` encoded arm does.
#[allow(clippy::too_many_arguments)]
fn encoded_oracle_rgb(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Vec<u8> {
  let cw = sw / 2;
  let mut uf = vec![0u8; sw * sh];
  let mut vf = vec![0u8; sw * sh];
  for r in 0..sh {
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&u[r * cw..r * cw + cw], cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&v[r * cw..r * cw + cw], cw));
  }
  let mut rgb = vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap();
    let f = Yuv444pFrame::new(
      y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
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
  // tiers. `Unspecified` is the baseline.
  let (y, u, v) = ramp(8, 8);
  for native in [true, false] {
    let base = run(
      &y,
      &u,
      &v,
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
      ChromaLocation::other("unassigned-7"),
    ] {
      let got = run(&y, &u, &v, 8, 8, 4, 4, loc.clone(), native, true);
      assert_eq!(got.0, base.0, "rgb {loc:?} native={native}");
      assert_eq!(got.1, base.1, "rgba {loc:?} native={native}");
      assert_eq!(got.2, base.2, "hsv {loc:?} native={native}");
      assert_eq!(got.3, base.3, "luma {loc:?} native={native}");
      assert_eq!(got.4, base.4, "luma_u16 {loc:?} native={native}");
    }
  }
}

// ---- centered native == the exact code-domain oracle -----------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_native_equals_code_domain_oracle() {
  // Clean 2:1 and a fractional ratio (both axes), for the whole centered group.
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 6, 4, 4)] {
    let (y, u, v) = ramp(sw, sh);
    let o = native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
    for loc in [
      ChromaLocation::Center,
      ChromaLocation::Top,
      ChromaLocation::Bottom,
    ] {
      let n = run(&y, &u, &v, sw, sh, ow, oh, loc.clone(), true, true);
      assert_eq!(n.0, o.0, "rgb {loc:?} {sw}x{sh}->{ow}x{oh}");
      assert_eq!(n.1, o.1, "rgba {loc:?} {sw}x{sh}->{ow}x{oh}");
      assert_eq!(n.2, o.2, "hsv {loc:?} {sw}x{sh}->{ow}x{oh}");
      assert_eq!(n.3, o.3, "luma {loc:?} {sw}x{sh}->{ow}x{oh}");
      assert_eq!(n.4, o.4, "luma_u16 {loc:?} {sw}x{sh}->{ow}x{oh}");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_native_simd_matches_scalar() {
  // Weights are precomputed integers, so the SIMD H/V passes must be 0-ULP.
  let (y, u, v) = ramp(8, 8);
  let s = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, true, false);
  let d = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, true, true);
  assert_eq!(s.0, d.0, "rgb scalar vs simd");
  assert_eq!(s.2, d.2, "hsv scalar vs simd");
  assert_eq!(s.3, d.3, "luma scalar vs simd");
}

// ---- centered encoded row-stage == RGB-domain reconstruct-then-bin ---------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_encoded_output_equals_rgb_reconstruct_then_bin() {
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3)] {
    let (y, u, v) = ramp(sw, sh);
    let oracle = encoded_oracle_rgb(&y, &u, &v, sw, sh, ow, oh, true);
    for loc in [
      ChromaLocation::Center,
      ChromaLocation::Top,
      ChromaLocation::Bottom,
    ] {
      let got = run(&y, &u, &v, sw, sh, ow, oh, loc.clone(), false, true);
      assert_eq!(got.0, oracle, "rgb {loc:?} {sw}x{sh}->{ow}x{oh}");
    }
  }
}

// ---- non-vacuous + flat-chroma sanity --------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_differs_from_cosited_on_a_chroma_ramp() {
  // The phase must actually DO something: on a horizontal chroma ramp the
  // centered decode diverges from co-sited on both tiers.
  let (y, u, v) = ramp(8, 8);
  for native in [true, false] {
    let cos = run(
      &y,
      &u,
      &v,
      8,
      8,
      4,
      4,
      ChromaLocation::Unspecified,
      native,
      true,
    );
    let cen = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
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
  // Sanity: on constant chroma the centered triangle is a no-op, so centered
  // and co-sited agree byte-for-byte (the phase machinery corrupts nothing).
  let (y, u, v) = flat_chroma(8, 8);
  for native in [true, false] {
    let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
    let cen = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
    assert_eq!(cen.0, cos.0, "flat-chroma rgb (native={native})");
    assert_eq!(cen.2, cos.2, "flat-chroma hsv (native={native})");
  }
}

// ---- the ≤1 LSB single-rounding note, pinned -------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_native_is_within_1_lsb_of_reconstruct_then_bin() {
  // The folded single-rounding native output and the #302 reconstruct-to-u8-
  // then-bin (TWO roundings) agree to ≤ 1 LSB per chroma sample — the
  // user-approved more-correct form. Compared in the chroma CODE domain so the
  // convert/clamp cannot mask or amplify the gap. `[0, 2, 0, 2]` chroma is a
  // crafted case that provably exercises the divergence: the odd-column
  // reconstruction lands on an exact `.5` (`(3·0 + 2)/4`), which the folded
  // single rounding averages down to 0 while the intermediate `>>2` rounds it
  // up to 1 first (so reconstruct-then-bin yields 1) — a genuine 1-LSB gap.
  let (cw, sh, ow, oh) = (4usize, 2usize, 4usize, 2usize);
  let u: Vec<u8> = (0..cw * sh)
    .map(|i| if i.is_multiple_of(2) { 0 } else { 2 })
    .collect();
  let folded = bin_chroma_centered(&u, cw, sh, ow, oh);
  // reconstruct-then-bin: #302 to u8 per row, then co-sited box-average.
  let mut recon = vec![0u8; 2 * cw * sh];
  for r in 0..sh {
    recon[r * 2 * cw..r * 2 * cw + 2 * cw]
      .copy_from_slice(&recon_full_row(&u[r * cw..r * cw + cw], cw));
  }
  let double = bin_cosited(&recon, 2 * cw, sh, ow, oh);
  let maxd = folded
    .iter()
    .zip(&double)
    .map(|(&a, &b)| a.abs_diff(b))
    .max()
    .unwrap();
  assert!(
    maxd <= 1,
    "folded vs reconstruct-then-bin max delta {maxd} must be ≤ 1 LSB"
  );
  // Non-vacuous: the two rounding orders DO diverge here (proving the folded
  // single-rounding form is a genuine, more-correct distinction from #302).
  assert_ne!(folded, double, "the ≤1 LSB gap must be exercised");
}

// ---- cross-frame sink reuse rebuilds the phased join (RFC #238) -------------
//
// The native / HSV-only joins cache a chroma plan built for ONE frame's siting
// and are only `reset` between frames; a reused sink whose `chroma_location`
// changed to a different phase must REBUILD the join, else frame 2 inherits
// frame 1's (folded centered ⇄ unscaled co-sited) weights.

/// Reuse ONE full-output native-tier sink across two frames of the SAME
/// content, siting `loc1` then `loc2` (via `set_chroma_location` between
/// walks), returning frame 2's outputs.
#[allow(clippy::too_many_arguments)]
fn run_reuse_native(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc1: ChromaLocation,
  loc2: ChromaLocation,
  simd: bool,
) -> Outs {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(true)
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
    let f = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    sink.set_chroma_location(loc1.clone());
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_chroma_location(loc2.clone());
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// One HSV-only (`with_native(false)` → the `HsvDirectPlanarYuv` join) frame.
#[allow(clippy::too_many_arguments)]
fn run_hsv_only(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  simd: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let f = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (hh, ss, vv)
}

/// Reuse ONE HSV-only sink across two frames, siting `loc1` then `loc2`.
#[allow(clippy::too_many_arguments)]
fn run_reuse_hsv_only(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc1: ChromaLocation,
  loc2: ChromaLocation,
  simd: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_simd(simd)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let f = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    sink.set_chroma_location(loc1.clone());
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_chroma_location(loc2.clone());
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (hh, ss, vv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_join_rebuilds_on_siting_change_across_frames() {
  // Reuse one native-tier sink flipping Left ⇄ Center (both directions): frame
  // 2 must match a FRESH sink for frame 2's siting — no stale-phase carryover.
  let (y, u, v) = ramp(8, 8);
  for (a, b) in [
    (ChromaLocation::Left, ChromaLocation::Center),
    (ChromaLocation::Center, ChromaLocation::Left),
  ] {
    let reused = run_reuse_native(&y, &u, &v, 8, 8, 4, 4, a.clone(), b.clone(), true);
    let fresh = run(&y, &u, &v, 8, 8, 4, 4, b.clone(), true, true);
    assert_eq!(
      reused.0, fresh.0,
      "native rgb {a:?}->{b:?} stale-phase carryover"
    );
    assert_eq!(reused.1, fresh.1, "native rgba {a:?}->{b:?}");
    assert_eq!(reused.2, fresh.2, "native hsv {a:?}->{b:?}");
    // Non-vacuous: the two sitings genuinely differ on this ramp, so matching
    // frame 2's siting (not frame 1's) is a real distinction.
    let stale = run(&y, &u, &v, 8, 8, 4, 4, a.clone(), true, true);
    assert_ne!(
      fresh.0, stale.0,
      "sitings {a:?} vs {b:?} must differ (non-vacuous)"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_only_join_rebuilds_on_siting_change_across_frames() {
  // The `HsvDirectPlanarYuv` twin: reuse one HSV-only sink flipping Left ⇄
  // Center; frame 2 must match a fresh sink for its siting.
  let (y, u, v) = ramp(8, 8);
  for (a, b) in [
    (ChromaLocation::Left, ChromaLocation::Center),
    (ChromaLocation::Center, ChromaLocation::Left),
  ] {
    let reused = run_reuse_hsv_only(&y, &u, &v, 8, 8, 4, 4, a.clone(), b.clone(), true);
    let fresh = run_hsv_only(&y, &u, &v, 8, 8, 4, 4, b.clone(), true);
    assert_eq!(reused, fresh, "hsv-only {a:?}->{b:?} stale-phase carryover");
    let stale = run_hsv_only(&y, &u, &v, 8, 8, 4, 4, a.clone(), true);
    assert_ne!(
      fresh, stale,
      "sitings {a:?} vs {b:?} must differ (non-vacuous)"
    );
  }
}

// ---- siting changed AFTER begin_frame (point-of-use invalidation) -----------
//
// A begin_frame-only invalidation misses a `set_chroma_location` /
// `set_color_spec` between `begin_frame` and row 0; the point-of-use re-check
// in `process` catches it. These drive the sink MANUALLY so the setter lands
// after `begin_frame`.

/// Apply the new siting via one of the two setters (both funnel to
/// `self.chroma_location`, the field the point-of-use check reads).
fn apply_siting<R>(
  sink: &mut MixedSinker<'_, Yuv422p, R>,
  loc: ChromaLocation,
  via_color_spec: bool,
) {
  if via_color_spec {
    let spec = ColorSpec::from_info(
      PixelFormat::Yuv422p,
      ColorInfo::new(
        Primaries::Unspecified,
        Transfer::Unspecified,
        crate::ColorMatrix::from(M),
        DynamicRange::Limited,
        loc,
      ),
    );
    sink.set_color_spec(&spec).unwrap();
  } else {
    sink.set_chroma_location(loc.clone());
  }
}

/// Reuse a native-tier sink: frame 1 at `loc1` (walker), then MANUALLY drive
/// frame 2 — `begin_frame` while still `loc1`, THEN switch to `loc2`, THEN feed
/// rows — returning frame 2's outputs.
#[allow(clippy::too_many_arguments)]
fn run_reuse_native_setter_after(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc1: ChromaLocation,
  loc2: ChromaLocation,
  via_color_spec: bool,
  simd: bool,
) -> Outs {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(true)
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
    let f = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    sink.set_chroma_location(loc1.clone());
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
    apply_siting(&mut sink, loc2, via_color_spec); // AFTER begin_frame, before row 0
    for r in 0..sh {
      let row = Yuv422pRow::for_tests(
        &y[r * sw..r * sw + sw],
        &u[r * cw..r * cw + cw],
        &v[r * cw..r * cw + cw],
        r,
        M,
        FR,
      );
      PixelSink::process(&mut sink, row).unwrap();
    }
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The HSV-only (`with_native(false)` → `HsvDirectPlanarYuv`) twin of
/// [`run_reuse_native_setter_after`].
#[allow(clippy::too_many_arguments)]
fn run_reuse_hsv_setter_after(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc1: ChromaLocation,
  loc2: ChromaLocation,
  via_color_spec: bool,
  simd: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_simd(simd)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let f = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    sink.set_chroma_location(loc1.clone());
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
    apply_siting(&mut sink, loc2, via_color_spec);
    for r in 0..sh {
      let row = Yuv422pRow::for_tests(
        &y[r * sw..r * sw + sw],
        &u[r * cw..r * cw + cw],
        &v[r * cw..r * cw + cw],
        r,
        M,
        FR,
      );
      PixelSink::process(&mut sink, row).unwrap();
    }
  }
  (hh, ss, vv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_join_rebuilds_on_siting_change_after_begin_frame() {
  // set_chroma_location AND set_color_spec, both Left ⇄ Center, applied AFTER
  // begin_frame: frame 2 must still match a FRESH sink for the new siting.
  let (y, u, v) = ramp(8, 8);
  for via_color_spec in [false, true] {
    for (a, b) in [
      (ChromaLocation::Left, ChromaLocation::Center),
      (ChromaLocation::Center, ChromaLocation::Left),
    ] {
      let reused = run_reuse_native_setter_after(
        &y,
        &u,
        &v,
        8,
        8,
        4,
        4,
        a.clone(),
        b.clone(),
        via_color_spec,
        true,
      );
      let fresh = run(&y, &u, &v, 8, 8, 4, 4, b.clone(), true, true);
      assert_eq!(
        reused.0, fresh.0,
        "native rgb {a:?}->{b:?} color_spec={via_color_spec}: stale after begin_frame"
      );
      assert_eq!(
        reused.2, fresh.2,
        "native hsv {a:?}->{b:?} color_spec={via_color_spec}"
      );
      let stale = run(&y, &u, &v, 8, 8, 4, 4, a, true, true);
      assert_ne!(fresh.0, stale.0, "sitings must differ (non-vacuous)");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_only_join_rebuilds_on_siting_change_after_begin_frame() {
  let (y, u, v) = ramp(8, 8);
  for via_color_spec in [false, true] {
    for (a, b) in [
      (ChromaLocation::Left, ChromaLocation::Center),
      (ChromaLocation::Center, ChromaLocation::Left),
    ] {
      let reused = run_reuse_hsv_setter_after(
        &y,
        &u,
        &v,
        8,
        8,
        4,
        4,
        a.clone(),
        b.clone(),
        via_color_spec,
        true,
      );
      let fresh = run_hsv_only(&y, &u, &v, 8, 8, 4, 4, b.clone(), true);
      assert_eq!(
        reused, fresh,
        "hsv-only {a:?}->{b:?} color_spec={via_color_spec}: stale after begin_frame"
      );
      let stale = run_hsv_only(&y, &u, &v, 8, 8, 4, 4, a, true);
      assert_ne!(fresh, stale, "sitings must differ (non-vacuous)");
    }
  }
}

// ---- atomicity: the centered reserve sits BEHIND the resample preflight -----

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn out_of_sequence_centered_first_row_is_rejected_before_the_chroma_reserve() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  // The centered chroma reservation must run AFTER the resample preflight, so an
  // out-of-sequence FIRST row is rejected BEFORE any allocation (#180) — a
  // primed allocator refusal is never reached (OutOfSequenceRow, not
  // AllocationFailed). `with_native(false)` forces the encoded convert path (the
  // native fast tier bins the folded chroma plan and never reserves).
  let (y, u, v) = ramp(8, 8);
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  super::super::arm_chroma_full_alloc_failure();
  // First process call is row 5 — the stream expects row 0.
  let bad = Yuv422pRow::for_tests(
    &y[5 * 8..6 * 8],
    &u[5 * 4..6 * 4],
    &v[5 * 4..6 * 4],
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
  // the reserve (proving the guard is ordering, not a disabled reserve) — the
  // primed refusal surfaces as a recoverable AllocationFailed.
  let good = Yuv422pRow::for_tests(&y[0..8], &u[0..4], &v[0..4], 0, M, FR);
  let err0 = PixelSink::process(&mut sink, good).unwrap_err();
  assert!(
    matches!(
      err0,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "a valid centered row reaches the reserve (failpoint fires), got {err0:?}"
  );
}

/// A centered COLOUR area sink whose first row DOES reach the chroma reserve,
/// firing (and thereby consuming) the thread-local chroma failpoint. Asserts it
/// was still armed — confirming a preceding path never reserved — and leaves it
/// disarmed for the next test.
fn assert_chroma_failpoint_armed_then_consume(y: &[u8], u: &[u8], v: &[u8]) {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  let f = Yuv422pFrame::new(y, u, v, 8, 8, 8, 4, 4);
  let err = yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the chroma failpoint must still be armed (the prior path never reserved), got {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_filter_bicublin_rejected_before_the_chroma_reserve() {
  use super::super::MixedSinkerError;
  use crate::resample::{Bicublin, ResampleError};
  // A BICUBLIN (multi-kernel) filter plan on Yuv422p must be rejected as
  // UnsupportedFilter BEFORE the centered chroma reserve. With the failpoint
  // armed, an unfixed reserve-first path would surface the wrong
  // AllocationFailed instead.
  let (y, u, v) = ramp(8, 8);
  let mut rgb = vec![0u8; 4 * 4 * 3];
  {
    let mut sink = MixedSinker::<Yuv422p, Bicublin>::with_resampler(8, 8, Bicublin::to(4, 4))
      .unwrap()
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
    PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
    super::super::arm_chroma_full_alloc_failure();
    let row = Yuv422pRow::for_tests(&y[0..8], &u[0..4], &v[0..4], 0, M, FR);
    let err = PixelSink::process(&mut sink, row).unwrap_err();
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
  }
  // The reject never reached the reserve, so the failpoint is still armed.
  assert_chroma_failpoint_armed_then_consume(&y, &u, &v);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_centered_area_does_not_reserve_chroma() {
  // A luma-only centered area resample (with_native(false), only luma attached)
  // never calls the RGB converter, so it must NOT reserve/reconstruct chroma:
  // with the failpoint armed it still succeeds (an unfixed path would reserve
  // and mask the luma output with a spurious chroma AllocationFailed).
  let (y, u, v) = ramp(8, 8);
  let mut luma = vec![0u8; 4 * 4];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(false)
        .with_chroma_location(ChromaLocation::Center)
        .with_luma(&mut luma)
        .unwrap();
    PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
    super::super::arm_chroma_full_alloc_failure();
    for r in 0..8 {
      let row = Yuv422pRow::for_tests(
        &y[r * 8..r * 8 + 8],
        &u[r * 4..r * 4 + 4],
        &v[r * 4..r * 4 + 4],
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
  // The luma-only path never reserved, so the failpoint is still armed.
  assert_chroma_failpoint_armed_then_consume(&y, &u, &v);
}

// ---- centered LINEAR folds the chroma phase (both decodes) ------------------

/// A `Yuv422p` linear-light area resample (`with_native(false)`) to RGB, at
/// `loc` siting and `mode` (display-referred clamped vs scene-referred f32).
fn run_linear_422(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  loc: ChromaLocation,
  mode: LinearMode,
  simd: bool,
) -> Vec<u8> {
  let (sw, sh, ow, oh, cw) = (8usize, 8usize, 4usize, 4usize, 4usize);
  let mut rgb = vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_native(false)
        .with_linear_mode(mode)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap();
    let f = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

/// The centered-Linear oracle: reconstruct U / V to full width with the #302
/// kernel (u8), then run that `Yuv444p` frame through the SAME linear-light
/// resample — i.e. reconstruct-then-linear-average, the exact operation the
/// centered `Yuv422p` Linear arm performs (decode over full-width chroma,
/// linearise, area-average, re-encode).
fn oracle_linear_reconstruct(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  mode: LinearMode,
  simd: bool,
) -> Vec<u8> {
  let (sw, sh, ow, oh, cw) = (8usize, 8usize, 4usize, 4usize, 4usize);
  let mut uf = vec![0u8; sw * sh];
  let mut vf = vec![0u8; sw * sh];
  for r in 0..sh {
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&u[r * cw..r * cw + cw], cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&v[r * cw..r * cw + cw], cw));
  }
  let mut rgb = vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_native(false)
        .with_linear_mode(mode)
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap();
    let f = Yuv444pFrame::new(
      y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_linear_folds_the_phase_for_both_decodes() {
  // Centered Linear must reconstruct full-width chroma and decode 4:4:4 (not
  // silently co-site) for BOTH the display-referred (clamped u8) and
  // scene-referred (f32 unclamped) decodes: it equals reconstruct-then-linear,
  // and differs from co-sited on a chroma ramp.
  let (y, u, v) = ramp(8, 8);
  for mode in [LinearMode::DisplayReferred, LinearMode::SceneReferred] {
    let centered = run_linear_422(&y, &u, &v, ChromaLocation::Center, mode, true);
    let oracle = oracle_linear_reconstruct(&y, &u, &v, mode, true);
    assert_eq!(
      centered, oracle,
      "centered Linear ({mode:?}) must equal reconstruct-then-linear-average"
    );
    let cosited = run_linear_422(&y, &u, &v, ChromaLocation::Left, mode, true);
    assert_ne!(
      centered, cosited,
      "centered Linear ({mode:?}) must differ from co-sited (non-vacuous)"
    );
  }
}

// ---- mid-frame phase change rejects WITHOUT dropping the cached stream ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_mid_frame_phase_change_rejection_keeps_the_stream_retryable() {
  use super::super::MixedSinkerError;
  // Advance rows 0,1 (Center), then flip siting mid-frame (Left). FIX 1 (R6):
  // the frozen-phase CHECK rejects it with ChromaSitingChanged at the choke
  // point, ahead of the out-of-sequence check — a mixed-phase frame is never
  // emitted; the frame must be restarted (a rejected row mutates no state).
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..2 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  sink.set_chroma_location(ChromaLocation::Left);
  let bad = Yuv422pRow::for_tests(
    &y[5 * 8..6 * 8],
    &u[5 * cw..6 * cw],
    &v[5 * cw..6 * cw],
    5,
    M,
    FR,
  );
  let err = PixelSink::process(&mut sink, bad).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
    "mid-frame siting change must be ChromaSitingChanged, got {err:?}"
  );
  // The rejected row mutated no stream state: begin_frame restarts cleanly and
  // a fresh frame at the new siting processes without error.
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_mid_frame_phase_change_rejection_keeps_the_stream_retryable() {
  use super::super::MixedSinkerError;
  // The HSV-only (`hsv_direct`) twin: a mid-frame siting flip is rejected with
  // ChromaSitingChanged (FIX 1), and begin_frame restarts cleanly.
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..2 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  sink.set_chroma_location(ChromaLocation::Left);
  let bad = Yuv422pRow::for_tests(
    &y[5 * 8..6 * 8],
    &u[5 * cw..6 * cw],
    &v[5 * cw..6 * cw],
    5,
    M,
    FR,
  );
  let err = PixelSink::process(&mut sink, bad).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
    "mid-frame siting change (HSV) must be ChromaSitingChanged, got {err:?}"
  );
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
}

// ---- IN-SEQUENCE mid-frame phase change is rejected (not silently mixed) -----
//
// freezing the phase per-frame is not enough to DROP a stale plan — an
// in-sequence row after a mid-frame `set_chroma_location` passes the sequence
// preflight, so without the frozen-phase CHECK it would reconstruct the new
// phase and the frame would bin a mixture. The effective siting is frozen on
// the first output-bearing row; a later in-sequence row observing a different
// phase must be rejected with `ChromaSitingChanged`, uniformly across tiers.

/// Drive one Yuv422p resample frame: `begin_frame`, accept row 0 at `loc1`
/// (freezes the phase), flip to `loc2`, then feed the IN-SEQUENCE row 1 and
/// return its `process` result.
fn in_sequence_flip_row1<R>(
  mut sink: MixedSinker<'_, Yuv422p, R>,
  y: &[u8],
  u: &[u8],
  v: &[u8],
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> Result<(), super::super::MixedSinkerError> {
  let cw = 4usize;
  sink.set_chroma_location(loc1.clone());
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0).unwrap();
  sink.set_chroma_location(loc2.clone());
  let row1 = Yuv422pRow::for_tests(&y[8..16], &u[cw..2 * cw], &v[cw..2 * cw], 1, M, FR);
  PixelSink::process(&mut sink, row1)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn in_sequence_mid_frame_phase_change_rejected_across_tiers() {
  use super::super::MixedSinkerError;
  let (y, u, v) = ramp(8, 8);
  // Both flip directions: Center->Left (drop the phase) and Left->Center (add
  // it). Each must reject the in-sequence row 1 with ChromaSitingChanged.
  for (loc1, loc2) in [
    (ChromaLocation::Center, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Center),
  ] {
    // Native fast tier.
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink = MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "native {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // Encoded row-stage RGB tier (`with_native(false)`).
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink = MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_rgb(&mut rgb)
      .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "encoded-rgb {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // HSV-only tier (the `HsvDirectPlanarYuv` join).
    let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
    let sink = MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "hsv {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // Linear averaging domain.
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink = MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_averaging_domain(AveragingDomain::Linear)
      .with_native(false)
      .with_rgb(&mut rgb)
      .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "linear {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // Filter tier (single-kernel Triangle FilteredResampler).
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink =
      MixedSinker::<Yuv422p, crate::resample::FilteredResampler<crate::resample::Triangle>>::with_resampler(
        8,
        8,
        crate::resample::FilteredResampler::new(4, 4, crate::resample::Triangle),
      )
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
  }
}

// ---- centered Linear preflight runs BEFORE the chroma reserve ---------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_linear_output_set_change_rejects_before_the_chroma_reserve() {
  use super::super::MixedSinkerError;
  // a luma-only centered Linear row 0 reserves no chroma (want_color
  // false). Attaching RGB mid-frame flips want_color true while chroma_full is
  // still empty, so the CENTERED Linear arm would reserve + reconstruct — but
  // the row is a mid-frame output-set change and must be rejected
  // (ResampleOutputsChanged) by the full Linear preflight BEFORE any allocation.
  // With the chroma failpoint armed, a reserve-first path would surface the
  // WRONG AllocationFailed instead (non-vacuous).
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut luma = vec![0u8; 4 * 4];
  let mut rgb = vec![0u8; 4 * 4 * 3];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_native(false)
        .with_chroma_location(ChromaLocation::Center)
        .with_luma(&mut luma)
        .unwrap();
    PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
    let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
    PixelSink::process(&mut sink, row0).unwrap(); // luma-only: reserves no chroma
    super::super::arm_chroma_full_alloc_failure();
    sink.set_rgb(&mut rgb).unwrap(); // attach RGB mid-frame → want_color true
    let row1 = Yuv422pRow::for_tests(&y[8..16], &u[cw..2 * cw], &v[cw..2 * cw], 1, M, FR);
    let err = PixelSink::process(&mut sink, row1).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
      "mid-frame output-set change must be ResampleOutputsChanged (reserve unreached), got {err:?}"
    );
    assert_eq!(
      sink.chroma_full.len(),
      0,
      "a rejected centered-Linear row reserves no chroma scratch"
    );
  }
  // The reject never reached the reserve, so the failpoint is still armed.
  assert_chroma_failpoint_armed_then_consume(&y, &u, &v);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_rowstage_reserve_oom_leaves_outputs_unfrozen_for_retry() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  // the centered row-stage / filter arm gates its chroma reserve with a
  // CHECK-ONLY preflight (no output-set freeze). A recoverable reserve OOM on
  // the first output-bearing row must leave `resample_outputs` uncommitted, so a
  // retry of row 0 with a CHANGED output attachment is accepted — not falsely
  // rejected as ResampleOutputsChanged (a committing preflight would freeze the
  // old output set before the failing reserve).
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut luma = vec![0u8; 4 * 4];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  super::super::arm_chroma_full_alloc_failure();
  let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  let err = PixelSink::process(&mut sink, row0).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the armed failpoint must surface AllocationFailed on the row-0 reserve, got {err:?}"
  );
  // Attach luma before retrying (changes the output set). The failpoint is
  // consumed, so the reserve now succeeds; because the OOM row never froze the
  // output set, row 0 is accepted with the new attachment.
  sink.set_luma(&mut luma).unwrap();
  let row0b = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0b).unwrap();
}

// ---- a genuine LATER-row output-set change is ResampleOutputsChanged --------
//
// the check-only preflight's ordering — the
// output-set compare precedes the general sequence reject on a later row, so
// attaching an output for row 1 (which selects a fresh, next_y==0 stream) is
// surfaced as ResampleOutputsChanged, NOT OutOfSequenceRow.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_rowstage_later_output_attach_is_resample_outputs_changed() {
  use super::super::MixedSinkerError;
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut luma = vec![0u8; 4 * 4];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0).unwrap(); // RGB-only, freezes {rgb}
  sink.set_luma(&mut luma).unwrap(); // attach luma → output set changed
  let row1 = Yuv422pRow::for_tests(&y[8..16], &u[cw..2 * cw], &v[cw..2 * cw], 1, M, FR);
  let err = PixelSink::process(&mut sink, row1).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "later output-set change (row-stage) must be ResampleOutputsChanged, got {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_filter_later_output_attach_is_resample_outputs_changed() {
  use super::super::MixedSinkerError;
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut luma = vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<
    Yuv422p,
    crate::resample::FilteredResampler<crate::resample::Triangle>,
  >::with_resampler(
    8,
    8,
    crate::resample::FilteredResampler::new(4, 4, crate::resample::Triangle),
  )
  .unwrap()
  .with_chroma_location(ChromaLocation::Center)
  .with_rgb(&mut rgb)
  .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0).unwrap(); // RGB-only, freezes {rgb}
  sink.set_luma(&mut luma).unwrap(); // attach luma → output set changed
  let row1 = Yuv422pRow::for_tests(&y[8..16], &u[cw..2 * cw], &v[cw..2 * cw], 1, M, FR);
  let err = PixelSink::process(&mut sink, row1).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "later output-set change (filter) must be ResampleOutputsChanged, got {err:?}"
  );
}

// ---- an out-of-sequence first row is state-atomic: it does NOT drop the join -
//
// the point-of-use phase invalidation must not clear a cached join before
// the row's sequence is validated. A reused sink, new frame, changed siting,
// then an out-of-sequence first process call must be rejected (OutOfSequenceRow)
// with the cached join INTACT, so a corrected row-0 retry rebuilds cleanly.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_out_of_sequence_first_row_does_not_drop_the_cached_join() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_chroma_location(ChromaLocation::Left)
      .with_rgb(&mut rgb)
      .unwrap();
  // Frame 1 at Left builds the native join.
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  assert!(
    sink.native_planar.is_some(),
    "frame 1 builds the native join"
  );
  // Frame 2: change siting to Center, then feed an OUT-OF-SEQUENCE first row.
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  sink.set_chroma_location(ChromaLocation::Center);
  let bad = Yuv422pRow::for_tests(
    &y[3 * 8..4 * 8],
    &u[3 * cw..4 * cw],
    &v[3 * cw..4 * cw],
    3,
    M,
    FR,
  );
  let err = PixelSink::process(&mut sink, bad).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "out-of-sequence first row must be OutOfSequenceRow, got {err:?}"
  );
  // the rejected row is state-atomic — the cached join is NOT dropped.
  assert!(
    sink.native_planar.is_some(),
    "an out-of-sequence first row must NOT drop the cached native join"
  );
  // The corrected retry (row 0, now rebuilding for Center) succeeds.
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_out_of_sequence_first_row_does_not_drop_the_cached_join() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Left)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  assert!(
    sink.hsv_planar.is_some(),
    "frame 1 builds the HSV-only join"
  );
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  sink.set_chroma_location(ChromaLocation::Center);
  let bad = Yuv422pRow::for_tests(
    &y[3 * 8..4 * 8],
    &u[3 * cw..4 * cw],
    &v[3 * cw..4 * cw],
    3,
    M,
    FR,
  );
  let err = PixelSink::process(&mut sink, bad).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "out-of-sequence first HSV row must be OutOfSequenceRow, got {err:?}"
  );
  assert!(
    sink.hsv_planar.is_some(),
    "an out-of-sequence first row must NOT drop the cached HSV join"
  );
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
}

// ---- a row-0 phase rebuild that OOMs is state-atomic: the old join survives ---
//
// a valid in-sequence row-0 siting change rebuilds the cached join. If that
// rebuild's allocation is refused, the OLD (prior-phase) join must be restored
// (Some), so the rejected row mutates nothing and a corrected retry can proceed.
// (Gated to the features that compile the native-join chroma failpoint.)

#[test]
#[cfg(any(
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  all(feature = "yuv-semi-planar", feature = "rgb")
))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_row0_siting_change_rebuild_oom_restores_the_old_join() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut luma = vec![0u8; 4 * 4];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_chroma_location(ChromaLocation::Left)
      .with_rgb(&mut rgb)
      .unwrap();
  // Frame 1 at Left builds the native join.
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  assert!(sink.native_planar.is_some());
  // Frame 2: change siting to Center; the in-sequence row-0 rebuild hits an
  // allocation refusal building the new (Center) chroma plan.
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  sink.set_chroma_location(ChromaLocation::Center);
  crate::sinker::mixed::arm_planar_native_chroma_failure();
  let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  let err = PixelSink::process(&mut sink, row0).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the row-0 rebuild OOM must surface AllocationFailed, got {err:?}"
  );
  // the rejected row is state-atomic — the intact prior-phase (Left) join
  // is restored (dropping the join up front then rebuilding would leave
  // native_planar None on a build failure).
  assert!(
    sink.native_planar.is_some(),
    "a row-0 rebuild OOM must restore the intact prior-phase join"
  );
  // the output-set freeze was also never committed (the delegate commits only
  // after its build succeeds) — a retry of row 0 with a CHANGED output attachment
  // (luma added) is accepted, not mis-rejected as ResampleOutputsChanged.
  sink.set_luma(&mut luma).unwrap();
  let row0b = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0b).unwrap();
}

#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_row0_rebuild_scratch_oom_is_state_atomic() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  // A stale-phase rebuild whose output-width RGB scratch grow OOMs must be
  // state-atomic. The delegate builds the replacement into a local and grows the
  // scratch BEFORE committing, so a scratch refusal leaves the field `None` (the
  // arm took the stale join) and `resample_outputs` uncommitted; the arm's
  // stale-gated restore brings the prior join back. Prime a native HSV-only join
  // (reserves no RGB scratch), switch to centered RGB, and refuse the (now-empty)
  // RGB-scratch grow on row 0.
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut luma = vec![0u8; 4 * 4];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_chroma_location(ChromaLocation::Left)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
  // Frame 1: native HSV-only join at Left (reserves no RGB scratch).
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  assert!(sink.native_planar.is_some());
  // Frame 2: switch to Center AND attach RGB → the row-0 rebuild inserts the new
  // join, then the (now-needed, empty) RGB-scratch grow is refused.
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  sink.set_chroma_location(ChromaLocation::Center);
  sink.set_rgb(&mut rgb).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  let err = PixelSink::process(&mut sink, row0).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the post-insert RGB-scratch OOM must surface AllocationFailed, got {err:?}"
  );
  // state-atomic — the stale-gated restore brings the prior join back (the
  // delegate never committed the rebuilt join, since the scratch grow precedes
  // the insert).
  assert!(
    sink.native_planar.is_some(),
    "a rejected rebuild must restore the prior join"
  );
  // the output-set freeze was never committed (the delegate commits only after
  // the scratch grows), so a retry of row 0 with a CHANGED output set (luma
  // added) is ACCEPTED, not mis-rejected as ResampleOutputsChanged.
  sink.set_luma(&mut luma).unwrap();
  let row0b = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0b).unwrap();
}

#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_first_build_scratch_oom_leaves_freeze_unfrozen_for_retry() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  // An ORDINARY first-ever row-0 native build (no prior/stale join) grows the
  // output RGB scratch BEFORE the delegate commits the output-set freeze. A
  // scratch OOM on that first build therefore leaves `resample_outputs`
  // uncommitted (the delegate never freezes on a pre-feed failure), with no arm
  // rollback needed, so a retry with a changed output attachment is accepted,
  // not mis-rejected as ResampleOutputsChanged.
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut luma = vec![0u8; 4 * 4];
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  let err = PixelSink::process(&mut sink, row0).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "first-build RGB-scratch OOM must surface AllocationFailed, got {err:?}"
  );
  // The freeze was never committed on the failed build — a retry of row 0 with
  // luma added (changed output set) is ACCEPTED, not ResampleOutputsChanged.
  sink.set_luma(&mut luma).unwrap();
  let row0b = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0b).unwrap();
}

#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_colour_capability_rebuild_scratch_oom_leaves_freeze_unfrozen_for_retry() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  // A cross-frame COLOUR-CAPABILITY rebuild: frame 1 builds a luma-only native
  // join (chroma absent, no output RGB scratch). Frame 2 attaches RGB, so the
  // delegate must rebuild the join WITH chroma and grow the (empty) output RGB
  // scratch — both before the row is accepted. A scratch OOM on that rebuild must
  // leave the cached luma-only join intact AND `resample_outputs` uncommitted, so
  // a row-0 retry with a CHANGED output set is accepted, not mis-rejected as
  // ResampleOutputsChanged. Pre-fix the committing preflight froze the luma-only
  // set and the up-front `*native = None` dropped the join before the failing
  // rebuild — both poisoning the retry.
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut luma = vec![0u8; 4 * 4];
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
  let mut sink =
    MixedSinker::<Yuv422p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();
  // Frame 1: a luma-only native join (chroma absent, reserves no RGB scratch).
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..8 {
    let row = Yuv422pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[r * cw..r * cw + cw],
      &v[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  assert!(
    sink.native_planar.is_some(),
    "frame 1 builds the luma-only native join"
  );
  // Frame 2: attach RGB → the row-0 rebuild must build the chroma half and grow
  // the (now-needed, empty) output RGB scratch; refuse that grow.
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  sink.set_rgb(&mut rgb).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let row0 = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  let err = PixelSink::process(&mut sink, row0).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the colour-capability rebuild scratch OOM must surface AllocationFailed, got {err:?}"
  );
  // The delegate built the replacement into a local and never cleared the field,
  // so the intact luma-only join survives the rejected rebuild.
  assert!(
    sink.native_planar.is_some(),
    "a rejected colour-capability rebuild must not drop the cached join"
  );
  // The output-set freeze was never committed (the delegate commits only after
  // the scratch grows), so a row-0 retry with ANOTHER output attached (add HSV)
  // is ACCEPTED, not mis-rejected as ResampleOutputsChanged.
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  let row0b = Yuv422pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0b).unwrap();
}
