//! RFC #238 S8b — chroma-siting-aware 4:4:0 **resample** for `Yuv440p`.
//!
//! 4:4:0 keeps FULL-width chroma, subsampled 2:1 **vertically only**, so the
//! siting reduces to its vertical axis: `Bottom`
//! ([`chroma_440_bottom_sited_v`](super::super::chroma_440_bottom_sited_v),
//! `v = 1`) folds the VERTICAL triangle through the resample (its even output
//! row box-blends the previous chroma row), while every co-sited / horizontal
//! siting keeps the vertical pairing co-sited (`v_phase = 0`, byte-identical to
//! the pre-siting resample):
//!  - the **native fast tier** folds the `v = 1` triangle into the chroma area
//!    weights ([`ResamplePlan::area_chroma_440`]) — one SINGLE-rounding phased
//!    box-average on the half-height grid (no horizontal reconstruction);
//!  - the **encoded row-stage tier** (`with_native(false)`) reconstructs
//!    full-height chroma then bins in RGB.
//!
//! ★ Oracle (native tier): the EXACT code-domain box-average of the UNROUNDED
//! vertical-`v = 1`-reconstructed chroma — a SINGLE rounding — pinned against a
//! YUV-domain oracle. The oracle uses EVEN source heights so the phased-V spans
//! (denominator `2·luma_h`) align with the `sh / 2` chroma rows.

use crate::{
  ChromaLocation, KernelMatrix, PixelSink,
  resample::{AreaResampler, AveragingDomain},
  sinker::MixedSinker,
  source::{Yuv440p, Yuv440pRow, Yuv444p, yuv440p_to, yuv444p_to},
};
use mediaframe::frame::{Yuv440pFrame, Yuv444pFrame};

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
/// `resample::AxisSpans::area`. Returns per output `(first source cell,
/// overlaps)`.
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
/// (round-half-up) — the reference for a phase-free plane (luma).
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

/// The EXACT bottom-sited (`v = 1`) 4:4:0 chroma oracle for the native tier: the
/// vertical `v = 1` triangle (×2 — even luma row `2i` box-blends chroma rows
/// `{i - 1, i}` with weights `{1, 1}`, odd row `2i + 1` takes chroma row `i` with
/// weight 2, top-edge clamp) kept UNROUNDED over the full-width `sw x sh` grid,
/// box-averaged to `ow x oh` — HORIZONTAL a plain box over `sw` (4:4:0 chroma is
/// full-width, no folded H triangle), VERTICAL over the `sh` luma rows — with a
/// SINGLE round-half-up over `2·sw·sh`. The code-domain twin the folded
/// [`ResamplePlan::area_chroma_440`] realizes when its V axis is
/// [`AxisSpans::area_chroma_phased_v`] (EVEN `sh` only, `ch = sh / 2`).
fn bin_chroma_bottom(c: &[u8], sw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u8> {
  let sh = 2 * ch;
  // ×2 UNROUNDED vertical reconstruction (`sw x sh`), full width (no horizontal
  // triangle for 4:4:0), so the box below applies the one rounding.
  let mut r2 = vec![0u64; sw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let prev = cr.saturating_sub(1);
    for j in 0..sw {
      r2[r * sw + j] = if r & 1 == 0 {
        u64::from(c[prev * sw + j]) + u64::from(c[cr * sw + j]) // even: {1, 1}
      } else {
        2 * u64::from(c[cr * sw + j]) // odd: {2}
      };
    }
  }
  let hw = area_weights(sw, ow);
  let vw = area_weights(sh, oh);
  let denom = (2 * sw * sh) as u64; // ×2 (V triangle) × the box normalization
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * r2[(vs + dy) * sw + hs + dx];
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// Reconstruct `Yuv440p` chroma to full height (`sw x sh`) for the bottom-sited
/// (`v = 1`) decode — the identity bottom kernel at source width: per luma row
/// the even rows vertically box-blend chroma rows `i - 1` (clamped) and `i`
/// (round-half-up), the odd rows take chroma row `i` straight through. No
/// horizontal reconstruction (full-width chroma). The shared reconstruction step
/// for the reconstruct-then-bin oracles.
fn recon_full_bottom(u: &[u8], v: &[u8], sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>) {
  let vblend = |plane: &[u8], cr: usize, prev: usize| -> Vec<u8> {
    (0..sw)
      .map(|c| {
        let a = u32::from(plane[prev * sw + c]);
        let b = u32::from(plane[cr * sw + c]);
        ((a + b + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  let mut uf = vec![0u8; sw * sh];
  let mut vf = vec![0u8; sw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let (uh, vh) = if r & 1 == 0 {
      let prev = cr.saturating_sub(1);
      (vblend(u, cr, prev), vblend(v, cr, prev))
    } else {
      (
        u[cr * sw..cr * sw + sw].to_vec(),
        v[cr * sw..cr * sw + sw].to_vec(),
      )
    };
    uf[r * sw..r * sw + sw].copy_from_slice(&uh);
    vf[r * sw..r * sw + sw].copy_from_slice(&vh);
  }
  (uf, vf)
}

/// A `Yuv440p` fixture (full-width chroma, `ch = sh / 2`) with a horizontal AND
/// vertical chroma ramp plus varying luma. `sh` must be even.
fn ramp(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let ch = sh / 2;
  let mut y = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i as u32 * 3) % 160) as u8;
  }
  let mut u = vec![0u8; sw * ch];
  let mut v = vec![0u8; sw * ch];
  for r in 0..ch {
    for c in 0..sw {
      u[r * sw + c] = (30 + c * 12 + r * 40).min(240) as u8;
      v[r * sw + c] = (230u32.saturating_sub((c * 10 + r * 40) as u32)).max(16) as u8;
    }
  }
  (y, u, v)
}

/// A `Yuv440p` fixture with flat luma and a strong per-ROW chroma step (flat
/// across columns), so the vertical bottom fold is observable in isolation.
/// `sh` must be even.
fn vramp(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let ch = sh / 2;
  let y = vec![128u8; sw * sh];
  let mut u = vec![0u8; sw * ch];
  let mut v = vec![0u8; sw * ch];
  for r in 0..ch {
    for c in 0..sw {
      u[r * sw + c] = (20 + r * 40).min(240) as u8;
      v[r * sw + c] = (220u32.saturating_sub((r * 40) as u32)).max(16) as u8;
    }
  }
  (y, u, v)
}

/// A flat-chroma fixture: the vertical blend of a constant is that constant, so
/// `Bottom` must equal co-sited. Luma still varies.
fn flat_chroma(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let ch = sh / 2;
  let mut y = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i as u32 * 7) % 170) as u8;
  }
  (y, vec![110u8; sw * ch], vec![140u8; sw * ch])
}

type Outs = (
  Vec<u8>,
  Vec<u8>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);

/// Drive a `Yuv440p` area resample (`sw x sh -> ow x oh`) for the full output
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
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
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
    let f = Yuv440pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The bottom-sited NATIVE oracle: bin Y co-sited and U / V through the exact
/// bottom V-fold chroma oracle to `ow x oh`, then convert ONCE at output width
/// via an identity `Yuv444p` sink — the byte-for-byte ground truth the native
/// tier reproduces for `ChromaLocation::Bottom` (EVEN `sh` only).
#[allow(clippy::too_many_arguments)]
fn bottom_native_oracle(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Outs {
  let ch = sh / 2;
  let yb = bin_cosited(y, sw, sh, ow, oh);
  let ub = bin_chroma_bottom(u, sw, ch, ow, oh);
  let vb = bin_chroma_bottom(v, sw, ch, ow, oh);
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

/// The bottom-sited ENCODED row-stage oracle: reconstruct U / V to full height
/// with the vertical bottom blend ([`recon_full_bottom`]) then run that
/// full-resolution `Yuv444p` frame through a `with_native(false)` RGB-domain
/// resample (convert-each-row-then-bin-RGB, exactly what the `Yuv440p` encoded
/// arm does for `Bottom`).
#[allow(clippy::too_many_arguments)]
fn encoded_oracle_rgb_bottom(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Vec<u8> {
  let (uf, vf) = recon_full_bottom(u, v, sw, sh);
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

/// Direct (non-resample) `Yuv440p` `Bottom` decode to RGB — the delay-line
/// kernel path, the already-validated identity Bottom reference the resample
/// path must match at identity dimensions.
fn direct_bottom_rgb(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize, simd: bool) -> Vec<u8> {
  let mut rgb = vec![0u8; sw * sh * 3];
  {
    let mut sink = MixedSinker::<Yuv440p>::new(sw, sh)
      .with_chroma_location(ChromaLocation::Bottom)
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap();
    let f = Yuv440pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
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
  // Every co-sited / vertically-central siting must produce the byte-identical
  // pre-siting resample (v_phase 0 && !v_top → neither phased-V plan is built), on
  // BOTH tiers. 4:4:0 has no horizontal phase, so `Center` (v=0.5) stays co-sited;
  // `Top` / `TopLeft` (v=0) now fold the forward triangle and left this group.
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
      ChromaLocation::Center,
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

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_left_resamples_as_bottom_across_tiers() {
  // 4:4:0 has no horizontal phase, so `BottomLeft` (v=1) drives the identical
  // vertical fold as `Bottom` on every tier. The ramp varies vertically, so a
  // `BottomLeft` that stayed co-sited would diverge from the folded `Bottom` —
  // making the equality a genuine regression guard.
  let (y, u, v) = ramp(8, 8);
  for native in [true, false] {
    let bottom = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
    let bl = run(
      &y,
      &u,
      &v,
      8,
      8,
      4,
      4,
      ChromaLocation::BottomLeft,
      native,
      true,
    );
    assert_eq!(bl.0, bottom.0, "rgb BottomLeft==Bottom native={native}");
    assert_eq!(bl.1, bottom.1, "rgba BottomLeft==Bottom native={native}");
    assert_eq!(bl.2, bottom.2, "hsv BottomLeft==Bottom native={native}");
    assert_eq!(bl.3, bottom.3, "luma BottomLeft==Bottom native={native}");
    assert_eq!(
      bl.4, bottom.4,
      "luma_u16 BottomLeft==Bottom native={native}"
    );
  }
}

// ---- bottom native == the exact code-domain oracle -------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_native_equals_code_domain_oracle() {
  // The native tier folds the vertical v=1 triangle into the chroma area
  // weights; its output is the EXACT code-domain box-average of the UNROUNDED
  // V reconstruction (single rounding), for clean 2:1 and fractional ratios
  // (EVEN source height).
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
    let (y, u, v) = ramp(sw, sh);
    let o = bottom_native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
    let n = run(
      &y,
      &u,
      &v,
      sw,
      sh,
      ow,
      oh,
      ChromaLocation::Bottom,
      true,
      true,
    );
    assert_eq!(n.0, o.0, "rgb {sw}x{sh}->{ow}x{oh}");
    assert_eq!(n.1, o.1, "rgba {sw}x{sh}->{ow}x{oh}");
    assert_eq!(n.2, o.2, "hsv {sw}x{sh}->{ow}x{oh}");
    assert_eq!(n.3, o.3, "luma {sw}x{sh}->{ow}x{oh}");
    assert_eq!(n.4, o.4, "luma_u16 {sw}x{sh}->{ow}x{oh}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_native_simd_matches_scalar() {
  // Precomputed integer weights → the SIMD H/V passes are 0-ULP against scalar.
  let (y, u, v) = ramp(8, 8);
  let s = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, true, false);
  let d = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, true, true);
  assert_eq!(s.0, d.0, "rgb scalar vs simd");
  assert_eq!(s.2, d.2, "hsv scalar vs simd");
  assert_eq!(s.3, d.3, "luma scalar vs simd");
}

// ---- bottom encoded row-stage / filter / linear == reconstruct-then-bin ----

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_encoded_output_equals_rgb_reconstruct_then_bin() {
  // The encoded row-stage tier reconstructs full-height chroma with the vertical
  // bottom blend then bins in RGB — the reconstruct-then-bin oracle.
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 6, 4)] {
    let (y, u, v) = ramp(sw, sh);
    let oracle = encoded_oracle_rgb_bottom(&y, &u, &v, sw, sh, ow, oh, true);
    let got = run(
      &y,
      &u,
      &v,
      sw,
      sh,
      ow,
      oh,
      ChromaLocation::Bottom,
      false,
      true,
    );
    assert_eq!(got.0, oracle, "rgb {sw}x{sh}->{ow}x{oh}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_filter_equals_reconstruct_then_filter() {
  // The single-kernel filter tier reconstructs full-height chroma with the
  // vertical bottom blend then Triangle-filters it — equal to feeding the same
  // reconstruction through a Yuv444p Triangle filter, and non-vacuously
  // different from the co-sited decode on a vertical ramp.
  use crate::resample::{FilteredResampler, Triangle};
  let (sw, sh, ow, oh) = (8usize, 8usize, 4usize, 4usize);
  let (y, u, v) = vramp(sw, sh);
  let filter_440 = |loc: ChromaLocation| -> Vec<u8> {
    let mut rgb = vec![0u8; ow * oh * 3];
    {
      let mut sink = MixedSinker::<Yuv440p, FilteredResampler<Triangle>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, Triangle),
      )
      .unwrap()
      .with_chroma_location(loc.clone())
      .with_simd(true)
      .with_rgb(&mut rgb)
      .unwrap();
      let f = Yuv440pFrame::new(
        &y, &u, &v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
      );
      yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    rgb
  };
  let got = filter_440(ChromaLocation::Bottom);
  let (uf, vf) = recon_full_bottom(&u, &v, sw, sh);
  let mut oracle = vec![0u8; ow * oh * 3];
  {
    let mut sink = MixedSinker::<Yuv444p, FilteredResampler<Triangle>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, Triangle),
    )
    .unwrap()
    .with_simd(true)
    .with_rgb(&mut oracle)
    .unwrap();
    let f = Yuv444pFrame::new(
      &y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(got, oracle, "filter-tier bottom == reconstruct-then-filter");
  assert_ne!(
    got,
    filter_440(ChromaLocation::Left),
    "filter-tier bottom must differ from co-sited on a vertical ramp"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_linear_equals_reconstruct_then_linear() {
  // The linear-light tier reconstructs full-height chroma with the vertical
  // bottom blend then resamples in linear light — equal to feeding the same
  // reconstruction through a Yuv444p linear-domain resample.
  let (sw, sh, ow, oh) = (8usize, 8usize, 4usize, 4usize);
  let (y, u, v) = ramp(sw, sh);
  let mut got = vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_chroma_location(ChromaLocation::Bottom)
        .with_rgb(&mut got)
        .unwrap();
    let f = Yuv440pFrame::new(
      &y, &u, &v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let (uf, vf) = recon_full_bottom(&u, &v, sw, sh);
  let mut oracle = vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut oracle)
        .unwrap();
    let f = Yuv444pFrame::new(
      &y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(got, oracle, "linear-tier bottom == reconstruct-then-linear");
}

// ---- identity resample == direct decode ------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_encoded_identity_matches_direct_decode() {
  // At identity dimensions (out == src) the encoded resample tier reconstructs
  // chroma with the SAME bottom kernel as the direct (non-resample) Yuv440p
  // decode and bins with pass-through area weights, so routing Bottom through the
  // resample preserves the decode byte-for-byte.
  let (y, u, v) = vramp(8, 8);
  let res = run(&y, &u, &v, 8, 8, 8, 8, ChromaLocation::Bottom, false, true);
  let direct = direct_bottom_rgb(&y, &u, &v, 8, 8, true);
  assert_eq!(
    res.0, direct,
    "identity encoded resample bottom == direct decode"
  );
}

// ---- non-vacuous + flat-chroma sanity --------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_differs_from_cosited_on_vertical_ramp() {
  // The v=1 fold must actually MOVE the chroma: on a purely-vertical chroma ramp
  // Bottom diverges from the co-sited decode on BOTH tiers.
  let (y, u, v) = vramp(8, 8);
  for native in [true, false] {
    let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
    let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
    assert_ne!(
      bot.0, cos.0,
      "bottom rgb must differ from co-sited (native={native})"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_equals_cosited_on_flat_chroma() {
  // On constant chroma the vertical blend is a no-op, so Bottom collapses to the
  // co-sited decode byte-for-byte.
  let (y, u, v) = flat_chroma(8, 8);
  for native in [true, false] {
    let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
    let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
    assert_eq!(bot.0, cos.0, "flat-chroma bottom rgb (native={native})");
    assert_eq!(bot.2, cos.2, "flat-chroma bottom hsv (native={native})");
  }
}

// ---- IN-SEQUENCE mid-frame phase change is rejected (not silently mixed) -----

/// Drive one Yuv440p resample frame: `begin_frame`, accept row 0 at `loc1`
/// (freezes the vertical phase), flip to `loc2`, then feed the IN-SEQUENCE row 1
/// (still chroma row 0) and return its `process` result.
fn in_sequence_flip_row1<R>(
  mut sink: MixedSinker<'_, Yuv440p, R>,
  y: &[u8],
  u: &[u8],
  v: &[u8],
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> Result<(), super::super::MixedSinkerError> {
  sink.set_chroma_location(loc1.clone());
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  let row0 = Yuv440pRow::for_tests(&y[0..8], &u[0..8], &v[0..8], 0, M, FR);
  PixelSink::process(&mut sink, row0).unwrap();
  sink.set_chroma_location(loc2.clone());
  let row1 = Yuv440pRow::for_tests(&y[8..16], &u[0..8], &v[0..8], 1, M, FR);
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
  // Every co-sited ⇆ bottom-vertical flip (both directions) changes the v=1 fold
  // and must reject the in-sequence row 1 with ChromaSitingChanged across all
  // tiers. `Bottom` (h=0.5) and `BottomLeft` (h=0) are both v=1 for 4:4:0.
  for (loc1, loc2) in [
    (ChromaLocation::Left, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Left),
    (ChromaLocation::Center, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Center),
    (ChromaLocation::Left, ChromaLocation::BottomLeft),
    (ChromaLocation::BottomLeft, ChromaLocation::Top),
  ] {
    // Native fast tier.
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink = MixedSinker::<Yuv440p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
    let err = in_sequence_flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "native {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // Encoded row-stage RGB tier.
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink = MixedSinker::<Yuv440p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
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
    let sink = MixedSinker::<Yuv440p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
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
    let sink = MixedSinker::<Yuv440p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
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
    let sink = MixedSinker::<
      Yuv440p,
      crate::resample::FilteredResampler<crate::resample::Triangle>,
    >::with_resampler(
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

// ---- cross-frame sink reuse rebuilds the phased join -----------------------

/// Reuse ONE native-tier sink across two frames of the SAME content, siting
/// `loc1` then `loc2`, returning frame 2's RGB.
fn run_reuse_native(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> Vec<u8> {
  let mut rgb = vec![0u8; 4 * 4 * 3];
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap();
    let f = Yuv440pFrame::new(y, u, v, 8, 8, 8, 8, 8);
    sink.set_chroma_location(loc1.clone());
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_chroma_location(loc2.clone());
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

/// One HSV-only (`with_native(false)` → the `HsvDirectPlanarYuv` join) fresh
/// frame at `loc`. HSV-only takes the direct YUV→HSV binning path, distinct from
/// the RGB-derived HSV a sink with RGB attached would produce, so the reuse
/// oracle must compare against THIS, not an RGB-bearing sink.
fn run_hsv_only(y: &[u8], u: &[u8], v: &[u8], loc: ChromaLocation) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(false)
        .with_chroma_location(loc.clone())
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let f = Yuv440pFrame::new(y, u, v, 8, 8, 8, 8, 8);
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (hh, ss, vv)
}

/// Reuse ONE HSV-only sink across two frames, siting `loc1` then `loc2`.
fn run_reuse_hsv(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(false)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let f = Yuv440pFrame::new(y, u, v, 8, 8, 8, 8, 8);
    sink.set_chroma_location(loc1.clone());
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_chroma_location(loc2.clone());
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (hh, ss, vv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_join_rebuilds_on_siting_change_across_frames() {
  // Reuse one native-tier sink flipping Left ⇆ Bottom (both directions): frame 2
  // must match a FRESH sink for frame 2's siting — no stale-phase carryover.
  let (y, u, v) = vramp(8, 8);
  for (loc1, loc2) in [
    (ChromaLocation::Left, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Left),
  ] {
    let reused = run_reuse_native(&y, &u, &v, loc1.clone(), loc2.clone());
    let fresh = run(&y, &u, &v, 8, 8, 4, 4, loc2.clone(), true, true).0;
    assert_eq!(
      reused, fresh,
      "native reuse {loc1:?}->{loc2:?} must rebuild"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_join_rebuilds_on_siting_change_across_frames() {
  // The `HsvDirectPlanarYuv` twin: reuse one HSV-only sink flipping Left ⇆
  // Bottom; frame 2 must match a fresh HSV-only sink for frame 2's siting.
  let (y, u, v) = vramp(8, 8);
  for (loc1, loc2) in [
    (ChromaLocation::Left, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Left),
  ] {
    let reused = run_reuse_hsv(&y, &u, &v, loc1.clone(), loc2.clone());
    let fresh = run_hsv_only(&y, &u, &v, loc2.clone());
    assert_eq!(reused, fresh, "hsv reuse {loc1:?}->{loc2:?} must rebuild");
  }
}

// ============ Top-sited (v = 0) FORWARD one-row delay =======================

/// The code-domain twin the folded [`ResamplePlan::area_chroma_440`] realizes
/// when its V axis is [`AxisSpans::area_chroma_phased_v_top`] — the FORWARD mirror
/// of [`bin_chroma_bottom`]: EVEN luma row `2i` is co-sited on `c[i]` (weight
/// `{2}`), ODD row `2i+1` folds `c[i]` and `c[i+1]` (weight `{1, 1}`, clamped to
/// `c[i]` at the bottom edge), a SINGLE round-half-up over `2·sw·sh`. EVEN `sh`
/// only (`ch = sh / 2`).
fn bin_chroma_top(c: &[u8], sw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u8> {
  let sh = 2 * ch;
  let mut r2 = vec![0u64; sw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let next = (cr + 1).min(ch - 1);
    for j in 0..sw {
      r2[r * sw + j] = if r & 1 == 0 {
        2 * u64::from(c[cr * sw + j]) // even: co-sited {2}
      } else {
        u64::from(c[cr * sw + j]) + u64::from(c[next * sw + j]) // odd: forward {1, 1}
      };
    }
  }
  let hw = area_weights(sw, ow);
  let vw = area_weights(sh, oh);
  let denom = (2 * sw * sh) as u64;
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * r2[(vs + dy) * sw + hs + dx];
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// Reconstruct `Yuv440p` chroma to full height (`sw x sh`) for the top-sited
/// (`v = 0`) decode — the FORWARD mirror of [`recon_full_bottom`]: even rows take
/// chroma row `i` straight, odd rows forward box-blend `c[i]` and `c[i+1]`
/// (round-half-up, bottom-edge clamp). The shared reconstruction step for the
/// reconstruct-then-bin/filter/linear Top oracles.
fn recon_full_top(u: &[u8], v: &[u8], sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>) {
  let ch = sh.div_ceil(2);
  let vblend = |plane: &[u8], a: usize, b: usize| -> Vec<u8> {
    (0..sw)
      .map(|c| {
        let x = u32::from(plane[a * sw + c]);
        let y = u32::from(plane[b * sw + c]);
        ((x + y + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  let mut uf = vec![0u8; sw * sh];
  let mut vf = vec![0u8; sw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let (uh, vh) = if r & 1 == 0 {
      (
        u[cr * sw..cr * sw + sw].to_vec(),
        v[cr * sw..cr * sw + sw].to_vec(),
      )
    } else {
      let next = (cr + 1).min(ch - 1);
      (vblend(u, cr, next), vblend(v, cr, next))
    };
    uf[r * sw..r * sw + sw].copy_from_slice(&uh);
    vf[r * sw..r * sw + sw].copy_from_slice(&vh);
  }
  (uf, vf)
}

/// The top-sited NATIVE oracle: bin Y co-sited and U / V through the exact forward
/// V-fold chroma oracle to `ow x oh`, then convert ONCE at output width via an
/// identity `Yuv444p` sink (EVEN `sh` only).
#[allow(clippy::too_many_arguments)]
fn top_native_oracle(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Outs {
  let ch = sh / 2;
  let yb = bin_cosited(y, sw, sh, ow, oh);
  let ub = bin_chroma_top(u, sw, ch, ow, oh);
  let vb = bin_chroma_top(v, sw, ch, ow, oh);
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

/// The top-sited ENCODED row-stage oracle: reconstruct U / V to full height with
/// the forward top blend ([`recon_full_top`]) then run that full-resolution
/// `Yuv444p` frame through a `with_native(false)` RGB-domain resample.
#[allow(clippy::too_many_arguments)]
fn encoded_oracle_rgb_top(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Vec<u8> {
  let (uf, vf) = recon_full_top(u, v, sw, sh);
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

/// Direct (non-resample) `Yuv440p` `Top` decode to RGB — the FORWARD-delay kernel
/// path (the already-validated identity Top reference).
fn direct_top_rgb(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize, simd: bool) -> Vec<u8> {
  let mut rgb = vec![0u8; sw * sh * 3];
  {
    let mut sink = MixedSinker::<Yuv440p>::new(sw, sh)
      .with_chroma_location(ChromaLocation::Top)
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap();
    let f = Yuv440pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_native_equals_code_domain_oracle() {
  // The native fast tier bins Y co-sited + U / V through the forward V-fold on the
  // half-height grid (single rounding), for clean 2:1 and fractional ratios.
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 6, 4), (8, 8, 8, 8)] {
    let (y, u, v) = ramp(sw, sh);
    let got = run(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Top, true, true);
    let oracle = top_native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
    assert_eq!(got.0, oracle.0, "rgb {sw}x{sh}->{ow}x{oh}");
    assert_eq!(got.1, oracle.1, "rgba {sw}x{sh}->{ow}x{oh}");
    assert_eq!(got.2, oracle.2, "hsv {sw}x{sh}->{ow}x{oh}");
    assert_eq!(got.3, oracle.3, "luma {sw}x{sh}->{ow}x{oh}");
    assert_eq!(got.4, oracle.4, "luma_u16 {sw}x{sh}->{ow}x{oh}");
    // `TopLeft` (v=0, no horizontal phase) must decode identically.
    let tl = run(
      &y,
      &u,
      &v,
      sw,
      sh,
      ow,
      oh,
      ChromaLocation::TopLeft,
      true,
      true,
    );
    assert_eq!(tl.0, got.0, "TopLeft==Top rgb {sw}x{sh}->{ow}x{oh}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_native_simd_matches_scalar() {
  let (y, u, v) = ramp(8, 8);
  let s = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, true, false);
  let d = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, true, true);
  assert_eq!(
    s, d,
    "top native must be bit-identical across SIMD and scalar"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_encoded_output_equals_rgb_reconstruct_then_bin() {
  // The encoded row-stage tier reconstructs full-height chroma with the forward
  // top blend then bins in RGB — the reconstruct-then-bin oracle.
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 6, 4)] {
    let (y, u, v) = ramp(sw, sh);
    let oracle = encoded_oracle_rgb_top(&y, &u, &v, sw, sh, ow, oh, true);
    let got = run(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Top, false, true);
    assert_eq!(got.0, oracle, "rgb {sw}x{sh}->{ow}x{oh}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_filter_equals_reconstruct_then_filter() {
  // The single-kernel filter tier reconstructs full-height chroma with the forward
  // top blend then Triangle-filters it — equal to feeding the same reconstruction
  // through a Yuv444p Triangle filter, and non-vacuously different from co-sited.
  use crate::resample::{FilteredResampler, Triangle};
  let (sw, sh, ow, oh) = (8usize, 8usize, 4usize, 4usize);
  let (y, u, v) = vramp(sw, sh);
  let filter_440 = |loc: ChromaLocation| -> Vec<u8> {
    let mut rgb = vec![0u8; ow * oh * 3];
    {
      let mut sink = MixedSinker::<Yuv440p, FilteredResampler<Triangle>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, Triangle),
      )
      .unwrap()
      .with_chroma_location(loc.clone())
      .with_simd(true)
      .with_rgb(&mut rgb)
      .unwrap();
      let f = Yuv440pFrame::new(
        &y, &u, &v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
      );
      yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    rgb
  };
  let got = filter_440(ChromaLocation::Top);
  let (uf, vf) = recon_full_top(&u, &v, sw, sh);
  let mut oracle = vec![0u8; ow * oh * 3];
  {
    let mut sink = MixedSinker::<Yuv444p, FilteredResampler<Triangle>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, Triangle),
    )
    .unwrap()
    .with_simd(true)
    .with_rgb(&mut oracle)
    .unwrap();
    let f = Yuv444pFrame::new(
      &y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(got, oracle, "filter-tier top == reconstruct-then-filter");
  assert_ne!(
    got,
    filter_440(ChromaLocation::Left),
    "filter-tier top must differ from co-sited on a vertical ramp"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_linear_equals_reconstruct_then_linear() {
  // The linear-light tier reconstructs full-height chroma with the forward top
  // blend then resamples in linear light — equal to feeding the same
  // reconstruction through a Yuv444p linear-domain resample.
  let (sw, sh, ow, oh) = (8usize, 8usize, 4usize, 4usize);
  let (y, u, v) = ramp(sw, sh);
  let mut got = vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_chroma_location(ChromaLocation::Top)
        .with_rgb(&mut got)
        .unwrap();
    let f = Yuv440pFrame::new(
      &y, &u, &v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let (uf, vf) = recon_full_top(&u, &v, sw, sh);
  let mut oracle = vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut oracle)
        .unwrap();
    let f = Yuv444pFrame::new(
      &y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(got, oracle, "linear-tier top == reconstruct-then-linear");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_encoded_identity_matches_direct_decode() {
  // At identity dimensions the encoded / filter resample tiers reconstruct chroma
  // with the SAME forward top kernel as the direct (non-resample) Yuv440p decode,
  // so routing Top through the resample preserves the decode byte-for-byte —
  // native, row-stage and filter all equal the validated direct Top decode.
  let (y, u, v) = vramp(8, 8);
  let direct = direct_top_rgb(&y, &u, &v, 8, 8, true);
  let row_stage = run(&y, &u, &v, 8, 8, 8, 8, ChromaLocation::Top, false, true);
  assert_eq!(
    row_stage.0, direct,
    "identity encoded resample top == direct decode"
  );
  use crate::resample::{FilteredResampler, Triangle};
  let mut filt = vec![0u8; 8 * 8 * 3];
  {
    let mut sink = MixedSinker::<Yuv440p, FilteredResampler<Triangle>>::with_resampler(
      8,
      8,
      FilteredResampler::new(8, 8, Triangle),
    )
    .unwrap()
    .with_chroma_location(ChromaLocation::Top)
    .with_rgb(&mut filt)
    .unwrap();
    let f = Yuv440pFrame::new(&y, &u, &v, 8, 8, 8, 8, 8);
    yuv440p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(
    filt, direct,
    "identity filter resample top == direct decode"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_differs_from_bottom_and_cosited_on_vertical_ramp() {
  // The v=0 forward fold must MOVE the chroma: on a vertical ramp Top diverges from
  // co-sited AND from the backward-folded Bottom, on BOTH tiers.
  let (y, u, v) = vramp(8, 8);
  for native in [true, false] {
    let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
    let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
    let top = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true);
    assert_ne!(
      top.0, cos.0,
      "top must differ from co-sited (native={native})"
    );
    assert_ne!(
      top.0, bot.0,
      "top must differ from bottom (native={native})"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_equals_cosited_on_flat_chroma() {
  // On constant chroma the forward vertical blend is a no-op, so Top collapses to
  // the co-sited decode byte-for-byte on both tiers.
  let (y, u, v) = flat_chroma(8, 8);
  for native in [true, false] {
    let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
    let top = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true);
    assert_eq!(top.0, cos.0, "flat-chroma top rgb (native={native})");
    assert_eq!(top.2, cos.2, "flat-chroma top hsv (native={native})");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_join_rebuilds_on_top_siting_change_across_frames() {
  // Reuse one native-tier sink flipping Left ⇆ Top (both directions): frame 2 must
  // match a FRESH sink for frame 2's siting — the Top phase is recorded separately
  // from Bottom in the cached join.
  let (y, u, v) = vramp(8, 8);
  for (loc1, loc2) in [
    (ChromaLocation::Left, ChromaLocation::Top),
    (ChromaLocation::Top, ChromaLocation::Left),
    (ChromaLocation::Bottom, ChromaLocation::Top),
    (ChromaLocation::Top, ChromaLocation::Bottom),
  ] {
    let reused = run_reuse_native(&y, &u, &v, loc1.clone(), loc2.clone());
    let fresh = run(&y, &u, &v, 8, 8, 4, 4, loc2.clone(), true, true).0;
    assert_eq!(
      reused, fresh,
      "native reuse {loc1:?}->{loc2:?} must rebuild"
    );
  }
}
