//! RFC #238 S3b — chroma-siting-aware 4:2:0 **resample** for the semi-planar
//! `Nv12` (`U V U V …`) and `Nv21` (`V U V U …`).
//!
//! The 4:2:0 resample twin of the identity-decode `chroma_siting_nv` and the
//! semi-planar sibling of the planar `chroma_siting_420_resample` (`Yuv420p`).
//! S3b routes the HORIZONTAL centered siting (`Center` / `Top` / `Bottom`,
//! [`chroma_420_center_sited_h`](super::super::chroma_420_center_sited_h))
//! through the resample; the VERTICAL chroma pairing stays co-sited
//! (`v_phase = 0`, today's box pairing — `Bottom`'s vertical blend is a later
//! stage). The interleaved chroma is de-interleaved to the SAME half-width U / V
//! planes a `Yuv420p` frame holds, so every centered `Nv12` / `Nv21` resample is
//! **bit-identical** to the centered `Yuv420p` resample of those planes — the
//! strongest catch for a U/V swap in the de-interleave (`swap_uv`).
//!
//! Oracles (independent of the production kernel):
//!  - the **native** tier is pinned to the EXACT code-domain box-average of the
//!    UNROUNDED triangle-reconstructed chroma — a SINGLE rounding, the
//!    user-approved more-correct form — computed in the chroma CODE domain
//!    (never RGB, which would prove the wrong averaging domain), on EVEN source
//!    heights so the luma-domain vertical pairing equals the co-sited box over
//!    the `sh / 2` chroma rows;
//!  - the **encoded row-stage** tier is pinned to the RGB-domain
//!    reconstruct-to-u8-then-bin;
//!  - both tiers are ALSO cross-checked bit-identical against the `Yuv420p`
//!    resample of the de-interleaved planes.

use super::*;
use crate::{
  ChromaLocation, PixelSink,
  resample::{AreaResampler, FilteredResampler, ResampleError, Triangle},
};

const M: ColorMatrix = ColorMatrix::Bt601;
const FR: bool = true;

/// Round `a / d` half-up (ties toward `+∞`) — the production
/// `round_div_half_up`, replicated here so the oracle is independent.
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

/// The EXACT centered chroma oracle for the native tier: reconstruct the
/// `cw x ch` chroma to full width with the #302 `1/4`–`3/4` triangle kept
/// UNROUNDED (scaled ×4 to stay integral: `r ∈ {1, 3, 4}`), then box-average to
/// `ow x oh` — HORIZONTAL over `2·cw`, VERTICAL over the `ch` chroma rows
/// (co-sited: 4:2:0 vertical stays a box pairing) — with a SINGLE round-half-up
/// over `4·(2·cw)·ch`. The code-domain twin the folded
/// [`ResamplePlan::area_chroma_420`] weights realize (for EVEN `sh`).
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

/// A `Yuv420p`/`Nv12`/`Nv21` fixture (`cw = sw / 2`, `ch = sh / 2`) with a strong
/// HORIZONTAL chroma ramp (so the centered triangle, which pulls neighbours,
/// genuinely differs from the co-sited nearest decode) plus a per-row tilt (a
/// vertical mistake would show). `sw` / `sh` must be even.
fn ramp(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let ch = sh / 2;
  let mut y = vec![0u8; sw * sh];
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 3) % 160) as u8;
  }
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (30 + c * 44 + r * 4).min(240) as u8;
      v[r * cw + c] = (230u32.saturating_sub((c * 44 + r * 4) as u32)).max(16) as u8;
    }
  }
  (y, u, v)
}

/// A flat-chroma fixture: the centered phase is a no-op on constant chroma, so
/// centered must equal co-sited. Luma still varies.
fn flat_chroma(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let ch = sh / 2;
  let mut y = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 7) % 170) as u8;
  }
  (y, vec![110u8; cw * ch], vec![140u8; cw * ch])
}

/// Interleave the half-width planar U / V into a 4:2:0 semi-planar chroma plane
/// (`2·cw` bytes per row, `ch` rows): `swap = false` packs Nv12 (`U` at the even
/// byte), `true` packs Nv21 (`V` at the even byte).
fn interleave(u: &[u8], v: &[u8], cw: usize, ch: usize, swap: bool) -> Vec<u8> {
  let full = 2 * cw;
  let mut uv = vec![0u8; full * ch];
  for r in 0..ch {
    for c in 0..cw {
      let (even, odd) = if swap {
        (v[r * cw + c], u[r * cw + c])
      } else {
        (u[r * cw + c], v[r * cw + c])
      };
      uv[r * full + 2 * c] = even;
      uv[r * full + 2 * c + 1] = odd;
    }
  }
  uv
}

type Outs = (
  Vec<u8>,
  Vec<u8>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);

/// The `Yuv420p` twin cross-check: the SAME planar U / V driven through a planar
/// 4:2:0 resample. A centered `Nv12` / `Nv21` decode must equal this
/// byte-for-byte (the de-interleave reconstructs the SAME planes).
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
) -> Outs {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
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
        .with_rgba(&mut rgba)
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
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The centered NATIVE oracle: bin Y co-sited and U / V through the exact
/// centered chroma oracle to `ow x oh`, then convert ONCE at output width via an
/// identity `Yuv444p` sink — the exact ground truth the native tier reproduces
/// (EVEN `sh` only).
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
  let ch = sh / 2;
  let yb = bin_cosited(y, sw, sh, ow, oh);
  let ub = bin_chroma_centered(u, cw, ch, ow, oh);
  let vb = bin_chroma_centered(v, cw, ch, ow, oh);
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
    yuv444p_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The centered ENCODED row-stage oracle: reconstruct U / V to full width with
/// the #302 kernel (u8), replicate each chroma row across its two luma rows (the
/// co-sited vertical pairing), then run that full-resolution `Yuv444p` frame
/// through a `with_native(false)` RGB-domain resample — convert-each-row-then-bin.
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
    let cr = r / 2;
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&u[cr * cw..cr * cw + cw], cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&v[cr * cw..cr * cw + cw], cw));
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
    yuv444p_to(&f, FR, M, &mut sink).unwrap();
  }
  rgb
}

// The full test set is identical bar the format marker, frame type, walker, row
// type, and chroma interleave order, so generate it once per format. Each test
// lands in its own `mod` so the names don't collide.
macro_rules! nv_resample_siting_tests {
  ($mod:ident, $Marker:ident, $Frame:ident, $walker:ident, $Row:ident, $swap:expr) => {
    mod $mod {
      use super::*;

      /// Drive an `Nv12` / `Nv21` area resample (`sw x sh -> ow x oh`) for the
      /// full output set, at `loc` siting and `native` tier — the interleaved
      /// chroma is built from the planar U / V fixture.
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
        let (cw, ch) = (sw / 2, sh / 2);
        let uv = interleave(u, v, cw, ch, $swap);
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgba = vec![0u8; ow * oh * 4];
        let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
        let mut luma = vec![0u8; ow * oh];
        let mut luma_u16 = vec![0u16; ow * oh];
        {
          let mut sink = MixedSinker::<$Marker, AreaResampler>::with_resampler(
            sw,
            sh,
            AreaResampler::to(ow, oh),
          )
          .unwrap()
          .with_native(native)
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
          let f = $Frame::new(y, &uv, sw as u32, sh as u32, sw as u32, sw as u32);
          $walker(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, (hh, ss, vv), luma, luma_u16)
      }

      /// A single-kernel Triangle FILTER resample of the fixture, as this NV
      /// format — the RGB output only.
      #[allow(clippy::too_many_arguments)]
      fn filter_rgb(
        y: &[u8],
        u: &[u8],
        v: &[u8],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
      ) -> Vec<u8> {
        let (cw, ch) = (sw / 2, sh / 2);
        let uv = interleave(u, v, cw, ch, $swap);
        let mut rgb = vec![0u8; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$Marker, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_chroma_location(loc)
          .with_rgb(&mut rgb)
          .unwrap();
          let f = $Frame::new(y, &uv, sw as u32, sh as u32, sw as u32, sw as u32);
          $walker(&f, FR, M, &mut sink).unwrap();
        }
        rgb
      }

      /// Reuse ONE native-tier sink across two frames of the SAME content, siting
      /// `loc1` then `loc2` (via `set_chroma_location` between walks).
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
      ) -> Outs {
        let (cw, ch) = (sw / 2, sh / 2);
        let uv = interleave(u, v, cw, ch, $swap);
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgba = vec![0u8; ow * oh * 4];
        let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
        let mut luma = vec![0u8; ow * oh];
        let mut luma_u16 = vec![0u16; ow * oh];
        {
          let mut sink = MixedSinker::<$Marker, AreaResampler>::with_resampler(
            sw,
            sh,
            AreaResampler::to(ow, oh),
          )
          .unwrap()
          .with_native(true)
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
          let f = $Frame::new(y, &uv, sw as u32, sh as u32, sw as u32, sw as u32);
          sink.set_chroma_location(loc1);
          $walker(&f, FR, M, &mut sink).unwrap();
          sink.set_chroma_location(loc2);
          $walker(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, (hh, ss, vv), luma, luma_u16)
      }

      // ---- co-sited byte-identity (the regression contract) ----------------

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
            ChromaLocation::Unknown(7),
          ] {
            let got = run(&y, &u, &v, 8, 8, 4, 4, loc, native, true);
            assert_eq!(got.0, base.0, "rgb {loc:?} native={native}");
            assert_eq!(got.1, base.1, "rgba {loc:?} native={native}");
            assert_eq!(got.2, base.2, "hsv {loc:?} native={native}");
            assert_eq!(got.3, base.3, "luma {loc:?} native={native}");
            assert_eq!(got.4, base.4, "luma_u16 {loc:?} native={native}");
          }
        }
      }

      // ---- centered native == the exact code-domain oracle -----------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_native_equals_code_domain_oracle() {
        // Clean 2:1 and fractional ratios (EVEN source height so the vertical
        // luma-domain pairing equals the co-sited chroma box), for the whole
        // centered group.
        for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
          let (y, u, v) = ramp(sw, sh);
          let o = native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
          for loc in [
            ChromaLocation::Center,
            ChromaLocation::Top,
            ChromaLocation::Bottom,
          ] {
            let n = run(&y, &u, &v, sw, sh, ow, oh, loc, true, true);
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

      // ---- centered encoded row-stage == RGB-domain reconstruct-then-bin ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_encoded_output_equals_rgb_reconstruct_then_bin() {
        for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 6, 4)] {
          let (y, u, v) = ramp(sw, sh);
          let oracle = encoded_oracle_rgb(&y, &u, &v, sw, sh, ow, oh, true);
          for loc in [
            ChromaLocation::Center,
            ChromaLocation::Top,
            ChromaLocation::Bottom,
          ] {
            let got = run(&y, &u, &v, sw, sh, ow, oh, loc, false, true);
            assert_eq!(got.0, oracle, "rgb {loc:?} {sw}x{sh}->{ow}x{oh}");
          }
        }
      }

      // ---- centered NV is bit-identical to centered Yuv420p (U/V-swap catch) --

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_equals_centered_yuv420p_area_tiers() {
        // De-interleaving the packed chroma back to planar U / V reconstructs the
        // SAME planes the planar twin holds, so a centered NV area resample is
        // byte-identical to the centered Yuv420p resample of those planes — on
        // BOTH the native fast tier and the encoded row-stage tier. The strongest
        // catch for a U/V swap in the de-interleave (`swap_uv`). Bt601 keeps this
        // on the shared matrix-tag path (ChromaDerivedNcl is the lone divergence).
        for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
          let (y, u, v) = ramp(sw, sh);
          // Center / Top keep the vertical pairing co-sited, so NV == Yuv420p.
          // Bottom is excluded here: Yuv420p now folds its vertical v=1 phase
          // through the resample while the semi-planar NV formats still co-site
          // it, so the two diverge on Bottom until the NV vertical fold lands.
          for loc in [ChromaLocation::Center, ChromaLocation::Top] {
            for native in [true, false] {
              let nv = run(&y, &u, &v, sw, sh, ow, oh, loc, native, true);
              let yuv420p = run_yuv420p(&y, &u, &v, sw, sh, ow, oh, loc, native, true);
              assert_eq!(
                nv, yuv420p,
                "NV vs Yuv420p {loc:?} native={native} {sw}x{sh}->{ow}x{oh}"
              );
            }
          }
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_equals_centered_yuv420p_filter_tier() {
        // The filter tier's centered reconstruction is the SAME as the planar
        // twin's, so the Triangle-filtered centered NV RGB equals the Yuv420p one;
        // the centered result also genuinely differs from co-sited on the ramp.
        for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3)] {
          let (y, u, v) = ramp(sw, sh);
          let cw = sw / 2;
          let mut rgb420 = vec![0u8; ow * oh * 3];
          {
            let mut sink = MixedSinker::<Yuv420p, FilteredResampler<Triangle>>::with_resampler(
              sw,
              sh,
              FilteredResampler::new(ow, oh, Triangle),
            )
            .unwrap()
            .with_chroma_location(ChromaLocation::Center)
            .with_rgb(&mut rgb420)
            .unwrap();
            let f = Yuv420pFrame::new(
              &y, &u, &v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
            );
            yuv420p_to(&f, FR, M, &mut sink).unwrap();
          }
          let nv = filter_rgb(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Center);
          assert_eq!(nv, rgb420, "filter centered {sw}x{sh}->{ow}x{oh}");
          let cosited = filter_rgb(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Left);
          assert_ne!(nv, cosited, "filter centered must differ from co-sited");
        }
      }

      // ---- non-vacuous + flat-chroma sanity --------------------------------

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

      // ---- cross-frame sink reuse rebuilds the phased native join ----------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_join_rebuilds_on_siting_change_across_frames() {
        // Reuse one native-tier sink flipping Left ⇄ Center (both directions):
        // frame 2 must match a FRESH sink for frame 2's siting — no stale-phase
        // carryover from the cached folded/co-sited chroma plan.
        let (y, u, v) = ramp(8, 8);
        for (a, b) in [
          (ChromaLocation::Left, ChromaLocation::Center),
          (ChromaLocation::Center, ChromaLocation::Left),
        ] {
          let reused = run_reuse_native(&y, &u, &v, 8, 8, 4, 4, a, b);
          let fresh = run(&y, &u, &v, 8, 8, 4, 4, b, true, true);
          assert_eq!(
            reused.0, fresh.0,
            "native rgb {a:?}->{b:?} stale-phase carryover"
          );
          assert_eq!(reused.2, fresh.2, "native hsv {a:?}->{b:?}");
          assert_eq!(reused.3, fresh.3, "native luma {a:?}->{b:?}");
          let stale = run(&y, &u, &v, 8, 8, 4, 4, a, true, true);
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
      fn native_out_of_sequence_first_row_does_not_drop_the_cached_join() {
        // The point-of-use phase drop is state-atomic: a reused sink, new frame,
        // changed siting, then an OUT-OF-SEQUENCE first process call must be
        // rejected (OutOfSequenceRow) with the cached join INTACT, so a corrected
        // row-0 retry rebuilds cleanly.
        let (y, u, v) = ramp(8, 8);
        let uv = interleave(&u, &v, 4, 4, $swap);
        let mut rgb = vec![0u8; 4 * 4 * 3];
        let mut sink =
          MixedSinker::<$Marker, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
            .unwrap()
            .with_native(true)
            .with_chroma_location(ChromaLocation::Left)
            .with_rgb(&mut rgb)
            .unwrap();
        // Frame 1 at Left builds the native join (chroma row `r / 2` per luma row).
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        for r in 0..8 {
          let row = $Row::new(
            &y[r * 8..r * 8 + 8],
            &uv[(r / 2) * 8..(r / 2) * 8 + 8],
            r,
            M,
            FR,
          );
          PixelSink::process(&mut sink, row).unwrap();
        }
        assert!(sink.native_420.is_some(), "frame 1 builds the native join");
        // Frame 2: change siting to Center, then feed an OUT-OF-SEQUENCE first row.
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        sink.set_chroma_location(ChromaLocation::Center);
        let bad = $Row::new(&y[4 * 8..5 * 8], &uv[2 * 8..3 * 8], 4, M, FR);
        let err = PixelSink::process(&mut sink, bad).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "out-of-sequence first row must be OutOfSequenceRow, got {err:?}"
        );
        assert!(
          sink.native_420.is_some(),
          "an out-of-sequence first row must NOT drop the cached native join"
        );
        // The corrected retry (row 0, now rebuilding for Center) succeeds.
        for r in 0..8 {
          let row = $Row::new(
            &y[r * 8..r * 8 + 8],
            &uv[(r / 2) * 8..(r / 2) * 8 + 8],
            r,
            M,
            FR,
          );
          PixelSink::process(&mut sink, row).unwrap();
        }
      }

      // ---- IN-SEQUENCE mid-frame phase change is rejected (not silently mixed) --

      /// Drive one frame: `begin_frame`, accept row 0 at `loc1` (freezes the
      /// phase), flip to `loc2`, then feed the IN-SEQUENCE row 1 and return its
      /// `process` result. In 4:2:0 rows 0 and 1 SHARE chroma row 0 (`uv[0..8]`).
      fn in_sequence_flip_row1<R>(
        mut sink: MixedSinker<'_, $Marker, R>,
        y: &[u8],
        uv: &[u8],
        loc1: ChromaLocation,
        loc2: ChromaLocation,
      ) -> Result<(), MixedSinkerError> {
        sink.set_chroma_location(loc1);
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        PixelSink::process(&mut sink, $Row::new(&y[0..8], &uv[0..8], 0, M, FR)).unwrap();
        sink.set_chroma_location(loc2);
        PixelSink::process(&mut sink, $Row::new(&y[8..16], &uv[0..8], 1, M, FR))
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn in_sequence_mid_frame_phase_change_rejected_across_tiers() {
        // The effective siting is frozen on the first output-bearing row; a later
        // IN-SEQUENCE row observing a different phase (after a mid-frame
        // `set_chroma_location`) passes the sequence preflight but must be rejected
        // with ChromaSitingChanged, uniformly across tiers — else the frame bins a
        // mixture of co-sited and centered chroma.
        let (y, u, v) = ramp(8, 8);
        let uv = interleave(&u, &v, 4, 4, $swap);
        for (loc1, loc2) in [
          (ChromaLocation::Center, ChromaLocation::Left),
          (ChromaLocation::Left, ChromaLocation::Center),
        ] {
          // Native fast tier.
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink =
            MixedSinker::<$Marker, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(true)
              .with_rgb(&mut rgb)
              .unwrap();
          let err = in_sequence_flip_row1(sink, &y, &uv, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "native {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Encoded row-stage RGB tier (`with_native(false)`).
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink =
            MixedSinker::<$Marker, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(false)
              .with_rgb(&mut rgb)
              .unwrap();
          let err = in_sequence_flip_row1(sink, &y, &uv, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "encoded-rgb {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // HSV-only row-stage tier (no separate hsv-direct join for NV 4:2:0).
          let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
          let sink =
            MixedSinker::<$Marker, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(false)
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
          let err = in_sequence_flip_row1(sink, &y, &uv, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "hsv {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Filter tier (single-kernel Triangle FilteredResampler).
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink = MixedSinker::<$Marker, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
          let err = in_sequence_flip_row1(sink, &y, &uv, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_mid_frame_phase_change_rejection_keeps_the_stream_retryable() {
        // Advance rows 0,1 (Center), then flip siting mid-frame (Left): the
        // frozen-phase CHECK rejects it with ChromaSitingChanged at the choke
        // point; a mixed-phase frame is never emitted and the frame restarts
        // cleanly (a rejected row mutates no state).
        let (y, u, v) = ramp(8, 8);
        let uv = interleave(&u, &v, 4, 4, $swap);
        let mut rgb = vec![0u8; 4 * 4 * 3];
        let mut sink =
          MixedSinker::<$Marker, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
            .unwrap()
            .with_native(true)
            .with_chroma_location(ChromaLocation::Center)
            .with_rgb(&mut rgb)
            .unwrap();
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        for r in 0..2 {
          let row = $Row::new(
            &y[r * 8..r * 8 + 8],
            &uv[(r / 2) * 8..(r / 2) * 8 + 8],
            r,
            M,
            FR,
          );
          PixelSink::process(&mut sink, row).unwrap();
        }
        sink.set_chroma_location(ChromaLocation::Left);
        let bad = $Row::new(&y[2 * 8..3 * 8], &uv[8..16], 2, M, FR);
        let err = PixelSink::process(&mut sink, bad).unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
          "mid-frame siting change must be ChromaSitingChanged, got {err:?}"
        );
        // The rejected row mutated no stream state: begin_frame restarts cleanly
        // and a fresh frame at the new siting processes without error.
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        for r in 0..8 {
          let row = $Row::new(
            &y[r * 8..r * 8 + 8],
            &uv[(r / 2) * 8..(r / 2) * 8 + 8],
            r,
            M,
            FR,
          );
          PixelSink::process(&mut sink, row).unwrap();
        }
      }
    }
  };
}

nv_resample_siting_tests!(nv12, Nv12, Nv12Frame, nv12_to, Nv12Row, false);
nv_resample_siting_tests!(nv21, Nv21, Nv21Frame, nv21_to, Nv21Row, true);
