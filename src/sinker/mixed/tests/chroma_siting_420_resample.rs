//! RFC #238 S3a — chroma-siting-aware 4:2:0 **resample** for `Yuv420p`.
//!
//! 4:2:0 subsamples chroma 2:1 horizontally AND vertically. The HORIZONTAL
//! centered siting (`Center` / `Top` / `Bottom`,
//! [`chroma_420_center_sited_h`](super::super::chroma_420_center_sited_h)) is
//! routed through the resample so a downscale keeps the correct horizontal
//! chroma phase; `Bottom`
//! ([`chroma_420_bottom_sited_v`](super::super::chroma_420_bottom_sited_v))
//! additionally folds the VERTICAL `v = 1` triangle through the resample (its
//! even output row box-blends the previous chroma row), while `Center` / `Top`
//! keep the vertical pairing co-sited (`v_phase = 0`):
//!  - the **native fast tier** folds the #302 `1/4`–`3/4` triangle into the
//!    chroma area weights ([`ResamplePlan::area_chroma_420`]) — one SINGLE-
//!    rounding phased box-average on the subsampled grid;
//!  - the **encoded row-stage tier** (`with_native(false)`) reconstructs
//!    full-width chroma then bins in RGB.
//!
//! The co-sited / unspecified group stays phase 0, byte-identical to the
//! pre-siting resample (the folded form at phase 0 = the plain box overlaps).
//!
//! ★ Oracle (native tier): the EXACT code-domain box-average of the UNROUNDED
//! triangle-reconstructed chroma — a SINGLE rounding — pinned against a
//! YUV-domain oracle (never the RGB-domain one, which would prove the wrong
//! averaging domain). 4:2:0 has no 4:2:2 sibling to cross-check bit-identically,
//! so the native tier is pinned to this code-domain YUV oracle and the encoded
//! row-stage tier to the RGB-domain reconstruct-then-bin. The oracle uses
//! EVEN source heights so the luma-domain vertical pairing
//! ([`AxisSpans::area_halved`]) equals the co-sited box over the `sh / 2` chroma
//! rows (its `2·` scale cancels the plan's `src_h = sh`).

use crate::{
  ChromaLocation, ColorInfo, ColorSpec, DynamicRange, KernelMatrix, PixelFormat, PixelSink,
  Primaries, Transfer,
  resample::{AreaResampler, AveragingDomain, LinearMode},
  sinker::MixedSinker,
  source::{Yuv420p, Yuv420pRow, Yuv444p, yuv420p_to, yuv444p_to},
};
use mediaframe::frame::{Yuv420pFrame, Yuv444pFrame};

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
/// `ow x oh` (round-half-up) — the reference for a phase-free plane (luma).
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
/// UNROUNDED (scaled ×4 to stay integral: `r ∈ {1, 3, 4}`), then box-average
/// to `ow x oh` — HORIZONTAL over `2·cw`, VERTICAL over the `ch` chroma rows
/// (co-sited: 4:2:0 vertical stays a box pairing) — with a SINGLE
/// round-half-up over `4·(2·cw)·ch`. This is the code-domain twin the folded
/// [`ResamplePlan::area_chroma_420`] weights realize (for EVEN `sh`, where the
/// plan's luma-domain `area_halved(sh, oh)` V axis is exactly `2·` the box over
/// `ch = sh / 2`, cancelled by its `src_h = sh`).
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
/// round-half-up to u8) — the RGB-domain oracle's reconstruction step. Matches
/// [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h).
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

/// A `Yuv420p` fixture (`cw = sw / 2`, `ch = sh / 2`) with a strong HORIZONTAL
/// chroma ramp (so the centered triangle, which pulls neighbours, genuinely
/// differs from the co-sited nearest decode) plus a per-row tilt (a vertical
/// mistake would show). `sw` / `sh` must be even.
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

/// A flat-chroma fixture: the centered phase is a no-op on constant chroma
/// (the triangle of a constant is that constant), so centered must equal
/// co-sited. Luma still varies.
fn flat_chroma(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let ch = sh / 2;
  let mut y = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 7) % 170) as u8;
  }
  (y, vec![110u8; cw * ch], vec![140u8; cw * ch])
}

type Outs = (
  Vec<u8>,
  Vec<u8>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);

/// Drive a `Yuv420p` area resample (`sw x sh -> ow x oh`) for the full output
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
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
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
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The centered NATIVE oracle: bin Y co-sited and U / V through the exact
/// centered chroma oracle to `ow x oh`, then convert ONCE at output width via
/// an identity `Yuv444p` sink — the exact ground truth the native tier
/// reproduces byte-for-byte (EVEN `sh` only).
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
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The centered ENCODED row-stage oracle: reconstruct U / V to full width with
/// the #302 kernel (u8) and replicate each chroma row across its two luma rows
/// (the co-sited vertical pairing), then run that full-resolution `Yuv444p`
/// frame through a `with_native(false)` RGB-domain resample — i.e.
/// convert-each-row-then-bin-RGB, exactly what the `Yuv420p` encoded arm does
/// (each luma row decodes its chroma row `idx / 2`, reconstructed horizontally).
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
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

/// The EXACT bottom-sited (`v = 1`) chroma oracle for the native tier: the
/// centered horizontal `1/4`–`3/4` triangle (×4, as [`bin_chroma_centered`])
/// composed with the vertical `v = 1` triangle (×2 — even luma row `2i`
/// box-blends chroma rows `{i - 1, i}` with weights `{1, 1}`, odd row `2i + 1`
/// takes chroma row `i` with weight 2, top-edge clamp), the combined ×8
/// UNROUNDED reconstruction box-averaged to `ow x oh` with a SINGLE
/// round-half-up. This is the code-domain twin the folded
/// [`ResamplePlan::area_chroma_420`] realizes when its V axis is
/// [`AxisSpans::area_chroma_phased_v`] (EVEN `sh` only, `ch = sh / 2`).
fn bin_chroma_bottom(c: &[u8], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u8> {
  let full = 2 * cw;
  let sh = 2 * ch;
  // ×8 reconstruction plane (`full x sh`): the vertical v=1 triangle (×2) then
  // the centered horizontal triangle (×4), both kept UNROUNDED so the box
  // below applies the one rounding.
  let mut r8 = vec![0u64; full * sh];
  for r in 0..sh {
    let cr = r / 2;
    let prev = cr.saturating_sub(1);
    let vrow: Vec<u32> = (0..cw)
      .map(|j| {
        if r & 1 == 0 {
          u32::from(c[prev * cw + j]) + u32::from(c[cr * cw + j]) // even: {1, 1}
        } else {
          2 * u32::from(c[cr * cw + j]) // odd: {2}
        }
      })
      .collect();
    for j in 0..cw {
      let l = vrow[j.saturating_sub(1)];
      let m = vrow[j];
      let rt = vrow[if j + 1 < cw { j + 1 } else { j }];
      r8[r * full + 2 * j] = u64::from(l + 3 * m); // even col: (v[j-1] + 3·v[j])
      r8[r * full + 2 * j + 1] = u64::from(3 * m + rt); // odd col: (3·v[j] + v[j+1])
    }
  }
  let hw = area_weights(full, ow);
  let vw = area_weights(sh, oh);
  let denom = (8 * full * sh) as u64; // ×8 (×2 V, ×4 H) × the box normalization
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * r8[(vs + dy) * full + hs + dx];
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
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
  let cw = sw / 2;
  let ch = sh / 2;
  let yb = bin_cosited(y, sw, sh, ow, oh);
  let ub = bin_chroma_bottom(u, cw, ch, ow, oh);
  let vb = bin_chroma_bottom(v, cw, ch, ow, oh);
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

/// Reconstruct `Yuv420p` chroma to full resolution (`sw x sh`) for the
/// bottom-sited (`v = 1`) decode — the identity bottom kernel at source width:
/// per luma row the even rows vertically box-blend chroma rows `i - 1`
/// (clamped) and `i` (round-half-up), the odd rows take chroma row `i`, each
/// then horizontally upsampled with the #302 centered `1/4`–`3/4` kernel. The
/// shared reconstruction step for every reconstruct-then-bin bottom oracle.
fn recon_full_bottom(u: &[u8], v: &[u8], sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let vblend = |plane: &[u8], cr: usize, prev: usize| -> Vec<u8> {
    (0..cw)
      .map(|c| {
        let a = u32::from(plane[prev * cw + c]);
        let b = u32::from(plane[cr * cw + c]);
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
        u[cr * cw..cr * cw + cw].to_vec(),
        v[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&uh, cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&vh, cw));
  }
  (uf, vf)
}

/// The bottom-sited ENCODED row-stage oracle: reconstruct U / V to full width
/// with the vertical bottom blend ([`recon_full_bottom`]) then run that
/// full-resolution `Yuv444p` frame through a `with_native(false)` RGB-domain
/// resample (convert-each-row-then-bin-RGB, exactly what the `Yuv420p` encoded
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

/// Direct (non-resample) `Yuv420p` `Bottom` decode to RGB — the delay-line
/// kernel path, the already-validated identity Bottom reference the resample
/// path must match at identity dimensions.
fn direct_bottom_rgb(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize, simd: bool) -> Vec<u8> {
  let cw = sw / 2;
  let mut rgb = vec![0u8; sw * sh * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(sw, sh)
      .with_chroma_location(ChromaLocation::Bottom)
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

/// A `Yuv420p` fixture with flat luma and a strong per-ROW chroma step (flat
/// across columns), so the vertical bottom fold is observable in isolation: a
/// horizontal-only siting leaves it untouched, the `v = 1` blend visibly moves
/// it. `sw` / `sh` must be even.
fn vramp(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let ch = sh / 2;
  let y = vec![128u8; sw * sh];
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (20 + r * 40).min(240) as u8;
      v[r * cw + c] = (220u32.saturating_sub((r * 40) as u32)).max(16) as u8;
    }
  }
  (y, u, v)
}

// ---- co-sited byte-identity (the regression contract) ----------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn cosited_group_is_byte_identical_across_tiers() {
  // Every FULLY co-sited / unspecified siting must produce the byte-identical
  // pre-siting resample (phase 0 → the folded plan is never built), on BOTH
  // tiers. `Unspecified` is the baseline. `BottomLeft` (`v = 1`) and `TopLeft`
  // (`v = 0`, the FORWARD fold) are EXCLUDED: both are co-sited horizontally but
  // VERTICALLY sited, so their folded plans ARE built and they are pinned to
  // their own oracles below.
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
    for loc in [ChromaLocation::Left, ChromaLocation::other("unassigned-7")] {
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
  // Clean 2:1 and fractional ratios (EVEN source height so the vertical
  // luma-domain pairing equals the co-sited chroma box), for the ONE
  // vertically-co-sited centered siting (Center). Top / Bottom fold the vertical
  // phase and are pinned to their own oracles below.
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
    let (y, u, v) = ramp(sw, sh);
    let o = native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
    let loc = ChromaLocation::Center;
    let n = run(&y, &u, &v, sw, sh, ow, oh, loc.clone(), true, true);
    assert_eq!(n.0, o.0, "rgb {loc:?} {sw}x{sh}->{ow}x{oh}");
    assert_eq!(n.1, o.1, "rgba {loc:?} {sw}x{sh}->{ow}x{oh}");
    assert_eq!(n.2, o.2, "hsv {loc:?} {sw}x{sh}->{ow}x{oh}");
    assert_eq!(n.3, o.3, "luma {loc:?} {sw}x{sh}->{ow}x{oh}");
    assert_eq!(n.4, o.4, "luma_u16 {loc:?} {sw}x{sh}->{ow}x{oh}");
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
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 6, 4)] {
    let (y, u, v) = ramp(sw, sh);
    let oracle = encoded_oracle_rgb(&y, &u, &v, sw, sh, ow, oh, true);
    let loc = ChromaLocation::Center;
    let got = run(&y, &u, &v, sw, sh, ow, oh, loc.clone(), false, true);
    assert_eq!(got.0, oracle, "rgb {loc:?} {sw}x{sh}->{ow}x{oh}");
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

// ---- bottom-sited (v = 1) vertical fold ------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_native_equals_code_domain_oracle() {
  // The native tier folds the vertical v=1 triangle into the chroma area
  // weights; its output is the EXACT code-domain box-average of the UNROUNDED
  // H⊗V reconstruction (single rounding), for clean 2:1 and fractional ratios
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

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_encoded_output_equals_rgb_reconstruct_then_bin() {
  // The encoded row-stage tier reconstructs full-width chroma with the vertical
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
fn bottom_differs_from_center_on_a_vertical_chroma_ramp() {
  // The v=1 fold must actually MOVE the chroma: on a purely-vertical chroma
  // ramp Bottom diverges from the vertically-co-sited Center on BOTH tiers.
  let (y, u, v) = vramp(8, 8);
  for native in [true, false] {
    let cen = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
    let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
    assert_ne!(
      bot.0, cen.0,
      "bottom rgb must differ from center (native={native})"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_equals_cosited_on_flat_chroma() {
  // On constant chroma the vertical blend (and the horizontal triangle) are
  // no-ops, so Bottom collapses to the co-sited decode byte-for-byte.
  let (y, u, v) = flat_chroma(8, 8);
  for native in [true, false] {
    let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
    let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
    assert_eq!(bot.0, cos.0, "flat-chroma bottom rgb (native={native})");
    assert_eq!(bot.2, cos.2, "flat-chroma bottom hsv (native={native})");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_encoded_identity_matches_direct_decode() {
  // At identity dimensions (out == src) the encoded resample tier reconstructs
  // chroma with the SAME bottom kernel as the direct (non-resample) Yuv420p
  // decode and bins with pass-through area weights, so routing Bottom through
  // the resample preserves the decode byte-for-byte.
  let (y, u, v) = vramp(8, 8);
  let res = run(&y, &u, &v, 8, 8, 8, 8, ChromaLocation::Bottom, false, true);
  let direct = direct_bottom_rgb(&y, &u, &v, 8, 8, true);
  assert_eq!(
    res.0, direct,
    "identity encoded resample bottom == direct decode"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_linear_final_even_row_tail_alloc_failure_retries_with_blend() {
  // ODD source height => the FINAL luma row is EVEN, so its Bottom vertical
  // blend reads the PREDECESSOR chroma row. If the linear-light tail allocation
  // fails on that final row, the retry must STILL blend with the predecessor
  // (not clamp to the current chroma row): the sited reconstruction reads the
  // lookback but must NOT advance it before the fallible commit accepts the row
  // (deferred staging, #180). Without the fix the failed attempt advances the
  // lookback, so the retry clamps and silently diverges.
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  let (sw, sh, ow, oh) = (8usize, 7usize, 4usize, 3usize);
  let cw = sw / 2;
  let ch = sh.div_ceil(2);
  // Flat luma, strong per-row chroma step so the blend differs from the clamp.
  let y = vec![128u8; sw * sh];
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (20 + r * 40).min(240) as u8;
      v[r * cw + c] = (220u32.saturating_sub((r * 40) as u32)).max(16) as u8;
    }
  }
  let run_frame = |arm_fail: bool, loc: ChromaLocation| -> Vec<u8> {
    let mut rgb = vec![0u8; ow * oh * 3];
    {
      let mut sink =
        MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
          .unwrap()
          .with_averaging_domain(AveragingDomain::Linear)
          .with_chroma_location(loc.clone())
          .with_rgb(&mut rgb)
          .unwrap();
      sink.begin_frame(sw as u32, sh as u32).unwrap();
      let feed = |sink: &mut MixedSinker<'_, Yuv420p, AreaResampler>, r: usize| {
        let yr = &y[r * sw..(r + 1) * sw];
        let cr = r / 2;
        let ur = &u[cr * cw..(cr + 1) * cw];
        let vr = &v[cr * cw..(cr + 1) * cw];
        sink.process(Yuv420pRow::for_tests(yr, ur, vr, r, M, FR))
      };
      for r in 0..sh - 1 {
        feed(&mut sink, r).unwrap();
      }
      if arm_fail {
        crate::sinker::mixed::linear_light::arm_linear_tail_alloc_failure();
        let err = feed(&mut sink, sh - 1).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "armed final-row tail alloc must surface AllocationFailed, got {err:?}"
        );
      }
      // Retry the SAME final row (the failpoint is one-shot, already taken).
      feed(&mut sink, sh - 1).unwrap();
    }
    rgb
  };
  let reference = run_frame(false, ChromaLocation::Bottom);
  let retried = run_frame(true, ChromaLocation::Bottom);
  assert_eq!(
    retried, reference,
    "the post-failure retry of the final EVEN row must blend with the predecessor (not clamp)"
  );
  // Non-vacuous: the bottom vertical blend is genuinely active on the final even
  // row, so the reference differs from the vertically-co-sited Center decode.
  assert_ne!(
    reference,
    run_frame(false, ChromaLocation::Center),
    "the bottom vertical blend must move the final even row vs co-sited Center"
  );
}

// ---- the ≤1 LSB single-rounding note, pinned -------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_native_is_within_1_lsb_of_reconstruct_then_bin() {
  // The folded single-rounding native output and the #302 reconstruct-to-u8-
  // then-bin (TWO roundings) agree to ≤ 1 LSB per chroma sample, compared in
  // the chroma CODE domain. `[0, 2, 0, 2]` chroma is a crafted case that
  // provably exercises the divergence: the odd-column reconstruction lands on
  // an exact `.5` (`(3·0 + 2)/4`), which the folded single rounding averages
  // down to 0 while the intermediate `>>2` rounds it up to 1 first.
  let (cw, ch, ow, oh) = (4usize, 2usize, 4usize, 2usize);
  let u: Vec<u8> = (0..cw * ch)
    .map(|i| if i.is_multiple_of(2) { 0 } else { 2 })
    .collect();
  let folded = bin_chroma_centered(&u, cw, ch, ow, oh);
  // reconstruct-then-bin: #302 to u8 per row, then co-sited box-average.
  let mut recon = vec![0u8; 2 * cw * ch];
  for r in 0..ch {
    recon[r * 2 * cw..r * 2 * cw + 2 * cw]
      .copy_from_slice(&recon_full_row(&u[r * cw..r * cw + cw], cw));
  }
  let double = bin_cosited(&recon, 2 * cw, ch, ow, oh);
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
  assert_ne!(folded, double, "the ≤1 LSB gap must be exercised");
}

// ---- cross-frame sink reuse rebuilds the phased join (RFC #238) -------------
//
// The native / HSV-only joins cache a chroma plan built for ONE frame's siting
// and are only `reset` between frames; a reused sink whose `chroma_location`
// changed to a different phase must REBUILD the join, else frame 2 inherits
// frame 1's (folded centered ⇄ unscaled co-sited) weights.

/// Reuse ONE full-output native-tier sink across two frames of the SAME
/// content, siting `loc1` then `loc2`, returning frame 2's outputs.
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
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
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
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    sink.set_chroma_location(loc1.clone());
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_chroma_location(loc2.clone());
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
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
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
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
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_simd(simd)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    sink.set_chroma_location(loc1.clone());
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_chroma_location(loc2.clone());
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
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
    // RFC #238: Top shares Center's horizontal phase but folds the v=0 forward
    // triangle, so a Center<->Top reuse must REBUILD the join (else frame 2
    // inherits frame 1's vertically-co-sited weights); Top<->Bottom flips only
    // the vertical fold direction.
    (ChromaLocation::Center, ChromaLocation::Top),
    (ChromaLocation::Top, ChromaLocation::Center),
    (ChromaLocation::Top, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Top),
  ] {
    let reused = run_reuse_native(&y, &u, &v, 8, 8, 4, 4, a.clone(), b.clone(), true);
    let fresh = run(&y, &u, &v, 8, 8, 4, 4, b.clone(), true, true);
    assert_eq!(
      reused.0, fresh.0,
      "native rgb {a:?}->{b:?} stale-phase carryover"
    );
    assert_eq!(reused.1, fresh.1, "native rgba {a:?}->{b:?}");
    assert_eq!(reused.2, fresh.2, "native hsv {a:?}->{b:?}");
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

/// Apply the new siting via one of the two setters (both funnel to
/// `self.chroma_location`, the field the point-of-use check reads).
fn apply_siting<R>(
  sink: &mut MixedSinker<'_, Yuv420p, R>,
  loc: ChromaLocation,
  via_color_spec: bool,
) {
  if via_color_spec {
    let spec = ColorSpec::from_info(
      PixelFormat::Yuv420p,
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

/// Feed all `sh` rows of the frame in order (each luma row `r` reads chroma row
/// `r / 2`). Concrete to the [`AreaResampler`] sink every caller uses.
fn feed_all(
  sink: &mut MixedSinker<'_, Yuv420p, AreaResampler>,
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
) {
  let cw = sw / 2;
  for r in 0..sh {
    let cr = r / 2;
    let row = Yuv420pRow::for_tests(
      &y[r * sw..r * sw + sw],
      &u[cr * cw..cr * cw + cw],
      &v[cr * cw..cr * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(sink, row).unwrap();
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
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
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
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    sink.set_chroma_location(loc1.clone());
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
    apply_siting(&mut sink, loc2, via_color_spec); // AFTER begin_frame, before row 0
    feed_all(&mut sink, y, u, v, sw, sh);
  }
  (rgb, rgba, (hh, ss, vv), luma, luma_u16)
}

/// The HSV-only twin of [`run_reuse_native_setter_after`].
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
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_simd(simd)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    sink.set_chroma_location(loc1.clone());
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
    apply_siting(&mut sink, loc2, via_color_spec);
    feed_all(&mut sink, y, u, v, sw, sh);
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
  // AllocationFailed). `with_native(false)` forces the encoded convert path.
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  super::super::arm_chroma_full_alloc_failure();
  // First process call is row 5 — the stream expects row 0.
  let bad = Yuv420pRow::for_tests(
    &y[5 * 8..6 * 8],
    &u[2 * cw..3 * cw],
    &v[2 * cw..3 * cw],
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
  let good = Yuv420pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
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
  // A luma-only centered area resample never calls the RGB converter, so it
  // must NOT reserve/reconstruct chroma: with the failpoint armed it still
  // succeeds (an unfixed path would reserve and mask the luma output).
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut luma = vec![0u8; 4 * 4];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(false)
        .with_chroma_location(ChromaLocation::Center)
        .with_luma(&mut luma)
        .unwrap();
    PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
    super::super::arm_chroma_full_alloc_failure();
    feed_all(&mut sink, &y, &u, &v, 8, 8);
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
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  let f = Yuv420pFrame::new(&y, &u, &v, 8, 8, 8, cw as u32, cw as u32);
  let _ = yuv420p_to(&f, FR, sink.set_kernel_matrix(M));
}

// ---- centered LINEAR folds the chroma phase (both decodes) ------------------

/// A `Yuv420p` linear-light area resample (`with_native(false)`) to RGB, at
/// `loc` siting and `mode`.
fn run_linear_420(
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
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_native(false)
        .with_linear_mode(mode)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

/// The centered-Linear oracle: reconstruct U / V to full width with the #302
/// kernel (u8, chroma row `idx / 2` per luma row), then run that full-res
/// `Yuv444p` frame through the SAME linear-light resample — i.e.
/// reconstruct-then-linear-average.
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
    let cr = r / 2;
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&u[cr * cw..cr * cw + cw], cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&v[cr * cw..cr * cw + cw], cw));
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
    let centered = run_linear_420(&y, &u, &v, ChromaLocation::Center, mode, true);
    let oracle = oracle_linear_reconstruct(&y, &u, &v, mode, true);
    assert_eq!(
      centered, oracle,
      "centered Linear ({mode:?}) must equal reconstruct-then-linear-average"
    );
    let cosited = run_linear_420(&y, &u, &v, ChromaLocation::Left, mode, true);
    assert_ne!(
      centered, cosited,
      "centered Linear ({mode:?}) must differ from co-sited (non-vacuous)"
    );
  }
}

/// The bottom-Linear oracle: reconstruct U / V to full width with the vertical
/// bottom blend ([`recon_full_bottom`]) then run that full-res `Yuv444p` frame
/// through the SAME linear-light resample — reconstruct-then-linear-average.
fn oracle_linear_reconstruct_bottom(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  mode: LinearMode,
  simd: bool,
) -> Vec<u8> {
  let (sw, sh, ow, oh) = (8usize, 8usize, 4usize, 4usize);
  let (uf, vf) = recon_full_bottom(u, v, sw, sh);
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
fn bottom_linear_folds_the_vertical_phase() {
  // Bottom Linear must fold the vertical v=1 phase (reconstruct full-width
  // chroma with the vertical blend, decode 4:4:4 in linear light) for both
  // decodes, and differ from the vertically-co-sited Center on a vertical ramp.
  let (y, u, v) = vramp(8, 8);
  for mode in [LinearMode::DisplayReferred, LinearMode::SceneReferred] {
    let bottom = run_linear_420(&y, &u, &v, ChromaLocation::Bottom, mode, true);
    let oracle = oracle_linear_reconstruct_bottom(&y, &u, &v, mode, true);
    assert_eq!(
      bottom, oracle,
      "bottom Linear ({mode:?}) must equal reconstruct-then-linear-average"
    );
    let centered = run_linear_420(&y, &u, &v, ChromaLocation::Center, mode, true);
    assert_ne!(
      bottom, centered,
      "bottom Linear ({mode:?}) must differ from center on a vertical ramp"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_filter_equals_reconstruct_then_filter() {
  // The single-kernel filter tier reconstructs full-width chroma with the
  // vertical bottom blend then Triangle-filters it — equal to feeding the same
  // reconstruction through a Yuv444p Triangle filter, and non-vacuously
  // different from the vertically-co-sited Center on a vertical ramp.
  use crate::resample::{FilteredResampler, Triangle};
  let (sw, sh, ow, oh, cw) = (8usize, 8usize, 4usize, 4usize, 4usize);
  let (y, u, v) = vramp(sw, sh);
  let filter_420 = |loc: ChromaLocation| -> Vec<u8> {
    let mut rgb = vec![0u8; ow * oh * 3];
    {
      let mut sink = MixedSinker::<Yuv420p, FilteredResampler<Triangle>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, Triangle),
      )
      .unwrap()
      .with_chroma_location(loc.clone())
      .with_simd(true)
      .with_rgb(&mut rgb)
      .unwrap();
      let f = Yuv420pFrame::new(
        &y, &u, &v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
      );
      yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    rgb
  };
  let got = filter_420(ChromaLocation::Bottom);
  // reconstruct-then-filter oracle.
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
    filter_420(ChromaLocation::Center),
    "filter-tier bottom must differ from center on a vertical ramp"
  );
}

// ---- IN-SEQUENCE mid-frame phase change is rejected (not silently mixed) -----
//
// Freezing the phase per-frame is not enough to DROP a stale plan — an
// in-sequence row after a mid-frame `set_chroma_location` passes the sequence
// preflight, so without the frozen-phase CHECK it would reconstruct the new
// phase and the frame would bin a mixture. The effective siting is frozen on
// the first output-bearing row; a later in-sequence row observing a different
// phase must be rejected with `ChromaSitingChanged`, uniformly across tiers.

/// Drive one Yuv420p resample frame: `begin_frame`, accept row 0 at `loc1`
/// (freezes the phase), flip to `loc2`, then feed the IN-SEQUENCE row 1 (chroma
/// row 0) and return its `process` result.
fn in_sequence_flip_row1<R>(
  mut sink: MixedSinker<'_, Yuv420p, R>,
  y: &[u8],
  u: &[u8],
  v: &[u8],
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> Result<(), super::super::MixedSinkerError> {
  let cw = 4usize;
  sink.set_chroma_location(loc1.clone());
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  let row0 = Yuv420pRow::for_tests(&y[0..8], &u[0..cw], &v[0..cw], 0, M, FR);
  PixelSink::process(&mut sink, row0).unwrap();
  sink.set_chroma_location(loc2.clone());
  let row1 = Yuv420pRow::for_tests(&y[8..16], &u[0..cw], &v[0..cw], 1, M, FR);
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
  // Both HORIZONTAL flip directions: Center->Left (drop the phase) and
  // Left->Center (add it), plus the VERTICAL flip Center<->Bottom (same
  // horizontal center, differing v=1 fold — caught only by the vertical-phase
  // freeze). BottomLeft adds two more: BottomLeft<->Left (same co-sited h,
  // differing v=1 fold — vertical-phase freeze) and BottomLeft<->Bottom (same
  // v=1 fold, differing horizontal phase — horizontal freeze). Each must reject
  // the in-sequence row 1 with ChromaSitingChanged.
  for (loc1, loc2) in [
    (ChromaLocation::Center, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Center),
    (ChromaLocation::Center, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Center),
    (ChromaLocation::Left, ChromaLocation::BottomLeft),
    (ChromaLocation::BottomLeft, ChromaLocation::Left),
    (ChromaLocation::BottomLeft, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::BottomLeft),
    // RFC #238 Top adds: Center<->Top (same horizontal center, differing v=0 vs
    // co-sited fold — caught only by the Top-phase freeze), Top<->Bottom (same
    // horizontal center, opposite vertical folds), TopLeft<->Left (same co-sited
    // h, differing v=0 fold), and TopLeft<->Top (same v=0 fold, differing
    // horizontal phase).
    (ChromaLocation::Center, ChromaLocation::Top),
    (ChromaLocation::Top, ChromaLocation::Center),
    (ChromaLocation::Top, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Top),
    (ChromaLocation::Left, ChromaLocation::TopLeft),
    (ChromaLocation::TopLeft, ChromaLocation::Left),
    (ChromaLocation::TopLeft, ChromaLocation::Top),
    (ChromaLocation::Top, ChromaLocation::TopLeft),
  ] {
    // Native fast tier.
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
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
    let sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
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
    let sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
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
    let sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
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
      MixedSinker::<Yuv420p, crate::resample::FilteredResampler<crate::resample::Triangle>>::with_resampler(
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

// ---- mid-frame phase change rejects WITHOUT dropping the cached stream ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_mid_frame_phase_change_rejection_keeps_the_stream_retryable() {
  use super::super::MixedSinkerError;
  // Advance rows 0,1 (Center), then flip siting mid-frame (Left): the
  // frozen-phase CHECK rejects it with ChromaSitingChanged at the choke point,
  // ahead of the out-of-sequence check — a mixed-phase frame is never emitted;
  // the frame must be restarted (a rejected row mutates no state).
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..2 {
    let row = Yuv420pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[(r / 2) * cw..(r / 2) * cw + cw],
      &v[(r / 2) * cw..(r / 2) * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  sink.set_chroma_location(ChromaLocation::Left);
  let bad = Yuv420pRow::for_tests(
    &y[5 * 8..6 * 8],
    &u[2 * cw..3 * cw],
    &v[2 * cw..3 * cw],
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
  feed_all(&mut sink, &y, &u, &v, 8, 8);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_mid_frame_phase_change_rejection_keeps_the_stream_retryable() {
  use super::super::MixedSinkerError;
  // The HSV-only (`hsv_direct`) twin: a mid-frame siting flip is rejected with
  // ChromaSitingChanged, and begin_frame restarts cleanly.
  let (y, u, v) = ramp(8, 8);
  let cw = 4usize;
  let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(false)
      .with_chroma_location(ChromaLocation::Center)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
  for r in 0..2 {
    let row = Yuv420pRow::for_tests(
      &y[r * 8..r * 8 + 8],
      &u[(r / 2) * cw..(r / 2) * cw + cw],
      &v[(r / 2) * cw..(r / 2) * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  sink.set_chroma_location(ChromaLocation::Left);
  let bad = Yuv420pRow::for_tests(
    &y[5 * 8..6 * 8],
    &u[2 * cw..3 * cw],
    &v[2 * cw..3 * cw],
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
  feed_all(&mut sink, &y, &u, &v, 8, 8);
}

// ---- bottom-LEFT-sited (co-sited h = 0 + bottom v = 1) resample =============

/// The EXACT bottom-left-sited (`h = 0`, `v = 1`) chroma oracle for the native
/// tier: the CO-SITED horizontal box (a plain `cw`-wide area, NOT the centered
/// `1/4`–`3/4` reconstruction of [`bin_chroma_bottom`]) composed with the same
/// vertical `v = 1` triangle (×2), box-averaged to `ow x oh` with a SINGLE
/// round-half-up. This is the code-domain twin the folded
/// [`ResamplePlan::area_chroma_420`] realizes at `h_phase = 0`, `v_phase = 1`
/// (EVEN `sh`, `ch = sh / 2`).
fn bin_chroma_bottomleft(c: &[u8], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u8> {
  let sh = 2 * ch;
  // ×2 vertical fold at HALF width, kept UNROUNDED. The horizontal axis is
  // co-sited (no reconstruction), so the plane stays `cw` wide.
  let mut r2 = vec![0u64; cw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let prev = cr.saturating_sub(1);
    for j in 0..cw {
      r2[r * cw + j] = if r & 1 == 0 {
        u64::from(c[prev * cw + j]) + u64::from(c[cr * cw + j]) // even: {1, 1}
      } else {
        2 * u64::from(c[cr * cw + j]) // odd: {2}
      };
    }
  }
  let hw = area_weights(cw, ow);
  let vw = area_weights(sh, oh);
  let denom = (2 * cw * sh) as u64; // ×2 V × the box normalization (H co-sited, ×1)
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * r2[(vs + dy) * cw + hs + dx];
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// The bottom-left-sited NATIVE oracle: bin Y co-sited and U / V through the
/// exact bottom-left fold to `ow x oh`, then convert ONCE at output width via an
/// identity `Yuv444p` sink — the byte-for-byte ground truth the native tier
/// reproduces for `ChromaLocation::BottomLeft` (EVEN `sh` only).
#[allow(clippy::too_many_arguments)]
fn bottomleft_native_oracle(
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
  let ub = bin_chroma_bottomleft(u, cw, ch, ow, oh);
  let vb = bin_chroma_bottomleft(v, cw, ch, ow, oh);
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

/// Co-sited horizontal reconstruction (a plain 2× replicate) — the co-sited
/// twin of [`recon_full_row`].
fn recon_full_row_cosited(c: &[u8], cw: usize) -> Vec<u8> {
  let mut out = vec![0u8; 2 * cw];
  for j in 0..cw {
    out[2 * j] = c[j];
    out[2 * j + 1] = c[j];
  }
  out
}

/// Reconstruct `Yuv420p` chroma to full resolution (`sw x sh`) for the
/// bottom-left-sited (`h = 0`, `v = 1`) decode: [`recon_full_bottom`]'s vertical
/// box blend, but the co-sited horizontal replicate instead of the centered
/// kernel. The shared reconstruction step for every reconstruct-then-* oracle.
fn recon_full_bottomleft(u: &[u8], v: &[u8], sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let vblend = |plane: &[u8], cr: usize, prev: usize| -> Vec<u8> {
    (0..cw)
      .map(|c| {
        let a = u32::from(plane[prev * cw + c]);
        let b = u32::from(plane[cr * cw + c]);
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
        u[cr * cw..cr * cw + cw].to_vec(),
        v[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_cosited(&uh, cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_cosited(&vh, cw));
  }
  (uf, vf)
}

/// The bottom-left-sited ENCODED row-stage oracle: reconstruct U / V to full
/// width ([`recon_full_bottomleft`]) then run that frame through a
/// `with_native(false)` RGB-domain resample — exactly what the `Yuv420p` encoded
/// arm does for `BottomLeft`.
#[allow(clippy::too_many_arguments)]
fn encoded_oracle_rgb_bottomleft(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Vec<u8> {
  let (uf, vf) = recon_full_bottomleft(u, v, sw, sh);
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

/// Direct (non-resample) `Yuv420p` `BottomLeft` decode to RGB — the delay-line
/// kernel path, the already-validated identity reference the resample path must
/// match at identity dimensions.
fn direct_bottomleft_rgb(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  simd: bool,
) -> Vec<u8> {
  let cw = sw / 2;
  let mut rgb = vec![0u8; sw * sh * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(sw, sh)
      .with_chroma_location(ChromaLocation::BottomLeft)
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_native_equals_code_domain_oracle() {
  // EVEN source height so the vertical luma-domain pairing equals the phased-V
  // fold; the horizontal axis is co-sited.
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
    let (y, u, v) = ramp(sw, sh);
    let o = bottomleft_native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
    let n = run(
      &y,
      &u,
      &v,
      sw,
      sh,
      ow,
      oh,
      ChromaLocation::BottomLeft,
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
fn bottomleft_native_simd_matches_scalar() {
  let (y, u, v) = ramp(8, 8);
  let s = run(
    &y,
    &u,
    &v,
    8,
    8,
    4,
    4,
    ChromaLocation::BottomLeft,
    true,
    false,
  );
  let d = run(
    &y,
    &u,
    &v,
    8,
    8,
    4,
    4,
    ChromaLocation::BottomLeft,
    true,
    true,
  );
  assert_eq!(
    s, d,
    "bottom-left native SIMD must be 0-ULP identical to scalar"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_encoded_output_equals_rgb_reconstruct_then_bin() {
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (16, 8, 6, 5)] {
    let (y, u, v) = ramp(sw, sh);
    let got = run(
      &y,
      &u,
      &v,
      sw,
      sh,
      ow,
      oh,
      ChromaLocation::BottomLeft,
      false,
      true,
    );
    let oracle = encoded_oracle_rgb_bottomleft(&y, &u, &v, sw, sh, ow, oh, true);
    assert_eq!(
      got.0, oracle,
      "encoded-rgb bottom-left {sw}x{sh}->{ow}x{oh}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_encoded_identity_matches_direct_decode() {
  // At identity dimensions the encoded row-stage reconstructs full-width chroma
  // and bins with pass-through area weights, so routing BottomLeft through the
  // resample must equal the already-validated direct BottomLeft decode.
  let (y, u, v) = vramp(8, 8);
  let res = run(
    &y,
    &u,
    &v,
    8,
    8,
    8,
    8,
    ChromaLocation::BottomLeft,
    false,
    true,
  );
  let direct = direct_bottomleft_rgb(&y, &u, &v, 8, 8, true);
  assert_eq!(
    res.0, direct,
    "encoded-identity bottom-left == direct decode"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_differs_from_bottom_and_cosited() {
  // On a horizontal ramp BottomLeft (co-sited h) diverges from Bottom (centered
  // h) on both tiers; on a vertical ramp it diverges from the co-sited baseline.
  for native in [true, false] {
    let (y, u, v) = ramp(8, 8);
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
    let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
    assert_ne!(
      bl.0, bot.0,
      "bottom-left rgb must differ from bottom (native={native})"
    );

    let (y, u, v) = vramp(8, 8);
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
    let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
    assert_ne!(
      bl.0, cos.0,
      "bottom-left rgb must differ from co-sited (native={native})"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_equals_cosited_on_flat_chroma() {
  // Flat chroma: both the vertical blend and the co-sited horizontal phase are
  // no-ops, so BottomLeft collapses to the co-sited decode on both tiers.
  for native in [true, false] {
    let (y, u, v) = flat_chroma(8, 8);
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
    assert_eq!(bl.0, cos.0, "flat-chroma bottom-left rgb (native={native})");
    assert_eq!(bl.2, cos.2, "flat-chroma bottom-left hsv (native={native})");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_filter_equals_reconstruct_then_filter() {
  // The single-kernel filter tier reconstructs full-width chroma (co-sited h +
  // vertical bottom blend) then Triangle-filters it — equal to feeding the same
  // reconstruction through a Yuv444p Triangle filter.
  use crate::resample::{FilteredResampler, Triangle};
  let (sw, sh, ow, oh, cw) = (8usize, 8usize, 4usize, 4usize, 4usize);
  let (y, u, v) = vramp(sw, sh);
  let mut got = vec![0u8; ow * oh * 3];
  {
    let mut sink = MixedSinker::<Yuv420p, FilteredResampler<Triangle>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, Triangle),
    )
    .unwrap()
    .with_chroma_location(ChromaLocation::BottomLeft)
    .with_simd(true)
    .with_rgb(&mut got)
    .unwrap();
    let f = Yuv420pFrame::new(
      &y, &u, &v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let (uf, vf) = recon_full_bottomleft(&u, &v, sw, sh);
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
  assert_eq!(
    got, oracle,
    "filter-tier bottom-left == reconstruct-then-filter"
  );
}

// ---- top-sited (v = 0, FORWARD fold) resample ==============================
//
// `Top` is the exact vertical MIRROR of `Bottom`: where Bottom's EVEN luma row
// box-blends the PREVIOUS chroma row (backward), Top's ODD luma row box-blends
// the NEXT chroma row (forward), with a BOTTOM-edge clamp on the trailing odd
// row. Horizontally, `Top` rides the centered `1/4`–`3/4` triangle (like
// Center/Bottom) and `TopLeft` the co-sited replicate (like BottomLeft). The
// reconstruction tiers realize the forward reach with a FORWARD one-row output
// delay (the mirror of Bottom's `chroma_prev` lookback), so the last row of an
// ODD-height frame emits TWO output rows.

/// The EXACT top-sited (`v = 0`) chroma oracle for the native tier: the centered
/// horizontal `1/4`–`3/4` triangle (×4) composed with the vertical `v = 0`
/// triangle (×2 — even luma row `2i` takes chroma row `i` with weight 2, odd row
/// `2i + 1` box-blends chroma rows `{i, i + 1}` with weights `{1, 1}` and a
/// BOTTOM-edge clamp), the combined ×8 UNROUNDED reconstruction box-averaged to
/// `ow x oh` with a SINGLE round-half-up. The vertical mirror of
/// [`bin_chroma_bottom`].
fn bin_chroma_top(c: &[u8], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u8> {
  let full = 2 * cw;
  let sh = 2 * ch;
  let mut r8 = vec![0u64; full * sh];
  for r in 0..sh {
    let cr = r / 2;
    let next = if cr + 1 < ch { cr + 1 } else { cr }; // bottom-edge clamp
    let vrow: Vec<u32> = (0..cw)
      .map(|j| {
        if r & 1 == 0 {
          2 * u32::from(c[cr * cw + j]) // even: {2} co-sited
        } else {
          u32::from(c[cr * cw + j]) + u32::from(c[next * cw + j]) // odd: {1, 1} forward
        }
      })
      .collect();
    for j in 0..cw {
      let l = vrow[j.saturating_sub(1)];
      let m = vrow[j];
      let rt = vrow[if j + 1 < cw { j + 1 } else { j }];
      r8[r * full + 2 * j] = u64::from(l + 3 * m);
      r8[r * full + 2 * j + 1] = u64::from(3 * m + rt);
    }
  }
  let hw = area_weights(full, ow);
  let vw = area_weights(sh, oh);
  let denom = (8 * full * sh) as u64;
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * r8[(vs + dy) * full + hs + dx];
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// The top-sited NATIVE oracle: bin Y co-sited and U / V through the exact top
/// V-fold chroma oracle, then convert ONCE at output width via an identity
/// `Yuv444p` sink — the byte-for-byte ground truth the native tier reproduces
/// for `ChromaLocation::Top` (EVEN `sh` only).
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
  let cw = sw / 2;
  let ch = sh / 2;
  let yb = bin_cosited(y, sw, sh, ow, oh);
  let ub = bin_chroma_top(u, cw, ch, ow, oh);
  let vb = bin_chroma_top(v, cw, ch, ow, oh);
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

/// Reconstruct `Yuv420p` chroma to full resolution (`sw x sh`) for the top-sited
/// (`v = 0`) decode — the identity top kernel at source width: per luma row the
/// even rows take chroma row `i` co-sited, the odd rows FORWARD box-blend chroma
/// rows `i` and `i + 1` (round-half-up, bottom-edge clamp), each then
/// horizontally upsampled with the #302 centered `1/4`–`3/4` kernel. The mirror
/// of [`recon_full_bottom`].
fn recon_full_top(u: &[u8], v: &[u8], sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let ch = sh.div_ceil(2);
  let vblend = |plane: &[u8], a: usize, b: usize| -> Vec<u8> {
    (0..cw)
      .map(|c| {
        let x = u32::from(plane[a * cw + c]);
        let z = u32::from(plane[b * cw + c]);
        ((x + z + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  let mut uf = vec![0u8; sw * sh];
  let mut vf = vec![0u8; sw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let (uh, vh) = if r & 1 == 0 {
      (
        u[cr * cw..cr * cw + cw].to_vec(),
        v[cr * cw..cr * cw + cw].to_vec(),
      )
    } else {
      let next = if cr + 1 < ch { cr + 1 } else { cr };
      (vblend(u, cr, next), vblend(v, cr, next))
    };
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&uh, cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row(&vh, cw));
  }
  (uf, vf)
}

/// The top-sited ENCODED row-stage oracle: reconstruct U / V to full width with
/// the vertical top blend ([`recon_full_top`]) then run that full-resolution
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

/// Direct (non-resample) `Yuv420p` `Top` decode to RGB — the delay-line kernel
/// path (the foundation's already-validated identity Top reference), the
/// resample path must match it at identity dimensions.
fn direct_top_rgb(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize, simd: bool) -> Vec<u8> {
  let cw = sw / 2;
  let mut rgb = vec![0u8; sw * sh * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(sw, sh)
      .with_chroma_location(ChromaLocation::Top)
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_native_equals_code_domain_oracle() {
  // The native tier folds the vertical v=0 FORWARD triangle into the chroma area
  // weights; its output is the EXACT code-domain box-average of the UNROUNDED
  // H⊗V reconstruction (single rounding). Includes ODD output heights (5x3) so
  // the trailing-odd bottom-edge clamp is exercised.
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
    let (y, u, v) = ramp(sw, sh);
    let o = top_native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
    let n = run(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Top, true, true);
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
fn top_native_simd_matches_scalar() {
  let (y, u, v) = ramp(8, 8);
  let s = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, true, false);
  let d = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, true, true);
  assert_eq!(s.0, d.0, "rgb scalar vs simd");
  assert_eq!(s.2, d.2, "hsv scalar vs simd");
  assert_eq!(s.3, d.3, "luma scalar vs simd");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_encoded_output_equals_rgb_reconstruct_then_bin() {
  // The encoded row-stage tier reconstructs full-width chroma with the FORWARD
  // one-row delay then bins in RGB — the reconstruct-then-bin oracle. The 8x8->4x4
  // and 12x8->6x4 cases have an EVEN final source row (trailing-odd co-sited);
  // 8x8->5x3 additionally downscales the delayed rows non-trivially.
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
fn top_encoded_identity_matches_direct_decode() {
  // At identity dimensions the encoded row-stage reconstructs chroma with the
  // SAME forward-delay top kernel as the direct (non-resample) Yuv420p decode and
  // bins with pass-through area weights — so routing Top through the resample
  // preserves the foundation's direct decode byte-for-byte (the cross-tier
  // consistency check that caught the earlier half-wired attempt).
  let (y, u, v) = vramp(8, 8);
  let res = run(&y, &u, &v, 8, 8, 8, 8, ChromaLocation::Top, false, true);
  let direct = direct_top_rgb(&y, &u, &v, 8, 8, true);
  assert_eq!(
    res.0, direct,
    "identity encoded resample top == direct decode"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_two_row_final_flush_odd_height_matches_direct_decode() {
  // ODD source height => the FINAL luma row is EVEN, so `process(height - 1)`
  // emits TWO output rows (the held odd predecessor + the last even row). At
  // identity dimensions the encoded resample must still equal the direct decode,
  // proving the two-row final flush writes the last two rows correctly.
  for sh in [5usize, 7] {
    // ODD height: chroma has `ceil(sh / 2)` rows. A strong per-ROW chroma step so
    // the forward vertical blend visibly moves the delayed rows.
    let (sw, cw, ch) = (8usize, 4usize, sh.div_ceil(2));
    let y = vec![128u8; sw * sh];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
      for c in 0..cw {
        u[r * cw + c] = (20 + r * 40).min(240) as u8;
        v[r * cw + c] = (220u32.saturating_sub((r * 40) as u32)).max(16) as u8;
      }
    }
    let res = run(&y, &u, &v, sw, sh, sw, sh, ChromaLocation::Top, false, true);
    let direct = direct_top_rgb(&y, &u, &v, sw, sh, true);
    assert_eq!(res.0, direct, "odd-height two-row final flush sh={sh}");
    // Non-vacuous: the forward fold genuinely moves the final even row vs the
    // vertically-co-sited Center decode.
    let center = run(
      &y,
      &u,
      &v,
      sw,
      sh,
      sw,
      sh,
      ChromaLocation::Center,
      false,
      true,
    );
    assert_ne!(res.0, center.0, "top must differ from center sh={sh}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_differs_from_center_and_bottom_on_a_vertical_ramp() {
  // The forward v=0 fold must actually MOVE the chroma and differ from BOTH the
  // vertically-co-sited Center AND the backward Bottom fold, on both tiers.
  let (y, u, v) = vramp(8, 8);
  for native in [true, false] {
    let top = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true);
    let cen = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
    let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
    assert_ne!(top.0, cen.0, "top != center (native={native})");
    assert_ne!(top.0, bot.0, "top != bottom (native={native})");
    assert_ne!(cen.0, bot.0, "center != bottom (native={native})");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_equals_cosited_on_flat_chroma() {
  // On constant chroma the vertical forward blend (and the horizontal triangle)
  // are no-ops, so Top collapses to the co-sited decode byte-for-byte.
  let (y, u, v) = flat_chroma(8, 8);
  for native in [true, false] {
    let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
    let top = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true);
    assert_eq!(top.0, cos.0, "flat-chroma top rgb (native={native})");
    assert_eq!(top.2, cos.2, "flat-chroma top hsv (native={native})");
  }
}

/// The top-Linear oracle: reconstruct U / V to full width with the vertical top
/// blend ([`recon_full_top`]) then run that full-res `Yuv444p` frame through the
/// SAME linear-light resample — reconstruct-then-linear-average.
#[allow(clippy::too_many_arguments)]
fn oracle_linear_reconstruct_top(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  mode: LinearMode,
  simd: bool,
) -> Vec<u8> {
  let (uf, vf) = recon_full_top(u, v, sw, sh);
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

/// A `Yuv420p` top-Linear area resample (`with_native(false)`) to RGB.
#[allow(clippy::too_many_arguments)]
fn run_linear_420_top(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  mode: LinearMode,
  simd: bool,
) -> Vec<u8> {
  let cw = sw / 2;
  let mut rgb = vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_native(false)
        .with_linear_mode(mode)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_linear_folds_the_vertical_phase() {
  // Top Linear must fold the vertical v=0 forward phase (reconstruct full-width
  // chroma with the forward delay, decode 4:4:4 in linear light) for both
  // decodes, and differ from the vertically-co-sited Center on a vertical ramp.
  // EVEN and ODD source heights (the odd case exercises the two-row final flush
  // where the FINAL even row's second feed allocates the bin tail).
  for (sw, sh, ow, oh) in [(8usize, 8usize, 4usize, 4usize), (8, 6, 4, 3)] {
    let (y, u, v) = vramp(sw, sh);
    for mode in [LinearMode::DisplayReferred, LinearMode::SceneReferred] {
      let top = run_linear_420_top(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Top, mode, true);
      let oracle = oracle_linear_reconstruct_top(&y, &u, &v, sw, sh, ow, oh, mode, true);
      assert_eq!(
        top, oracle,
        "top Linear ({mode:?}) {sw}x{sh}->{ow}x{oh} must equal reconstruct-then-linear-average"
      );
      let center = run_linear_420_top(
        &y,
        &u,
        &v,
        sw,
        sh,
        ow,
        oh,
        ChromaLocation::Center,
        mode,
        true,
      );
      assert_ne!(top, center, "top Linear ({mode:?}) must differ from center");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_linear_final_even_row_tail_alloc_failure_retries_atomically() {
  // ODD source height => the FINAL luma row is EVEN and drives a TWO-row flush
  // (held odd predecessor + last even row) whose SECOND (final) feed allocates
  // the bin tail. That tail is pre-reserved BEFORE the first feed, so an armed
  // allocation refusal returns BEFORE either feed advances the frame cursor or
  // consumes the held row — the SAME row retries cleanly and reproduces the
  // reference. Without the pre-reservation the failed final feed would strand the
  // held row (its predecessor consumed, the cursor advanced) and the retry would
  // reject / diverge.
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;
  let (sw, sh, ow, oh) = (8usize, 7usize, 4usize, 3usize);
  let cw = sw / 2;
  let ch = sh.div_ceil(2);
  let y = vec![128u8; sw * sh];
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (20 + r * 40).min(240) as u8;
      v[r * cw + c] = (220u32.saturating_sub((r * 40) as u32)).max(16) as u8;
    }
  }
  let run_frame = |arm_fail: bool, loc: ChromaLocation| -> Vec<u8> {
    let mut rgb = vec![0u8; ow * oh * 3];
    {
      let mut sink =
        MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
          .unwrap()
          .with_averaging_domain(AveragingDomain::Linear)
          .with_native(false)
          .with_chroma_location(loc.clone())
          .with_rgb(&mut rgb)
          .unwrap();
      sink.begin_frame(sw as u32, sh as u32).unwrap();
      let feed = |sink: &mut MixedSinker<'_, Yuv420p, AreaResampler>, r: usize| {
        let yr = &y[r * sw..(r + 1) * sw];
        let cr = r / 2;
        let ur = &u[cr * cw..(cr + 1) * cw];
        let vr = &v[cr * cw..(cr + 1) * cw];
        sink.process(Yuv420pRow::for_tests(yr, ur, vr, r, M, FR))
      };
      for r in 0..sh - 1 {
        feed(&mut sink, r).unwrap();
      }
      if arm_fail {
        crate::sinker::mixed::linear_light::arm_linear_tail_alloc_failure();
        let err = feed(&mut sink, sh - 1).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "armed final-row tail alloc must surface AllocationFailed, got {err:?}"
        );
      }
      // Retry the SAME final row (the failpoint is one-shot, already taken).
      feed(&mut sink, sh - 1).unwrap();
    }
    rgb
  };
  let reference = run_frame(false, ChromaLocation::Top);
  let retried = run_frame(true, ChromaLocation::Top);
  assert_eq!(
    retried, reference,
    "the post-failure retry of the two-row FINAL flush must reproduce the reference"
  );
  assert_ne!(
    reference,
    run_frame(false, ChromaLocation::Center),
    "the top forward fold must move the final even row vs co-sited Center"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_filter_equals_reconstruct_then_filter() {
  // The single-kernel filter tier reconstructs full-width chroma with the forward
  // top delay then Triangle-filters it — equal to feeding the same reconstruction
  // through a Yuv444p Triangle filter, and different from the vertically-co-sited
  // Center on a vertical ramp.
  use crate::resample::{FilteredResampler, Triangle};
  let (sw, sh, ow, oh, cw) = (8usize, 8usize, 4usize, 4usize, 4usize);
  let (y, u, v) = vramp(sw, sh);
  let filter_420 = |loc: ChromaLocation| -> Vec<u8> {
    let mut rgb = vec![0u8; ow * oh * 3];
    {
      let mut sink = MixedSinker::<Yuv420p, FilteredResampler<Triangle>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, Triangle),
      )
      .unwrap()
      .with_chroma_location(loc.clone())
      .with_simd(true)
      .with_rgb(&mut rgb)
      .unwrap();
      let f = Yuv420pFrame::new(
        &y, &u, &v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
      );
      yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    rgb
  };
  let got = filter_420(ChromaLocation::Top);
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
    filter_420(ChromaLocation::Center),
    "filter-tier top must differ from center on a vertical ramp"
  );
}

// ---- top-LEFT-sited (co-sited h = 0 + top v = 0) resample ==================

/// The EXACT top-left-sited (`h = 0`, `v = 0`) chroma oracle for the native tier:
/// the CO-SITED horizontal box composed with the vertical `v = 0` FORWARD
/// triangle (×2), box-averaged with a SINGLE round-half-up. The vertical mirror
/// of [`bin_chroma_bottomleft`].
fn bin_chroma_topleft(c: &[u8], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u8> {
  let sh = 2 * ch;
  let mut r2 = vec![0u64; cw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let next = if cr + 1 < ch { cr + 1 } else { cr };
    for j in 0..cw {
      r2[r * cw + j] = if r & 1 == 0 {
        2 * u64::from(c[cr * cw + j]) // even: {2}
      } else {
        u64::from(c[cr * cw + j]) + u64::from(c[next * cw + j]) // odd: {1, 1} forward
      };
    }
  }
  let hw = area_weights(cw, ow);
  let vw = area_weights(sh, oh);
  let denom = (2 * cw * sh) as u64;
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * r2[(vs + dy) * cw + hs + dx];
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// The top-left-sited NATIVE oracle (mirror of [`bottomleft_native_oracle`]).
#[allow(clippy::too_many_arguments)]
fn topleft_native_oracle(
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
  let ub = bin_chroma_topleft(u, cw, ch, ow, oh);
  let vb = bin_chroma_topleft(v, cw, ch, ow, oh);
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

/// Reconstruct `Yuv420p` chroma to full resolution for the top-left-sited
/// (`h = 0`, `v = 0`) decode: [`recon_full_top`]'s forward vertical blend, but a
/// co-sited horizontal replicate.
fn recon_full_topleft(u: &[u8], v: &[u8], sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let ch = sh.div_ceil(2);
  let vblend = |plane: &[u8], a: usize, b: usize| -> Vec<u8> {
    (0..cw)
      .map(|c| {
        let x = u32::from(plane[a * cw + c]);
        let z = u32::from(plane[b * cw + c]);
        ((x + z + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  let mut uf = vec![0u8; sw * sh];
  let mut vf = vec![0u8; sw * sh];
  for r in 0..sh {
    let cr = r / 2;
    let (uh, vh) = if r & 1 == 0 {
      (
        u[cr * cw..cr * cw + cw].to_vec(),
        v[cr * cw..cr * cw + cw].to_vec(),
      )
    } else {
      let next = if cr + 1 < ch { cr + 1 } else { cr };
      (vblend(u, cr, next), vblend(v, cr, next))
    };
    uf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_cosited(&uh, cw));
    vf[r * sw..r * sw + sw].copy_from_slice(&recon_full_row_cosited(&vh, cw));
  }
  (uf, vf)
}

/// The top-left-sited ENCODED row-stage oracle.
#[allow(clippy::too_many_arguments)]
fn encoded_oracle_rgb_topleft(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  simd: bool,
) -> Vec<u8> {
  let (uf, vf) = recon_full_topleft(u, v, sw, sh);
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

/// Direct (non-resample) `Yuv420p` `TopLeft` decode to RGB.
fn direct_topleft_rgb(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize, simd: bool) -> Vec<u8> {
  let cw = sw / 2;
  let mut rgb = vec![0u8; sw * sh * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(sw, sh)
      .with_chroma_location(ChromaLocation::TopLeft)
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap();
    let f = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    yuv420p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn topleft_native_equals_code_domain_oracle() {
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)] {
    let (y, u, v) = ramp(sw, sh);
    let o = topleft_native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
    let n = run(
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
fn topleft_native_simd_matches_scalar() {
  let (y, u, v) = ramp(8, 8);
  let s = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::TopLeft, true, false);
  let d = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::TopLeft, true, true);
  assert_eq!(
    s, d,
    "top-left native SIMD must be 0-ULP identical to scalar"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn topleft_encoded_output_equals_rgb_reconstruct_then_bin() {
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (16, 8, 6, 5)] {
    let (y, u, v) = ramp(sw, sh);
    let got = run(
      &y,
      &u,
      &v,
      sw,
      sh,
      ow,
      oh,
      ChromaLocation::TopLeft,
      false,
      true,
    );
    let oracle = encoded_oracle_rgb_topleft(&y, &u, &v, sw, sh, ow, oh, true);
    assert_eq!(got.0, oracle, "encoded-rgb top-left {sw}x{sh}->{ow}x{oh}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn topleft_encoded_identity_matches_direct_decode() {
  let (y, u, v) = vramp(8, 8);
  let res = run(&y, &u, &v, 8, 8, 8, 8, ChromaLocation::TopLeft, false, true);
  let direct = direct_topleft_rgb(&y, &u, &v, 8, 8, true);
  assert_eq!(res.0, direct, "encoded-identity top-left == direct decode");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn topleft_differs_from_top_and_bottomleft() {
  // On a horizontal ramp TopLeft (co-sited h) diverges from Top (centered h); on
  // a vertical ramp it diverges from BottomLeft (opposite vertical fold).
  for native in [true, false] {
    let (y, u, v) = ramp(8, 8);
    let tl = run(
      &y,
      &u,
      &v,
      8,
      8,
      4,
      4,
      ChromaLocation::TopLeft,
      native,
      true,
    );
    let tp = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true);
    assert_ne!(tl.0, tp.0, "top-left != top (native={native})");

    let (y, u, v) = vramp(8, 8);
    let tl = run(
      &y,
      &u,
      &v,
      8,
      8,
      4,
      4,
      ChromaLocation::TopLeft,
      native,
      true,
    );
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
    assert_ne!(tl.0, bl.0, "top-left != bottom-left (native={native})");
  }
}
