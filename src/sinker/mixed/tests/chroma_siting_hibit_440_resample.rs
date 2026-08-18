//! RFC #238 S8c — chroma-siting-aware fused-downscale coverage for the HIGH-BIT
//! **planar** 4:4:0 YUV family — `Yuv440p10` / `Yuv440p12` (LE + BE wire), the
//! `u16` twin of the 8-bit `chroma_siting_440_resample` suite and the vertical
//! (`Bottom`) sibling of the high-bit 4:2:0 `chroma_siting_hibit_420_resample`
//! suite. This is the LAST 4:4:0 piece.
//!
//! 4:4:0 keeps FULL-width chroma, subsampled 2:1 **vertically only**, so the
//! siting reduces to its vertical axis: `Bottom` / `BottomLeft`
//! ([`chroma_440_bottom_sited_v`](super::super::chroma_440_bottom_sited_v),
//! `v = 1`) folds the VERTICAL triangle through the resample (its even output row
//! box-blends the previous chroma row), while every co-sited / horizontal siting
//! keeps the vertical pairing co-sited (`v_phase = 0`, byte-identical to the
//! pre-siting resample):
//!  - the **native fast tier** folds the `v = 1` triangle into the chroma area
//!    weights ([`ResamplePlan::area_chroma_440`]) — one SINGLE-rounding phased
//!    box-average on the half-height grid, NO horizontal reconstruction;
//!  - the **encoded row-stage** and **single-kernel filter** tiers reconstruct
//!    full-height `u16` chroma (the `chroma_prev_u16` lookback) then decode 4:4:4.
//!
//! Both `u8` and native-`u16` colour are exercised; the wire endianness (`BE`) is
//! siting-independent (decode normalizes before the phase math), pinned by the
//! BE↔LE parity test. Representative depths `p10` (low-packed) and `p12` cover the
//! family; every bottom assertion is an EXACT match (single rounding), never a
//! tolerance. Source heights are EVEN so the phased-V spans (denominator
//! `2·luma_h`) align with the `sh / 2` chroma rows.

use crate::{
  ChromaLocation, KernelMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, FilteredResampler, Triangle},
  sinker::{MixedSinker, MixedSinkerError},
  source::*,
};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Round-half-up integer divide — the production `round_div_half_up`, replicated
/// so the oracle stays independent.
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

/// Co-sited box-average of a full-resolution `sw x sh` `u16` plane to `ow x oh`
/// (round-half-up) — the reference for a phase-free plane (luma, co-sited chroma).
fn bin_cosited_u16(plane: &[u16], sw: usize, sh: usize, ow: usize, oh: usize) -> Vec<u16> {
  let hw = area_weights(sw, ow);
  let vw = area_weights(sh, oh);
  let denom = (sw * sh) as u64;
  let mut out = vec![0u16; ow * oh];
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
      out[oy * ow + ox] = rdhu(s, denom) as u16;
    }
  }
  out
}

/// The EXACT bottom-sited (`v = 1`) 4:4:0 chroma oracle (`u16`) for the native
/// tier: the vertical `v = 1` triangle (×2 — even luma row `2i` box-blends chroma
/// rows `{i - 1, i}` with weights `{1, 1}`, odd row `2i + 1` takes chroma row `i`
/// with weight 2, top-edge clamp) kept UNROUNDED over the full-width `sw x sh`
/// grid, box-averaged to `ow x oh` — HORIZONTAL a plain box over `sw` (4:4:0
/// chroma is full-width, NO folded H triangle), VERTICAL over the `sh` luma rows —
/// with a SINGLE round-half-up over `2·sw·sh`. The code-domain twin the folded
/// [`ResamplePlan::area_chroma_440`] realizes when its V axis is
/// `AxisSpans::area_chroma_phased_v` (EVEN `sh` only, `ch = sh / 2`).
fn bin_chroma_bottom_u16(c: &[u16], sw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u16> {
  let sh = 2 * ch;
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
  let mut out = vec![0u16; ow * oh];
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
      out[oy * ow + ox] = rdhu(s, denom) as u16;
    }
  }
  out
}

/// Reconstruct `Yuv440pN` chroma to full height (`sw x sh`, `u16`) for the
/// bottom-sited (`v = 1`) decode — the identity bottom kernel at source width: per
/// luma row the even rows vertically box-blend chroma rows `i - 1` (clamped) and
/// `i` via the production scalar kernel
/// [`chroma_upsample_440_bottom_v_u16`](crate::row::scalar::chroma_upsample_440_bottom_v_u16),
/// the odd rows take chroma row `i` straight through. NO horizontal reconstruction
/// (full-width chroma). The shared reconstruction step for the reconstruct-then-bin
/// oracles — reusing the S8a kernel so the oracle is pinned to production math.
fn recon_full_bottom_u16(u: &[u16], v: &[u16], sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>) {
  let vblend = |plane: &[u16], cr: usize, prev: usize| -> Vec<u16> {
    let mut out = vec![0u16; sw];
    crate::row::scalar::chroma_upsample_440_bottom_v_u16(
      &plane[prev * sw..prev * sw + sw],
      &plane[cr * sw..cr * sw + sw],
      &mut out,
      sw,
    );
    out
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
        u[cr * sw..cr * sw + sw].to_vec(),
        v[cr * sw..cr * sw + sw].to_vec(),
      )
    };
    uf[r * sw..r * sw + sw].copy_from_slice(&uh);
    vf[r * sw..r * sw + sw].copy_from_slice(&vh);
  }
  (uf, vf)
}

/// The FORWARD (`Top`, `v = 0`) code-domain twin of [`bin_chroma_bottom_u16`]:
/// EVEN luma row `2i` is co-sited on `c[i]` (weight `{2}`), ODD row `2i+1` folds
/// `c[i]` and `c[i+1]` (weight `{1, 1}`, bottom-edge clamp), a SINGLE round-half-up
/// over `2·sw·sh`. EVEN `sh` only (`ch = sh / 2`).
fn bin_chroma_top_u16(c: &[u16], sw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u16> {
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
  let mut out = vec![0u16; ow * oh];
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
      out[oy * ow + ox] = rdhu(s, denom) as u16;
    }
  }
  out
}

/// Reconstruct `Yuv440pN` chroma to full height (`sw x sh`, `u16`) for the
/// top-sited (`v = 0`) decode — the FORWARD mirror of [`recon_full_bottom_u16`]:
/// even rows take chroma row `i` straight, odd rows forward box-blend `c[i]` and
/// `c[i+1]` (bottom-edge clamp) via the production scalar kernel
/// [`chroma_upsample_440_bottom_v_u16`](crate::row::scalar::chroma_upsample_440_bottom_v_u16)
/// (a symmetric box average, so applied to the forward pair). NO horizontal
/// reconstruction (full-width chroma).
fn recon_full_top_u16(u: &[u16], v: &[u16], sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>) {
  let ch = sh.div_ceil(2);
  let vblend = |plane: &[u16], a: usize, b: usize| -> Vec<u16> {
    let mut out = vec![0u16; sw];
    crate::row::scalar::chroma_upsample_440_bottom_v_u16(
      &plane[a * sw..a * sw + sw],
      &plane[b * sw..b * sw + sw],
      &mut out,
      sw,
    );
    out
  };
  let mut uf = vec![0u16; sw * sh];
  let mut vf = vec![0u16; sw * sh];
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

/// Re-encode a host-native `u16` slice as host-independent BE-wire storage.
fn as_be(host: &[u16]) -> Vec<u16> {
  host.iter().map(|v| v.to_be()).collect()
}

/// The geometries the oracle-equality tests sweep: clean 2:1 and fractional
/// ratios, EVEN source heights only (so the vertical phased-V spans align with the
/// `ch = sh / 2` chroma rows).
const GEOMS: [(usize, usize, usize, usize); 4] =
  [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)];

macro_rules! hibit_440_resample_siting {
  (
    $mod:ident, $bits:expr,
    $M440:ident, $F440:ident, $w440:ident,
    $M440be:ty, $F440be:ident, $w440be:ident,
    $M444:ident, $F444:ident, $w444:ident,
    $Row:ident
  ) => {
    mod $mod {
      use super::*;

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = 1u16 << ($bits - 1);

      /// A `Yuv440pN` fixture (full-width chroma, `ch = sh / 2`) with a horizontal
      /// AND vertical chroma ramp plus varying luma. Low-packed native codes so
      /// every kernel sees real math. `sh` must be even.
      fn ramp(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let ch = sh / 2;
        let step = (MASK as u32 / 16).max(1);
        let mut y = vec![0u16; sw * sh];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 37) & MASK as u32) as u16;
        }
        let mut u = vec![0u16; sw * ch];
        let mut v = vec![0u16; sw * ch];
        for r in 0..ch {
          for c in 0..sw {
            u[r * sw + c] = (step * c as u32 + step + r as u32 * step * 2).min(MASK as u32) as u16;
            v[r * sw + c] = (MASK as u32)
              .saturating_sub(step * c as u32 + r as u32 * step * 2)
              .max(step) as u16;
          }
        }
        (y, u, v)
      }

      /// Flat-luma fixture with a strong per-ROW chroma step (flat across columns),
      /// so the vertical `Bottom` fold is observable in isolation: a horizontal-only
      /// siting leaves it untouched, the `v = 1` blend visibly moves it. `sh` even.
      fn vramp(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let ch = sh / 2;
        let step = (MASK as u32 / 8).max(1);
        let y = vec![MID; sw * sh];
        let mut u = vec![0u16; sw * ch];
        let mut v = vec![0u16; sw * ch];
        for r in 0..ch {
          for c in 0..sw {
            // Flat across columns, a strong step per chroma row.
            u[r * sw + c] = (step + r as u32 * step).min(MASK as u32) as u16;
            v[r * sw + c] = (MASK as u32).saturating_sub(r as u32 * step).max(step) as u16;
          }
        }
        (y, u, v)
      }

      /// Flat chroma: the vertical blend of a constant is that constant, so
      /// `Bottom` must equal co-sited. Luma still varies.
      fn flat(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let ch = sh / 2;
        let mut y = vec![0u16; sw * sh];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 29) & MASK as u32) as u16;
        }
        (y, vec![MID; sw * ch], vec![MID; sw * ch])
      }

      /// Drive an LE `Yuv440pN` area resample for `rgb` (u8) + `rgb_u16`.
      #[allow(clippy::too_many_arguments)]
      fn run(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
        native: bool,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink =
            MixedSinker::<$M440, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
              .unwrap()
              .with_native(native)
              .with_chroma_location(loc.clone())
              .with_simd(simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgb_u16(&mut rgb16)
              .unwrap();
          let f = $F440::new(
            y, u, v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
          );
          $w440(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// Drive an LE `Yuv440pN` single-kernel filter resample (Triangle).
      #[allow(clippy::too_many_arguments)]
      fn run_filter(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$M440, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_chroma_location(loc.clone())
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          let f = $F440::new(
            y, u, v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
          );
          $w440(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// Drive a BE `Yuv440pN` area resample (planes re-encoded BE-wire).
      #[allow(clippy::too_many_arguments)]
      fn run_be(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
        native: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let (yb, ub, vb) = (as_be(y), as_be(u), as_be(v));
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$M440be, AreaResampler>::with_resampler(
            sw,
            sh,
            AreaResampler::to(ow, oh),
          )
          .unwrap()
          .with_native(native)
          .with_chroma_location(loc.clone())
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          let f = $F440be::try_new(
            &yb, &ub, &vb, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
          )
          .unwrap();
          $w440be(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// The bottom-sited NATIVE oracle: bin Y co-sited and U / V through the exact
      /// bottom V-fold oracle to `ow x oh`, then convert ONCE at output width via an
      /// identity `Yuv444pN` sink.
      #[allow(clippy::too_many_arguments)]
      fn bottom_native_oracle(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let ch = sh / 2;
        let yb = bin_cosited_u16(y, sw, sh, ow, oh);
        let ub = bin_chroma_bottom_u16(u, sw, ch, ow, oh);
        let vb = bin_chroma_bottom_u16(v, sw, ch, ow, oh);
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$M444>::new(ow, oh)
            .with_simd(simd)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb16)
            .unwrap();
          let f = $F444::new(
            &yb, &ub, &vb, ow as u32, oh as u32, ow as u32, ow as u32, ow as u32,
          );
          $w444(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// The bottom-sited RGB-domain oracle: reconstruct U / V to full height with
      /// the vertical bottom blend then run that `Yuv444pN` frame through the given
      /// resampler (convert-each-row-then-bin) — exactly what the row-stage / filter
      /// arms do for `Bottom`. `filter = true` uses the Triangle tier.
      #[allow(clippy::too_many_arguments)]
      fn bottom_rgb_domain_oracle(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        simd: bool,
        filter: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let (uf, vf) = recon_full_bottom_u16(u, v, sw, sh);
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        let f = $F444::new(
          y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
        );
        if filter {
          let mut sink = MixedSinker::<$M444, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          $w444(&f, FR, M, &mut sink).unwrap();
        } else {
          let mut sink =
            MixedSinker::<$M444, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
              .unwrap()
              .with_native(false)
              .with_simd(simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgb_u16(&mut rgb16)
              .unwrap();
          $w444(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// Direct (non-resample, identity dims) `Yuv440pN` `Bottom` decode over the
      /// FULL output set (u8 rgb / rgba / hsv / luma + u16 rgb / rgba) — the
      /// delay-line kernel path, the already-validated identity reference the
      /// identity resample must match at identity dimensions.
      #[allow(clippy::type_complexity)]
      fn direct_bottom_full(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        loc: ChromaLocation,
        simd: bool,
      ) -> (
        Vec<u8>,
        Vec<u8>,
        (Vec<u8>, Vec<u8>, Vec<u8>),
        Vec<u8>,
        Vec<u16>,
        Vec<u16>,
      ) {
        let mut rgb = vec![0u8; sw * sh * 3];
        let mut rgba = vec![0u8; sw * sh * 4];
        let (mut hh, mut ss, mut vv) = (vec![0u8; sw * sh], vec![0u8; sw * sh], vec![0u8; sw * sh]);
        let mut luma = vec![0u8; sw * sh];
        let mut rgb16 = vec![0u16; sw * sh * 3];
        let mut rgba16 = vec![0u16; sw * sh * 4];
        {
          let mut sink = MixedSinker::<$M440>::new(sw, sh)
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
            .with_rgb_u16(&mut rgb16)
            .unwrap()
            .with_rgba_u16(&mut rgba16)
            .unwrap();
          let f = $F440::new(
            y, u, v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
          );
          $w440(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, (hh, ss, vv), luma, rgb16, rgba16)
      }

      /// Identity-dims (`ow = sw`, `oh = sh`) row-stage resample over the FULL
      /// output set — must byte-match [`direct_bottom_full`].
      #[allow(clippy::type_complexity)]
      fn identity_resample_full(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        loc: ChromaLocation,
        simd: bool,
      ) -> (
        Vec<u8>,
        Vec<u8>,
        (Vec<u8>, Vec<u8>, Vec<u8>),
        Vec<u8>,
        Vec<u16>,
        Vec<u16>,
      ) {
        let mut rgb = vec![0u8; sw * sh * 3];
        let mut rgba = vec![0u8; sw * sh * 4];
        let (mut hh, mut ss, mut vv) = (vec![0u8; sw * sh], vec![0u8; sw * sh], vec![0u8; sw * sh]);
        let mut luma = vec![0u8; sw * sh];
        let mut rgb16 = vec![0u16; sw * sh * 3];
        let mut rgba16 = vec![0u16; sw * sh * 4];
        {
          let mut sink =
            MixedSinker::<$M440, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(sw, sh))
              .unwrap()
              .with_native(false)
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
              .with_rgb_u16(&mut rgb16)
              .unwrap()
              .with_rgba_u16(&mut rgba16)
              .unwrap();
          let f = $F440::new(
            y, u, v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
          );
          $w440(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgba, (hh, ss, vv), luma, rgb16, rgba16)
      }

      // ---- co-sited byte-identity (the regression contract) ----------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn cosited_group_is_byte_identical_across_tiers() {
        // Every co-sited / vertically-central siting must produce the byte-identical
        // pre-siting resample (v_phase 0 && !v_top → neither phased-V plan is built),
        // on BOTH tiers. 4:4:0 has no horizontal phase, so `Center` (v=0.5) stays
        // co-sited; `Top` / `TopLeft` (v=0) now fold the forward triangle and left
        // this group.
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          for native in [true, false] {
            let base = run(
              &y,
              &u,
              &v,
              sw,
              sh,
              ow,
              oh,
              ChromaLocation::Unspecified,
              native,
              true,
            );
            for loc in [
              ChromaLocation::Left,
              ChromaLocation::Center,
              ChromaLocation::other("unassigned-7"),
            ] {
              assert_eq!(
                run(&y, &u, &v, sw, sh, ow, oh, loc.clone(), native, true),
                base,
                "co-sited {loc:?} must keep the byte-identical decode \
                 (native={native}, {sw}x{sh}->{ow}x{oh})"
              );
            }
          }
          // The filter tier too.
          let base = run_filter(
            &y,
            &u,
            &v,
            sw,
            sh,
            ow,
            oh,
            ChromaLocation::Unspecified,
            true,
          );
          assert_eq!(
            run_filter(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Center, true),
            base,
            "co-sited filter must stay byte-identical ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_left_resamples_as_bottom_across_tiers() {
        // 4:4:0 has no horizontal phase, so `BottomLeft` (v=1) drives the identical
        // vertical fold as `Bottom` on every tier. The ramp varies vertically, so a
        // `BottomLeft` that stayed co-sited would diverge — a genuine guard.
        let (y, u, v) = ramp(8, 8);
        for native in [true, false] {
          let bottom = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
          assert_eq!(
            run(
              &y,
              &u,
              &v,
              8,
              8,
              4,
              4,
              ChromaLocation::BottomLeft,
              native,
              true
            ),
            bottom,
            "BottomLeft must resample as Bottom (native={native})"
          );
        }
        // Filter tier too.
        let bottom = run_filter(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, true);
        assert_eq!(
          run_filter(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::BottomLeft, true),
          bottom,
          "BottomLeft must filter as Bottom"
        );
      }

      // ---- bottom native == the exact code-domain oracle -------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_native_equals_code_domain_oracle() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          let o = bottom_native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
          assert_eq!(
            run(
              &y,
              &u,
              &v,
              sw,
              sh,
              ow,
              oh,
              ChromaLocation::Bottom,
              true,
              true
            ),
            o,
            "bottom native must equal the V-fold code-domain oracle \
             ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      // ---- bottom row-stage / filter == reconstruct-then-bin ---------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_row_stage_equals_rgb_reconstruct_then_bin() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          let o = bottom_rgb_domain_oracle(&y, &u, &v, sw, sh, ow, oh, true, false);
          assert_eq!(
            run(
              &y,
              &u,
              &v,
              sw,
              sh,
              ow,
              oh,
              ChromaLocation::Bottom,
              false,
              true
            ),
            o,
            "bottom row-stage must equal the RGB-domain reconstruct-then-bin \
             oracle ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_filter_equals_rgb_reconstruct_then_filter() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          let o = bottom_rgb_domain_oracle(&y, &u, &v, sw, sh, ow, oh, true, true);
          assert_eq!(
            run_filter(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Bottom, true),
            o,
            "bottom filter must equal the RGB-domain Triangle oracle \
             ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      // ---- SIMD == scalar, BE == LE ----------------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_simd_matches_scalar() {
        let (y, u, v) = vramp(8, 8);
        for native in [true, false] {
          assert_eq!(
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true),
            run(
              &y,
              &u,
              &v,
              8,
              8,
              4,
              4,
              ChromaLocation::Bottom,
              native,
              false
            ),
            "bottom SIMD vs scalar must agree (native={native})"
          );
        }
        assert_eq!(
          run_filter(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, true),
          run_filter(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, false),
          "bottom filter SIMD vs scalar must agree"
        );
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_be_matches_le() {
        let (y, u, v) = vramp(8, 8);
        for native in [true, false] {
          assert_eq!(
            run_be(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native),
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true),
            "BE bottom decode must equal LE (native={native})"
          );
        }
      }

      // ---- non-vacuous + flat-chroma sanity --------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_differs_from_cosited_on_a_vertical_chroma_ramp() {
        let (y, u, v) = vramp(8, 8);
        for native in [true, false] {
          let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
          let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
          assert_ne!(
            cos, bot,
            "bottom must differ from co-sited on a vertical chroma ramp (native={native})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_equals_cosited_on_flat_chroma() {
        let (y, u, v) = flat(8, 8);
        for native in [true, false] {
          assert_eq!(
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true),
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true),
            "bottom must equal co-sited on flat chroma (native={native})"
          );
        }
      }

      // ---- identity resample == direct decode (ALL outputs) ----------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_identity_resample_matches_direct_decode_all_outputs() {
        // Identity dims (ow=sw, oh=sh): the row-stage resample reconstructs
        // full-width chroma with the SAME bottom kernel as the direct (non-resample)
        // decode and bins with pass-through area weights, so routing Bottom through
        // the resample preserves the decode byte-for-byte across EVERY output
        // (u8 rgb / rgba / hsv / luma + u16 rgb / rgba — the direct AND row-stage
        // reconstruction arms). BottomLeft too (identical v=1 fold).
        let (y, u, v) = vramp(8, 8);
        for loc in [ChromaLocation::Bottom, ChromaLocation::BottomLeft] {
          assert_eq!(
            identity_resample_full(&y, &u, &v, 8, 8, loc.clone(), true),
            direct_bottom_full(&y, &u, &v, 8, 8, loc.clone(), true),
            "identity resample {loc:?} == direct decode (all outputs)"
          );
        }
      }

      // ---- IN-SEQUENCE mid-frame phase change rejected across tiers ---------

      /// Accept row 0 at `loc1` (freezes the vertical phase), flip to `loc2`, feed
      /// the IN-SEQUENCE row 1 (still chroma row 0), and return its result.
      fn flip_row1<R>(
        mut sink: MixedSinker<'_, $M440, R>,
        y: &[u16],
        u: &[u16],
        v: &[u16],
        loc1: ChromaLocation,
        loc2: ChromaLocation,
      ) -> Result<(), MixedSinkerError> {
        sink.set_chroma_location(loc1.clone());
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        let row0 = $Row::new(&y[0..8], &u[0..8], &v[0..8], 0, M, FR);
        PixelSink::process(&mut sink, row0).unwrap();
        sink.set_chroma_location(loc2.clone());
        let row1 = $Row::new(&y[8..16], &u[0..8], &v[0..8], 1, M, FR);
        PixelSink::process(&mut sink, row1)
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn in_sequence_mid_frame_phase_change_rejected_across_tiers() {
        let (y, u, v) = ramp(8, 8);
        // Every co-sited ⇆ bottom-vertical flip (both directions, incl BottomLeft)
        // changes the v=1 fold and must reject the in-sequence row 1 with
        // ChromaSitingChanged across all tiers AND the identity (direct) path.
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
          let sink =
            MixedSinker::<$M440, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(true)
              .with_rgb(&mut rgb)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "native {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Encoded row-stage tier.
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink =
            MixedSinker::<$M440, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(false)
              .with_rgb(&mut rgb)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "row-stage {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Filter tier.
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink = MixedSinker::<$M440, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Identity (direct) path — no resampler.
          let mut rgb = vec![0u8; 8 * 8 * 3];
          let sink = MixedSinker::<$M440>::new(8, 8).with_rgb(&mut rgb).unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "identity {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
        }
      }

      // ---- cross-frame sink reuse rebuilds the phased native join ----------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn native_join_rebuilds_on_siting_change_across_frames() {
        // Reuse one native-tier sink flipping Left ⇆ Bottom (both directions): frame
        // 2 must match a FRESH sink for frame 2's siting — no stale-phase carryover,
        // and no cross-frame stale chroma blend.
        let (y, u, v) = vramp(8, 8);
        for (loc1, loc2) in [
          (ChromaLocation::Left, ChromaLocation::Bottom),
          (ChromaLocation::Bottom, ChromaLocation::Left),
        ] {
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let mut rgb16 = vec![0u16; 4 * 4 * 3];
          {
            let mut sink =
              MixedSinker::<$M440, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
                .unwrap()
                .with_native(true)
                .with_rgb(&mut rgb)
                .unwrap()
                .with_rgb_u16(&mut rgb16)
                .unwrap();
            let f = $F440::new(&y, &u, &v, 8, 8, 8, 8, 8);
            sink.set_chroma_location(loc1.clone());
            $w440(&f, FR, M, &mut sink).unwrap();
            sink.set_chroma_location(loc2.clone());
            $w440(&f, FR, M, &mut sink).unwrap();
          }
          let fresh = run(&y, &u, &v, 8, 8, 4, 4, loc2.clone(), true, true);
          assert_eq!(
            (rgb, rgb16),
            fresh,
            "native reuse {loc1:?}->{loc2:?} must rebuild the phased join"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn row_stage_bottom_reuse_across_frames_no_stale_blend() {
        // Reuse ONE row-stage `Bottom` sink across two frames of DIFFERENT content:
        // frame 2 must byte-match a FRESH sink for frame 2, proving `begin_frame`
        // reset the streams AND cleared the `chroma_prev_u16` lookback tag — so
        // frame 2's first even row never box-blends frame 1's last chroma row.
        let (ya, ua, va) = vramp(8, 8);
        let (yb, ub, vb) = ramp(8, 8);
        let mut rgb = vec![0u8; 4 * 4 * 3];
        let mut rgb16 = vec![0u16; 4 * 4 * 3];
        {
          let mut sink =
            MixedSinker::<$M440, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(false)
              .with_chroma_location(ChromaLocation::Bottom)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgb_u16(&mut rgb16)
              .unwrap();
          let fa = $F440::new(&ya, &ua, &va, 8, 8, 8, 8, 8);
          $w440(&fa, FR, M, &mut sink).unwrap();
          let fb = $F440::new(&yb, &ub, &vb, 8, 8, 8, 8, 8);
          $w440(&fb, FR, M, &mut sink).unwrap();
        }
        let fresh = run(
          &yb,
          &ub,
          &vb,
          8,
          8,
          4,
          4,
          ChromaLocation::Bottom,
          false,
          true,
        );
        assert_eq!(
          (rgb, rgb16),
          fresh,
          "row-stage Bottom sink reused across frames must not carry stale chroma"
        );
      }

      // ======== Top-sited (v = 0) FORWARD one-row delay ====================

      /// The top-sited NATIVE oracle: bin Y co-sited and U / V through the exact
      /// forward V-fold oracle to `ow x oh`, then convert ONCE via a `Yuv444pN` sink.
      #[allow(clippy::too_many_arguments)]
      fn top_native_oracle(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let ch = sh / 2;
        let yb = bin_cosited_u16(y, sw, sh, ow, oh);
        let ub = bin_chroma_top_u16(u, sw, ch, ow, oh);
        let vb = bin_chroma_top_u16(v, sw, ch, ow, oh);
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$M444>::new(ow, oh)
            .with_simd(simd)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb16)
            .unwrap();
          let f = $F444::new(
            &yb, &ub, &vb, ow as u32, oh as u32, ow as u32, ow as u32, ow as u32,
          );
          $w444(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// The top-sited RGB-domain oracle: reconstruct U / V to full height with the
      /// forward top blend then run that `Yuv444pN` frame through the given resampler
      /// (convert-each-row-then-bin). `filter = true` uses the Triangle tier.
      #[allow(clippy::too_many_arguments)]
      fn top_rgb_domain_oracle(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        simd: bool,
        filter: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let (uf, vf) = recon_full_top_u16(u, v, sw, sh);
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        let f = $F444::new(
          y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
        );
        if filter {
          let mut sink = MixedSinker::<$M444, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          $w444(&f, FR, M, &mut sink).unwrap();
        } else {
          let mut sink =
            MixedSinker::<$M444, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
              .unwrap()
              .with_native(false)
              .with_simd(simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgb_u16(&mut rgb16)
              .unwrap();
          $w444(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn top_native_equals_code_domain_oracle() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          assert_eq!(
            run(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Top, true, true),
            top_native_oracle(&y, &u, &v, sw, sh, ow, oh, true),
            "native top == code-domain oracle ({sw}x{sh}->{ow}x{oh})"
          );
          // `TopLeft` (v=0, no horizontal phase) decodes identically.
          assert_eq!(
            run(
              &y,
              &u,
              &v,
              sw,
              sh,
              ow,
              oh,
              ChromaLocation::TopLeft,
              true,
              true
            ),
            run(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Top, true, true),
            "TopLeft == Top ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn top_row_stage_equals_rgb_reconstruct_then_bin() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          assert_eq!(
            run(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Top, false, true).0,
            top_rgb_domain_oracle(&y, &u, &v, sw, sh, ow, oh, true, false).0,
            "row-stage top == reconstruct-then-bin ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn top_filter_equals_rgb_reconstruct_then_filter() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          assert_eq!(
            run_filter(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Top, true).0,
            top_rgb_domain_oracle(&y, &u, &v, sw, sh, ow, oh, true, true).0,
            "filter top == reconstruct-then-filter ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn top_simd_matches_scalar() {
        let (y, u, v) = vramp(8, 8);
        for native in [true, false] {
          assert_eq!(
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true),
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, false),
            "top SIMD vs scalar must agree (native={native})"
          );
        }
        assert_eq!(
          run_filter(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, true),
          run_filter(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, false),
          "top filter SIMD vs scalar must agree"
        );
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn top_be_matches_le() {
        let (y, u, v) = vramp(8, 8);
        for native in [true, false] {
          assert_eq!(
            run_be(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native),
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true),
            "BE top decode must equal LE (native={native})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn top_differs_from_cosited_and_bottom_on_a_vertical_chroma_ramp() {
        let (y, u, v) = vramp(8, 8);
        for native in [true, false] {
          let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
          let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
          let top = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true);
          assert_ne!(top, cos, "top must differ from co-sited (native={native})");
          assert_ne!(top, bot, "top must differ from bottom (native={native})");
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn top_equals_cosited_on_flat_chroma() {
        let (y, u, v) = flat(8, 8);
        for native in [true, false] {
          assert_eq!(
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true),
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Top, native, true),
            "top must equal co-sited on flat chroma (native={native})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn top_identity_resample_matches_direct_decode_all_outputs() {
        // Identity dims: the row-stage resample reconstructs full-width chroma with
        // the SAME forward top kernel as the direct (non-resample) decode, so routing
        // Top through the resample preserves EVERY output byte-for-byte. TopLeft too
        // (identical v=0 fold).
        let (y, u, v) = vramp(8, 8);
        for loc in [ChromaLocation::Top, ChromaLocation::TopLeft] {
          assert_eq!(
            identity_resample_full(&y, &u, &v, 8, 8, loc.clone(), true),
            direct_bottom_full(&y, &u, &v, 8, 8, loc.clone(), true),
            "identity resample {loc:?} == direct decode (all outputs)"
          );
        }
      }
    }
  };
}

hibit_440_resample_siting!(
  p10,
  10,
  Yuv440p10,
  Yuv440p10Frame,
  yuv440p10_to,
  Yuv440p10<true>,
  Yuv440p10BeFrame,
  yuv440p10_to_endian,
  Yuv444p10,
  Yuv444p10Frame,
  yuv444p10_to,
  Yuv440p10Row
);
hibit_440_resample_siting!(
  p12,
  12,
  Yuv440p12,
  Yuv440p12Frame,
  yuv440p12_to,
  Yuv440p12<true>,
  Yuv440p12BeFrame,
  yuv440p12_to_endian,
  Yuv444p12,
  Yuv444p12Frame,
  yuv444p12_to,
  Yuv440p12Row
);
