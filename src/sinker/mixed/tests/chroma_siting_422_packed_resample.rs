//! RFC #238 — chroma-siting-aware 4:2:2 **resample** for the PACKED
//! `Yuyv422` (`Y0 U Y1 V …`), `Uyvy422` (`U Y0 V Y1 …`) and `Yvyu422`
//! (`Y0 V Y1 U …`) formats.
//!
//! The 4:2:2 resample twin of the identity-decode `chroma_siting_422_packed`
//! and the packed sibling of the planar `chroma_siting_422_resample`
//! (`Yuv422p`) / semi-planar `chroma_siting_nv16_resample` (`Nv16`). The
//! fully-interleaved chroma is de-interleaved to the SAME half-width U / V
//! planes a `Yuv422p` frame holds, so every centered packed resample is
//! **bit-identical** to the centered `Yuv422p` resample of those planes — the
//! strongest catch for a U/V swap in the de-interleave. Covers the same tiers
//! as the planar twin minus the two the packed family lacks: there is no
//! separate `hsv_direct` join (native HSV rides `yuv_planar_process_native`,
//! and packed HSV-only is RGB-staged) and no Linear averaging tier.
//!
//! The native oracle (`native_oracle`) is the EXACT code-domain box-average of
//! the UNROUNDED triangle-reconstructed chroma — a SINGLE rounding, the
//! user-approved more-correct form (≤ 1 LSB from reconstruct-to-u8-then-bin).
//! The encoded row-stage oracle (`encoded_oracle_rgb`) IS the RGB-domain
//! reconstruct-then-bin. Both are written independently of the production
//! kernel.

use crate::{
  ChromaLocation, KernelMatrix, PixelSink,
  resample::{AreaResampler, FilteredResampler, Triangle},
  sinker::MixedSinker,
  source::{
    Uyvy422, Uyvy422Row, Yuv422p, Yuv444p, Yuyv422, Yuyv422Row, Yvyu422, Yvyu422Row, uyvy422_to,
    yuv422p_to, yuv444p_to, yuyv422_to, yvyu422_to,
  },
};
use mediaframe::frame::{Uyvy422Frame, Yuv422pFrame, Yuv444pFrame, Yuyv422Frame, Yvyu422Frame};

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
/// (round-half-up) — the reference for a phase-free plane (luma, co-sited chroma).
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

/// The EXACT centered chroma oracle: reconstruct the half-width `cw x ch` chroma
/// to full width with the #302 `1/4`–`3/4` triangle kept UNROUNDED (scaled ×4 to
/// stay integral: `r ∈ {1, 3, 4}`), then box-average to `ow x oh` with a SINGLE
/// round-half-up over `4·(2·cw)·ch`. This is the code-domain twin the folded
/// `ResamplePlan::area_chroma_422` weights realize.
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

/// A `Yuv422p`-plane fixture with a strong HORIZONTAL chroma ramp (so the
/// centered triangle, which pulls neighbours, genuinely differs from the
/// co-sited nearest decode) plus a per-row tilt (a vertical mistake would show).
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

/// A flat-chroma fixture: the centered phase is a no-op on constant chroma, so
/// centered must equal co-sited. Luma still varies.
fn flat_chroma(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / 2;
  let mut y = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 7) % 170) as u8;
  }
  (y, vec![110u8; cw * sh], vec![140u8; cw * sh])
}

/// Pack the half-width planar U / V + full-width Y into a `Yuyv422` plane
/// (`Y0 U Y1 V` per 2-pixel group), `2·sw` bytes per row.
fn pack_yuyv(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize) -> Vec<u8> {
  let cw = sw / 2;
  let mut buf = vec![0u8; 2 * sw * sh];
  for r in 0..sh {
    for c in 0..cw {
      let base = r * 2 * sw + c * 4;
      buf[base] = y[r * sw + c * 2];
      buf[base + 1] = u[r * cw + c];
      buf[base + 2] = y[r * sw + c * 2 + 1];
      buf[base + 3] = v[r * cw + c];
    }
  }
  buf
}

/// Pack into a `Uyvy422` plane (`U Y0 V Y1` per 2-pixel group).
fn pack_uyvy(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize) -> Vec<u8> {
  let cw = sw / 2;
  let mut buf = vec![0u8; 2 * sw * sh];
  for r in 0..sh {
    for c in 0..cw {
      let base = r * 2 * sw + c * 4;
      buf[base] = u[r * cw + c];
      buf[base + 1] = y[r * sw + c * 2];
      buf[base + 2] = v[r * cw + c];
      buf[base + 3] = y[r * sw + c * 2 + 1];
    }
  }
  buf
}

/// Pack into a `Yvyu422` plane (`Y0 V Y1 U` per 2-pixel group).
fn pack_yvyu(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize) -> Vec<u8> {
  let cw = sw / 2;
  let mut buf = vec![0u8; 2 * sw * sh];
  for r in 0..sh {
    for c in 0..cw {
      let base = r * 2 * sw + c * 4;
      buf[base] = y[r * sw + c * 2];
      buf[base + 1] = v[r * cw + c];
      buf[base + 2] = y[r * sw + c * 2 + 1];
      buf[base + 3] = u[r * cw + c];
    }
  }
  buf
}

type Outs = (
  Vec<u8>,
  Vec<u8>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
  Vec<u8>,
  Vec<u16>,
);

/// The `Yuv422p` twin runner: the SAME planar U / V driven through a planar
/// 4:2:2 resample. A centered packed decode must equal this byte-for-byte.
#[allow(clippy::too_many_arguments)]
fn run_yuv422p(
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
/// centered chroma oracle to `ow x oh`, then convert ONCE at output width via an
/// identity `Yuv444p` sink — the exact ground truth the native tier reproduces.
fn native_oracle(y: &[u8], u: &[u8], v: &[u8], sw: usize, sh: usize, ow: usize, oh: usize) -> Outs {
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
/// the #302 kernel (u8), then run that `Yuv444p` frame through a
/// `with_native(false)` RGB-domain resample — convert-each-row-then-bin-RGB.
fn encoded_oracle_rgb(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
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
        .with_rgb(&mut rgb)
        .unwrap();
    let f = Yuv444pFrame::new(
      y, &uf, &vf, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    yuv444p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgb
}

// A per-format macro keeps the two near-identical packed suites in lockstep
// while naming each test after its format (so a failure points at the exact
// byte order — YUYV vs UYVY). `$pack` builds the packed plane from planar
// U / V, `$drv` is the source walker, `$frame` the frame builder, `$row` the
// per-row constructor.
macro_rules! packed_422_siting_resample_suite {
  (
    $modname:ident,
    $fmt:ident,
    $row:ident,
    $frame:ident,
    $drv:ident,
    $pack:ident,
  ) => {
    mod $modname {
      use super::*;

      /// Drive a packed area resample (`sw x sh -> ow x oh`) for the full output
      /// set, at `loc` siting and `native` tier — the packed plane is built from
      /// the planar U / V fixture.
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
        let packed = $pack(y, u, v, sw, sh);
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgba = vec![0u8; ow * oh * 4];
        let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
        let mut luma = vec![0u8; ow * oh];
        let mut luma_u16 = vec![0u16; ow * oh];
        {
          let mut sink =
            MixedSinker::<$fmt, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
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
          let f = $frame::new(&packed, sw as u32, sh as u32, (2 * sw) as u32);
          $drv(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgba, (hh, ss, vv), luma, luma_u16)
      }

      // ---- co-sited byte-identity (the regression contract) ------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn cosited_group_is_byte_identical_across_tiers() {
        // Every co-sited / unspecified siting must produce the byte-identical
        // pre-siting resample (phase 0 → the folded plan is never built), on
        // BOTH tiers. `Unspecified` is the baseline.
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

      // ---- centered native == the exact code-domain oracle -------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_native_equals_code_domain_oracle() {
        // Clean 2:1 and fractional ratios (both axes), for the centered group.
        for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 6, 4, 4)] {
          let (y, u, v) = ramp(sw, sh);
          let o = native_oracle(&y, &u, &v, sw, sh, ow, oh);
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

      // ---- centered encoded row-stage == RGB-domain reconstruct-then-bin -----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_encoded_output_equals_rgb_reconstruct_then_bin() {
        for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3)] {
          let (y, u, v) = ramp(sw, sh);
          let oracle = encoded_oracle_rgb(&y, &u, &v, sw, sh, ow, oh);
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

      // ---- centered packed is bit-identical to centered Yuv422p (swap catch) --

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_packed_equals_centered_yuv422p_area_tiers() {
        // De-interleaving the packed chroma back to planar U / V reconstructs the
        // SAME planes the planar twin holds, so a centered packed area resample
        // is byte-identical to the centered Yuv422p resample of those planes — on
        // BOTH the native fast tier and the encoded row-stage tier. The strongest
        // catch for a U/V swap in the de-pack (`u_off` / `v_off`).
        for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3), (12, 6, 4, 4)] {
          let (y, u, v) = ramp(sw, sh);
          for loc in [
            ChromaLocation::Center,
            ChromaLocation::Top,
            ChromaLocation::Bottom,
          ] {
            for native in [true, false] {
              let packed = run(&y, &u, &v, sw, sh, ow, oh, loc.clone(), native, true);
              let yuv422p = run_yuv422p(&y, &u, &v, sw, sh, ow, oh, loc.clone(), native, true);
              assert_eq!(
                packed, yuv422p,
                "packed vs Yuv422p {loc:?} native={native} {sw}x{sh}->{ow}x{oh}"
              );
            }
          }
        }
      }

      /// A single-kernel Triangle FILTER resample of the centered chroma, as this
      /// packed format and as `Yuv422p` (planar) — the two must agree.
      #[allow(clippy::too_many_arguments)]
      fn filter_rgb_packed(
        y: &[u8],
        u: &[u8],
        v: &[u8],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
      ) -> Vec<u8> {
        let packed = $pack(y, u, v, sw, sh);
        let mut rgb = vec![0u8; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$fmt, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_chroma_location(loc.clone())
          .with_rgb(&mut rgb)
          .unwrap();
          let f = $frame::new(&packed, sw as u32, sh as u32, (2 * sw) as u32);
          $drv(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        rgb
      }

      #[allow(clippy::too_many_arguments)]
      fn filter_rgb_yuv422p(
        y: &[u8],
        u: &[u8],
        v: &[u8],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        loc: ChromaLocation,
      ) -> Vec<u8> {
        let cw = sw / 2;
        let mut rgb = vec![0u8; ow * oh * 3];
        {
          let mut sink = MixedSinker::<Yuv422p, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_chroma_location(loc.clone())
          .with_rgb(&mut rgb)
          .unwrap();
          let f = Yuv422pFrame::new(
            y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
          );
          yuv422p_to(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        rgb
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_packed_equals_centered_yuv422p_filter_tier() {
        // The filter tier's centered reconstruction is the SAME as the planar
        // twin's, so the Triangle-filtered centered packed RGB equals the
        // Yuv422p one; the centered result also genuinely differs from co-sited.
        for (sw, sh, ow, oh) in [(8, 8, 4, 4), (8, 8, 5, 3)] {
          let (y, u, v) = ramp(sw, sh);
          let packed = filter_rgb_packed(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Center);
          let yuv422p = filter_rgb_yuv422p(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Center);
          assert_eq!(packed, yuv422p, "filter centered {sw}x{sh}->{ow}x{oh}");
          let cosited = filter_rgb_packed(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Left);
          assert_ne!(packed, cosited, "filter centered must differ from co-sited");
        }
      }

      // ---- non-vacuous + flat-chroma sanity ----------------------------------

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
        // Sanity: on constant chroma the centered triangle is a no-op, so
        // centered and co-sited agree byte-for-byte.
        let (y, u, v) = flat_chroma(8, 8);
        for native in [true, false] {
          let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
          let cen = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
          assert_eq!(cen.0, cos.0, "flat-chroma rgb (native={native})");
          assert_eq!(cen.2, cos.2, "flat-chroma hsv (native={native})");
        }
      }

      // ---- cross-frame sink reuse rebuilds the phased native join ------------

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
        let packed = $pack(y, u, v, sw, sh);
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgba = vec![0u8; ow * oh * 4];
        let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
        let mut luma = vec![0u8; ow * oh];
        let mut luma_u16 = vec![0u16; ow * oh];
        {
          let mut sink =
            MixedSinker::<$fmt, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
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
          let f = $frame::new(&packed, sw as u32, sh as u32, (2 * sw) as u32);
          sink.set_chroma_location(loc1.clone());
          $drv(&f, FR, sink.set_kernel_matrix(M)).unwrap();
          sink.set_chroma_location(loc2.clone());
          $drv(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgba, (hh, ss, vv), luma, luma_u16)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_join_rebuilds_on_siting_change_across_frames() {
        // Reuse one native-tier sink flipping Left ⇄ Center (both directions):
        // frame 2 must match a FRESH sink for frame 2's siting — no stale-phase
        // carryover (the point-of-use stale-phase drop).
        let (y, u, v) = ramp(8, 8);
        for (a, b) in [
          (ChromaLocation::Left, ChromaLocation::Center),
          (ChromaLocation::Center, ChromaLocation::Left),
        ] {
          let reused = run_reuse_native(&y, &u, &v, 8, 8, 4, 4, a.clone(), b.clone());
          let fresh = run(&y, &u, &v, 8, 8, 4, 4, b.clone(), true, true);
          assert_eq!(
            reused.0, fresh.0,
            "native rgb {a:?}->{b:?} stale-phase carryover"
          );
          assert_eq!(reused.2, fresh.2, "native hsv {a:?}->{b:?}");
          assert_eq!(reused.3, fresh.3, "native luma {a:?}->{b:?}");
          // Non-vacuous: the two sitings genuinely differ on this ramp.
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
      fn native_out_of_sequence_first_row_does_not_drop_the_cached_join() {
        use super::super::super::MixedSinkerError;
        use crate::resample::ResampleError;
        // The point-of-use phase drop is state-atomic: a reused sink, new frame,
        // changed siting, then an OUT-OF-SEQUENCE first process call must be
        // rejected (OutOfSequenceRow) with the cached join INTACT, so a corrected
        // row-0 retry rebuilds cleanly.
        let (y, u, v) = ramp(8, 8);
        let packed = $pack(&y, &u, &v, 8, 8);
        let mut rgb = vec![0u8; 4 * 4 * 3];
        let mut sink =
          MixedSinker::<$fmt, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
            .unwrap()
            .with_native(true)
            .with_chroma_location(ChromaLocation::Left)
            .with_rgb(&mut rgb)
            .unwrap();
        // Frame 1 at Left builds the native join.
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        for r in 0..8 {
          let row = $row::for_tests(&packed[r * 16..r * 16 + 16], r, M, FR);
          PixelSink::process(&mut sink, row).unwrap();
        }
        assert!(
          sink.native_planar.is_some(),
          "frame 1 builds the native join"
        );
        // Frame 2: change siting to Center, then feed an OUT-OF-SEQUENCE first row.
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        sink.set_chroma_location(ChromaLocation::Center);
        let bad = $row::for_tests(&packed[3 * 16..4 * 16], 3, M, FR);
        let err = PixelSink::process(&mut sink, bad).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "out-of-sequence first row must be OutOfSequenceRow, got {err:?}"
        );
        assert!(
          sink.native_planar.is_some(),
          "an out-of-sequence first row must NOT drop the cached native join"
        );
        // The corrected retry (row 0, now rebuilding for Center) succeeds.
        for r in 0..8 {
          let row = $row::for_tests(&packed[r * 16..r * 16 + 16], r, M, FR);
          PixelSink::process(&mut sink, row).unwrap();
        }
      }

      // ---- IN-SEQUENCE mid-frame phase change is rejected across tiers -------

      /// Drive one packed resample frame: `begin_frame`, accept row 0 at `loc1`
      /// (freezes the phase), flip to `loc2`, then feed the IN-SEQUENCE row 1 and
      /// return its `process` result.
      fn in_sequence_flip_row1<R>(
        mut sink: MixedSinker<'_, $fmt, R>,
        packed: &[u8],
        loc1: ChromaLocation,
        loc2: ChromaLocation,
      ) -> Result<(), super::super::super::MixedSinkerError> {
        sink.set_chroma_location(loc1.clone());
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        let row0 = $row::for_tests(&packed[0..16], 0, M, FR);
        PixelSink::process(&mut sink, row0).unwrap();
        sink.set_chroma_location(loc2.clone());
        let row1 = $row::for_tests(&packed[16..32], 1, M, FR);
        PixelSink::process(&mut sink, row1)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn in_sequence_mid_frame_phase_change_rejected_across_tiers() {
        use super::super::super::MixedSinkerError;
        let (y, u, v) = ramp(8, 8);
        let packed = $pack(&y, &u, &v, 8, 8);
        // Both flip directions: Center->Left (drop the phase) and Left->Center
        // (add it). Each must reject the in-sequence row 1 with
        // ChromaSitingChanged.
        for (loc1, loc2) in [
          (ChromaLocation::Center, ChromaLocation::Left),
          (ChromaLocation::Left, ChromaLocation::Center),
        ] {
          // Native fast tier.
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink =
            MixedSinker::<$fmt, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(true)
              .with_rgb(&mut rgb)
              .unwrap();
          let err = in_sequence_flip_row1(sink, &packed, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "native {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Encoded row-stage RGB tier (`with_native(false)`).
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink =
            MixedSinker::<$fmt, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(false)
              .with_rgb(&mut rgb)
              .unwrap();
          let err = in_sequence_flip_row1(sink, &packed, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "encoded-rgb {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // HSV-only row-stage tier (packed HSV-only is RGB-staged, no separate
          // hsv-direct join).
          let (mut hh, mut ss, mut vv) = (vec![0u8; 4 * 4], vec![0u8; 4 * 4], vec![0u8; 4 * 4]);
          let sink =
            MixedSinker::<$fmt, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(false)
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
          let err = in_sequence_flip_row1(sink, &packed, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "hsv {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Filter tier (single-kernel Triangle FilteredResampler).
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink = MixedSinker::<$fmt, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
          let err = in_sequence_flip_row1(sink, &packed, loc1.clone(), loc2.clone()).unwrap_err();
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
        use super::super::super::MixedSinkerError;
        // Advance rows 0,1 (Center), then flip siting mid-frame (Left): the
        // frozen-phase CHECK rejects it with ChromaSitingChanged at the choke
        // point; a mixed-phase frame is never emitted and the frame restarts
        // cleanly (a rejected row mutates no state).
        let (y, u, v) = ramp(8, 8);
        let packed = $pack(&y, &u, &v, 8, 8);
        let mut rgb = vec![0u8; 4 * 4 * 3];
        let mut sink =
          MixedSinker::<$fmt, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
            .unwrap()
            .with_native(true)
            .with_chroma_location(ChromaLocation::Center)
            .with_rgb(&mut rgb)
            .unwrap();
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        for r in 0..2 {
          let row = $row::for_tests(&packed[r * 16..r * 16 + 16], r, M, FR);
          PixelSink::process(&mut sink, row).unwrap();
        }
        sink.set_chroma_location(ChromaLocation::Left);
        let bad = $row::for_tests(&packed[5 * 16..6 * 16], 5, M, FR);
        let err = PixelSink::process(&mut sink, bad).unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
          "mid-frame siting change must be ChromaSitingChanged, got {err:?}"
        );
        // The rejected row mutated no stream state: begin_frame restarts cleanly
        // and a fresh frame at the new siting processes without error.
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        for r in 0..8 {
          let row = $row::for_tests(&packed[r * 16..r * 16 + 16], r, M, FR);
          PixelSink::process(&mut sink, row).unwrap();
        }
      }
    }
  };
}

packed_422_siting_resample_suite!(
  yuyv422,
  Yuyv422,
  Yuyv422Row,
  Yuyv422Frame,
  yuyv422_to,
  pack_yuyv,
);
packed_422_siting_resample_suite!(
  uyvy422,
  Uyvy422,
  Uyvy422Row,
  Uyvy422Frame,
  uyvy422_to,
  pack_uyvy,
);
packed_422_siting_resample_suite!(
  yvyu422,
  Yvyu422,
  Yvyu422Row,
  Yvyu422Frame,
  yvyu422_to,
  pack_yvyu,
);

// ---- the ≤1 LSB single-rounding note, pinned (format-independent) -----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_native_is_within_1_lsb_of_reconstruct_then_bin() {
  // The folded single-rounding native output and the #302 reconstruct-to-u8-
  // then-bin (TWO roundings) agree to ≤ 1 LSB per chroma sample. Compared in the
  // chroma CODE domain so the convert/clamp cannot mask or amplify the gap.
  // `[0, 2, 0, 2]` chroma provably exercises the divergence. The de-pack lands
  // both packed formats on the identical planar U / V, so this is shared.
  let (cw, sh, ow, oh) = (4usize, 2usize, 4usize, 2usize);
  let u: Vec<u8> = (0..cw * sh)
    .map(|i| if i.is_multiple_of(2) { 0 } else { 2 })
    .collect();
  let folded = bin_chroma_centered(&u, cw, sh, ow, oh);
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
  assert_ne!(folded, double, "the ≤1 LSB gap must be exercised");
}
