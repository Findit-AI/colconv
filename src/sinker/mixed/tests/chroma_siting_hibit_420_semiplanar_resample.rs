//! Chroma-siting-aware fused-downscale coverage for the HIGH-BIT **semi-planar**
//! 4:2:0 P-format family — `P010` / `P012` / `P016` (RFC #238 S6b), the `u16`
//! semi-planar twin of the planar `chroma_siting_hibit_420_resample` (S6a) and
//! the high-bit sibling of the 8-bit `chroma_siting_nv12_resample` (S3b).
//!
//! 4:2:0 subsamples chroma 2:1 horizontally AND vertically. S6b routes the
//! HORIZONTAL centered siting (`Center` / `Top` / `Bottom`,
//! [`chroma_420_center_sited_h`](super::super::chroma_420_center_sited_h))
//! through the resample; the VERTICAL chroma pairing stays co-sited
//! (`v_phase = 0`, today's box — `Bottom`'s vertical blend is a later high-bit
//! stage). The interleaved `U V U V …` chroma is de-interleaved to the SAME
//! half-width U / V planes a `Yuv420pN` frame holds, so every centered P-format
//! resample is **bit-identical** to the (S6a-verified) centered `Yuv420pN`
//! resample of those planes — the strongest catch for a U/V swap in the
//! de-interleave. Concretely:
//!  - the **native fast tier** (`with_native(true)`) folds the #302 `1/4`–`3/4`
//!    triangle into the [`ResamplePlan::area_chroma_420`] chroma weights and bins
//!    the SUBSAMPLED `u16` chroma directly — its code-domain twin is
//!    [`bin_chroma_centered_u16`] (a single round over the ×4 reconstruction,
//!    HORIZONTAL over `2·cw`, VERTICAL a co-sited box over the `ch` chroma rows);
//!  - the **encoded row-stage tier** (`with_native(false)`) and the
//!    **single-kernel filter tier** reconstruct full-width interleaved `u16`
//!    chroma per source row (each luma row `r` takes chroma row `r / 2`,
//!    horizontally upsampled) then convert-then-bin 4:4:4 — the RGB-domain
//!    reconstruct-then-bin, pinned against a `Yuv444pN` resample of the
//!    independently reconstructed planes.
//!
//! The P-format wire is MSB-aligned (`logical << (16 - BITS)`); the logical
//! values decode identically to the low-packed `Yuv420pN` / `Yuv444pN` oracle
//! planes, so the siting math is depth- and endian-independent (pinned by the
//! BE↔LE parity test). Representative depths `p010` (low-packed native codes) and
//! `p016` (full u16) cover the family; every centered assertion is an EXACT match
//! (single rounding), never a tolerance. Source heights are EVEN so the plan's
//! luma-domain `area_halved(sh, oh)` V axis equals the co-sited box over the
//! `ch = sh / 2` chroma rows.

use crate::{
  ChromaLocation, ColorMatrix, PixelSink,
  resample::{AreaResampler, FilteredResampler, Triangle},
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    P010, P010Row, P012, P012Row, P016, P016Row, Yuv420p10, Yuv420p12, Yuv420p16, Yuv444p10,
    Yuv444p12, Yuv444p16, p010_to, p010_to_endian, p012_to, p012_to_endian, p016_to,
    p016_to_endian, yuv420p10_to, yuv420p12_to, yuv420p16_to, yuv444p10_to, yuv444p12_to,
    yuv444p16_to,
  },
};
use mediaframe::frame::{
  P010BeFrame, P010Frame, P012BeFrame, P012Frame, P016BeFrame, P016Frame, Yuv420p10Frame,
  Yuv420p12Frame, Yuv420p16Frame, Yuv444p10Frame, Yuv444p12Frame, Yuv444p16Frame,
};

const M: ColorMatrix = ColorMatrix::Bt601;
const FR: bool = true;

/// Round-half-up integer divide.
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

/// The EXACT centered chroma oracle (`u16`) for the native tier: reconstruct the
/// half-width `cw x ch` chroma to full width with the #302 `1/4`–`3/4` triangle
/// kept UNROUNDED (scaled ×4 to stay integral), then box-average to `ow x oh` —
/// HORIZONTAL over `2·cw`, VERTICAL a co-sited box over the `ch` chroma rows
/// (4:2:0 vertical stays co-sited) — with a SINGLE round-half-up over
/// `4·(2·cw)·ch`. The code-domain twin the folded
/// [`ResamplePlan::area_chroma_420`] weights realize (EVEN `sh`, `ch = sh / 2`,
/// where the plan's luma-domain `area_halved(sh, oh)` V axis is exactly `2·` the
/// box over `ch`, cancelled by its `src_h = sh`).
fn bin_chroma_centered_u16(c: &[u16], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u16> {
  let full = 2 * cw;
  let mut r4 = vec![0u64; full * ch];
  for r in 0..ch {
    let row = &c[r * cw..r * cw + cw];
    for j in 0..cw {
      let l = u64::from(row[j.saturating_sub(1)]);
      let m = u64::from(row[j]);
      let rt = u64::from(row[if j + 1 < cw { j + 1 } else { j }]);
      r4[r * full + 2 * j] = l + 3 * m;
      r4[r * full + 2 * j + 1] = 3 * m + rt;
    }
  }
  let hw = area_weights(full, ow);
  let vw = area_weights(ch, oh);
  let denom = (4 * full * ch) as u64;
  let mut out = vec![0u16; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * r4[(vs + dy) * full + hs + dx];
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u16;
    }
  }
  out
}

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
/// ratios, EVEN source heights only (so the vertical luma-domain pairing equals
/// the co-sited chroma box). Widths are even (4:2:0 requires it).
const GEOMS: [(usize, usize, usize, usize); 4] =
  [(8, 8, 4, 4), (8, 8, 5, 3), (12, 8, 4, 4), (16, 8, 6, 5)];

macro_rules! hibit_420_semiplanar_resample_siting {
  (
    $mod:ident, $bits:expr,
    $M420:ident, $F420:ident, $w420:ident,
    $M420be:ty, $F420be:ident, $w420be:ident, $Row:ident,
    $Mplanar:ident, $Fplanar:ident, $wplanar:ident,
    $M444:ident, $F444:ident, $w444:ident
  ) => {
    mod $mod {
      use super::*;

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = 1u16 << ($bits - 1);

      /// MSB-aligns a LOGICAL sample into the P-format wire `u16`:
      /// `value << (16 - BITS)` (P010 `<< 6`, P016 `<< 0`).
      fn pack(v: u16) -> u16 {
        v << (16 - $bits)
      }

      /// Pack the logical Y plane into the MSB-aligned wire form.
      fn pack_y(y: &[u16]) -> Vec<u16> {
        y.iter().map(|&v| pack(v)).collect()
      }

      /// Interleave the half-width logical U / V into a P-format semi-planar
      /// chroma plane (`U` at the even element, `V` at the odd), MSB-aligned,
      /// `2·cw = sw` u16 per row, HALF-height (`ch = sh / 2` rows — one chroma
      /// row per luma-row pair).
      fn interleave_pack(u: &[u16], v: &[u16], sw: usize, sh: usize) -> Vec<u16> {
        let (cw, ch) = (sw / 2, sh / 2);
        let mut uv = vec![0u16; sw * ch];
        for r in 0..ch {
          for c in 0..cw {
            uv[r * sw + 2 * c] = pack(u[r * cw + c]);
            uv[r * sw + 2 * c + 1] = pack(v[r * cw + c]);
          }
        }
        uv
      }

      /// A `Yuv420pN` fixture (`cw = sw / 2`, `ch = sh / 2`) with a strong
      /// HORIZONTAL chroma ramp (so the centered triangle genuinely differs from
      /// the co-sited nearest decode) plus a per-row tilt (a vertical mistake
      /// would show). Low-packed native codes so every kernel sees real math.
      /// Returns LOGICAL planes.
      fn ramp(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let (cw, ch) = (sw / 2, sh / 2);
        let step = (MASK as u32 / 16).max(1);
        let mut y = vec![0u16; sw * sh];
        let mut u = vec![0u16; cw * ch];
        let mut v = vec![0u16; cw * ch];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 37) & MASK as u32) as u16;
        }
        for r in 0..ch {
          for c in 0..cw {
            u[r * cw + c] = (step * c as u32 + step + r as u32 * 5).min(MASK as u32) as u16;
            v[r * cw + c] = (MASK as u32).saturating_sub(step * c as u32).max(step) as u16;
          }
        }
        (y, u, v)
      }

      /// Flat chroma: the centered triangle of a constant is that constant, so
      /// centered must equal co-sited. Luma still varies.
      fn flat(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let (cw, ch) = (sw / 2, sh / 2);
        let mut y = vec![0u16; sw * sh];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 29) & MASK as u32) as u16;
        }
        (y, vec![MID; cw * ch], vec![MID; cw * ch])
      }

      /// Drive an LE `P0xx` area resample for `rgb` (u8) + `rgb_u16`, at `loc`
      /// siting and `native` tier. Logical planes are packed to the MSB-aligned
      /// interleaved wire form.
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
        let (y_wire, uv_wire) = (pack_y(y), interleave_pack(u, v, sw, sh));
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink =
            MixedSinker::<$M420, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
              .unwrap()
              .with_native(native)
              .with_chroma_location(loc)
              .with_simd(simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgb_u16(&mut rgb16)
              .unwrap();
          let f = $F420::new(
            &y_wire, &uv_wire, sw as u32, sh as u32, sw as u32, sw as u32,
          );
          $w420(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// Drive an LE `P0xx` single-kernel filter resample (Triangle).
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
        let (y_wire, uv_wire) = (pack_y(y), interleave_pack(u, v, sw, sh));
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$M420, FilteredResampler<Triangle>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, Triangle),
          )
          .unwrap()
          .with_chroma_location(loc)
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          let f = $F420::new(
            &y_wire, &uv_wire, sw as u32, sh as u32, sw as u32, sw as u32,
          );
          $w420(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// Drive a BE `P0xx` area resample (wire re-encoded BE).
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
        let (y_wire, uv_wire) = (as_be(&pack_y(y)), as_be(&interleave_pack(u, v, sw, sh)));
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$M420be, AreaResampler>::with_resampler(
            sw,
            sh,
            AreaResampler::to(ow, oh),
          )
          .unwrap()
          .with_native(native)
          .with_chroma_location(loc)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          let f = $F420be::try_new(
            &y_wire, &uv_wire, sw as u32, sh as u32, sw as u32, sw as u32,
          )
          .unwrap();
          $w420be(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// The planar `Yuv420pN` twin of [`run`]: the SAME logical U / V driven
      /// through a planar 4:2:0 resample (S6a). A centered semi-planar decode
      /// must equal this byte-for-byte — the de-interleave / U-V-swap catch.
      #[allow(clippy::too_many_arguments)]
      fn run_yuv420p(
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
        let cw = sw / 2;
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb16 = vec![0u16; ow * oh * 3];
        {
          let mut sink = MixedSinker::<$Mplanar, AreaResampler>::with_resampler(
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
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          let f = $Fplanar::new(
            y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
          );
          $wplanar(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      /// The centered NATIVE code-domain oracle: bin Y co-sited and U / V through
      /// the exact centered chroma oracle to `ow x oh` (LOGICAL), then convert
      /// ONCE at output width via an identity `Yuv444pN` sink.
      #[allow(clippy::too_many_arguments)]
      fn native_oracle(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let (cw, ch) = (sw / 2, sh / 2);
        let yb = bin_cosited_u16(y, sw, sh, ow, oh);
        let ub = bin_chroma_centered_u16(u, cw, ch, ow, oh);
        let vb = bin_chroma_centered_u16(v, cw, ch, ow, oh);
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

      /// The centered RGB-domain oracle: reconstruct U / V to full width (LOGICAL
      /// u16) with the #302 kernel — each luma row `r` taking chroma row `r / 2`
      /// (the co-sited vertical replication) — then run that full-resolution
      /// `Yuv444pN` frame through the given resampler (convert-each-row-then-bin,
      /// exactly what the row-stage / filter arms do). `filter = true` uses the
      /// Triangle filter tier.
      #[allow(clippy::too_many_arguments)]
      fn rgb_domain_oracle(
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

      // ---- co-sited byte-identity (the regression contract) ----------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn cosited_group_is_byte_identical_across_tiers() {
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
              ChromaLocation::TopLeft,
              ChromaLocation::BottomLeft,
              ChromaLocation::Unknown(7),
            ] {
              assert_eq!(
                run(&y, &u, &v, sw, sh, ow, oh, loc, native, true),
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
            run_filter(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Left, true),
            base,
            "co-sited filter must stay byte-identical ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      // ---- centered semi-planar == centered planar Yuv420p (U/V-swap catch) -

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_equals_planar_yuv420p_across_tiers() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          // Center / Top are horizontal-only; Bottom folds the vertical `v = 1`
          // blend. RFC #238 S6e routes Bottom through the semi-planar P0xx
          // resample too (matching the planar S6d fold), so P0xx Bottom is once
          // again byte-identical to the planar `Yuv420pN` Bottom across every
          // tier — the strongest catch for a U/V swap or a mis-sited vertical
          // blend in the de-interleave.
          for loc in [
            ChromaLocation::Center,
            ChromaLocation::Top,
            ChromaLocation::Bottom,
          ] {
            for native in [true, false] {
              assert_eq!(
                run(&y, &u, &v, sw, sh, ow, oh, loc, native, true),
                run_yuv420p(&y, &u, &v, sw, sh, ow, oh, loc, native, true),
                "centered semi-planar {loc:?} must equal centered planar Yuv420p \
                 (native={native}, {sw}x{sh}->{ow}x{oh})"
              );
            }
          }
          // The filter tier too.
          assert_eq!(
            run_filter(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Center, true),
            {
              let cw = sw / 2;
              let mut rgb = vec![0u8; ow * oh * 3];
              let mut rgb16 = vec![0u16; ow * oh * 3];
              let mut sink = MixedSinker::<$Mplanar, FilteredResampler<Triangle>>::with_resampler(
                sw,
                sh,
                FilteredResampler::new(ow, oh, Triangle),
              )
              .unwrap()
              .with_chroma_location(ChromaLocation::Center)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgb_u16(&mut rgb16)
              .unwrap();
              let f = $Fplanar::new(
                &y, &u, &v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
              );
              $wplanar(&f, FR, M, &mut sink).unwrap();
              (rgb, rgb16)
            },
            "centered semi-planar filter must equal centered planar Yuv420p filter \
             ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      // ---- centered native == the exact code-domain oracle -----------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_native_equals_code_domain_oracle() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          let want = native_oracle(&y, &u, &v, sw, sh, ow, oh, true);
          // Center / Top: the horizontal centered phase, vertical co-sited.
          for loc in [ChromaLocation::Center, ChromaLocation::Top] {
            assert_eq!(
              run(&y, &u, &v, sw, sh, ow, oh, loc, true, true),
              want,
              "centered native {loc:?} must equal the code-domain oracle \
               ({sw}x{sh}->{ow}x{oh})"
            );
          }
        }
      }

      // ---- centered row-stage / filter == RGB-domain reconstruct-then-bin --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_row_stage_equals_rgb_reconstruct_then_bin() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          let want = rgb_domain_oracle(&y, &u, &v, sw, sh, ow, oh, true, false);
          for loc in [ChromaLocation::Center, ChromaLocation::Top] {
            assert_eq!(
              run(&y, &u, &v, sw, sh, ow, oh, loc, false, true),
              want,
              "centered row-stage {loc:?} must equal the RGB-domain oracle \
               ({sw}x{sh}->{ow}x{oh})"
            );
          }
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_filter_equals_rgb_reconstruct_then_bin() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = ramp(sw, sh);
          let want = rgb_domain_oracle(&y, &u, &v, sw, sh, ow, oh, true, true);
          assert_eq!(
            run_filter(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Center, true),
            want,
            "centered filter must equal the RGB-domain Triangle oracle \
             ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      // ---- SIMD == scalar --------------------------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_simd_matches_scalar() {
        let (y, u, v) = ramp(8, 8);
        for native in [true, false] {
          assert_eq!(
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true),
            run(
              &y,
              &u,
              &v,
              8,
              8,
              4,
              4,
              ChromaLocation::Center,
              native,
              false
            ),
            "centered SIMD vs scalar must agree (native={native})"
          );
        }
        assert_eq!(
          run_filter(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, true),
          run_filter(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, false),
          "centered filter SIMD vs scalar must agree"
        );
      }

      // ---- centered differs from co-sited on a ramp, equals it on flat -----

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_differs_from_cosited_on_a_chroma_ramp() {
        let (y, u, v) = ramp(8, 8);
        for native in [true, false] {
          let cos = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true);
          let cen = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
          assert_ne!(
            cos, cen,
            "centered must differ from co-sited on a chroma ramp (native={native})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_equals_cosited_on_flat_chroma() {
        let (y, u, v) = flat(8, 8);
        for native in [true, false] {
          assert_eq!(
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Left, native, true),
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true),
            "centered must equal co-sited on flat chroma (native={native})"
          );
        }
      }

      // ---- wire endianness is siting-independent ---------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn be_centered_matches_le() {
        let (y, u, v) = ramp(8, 8);
        for native in [true, false] {
          assert_eq!(
            run_be(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native),
            run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true),
            "BE centered decode must equal LE for the same logical planes (native={native})"
          );
        }
      }

      // ---- mid-frame siting change is rejected across tiers ----------------

      /// Accept row 0 at `loc1` (freezes the phase), flip to `loc2`, feed the
      /// IN-SEQUENCE row 1 (chroma row `1 / 2 = 0`, the SAME interleaved chroma
      /// row the walker shares), and return its `process` result.
      fn flip_row1<R>(
        mut sink: MixedSinker<'_, $M420, R>,
        y: &[u16],
        u: &[u16],
        v: &[u16],
        loc1: ChromaLocation,
        loc2: ChromaLocation,
      ) -> Result<(), MixedSinkerError> {
        let (y_wire, uv_wire) = (pack_y(y), interleave_pack(u, v, 8, 8));
        sink.set_chroma_location(loc1);
        PixelSink::begin_frame(&mut sink, 8, 8).unwrap();
        let row0 = $Row::new(&y_wire[0..8], &uv_wire[0..8], 0, M, FR);
        PixelSink::process(&mut sink, row0).unwrap();
        sink.set_chroma_location(loc2);
        let row1 = $Row::new(&y_wire[8..16], &uv_wire[0..8], 1, M, FR);
        PixelSink::process(&mut sink, row1)
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn mid_frame_siting_change_rejected() {
        let (y, u, v) = ramp(8, 8);
        for (loc1, loc2) in [
          (ChromaLocation::Center, ChromaLocation::Left),
          (ChromaLocation::Left, ChromaLocation::Center),
        ] {
          // Native fast tier.
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink =
            MixedSinker::<$M420, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(true)
              .with_rgb(&mut rgb)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "native {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Encoded row-stage tier.
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink =
            MixedSinker::<$M420, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
              .unwrap()
              .with_native(false)
              .with_rgb(&mut rgb)
              .unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "row-stage {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );

          // Filter tier.
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink = MixedSinker::<$M420, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
        }
      }

      // ==== RFC #238 S6e: bottom-sited (v = 1) VERTICAL fold ================
      //
      // The `Bottom` vertical blend runs on the SAME logical U / V a `Yuv420pN`
      // frame holds (the de-interleave produces the identical half-width planes),
      // so every oracle here operates on those LOGICAL planes and decodes through
      // an identity / resampled `Yuv444pN` sink — byte-identical to the planar
      // S6d oracles. The P-format wire is MSB-aligned, but the reconstructed
      // logical chroma is depth- and endian-independent (pinned by the BE↔LE and
      // planar cross-checks).

      /// Flat-luma fixture with a strong per-ROW chroma step (flat across
      /// columns), so the vertical `Bottom` fold is observable in isolation: a
      /// horizontal-only siting leaves it untouched, the `v = 1` blend visibly
      /// moves it. `sw` / `sh` must be even. Returns LOGICAL planes.
      fn vramp(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let (cw, ch) = (sw / 2, sh / 2);
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
        (y, u, v)
      }

      /// The EXACT bottom-sited (`v = 1`) chroma oracle for the native tier: the
      /// centered horizontal `1/4`–`3/4` triangle (×4) composed with the vertical
      /// `v = 1` triangle (×2 — even luma row `2i` box-blends chroma rows
      /// `{i-1, i}` with weights `{1, 1}`, odd row `2i+1` takes chroma row `i`
      /// with weight 2, top-edge clamp), the combined ×8 UNROUNDED reconstruction
      /// box-averaged to `ow x oh` with a SINGLE round-half-up — the code-domain
      /// twin the folded `area_chroma_phased_v` V axis realizes (EVEN `sh`,
      /// `ch = sh / 2`).
      fn bin_chroma_bottom_u16(c: &[u16], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u16> {
        let full = 2 * cw;
        let sh = 2 * ch;
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
            r8[r * full + 2 * j] = u64::from(l + 3 * m);
            r8[r * full + 2 * j + 1] = u64::from(3 * m + rt);
          }
        }
        let hw = area_weights(full, ow);
        let vw = area_weights(sh, oh);
        let denom = (8 * full * sh) as u64; // ×8 (×2 V, ×4 H) × the box normalization
        let mut out = vec![0u16; ow * oh];
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
            out[oy * ow + ox] = rdhu(s, denom) as u16;
          }
        }
        out
      }

      /// The bottom-sited NATIVE code-domain oracle: bin Y co-sited and U / V
      /// through the exact bottom V-fold oracle to `ow x oh` (LOGICAL), then
      /// convert ONCE at output width via an identity `Yuv444pN` sink.
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
        let (cw, ch) = (sw / 2, sh / 2);
        let yb = bin_cosited_u16(y, sw, sh, ow, oh);
        let ub = bin_chroma_bottom_u16(u, cw, ch, ow, oh);
        let vb = bin_chroma_bottom_u16(v, cw, ch, ow, oh);
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

      /// Reconstruct `Yuv420pN` chroma to full resolution (`sw x sh`, `u16`) for
      /// the bottom-sited decode — even rows box-blend chroma rows `i - 1`
      /// (clamped) and `i` (round-half-up), odd rows take chroma row `i`, each then
      /// horizontally upsampled with the #302 centered kernel. The shared
      /// reconstruction step for every reconstruct-then-bin bottom oracle.
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

      /// The bottom-sited RGB-domain oracle: reconstruct U / V to full width with
      /// the vertical bottom blend then run that `Yuv444pN` frame through the given
      /// resampler (convert-each-row-then-bin) — exactly what the row-stage /
      /// filter arms do for `Bottom`. `filter = true` uses the Triangle tier.
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

      /// Direct (non-resample, identity dims) `P0xx` `Bottom` decode — the
      /// delay-line kernel path (de-interleave + vertical blend), the
      /// already-validated identity reference the resample path must match at
      /// identity dimensions. Logical planes are packed to the MSB-aligned
      /// interleaved wire form.
      fn direct_bottom(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        sw: usize,
        sh: usize,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let (y_wire, uv_wire) = (pack_y(y), interleave_pack(u, v, sw, sh));
        let mut rgb = vec![0u8; sw * sh * 3];
        let mut rgb16 = vec![0u16; sw * sh * 3];
        {
          let mut sink = MixedSinker::<$M420>::new(sw, sh)
            .with_chroma_location(ChromaLocation::Bottom)
            .with_simd(simd)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb16)
            .unwrap();
          let f = $F420::new(
            &y_wire, &uv_wire, sw as u32, sh as u32, sw as u32, sw as u32,
          );
          $w420(&f, FR, M, &mut sink).unwrap();
        }
        (rgb, rgb16)
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_native_equals_code_domain_oracle() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = vramp(sw, sh);
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

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_row_stage_equals_rgb_reconstruct_then_bin() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = vramp(sw, sh);
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
            "bottom semi-planar row-stage must equal the RGB-domain reconstruct-then-bin \
             oracle ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_filter_equals_rgb_reconstruct_then_bin() {
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = vramp(sw, sh);
          let o = bottom_rgb_domain_oracle(&y, &u, &v, sw, sh, ow, oh, true, true);
          assert_eq!(
            run_filter(&y, &u, &v, sw, sh, ow, oh, ChromaLocation::Bottom, true),
            o,
            "bottom semi-planar filter must equal the RGB-domain Triangle oracle \
             ({sw}x{sh}->{ow}x{oh})"
          );
        }
      }

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

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_equals_planar_yuv420p_across_tiers() {
        // The de-interleave / U-V-swap / mis-sited-vertical-blend catch: a
        // semi-planar `P0xx` Bottom decode must equal the (S6d-verified) planar
        // `Yuv420pN` Bottom decode of the SAME logical planes, on every tier.
        for (sw, sh, ow, oh) in GEOMS {
          let (y, u, v) = vramp(sw, sh);
          for native in [true, false] {
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
                native,
                true
              ),
              run_yuv420p(
                &y,
                &u,
                &v,
                sw,
                sh,
                ow,
                oh,
                ChromaLocation::Bottom,
                native,
                true
              ),
              "bottom semi-planar must equal bottom planar Yuv420p \
               (native={native}, {sw}x{sh}->{ow}x{oh})"
            );
          }
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_differs_from_center_on_a_vertical_chroma_ramp() {
        let (y, u, v) = vramp(8, 8);
        for native in [true, false] {
          let cen = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Center, native, true);
          let bot = run(&y, &u, &v, 8, 8, 4, 4, ChromaLocation::Bottom, native, true);
          assert_ne!(
            cen, bot,
            "bottom must differ from center on a vertical chroma ramp (native={native})"
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

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_identity_resample_matches_direct_decode() {
        // Identity dims (ow=sw, oh=sh): the resample reconstructs full-width
        // interleaved chroma with the SAME bottom kernel as the direct decode and
        // bins with pass-through area weights, so routing Bottom through the
        // resample preserves the direct decode byte-for-byte.
        let (y, u, v) = vramp(8, 8);
        assert_eq!(
          run(&y, &u, &v, 8, 8, 8, 8, ChromaLocation::Bottom, false, true),
          direct_bottom(&y, &u, &v, 8, 8, true),
          "identity resample bottom == direct decode"
        );
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn bottom_final_even_row_reserve_alloc_failure_retries_with_blend() {
        // ODD source height => the FINAL luma row is EVEN, so its Bottom vertical
        // blend reads the PREDECESSOR chroma row. If the `chroma_prev` reserve
        // fails on that final row, the retry must STILL blend with the predecessor
        // (not clamp): the reconstruction reads the lookback only AFTER the
        // reserve, and the resample arms DEFER the stage until the commit accepts
        // the row (#180), so a rejected row leaves the lookback untouched. Feeds
        // MSB-aligned wire rows (packed Y + interleaved UV) per row.
        use super::super::MixedSinkerError;
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
        // Pack to the P-format wire: MSB-aligned Y + interleaved MSB-aligned UV,
        // `ch = div_ceil(sh, 2)` chroma rows (one per luma-row pair, incl. the
        // trailing odd row's).
        let y_wire = pack_y(&y);
        let mut uv_wire = vec![0u16; sw * ch];
        for r in 0..ch {
          for c in 0..cw {
            uv_wire[r * sw + 2 * c] = pack(u[r * cw + c]);
            uv_wire[r * sw + 2 * c + 1] = pack(v[r * cw + c]);
          }
        }
        let run_frame = |arm_fail: bool, loc: ChromaLocation| -> Vec<u8> {
          let mut rgb = vec![0u8; ow * oh * 3];
          {
            let mut sink = MixedSinker::<$M420, AreaResampler>::with_resampler(
              sw,
              sh,
              AreaResampler::to(ow, oh),
            )
            .unwrap()
            .with_native(false)
            .with_chroma_location(loc)
            .with_rgb(&mut rgb)
            .unwrap();
            sink.begin_frame(sw as u32, sh as u32).unwrap();
            let feed = |sink: &mut MixedSinker<'_, $M420, AreaResampler>, r: usize| {
              let yr = &y_wire[r * sw..(r + 1) * sw];
              let cr = r / 2;
              let uvr = &uv_wire[cr * sw..(cr + 1) * sw];
              sink.process($Row::new(yr, uvr, r, M, FR))
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
                "armed final-row chroma_prev reserve must surface AllocationFailed, got {err:?}"
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
        // Non-vacuous: the bottom vertical blend genuinely moves the final even
        // row vs the vertically-co-sited Center decode.
        assert_ne!(
          reference,
          run_frame(false, ChromaLocation::Center),
          "the bottom vertical blend must move the final even row vs co-sited Center"
        );
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn mid_frame_center_bottom_flip_rejected() {
        // Center and Bottom are BOTH horizontally centered, so the centered flag
        // alone cannot tell them apart; the frozen VERTICAL phase rejects the flip.
        let (y, u, v) = vramp(8, 8);
        for (loc1, loc2) in [
          (ChromaLocation::Center, ChromaLocation::Bottom),
          (ChromaLocation::Bottom, ChromaLocation::Center),
        ] {
          for native in [true, false] {
            let mut rgb = vec![0u8; 4 * 4 * 3];
            let sink =
              MixedSinker::<$M420, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
                .unwrap()
                .with_native(native)
                .with_rgb(&mut rgb)
                .unwrap();
            let err = flip_row1(sink, &y, &u, &v, loc1, loc2).unwrap_err();
            assert!(
              matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
              "native={native} {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
            );
          }
          let mut rgb = vec![0u8; 4 * 4 * 3];
          let sink = MixedSinker::<$M420, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1, loc2).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
        }
      }
    }
  };
}

hibit_420_semiplanar_resample_siting!(
  p010,
  10,
  P010,
  P010Frame,
  p010_to,
  P010<true>,
  P010BeFrame,
  p010_to_endian,
  P010Row,
  Yuv420p10,
  Yuv420p10Frame,
  yuv420p10_to,
  Yuv444p10,
  Yuv444p10Frame,
  yuv444p10_to
);
hibit_420_semiplanar_resample_siting!(
  p012,
  12,
  P012,
  P012Frame,
  p012_to,
  P012<true>,
  P012BeFrame,
  p012_to_endian,
  P012Row,
  Yuv420p12,
  Yuv420p12Frame,
  yuv420p12_to,
  Yuv444p12,
  Yuv444p12Frame,
  yuv444p12_to
);
hibit_420_semiplanar_resample_siting!(
  p016,
  16,
  P016,
  P016Frame,
  p016_to,
  P016<true>,
  P016BeFrame,
  p016_to_endian,
  P016Row,
  Yuv420p16,
  Yuv420p16Frame,
  yuv420p16_to,
  Yuv444p16,
  Yuv444p16Frame,
  yuv444p16_to
);
