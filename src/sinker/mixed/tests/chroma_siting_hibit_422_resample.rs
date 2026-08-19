//! Chroma-siting-aware fused-downscale coverage for the HIGH-BIT **planar**
//! 4:2:2 YUV family — `Yuv422p9` … `Yuv422p16` (RFC #238 S5a), the `u16` twin of
//! the 8-bit `chroma_siting_422_resample` suite.
//!
//! 4:2:2 subsamples chroma 2:1 horizontally only. Routing the centered siting
//! (`Center` / `Top` / `Bottom`, [`chroma_422_center_sited_h`]) through the
//! resample means:
//!  - the **native fast tier** (`with_native(true)`) folds the #302 `1/4`–`3/4`
//!    triangle into the [`ResamplePlan::area_chroma_422`] chroma weights and
//!    bins the SUBSAMPLED `u16` chroma directly — its code-domain twin is
//!    [`bin_chroma_centered_u16`] (a single round over the ×4 reconstruction);
//!  - the **encoded row-stage tier** (`with_native(false)`) and the
//!    **single-kernel filter tier** reconstruct full-width `u16` chroma per
//!    source row then convert-then-bin — the RGB-domain
//!    reconstruct-then-bin, pinned against a `Yuv444pN` resample of the
//!    independently reconstructed planes.
//!
//! The co-sited / unspecified group is phase 0 and stays byte-identical to the
//! pre-siting resample (the folded plan at phase 0 = the plain box). Both `u8`
//! and native-`u16` colour are exercised; the wire endianness (`BE`) is siting-
//! independent (decode normalizes before the phase math), pinned by the BE↔LE
//! parity test. Representative depths `p10` (low-packed) and `p16` (full u16)
//! cover the family; every centered assertion is an EXACT match (single
//! rounding), never a tolerance.

use crate::{
  ChromaLocation, KernelMatrix, PixelSink,
  frame::*,
  resample::{AreaResampler, FilteredResampler, Triangle},
  sinker::{MixedSinker, MixedSinkerError},
  source::*,
};

const SRC: usize = 8;
const CW: usize = SRC / 2;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
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

/// The EXACT centered chroma oracle (`u16`): reconstruct the half-width `cw x ch`
/// chroma to full width with the #302 `1/4`–`3/4` triangle kept UNROUNDED
/// (scaled ×4 to stay integral), then box-average to `ow x oh` with a SINGLE
/// round-half-up over `4·(2·cw)·ch`. The code-domain twin the folded
/// [`ResamplePlan::area_chroma_422`] weights realize.
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

macro_rules! hibit_422_resample_siting {
  (
    $mod:ident, $bits:expr,
    $M422:ident, $F422:ident, $w422:ident,
    $M422be:ty, $F422be:ident, $w422be:ident,
    $M444:ident, $F444:ident, $w444:ident,
    $Row:ident
  ) => {
    mod $mod {
      use super::*;

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = 1u16 << ($bits - 1);

      /// A strong HORIZONTAL chroma ramp (so the centered triangle genuinely
      /// differs from the co-sited nearest decode) plus a per-row tilt (a
      /// vertical mistake would show). Low-packed native codes so every kernel
      /// sees real math.
      fn ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let step = (MASK as u32 / 16).max(1);
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; CW * SRC];
        let mut v = vec![0u16; CW * SRC];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 37) & MASK as u32) as u16;
        }
        for r in 0..SRC {
          for c in 0..CW {
            u[r * CW + c] = (step * c as u32 + step + r as u32 * 5).min(MASK as u32) as u16;
            v[r * CW + c] = (MASK as u32).saturating_sub(step * c as u32).max(step) as u16;
          }
        }
        (y, u, v)
      }

      /// Flat chroma: the centered triangle of a constant is that constant, so
      /// centered must equal co-sited. Luma still varies.
      fn flat() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        for (i, p) in y.iter_mut().enumerate() {
          *p = ((40 + i as u32 * 29) & MASK as u32) as u16;
        }
        (y, vec![MID; CW * SRC], vec![MID; CW * SRC])
      }

      /// Drive an LE `Yuv422pN` area resample for `rgb` (u8) + `rgb_u16`, at
      /// `loc` siting and `native` tier.
      fn run(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        loc: ChromaLocation,
        native: bool,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb16 = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$M422, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(native)
          .with_chroma_location(loc.clone())
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          let f = $F422::new(
            y, u, v, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
          );
          $w422(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb16)
      }

      /// Drive an LE `Yuv422pN` single-kernel filter resample (Triangle).
      fn run_filter(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        loc: ChromaLocation,
        simd: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb16 = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$M422, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_chroma_location(loc.clone())
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          let f = $F422::new(
            y, u, v, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
          );
          $w422(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb16)
      }

      /// Drive a BE `Yuv422pN` area resample (planes re-encoded BE-wire).
      fn run_be(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        loc: ChromaLocation,
        native: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let (yb, ub, vb) = (as_be(y), as_be(u), as_be(v));
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb16 = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$M422be, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(native)
          .with_chroma_location(loc.clone())
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          let f = $F422be::try_new(
            &yb, &ub, &vb, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
          )
          .unwrap();
          $w422be(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb16)
      }

      /// The centered NATIVE code-domain oracle: bin Y co-sited and U / V through
      /// the exact centered chroma oracle to `OUT x OUT`, then convert ONCE at
      /// output width via an identity `Yuv444pN` sink.
      fn native_oracle(y: &[u16], u: &[u16], v: &[u16], simd: bool) -> (Vec<u8>, Vec<u16>) {
        let yb = bin_cosited_u16(y, SRC, SRC, OUT, OUT);
        let ub = bin_chroma_centered_u16(u, CW, SRC, OUT, OUT);
        let vb = bin_chroma_centered_u16(v, CW, SRC, OUT, OUT);
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb16 = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$M444>::new(OUT, OUT)
            .with_simd(simd)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb16)
            .unwrap();
          let f = $F444::new(
            &yb, &ub, &vb, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
          );
          $w444(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb16)
      }

      /// The centered RGB-domain oracle: reconstruct U / V to full width (u16)
      /// with the #302 kernel, then run that `Yuv444pN` frame through the given
      /// resampler — i.e. convert-each-row-then-bin, exactly what the row-stage /
      /// filter arms do. `filter = true` uses the Triangle filter tier.
      fn rgb_domain_oracle(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        simd: bool,
        filter: bool,
      ) -> (Vec<u8>, Vec<u16>) {
        let mut uf = vec![0u16; SRC * SRC];
        let mut vf = vec![0u16; SRC * SRC];
        for r in 0..SRC {
          uf[r * SRC..r * SRC + SRC]
            .copy_from_slice(&recon_full_row_u16(&u[r * CW..r * CW + CW], CW));
          vf[r * SRC..r * SRC + SRC]
            .copy_from_slice(&recon_full_row_u16(&v[r * CW..r * CW + CW], CW));
        }
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb16 = vec![0u16; OUT * OUT * 3];
        let f = $F444::new(
          y, &uf, &vf, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
        );
        if filter {
          let mut sink = MixedSinker::<$M444, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          $w444(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        } else {
          let mut sink = MixedSinker::<$M444, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_simd(simd)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb16)
          .unwrap();
          $w444(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb16)
      }

      // ---- co-sited byte-identity (the regression contract) ------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn cosited_group_is_byte_identical_across_tiers() {
        let (y, u, v) = ramp();
        for native in [true, false] {
          let base = run(&y, &u, &v, ChromaLocation::Unspecified, native, true);
          for loc in [
            ChromaLocation::Left,
            ChromaLocation::TopLeft,
            ChromaLocation::BottomLeft,
            ChromaLocation::other("unassigned-7"),
          ] {
            assert_eq!(
              run(&y, &u, &v, loc.clone(), native, true),
              base,
              "co-sited siting {loc:?} must keep the byte-identical decode (native={native})"
            );
          }
        }
        // The filter tier too.
        let base = run_filter(&y, &u, &v, ChromaLocation::Unspecified, true);
        assert_eq!(run_filter(&y, &u, &v, ChromaLocation::Left, true), base);
      }

      // ---- centered native == the exact code-domain oracle -------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_native_equals_code_domain_oracle() {
        let (y, u, v) = ramp();
        let want = native_oracle(&y, &u, &v, true);
        for loc in [
          ChromaLocation::Center,
          ChromaLocation::Top,
          ChromaLocation::Bottom,
        ] {
          assert_eq!(
            run(&y, &u, &v, loc.clone(), true, true),
            want,
            "centered native {loc:?} must equal the code-domain reconstruct-then-bin oracle"
          );
        }
      }

      // ---- centered row-stage / filter == RGB-domain reconstruct-then-bin ----

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_row_stage_equals_rgb_reconstruct_then_bin() {
        let (y, u, v) = ramp();
        let want = rgb_domain_oracle(&y, &u, &v, true, false);
        for loc in [
          ChromaLocation::Center,
          ChromaLocation::Top,
          ChromaLocation::Bottom,
        ] {
          assert_eq!(
            run(&y, &u, &v, loc.clone(), false, true),
            want,
            "centered row-stage {loc:?} must equal the RGB-domain reconstruct-then-bin oracle"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_filter_equals_rgb_reconstruct_then_bin() {
        let (y, u, v) = ramp();
        let want = rgb_domain_oracle(&y, &u, &v, true, true);
        assert_eq!(
          run_filter(&y, &u, &v, ChromaLocation::Center, true),
          want,
          "centered filter must equal the RGB-domain Triangle reconstruct-then-bin oracle"
        );
      }

      // ---- SIMD == scalar ----------------------------------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_simd_matches_scalar() {
        let (y, u, v) = ramp();
        for native in [true, false] {
          assert_eq!(
            run(&y, &u, &v, ChromaLocation::Center, native, true),
            run(&y, &u, &v, ChromaLocation::Center, native, false),
            "centered SIMD vs scalar must agree (native={native})"
          );
        }
        assert_eq!(
          run_filter(&y, &u, &v, ChromaLocation::Center, true),
          run_filter(&y, &u, &v, ChromaLocation::Center, false),
          "centered filter SIMD vs scalar must agree"
        );
      }

      // ---- centered differs from co-sited on a ramp, equals it on flat -------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_differs_from_cosited_on_a_chroma_ramp() {
        let (y, u, v) = ramp();
        for native in [true, false] {
          let cos = run(&y, &u, &v, ChromaLocation::Left, native, true);
          let cen = run(&y, &u, &v, ChromaLocation::Center, native, true);
          assert_ne!(
            cos, cen,
            "centered must differ from co-sited on a chroma ramp (native={native})"
          );
        }
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn centered_equals_cosited_on_flat_chroma() {
        let (y, u, v) = flat();
        for native in [true, false] {
          assert_eq!(
            run(&y, &u, &v, ChromaLocation::Left, native, true),
            run(&y, &u, &v, ChromaLocation::Center, native, true),
            "centered must equal co-sited on flat chroma (native={native})"
          );
        }
      }

      // ---- wire endianness is siting-independent -----------------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn be_centered_matches_le() {
        let (y, u, v) = ramp();
        for native in [true, false] {
          assert_eq!(
            run_be(&y, &u, &v, ChromaLocation::Center, native),
            run(&y, &u, &v, ChromaLocation::Center, native, true),
            "BE centered decode must equal LE for the same logical planes (native={native})"
          );
        }
      }

      // ---- mid-frame siting change is rejected across tiers ------------------

      /// Accept row 0 at `loc1` (freezes the phase), flip to `loc2`, feed the
      /// IN-SEQUENCE row 1, and return its `process` result.
      fn flip_row1<R>(
        mut sink: MixedSinker<'_, $M422, R>,
        y: &[u16],
        u: &[u16],
        v: &[u16],
        loc1: ChromaLocation,
        loc2: ChromaLocation,
      ) -> Result<(), MixedSinkerError> {
        sink.set_chroma_location(loc1.clone());
        PixelSink::begin_frame(&mut sink, SRC as u32, SRC as u32).unwrap();
        let row0 = $Row::for_tests(&y[0..SRC], &u[0..CW], &v[0..CW], 0, M, FR);
        PixelSink::process(&mut sink, row0).unwrap();
        sink.set_chroma_location(loc2.clone());
        let row1 = $Row::for_tests(&y[SRC..2 * SRC], &u[CW..2 * CW], &v[CW..2 * CW], 1, M, FR);
        PixelSink::process(&mut sink, row1)
      }

      #[test]
      #[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
      fn mid_frame_siting_change_rejected() {
        let (y, u, v) = ramp();
        for (loc1, loc2) in [
          (ChromaLocation::Center, ChromaLocation::Left),
          (ChromaLocation::Left, ChromaLocation::Center),
        ] {
          // Native fast tier.
          let mut rgb = vec![0u8; OUT * OUT * 3];
          let sink = MixedSinker::<$M422, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
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
          let mut rgb = vec![0u8; OUT * OUT * 3];
          let sink = MixedSinker::<$M422, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
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
          let mut rgb = vec![0u8; OUT * OUT * 3];
          let sink = MixedSinker::<$M422, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
          let err = flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "filter {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
        }
      }
    }
  };
}

hibit_422_resample_siting!(
  p10,
  10,
  Yuv422p10,
  Yuv422p10Frame,
  yuv422p10_to,
  Yuv422p10<true>,
  Yuv422p10BeFrame,
  yuv422p10_to_endian,
  Yuv444p10,
  Yuv444p10Frame,
  yuv444p10_to,
  Yuv422p10Row
);
hibit_422_resample_siting!(
  p16,
  16,
  Yuv422p16,
  Yuv422p16Frame,
  yuv422p16_to,
  Yuv422p16<true>,
  Yuv422p16BeFrame,
  yuv422p16_to_endian,
  Yuv444p16,
  Yuv444p16Frame,
  yuv444p16_to,
  Yuv422p16Row
);
