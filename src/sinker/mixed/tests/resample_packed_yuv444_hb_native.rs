//! Fused-downscale coverage for the high-bit **packed 4:4:4** YUV NATIVE fast
//! tier (issue #123) — [`V410`](crate::source::V410) (10-bit, one bit-field-packed
//! u32 word per pixel) / [`Xv36`](crate::source::Xv36) (12-bit, four MSB-aligned
//! u16 slots `U Y V A` per pixel, `A` padding), LE + BE wire. The 4:4:4 twin of
//! the high-bit packed 4:2:2 [`y2xx_process_native`](crate::sinker::mixed):
//! both reuse the high-bit non-4:2:0 PLANAR join
//! ([`yuv_planar16_process_native`](crate::sinker::mixed::planar_high_bit_native))
//! after de-PACKING each format's OWN wire layout into wrapper-owned host-native
//! LOGICAL u16 scratch — but at 4:4:4 (full-width chroma, `chroma_vsub = 1`,
//! `chroma_w = w`) rather than 4:2:2.
//!
//! The native tier bins those planes straight to the output grid and converts
//! ONCE per output row at output width (4:4:4 kernels) — vs the row-stage tier
//! ([`packed_yuv444_triple_resample`](crate::sinker::mixed::packed_yuv444_triple_resample)),
//! which converts each source row at source width then bins. The tiers differ in
//! colour SEMANTICS (native averages in YUV then converts; row-stage converts
//! then averages in RGB), so native is NOT byte-identical to row-stage — only
//! within a small tolerance in-gamut. Luma is bit-identical (both bin the same
//! de-packed native Y then narrow `>> (BITS - 8)`).
//!
//! Per format (LE + BE):
//! - `native_equals_bin_then_convert_oracle`: native EXACTLY equals an
//!   independent bin-then-convert oracle that de-packs the wire, area-bins each
//!   plane to OUTPUT resolution (all 4:4:4 — full width), then converts ONCE
//!   through an identity-resolution `Yuv444pN` sink with the SAME native-depth
//!   kernels + `(1 << BITS) - 1` clamp the native tier finalizes with. The luma
//!   oracle clamps INDEPENDENTLY.
//! - `native_equals_planar_twin`: native == native `Yuv444pN` on the de-packed
//!   planes (the strong cross-check that the packed wrapper is a faithful de-pack
//!   in front of the planar join).
//! - `native_within_tolerance_of_rowstage` + the `BE = HOST_NATIVE_BE` handoff
//!   proof.
//! - `native_luma_clamps_full_scale_y` (both tiers): the native-depth luma clamp
//!   at the achievable full-scale boundary.
//! - `native_luma_u16_equals_clamped_binned_y`: the native-depth `luma_u16`
//!   emit + the route-stays-native-with-luma_u16 contract, LE + BE.
//! - the atomicity regressions on [`arm_packed_444_alloc_failure`] + the route
//!   freeze guard, and the luma-only lazy-chroma carry-through.

use crate::{
  KernelMatrix, PixelSink,
  frame::{Yuv444p10LeFrame, Yuv444p12LeFrame},
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    V410, V410Row, Xv36, Xv36Row, Yuv444p10, Yuv444p12, v410_to, v410_to_endian, xv36_to,
    xv36_to_endian, yuv444p10_to, yuv444p12_to,
  },
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// In-gamut per-channel u8 tolerance between the native and row-stage tiers — the
/// two average in different domains (YUV vs RGB) and round independently per
/// output pixel; native correctness itself is pinned EXACTLY by
/// `native_equals_bin_then_convert_oracle`. Matches the y2xx suite's bound.
const TOL_U8: u8 = 5;

/// Exact integer-ratio area mean (round-half-up) of an `SRC x SRC` u16 plane down
/// to `OUT x OUT` (square ratio).
fn bin_to_out(plane: &[u16]) -> Vec<u16> {
  let r = SRC / OUT;
  let denom = (r * r) as u32;
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..r {
        for dx in 0..r {
          s += plane[(oy * r + dy) * SRC + ox * r + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + denom / 2) / denom) as u16;
    }
  }
  out
}

fn max_delta_u8(a: &[u8], b: &[u8]) -> u8 {
  a.iter()
    .zip(b)
    .map(|(&x, &y)| x.abs_diff(y))
    .max()
    .unwrap_or(0)
}
fn max_delta_u16(a: &[u16], b: &[u16]) -> u16 {
  a.iter()
    .zip(b)
    .map(|(&x, &y)| x.abs_diff(y))
    .max()
    .unwrap_or(0)
}

/// Generates the per-format high-bit packed 4:4:4 native suite. The format
/// supplies, as free helpers defined before the invocation:
/// - `$pack(&[u16] u, &[u16] y, &[u16] v) -> Vec<ELEM>` — logical planes → wire
///   (host-native ELEM).
/// - `$as_le(&[ELEM]) -> Vec<ELEM>` / `$as_be(&[ELEM]) -> Vec<ELEM>` — re-encode
///   the host-native wire as host-independent LE / BE byte storage.
/// - `$logical_y(&[ELEM]) -> Vec<u16>` / `$depack_uv(&[ELEM]) -> (Vec<u16>,
///   Vec<u16>)` — de-pack the wire to logical planes (the oracle's de-pack).
/// - `$frame(&[ELEM]) -> Frame<false>` / `$frame_be(&[ELEM]) -> Frame<true>`.
macro_rules! packed_444_hb_suite {
  (
    $mod:ident, $marker:ident, $row:ident, $walker:ident, $walker_be:ident,
    $planar_marker:ident, $planar_frame:ident, $planar_walker:ident, $bits:literal, $elem:ty,
    $pack:ident, $as_le:ident, $as_be:ident, $logical_y:ident, $depack_uv:ident,
    $frame:ident, $frame_be:ident, $row_slice:ident,
  ) => {
    mod $mod {
      use super::*;

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MID: u16 = 1u16 << ($bits - 1);

      /// Per-pixel logical Y / U / V ramp kept near the legal-range middle so the
      /// converted RGB stays in gamut (4:4:4 — every pixel its own chroma).
      fn ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; SRC * SRC];
        let mut v = vec![0u16; SRC * SRC];
        for i in 0..SRC * SRC {
          y[i] = (MID as u32 + ((i as u32 * 37) % (MASK as u32 / 4))) as u16 & MASK;
          u[i] = (MID as u32 + ((i as u32 * 53) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16
            & MASK;
          v[i] = (MID as u32 + ((i as u32 * 41) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16
            & MASK;
        }
        (u, y, v)
      }

      /// Uniform-gray: constant logical Y, neutral chroma.
      fn uniform_gray(y: u16) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        (
          vec![MID & MASK; SRC * SRC],
          vec![y & MASK; SRC * SRC],
          vec![MID & MASK; SRC * SRC],
        )
      }

      /// Crafted VARYING illegal-chroma fixture: extreme alternating chroma over a
      /// super-black -> super-white Y ramp — many 2x2 blocks straddle the RGB
      /// clamp where native and row-stage genuinely diverge.
      fn out_of_gamut() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; SRC * SRC];
        let mut v = vec![0u16; SRC * SRC];
        for i in 0..SRC * SRC {
          y[i] = ((i as u32 * MASK as u32) / (SRC * SRC) as u32) as u16 & MASK;
          let hi = i % 2 == 0;
          u[i] = if hi { MASK } else { 0 };
          v[i] = if hi { 0 } else { MASK };
        }
        (u, y, v)
      }

      /// Drive the LE source through a tier for the native output set (u8 RGB +
      /// u16 RGB + u8 luma). `native` toggles the bin-then-convert native fast
      /// tier vs the convert-then-bin row-stage tier.
      fn run(packed: &[$elem], native: bool) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let p = $as_le(packed);
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(native)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&$frame(&p), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the BE source through the native tier (the host-native-endian guard
      /// reference).
      fn native_be_run(packed: &[$elem]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let p = $as_be(packed);
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker_be::<_, true>(&$frame_be(&p), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// Drive the row-stage tier (BE) — the correct host-independent reference.
      fn rowstage_be_run(packed: &[$elem]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let p = $as_be(packed);
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(false)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker_be::<_, true>(&$frame_be(&p), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      /// The bin-then-convert oracle: de-pack the wire, area-bin every plane to
      /// OUTPUT resolution (all 4:4:4 — full width), then convert the full-output
      /// host-native planes ONCE through an identity-resolution `Yuv444pN` sink.
      /// The luma oracle clamps INDEPENDENTLY of the sink.
      fn oracle(packed: &[$elem]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let yl = $logical_y(packed);
        let (u, v) = $depack_uv(packed);
        let yb = bin_to_out(&yl);
        let ub = bin_to_out(&u);
        let vb = bin_to_out(&v);
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$planar_marker>::new(OUT, OUT)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb_u16)
            .unwrap();
          let f = $planar_frame::try_new(
            &yb, &ub, &vb, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
          )
          .unwrap();
          $planar_walker(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let luma: Vec<u8> = yb
          .iter()
          .map(|&by| (by.min(MASK) >> ($bits - 8)) as u8)
          .collect();
        (rgb, rgb_u16, luma)
      }

      /// Native `Yuv444pN` reference on the de-packed planes — the twin-parity
      /// cross-check (same join, fed planar instead of through the packed de-pack).
      fn planar_twin_native(packed: &[$elem]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
        let yl = $logical_y(packed);
        let (u, v) = $depack_uv(packed);
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$planar_marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          let f = $planar_frame::try_new(
            &yl, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
          )
          .unwrap();
          $planar_walker(&f, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (rgb, rgb_u16, luma)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_equals_bin_then_convert_oracle() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
        let (o_rgb, o_rgb16, o_luma) = oracle(&packed);
        assert_eq!(n_rgb, o_rgb, "u8 rgb must equal the bin-then-convert oracle");
        assert_eq!(
          n_rgb16, o_rgb16,
          "u16 rgb must equal the bin-then-convert oracle (clamp-for-clamp)"
        );
        assert_eq!(n_luma, o_luma, "luma must equal the binned-then-narrowed Y");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_equals_planar_twin() {
        // The packed wrapper IS a de-pack in front of the planar join, so its
        // output must be bit-identical to feeding the de-packed planes straight to
        // the native Yuv444pN.
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
        let (t_rgb, t_rgb16, t_luma) = planar_twin_native(&packed);
        assert_eq!(n_rgb, t_rgb, "u8 rgb must match the planar twin");
        assert_eq!(n_rgb16, t_rgb16, "u16 rgb must match the planar twin");
        assert_eq!(n_luma, t_luma, "luma must match the planar twin");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_within_tolerance_of_rowstage() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
        let (r_rgb, r_rgb16, r_luma) = run(&packed, false);
        assert_eq!(n_luma, r_luma, "luma must be bit-identical across tiers");
        let d_u8 = max_delta_u8(&n_rgb, &r_rgb);
        assert!(
          d_u8 <= TOL_U8,
          "u8 native-vs-rowstage max delta {d_u8} exceeds tolerance {TOL_U8}"
        );
        let tol_u16: u16 = (TOL_U8 as u16) << ($bits - 8);
        let d_u16 = max_delta_u16(&n_rgb16, &r_rgb16);
        assert!(
          d_u16 <= tol_u16,
          "u16 native-vs-rowstage max delta {d_u16} exceeds tolerance {tol_u16}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_be_matches_native_le() {
        // The native tier de-packs the wire to host-native LOGICAL u16 BEFORE
        // binning, so BE and LE sources produce identical output.
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let le = run(&packed, true);
        let be = native_be_run(&packed);
        assert_eq!(be.0, le.0, "BE u8 colour must match LE");
        assert_eq!(be.1, le.1, "BE u16 colour must match LE");
        assert_eq!(be.2, le.2, "BE luma must match LE");
      }

      /// The host-native-endian regression: BE native vs the correct BE row-stage
      /// reference. Proves the `BE = HOST_NATIVE_BE` handoff — a wrapper forwarding
      /// the source `BE` to the delegate would byte-swap the already-native scratch
      /// on a big-endian host.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_be_within_tolerance_of_rowstage_be() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let (n_rgb, n_rgb16, n_luma) = native_be_run(&packed);
        let (r_rgb, r_rgb16, r_luma) = rowstage_be_run(&packed);
        assert_eq!(n_luma, r_luma, "BE luma must be bit-identical across tiers");
        let d_u8 = max_delta_u8(&n_rgb, &r_rgb);
        assert!(
          d_u8 <= TOL_U8,
          "BE u8 native-vs-rowstage max delta {d_u8} exceeds tolerance {TOL_U8}"
        );
        let tol_u16: u16 = (TOL_U8 as u16) << ($bits - 8);
        let d_u16 = max_delta_u16(&n_rgb16, &r_rgb16);
        assert!(
          d_u16 <= tol_u16,
          "BE u16 native-vs-rowstage max delta {d_u16} exceeds tolerance {tol_u16}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_matches_inter_area_oracle() {
        // cv2 INTER_AREA parity for luma: the area-bin of the DE-PACKED logical Y,
        // narrowed. Guards the Y de-pack.
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let (_, _, n_luma) = run(&packed, true);
        let y_ref = bin_to_out(&$logical_y(&packed));
        let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_eq!(
          n_luma, luma_ref,
          "native luma must equal the INTER_AREA bin of the de-packed Y"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_clamps_full_scale_y() {
        // A full-scale binned Y must SATURATE through the native-depth clamp +
        // `>> (BITS - 8)` narrowing, never wrap. (A bit-field / MSB extraction
        // CANNOT exceed MASK, and an area mean of `<= MASK` stays `<= MASK`, so
        // the achievable boundary is the legal max.) The oracle clamps
        // independently.
        let (u, _, v) = ramp();
        let y = vec![MASK; SRC * SRC];
        let packed = $pack(&u, &y, &v);
        let (_, _, n_luma) = run(&packed, true);
        let yb = bin_to_out(&$logical_y(&packed));
        let expect: Vec<u8> = yb
          .iter()
          .map(|&by| (by.min(MASK) >> ($bits - 8)) as u8)
          .collect();
        assert_eq!(
          n_luma, expect,
          "full-scale binned Y must clamp to native-max before narrowing, not wrap"
        );
        let sat = (MASK >> ($bits - 8)) as u8;
        assert!(
          n_luma.iter().all(|&l| l == sat),
          "all full-scale luma must saturate to {sat}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rowstage_luma_clamps_full_scale_y() {
        let (u, _, v) = ramp();
        let y = vec![MASK; SRC * SRC];
        let packed = $pack(&u, &y, &v);
        let (_, _, r_luma) = run(&packed, false);
        let yb = bin_to_out(&$logical_y(&packed));
        let expect: Vec<u8> = yb
          .iter()
          .map(|&by| (by.min(MASK) >> ($bits - 8)) as u8)
          .collect();
        assert_eq!(
          r_luma, expect,
          "row-stage full-scale luma must clamp to native-max before narrowing, not wrap"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_gamut_native_vs_rowstage_pinned() {
        let (u, y, v) = out_of_gamut();
        let packed = $pack(&u, &y, &v);
        let (n_rgb, _, n_luma) = run(&packed, true);
        let (r_rgb, _, r_luma) = run(&packed, false);
        assert_eq!(n_luma, r_luma, "luma stays bit-identical out of gamut");
        let d = max_delta_u8(&n_rgb, &r_rgb);
        assert!(
          d > TOL_U8,
          "crafted out-of-gamut case must diverge beyond the in-gamut tolerance {TOL_U8}, got {d}"
        );
        assert!(d < u8::MAX, "out-of-gamut delta stays bounded, got {d}");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn uniform_gray_leaves_color_unchanged() {
        // Independent-kernel guard (#37): a uniform-gray downscale must leave every
        // colour output equal to the direct conversion of a single pixel.
        let (u, y, v) = uniform_gray((MID as u32 + (MASK as u32 / 8)) as u16 & MASK);
        let packed = $pack(&u, &y, &v);
        let (n_rgb, n_rgb16, _) = run(&packed, true);
        let p = $as_le(&packed);
        let mut ref_rgb = vec![0u8; SRC * SRC * 3];
        let mut ref_rgb16 = vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut ref_rgb)
            .unwrap()
            .with_rgb_u16(&mut ref_rgb16)
            .unwrap();
          $walker(&$frame(&p), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        for px in n_rgb.chunks_exact(3) {
          assert_eq!(px, &ref_rgb[..3], "uniform-gray u8 colour drifted");
        }
        for px in n_rgb16.chunks_exact(3) {
          assert_eq!(px, &ref_rgb16[..3], "uniform-gray u16 colour drifted");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn u8_and_u16_color_are_independent_bins() {
        // Independent-kernel guard (#37): narrowing the binned u16 colour to u8
        // diverges from the genuine u8 bin over a varying ramp.
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let (n_rgb, n_rgb16, _) = run(&packed, true);
        let narrowed: Vec<u8> = n_rgb16.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_ne!(
          n_rgb, narrowed,
          "u8 colour must be an independent bin, not a narrowed u16 bin"
        );
      }

      /// The native fast tier emits the native-depth `luma_u16` directly (the
      /// clamped binned Y, host-native u16 — NOT narrowed), so it equals an
      /// INDEPENDENT clamped-binned-Y oracle bit-for-bit, LE + BE. AND a `rgb` +
      /// `luma_u16` sink takes the NATIVE route for BOTH: the rgb matches the
      /// native bin-then-convert oracle (not row-stage), proving that attaching
      /// `luma_u16` no longer changes the rgb colour semantics.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_luma_u16_equals_clamped_binned_y() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let luma_u16_oracle: Vec<u16> = bin_to_out(&$logical_y(&packed))
          .iter()
          .map(|&by| by.min(MASK))
          .collect();

        // LE: a `luma_u16`-only native sink.
        {
          let p = $as_le(&packed);
          let mut luma_u16 = vec![0u16; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(true)
            .with_luma_u16(&mut luma_u16)
            .unwrap();
            $walker(&$frame(&p), FR, sink.set_kernel_matrix(M)).unwrap();
          }
          assert_eq!(
            luma_u16, luma_u16_oracle,
            "native luma_u16 (LE) must equal the clamped-binned-Y oracle"
          );
        }

        // BE: same, through the BE wire.
        {
          let p = $as_be(&packed);
          let mut luma_u16 = vec![0u16; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(true)
            .with_luma_u16(&mut luma_u16)
            .unwrap();
            $walker_be::<_, true>(&$frame_be(&p), FR, sink.set_kernel_matrix(M)).unwrap();
          }
          assert_eq!(
            luma_u16, luma_u16_oracle,
            "native luma_u16 (BE) must equal the clamped-binned-Y oracle"
          );
        }

        // A `rgb` + `luma_u16` sink uses the NATIVE route for BOTH.
        {
          let p = $as_le(&packed);
          let mut rgb = vec![0u8; OUT * OUT * 3];
          let mut luma_u16 = vec![0u16; OUT * OUT];
          {
            let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(OUT, OUT),
            )
            .unwrap()
            .with_native(true)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_luma_u16(&mut luma_u16)
            .unwrap();
            $walker(&$frame(&p), FR, sink.set_kernel_matrix(M)).unwrap();
          }
          let (o_rgb, _, _) = oracle(&packed);
          assert_eq!(
            rgb, o_rgb,
            "rgb + luma_u16 must take the NATIVE route — rgb equals the native \
             oracle, not row-stage"
          );
          assert_eq!(
            luma_u16, luma_u16_oracle,
            "rgb + luma_u16: luma_u16 must equal the clamped-binned-Y oracle"
          );
        }
      }

      #[test]
      fn no_outputs_is_a_no_op() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let p = $as_le(&packed);
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true);
        $walker(&$frame(&p), FR, sink.set_kernel_matrix(M)).unwrap();
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resets_join_across_frames() {
        let (u, y1, v) = ramp();
        let mut y2 = y1.clone();
        for p in y2.iter_mut() {
          *p = MASK - *p;
        }
        let packed1 = $pack(&u, &y1, &v);
        let packed2 = $pack(&u, &y2, &v);
        let p1 = $as_le(&packed1);
        let p2 = $as_le(&packed2);
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_native(true)
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&$frame(&p1), FR, sink.set_kernel_matrix(M)).unwrap();
          $walker(&$frame(&p2), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let y_ref = bin_to_out(&$logical_y(&packed2));
        let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "join did not reset between frames");
      }

      // ---- atomicity ----------------------------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn frozen_mid_frame_change_rejected_before_scratch_alloc() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let p = $as_le(&packed);
        let mut luma = vec![0u8; OUT * OUT];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Luma-only rows 0 and 1 freeze a luma-only output set.
        for r in 0..2 {
          sink
            .process($row::for_tests($row_slice(&p, r), r, M, FR))
            .expect("luma-only rows freeze a luma-only output set");
        }
        // Attach u16 colour mid-frame, changing the output set, and arm the
        // wrapper scratch failpoint.
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        crate::sinker::mixed::arm_packed_444_alloc_failure();
        let err = sink.process($row::for_tests($row_slice(&p, 2), 2, M, FR)).unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "mid-frame output change must reject as ResampleOutputsChanged before \
           the scratch alloc, got {err:?}"
        );
        assert!(
          rgb_u16.iter().all(|&b| b == 0),
          "rejected mid-frame-change row touched the new colour output"
        );
        // The failpoint is single-shot; prove it was NOT consumed via a fresh
        // in-sequence colour row that DOES fire it.
        let mut rgb_u16b = vec![0u16; OUT * OUT * 3];
        let mut sink2 = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_rgb_u16(&mut rgb_u16b)
        .unwrap();
        let err2 = sink2.process($row::for_tests($row_slice(&p, 0), 0, M, FR)).unwrap_err();
        assert!(
          matches!(
            err2,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "armed failpoint must still be live and fire on the first in-sequence \
           colour reserve, got {err2:?}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn oos_after_recoverable_alloc_failure_rejected_before_scratch_alloc() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let p = $as_le(&packed);
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
        crate::sinker::mixed::arm_packed_444_alloc_failure();
        let err0 = sink.process($row::for_tests($row_slice(&p, 0), 0, M, FR)).unwrap_err();
        assert!(
          matches!(
            err0,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the recoverable scratch failure on row 0 must surface AllocationFailed, got {err0:?}"
        );
        crate::sinker::mixed::arm_packed_444_alloc_failure();
        let err2 = sink.process($row::for_tests($row_slice(&p, 2), 2, M, FR)).unwrap_err();
        assert!(
          matches!(
            err2,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "an out-of-sequence row after a recoverable scratch failure must reject \
           as OutOfSequenceRow, never AllocationFailed, got {err2:?}"
        );
        assert!(
          rgb_u16.iter().all(|&b| b == 0),
          "neither the recoverable-failure nor the out-of-sequence row touched the colour output"
        );
        let mut rgb_u16b = vec![0u16; OUT * OUT * 3];
        let mut sink2 = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_rgb_u16(&mut rgb_u16b)
        .unwrap();
        let err3 = sink2.process($row::for_tests($row_slice(&p, 0), 0, M, FR)).unwrap_err();
        assert!(
          matches!(
            err3,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "the failpoint re-armed in step 2 must still be live and fire on the \
           first in-sequence colour reserve, got {err3:?}"
        );
      }

      // ---- frozen native-vs-row-stage route -----------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_to_rowstage_route_flip_mid_frame_rejected() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let p = $as_le(&packed);
        let mut luma = vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::for_tests($row_slice(&p, 0), 0, M, FR))
          .expect("native row 0 freezes the route and succeeds");
        sink.set_native(false);
        let err = sink.process($row::for_tests($row_slice(&p, 1), 1, M, FR)).unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::NativeRouteChanged(_)),
          "a native -> row-stage mid-frame route flip must reject as NativeRouteChanged, got {err:?}"
        );
      }

      /// Attaching `luma_u16` MID-FRAME (after a native u8-luma row froze the
      /// output set) must be classified by the FROZEN-OUTPUT check as
      /// `ResampleOutputsChanged`, NOT by the route guard as `NativeRouteChanged`
      /// (`take_native = native` is invariant to `luma_u16`).
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_u16_attach_mid_frame_rejected_as_outputs_changed() {
        let (u, y, v) = ramp();
        let packed = $pack(&u, &y, &v);
        let p = $as_le(&packed);
        let mut luma = vec![0u8; OUT * OUT];
        let mut luma_u16 = vec![0u16; OUT * OUT];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::for_tests($row_slice(&p, 0), 0, M, FR))
          .expect("native luma row 0 freezes the output set and the route");
        sink.set_luma_u16(&mut luma_u16).unwrap();
        let err = sink.process($row::for_tests($row_slice(&p, 1), 1, M, FR)).unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "a mid-frame luma_u16 attach must reject as ResampleOutputsChanged (the \
           frozen-output check), not NativeRouteChanged, got {err:?}"
        );
        assert!(
          luma_u16.iter().all(|&b| b == 0),
          "the rejected mid-frame-change row must not touch the new luma_u16 output"
        );
      }
    }
  };
}

// ---- V410: 10-bit, one u32 word per pixel (`(V << 20) | (Y << 10) | U`) ----

fn v410_pack(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u32> {
  (0..u.len())
    .map(|i| {
      let u = (u[i] & 0x3FF) as u32;
      let y = (y[i] & 0x3FF) as u32;
      let v = (v[i] & 0x3FF) as u32;
      (v << 20) | (y << 10) | u
    })
    .collect()
}
fn v410_as_le(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}
fn v410_as_be(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}
fn v410_logical_y(packed: &[u32]) -> Vec<u16> {
  packed.iter().map(|&w| ((w >> 10) & 0x3FF) as u16).collect()
}
fn v410_depack_uv(packed: &[u32]) -> (Vec<u16>, Vec<u16>) {
  let u = packed.iter().map(|&w| (w & 0x3FF) as u16).collect();
  let v = packed.iter().map(|&w| ((w >> 20) & 0x3FF) as u16).collect();
  (u, v)
}
fn v410_frame(buf: &[u32]) -> crate::frame::V410Frame<'_> {
  crate::frame::V410Frame::new(buf, SRC as u32, SRC as u32, SRC as u32)
}
fn v410_frame_be(buf: &[u32]) -> crate::frame::V410BeFrame<'_> {
  crate::frame::V410BeFrame::try_new(buf, SRC as u32, SRC as u32, SRC as u32).unwrap()
}
/// Row `r` of a V410 wire buffer — one u32 per pixel, so `SRC` elements per row.
fn v410_row_slice(p: &[u32], r: usize) -> &[u32] {
  &p[r * SRC..(r + 1) * SRC]
}

// ---- Xv36: 12-bit, four MSB-aligned u16 slots `U Y V A` per pixel ----

fn xv36_pack(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u16> {
  let mut buf = vec![0u16; SRC * SRC * 4];
  for i in 0..u.len() {
    let base = i * 4;
    buf[base] = (u[i] & 0xFFF) << 4; // U, MSB-aligned
    buf[base + 1] = (y[i] & 0xFFF) << 4; // Y
    buf[base + 2] = (v[i] & 0xFFF) << 4; // V
    buf[base + 3] = 0; // A padding
  }
  buf
}
fn xv36_as_le(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}
fn xv36_as_be(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}
fn xv36_logical_y(packed: &[u16]) -> Vec<u16> {
  (0..packed.len() / 4)
    .map(|i| packed[i * 4 + 1] >> 4)
    .collect()
}
fn xv36_depack_uv(packed: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let u = (0..packed.len() / 4).map(|i| packed[i * 4] >> 4).collect();
  let v = (0..packed.len() / 4)
    .map(|i| packed[i * 4 + 2] >> 4)
    .collect();
  (u, v)
}
fn xv36_frame(buf: &[u16]) -> crate::frame::Xv36Frame<'_> {
  crate::frame::Xv36Frame::new(buf, SRC as u32, SRC as u32, (SRC * 4) as u32)
}
fn xv36_frame_be(buf: &[u16]) -> crate::frame::Xv36BeFrame<'_> {
  crate::frame::Xv36BeFrame::try_new(buf, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap()
}
/// Row `r` of an Xv36 wire buffer — four u16 per pixel, so `SRC * 4` per row.
fn xv36_row_slice(p: &[u16], r: usize) -> &[u16] {
  &p[r * SRC * 4..(r + 1) * SRC * 4]
}

packed_444_hb_suite!(
  v410,
  V410,
  V410Row,
  v410_to,
  v410_to_endian,
  Yuv444p10,
  Yuv444p10LeFrame,
  yuv444p10_to,
  10,
  u32,
  v410_pack,
  v410_as_le,
  v410_as_be,
  v410_logical_y,
  v410_depack_uv,
  v410_frame,
  v410_frame_be,
  v410_row_slice,
);

packed_444_hb_suite!(
  xv36,
  Xv36,
  Xv36Row,
  xv36_to,
  xv36_to_endian,
  Yuv444p12,
  Yuv444p12LeFrame,
  yuv444p12_to,
  12,
  u16,
  xv36_pack,
  xv36_as_le,
  xv36_as_be,
  xv36_logical_y,
  xv36_depack_uv,
  xv36_frame,
  xv36_frame_be,
  xv36_row_slice,
);

/// A luma-only high-bit packed 4:4:4 native sink must NOT plan or allocate any
/// chroma state. Armed with the planar join's chroma-planning failpoint: a
/// luma-only row leaves it unconsumed (the run succeeds), while a colour row
/// reaches chroma planning and fires. (Tested on V410 — the lazy-chroma contract
/// is shared across both formats via the same join.)
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_native_skips_chroma_planning() {
  let mut u = vec![0u16; SRC * SRC];
  let mut y = vec![0u16; SRC * SRC];
  let mut v = vec![0u16; SRC * SRC];
  for i in 0..SRC * SRC {
    u[i] = 512;
    y[i] = 512;
    v[i] = 512;
  }
  let packed = v410_pack(&u, &y, &v);
  let p = v410_as_le(&packed);
  let frame = v410_frame(&p);

  crate::sinker::mixed::arm_planar_hb_native_chroma_failure();

  // Luma-only: the chroma failpoint is armed but never reached -> Ok.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    v410_to(&frame, FR, sink.set_kernel_matrix(M)).expect("luma-only native must not plan chroma");
  }

  // Colour: the still-armed failpoint fires at chroma planning -> Err.
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
  assert!(
    v410_to(&frame, FR, sink.set_kernel_matrix(M)).is_err(),
    "colour native must reach chroma planning (the armed failpoint fires)"
  );
}
