//! Chroma-siting-aware **high-bit** 4:2:2 upsampling for `Yuv422p9` …
//! `Yuv422p16` (#302).
//!
//! 4:2:2 subsamples chroma 2:1 horizontally ONLY (one chroma row per luma row,
//! no vertical subsampling), reusing the SAME wire-format `u16` phase-0.5
//! upsample kernel 4:2:0 uses
//! ([`chroma_upsample_2to1_center_h_u16`](crate::row::scalar::chroma_upsample_2to1_center_h_u16)) —
//! whose bit-exact / endianness / dirty-upper-bit masking behaviour is oracle-
//! tested in `chroma_siting_hibit_420`. Covers here, per bit depth (9 / 10 / 12 /
//! 14 / 16, via the macro): the default / co-sited path staying byte-identical to
//! the pre-#302 nearest-neighbor decode (+ its negative control); the centered
//! RGB / RGBA / HSV decodes — and their `u16` twins — matching an independent
//! "upsample-then-4:4:4" reference; SIMD-vs-scalar parity; that every centered
//! siting (Center / Top / Bottom) agrees horizontally (4:2:2 has no vertical
//! axis); the dirty-upper-bit sanitization end-to-end on BOTH wire endians (the
//! `BE` flag threads through the 4:2:2 wiring); the preflight-ordering atomicity
//! (a centered chroma-scratch alloc failure leaves luma AND colour untouched);
//! and the `ChromaDerivedNcl` consistency invariant (NOT primaries-wired — both
//! paths use the BT.709 matrix-tag fallback).
//!
//! The macro instantiates each bit depth with its **little-endian** marker, so a
//! sample's wire `u16` equals its logical value on the (little-endian) test host;
//! the references compute in that logical domain. The BE re-encode is exercised
//! by the `*_be` dirty-bit tests.

use super::*;
use crate::{ChromaLocation, ColorMatrix};

const W: u32 = 16;
const H: u32 = 8;

/// Builds a high-bit 4:2:2 frame's logical planes: flat mid-gray luma plus a
/// per-column chroma ramp on a half-width, **full-height** chroma plane (one
/// chroma row per luma row). Distinct adjacent columns make the horizontal phase
/// observable; the `+ r` term keeps chroma rows distinct. Clamped to `maxv =
/// (1 << BITS) - 1`.
fn ramp_planes_n(maxv: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let step = (maxv / 16).max(1);
  let y = std::vec![(maxv / 2) as u16; w * h];
  let mut u = std::vec![0u16; cw * h];
  let mut v = std::vec![0u16; cw * h];
  for r in 0..h {
    for c in 0..cw {
      u[r * cw + c] = (step * c as u32 + step + r as u32 * 5).min(maxv) as u16;
      v[r * cw + c] = maxv.saturating_sub(step * c as u32).max(step) as u16;
    }
  }
  (y, u, v)
}

/// Independent reference for the centered horizontal upsample — phase-0.5
/// `1/4`–`3/4` with edge clamp, on logical `u16`. Written separately from the
/// production kernel so it is a real oracle.
fn ref_upsample_center_h_u16(c_half: &[u16], width: usize) -> Vec<u16> {
  let half = width / 2;
  let mut out = std::vec![0u16; width];
  for j in 0..half {
    let l = c_half[j.saturating_sub(1)] as u32;
    let m = c_half[j] as u32;
    let r = c_half[if j + 1 < half { j + 1 } else { j }] as u32;
    out[2 * j] = ((l + 3 * m + 2) >> 2) as u16;
    out[2 * j + 1] = ((3 * m + r + 2) >> 2) as u16;
  }
  out
}

/// Full-resolution U / V a centered high-bit 4:2:2 decode reconstructs: each luma
/// row `r` takes chroma row `r` (1:1 — NO vertical subsampling, unlike 4:2:0's
/// `r / 2`) horizontally upsampled with the centered weights. Feeding these to
/// the matching `Yuv444pN` conversion is the end-to-end oracle.
fn ref_full_chroma_u16(u422: &[u16], v422: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u16; w * h];
  let mut v444 = std::vec![0u16; w * h];
  for r in 0..h {
    let urow = ref_upsample_center_h_u16(&u422[r * cw..r * cw + cw], w);
    let vrow = ref_upsample_center_h_u16(&v422[r * cw..r * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

// ---- per-bit-depth suite ---------------------------------------------------

// Identical bar the bit depth, format marker, frame type, and walker, so
// generate it once per depth. Each lands in its own `mod` so the names don't
// collide.
macro_rules! hibit_422_chroma_tests {
  ($mod:ident, $bits:expr, $Marker:ident, $Frame:ident, $walker:ident, $Ref:ident, $RefFrame:ident, $ref_walker:ident, $MarkerBe:ty, $FrameBe:ident, $walker_be:ident, $Row:ident) => {
    mod $mod {
      use super::*;

      const MAXV: u32 = (1u32 << $bits) - 1;

      /// Centered/default identity-decode RGB for a siting + SIMD toggle.
      fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(loc.clone())
          .with_simd(simd);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        rgb
      }

      // ---- default / co-sited path is byte-identical (regression guard) ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn default_and_cosited_sitings_are_byte_identical() {
        let baseline = convert_rgb(ChromaLocation::Unspecified, true);
        for loc in [
          ChromaLocation::Unspecified,
          ChromaLocation::other("unassigned-99"),
          ChromaLocation::Left,
          ChromaLocation::TopLeft,
          ChromaLocation::BottomLeft,
        ] {
          assert_eq!(
            convert_rgb(loc.clone(), true),
            baseline,
            "siting {loc:?} must keep the byte-identical default decode"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn default_path_does_not_allocate_chroma_scratch() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Left);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(chroma_len, 0, "co-sited path must not grow the u16 chroma scratch");
      }

      // ---- centered path correctness ---------------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_grows_chroma_scratch_to_full_width() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(
          chroma_len,
          2 * W as usize,
          "centered path stages U+V at full width"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_matches_upsample_then_444_reference() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          rgb_ref,
          "centered high-bit 4:2:2 RGB must equal upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_u16_matches_upsample_then_444_reference() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);

        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();

        assert_eq!(
          rgb16, rgb16_ref,
          "centered high-bit 4:2:2 RGB(u16) must equal upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgba_rgba_u16_and_hsv_match_444_reference() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);

        // RGBA (u8).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgba = std::vec![0u8; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba(&mut rgba_ref)
            .unwrap();
          $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(rgba, rgba_ref, "centered RGBA must equal upsample-then-4:4:4");
        }

        // RGBA (u16).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgba16 = std::vec![0u16; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let mut rgba16_ref = std::vec![0u16; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16_ref)
            .unwrap();
          $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(
            rgba16, rgba16_ref,
            "centered RGBA(u16) must equal upsample-then-4:4:4"
          );
        }

        // HSV-direct (no RGB / RGBA attached).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let (mut h, mut s, mut v) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_hsv(&mut h, &mut s, &mut v)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let (mut hr, mut sr, mut vr) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_hsv(&mut hr, &mut sr, &mut vr)
            .unwrap();
          $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(
            (h, s, v),
            (hr, sr, vr),
            "centered HSV must equal upsample-then-4:4:4"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn every_centered_siting_agrees_horizontally() {
        // 4:2:2 has no vertical axis, so Center / Top / Bottom all reduce to the
        // same horizontal phase — all three agree.
        let center = convert_rgb(ChromaLocation::Center, true);
        assert_eq!(convert_rgb(ChromaLocation::Top, true), center);
        assert_eq!(convert_rgb(ChromaLocation::Bottom, true), center);
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_phase_differs_from_default() {
        // Negative control: on a chroma ramp the centered phase must move chroma
        // relative to the co-sited / nearest-neighbor default — otherwise the
        // byte-identity assertions above would be vacuous.
        assert_ne!(
          convert_rgb(ChromaLocation::Center, true),
          convert_rgb(ChromaLocation::Left, true),
          "centered siting must shift chroma vs the co-sited default"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_path_simd_matches_scalar() {
        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          convert_rgb(ChromaLocation::Center, false),
          "centered path must be bit-identical across the SIMD and scalar tiers"
        );
      }

      // ---- dirty-upper-bit sanitization end-to-end (BE flag threading) -----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_upper_bits_le() {
        // A malformed-but-accepted low-packed frame with bits set ABOVE BITS must
        // decode (centered) identically to the masked clean frame: the centered
        // upsample masks each sample to BITS BEFORE the 1/4-3/4 blend. (At
        // BITS = 16 `upper` is 0, so this is the clean == clean identity.)
        let upper = !(MAXV as u16);
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let decode = |u: &[u16], v: &[u16]| -> Vec<u8> {
          let src = $Frame::new(&yp, u, v, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let up_dirty: Vec<u16> = up.iter().map(|&x| x | upper).collect();
        let vp_dirty: Vec<u16> = vp.iter().map(|&x| x | upper).collect();
        assert_eq!(
          decode(&up_dirty, &vp_dirty),
          decode(&up, &vp),
          "centered LE decode must sanitize dirty upper bits (mask before blend)"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_upper_bits_be() {
        // Same invariant on the big-endian wire path: the mask is applied in the
        // logical domain (after the endian load), so dirty bits are stripped for
        // BE inputs too — confirming the `BE` flag threads through the 4:2:2
        // centered staging. Planes are BE-encoded and decoded via the BE marker /
        // frame / walker.
        let upper = !(MAXV as u16);
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let y_be: Vec<u16> = yp.iter().map(|&x| x.to_be()).collect();
        let decode = |u_logical: &[u16], v_logical: &[u16]| -> Vec<u8> {
          let u_be: Vec<u16> = u_logical.iter().map(|&x| x.to_be()).collect();
          let v_be: Vec<u16> = v_logical.iter().map(|&x| x.to_be()).collect();
          let src = $FrameBe::try_new(&y_be, &u_be, &v_be, W, H, W, W / 2, W / 2).unwrap();
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$MarkerBe>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker_be(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let up_dirty: Vec<u16> = up.iter().map(|&x| x | upper).collect();
        let vp_dirty: Vec<u16> = vp.iter().map(|&x| x | upper).collect();
        assert_eq!(
          decode(&up_dirty, &vp_dirty),
          decode(&up, &vp),
          "centered BE decode must sanitize dirty upper bits (mask before blend)"
        );
      }

      // ---- preflight-ordering atomicity (#302 / #314, cf. #180) ------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_alloc_failure_leaves_outputs_untouched() {
        use crate::resample::ResampleError;

        // luma PLUS a centered RGB decode whose u16 chroma-scratch allocation
        // fails must leave EVERY output buffer — luma included — untouched: the
        // centered scratch is reserved (fallibly) BEFORE any output row is
        // written, so a refusal can't half-update the frame.
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut luma = std::vec![0xABu8; (W * H) as usize];
        let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_luma(&mut luma)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);

        super::super::super::arm_chroma_full_alloc_failure();
        let err = $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
        drop(sink);

        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
          "centered chroma-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
        );
        assert!(
          luma.iter().all(|&b| b == 0xAB),
          "luma must be untouched on the centered alloc-failure path"
        );
        assert!(
          rgb.iter().all(|&b| b == 0xCD),
          "rgb must be untouched on the centered alloc-failure path"
        );
      }

      // ---- ChromaDerivedNcl consistency (#302 / #303 cross-feature seam) ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_chroma_derived_ncl_uses_matrix_tag_fallback() {
        // The high-bit Yuv422p formats are NOT ChromaDerivedNcl-primaries-wired.
        // BOTH paths — the default fused 4:2:2 kernel AND the centered 4:4:4
        // kernel — resolve ChromaDerivedNcl via the shared BT.709 matrix-tag
        // fallback (`Coefficients::for_matrix`), IGNORING the ColorSpec primaries,
        // so default and centered stay internally consistent (the centered phase
        // shift is the ONLY difference). Guards that consistency AND that the
        // centered path did not accidentally half-adopt primaries on one tier.
        use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

        let (yp, up, vp) = ramp_planes_n(MAXV);
        let spec = |loc: ChromaLocation| {
          ColorSpec::from_info(
            PixelFormat::Yuv422p,
            ColorInfo::new(
              Primaries::Bt2020,
              Transfer::Bt709,
              ColorMatrix::ChromaDerivedNcl,
              DynamicRange::Limited,
              loc,
            ),
          )
        };
        let decode_cdn = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_color_spec(&spec(loc));
          $walker(&src, false, KernelMatrix::ChromaDerivedNcl, &mut sink).unwrap();
          rgb
        };
        let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc.clone());
          $walker(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
          rgb
        };

        assert_eq!(
          decode_cdn(ChromaLocation::Center),
          decode_bt709(ChromaLocation::Center),
          "centered high-bit ChromaDerivedNcl must resolve via the BT.709 matrix-tag fallback"
        );
        assert_eq!(
          decode_cdn(ChromaLocation::Left),
          decode_bt709(ChromaLocation::Left),
          "default high-bit ChromaDerivedNcl must resolve via the same BT.709 fallback"
        );
      }

      // ---- no-output invariant: guard runs before the row-offset math --------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "constructs an absurd geometry; the no-op contract is the point, not Miri"
      )]
      fn no_output_row_large_geometry_does_not_overflow() {
        // The no-output guard must run BEFORE the `idx * w` single-plane offset
        // arithmetic. A no-output `process` call never ran an attach-time
        // `w x h x 1` validation, so on a 32-bit target (`usize == u32`) an absurd
        // geometry where `idx * w` exceeds `u32::MAX` would overflow that offset
        // and panic under overflow checks. With no outputs attached, `process`
        // must return `Ok(())` having done NO row math and NO allocation.
        //
        // w = 4, idx = 2^30 -> idx * w = 2^32 = u32::MAX + 1 (overflows u32).
        let w: usize = 4;
        let idx: usize = 1 << 30;
        let h: usize = idx + 1; // idx < height so the row-index check passes
        assert!(
          (idx as u64) * (w as u64) > u32::MAX as u64,
          "test geometry must exceed u32::MAX to exercise the 32-bit offset overflow"
        );

        let y = std::vec![(MAXV / 2) as u16; w];
        let c = std::vec![(MAXV / 2) as u16; w / 2];
        let mut sink =
          MixedSinker::<$Marker>::new(w, h).with_chroma_location(ChromaLocation::Center);
        // No outputs attached: the guard returns before `idx * w` (no overflow
        // panic) and before the centered preflight (no allocation).
        let row = $Row::new(&y, &c, &c, idx, KernelMatrix::Bt601, false);
        crate::PixelSink::process(&mut sink, row).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(
          chroma_len, 0,
          "a no-output large-geometry high-bit row must allocate nothing"
        );
      }

      // ---- mid-frame siting-flip rejection (identity path freeze) ---------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn direct_path_mid_frame_siting_flip_is_rejected() {
        // The identity (no-resample) high-bit planar 4:2:2 decode freezes the
        // horizontal siting on its first output-bearing row. Flipping co-sited ⇆
        // centered mid-frame must reject the next in-sequence row with
        // `ChromaSitingChanged` (leaving the centered chroma scratch untouched);
        // flipping back and retrying then matches a clean single-phase decode
        // byte-for-byte.
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        for (loc1, loc2) in [
          (ChromaLocation::Left, ChromaLocation::Center),
          (ChromaLocation::Center, ChromaLocation::Left),
        ] {
          let want = convert_rgb(loc1.clone(), true);
          let mut rgb = std::vec![0u8; w * h * 3];
          let mut sink = MixedSinker::<$Marker>::new(w, h)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc1.clone())
            .with_simd(true);
          crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
          let row0 = $Row::new(&yp[0..w], &up[0..cw], &vp[0..cw], 0, KernelMatrix::Bt601, false);
          crate::PixelSink::process(&mut sink, row0).unwrap();
          let scratch_len = sink.chroma_full_u16.len();

          sink.set_chroma_location(loc2.clone());
          let row1 = $Row::new(
            &yp[w..2 * w],
            &up[cw..2 * cw],
            &vp[cw..2 * cw],
            1,
            KernelMatrix::Bt601,
            false,
          );
          let err = crate::PixelSink::process(&mut sink, row1).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "direct path {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
          assert_eq!(
            sink.chroma_full_u16.len(),
            scratch_len,
            "{loc1:?}->{loc2:?}: a rejected flip must not grow the chroma scratch"
          );

          sink.set_chroma_location(loc1.clone());
          for r in 1..h {
            let row = $Row::new(
              &yp[r * w..(r + 1) * w],
              &up[r * cw..(r + 1) * cw],
              &vp[r * cw..(r + 1) * cw],
              r,
              KernelMatrix::Bt601,
              false,
            );
            crate::PixelSink::process(&mut sink, row).unwrap();
          }
          drop(sink);
          assert_eq!(
            rgb, want,
            "{loc1:?}: retry after a rejected flip must match a clean in-order decode"
          );
        }
      }
    }
  };
}

hibit_422_chroma_tests!(
  p9,
  9,
  Yuv422p9,
  Yuv422p9Frame,
  yuv422p9_to,
  Yuv444p9,
  Yuv444p9Frame,
  yuv444p9_to,
  Yuv422p9<true>,
  Yuv422p9BeFrame,
  yuv422p9_to_endian,
  Yuv422p9Row
);
hibit_422_chroma_tests!(
  p10,
  10,
  Yuv422p10,
  Yuv422p10Frame,
  yuv422p10_to,
  Yuv444p10,
  Yuv444p10Frame,
  yuv444p10_to,
  Yuv422p10<true>,
  Yuv422p10BeFrame,
  yuv422p10_to_endian,
  Yuv422p10Row
);
hibit_422_chroma_tests!(
  p12,
  12,
  Yuv422p12,
  Yuv422p12Frame,
  yuv422p12_to,
  Yuv444p12,
  Yuv444p12Frame,
  yuv444p12_to,
  Yuv422p12<true>,
  Yuv422p12BeFrame,
  yuv422p12_to_endian,
  Yuv422p12Row
);
hibit_422_chroma_tests!(
  p14,
  14,
  Yuv422p14,
  Yuv422p14Frame,
  yuv422p14_to,
  Yuv444p14,
  Yuv444p14Frame,
  yuv444p14_to,
  Yuv422p14<true>,
  Yuv422p14BeFrame,
  yuv422p14_to_endian,
  Yuv422p14Row
);
hibit_422_chroma_tests!(
  p16,
  16,
  Yuv422p16,
  Yuv422p16Frame,
  yuv422p16_to,
  Yuv444p16,
  Yuv444p16Frame,
  yuv444p16_to,
  Yuv422p16<true>,
  Yuv422p16BeFrame,
  yuv422p16_to_endian,
  Yuv422p16Row
);
