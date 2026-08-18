//! Chroma-siting-aware 4:2:0 upsampling for the semi-planar `Nv12` / `Nv21`
//! (#302) — the semi-planar siblings of `chroma_siting_420` (planar
//! `Yuv420p`).
//!
//! Per format (`Nv12` `U V U V …`, `Nv21` `V U V U …`): the default / co-sited
//! path staying byte-identical to the pre-#302 fused decode (the regression
//! guard, negative-controlled by a ramp that makes the phase observable); the
//! centered RGB / RGBA / HSV identity decodes matching an independent
//! upsample-then-4:4:4 reference; cross-format equivalence to the planar
//! `Yuv420p` centered decode (catches a U/V-swap in the de-interleave); SIMD-vs-
//! scalar parity; the `ColorSpec` siting flow; and the preflight-ordering
//! alloc-failure atomicity (luma untouched), negative-controlled by an unarmed
//! run that DOES write luma.

use super::*;
use crate::{ChromaLocation, ColorMatrix};

const W: u32 = 16;
const H: u32 = 8;

/// Flat luma + per-column chroma ramp (planar U / V), so the horizontal chroma
/// phase is observable — a solid chroma frame would make every siting
/// identical. Identical fixture to `chroma_siting_420::ramp_yuv420p`, so an
/// NV decode of the interleaved form must match the planar twin.
fn ramp_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let y = std::vec![128u8; w * h];
  let mut u = std::vec![0u8; cw * ch];
  let mut v = std::vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (16 + c * 24 + r * 3).min(240) as u8;
      v[r * cw + c] = (240 - c * 24).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Interleaves the half-width planar U / V into a semi-planar chroma plane:
/// `swap = false` packs Nv12 (`U` at the even byte), `true` packs Nv21 (`V` at
/// the even byte). `width` bytes per chroma row, `height / 2` rows.
fn interleave(u: &[u8], v: &[u8], swap: bool) -> Vec<u8> {
  let w = W as usize;
  let cw = w / 2;
  let ch = (H / 2) as usize;
  let mut uv = std::vec![0u8; w * ch];
  for r in 0..ch {
    for c in 0..cw {
      let (even, odd) = if swap {
        (v[r * cw + c], u[r * cw + c])
      } else {
        (u[r * cw + c], v[r * cw + c])
      };
      uv[r * w + 2 * c] = even;
      uv[r * w + 2 * c + 1] = odd;
    }
  }
  uv
}

fn interleave_nv12(u: &[u8], v: &[u8]) -> Vec<u8> {
  interleave(u, v, false)
}

fn interleave_nv21(u: &[u8], v: &[u8]) -> Vec<u8> {
  interleave(u, v, true)
}

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp. Written
/// separately from the production kernel so it is a real oracle.
fn ref_upsample_center_h(c_half: &[u8], width: usize) -> Vec<u8> {
  let half = width / 2;
  let mut out = std::vec![0u8; width];
  for j in 0..half {
    let l = c_half[j.saturating_sub(1)] as i32;
    let m = c_half[j] as i32;
    let r = c_half[if j + 1 < half { j + 1 } else { j }] as i32;
    out[2 * j] = ((l + 3 * m + 2) >> 2) as u8;
    out[2 * j + 1] = ((3 * m + r + 2) >> 2) as u8;
  }
  out
}

/// Full-resolution U / V planes a centered-siting 4:2:0 decode should
/// reconstruct: each luma row `r` takes chroma row `r / 2` (the walker's
/// vertical replication, unchanged by #302) upsampled with the centered
/// weights. Feeding these to a `Yuv444p` conversion is the end-to-end oracle.
fn ref_full_chroma(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  for r in 0..h {
    let cr = r / 2;
    let urow = ref_upsample_center_h(&u420[cr * cw..cr * cw + cw], w);
    let vrow = ref_upsample_center_h(&v420[cr * cw..cr * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

// The full test set is identical bar the format marker, frame type, walker,
// and chroma interleave order, so generate it once per format. Each test fn
// lands in its own `mod` so the names don't collide.
macro_rules! nv_chroma_siting_tests {
  ($mod:ident, $Marker:ident, $Frame:ident, $walker:ident, $interleave:ident) => {
    mod $mod {
      use super::*;

      /// Centered identity-decode RGB for a siting + SIMD toggle.
      fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
        let (yp, up, vp) = ramp_planes();
        let uvp = $interleave(&up, &vp);
        let src = $Frame::new(&yp, &uvp, W, H, W, W);
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
        // The pre-#302 baseline: a sink that never sets a chroma location.
        let baseline = convert_rgb(ChromaLocation::Unspecified, true);
        // Unspecified / Unknown and every FULLY co-sited value keep the exact
        // fused nearest-neighbor decode — bit-for-bit equal even though the
        // chroma plane is a non-trivial ramp. `BottomLeft` is EXCLUDED: co-sited
        // horizontally but bottom-sited vertically (`v = 1`), so it folds the
        // even-row vertical blend (covered by its own test below). `TopLeft` is
        // EXCLUDED for the mirror reason (RFC #238): co-sited horizontally but
        // top-sited vertically (`v = 0`), so it folds the FORWARD odd-row blend.
        for loc in [
          ChromaLocation::Unspecified,
          ChromaLocation::other("unassigned-99"),
          ChromaLocation::Left,
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
        let (yp, up, vp) = ramp_planes();
        let uvp = $interleave(&up, &vp);
        let src = $Frame::new(&yp, &uvp, W, H, W, W);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Left);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full.len();
        let half_len = sink.semi_planar_u_half.len();
        drop(sink);
        assert_eq!(chroma_len, 0, "co-sited path must not grow the chroma scratch");
        assert_eq!(half_len, 0, "co-sited path must not grow the de-interleave scratch");
      }

      // ---- centered path correctness ---------------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_matches_upsample_then_444_reference() {
        let (yp, up, vp) = ramp_planes();
        // Reference: upsample chroma (centered weights) to full resolution,
        // then run the ordinary 4:4:4 decode.
        let (u444, v444) = ref_full_chroma(&up, &vp);
        let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        yuv444p_to(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();

        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          rgb_ref,
          "centered semi-planar RGB must equal upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_grows_chroma_scratch_to_full_width() {
        let (yp, up, vp) = ramp_planes();
        let uvp = $interleave(&up, &vp);
        let src = $Frame::new(&yp, &uvp, W, H, W, W);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full.len();
        let half_len = sink.semi_planar_u_half.len();
        drop(sink);
        assert_eq!(chroma_len, 2 * W as usize, "centered path stages U+V at full width");
        assert_eq!(
          half_len,
          (W / 2) as usize,
          "centered path stages the de-interleaved half-width chroma"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_forward_vblend_differs_from_center_and_bottom() {
        // Top shares Center's horizontal (centered) phase and now consumes its
        // vertical `v = 0` FORWARD fold (RFC #238): the ODD output row box-blends
        // its chroma with the NEXT chroma row. On `ramp_planes`' vertically-varying
        // chroma Top therefore DIFFERS from Center (horizontal-only) AND from
        // Bottom (the BACKWARD even-row blend). Its exact value is pinned by the
        // resample identity == direct-decode cross-check in
        // `chroma_siting_nv12_resample`.
        let center = convert_rgb(ChromaLocation::Center, true);
        assert_ne!(
          convert_rgb(ChromaLocation::Top, true),
          center,
          "Top's forward vertical box blend must differ from Center on a vertically-varying ramp"
        );
        assert_ne!(
          convert_rgb(ChromaLocation::Top, true),
          convert_rgb(ChromaLocation::Bottom, true),
          "Top (forward, odd rows) must differ from Bottom (backward, even rows)"
        );
        assert_ne!(
          convert_rgb(ChromaLocation::Bottom, true),
          center,
          "Bottom's vertical box blend must differ from Center on a vertically-varying chroma ramp"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottomleft_folds_vertical_and_matches_yuv420p() {
        // BottomLeft (h=0 co-sited + v=1) must (a) differ from Bottom (h=0.5) —
        // the horizontal phase — and from TopLeft (v=0.5 replicate) — the
        // vertical fold; (b) equal the SAME BottomLeft decode of the planar
        // Yuv420p twin (the strongest catch for a U/V swap in the de-interleave,
        // and pins the exact value against the validated Yuv420p BottomLeft); and
        // (c) be bit-identical across SIMD and scalar.
        let bl = convert_rgb(ChromaLocation::BottomLeft, true);
        assert_ne!(
          bl,
          convert_rgb(ChromaLocation::Bottom, true),
          "BottomLeft (h=0) must differ from Bottom (h=0.5)"
        );
        assert_ne!(
          bl,
          convert_rgb(ChromaLocation::TopLeft, true),
          "BottomLeft (v=1) must differ from TopLeft (vertical-replicate)"
        );
        assert_eq!(
          bl,
          convert_rgb(ChromaLocation::BottomLeft, false),
          "BottomLeft path must be bit-identical across the SIMD and scalar tiers"
        );

        let (yp, up, vp) = ramp_planes();
        let src420 = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb420 = std::vec![0u8; (W * H * 3) as usize];
        let mut sink420 = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
          .with_rgb(&mut rgb420)
          .unwrap()
          .with_chroma_location(ChromaLocation::BottomLeft);
        yuv420p_to(&src420, false, KernelMatrix::Bt601, &mut sink420).unwrap();
        assert_eq!(
          bl, rgb420,
          "BottomLeft semi-planar must equal BottomLeft Yuv420p of the same planes"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_phase_differs_from_default() {
        // Negative control for the byte-identity guard: on a chroma ramp the
        // centered phase must move chroma relative to the co-sited default.
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

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_matches_yuv420p_centered() {
        // De-interleaving the interleaved chroma back to planar U / V must
        // reconstruct the SAME planes the planar twin holds, so a centered NV
        // decode is byte-identical to a centered Yuv420p decode — the strongest
        // catch for a U/V swap in the de-interleave (`swap_uv`). Uses Bt601:
        // this equivalence holds on the shared matrix-tag path. `ChromaDerivedNcl`
        // is the lone divergence (Yuv420p is primaries-wired, NV is not) and is
        // covered by `centered_chroma_derived_ncl_uses_matrix_tag_fallback`.
        let (yp, up, vp) = ramp_planes();
        let src420 = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb420 = std::vec![0u8; (W * H * 3) as usize];
        let mut sink420 = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
          .with_rgb(&mut rgb420)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        yuv420p_to(&src420, false, KernelMatrix::Bt601, &mut sink420).unwrap();

        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          rgb420,
          "centered semi-planar must equal centered Yuv420p of the same planes"
        );
      }

      // ---- ChromaDerivedNcl: NV uses the matrix-tag fallback, not primaries ---

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_chroma_derived_ncl_uses_matrix_tag_fallback() {
        // NV is NOT ChromaDerivedNcl-primaries-wired — only Yuv420p is (#316 /
        // #303). BOTH NV paths (the default fused kernel AND the centered 4:4:4
        // kernel) resolve ChromaDerivedNcl via the shared BT.709 matrix-tag
        // fallback (`Coefficients::for_matrix`), IGNORING the ColorSpec
        // primaries — so default and centered stay internally consistent (the
        // centered phase shift is the ONLY difference between them). A
        // primaries-derived NV decode (the Yuv420p behaviour) is a documented
        // #302 / #303 follow-up. This guards that consistency AND that NV does
        // not half-adopt primaries on only one path — the inverse of the
        // Yuv420p #316 test, which asserts the primaries-derived path.
        use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

        let (yp, up, vp) = ramp_planes();
        let uvp = $interleave(&up, &vp);

        // ChromaDerivedNcl + Bt2020 primaries: were NV to honour the primaries
        // (it must NOT), the decode would diverge from BT.709 (~Bt2020Ncl). The
        // PixelFormat in the spec is cosmetic — the sink consumes only
        // chroma_location + primaries.
        let spec = |loc: ChromaLocation| {
          ColorSpec::from_info(
            PixelFormat::Yuv420p,
            ColorInfo::new(
              Primaries::Bt2020,
              Transfer::Bt709,
              ColorMatrix::ChromaDerivedNcl,
              DynamicRange::Limited,
              loc,
            ),
          )
        };
        // ChromaDerivedNcl(Bt2020) decode via the ColorSpec path.
        let decode_cdn = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&yp, &uvp, W, H, W, W);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_color_spec(&spec(loc));
          $walker(&src, false, KernelMatrix::ChromaDerivedNcl, &mut sink).unwrap();
          rgb
        };
        // The BT.709 reference the matrix-tag fallback must equal.
        let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&yp, &uvp, W, H, W, W);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc.clone());
          $walker(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
          rgb
        };

        // Centered: ChromaDerivedNcl(Bt2020) == BT.709 → the fallback IS BT.709,
        // NOT the Bt2020-primaries-derived coefficients.
        assert_eq!(
          decode_cdn(ChromaLocation::Center),
          decode_bt709(ChromaLocation::Center),
          "centered NV ChromaDerivedNcl must resolve via the BT.709 matrix-tag fallback, not primaries"
        );
        // Default (co-sited): same fallback → default and centered NV agree on
        // the coefficient path (neither half-adopts primaries).
        assert_eq!(
          decode_cdn(ChromaLocation::Left),
          decode_bt709(ChromaLocation::Left),
          "default NV ChromaDerivedNcl must resolve via the same BT.709 matrix-tag fallback"
        );
      }

      // ---- RGBA / HSV identity outputs also honor siting -------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgba_and_hsv_match_444_reference() {
        let (yp, up, vp) = ramp_planes();
        let uvp = $interleave(&up, &vp);
        let (u444, v444) = ref_full_chroma(&up, &vp);

        // RGBA-only path.
        {
          let src = $Frame::new(&yp, &uvp, W, H, W, W);
          let mut rgba = std::vec![0u8; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

          let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
            .with_rgba(&mut rgba_ref)
            .unwrap();
          yuv444p_to(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(rgba, rgba_ref, "centered RGBA must equal upsample-then-4:4:4");
        }

        // HSV-direct path (no RGB / RGBA attached).
        {
          let src = $Frame::new(&yp, &uvp, W, H, W, W);
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

          let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let (mut hr, mut sr, mut vr) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
            .with_hsv(&mut hr, &mut sr, &mut vr)
            .unwrap();
          yuv444p_to(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!((h, s, v), (hr, sr, vr), "centered HSV must equal upsample-then-4:4:4");
        }
      }

      // ---- end-to-end ColorSpec flow (no manual with_chroma_location) ------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn color_spec_center_drives_decode_without_manual_chroma_call() {
        use crate::{
          ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer, YuvOptions,
        };

        let (yp, up, vp) = ramp_planes();
        let uvp = $interleave(&up, &vp);
        let src = $Frame::new(&yp, &uvp, W, H, W, W);

        // Drive the decode from a ColorSpec carrying ChromaLocation::Center via
        // the NORMAL path: YuvOptions::from_color_spec for the walk and the
        // sink's ColorSpec entry point for the siting — no manual call.
        let info = ColorInfo::new(
          Primaries::Bt709,
          Transfer::Bt709,
          ColorMatrix::Bt601,
          DynamicRange::Limited,
          ChromaLocation::Center,
        );
        let spec = ColorSpec::from_info(PixelFormat::Yuv420p, info);
        let opts = YuvOptions::from_color_spec(&spec).unwrap();
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_color_spec(&spec);
        $walker(&src, opts.full_range(), opts.matrix(), &mut sink).unwrap();
        drop(sink);

        assert_ne!(
          rgb,
          convert_rgb(ChromaLocation::Unspecified, true),
          "ColorSpec ChromaLocation::Center must change the decode via the options path"
        );
        assert_eq!(
          rgb,
          convert_rgb(ChromaLocation::Center, true),
          "ColorSpec-driven centered decode must equal the explicit centered path"
        );
      }

      // ---- preflight-ordering atomicity (#302, cf. #180 / #308) ------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_alloc_failure_leaves_outputs_untouched() {
        use crate::resample::ResampleError;

        let (yp, up, vp) = ramp_planes();
        let uvp = $interleave(&up, &vp);
        let src = $Frame::new(&yp, &uvp, W, H, W, W);

        // Negative control: unarmed, the SAME luma + centered-RGB config DOES
        // write luma — so the armed "untouched" assertion below is non-vacuous.
        {
          let mut luma_ok = std::vec![0xABu8; (W * H) as usize];
          let mut rgb_ok = std::vec![0xCDu8; (W * H * 3) as usize];
          let mut sink_ok = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_luma(&mut luma_ok)
            .unwrap()
            .with_rgb(&mut rgb_ok)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink_ok).unwrap();
          drop(sink_ok);
          assert!(
            luma_ok.iter().any(|&b| b != 0xAB),
            "control: the centered path writes luma when the scratch alloc is not armed"
          );
        }

        // Armed: a centered RGB decode whose chroma-scratch allocation fails
        // must leave EVERY output — luma included — untouched, because the
        // scratch is reserved (fallibly) BEFORE any output row is written.
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
          matches!(err, MixedSinkerError::Resample(ResampleError::AllocationFailed(_))),
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
    }
  };
}

nv_chroma_siting_tests!(nv12, Nv12, Nv12Frame, nv12_to, interleave_nv12);
nv_chroma_siting_tests!(nv21, Nv21, Nv21Frame, nv21_to, interleave_nv21);

// ---- mid-frame siting-flip rejection (identity path freeze) -----------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_direct_path_mid_frame_siting_flip_is_rejected() {
  // The identity (no-resample) semi-planar 4:2:0 `Nv12` decode freezes the effective
  // phase — the horizontal centered flag and the vertical `Bottom` / `Top` flags —
  // on its first output-bearing row, mirroring the planar `Yuv420p` twin. Flipping
  // `Bottom` / `Top` ⇆ co-sited / `Center` mid-frame must reject the next
  // in-sequence row with `ChromaSitingChanged`, WITHOUT growing the chroma scratch
  // OR advancing the stateful `chroma_prev` vertical lookback (its validity tag
  // included). For a held `Top` odd row the flip is rejected BEFORE the held row is
  // touched (RFC #238). Flipping back and retrying then matches a clean
  // single-phase decode byte-for-byte.
  use super::super::MixedSinkerError;
  let (y, u, v) = ramp_planes();
  let uv = interleave_nv12(&u, &v);
  let w = W as usize;
  let h = H as usize;
  for (loc1, loc2) in [
    (ChromaLocation::Bottom, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Center),
    (ChromaLocation::Top, ChromaLocation::Center),
    (ChromaLocation::Center, ChromaLocation::Top),
    (ChromaLocation::TopLeft, ChromaLocation::Left),
  ] {
    // Reference: a clean whole-frame decode at the held siting.
    let mut want = std::vec![0u8; w * h * 3];
    {
      let src = Nv12Frame::new(&y, &uv, W, H, W, W);
      let mut ref_sink = MixedSinker::<Nv12>::new(w, h)
        .with_rgb(&mut want)
        .unwrap()
        .with_chroma_location(loc1.clone())
        .with_simd(true);
      nv12_to(&src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
    }

    let mut rgb = std::vec![0u8; w * h * 3];
    let mut sink = MixedSinker::<Nv12>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc1.clone())
      .with_simd(true);
    crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
    for r in 0..2 {
      let cr = r / 2;
      let row = Nv12Row::new(
        &y[r * w..(r + 1) * w],
        &uv[cr * w..(cr + 1) * w],
        r,
        KernelMatrix::Bt601,
        false,
      );
      crate::PixelSink::process(&mut sink, row).unwrap();
    }
    let scratch_len = sink.chroma_full.len();
    let prev_len = sink.chroma_prev.len();
    let prev_tag = sink.chroma_prev_row;

    sink.set_chroma_location(loc2.clone());
    let row2 = Nv12Row::new(
      &y[2 * w..3 * w],
      &uv[w..2 * w],
      2,
      KernelMatrix::Bt601,
      false,
    );
    let err = crate::PixelSink::process(&mut sink, row2).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "direct path {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
    assert_eq!(
      sink.chroma_full.len(),
      scratch_len,
      "{loc1:?}->{loc2:?}: a rejected flip must not grow the chroma scratch"
    );
    assert_eq!(
      sink.chroma_prev.len(),
      prev_len,
      "{loc1:?}->{loc2:?}: a rejected flip must not grow the vertical lookback"
    );
    assert_eq!(
      sink.chroma_prev_row, prev_tag,
      "{loc1:?}->{loc2:?}: a rejected flip must not advance the lookback tag"
    );

    sink.set_chroma_location(loc1.clone());
    for r in 2..h {
      let cr = r / 2;
      let row = Nv12Row::new(
        &y[r * w..(r + 1) * w],
        &uv[cr * w..(cr + 1) * w],
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

#[test]
fn nv21_begin_frame_drops_held_top_row() {
  // RFC #238 Top forward delay: an `Nv21` odd row is HELD (`chroma_top_pending`)
  // to be emitted at the following even row. If a Top frame is interrupted after
  // a held odd non-last row, `begin_frame` MUST drop that held state (and the
  // frozen Top phase) so frame N's deferred colour row can never flush into frame
  // N+1. Regression for the Nv21 `begin_frame` that (unlike Nv12) omitted the Top
  // resets.
  let (y, u, v) = ramp_planes();
  let vu = interleave_nv21(&u, &v);
  let w = W as usize;
  let mut rgb = std::vec![0u8; w * H as usize * 3];
  let mut sink = MixedSinker::<Nv21>::new(w, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Top)
    .with_simd(true);
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  // Row 0 (even, co-sited, emitted) then row 1 (odd, HELD).
  for r in 0..2 {
    let cr = r / 2;
    crate::PixelSink::process(
      &mut sink,
      Nv21Row::new(
        &y[r * w..(r + 1) * w],
        &vu[cr * w..(cr + 1) * w],
        r,
        KernelMatrix::Bt601,
        false,
      ),
    )
    .unwrap();
  }
  assert!(
    sink.chroma_top_pending.is_some(),
    "an odd Top row must be held before the next even row"
  );

  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  assert!(
    sink.chroma_top_pending.is_none(),
    "begin_frame must drop the held Top odd row so it can't flush into the next frame"
  );
  assert!(
    sink.frozen_chroma_top_v.is_none(),
    "begin_frame must clear the frozen Top phase so the next frame may re-pick siting"
  );
}

/// Size-parametric ramp planes (Y + half-width U / V) for the identity `Top`
/// cross-format test — a two-axis chroma ramp so the FORWARD vertical fold is
/// observable, and a varying luma so the co-sited even rows are non-trivial too.
#[cfg(test)]
fn ramp_planes_hw(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = w / 2;
  let ch = h.div_ceil(2);
  let y: Vec<u8> = (0..w * h).map(|i| (40 + (i * 7) % 180) as u8).collect();
  let mut u = std::vec![0u8; cw * ch];
  let mut v = std::vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (16 + c * 20 + r * 17).min(240) as u8;
      v[r * cw + c] = (240 - c * 18 - r * 11).max(16) as u8;
    }
  }
  (y, u, v)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv_direct_top_matches_yuv420p_top() {
  // The identity FORWARD one-row-delay `Top` / `TopLeft` decode (RFC #238) must
  // reproduce the validated planar `Yuv420p` `Top` / `TopLeft` decode of the same
  // de-interleaved planes byte-for-byte — de-interleaving Nv12 (`U V …`) and Nv21
  // (`V U …`) both back to the same planar U / V (the strongest catch for a U/V
  // swap in the delayed path). EVEN heights end on the trailing-odd co-sited
  // clamp; ODD heights end on the final even-row flush of the held odd row — both
  // exercised. Bt601 keeps this on the shared matrix-tag path.
  for (w, h) in [(16usize, 8usize), (16, 6), (16, 5), (8, 3), (8, 7)] {
    let (y, u, v) = ramp_planes_hw(w, h);
    let cw = w / 2;
    let uv12 = interleave_hw(&u, &v, w, h, false);
    let uv21 = interleave_hw(&u, &v, w, h, true);
    for loc in [ChromaLocation::Top, ChromaLocation::TopLeft] {
      // Planar Yuv420p reference (validated Top forward decode).
      let mut want = std::vec![0u8; w * h * 3];
      {
        let src = Yuv420pFrame::new(
          &y, &u, &v, w as u32, h as u32, w as u32, cw as u32, cw as u32,
        );
        let mut sink = MixedSinker::<Yuv420p>::new(w, h)
          .with_rgb(&mut want)
          .unwrap()
          .with_chroma_location(loc.clone());
        yuv420p_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
      }
      for simd in [true, false] {
        let mut nv12 = std::vec![0u8; w * h * 3];
        {
          let src = Nv12Frame::new(&y, &uv12, w as u32, h as u32, w as u32, w as u32);
          let mut sink = MixedSinker::<Nv12>::new(w, h)
            .with_rgb(&mut nv12)
            .unwrap()
            .with_chroma_location(loc.clone())
            .with_simd(simd);
          nv12_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        }
        assert_eq!(
          nv12, want,
          "Nv12 {loc:?} identity {w}x{h} simd={simd} must equal Yuv420p {loc:?}"
        );
        let mut nv21 = std::vec![0u8; w * h * 3];
        {
          let src = Nv21Frame::new(&y, &uv21, w as u32, h as u32, w as u32, w as u32);
          let mut sink = MixedSinker::<Nv21>::new(w, h)
            .with_rgb(&mut nv21)
            .unwrap()
            .with_chroma_location(loc.clone())
            .with_simd(simd);
          nv21_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        }
        assert_eq!(
          nv21, want,
          "Nv21 {loc:?} identity {w}x{h} simd={simd} must equal Yuv420p {loc:?}"
        );
      }
    }
  }
}

/// Size-parametric interleave: `swap = false` packs Nv12 (`U` at the even byte),
/// `true` packs Nv21 (`V` at the even byte). `w` bytes per chroma row,
/// `h.div_ceil(2)` rows.
#[cfg(test)]
fn interleave_hw(u: &[u8], v: &[u8], w: usize, h: usize, swap: bool) -> Vec<u8> {
  let cw = w / 2;
  let ch = h.div_ceil(2);
  let mut uv = std::vec![0u8; w * ch];
  for r in 0..ch {
    for c in 0..cw {
      let (even, odd) = if swap {
        (v[r * cw + c], u[r * cw + c])
      } else {
        (u[r * cw + c], v[r * cw + c])
      };
      uv[r * w + 2 * c] = even;
      uv[r * w + 2 * c + 1] = odd;
    }
  }
  uv
}
