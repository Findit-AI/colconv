//! Chroma-siting-aware 4:2:0 upsampling for the planar **YUVA** family
//! (#302): `Yuva420p` (8-bit) + `Yuva420p9` / `Yuva420p10` / `Yuva420p12` /
//! `Yuva420p16` (high-bit, low-packed).
//!
//! YUVA is planar 4:2:0 YUV (separate half-width / half-height U & V planes)
//! PLUS a **full-resolution** alpha plane that is never subsampled — so the
//! chroma siting is IDENTICAL to the non-alpha `Yuv420p` twin, and the alpha
//! plane passes through unchanged on every path. These tests therefore assert,
//! per format:
//!   * the default / co-sited / unspecified sitings stay byte-identical to the
//!     pre-#302 nearest-neighbor decode (the regression guard) + a negative
//!     control that the centered phase actually moves chroma;
//!   * the centered RGBA decode carries the **real source alpha** (not opaque
//!     `0xFF`), matching an independent "upsample-then-4:4:4-with-alpha"
//!     reference;
//!   * the centered RGB / HSV (and the high-bit `u16` twins) match the
//!     upsample-then-4:4:4 alpha-drop reference;
//!   * **alpha preservation**: the centered RGBA's alpha channel equals BOTH
//!     the source alpha plane AND the default path's alpha (siting never
//!     touches alpha);
//!   * SIMD == scalar on the centered path;
//!   * the preflight-ordering atomicity (a centered chroma-scratch alloc
//!     failure leaves luma AND colour untouched);
//!   * `ChromaDerivedNcl` consistency (YUVA is NOT primaries-wired, so BOTH the
//!     default and centered paths resolve it via the BT.709 matrix-tag
//!     fallback — they agree bar the centered phase shift);
//!   * (high-bit) dirty-upper-bit sanitization (mask before the blend), LE+BE.

use super::*;
use crate::ChromaLocation;

const W: u32 = 16;
const H: u32 = 8;

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp, on logical
/// `u32` samples. Written separately from the production kernel so it is a real
/// oracle; shared by the 8-bit (`u8`) and high-bit (`u16`) suites.
fn ref_upsample_center_h(c_half: &[u32], width: usize) -> Vec<u32> {
  let half = width / 2;
  let mut out = std::vec![0u32; width];
  for j in 0..half {
    let l = c_half[j.saturating_sub(1)];
    let m = c_half[j];
    let r = c_half[if j + 1 < half { j + 1 } else { j }];
    out[2 * j] = (l + 3 * m + 2) >> 2;
    out[2 * j + 1] = (3 * m + r + 2) >> 2;
  }
  out
}

// ===========================================================================
// 8-bit Yuva420p
// ===========================================================================

mod p8 {
  use super::*;
  use crate::{
    frame::{Yuva420pFrame, Yuva444pFrame},
    source::{Yuva420p, Yuva444p, yuva420p_to, yuva444p_to},
  };

  /// Flat mid-gray luma + per-column chroma ramp (distinct adjacent columns so
  /// the horizontal phase is observable; `+ r * 5` keeps chroma rows distinct
  /// so a vertical mistake would surface) + a per-pixel alpha gradient that is
  /// NOT all-opaque (so the alpha-preservation assertions are non-vacuous).
  fn ramp_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let ch = h / 2;
    let y = std::vec![128u8; w * h];
    let mut u = std::vec![0u8; cw * ch];
    let mut v = std::vec![0u8; cw * ch];
    let mut a = std::vec![0u8; w * h];
    for r in 0..ch {
      for c in 0..cw {
        u[r * cw + c] = (16 * c + 16 + r * 5).min(255) as u8;
        v[r * cw + c] = (255u32.saturating_sub(16 * c as u32)).max(16) as u8;
      }
    }
    for r in 0..h {
      for c in 0..w {
        // A varying, non-opaque alpha so a dropped / opaqued alpha is caught.
        a[r * w + c] = ((r * w + c) % 251 + 3) as u8;
      }
    }
    (y, u, v, a)
  }

  /// The full-resolution U / V a centered 4:2:0 decode reconstructs: each luma
  /// row `r` takes chroma row `r / 2` (the walker's vertical replication,
  /// unchanged by #302) horizontally upsampled with the centered weights.
  fn ref_full_chroma(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let mut u444 = std::vec![0u8; w * h];
    let mut v444 = std::vec![0u8; w * h];
    for r in 0..h {
      let cr = r / 2;
      let urow: Vec<u32> = u420[cr * cw..cr * cw + cw]
        .iter()
        .map(|&x| x as u32)
        .collect();
      let vrow: Vec<u32> = v420[cr * cw..cr * cw + cw]
        .iter()
        .map(|&x| x as u32)
        .collect();
      let uo = ref_upsample_center_h(&urow, w);
      let vo = ref_upsample_center_h(&vrow, w);
      for c in 0..w {
        u444[r * w + c] = uo[c] as u8;
        v444[r * w + c] = vo[c] as u8;
      }
    }
    (u444, v444)
  }

  /// The full-resolution U / V a **`Bottom`** (`v = 1`) 4:2:0 decode reconstructs
  /// (RFC #238 S4-D): the even luma row `2i` vertically box-blends chroma rows
  /// `{i - 1, i}` (round-half-up, top-edge clamp) BEFORE the centered horizontal
  /// upsample; the odd luma row `2i + 1` is co-sited with chroma row `i` and keeps
  /// the plain centered upsample. Mirrors the streaming delay-line kernel
  /// (vertical blend rounded to u8, then horizontal), so it is the direct-decode
  /// ground truth.
  fn ref_full_chroma_bottom(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let mut u444 = std::vec![0u8; w * h];
    let mut v444 = std::vec![0u8; w * h];
    let vblend = |plane: &[u8], cr: usize, prev: usize| -> Vec<u32> {
      (0..cw)
        .map(|c| (u32::from(plane[prev * cw + c]) + u32::from(plane[cr * cw + c]) + 1) >> 1)
        .collect()
    };
    for r in 0..h {
      let cr = r / 2;
      let (urow, vrow) = if r & 1 == 0 {
        let prev = cr.saturating_sub(1);
        (vblend(u420, cr, prev), vblend(v420, cr, prev))
      } else {
        (
          u420[cr * cw..cr * cw + cw]
            .iter()
            .map(|&x| u32::from(x))
            .collect(),
          v420[cr * cw..cr * cw + cw]
            .iter()
            .map(|&x| u32::from(x))
            .collect(),
        )
      };
      let uo = ref_upsample_center_h(&urow, w);
      let vo = ref_upsample_center_h(&vrow, w);
      for c in 0..w {
        u444[r * w + c] = uo[c] as u8;
        v444[r * w + c] = vo[c] as u8;
      }
    }
    (u444, v444)
  }

  fn frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], a: &'a [u8]) -> Yuva420pFrame<'a> {
    Yuva420pFrame::try_new(y, u, v, a, W, H, W, W / 2, W / 2, W).unwrap()
  }

  fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
    let (y, u, v, a) = ramp_planes();
    let mut rgb = std::vec![0u8; (W * H * 3) as usize];
    let mut sink = MixedSinker::<Yuva420p>::new(W as usize, H as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc)
      .with_simd(simd);
    yuva420p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
    rgb
  }

  fn convert_rgba(loc: ChromaLocation, simd: bool) -> Vec<u8> {
    let (y, u, v, a) = ramp_planes();
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuva420p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(loc)
      .with_simd(simd);
    yuva420p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
    rgba
  }

  // ---- default / co-sited path is byte-identical (regression guard) ----

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn default_and_cosited_sitings_are_byte_identical() {
    let baseline = convert_rgba(ChromaLocation::Unspecified, true);
    for loc in [
      ChromaLocation::Unspecified,
      ChromaLocation::Unknown(99),
      ChromaLocation::Left,
      ChromaLocation::TopLeft,
      ChromaLocation::BottomLeft,
    ] {
      assert_eq!(
        convert_rgba(loc, true),
        baseline,
        "siting {loc:?} must keep the byte-identical default YUVA decode"
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn default_path_does_not_allocate_chroma_scratch() {
    let (y, u, v, a) = ramp_planes();
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuva420p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Left);
    yuva420p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
    let chroma_len = sink.chroma_full.len();
    drop(sink);
    assert_eq!(
      chroma_len, 0,
      "co-sited path must not grow the chroma scratch"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn center_grows_chroma_scratch_to_full_width() {
    let (y, u, v, a) = ramp_planes();
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuva420p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuva420p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
    let chroma_len = sink.chroma_full.len();
    drop(sink);
    assert_eq!(
      chroma_len,
      2 * W as usize,
      "centered path stages U+V at full width"
    );
  }

  // ---- centered path correctness (vs the upsample-then-4:4:4 oracle) ----

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn center_rgba_matches_upsample_then_444_with_real_alpha() {
    let (y, u, v, a) = ramp_planes();
    let (u444, v444) = ref_full_chroma(&u, &v);
    // The reference is a 4:4:4 YUVA decode on the upsampled chroma + the SAME
    // full-res alpha plane — so its RGBA carries the real source alpha.
    let ref_src = Yuva444pFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuva444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuva444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      convert_rgba(ChromaLocation::Center, true),
      rgba_ref,
      "centered YUVA RGBA must equal upsample-then-4:4:4 (real source alpha)"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn center_rgb_matches_upsample_then_444_reference() {
    let (y, u, v, a) = ramp_planes();
    let (u444, v444) = ref_full_chroma(&u, &v);
    let ref_src = Yuva444pFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
    let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
    let mut ref_sink = MixedSinker::<Yuva444p>::new(W as usize, H as usize)
      .with_rgb(&mut rgb_ref)
      .unwrap();
    yuva444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      convert_rgb(ChromaLocation::Center, true),
      rgb_ref,
      "centered YUVA RGB (alpha-drop) must equal upsample-then-4:4:4"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn center_hsv_matches_upsample_then_444_reference() {
    let (y, u, v, a) = ramp_planes();
    let (u444, v444) = ref_full_chroma(&u, &v);
    let (mut h, mut s, mut vv) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let src = frame(&y, &u, &v, &a);
    let mut sink = MixedSinker::<Yuva420p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut vv)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuva420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuva444pFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
    let (mut hr, mut sr, mut vr) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut ref_sink = MixedSinker::<Yuva444p>::new(W as usize, H as usize)
      .with_hsv(&mut hr, &mut sr, &mut vr)
      .unwrap();
    yuva444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      (h, s, vv),
      (hr, sr, vr),
      "centered YUVA HSV must equal upsample-then-4:4:4"
    );
  }

  // ---- alpha preservation (siting never touches alpha) -----------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn centered_alpha_equals_source_and_default_alpha() {
    let (_, _, _, a) = ramp_planes();
    let center = convert_rgba(ChromaLocation::Center, true);
    let default = convert_rgba(ChromaLocation::Left, true);
    for (i, &src_a) in a.iter().enumerate() {
      assert_eq!(
        center[i * 4 + 3],
        src_a,
        "centered alpha at px {i} must equal the source alpha plane"
      );
      assert_eq!(
        center[i * 4 + 3],
        default[i * 4 + 3],
        "centered alpha at px {i} must equal the default-path alpha"
      );
    }
    // The colour channels DO differ (negative control for the chroma shift);
    // only alpha is invariant across the siting.
    assert_ne!(
      center, default,
      "centered colour must differ from the default"
    );
  }

  // ---- negative control + SIMD parity ----------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn top_routes_like_center_and_bottom_folds_vertically() {
    // Top shares the horizontal centered phase AND keeps the vertical pairing
    // co-sited (RFC #238 S4-D folds only Bottom's `v = 1`), so Top == Center
    // byte-for-byte. Bottom additionally applies the vertical `v = 1` blend, so on
    // the vertically-varying fixture it diverges from Center.
    let center = convert_rgba(ChromaLocation::Center, true);
    assert_eq!(convert_rgba(ChromaLocation::Top, true), center);
    assert_ne!(convert_rgba(ChromaLocation::Bottom, true), center);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn bottom_rgba_matches_vfold_upsample_then_444_with_real_alpha() {
    // RFC #238 S4-D: the `Bottom` (`v = 1`) direct decode reconstructs chroma
    // with the vertical box-blend + centered horizontal upsample, so its RGBA
    // equals a 4:4:4 YUVA decode on that v-fold-reconstructed chroma + the SAME
    // full-res source alpha — on BOTH the SIMD and scalar tiers (0-ULP).
    let (y, u, v, a) = ramp_planes();
    let (u444, v444) = ref_full_chroma_bottom(&u, &v);
    let ref_src = Yuva444pFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuva444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuva444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      convert_rgba(ChromaLocation::Bottom, true),
      rgba_ref,
      "bottom YUVA RGBA (SIMD) must equal v-fold-upsample-then-4:4:4 (real source alpha)"
    );
    assert_eq!(
      convert_rgba(ChromaLocation::Bottom, false),
      rgba_ref,
      "bottom YUVA RGBA (scalar) must equal v-fold-upsample-then-4:4:4"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn centered_phase_differs_from_default() {
    assert_ne!(
      convert_rgba(ChromaLocation::Center, true),
      convert_rgba(ChromaLocation::Left, true),
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
      convert_rgba(ChromaLocation::Center, true),
      convert_rgba(ChromaLocation::Center, false),
      "centered RGBA must be bit-identical across the SIMD and scalar tiers"
    );
    assert_eq!(
      convert_rgb(ChromaLocation::Center, true),
      convert_rgb(ChromaLocation::Center, false),
      "centered RGB must be bit-identical across the SIMD and scalar tiers"
    );
  }

  // ---- preflight-ordering atomicity (#302, cf. #180) -------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn centered_alloc_failure_leaves_outputs_untouched() {
    use crate::resample::ResampleError;

    let (y, u, v, a) = ramp_planes();
    let src = frame(&y, &u, &v, &a);
    let mut luma = std::vec![0xABu8; (W * H) as usize];
    let mut rgba = std::vec![0xCDu8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuva420p>::new(W as usize, H as usize)
      .with_luma(&mut luma)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);

    super::super::super::arm_chroma_full_alloc_failure();
    let err = yuva420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
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
      rgba.iter().all(|&b| b == 0xCD),
      "rgba must be untouched on the centered alloc-failure path"
    );
  }

  // ---- ChromaDerivedNcl consistency (#302 / #303 cross-feature seam) ----

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn centered_chroma_derived_ncl_uses_matrix_tag_fallback() {
    // YUVA is NOT ChromaDerivedNcl-primaries-wired. BOTH paths — the default
    // fused 4:2:0 kernel AND the centered 4:4:4 kernel — resolve
    // ChromaDerivedNcl via the shared BT.709 matrix-tag fallback, IGNORING the
    // ColorSpec primaries, so default and centered stay internally consistent
    // (the centered phase shift is the ONLY difference between them).
    use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

    let (y, u, v, a) = ramp_planes();
    let spec = |loc: ChromaLocation| {
      ColorSpec::from_info(
        PixelFormat::Yuva420p,
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
      let mut rgb = std::vec![0u8; (W * H * 3) as usize];
      let mut sink = MixedSinker::<Yuva420p>::new(W as usize, H as usize)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_color_spec(spec(loc));
      yuva420p_to(
        &frame(&y, &u, &v, &a),
        false,
        ColorMatrix::ChromaDerivedNcl,
        &mut sink,
      )
      .unwrap();
      rgb
    };
    let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
      let mut rgb = std::vec![0u8; (W * H * 3) as usize];
      let mut sink = MixedSinker::<Yuva420p>::new(W as usize, H as usize)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_chroma_location(loc);
      yuva420p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt709, &mut sink).unwrap();
      rgb
    };

    assert_eq!(
      decode_cdn(ChromaLocation::Center),
      decode_bt709(ChromaLocation::Center),
      "centered YUVA ChromaDerivedNcl must resolve via the BT.709 matrix-tag fallback"
    );
    assert_eq!(
      decode_cdn(ChromaLocation::Left),
      decode_bt709(ChromaLocation::Left),
      "default YUVA ChromaDerivedNcl must resolve via the same BT.709 fallback"
    );
  }
}

// ===========================================================================
// High-bit Yuva420p9 / 10 / 12 / 16 (low-packed)
// ===========================================================================

// Identical bar the bit depth, format marker, frame type, and walker — so
// generate the suite once per depth. The macro instantiates each with its
// little-endian marker (a sample's wire `u16` equals its logical value on the
// LE test host); the references compute in that logical domain. The endianness
// re-encode is exercised by the dirty-bit BE test (via the BE frame / walker).
macro_rules! hibit_yuva420_chroma_tests {
  (
    $mod:ident,
    $bits:expr,
    $Marker:ident,
    $LeFrame:ident,
    $BeFrame:ident,
    $walker:ident,
    $walker_be:ident,
    $Ref:ident,
    $RefFrame:ident,
    $ref_walker:ident,
    $MarkerBe:ty
  ) => {
    mod $mod {
      use super::*;
      use crate::{
        frame::{$BeFrame, $LeFrame, $RefFrame},
        source::{$Marker, $Ref, $ref_walker, $walker, $walker_be},
      };

      const MAXV: u32 = (1u32 << $bits) - 1;

      /// Flat mid-gray luma + per-column chroma ramp + a varying (non-opaque)
      /// alpha plane, all low-packed at `$bits`.
      fn ramp_planes() -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        let ch = h / 2;
        let step = (MAXV / 16).max(1);
        let y = std::vec![(MAXV / 2) as u16; w * h];
        let mut u = std::vec![0u16; cw * ch];
        let mut v = std::vec![0u16; cw * ch];
        let mut a = std::vec![0u16; w * h];
        for r in 0..ch {
          for c in 0..cw {
            u[r * cw + c] = (step * c as u32 + step + r as u32 * 5).min(MAXV) as u16;
            v[r * cw + c] = MAXV.saturating_sub(step * c as u32).max(step) as u16;
          }
        }
        for r in 0..h {
          for c in 0..w {
            a[r * w + c] = (((r * w + c) as u32 * 97 + 5) % (MAXV + 1)) as u16;
          }
        }
        (y, u, v, a)
      }

      /// The full-resolution U / V a centered high-bit 4:2:0 decode
      /// reconstructs (logical `u16`).
      fn ref_full_chroma(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        let mut u444 = std::vec![0u16; w * h];
        let mut v444 = std::vec![0u16; w * h];
        for r in 0..h {
          let cr = r / 2;
          let urow: Vec<u32> =
            u420[cr * cw..cr * cw + cw].iter().map(|&x| x as u32).collect();
          let vrow: Vec<u32> =
            v420[cr * cw..cr * cw + cw].iter().map(|&x| x as u32).collect();
          let uo = ref_upsample_center_h(&urow, w);
          let vo = ref_upsample_center_h(&vrow, w);
          for c in 0..w {
            u444[r * w + c] = uo[c] as u16;
            v444[r * w + c] = vo[c] as u16;
          }
        }
        (u444, v444)
      }

      /// The full-resolution U / V a bottom-sited (`v = 1`) high-bit 4:2:0 decode
      /// reconstructs (logical `u16`): the even luma row box-blends its predecessor
      /// chroma row (round-half-up), the odd row takes the current chroma row, each
      /// then horizontally centered — the RFC #238 S6f vertical fold.
      fn ref_full_chroma_bottom(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        let mut u444 = std::vec![0u16; w * h];
        let mut v444 = std::vec![0u16; w * h];
        let vblend = |plane: &[u16], cr: usize, prev: usize| -> Vec<u32> {
          (0..cw)
            .map(|c| (u32::from(plane[prev * cw + c]) + u32::from(plane[cr * cw + c]) + 1) >> 1)
            .collect()
        };
        for r in 0..h {
          let cr = r / 2;
          let (urow, vrow) = if r & 1 == 0 {
            let prev = cr.saturating_sub(1);
            (vblend(u420, cr, prev), vblend(v420, cr, prev))
          } else {
            (
              u420[cr * cw..cr * cw + cw]
                .iter()
                .map(|&x| u32::from(x))
                .collect(),
              v420[cr * cw..cr * cw + cw]
                .iter()
                .map(|&x| u32::from(x))
                .collect(),
            )
          };
          let uo = ref_upsample_center_h(&urow, w);
          let vo = ref_upsample_center_h(&vrow, w);
          for c in 0..w {
            u444[r * w + c] = uo[c] as u16;
            v444[r * w + c] = vo[c] as u16;
          }
        }
        (u444, v444)
      }

      fn frame<'a>(
        y: &'a [u16],
        u: &'a [u16],
        v: &'a [u16],
        a: &'a [u16],
      ) -> $LeFrame<'a> {
        $LeFrame::try_new(y, u, v, a, W, H, W, W / 2, W / 2, W).unwrap()
      }

      fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
        let (y, u, v, a) = ramp_planes();
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(loc)
          .with_simd(simd);
        $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
        rgb
      }

      fn convert_rgba(loc: ChromaLocation, simd: bool) -> Vec<u8> {
        let (y, u, v, a) = ramp_planes();
        let mut rgba = std::vec![0u8; (W * H * 4) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgba(&mut rgba)
          .unwrap()
          .with_chroma_location(loc)
          .with_simd(simd);
        $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
        rgba
      }

      // ---- default / co-sited path byte-identity + scratch discipline ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn default_and_cosited_sitings_are_byte_identical() {
        let baseline = convert_rgba(ChromaLocation::Unspecified, true);
        for loc in [
          ChromaLocation::Unspecified,
          ChromaLocation::Unknown(99),
          ChromaLocation::Left,
          ChromaLocation::TopLeft,
          ChromaLocation::BottomLeft,
        ] {
          assert_eq!(
            convert_rgba(loc, true),
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
        let (y, u, v, a) = ramp_planes();
        let mut rgba = std::vec![0u8; (W * H * 4) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgba(&mut rgba)
          .unwrap()
          .with_chroma_location(ChromaLocation::Left);
        $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(chroma_len, 0, "co-sited path must not grow the u16 chroma scratch");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_grows_chroma_scratch_to_full_width() {
        let (y, u, v, a) = ramp_planes();
        let mut rgba = std::vec![0u8; (W * H * 4) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgba(&mut rgba)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(chroma_len, 2 * W as usize, "centered path stages U+V at full width");
      }

      // ---- centered path correctness (upsample-then-4:4:4 oracle) ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgba_matches_upsample_then_444_with_real_alpha() {
        let (y, u, v, a) = ramp_planes();
        let (u444, v444) = ref_full_chroma(&u, &v);
        let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
        let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgba(&mut rgba_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          convert_rgba(ChromaLocation::Center, true),
          rgba_ref,
          "centered high-bit YUVA RGBA(u8) must equal upsample-then-4:4:4 (real alpha)"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgba_u16_matches_upsample_then_444_with_real_alpha() {
        let (y, u, v, a) = ramp_planes();
        let (u444, v444) = ref_full_chroma(&u, &v);

        let src = frame(&y, &u, &v, &a);
        let mut rgba16 = std::vec![0u16; (W * H * 4) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgba_u16(&mut rgba16)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

        let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
        let mut rgba16_ref = std::vec![0u16; (W * H * 4) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgba_u16(&mut rgba16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          rgba16, rgba16_ref,
          "centered high-bit YUVA RGBA(u16) must equal upsample-then-4:4:4 (real alpha)"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_and_rgb_u16_and_hsv_match_444_reference() {
        let (y, u, v, a) = ramp_planes();
        let (u444, v444) = ref_full_chroma(&u, &v);

        // RGB (u8, alpha-drop).
        {
          let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
          let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgb(&mut rgb_ref)
            .unwrap();
          $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(convert_rgb(ChromaLocation::Center, true), rgb_ref, "centered RGB");
        }

        // RGB (u16, alpha-drop).
        {
          let src = frame(&y, &u, &v, &a);
          let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb_u16(&mut rgb16)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
          let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgb_u16(&mut rgb16_ref)
            .unwrap();
          $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(rgb16, rgb16_ref, "centered RGB(u16)");
        }

        // HSV-direct.
        {
          let src = frame(&y, &u, &v, &a);
          let (mut h, mut s, mut vv) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_hsv(&mut h, &mut s, &mut vv)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
          let (mut hr, mut sr, mut vr) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_hsv(&mut hr, &mut sr, &mut vr)
            .unwrap();
          $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!((h, s, vv), (hr, sr, vr), "centered HSV");
        }
      }

      // ---- alpha preservation (native depth; siting never touches alpha) ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_alpha_u16_equals_source_and_default_alpha() {
        let (y, u, v, a) = ramp_planes();
        let decode = |loc: ChromaLocation| -> Vec<u16> {
          let mut rgba = std::vec![0u16; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba)
            .unwrap()
            .with_chroma_location(loc);
          $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
          rgba
        };
        let center = decode(ChromaLocation::Center);
        let default = decode(ChromaLocation::Left);
        // u16 RGBA carries alpha at native depth — equal to the source plane.
        for (i, &src_a) in a.iter().enumerate() {
          assert_eq!(center[i * 4 + 3], src_a, "centered native alpha at px {i}");
          assert_eq!(center[i * 4 + 3], default[i * 4 + 3], "alpha invariant to siting");
        }
        assert_ne!(center, default, "centered colour must differ from the default");
      }

      // ---- negative control + SIMD parity ----------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_routes_like_center_and_bottom_folds_vertically() {
        // Top shares the horizontal centered phase AND keeps the vertical pairing
        // co-sited (RFC #238 S6f folds only Bottom's `v = 1`), so Top == Center
        // byte-for-byte. Bottom additionally applies the vertical `v = 1` blend, so
        // on the vertically-varying fixture it diverges from Center — asserted at
        // NATIVE u16 depth: at high bit depths the fold's sub-LSB effect can round
        // away in the 8-bit RGBA, but it is exact in the native-depth output.
        let center = convert_rgba(ChromaLocation::Center, true);
        assert_eq!(convert_rgba(ChromaLocation::Top, true), center);
        let (y, u, v, a) = ramp_planes();
        let decode_u16 = |loc: ChromaLocation| -> Vec<u16> {
          let mut rgba = std::vec![0u16; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba)
            .unwrap()
            .with_chroma_location(loc);
          $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
          rgba
        };
        assert_eq!(
          decode_u16(ChromaLocation::Top),
          decode_u16(ChromaLocation::Center),
          "Top must route like Center at native depth (co-sited vertical)"
        );
        assert_ne!(
          decode_u16(ChromaLocation::Bottom),
          decode_u16(ChromaLocation::Center),
          "Bottom must fold vertically (diverge from Center) at native depth"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottom_rgba_matches_vfold_upsample_then_444_with_real_alpha() {
        // RFC #238 S6f: the `Bottom` (`v = 1`) high-bit direct decode reconstructs
        // chroma with the vertical box-blend + centered horizontal upsample, so its
        // RGBA equals a `Yuva444pN` decode on that v-fold-reconstructed chroma + the
        // SAME full-res source alpha — on BOTH the SIMD and scalar tiers (0-ULP).
        let (y, u, v, a) = ramp_planes();
        let (u444, v444) = ref_full_chroma_bottom(&u, &v);
        let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
        let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgba(&mut rgba_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          convert_rgba(ChromaLocation::Bottom, true),
          rgba_ref,
          "bottom high-bit YUVA RGBA (SIMD) must equal v-fold-upsample-then-4:4:4"
        );
        assert_eq!(
          convert_rgba(ChromaLocation::Bottom, false),
          rgba_ref,
          "bottom high-bit YUVA RGBA (scalar) must equal v-fold-upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_phase_differs_from_default() {
        assert_ne!(
          convert_rgba(ChromaLocation::Center, true),
          convert_rgba(ChromaLocation::Left, true),
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
          convert_rgba(ChromaLocation::Center, true),
          convert_rgba(ChromaLocation::Center, false),
          "centered RGBA must be bit-identical across the SIMD and scalar tiers"
        );
        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          convert_rgb(ChromaLocation::Center, false),
          "centered RGB must be bit-identical across the SIMD and scalar tiers"
        );
      }

      // ---- dirty-upper-bit sanitization (mask before the blend), LE + BE ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_upper_bits_le() {
        // A malformed-but-accepted low-packed frame with bits set ABOVE BITS
        // must decode (centered) identically to the masked clean frame: the
        // centered upsample masks each sample to BITS BEFORE the 1/4-3/4 blend.
        // (At BITS = 16 `upper` is 0, so this is the clean == clean identity.)
        let upper = !(MAXV as u16);
        let (y, u, v, a) = ramp_planes();
        let decode = |u: &[u16], v: &[u16]| -> Vec<u8> {
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&frame(&y, u, v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let u_dirty: Vec<u16> = u.iter().map(|&x| x | upper).collect();
        let v_dirty: Vec<u16> = v.iter().map(|&x| x | upper).collect();
        assert_eq!(
          decode(&u_dirty, &v_dirty),
          decode(&u, &v),
          "centered LE decode must sanitize dirty upper bits (mask before blend)"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_upper_bits_be() {
        // Same invariant on the big-endian wire path: the mask is applied in
        // the logical domain (after the endian load). Planes are BE-encoded and
        // decoded via the BE marker / frame / walker.
        let upper = !(MAXV as u16);
        let (y, u, v, a) = ramp_planes();
        let y_be: Vec<u16> = y.iter().map(|&x| x.to_be()).collect();
        let a_be: Vec<u16> = a.iter().map(|&x| x.to_be()).collect();
        let decode = |u_logical: &[u16], v_logical: &[u16]| -> Vec<u8> {
          let u_be: Vec<u16> = u_logical.iter().map(|&x| x.to_be()).collect();
          let v_be: Vec<u16> = v_logical.iter().map(|&x| x.to_be()).collect();
          let src =
            $BeFrame::try_new(&y_be, &u_be, &v_be, &a_be, W, H, W, W / 2, W / 2, W).unwrap();
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$MarkerBe>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker_be(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let u_dirty: Vec<u16> = u.iter().map(|&x| x | upper).collect();
        let v_dirty: Vec<u16> = v.iter().map(|&x| x | upper).collect();
        assert_eq!(
          decode(&u_dirty, &v_dirty),
          decode(&u, &v),
          "centered BE decode must sanitize dirty upper bits (mask before blend)"
        );
      }

      // ---- preflight-ordering atomicity (#302, cf. #180) -------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_alloc_failure_leaves_outputs_untouched() {
        use crate::resample::ResampleError;

        let (y, u, v, a) = ramp_planes();
        let src = frame(&y, &u, &v, &a);
        let mut luma = std::vec![0xABu8; (W * H) as usize];
        let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_luma(&mut luma)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);

        super::super::super::arm_chroma_full_alloc_failure();
        let err = $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
        drop(sink);

        assert!(
          matches!(err, MixedSinkerError::Resample(ResampleError::AllocationFailed(_))),
          "centered chroma-scratch refusal must surface as AllocationFailed, got {err:?}"
        );
        assert!(luma.iter().all(|&b| b == 0xAB), "luma untouched on alloc-failure");
        assert!(rgb.iter().all(|&b| b == 0xCD), "rgb untouched on alloc-failure");
      }

      // ---- ChromaDerivedNcl consistency ------------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_chroma_derived_ncl_uses_matrix_tag_fallback() {
        use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

        let (y, u, v, a) = ramp_planes();
        let spec = |loc: ChromaLocation| {
          ColorSpec::from_info(
            PixelFormat::Yuva420p,
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
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_color_spec(spec(loc));
          $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::ChromaDerivedNcl, &mut sink)
            .unwrap();
          rgb
        };
        let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc);
          $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt709, &mut sink).unwrap();
          rgb
        };
        assert_eq!(
          decode_cdn(ChromaLocation::Center),
          decode_bt709(ChromaLocation::Center),
          "centered high-bit YUVA ChromaDerivedNcl must use the BT.709 matrix-tag fallback"
        );
        assert_eq!(
          decode_cdn(ChromaLocation::Left),
          decode_bt709(ChromaLocation::Left),
          "default high-bit YUVA ChromaDerivedNcl must use the same BT.709 fallback"
        );
      }
    }
  };
}

hibit_yuva420_chroma_tests!(
  p9,
  9,
  Yuva420p9,
  Yuva420p9LeFrame,
  Yuva420p9BeFrame,
  yuva420p9_to,
  yuva420p9_to_endian,
  Yuva444p9,
  Yuva444p9Frame,
  yuva444p9_to,
  Yuva420p9<true>
);
hibit_yuva420_chroma_tests!(
  p10,
  10,
  Yuva420p10,
  Yuva420p10LeFrame,
  Yuva420p10BeFrame,
  yuva420p10_to,
  yuva420p10_to_endian,
  Yuva444p10,
  Yuva444p10Frame,
  yuva444p10_to,
  Yuva420p10<true>
);
hibit_yuva420_chroma_tests!(
  p12,
  12,
  Yuva420p12,
  Yuva420p12LeFrame,
  Yuva420p12BeFrame,
  yuva420p12_to,
  yuva420p12_to_endian,
  Yuva444p12,
  Yuva444p12Frame,
  yuva444p12_to,
  Yuva420p12<true>
);
hibit_yuva420_chroma_tests!(
  p16,
  16,
  Yuva420p16,
  Yuva420p16LeFrame,
  Yuva420p16BeFrame,
  yuva420p16_to,
  yuva420p16_to_endian,
  Yuva444p16,
  Yuva444p16Frame,
  yuva444p16_to,
  Yuva420p16<true>
);
