//! Chroma-siting-aware 4:2:2 upsampling for the planar **YUVA** family
//! (#302): `Yuva422p` (8-bit) + `Yuva422p9` / `Yuva422p10` / `Yuva422p12` /
//! `Yuva422p16` (high-bit, low-packed). The 4:2:2 sibling of
//! `chroma_siting_yuva` (4:2:0) and the alpha-bearing twin of
//! `chroma_siting_422` (no-alpha `Yuv422p`).
//!
//! YUVA 4:2:2 is planar 4:2:2 YUV (half-width, FULL-height U & V planes — one
//! chroma row per luma row) PLUS a **full-resolution** alpha plane that is never
//! subsampled — so the chroma siting reduces to its horizontal axis (identical
//! to the non-alpha `Yuv422p` twin), and the alpha plane passes through unchanged
//! on every path. These tests therefore assert, per format:
//!   * the default / co-sited / unspecified sitings stay byte-identical to the
//!     pre-#302 nearest-neighbor decode (the regression guard) + a negative
//!     control that the centered phase actually moves chroma;
//!   * the centered RGBA decode carries the **real source alpha** (not opaque
//!     `0xFF`), matching an independent "upsample-then-4:4:4-with-alpha" oracle;
//!   * the centered RGB / HSV (and the high-bit `u16` twins) match the
//!     upsample-then-4:4:4 alpha-drop reference;
//!   * **alpha preservation**: the centered RGBA's alpha channel equals BOTH the
//!     source alpha plane AND the default path's alpha (siting never touches
//!     alpha), under STRAIGHT and PREMULTIPLIED alpha (the identity decode is
//!     alpha-mode-agnostic — it never premultiplies);
//!   * SIMD == scalar on the centered path;
//!   * the no-output invariant — a no-output `process` reserves no scratch AND
//!     skips the `idx * w` offset math (the 32-bit overflow guard), per depth;
//!   * the preflight-ordering atomicity (a centered chroma-scratch alloc failure
//!     leaves luma AND colour untouched);
//!   * `ChromaDerivedNcl` consistency (YUVA is NOT primaries-wired, so BOTH the
//!     default and centered paths resolve it via the BT.709 matrix-tag fallback);
//!   * (high-bit) dirty-upper-bit sanitization (mask before the blend), LE + BE.

use super::*;
use crate::ChromaLocation;

const W: u32 = 16;
const H: u32 = 8;

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp, on logical `u32`
/// samples. Written separately from the production kernel so it is a real oracle;
/// shared by the 8-bit (`u8`) and high-bit (`u16`) suites.
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
// 8-bit Yuva422p
// ===========================================================================

mod p8 {
  use super::*;
  use crate::{
    frame::{Yuva422pFrame, Yuva444pFrame},
    source::{Yuva422p, Yuva422pRow, Yuva444p, yuva422p_to, yuva444p_to},
  };

  /// Flat mid-gray luma + per-column chroma ramp (distinct adjacent columns so
  /// the horizontal phase is observable; `+ r * 5` keeps chroma rows distinct so
  /// a vertical mistake would surface) + a per-pixel non-opaque alpha gradient
  /// (so the alpha-preservation assertions are non-vacuous). 4:2:2 chroma is
  /// half-width, FULL-height (`cw * h`).
  fn ramp_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let y = std::vec![128u8; w * h];
    let mut u = std::vec![0u8; cw * h];
    let mut v = std::vec![0u8; cw * h];
    let mut a = std::vec![0u8; w * h];
    for r in 0..h {
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

  /// The full-resolution U / V a centered 4:2:2 decode reconstructs: each luma
  /// row `r` takes chroma row `r` (4:2:2 has one chroma row per luma row — NO
  /// vertical subsampling) horizontally upsampled with the centered weights.
  fn ref_full_chroma(u422: &[u8], v422: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let mut u444 = std::vec![0u8; w * h];
    let mut v444 = std::vec![0u8; w * h];
    for r in 0..h {
      let cr = r;
      let urow: Vec<u32> = u422[cr * cw..cr * cw + cw]
        .iter()
        .map(|&x| x as u32)
        .collect();
      let vrow: Vec<u32> = v422[cr * cw..cr * cw + cw]
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

  fn frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], a: &'a [u8]) -> Yuva422pFrame<'a> {
    Yuva422pFrame::try_new(y, u, v, a, W, H, W, W / 2, W / 2, W).unwrap()
  }

  fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
    let (y, u, v, a) = ramp_planes();
    let mut rgb = std::vec![0u8; (W * H * 3) as usize];
    let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc)
      .with_simd(simd);
    yuva422p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
    rgb
  }

  fn convert_rgba(loc: ChromaLocation, simd: bool) -> Vec<u8> {
    let (y, u, v, a) = ramp_planes();
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(loc)
      .with_simd(simd);
    yuva422p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
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
        "siting {loc:?} must keep the byte-identical default YUVA 4:2:2 decode"
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
    let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Left);
    yuva422p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
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
    let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuva422p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
    let chroma_len = sink.chroma_full.len();
    drop(sink);
    assert_eq!(
      chroma_len,
      2 * W as usize,
      "centered path stages U+V at full width"
    );
  }

  // ---- no-output invariant (no scratch, no row-offset overflow) --------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn no_output_row_does_not_allocate_chroma_scratch() {
    // A centered-siting sink with NO outputs attached honours the repo-wide
    // no-output invariant: every `process` returns before the preflight, so the
    // centered chroma scratch is NEVER reserved.
    let (y, u, v, a) = ramp_planes();
    let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
      .with_chroma_location(ChromaLocation::Center);
    yuva422p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
    let chroma_len = sink.chroma_full.len();
    drop(sink);
    assert_eq!(
      chroma_len, 0,
      "a no-output centered row must not reserve the chroma scratch"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "constructs an absurd geometry; the no-op contract is the point, not Miri"
  )]
  fn no_output_row_large_geometry_does_not_overflow() {
    // The no-output guard must run BEFORE the `idx * w` single-plane offset
    // arithmetic. A no-output `process` call never ran an attach-time `w x h x 1`
    // validation, so on a 32-bit target (`usize == u32`) an absurd geometry where
    // `idx * w` exceeds `u32::MAX` would overflow that offset and panic under
    // overflow checks. With no outputs attached, `process` returns `Ok(())`
    // having done NO row math and NO allocation.
    //
    // w = 4, idx = 2^30 -> idx * w = 2^32 = u32::MAX + 1 (overflows u32).
    let w: usize = 4;
    let idx: usize = 1 << 30;
    let h: usize = idx + 1; // idx < height so the row-index check passes
    assert!(
      (idx as u64) * (w as u64) > u32::MAX as u64,
      "test geometry must exceed u32::MAX to exercise the 32-bit offset overflow"
    );

    let y = std::vec![128u8; w];
    let c = std::vec![128u8; w / 2];
    let a = std::vec![128u8; w];
    let mut sink = MixedSinker::<Yuva422p>::new(w, h).with_chroma_location(ChromaLocation::Center);
    // No outputs attached: the guard returns before `idx * w` (no overflow panic)
    // and before the centered preflight (no allocation).
    let row = Yuva422pRow::new(&y, &c, &c, &a, idx, ColorMatrix::Bt601, false);
    crate::PixelSink::process(&mut sink, row).unwrap();
    let chroma_len = sink.chroma_full.len();
    drop(sink);
    assert_eq!(
      chroma_len, 0,
      "a no-output large-geometry row must allocate nothing"
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
      "centered YUVA 4:2:2 RGBA must equal upsample-then-4:4:4 (real source alpha)"
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
      "centered YUVA 4:2:2 RGB (alpha-drop) must equal upsample-then-4:4:4"
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
    let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut vv)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuva422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

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
      "centered YUVA 4:2:2 HSV must equal upsample-then-4:4:4"
    );
  }

  // ---- alpha preservation (siting never touches alpha; mode-agnostic) ----

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

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn centered_premultiplied_alpha_passes_through() {
    // The identity centered decode is alpha-mode-agnostic — it NEVER
    // premultiplies (premultiplied binning is a resample-path concern). So a
    // Premultiplied-mode sink emits the SAME centered RGBA as a Straight-mode
    // sink, with the alpha channel equal to the source plane bit-for-bit.
    use crate::sinker::AlphaMode;
    let (_, _, _, a) = ramp_planes();
    let decode = |mode: AlphaMode| -> Vec<u8> {
      let (y, u, v, a) = ramp_planes();
      let mut rgba = std::vec![0u8; (W * H * 4) as usize];
      let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_alpha_mode(mode)
        .with_chroma_location(ChromaLocation::Center);
      yuva422p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
      rgba
    };
    let premult = decode(AlphaMode::Premultiplied);
    let straight = decode(AlphaMode::Straight);
    for (i, &src_a) in a.iter().enumerate() {
      assert_eq!(
        premult[i * 4 + 3],
        src_a,
        "premultiplied-mode centered alpha at px {i} must pass the source plane through"
      );
    }
    assert_eq!(
      premult, straight,
      "the identity centered decode must be alpha-mode-agnostic (no premultiply)"
    );
  }

  // ---- negative control + SIMD parity ----------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn top_and_bottom_route_like_center_horizontally() {
    let center = convert_rgba(ChromaLocation::Center, true);
    assert_eq!(convert_rgba(ChromaLocation::Top, true), center);
    assert_eq!(convert_rgba(ChromaLocation::Bottom, true), center);
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
    let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
      .with_luma(&mut luma)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);

    super::super::super::arm_chroma_full_alloc_failure();
    let err = yuva422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
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
    // fused 4:2:2 kernel AND the centered 4:4:4 kernel — resolve ChromaDerivedNcl
    // via the shared BT.709 matrix-tag fallback, IGNORING the ColorSpec
    // primaries, so default and centered stay internally consistent (the centered
    // phase shift is the ONLY difference between them).
    use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

    let (y, u, v, a) = ramp_planes();
    let spec = |loc: ChromaLocation| {
      ColorSpec::from_info(
        PixelFormat::Yuva422p,
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
      let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_color_spec(spec(loc));
      yuva422p_to(
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
      let mut sink = MixedSinker::<Yuva422p>::new(W as usize, H as usize)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_chroma_location(loc);
      yuva422p_to(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt709, &mut sink).unwrap();
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

  // ---- mid-frame siting-flip rejection (identity path freeze) ---------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn direct_path_mid_frame_siting_flip_is_rejected() {
    // The identity (no-resample) 4:2:2 YUVA decode freezes the horizontal siting on
    // its first output-bearing row. Flipping co-sited ⇆ centered mid-frame must
    // reject the next in-sequence row with `ChromaSitingChanged` (leaving the
    // centered chroma scratch untouched); flipping back and retrying then matches a
    // clean single-phase decode byte-for-byte.
    let (y, u, v, a) = ramp_planes();
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let row_at = |r: usize| {
      Yuva422pRow::new(
        &y[r * w..(r + 1) * w],
        &u[r * cw..(r + 1) * cw],
        &v[r * cw..(r + 1) * cw],
        &a[r * w..(r + 1) * w],
        r,
        ColorMatrix::Bt601,
        false,
      )
    };
    for (loc1, loc2) in [
      (ChromaLocation::Left, ChromaLocation::Center),
      (ChromaLocation::Center, ChromaLocation::Left),
    ] {
      let want = convert_rgb(loc1, true);
      let mut rgb = std::vec![0u8; w * h * 3];
      let mut sink = MixedSinker::<Yuva422p>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_chroma_location(loc1)
        .with_simd(true);
      crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
      crate::PixelSink::process(&mut sink, row_at(0)).unwrap();
      let scratch_len = sink.chroma_full.len();

      sink.set_chroma_location(loc2);
      let err = crate::PixelSink::process(&mut sink, row_at(1)).unwrap_err();
      assert!(
        matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
        "direct path {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
      );
      assert_eq!(
        sink.chroma_full.len(),
        scratch_len,
        "{loc1:?}->{loc2:?}: a rejected flip must not grow the chroma scratch"
      );

      sink.set_chroma_location(loc1);
      for r in 1..h {
        crate::PixelSink::process(&mut sink, row_at(r)).unwrap();
      }
      drop(sink);
      assert_eq!(
        rgb, want,
        "{loc1:?}: retry after a rejected flip must match a clean in-order decode"
      );
    }
  }
}

// ===========================================================================
// High-bit Yuva422p9 / 10 / 12 / 16 (low-packed)
// ===========================================================================

// Identical bar the bit depth, format marker, frame type, and walker — so
// generate the suite once per depth. The macro instantiates each with its
// little-endian marker (a sample's wire `u16` equals its logical value on the LE
// test host); the references compute in that logical domain. The endianness
// re-encode is exercised by the dirty-bit BE test (via the BE frame / walker).
macro_rules! hibit_yuva422_chroma_tests {
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
    $MarkerBe:ty,
    $Row:ident
  ) => {
    mod $mod {
      use super::*;
      use crate::{
        frame::{$BeFrame, $LeFrame, $RefFrame},
        source::{$Marker, $Ref, $Row, $ref_walker, $walker, $walker_be},
      };

      const MAXV: u32 = (1u32 << $bits) - 1;

      /// Flat mid-gray luma + per-column chroma ramp + a varying (non-opaque)
      /// alpha plane, all low-packed at `$bits`. 4:2:2 chroma is half-width,
      /// FULL-height (`cw * h`).
      fn ramp_planes() -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        let step = (MAXV / 16).max(1);
        let y = std::vec![(MAXV / 2) as u16; w * h];
        let mut u = std::vec![0u16; cw * h];
        let mut v = std::vec![0u16; cw * h];
        let mut a = std::vec![0u16; w * h];
        for r in 0..h {
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

      /// The full-resolution U / V a centered high-bit 4:2:2 decode reconstructs
      /// (logical `u16`): chroma row `r` per luma row `r`.
      fn ref_full_chroma(u422: &[u16], v422: &[u16]) -> (Vec<u16>, Vec<u16>) {
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        let mut u444 = std::vec![0u16; w * h];
        let mut v444 = std::vec![0u16; w * h];
        for r in 0..h {
          let cr = r;
          let urow: Vec<u32> =
            u422[cr * cw..cr * cw + cw].iter().map(|&x| x as u32).collect();
          let vrow: Vec<u32> =
            v422[cr * cw..cr * cw + cw].iter().map(|&x| x as u32).collect();
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
        // returns `Ok(())` having done NO row math and NO allocation.
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
        let a = std::vec![(MAXV / 2) as u16; w];
        let mut sink =
          MixedSinker::<$Marker>::new(w, h).with_chroma_location(ChromaLocation::Center);
        // No outputs attached: the guard returns before `idx * w` (no overflow
        // panic) and before the centered preflight (no allocation).
        let row = $Row::new(&y, &c, &c, &a, idx, ColorMatrix::Bt601, false);
        crate::PixelSink::process(&mut sink, row).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(
          chroma_len, 0,
          "a no-output large-geometry high-bit row must allocate nothing"
        );
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
          "centered high-bit YUVA 4:2:2 RGBA(u8) must equal upsample-then-4:4:4 (real alpha)"
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
          "centered high-bit YUVA 4:2:2 RGBA(u16) must equal upsample-then-4:4:4 (real alpha)"
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

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_premultiplied_alpha_passes_through() {
        // The identity centered decode never premultiplies — Premultiplied mode
        // emits the SAME centered RGBA(u16) as Straight, alpha = source plane.
        use crate::sinker::AlphaMode;
        let (_, _, _, a) = ramp_planes();
        let decode = |mode: AlphaMode| -> Vec<u16> {
          let (y, u, v, a) = ramp_planes();
          let mut rgba = std::vec![0u16; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba)
            .unwrap()
            .with_alpha_mode(mode)
            .with_chroma_location(ChromaLocation::Center);
          $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
          rgba
        };
        let premult = decode(AlphaMode::Premultiplied);
        let straight = decode(AlphaMode::Straight);
        for (i, &src_a) in a.iter().enumerate() {
          assert_eq!(premult[i * 4 + 3], src_a, "premult-mode centered alpha at px {i}");
        }
        assert_eq!(
          premult, straight,
          "the identity centered decode must be alpha-mode-agnostic (no premultiply)"
        );
      }

      // ---- negative control + SIMD parity ----------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_and_bottom_route_like_center_horizontally() {
        let center = convert_rgba(ChromaLocation::Center, true);
        assert_eq!(convert_rgba(ChromaLocation::Top, true), center);
        assert_eq!(convert_rgba(ChromaLocation::Bottom, true), center);
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
        // A malformed-but-accepted low-packed frame with bits set ABOVE BITS must
        // decode (centered) identically to the masked clean frame: the centered
        // upsample masks each sample to BITS BEFORE the 1/4-3/4 blend. (At
        // BITS = 16 `upper` is 0, so this is the clean == clean identity.)
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
        // Same invariant on the big-endian wire path: the mask is applied in the
        // logical domain (after the endian load). Planes are BE-encoded and
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

      // ---- dirty-upper-bit sanitization of the LUMA outputs, LE + BE --------
      //
      // The chroma dirty-bit tests above dirty U / V and read RGB; these dirty
      // the Y plane and read the LUMA outputs. A malformed-but-accepted
      // low-packed frame with Y bits set ABOVE BITS must produce native
      // `luma_u16` AND narrowed 8-bit `luma` equal to the in-range (masked) Y —
      // the luma path masks each decoded Y to `(1 << BITS) - 1` BEFORE the native
      // write and the `>> (BITS - 8)` downshift, NOT the raw / wrapped value
      // (#302 follow-up; mirrors the high-bit `Yuva420p` sibling). `ramp_planes`'
      // Y is flat in-range mid-gray, so the masked dirty output equals the clean
      // source Y exactly. (At BITS = 16 `upper` is 0, so this is the clean
      // identity; the sub-16-bit depths exercise the real sanitization.)

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn sanitizes_dirty_y_luma_le() {
        let upper = !(MAXV as u16);
        let (y, u, v, a) = ramp_planes();
        let decode = |yp: &[u16]| -> (Vec<u8>, Vec<u16>) {
          let mut luma = std::vec![0u8; (W * H) as usize];
          let mut luma16 = std::vec![0u16; (W * H) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut luma16)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&frame(yp, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
          (luma, luma16)
        };
        let y_dirty: Vec<u16> = y.iter().map(|&x| x | upper).collect();
        let (ld, l16d) = decode(&y_dirty);
        let (lc, l16c) = decode(&y);
        // The in-range Y is the oracle: native luma_u16 == source Y (masked),
        // never the raw dirty value (which without the mask would be `y | upper`).
        assert_eq!(l16d, y, "native luma_u16 must equal the in-range (masked) Y");
        assert_eq!(l16d, l16c, "dirty-Y luma_u16 must equal the clean-Y luma_u16");
        assert_eq!(ld, lc, "dirty-Y 8-bit luma must equal the clean-Y 8-bit luma");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn sanitizes_dirty_y_luma_be() {
        // Same invariant on the big-endian wire path: the mask is applied in the
        // logical domain (after the endian load), so the host-native luma_u16
        // equals the in-range logical Y.
        let upper = !(MAXV as u16);
        let (y, u, v, a) = ramp_planes();
        let u_be: Vec<u16> = u.iter().map(|&x| x.to_be()).collect();
        let v_be: Vec<u16> = v.iter().map(|&x| x.to_be()).collect();
        let a_be: Vec<u16> = a.iter().map(|&x| x.to_be()).collect();
        let decode = |y_logical: &[u16]| -> (Vec<u8>, Vec<u16>) {
          let y_be: Vec<u16> = y_logical.iter().map(|&x| x.to_be()).collect();
          let src =
            $BeFrame::try_new(&y_be, &u_be, &v_be, &a_be, W, H, W, W / 2, W / 2, W).unwrap();
          let mut luma = std::vec![0u8; (W * H) as usize];
          let mut luma16 = std::vec![0u16; (W * H) as usize];
          let mut sink = MixedSinker::<$MarkerBe>::new(W as usize, H as usize)
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut luma16)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker_be(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
          (luma, luma16)
        };
        let y_dirty: Vec<u16> = y.iter().map(|&x| x | upper).collect();
        let (ld, l16d) = decode(&y_dirty);
        let (lc, l16c) = decode(&y);
        assert_eq!(l16d, y, "BE native luma_u16 must equal the in-range (masked) Y");
        assert_eq!(l16d, l16c, "BE dirty-Y luma_u16 must equal the clean-Y luma_u16");
        assert_eq!(ld, lc, "BE dirty-Y 8-bit luma must equal the clean-Y 8-bit luma");
      }

      // ---- RESAMPLE-path dirty-Y luma sanitization, area + filter, LE + BE ---
      //
      // The R1 tests above cover the IDENTITY luma path; these cover the
      // non-identity (resample) branch. Under an area / filter downscale plan
      // with with_luma + with_luma_u16 attached, the native-Y luma stream feeds
      // `deinterleave_y_high_bit_masked::<BITS, BE>`, so dirty upper bits in a
      // malformed-but-accepted Y are masked BEFORE binning / filtering — NOT
      // binned-then-clamped to native_max. The resampled luma must therefore
      // reflect the in-range (masked) Y, identical to the clean frame. With the
      // flat in-range Y = MAXV/2, the clean resampled luma_u16 is uniformly
      // MAXV/2; the unmasked path would instead clamp the dirty value to MAXV, so
      // these are non-vacuous (fail before the R2 fix). (At BITS = 16 `upper` is
      // 0 → clean identity; the sub-16-bit depths exercise the real path.)

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resample_area_sanitizes_dirty_y_luma_le() {
        use crate::resample::AreaResampler;
        const OW: usize = (W / 2) as usize;
        const OH: usize = (H / 2) as usize;
        let upper = !(MAXV as u16);
        let (y, u, v, a) = ramp_planes();
        let decode = |yp: &[u16]| -> (Vec<u8>, Vec<u16>) {
          let mut luma = std::vec![0u8; OW * OH];
          let mut luma16 = std::vec![0u16; OW * OH];
          {
            let mut sink = MixedSinker::<$Marker, AreaResampler>::with_resampler(
              W as usize,
              H as usize,
              AreaResampler::to(OW, OH),
            )
            .unwrap()
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut luma16)
            .unwrap();
            $walker(&frame(yp, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
          }
          (luma, luma16)
        };
        let y_dirty: Vec<u16> = y.iter().map(|&x| x | upper).collect();
        let (ld, l16d) = decode(&y_dirty);
        let (lc, l16c) = decode(&y);
        assert!(
          l16c.iter().all(|&p| p == (MAXV / 2) as u16),
          "clean area-resample luma_u16 sanity (flat in-range Y stays flat)"
        );
        assert_eq!(l16d, l16c, "area-resample dirty-Y luma_u16 must equal clean (masked before binning)");
        assert_eq!(ld, lc, "area-resample dirty-Y 8-bit luma must equal clean");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resample_area_sanitizes_dirty_y_luma_be() {
        use crate::resample::AreaResampler;
        const OW: usize = (W / 2) as usize;
        const OH: usize = (H / 2) as usize;
        let upper = !(MAXV as u16);
        let (y, u, v, a) = ramp_planes();
        let u_be: Vec<u16> = u.iter().map(|&x| x.to_be()).collect();
        let v_be: Vec<u16> = v.iter().map(|&x| x.to_be()).collect();
        let a_be: Vec<u16> = a.iter().map(|&x| x.to_be()).collect();
        let decode = |y_logical: &[u16]| -> (Vec<u8>, Vec<u16>) {
          let y_be: Vec<u16> = y_logical.iter().map(|&x| x.to_be()).collect();
          let mut luma = std::vec![0u8; OW * OH];
          let mut luma16 = std::vec![0u16; OW * OH];
          {
            let src =
              $BeFrame::try_new(&y_be, &u_be, &v_be, &a_be, W, H, W, W / 2, W / 2, W).unwrap();
            let mut sink = MixedSinker::<$MarkerBe, AreaResampler>::with_resampler(
              W as usize,
              H as usize,
              AreaResampler::to(OW, OH),
            )
            .unwrap()
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut luma16)
            .unwrap();
            $walker_be(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
          }
          (luma, luma16)
        };
        let y_dirty: Vec<u16> = y.iter().map(|&x| x | upper).collect();
        let (ld, l16d) = decode(&y_dirty);
        let (lc, l16c) = decode(&y);
        assert!(
          l16c.iter().all(|&p| p == (MAXV / 2) as u16),
          "clean BE area-resample luma_u16 sanity"
        );
        assert_eq!(l16d, l16c, "BE area-resample dirty-Y luma_u16 must equal clean");
        assert_eq!(ld, lc, "BE area-resample dirty-Y 8-bit luma must equal clean");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resample_filter_sanitizes_dirty_y_luma_le() {
        use crate::resample::{FilteredResampler, Triangle};
        const OW: usize = (W / 2) as usize;
        const OH: usize = (H / 2) as usize;
        let upper = !(MAXV as u16);
        let (y, u, v, a) = ramp_planes();
        let decode = |yp: &[u16]| -> (Vec<u8>, Vec<u16>) {
          let mut luma = std::vec![0u8; OW * OH];
          let mut luma16 = std::vec![0u16; OW * OH];
          {
            let mut sink =
              MixedSinker::<$Marker, FilteredResampler<Triangle>>::with_resampler(
                W as usize,
                H as usize,
                FilteredResampler::new(OW, OH, Triangle),
              )
              .unwrap()
              .with_luma(&mut luma)
              .unwrap()
              .with_luma_u16(&mut luma16)
              .unwrap();
            $walker(&frame(yp, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
          }
          (luma, luma16)
        };
        let y_dirty: Vec<u16> = y.iter().map(|&x| x | upper).collect();
        let (ld, l16d) = decode(&y_dirty);
        let (lc, l16c) = decode(&y);
        // A linear filter of the flat in-range Y stays flat MAXV/2; the unmasked
        // path would filter the dirty value and clamp it to MAXV instead.
        assert!(
          l16c.iter().all(|&p| p == (MAXV / 2) as u16),
          "clean filter-resample luma_u16 sanity (flat in-range Y stays flat)"
        );
        assert_eq!(l16d, l16c, "filter-resample dirty-Y luma_u16 must equal clean (masked before filter)");
        assert_eq!(ld, lc, "filter-resample dirty-Y 8-bit luma must equal clean");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resample_filter_sanitizes_dirty_y_luma_be() {
        use crate::resample::{FilteredResampler, Triangle};
        const OW: usize = (W / 2) as usize;
        const OH: usize = (H / 2) as usize;
        let upper = !(MAXV as u16);
        let (y, u, v, a) = ramp_planes();
        let u_be: Vec<u16> = u.iter().map(|&x| x.to_be()).collect();
        let v_be: Vec<u16> = v.iter().map(|&x| x.to_be()).collect();
        let a_be: Vec<u16> = a.iter().map(|&x| x.to_be()).collect();
        let decode = |y_logical: &[u16]| -> (Vec<u8>, Vec<u16>) {
          let y_be: Vec<u16> = y_logical.iter().map(|&x| x.to_be()).collect();
          let mut luma = std::vec![0u8; OW * OH];
          let mut luma16 = std::vec![0u16; OW * OH];
          {
            let src =
              $BeFrame::try_new(&y_be, &u_be, &v_be, &a_be, W, H, W, W / 2, W / 2, W).unwrap();
            let mut sink =
              MixedSinker::<$MarkerBe, FilteredResampler<Triangle>>::with_resampler(
                W as usize,
                H as usize,
                FilteredResampler::new(OW, OH, Triangle),
              )
              .unwrap()
              .with_luma(&mut luma)
              .unwrap()
              .with_luma_u16(&mut luma16)
              .unwrap();
            $walker_be(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
          }
          (luma, luma16)
        };
        let y_dirty: Vec<u16> = y.iter().map(|&x| x | upper).collect();
        let (ld, l16d) = decode(&y_dirty);
        let (lc, l16c) = decode(&y);
        assert!(
          l16c.iter().all(|&p| p == (MAXV / 2) as u16),
          "clean BE filter-resample luma_u16 sanity"
        );
        assert_eq!(l16d, l16c, "BE filter-resample dirty-Y luma_u16 must equal clean");
        assert_eq!(ld, lc, "BE filter-resample dirty-Y 8-bit luma must equal clean");
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
            PixelFormat::Yuva422p,
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

hibit_yuva422_chroma_tests!(
  p9,
  9,
  Yuva422p9,
  Yuva422p9LeFrame,
  Yuva422p9BeFrame,
  yuva422p9_to,
  yuva422p9_to_endian,
  Yuva444p9,
  Yuva444p9Frame,
  yuva444p9_to,
  Yuva422p9<true>,
  Yuva422p9Row
);
hibit_yuva422_chroma_tests!(
  p10,
  10,
  Yuva422p10,
  Yuva422p10LeFrame,
  Yuva422p10BeFrame,
  yuva422p10_to,
  yuva422p10_to_endian,
  Yuva444p10,
  Yuva444p10Frame,
  yuva444p10_to,
  Yuva422p10<true>,
  Yuva422p10Row
);
hibit_yuva422_chroma_tests!(
  p12,
  12,
  Yuva422p12,
  Yuva422p12LeFrame,
  Yuva422p12BeFrame,
  yuva422p12_to,
  yuva422p12_to_endian,
  Yuva444p12,
  Yuva444p12Frame,
  yuva444p12_to,
  Yuva422p12<true>,
  Yuva422p12Row
);
hibit_yuva422_chroma_tests!(
  p16,
  16,
  Yuva422p16,
  Yuva422p16LeFrame,
  Yuva422p16BeFrame,
  yuva422p16_to,
  yuva422p16_to_endian,
  Yuva444p16,
  Yuva444p16Frame,
  yuva444p16_to,
  Yuva422p16<true>,
  Yuva422p16Row
);
