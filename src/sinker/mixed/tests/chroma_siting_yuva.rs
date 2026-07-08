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

  /// The full-resolution U / V a **`BottomLeft`** (`h = 0`, `v = 1`) 4:2:0 decode
  /// reconstructs: [`ref_full_chroma_bottom`]'s even-row vertical box blend, but
  /// fed to the CO-SITED horizontal 2× replicate instead of the centered kernel.
  fn ref_full_chroma_bottomleft(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let mut u444 = std::vec![0u8; w * h];
    let mut v444 = std::vec![0u8; w * h];
    let vblend = |plane: &[u8], cr: usize, prev: usize| -> Vec<u8> {
      (0..cw)
        .map(|c| ((u32::from(plane[prev * cw + c]) + u32::from(plane[cr * cw + c]) + 1) >> 1) as u8)
        .collect()
    };
    for r in 0..h {
      let cr = r / 2;
      let (uh, vh) = if r & 1 == 0 {
        let prev = cr.saturating_sub(1);
        (vblend(u420, cr, prev), vblend(v420, cr, prev))
      } else {
        (
          u420[cr * cw..cr * cw + cw].to_vec(),
          v420[cr * cw..cr * cw + cw].to_vec(),
        )
      };
      for j in 0..cw {
        u444[r * w + 2 * j] = uh[j];
        u444[r * w + 2 * j + 1] = uh[j];
        v444[r * w + 2 * j] = vh[j];
        v444[r * w + 2 * j + 1] = vh[j];
      }
    }
    (u444, v444)
  }

  /// The full-resolution U / V a **`Top`** (`v = 0`, FORWARD fold) 4:2:0 decode
  /// reconstructs (RFC #238) — the vertical MIRROR of [`ref_full_chroma_bottom`]:
  /// the EVEN luma row `2i` is co-sited with chroma row `i`; the ODD luma row
  /// `2i + 1` vertically box-blends chroma rows `{i, i + 1}` (round-half-up, bottom-
  /// edge clamp) BEFORE the centered horizontal upsample. Generalised over `h` so
  /// the odd-height two-row-flush test can reuse it.
  fn ref_full_chroma_top_geom(u420: &[u8], v420: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let ch = h.div_ceil(2);
    let mut u444 = std::vec![0u8; w * h];
    let mut v444 = std::vec![0u8; w * h];
    let vblend = |plane: &[u8], cr: usize, next: usize| -> Vec<u32> {
      (0..cw)
        .map(|c| (u32::from(plane[cr * cw + c]) + u32::from(plane[next * cw + c]) + 1) >> 1)
        .collect()
    };
    for r in 0..h {
      let cr = r / 2;
      let (urow, vrow) = if r & 1 == 1 {
        let next = (cr + 1).min(ch - 1);
        (vblend(u420, cr, next), vblend(v420, cr, next))
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

  fn ref_full_chroma_top(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
    ref_full_chroma_top_geom(u420, v420, W as usize, H as usize)
  }

  /// The full-resolution U / V a **`TopLeft`** (`h = 0`, `v = 0`) 4:2:0 decode
  /// reconstructs: [`ref_full_chroma_top`]'s odd-row forward vertical box blend fed
  /// to the CO-SITED horizontal 2× replicate instead of the centered kernel.
  fn ref_full_chroma_topleft(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let ch = h.div_ceil(2);
    let mut u444 = std::vec![0u8; w * h];
    let mut v444 = std::vec![0u8; w * h];
    let vblend = |plane: &[u8], cr: usize, next: usize| -> Vec<u8> {
      (0..cw)
        .map(|c| ((u32::from(plane[cr * cw + c]) + u32::from(plane[next * cw + c]) + 1) >> 1) as u8)
        .collect()
    };
    for r in 0..h {
      let cr = r / 2;
      let (uh, vh) = if r & 1 == 1 {
        let next = (cr + 1).min(ch - 1);
        (vblend(u420, cr, next), vblend(v420, cr, next))
      } else {
        (
          u420[cr * cw..cr * cw + cw].to_vec(),
          v420[cr * cw..cr * cw + cw].to_vec(),
        )
      };
      for j in 0..cw {
        u444[r * w + 2 * j] = uh[j];
        u444[r * w + 2 * j + 1] = uh[j];
        v444[r * w + 2 * j] = vh[j];
        v444[r * w + 2 * j + 1] = vh[j];
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
    // `BottomLeft` / `TopLeft` are EXCLUDED: both are co-sited horizontally but
    // vertically sited (`BottomLeft` v = 1, `TopLeft` v = 0 forward fold), so each
    // folds a vertical blend on its odd/even rows (their own tests below).
    for loc in [
      ChromaLocation::Unspecified,
      ChromaLocation::Unknown(99),
      ChromaLocation::Left,
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
  fn top_bottom_center_are_all_distinct() {
    // RFC #238 Top: `Top` folds the FORWARD (`v = 0`) vertical triangle, `Bottom`
    // the BACKWARD (`v = 1`) one, `Center` neither — so on the vertically-varying
    // fixture all three diverge from one another.
    let center = convert_rgba(ChromaLocation::Center, true);
    let top = convert_rgba(ChromaLocation::Top, true);
    let bottom = convert_rgba(ChromaLocation::Bottom, true);
    assert_ne!(
      top, center,
      "Top (v=0 forward fold) must differ from Center"
    );
    assert_ne!(bottom, center, "Bottom (v=1 fold) must differ from Center");
    assert_ne!(top, bottom, "Top and Bottom fold in opposite directions");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn top_rgba_matches_forward_vfold_upsample_then_444_with_real_alpha() {
    // RFC #238 Top (`v = 0`): the forward one-row-delay decode reconstructs chroma
    // with the ODD-row forward vertical box-blend + centered horizontal upsample,
    // so its RGBA equals a 4:4:4 YUVA decode on that reconstruction + the SAME
    // full-res source alpha — on BOTH the SIMD and scalar tiers (the alpha is held
    // with each deferred odd row and emitted in order).
    let (y, u, v, a) = ramp_planes();
    let (u444, v444) = ref_full_chroma_top(&u, &v);
    let ref_src = Yuva444pFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuva444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuva444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      convert_rgba(ChromaLocation::Top, true),
      rgba_ref,
      "top YUVA RGBA (SIMD) must equal forward-vfold-upsample-then-4:4:4 (real source alpha)"
    );
    assert_eq!(
      convert_rgba(ChromaLocation::Top, false),
      rgba_ref,
      "top YUVA RGBA (scalar) must equal forward-vfold-upsample-then-4:4:4"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn topleft_rgba_matches_cosited_forward_vfold_then_444_with_real_alpha() {
    // TopLeft (h=0 co-sited + v=0 forward fold): RGBA equals a 4:4:4 YUVA decode
    // over the co-sited-replicate + forward-vertical-blend reconstruction with the
    // SAME source α, on both tiers; and it genuinely differs from Top (co-sited h)
    // and from the co-sited default (the v=0 fold).
    let (y, u, v, a) = ramp_planes();
    let (u444, v444) = ref_full_chroma_topleft(&u, &v);
    let ref_src = Yuva444pFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuva444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuva444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      convert_rgba(ChromaLocation::TopLeft, true),
      rgba_ref,
      "top-left YUVA RGBA (SIMD) must equal cosited-forward-vfold-then-4:4:4 (real source alpha)"
    );
    assert_eq!(
      convert_rgba(ChromaLocation::TopLeft, false),
      rgba_ref,
      "top-left YUVA RGBA (scalar) must equal cosited-forward-vfold-then-4:4:4"
    );
    assert_ne!(
      convert_rgba(ChromaLocation::TopLeft, true),
      convert_rgba(ChromaLocation::Top, true),
      "TopLeft (h=0) must differ from Top (h=0.5)"
    );
    assert_ne!(
      convert_rgba(ChromaLocation::TopLeft, true),
      convert_rgba(ChromaLocation::Left, true),
      "TopLeft (v=0 forward fold) must differ from the co-sited default"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn top_rgba_alpha_channel_equals_source_plane() {
    // The full-resolution alpha is held with each deferred odd row and emitted in
    // order, so the Top RGBA alpha channel is the source plane verbatim (never
    // opaqued, never mis-ordered by the forward delay).
    let (_, _, _, a) = ramp_planes();
    let top = convert_rgba(ChromaLocation::Top, true);
    for (i, &src_a) in a.iter().enumerate() {
      assert_eq!(
        top[i * 4 + 3],
        src_a,
        "top alpha at px {i} must equal the source plane"
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn top_odd_height_two_row_final_flush_matches_reference() {
    // ODD height: the frame ends on an EVEN row that must flush its held odd
    // predecessor AND emit itself (the two-row FINAL flush). Compare the full
    // decode to the forward-fold oracle at that geometry.
    const OW: u32 = 8;
    const OH: u32 = 7;
    let w = OW as usize;
    let h = OH as usize;
    let cw = w / 2;
    let ch = h.div_ceil(2);
    let y = std::vec![128u8; w * h];
    let mut u = std::vec![0u8; cw * ch];
    let mut v = std::vec![0u8; cw * ch];
    let mut a = std::vec![0u8; w * h];
    for r in 0..ch {
      for c in 0..cw {
        u[r * cw + c] = (16 * c + 16 + r * 7).min(255) as u8;
        v[r * cw + c] = (255u32.saturating_sub(16 * c as u32 + r as u32 * 9)).max(16) as u8;
      }
    }
    for (i, e) in a.iter_mut().enumerate() {
      *e = (i % 251 + 3) as u8;
    }
    let (u444, v444) = ref_full_chroma_top_geom(&u, &v, w, h);
    let ref_src = Yuva444pFrame::try_new(&y, &u444, &v444, &a, OW, OH, OW, OW, OW, OW).unwrap();
    let mut rgba_ref = std::vec![0u8; (OW * OH * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuva444p>::new(w, h)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuva444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    let src = Yuva420pFrame::try_new(&y, &u, &v, &a, OW, OH, OW, OW / 2, OW / 2, OW).unwrap();
    for simd in [true, false] {
      let mut rgba = std::vec![0u8; (OW * OH * 4) as usize];
      let mut sink = MixedSinker::<Yuva420p>::new(w, h)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_chroma_location(ChromaLocation::Top)
        .with_simd(simd);
      yuva420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
      assert_eq!(
        rgba, rgba_ref,
        "odd-height Top two-row flush (simd={simd}) must equal the forward-fold reference"
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn begin_frame_after_held_odd_top_row_clears_state() {
    // Nv21 cross-frame-corruption regression class (REQUIRED): a Top frame that
    // ends holding an odd row must clear that held state at the next `begin_frame`,
    // so frame N's deferred row never leaks into frame N+1. Feed a SHORT partial
    // frame (rows 0, 1 — leaving row 1 held), then start a fresh full frame; the
    // fresh frame must decode identically to a clean single-frame Top decode.
    let (y, u, v, a) = ramp_planes();
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let row_at = |r: usize| {
      let cr = r / 2;
      Yuva420pRow::new(
        &y[r * w..(r + 1) * w],
        &u[cr * cw..cr * cw + cw],
        &v[cr * cw..cr * cw + cw],
        &a[r * w..(r + 1) * w],
        r,
        ColorMatrix::Bt601,
        false,
      )
    };
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuva420p>::new(w, h)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Top);
    // Partial frame: begin, feed rows 0 (even) + 1 (odd → HELD), then abandon.
    crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
    crate::PixelSink::process(&mut sink, row_at(0)).unwrap();
    crate::PixelSink::process(&mut sink, row_at(1)).unwrap();
    assert!(
      sink.chroma_top_pending.is_some(),
      "row 1 (odd) must be held pending after the partial frame"
    );
    // Fresh frame: begin_frame MUST clear the held row + frozen phase.
    crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
    assert!(
      sink.chroma_top_pending.is_none(),
      "begin_frame must drop the held Top odd row (Nv21 regression class)"
    );
    for r in 0..h {
      crate::PixelSink::process(&mut sink, row_at(r)).unwrap();
    }
    drop(sink);
    assert_eq!(
      rgba,
      convert_rgba(ChromaLocation::Top, true),
      "the post-clear fresh frame must equal a clean single-frame Top decode"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn mid_frame_flip_to_from_top_is_rejected() {
    // A mid-frame vertical-phase flip to/from Top decodes a mixture of phases (or
    // leaks a held row), so the second row's flipped siting is rejected with
    // ChromaSitingChanged.
    let (y, u, v, a) = ramp_planes();
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let row_at = |r: usize| {
      let cr = r / 2;
      Yuva420pRow::new(
        &y[r * w..(r + 1) * w],
        &u[cr * cw..cr * cw + cw],
        &v[cr * cw..cr * cw + cw],
        &a[r * w..(r + 1) * w],
        r,
        ColorMatrix::Bt601,
        false,
      )
    };
    for (first, second) in [
      (ChromaLocation::Top, ChromaLocation::Center),
      (ChromaLocation::Center, ChromaLocation::Top),
      (ChromaLocation::Top, ChromaLocation::Bottom),
    ] {
      let mut rgba = std::vec![0u8; (W * H * 4) as usize];
      let mut sink = MixedSinker::<Yuva420p>::new(w, h)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_chroma_location(first);
      crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
      crate::PixelSink::process(&mut sink, row_at(0)).unwrap();
      sink.set_chroma_location(second);
      let err = crate::PixelSink::process(&mut sink, row_at(1)).unwrap_err();
      assert!(
        matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
        "a mid-frame {first:?} -> {second:?} flip must be rejected"
      );
    }
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
  fn bottomleft_rgba_matches_cosited_vfold_upsample_then_444_with_real_alpha() {
    // BottomLeft (h=0 co-sited + v=1): RGBA equals a 4:4:4 YUVA decode over the
    // co-sited-replicate + vertical-blend reconstruction with the SAME source α,
    // on both tiers; and it genuinely differs from Bottom (co-sited h) and from
    // the co-sited default (the v=1 fold).
    let (y, u, v, a) = ramp_planes();
    let (u444, v444) = ref_full_chroma_bottomleft(&u, &v);
    let ref_src = Yuva444pFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuva444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuva444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      convert_rgba(ChromaLocation::BottomLeft, true),
      rgba_ref,
      "bottom-left YUVA RGBA (SIMD) must equal cosited-vfold-then-4:4:4 (real source alpha)"
    );
    assert_eq!(
      convert_rgba(ChromaLocation::BottomLeft, false),
      rgba_ref,
      "bottom-left YUVA RGBA (scalar) must equal cosited-vfold-then-4:4:4"
    );
    assert_ne!(
      convert_rgba(ChromaLocation::BottomLeft, true),
      convert_rgba(ChromaLocation::Bottom, true),
      "BottomLeft (h=0) must differ from Bottom (h=0.5)"
    );
    assert_ne!(
      convert_rgba(ChromaLocation::BottomLeft, true),
      convert_rgba(ChromaLocation::Left, true),
      "BottomLeft (v=1) must differ from the co-sited default"
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

  // ---- mid-frame siting-flip rejection (identity path freeze) ---------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn direct_path_mid_frame_siting_flip_is_rejected() {
    // The identity (no-resample) 4:2:0 YUVA decode freezes the effective phase —
    // BOTH the horizontal centered flag and the vertical `Bottom` flag — on its
    // first output-bearing row. `set_chroma_location` is public, so flipping
    // `Bottom` ⇆ co-sited mid-frame must reject the next in-sequence row with
    // `ChromaSitingChanged`, WITHOUT growing the chroma scratch OR advancing the
    // stateful `chroma_prev` vertical lookback (its validity tag included) — else a
    // later even row would box-blend against a stale, gap-advanced chroma row.
    // Flipping back and retrying then decodes byte-for-byte like a clean single-phase
    // frame.
    let (y, u, v, a) = ramp_planes();
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let row_at = |r: usize| {
      let cr = r / 2;
      Yuva420pRow::new(
        &y[r * w..(r + 1) * w],
        &u[cr * cw..cr * cw + cw],
        &v[cr * cw..cr * cw + cw],
        &a[r * w..(r + 1) * w],
        r,
        ColorMatrix::Bt601,
        false,
      )
    };
    for (loc1, loc2) in [
      (ChromaLocation::Bottom, ChromaLocation::Left),
      (ChromaLocation::Left, ChromaLocation::Bottom),
      (ChromaLocation::Bottom, ChromaLocation::Center),
    ] {
      let want = convert_rgb(loc1, true);
      let mut rgb = std::vec![0u8; w * h * 3];
      let mut sink = MixedSinker::<Yuva420p>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_chroma_location(loc1)
        .with_simd(true);
      crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
      // Rows 0, 1 at the held siting; row 0 (even) primes the `Bottom` lookback.
      crate::PixelSink::process(&mut sink, row_at(0)).unwrap();
      crate::PixelSink::process(&mut sink, row_at(1)).unwrap();
      let scratch_len = sink.chroma_full.len();
      let prev_len = sink.chroma_prev.len();
      let prev_tag = sink.chroma_prev_row;

      // Flip and deliver the next EVEN row (idx 2) — the one a `Bottom` decode
      // box-blends against the lookback.
      sink.set_chroma_location(loc2);
      let err = crate::PixelSink::process(&mut sink, row_at(2)).unwrap_err();
      assert!(
        matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
        "direct path {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
      );
      // Retry-atomic (#180): the reject grew no scratch and left the vertical
      // lookback (buffer + validity tag) exactly as the accepted rows left it.
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
        "{loc1:?}->{loc2:?}: a rejected flip must not advance the lookback validity tag"
      );

      // Flip back and drive the rest in order; the output must match a clean decode.
      sink.set_chroma_location(loc1);
      for r in 2..h {
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

      /// The full-resolution U / V a **`BottomLeft`** (`h = 0`, `v = 1`) high-bit
      /// 4:2:0 decode reconstructs: the even-row vertical box blend of
      /// [`ref_full_chroma_bottom`], but the CO-SITED horizontal 2× replicate
      /// instead of the centered kernel.
      fn ref_full_chroma_bottomleft(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        let mut u444 = std::vec![0u16; w * h];
        let mut v444 = std::vec![0u16; w * h];
        let vblend = |plane: &[u16], cr: usize, prev: usize| -> Vec<u16> {
          (0..cw)
            .map(|c| {
              ((u32::from(plane[prev * cw + c]) + u32::from(plane[cr * cw + c]) + 1) >> 1) as u16
            })
            .collect()
        };
        for r in 0..h {
          let cr = r / 2;
          let (uh, vh) = if r & 1 == 0 {
            let prev = cr.saturating_sub(1);
            (vblend(u420, cr, prev), vblend(v420, cr, prev))
          } else {
            (
              u420[cr * cw..cr * cw + cw].to_vec(),
              v420[cr * cw..cr * cw + cw].to_vec(),
            )
          };
          for j in 0..cw {
            u444[r * w + 2 * j] = uh[j];
            u444[r * w + 2 * j + 1] = uh[j];
            v444[r * w + 2 * j] = vh[j];
            v444[r * w + 2 * j + 1] = vh[j];
          }
        }
        (u444, v444)
      }

      /// The full-resolution U / V a **`Top`** (`v = 0`, FORWARD fold) high-bit
      /// 4:2:0 decode reconstructs (logical `u16`) — the vertical MIRROR of
      /// [`ref_full_chroma_bottom`]: the EVEN luma row is co-sited with its chroma
      /// row, the ODD luma row forward box-blends `{cur, next}` (round-half-up,
      /// bottom clamp), each then horizontally centered. Generalised over `h` for
      /// the odd-height two-row-flush test.
      fn ref_full_chroma_top_geom(
        u420: &[u16],
        v420: &[u16],
        w: usize,
        h: usize,
      ) -> (Vec<u16>, Vec<u16>) {
        let cw = w / 2;
        let ch = h.div_ceil(2);
        let mut u444 = std::vec![0u16; w * h];
        let mut v444 = std::vec![0u16; w * h];
        let vblend = |plane: &[u16], cr: usize, next: usize| -> Vec<u32> {
          (0..cw)
            .map(|c| (u32::from(plane[cr * cw + c]) + u32::from(plane[next * cw + c]) + 1) >> 1)
            .collect()
        };
        for r in 0..h {
          let cr = r / 2;
          let (urow, vrow) = if r & 1 == 1 {
            let next = (cr + 1).min(ch - 1);
            (vblend(u420, cr, next), vblend(v420, cr, next))
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

      fn ref_full_chroma_top(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
        ref_full_chroma_top_geom(u420, v420, W as usize, H as usize)
      }

      /// The full-resolution U / V a **`TopLeft`** (`h = 0`, `v = 0`) high-bit
      /// 4:2:0 decode reconstructs: the odd-row forward vertical box blend of
      /// [`ref_full_chroma_top`], fed to the CO-SITED horizontal 2× replicate.
      fn ref_full_chroma_topleft(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        let ch = h.div_ceil(2);
        let mut u444 = std::vec![0u16; w * h];
        let mut v444 = std::vec![0u16; w * h];
        let vblend = |plane: &[u16], cr: usize, next: usize| -> Vec<u16> {
          (0..cw)
            .map(|c| {
              ((u32::from(plane[cr * cw + c]) + u32::from(plane[next * cw + c]) + 1) >> 1) as u16
            })
            .collect()
        };
        for r in 0..h {
          let cr = r / 2;
          let (uh, vh) = if r & 1 == 1 {
            let next = (cr + 1).min(ch - 1);
            (vblend(u420, cr, next), vblend(v420, cr, next))
          } else {
            (
              u420[cr * cw..cr * cw + cw].to_vec(),
              v420[cr * cw..cr * cw + cw].to_vec(),
            )
          };
          for j in 0..cw {
            u444[r * w + 2 * j] = uh[j];
            u444[r * w + 2 * j + 1] = uh[j];
            v444[r * w + 2 * j] = vh[j];
            v444[r * w + 2 * j + 1] = vh[j];
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
        // `BottomLeft` / `TopLeft` are EXCLUDED: co-sited horizontally but
        // vertically sited (`v = 1` / `v = 0` forward fold), so each folds a
        // vertical blend (their own tests below).
        for loc in [
          ChromaLocation::Unspecified,
          ChromaLocation::Unknown(99),
          ChromaLocation::Left,
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
      fn top_bottom_center_are_all_distinct_at_native_depth() {
        // RFC #238 Top: `Top` folds the FORWARD (`v = 0`) triangle, `Bottom` the
        // BACKWARD (`v = 1`) one, `Center` neither — so all three diverge. Asserted
        // at NATIVE u16 depth: at high bit depths the fold's sub-LSB effect can round
        // away in the 8-bit RGBA, but it is exact in the native-depth output.
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
        let center = decode_u16(ChromaLocation::Center);
        assert_ne!(decode_u16(ChromaLocation::Top), center, "Top (v=0) must diverge from Center");
        assert_ne!(decode_u16(ChromaLocation::Bottom), center, "Bottom (v=1) must diverge from Center");
        assert_ne!(
          decode_u16(ChromaLocation::Top),
          decode_u16(ChromaLocation::Bottom),
          "Top and Bottom fold in opposite directions"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_rgba_matches_forward_vfold_upsample_then_444_with_real_alpha() {
        // RFC #238 Top: the forward one-row-delay high-bit decode reconstructs chroma
        // with the ODD-row forward box-blend + centered horizontal upsample, so its
        // RGBA equals a `Yuva444pN` decode on that reconstruction + the SAME full-res
        // source alpha (held with each deferred odd row), on BOTH tiers.
        let (y, u, v, a) = ramp_planes();
        let (u444, v444) = ref_full_chroma_top(&u, &v);
        let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
        let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgba(&mut rgba_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          convert_rgba(ChromaLocation::Top, true),
          rgba_ref,
          "top high-bit YUVA RGBA (SIMD) must equal forward-vfold-upsample-then-4:4:4"
        );
        assert_eq!(
          convert_rgba(ChromaLocation::Top, false),
          rgba_ref,
          "top high-bit YUVA RGBA (scalar) must equal forward-vfold-upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn topleft_rgba_matches_cosited_forward_vfold_then_444_with_real_alpha() {
        // TopLeft (h=0 co-sited + v=0 forward fold) high-bit: RGBA equals a
        // `Yuva444pN` decode over the co-sited-replicate + forward-vertical-blend
        // reconstruction with the SAME source α, on both tiers; and differs from Top.
        let (y, u, v, a) = ramp_planes();
        let (u444, v444) = ref_full_chroma_topleft(&u, &v);
        let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
        let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgba(&mut rgba_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          convert_rgba(ChromaLocation::TopLeft, true),
          rgba_ref,
          "top-left high-bit YUVA RGBA (SIMD) must equal cosited-forward-vfold-then-4:4:4"
        );
        assert_eq!(
          convert_rgba(ChromaLocation::TopLeft, false),
          rgba_ref,
          "top-left high-bit YUVA RGBA (scalar) must equal cosited-forward-vfold-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_rgba_u16_alpha_channel_equals_source_plane() {
        // Native-depth alpha is held with each deferred odd row and emitted in order,
        // so the Top RGBA u16 alpha channel is the source plane verbatim.
        let (y, u, v, a) = ramp_planes();
        let mut rgba = std::vec![0u16; (W * H * 4) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgba_u16(&mut rgba)
          .unwrap()
          .with_chroma_location(ChromaLocation::Top);
        $walker(&frame(&y, &u, &v, &a), false, ColorMatrix::Bt601, &mut sink).unwrap();
        for (i, &src_a) in a.iter().enumerate() {
          assert_eq!(rgba[i * 4 + 3], src_a, "top native alpha at px {i} must equal the source plane");
        }
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
      fn bottomleft_rgba_matches_cosited_vfold_upsample_then_444_with_real_alpha() {
        // BottomLeft (h=0 co-sited + v=1) high-bit: RGBA equals a `Yuva444pN`
        // decode over the co-sited-replicate + vertical-blend reconstruction with
        // the SAME source α, on both tiers; and differs from Bottom (co-sited h)
        // and from the co-sited default (the v=1 fold).
        let (y, u, v, a) = ramp_planes();
        let (u444, v444) = ref_full_chroma_bottomleft(&u, &v);
        let ref_src = $RefFrame::try_new(&y, &u444, &v444, &a, W, H, W, W, W, W).unwrap();
        let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgba(&mut rgba_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          convert_rgba(ChromaLocation::BottomLeft, true),
          rgba_ref,
          "bottom-left high-bit YUVA RGBA (SIMD) must equal cosited-vfold-then-4:4:4"
        );
        assert_eq!(
          convert_rgba(ChromaLocation::BottomLeft, false),
          rgba_ref,
          "bottom-left high-bit YUVA RGBA (scalar) must equal cosited-vfold-then-4:4:4"
        );
        assert_ne!(
          convert_rgba(ChromaLocation::BottomLeft, true),
          convert_rgba(ChromaLocation::Bottom, true),
          "BottomLeft (h=0) must differ from Bottom (h=0.5)"
        );
        // (The v=1 fold's non-vacuity + exact value are pinned across all depths
        // by the oracle match above and the resample suite's vramp tests; this
        // direct ramp's per-row step is sub-8-bit-RGB-quantization at 16-bit.)
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
