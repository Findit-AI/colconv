//! Chroma-siting-aware 4:2:2 upsampling for the semi-planar `Nv16` (#302) — the
//! 4:2:2 sibling of `chroma_siting_nv` (semi-planar 4:2:0 `Nv12` / `Nv21`) and
//! the semi-planar twin of `chroma_siting_422` (planar 4:2:2 `Yuv422p`).
//!
//! 4:2:2 subsamples chroma 2:1 horizontally ONLY (one chroma row per luma row,
//! no vertical subsampling), so the centered reconstruction reuses the SAME
//! de-interleave + phase-0.5 horizontal upsample 4:2:0 uses — only the walker's
//! per-row chroma differs. Covers: the default / co-sited path staying
//! byte-identical to the pre-#302 fused decode (the regression guard, negative-
//! controlled by a ramp that makes the phase observable); the centered RGB /
//! RGBA / HSV identity decodes matching an independent upsample-then-4:4:4
//! reference; cross-format equivalence to the planar `Yuv422p` centered decode
//! (catches a U/V swap in the de-interleave); SIMD-vs-scalar parity; the
//! `ColorSpec` siting flow; the `ChromaDerivedNcl` matrix-tag-fallback
//! consistency; the preflight-ordering alloc-failure atomicity (luma untouched);
//! and the no-output invariant (no row math / no allocation when nothing is
//! attached, including a 32-bit-offset-overflow geometry).

use super::*;
use crate::{ChromaLocation, ColorMatrix};

const W: u32 = 16;
const H: u32 = 8;

/// Flat luma + per-column chroma ramp on a half-width, **full-height** chroma
/// plane (one chroma row per luma row — 4:2:2). Distinct adjacent columns make
/// the horizontal phase observable; the `+ r` term keeps chroma rows distinct so
/// a vertical mistake would surface.
fn ramp_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let y = std::vec![128u8; w * h];
  let mut u = std::vec![0u8; cw * h];
  let mut v = std::vec![0u8; cw * h];
  for r in 0..h {
    for c in 0..cw {
      u[r * cw + c] = (16 + c * 24 + r * 3).min(240) as u8;
      v[r * cw + c] = (240 - c * 24).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Interleaves the half-width planar U / V into an `Nv16` semi-planar chroma
/// plane (`U` at the even byte), `width` bytes per chroma row, `height` rows.
fn interleave_nv16(u: &[u8], v: &[u8]) -> Vec<u8> {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut uv = std::vec![0u8; w * h];
  for r in 0..h {
    for c in 0..cw {
      uv[r * w + 2 * c] = u[r * cw + c];
      uv[r * w + 2 * c + 1] = v[r * cw + c];
    }
  }
  uv
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

/// Full-resolution U / V planes a centered-siting 4:2:2 decode reconstructs:
/// each luma row `r` takes chroma row `r` (1:1 — NO vertical subsampling, unlike
/// 4:2:0's `r / 2`) upsampled with the centered weights. Feeding these to a
/// `Yuv444p` conversion is the end-to-end oracle.
fn ref_full_chroma(u422: &[u8], v422: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  for r in 0..h {
    let urow = ref_upsample_center_h(&u422[r * cw..r * cw + cw], w);
    let vrow = ref_upsample_center_h(&v422[r * cw..r * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

/// Centered identity-decode RGB for a siting + SIMD toggle.
fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
  let (yp, up, vp) = ramp_planes();
  let uvp = interleave_nv16(&up, &vp);
  let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(loc.clone())
    .with_simd(simd);
  nv16_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
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
  let (yp, up, vp) = ramp_planes();
  let uvp = interleave_nv16(&up, &vp);
  let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Left);
  nv16_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
  let chroma_len = sink.chroma_full.len();
  let half_len = sink.semi_planar_u_half.len();
  drop(sink);
  assert_eq!(
    chroma_len, 0,
    "co-sited path must not grow the chroma scratch"
  );
  assert_eq!(
    half_len, 0,
    "co-sited path must not grow the de-interleave scratch"
  );
}

// ---- centered path correctness ---------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_rgb_matches_upsample_then_444_reference() {
  let (yp, up, vp) = ramp_planes();
  // Reference: upsample chroma (centered weights, per-row) to full resolution,
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
  let uvp = interleave_nv16(&up, &vp);
  let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);
  nv16_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
  let chroma_len = sink.chroma_full.len();
  let half_len = sink.semi_planar_u_half.len();
  drop(sink);
  assert_eq!(
    chroma_len,
    2 * W as usize,
    "centered path stages U+V at full width"
  );
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
fn top_and_bottom_route_like_center_horizontally() {
  // Top / Bottom share Center's horizontal (centered) phase; 4:2:2 has no
  // vertical axis to drive, so all three agree.
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
  // Negative control for the byte-identity guard: on a chroma ramp the centered
  // phase must move chroma relative to the co-sited default.
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
fn centered_matches_yuv422p_centered() {
  // De-interleaving the interleaved chroma back to planar U / V must reconstruct
  // the SAME planes the planar twin holds, so a centered Nv16 decode is
  // byte-identical to a centered Yuv422p decode — the strongest catch for a U/V
  // swap in the de-interleave (`swap_uv`). Uses Bt601: this equivalence holds on
  // the shared matrix-tag path (`ChromaDerivedNcl` is covered separately).
  let (yp, up, vp) = ramp_planes();
  let src422 = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb422 = std::vec![0u8; (W * H * 3) as usize];
  let mut sink422 = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb422)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);
  yuv422p_to(&src422, false, KernelMatrix::Bt601, &mut sink422).unwrap();

  assert_eq!(
    convert_rgb(ChromaLocation::Center, true),
    rgb422,
    "centered semi-planar must equal centered Yuv422p of the same planes"
  );
}

// ---- ChromaDerivedNcl: NV uses the matrix-tag fallback, not primaries ---

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_chroma_derived_ncl_uses_matrix_tag_fallback() {
  // Nv16 is NOT ChromaDerivedNcl-primaries-wired. BOTH paths (the default fused
  // kernel AND the centered 4:4:4 kernel) resolve ChromaDerivedNcl via the
  // shared BT.709 matrix-tag fallback (`Coefficients::for_matrix`), IGNORING the
  // ColorSpec primaries — so default and centered stay internally consistent
  // (the centered phase shift is the ONLY difference between them).
  use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

  let (yp, up, vp) = ramp_planes();
  let uvp = interleave_nv16(&up, &vp);

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
    let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);
    let mut rgb = std::vec![0u8; (W * H * 3) as usize];
    let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(&spec(loc));
    nv16_to(&src, false, KernelMatrix::ChromaDerivedNcl, &mut sink).unwrap();
    rgb
  };
  let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
    let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);
    let mut rgb = std::vec![0u8; (W * H * 3) as usize];
    let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc.clone());
    nv16_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
    rgb
  };

  assert_eq!(
    decode_cdn(ChromaLocation::Center),
    decode_bt709(ChromaLocation::Center),
    "centered Nv16 ChromaDerivedNcl must resolve via the BT.709 matrix-tag fallback, not primaries"
  );
  assert_eq!(
    decode_cdn(ChromaLocation::Left),
    decode_bt709(ChromaLocation::Left),
    "default Nv16 ChromaDerivedNcl must resolve via the same BT.709 matrix-tag fallback"
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
  let uvp = interleave_nv16(&up, &vp);
  let (u444, v444) = ref_full_chroma(&up, &vp);

  // RGBA-only path.
  {
    let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    nv16_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuv444p_to(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      rgba, rgba_ref,
      "centered RGBA must equal upsample-then-4:4:4"
    );
  }

  // HSV-direct path (no RGB / RGBA attached).
  {
    let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);
    let (mut h, mut s, mut v) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    nv16_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

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
    assert_eq!(
      (h, s, v),
      (hr, sr, vr),
      "centered HSV must equal upsample-then-4:4:4"
    );
  }
}

// ---- end-to-end ColorSpec flow (no manual with_chroma_location) ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn color_spec_center_drives_decode_without_manual_chroma_call() {
  use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer, YuvOptions};

  let (yp, up, vp) = ramp_planes();
  let uvp = interleave_nv16(&up, &vp);
  let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);

  let info = ColorInfo::new(
    Primaries::Bt709,
    Transfer::Bt709,
    ColorMatrix::Bt601,
    DynamicRange::Limited,
    ChromaLocation::Center,
  );
  let spec = ColorSpec::from_info(PixelFormat::Yuv422p, info);
  let opts = YuvOptions::from_color_spec(&spec).unwrap();
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_color_spec(&spec);
  nv16_to(&src, opts.full_range(), opts.matrix(), &mut sink).unwrap();
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
  let uvp = interleave_nv16(&up, &vp);
  let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);

  // Negative control: unarmed, the SAME luma + centered-RGB config DOES write
  // luma — so the armed "untouched" assertion below is non-vacuous.
  {
    let mut luma_ok = std::vec![0xABu8; (W * H) as usize];
    let mut rgb_ok = std::vec![0xCDu8; (W * H * 3) as usize];
    let mut sink_ok = MixedSinker::<Nv16>::new(W as usize, H as usize)
      .with_luma(&mut luma_ok)
      .unwrap()
      .with_rgb(&mut rgb_ok)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    nv16_to(&src, false, KernelMatrix::Bt601, &mut sink_ok).unwrap();
    drop(sink_ok);
    assert!(
      luma_ok.iter().any(|&b| b != 0xAB),
      "control: the centered path writes luma when the scratch alloc is not armed"
    );
  }

  // Armed: a centered RGB decode whose chroma-scratch allocation fails must leave
  // EVERY output — luma included — untouched, because the scratch is reserved
  // (fallibly) BEFORE any output row is written.
  let mut luma = std::vec![0xABu8; (W * H) as usize];
  let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Nv16>::new(W as usize, H as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);

  super::super::arm_chroma_full_alloc_failure();
  let err = nv16_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
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

// ---- no-output invariant: guard runs before the row-offset math --------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn no_output_row_does_not_allocate_chroma_scratch() {
  // A centered-siting sink with NO outputs attached returns before the centered
  // preflight, so neither chroma scratch is ever reserved.
  let (yp, up, vp) = ramp_planes();
  let uvp = interleave_nv16(&up, &vp);
  let src = Nv16Frame::new(&yp, &uvp, W, H, W, W);
  let mut sink =
    MixedSinker::<Nv16>::new(W as usize, H as usize).with_chroma_location(ChromaLocation::Center);
  nv16_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
  let chroma_len = sink.chroma_full.len();
  let half_len = sink.semi_planar_u_half.len();
  drop(sink);
  assert_eq!(
    chroma_len, 0,
    "a no-output centered row must not reserve the chroma scratch"
  );
  assert_eq!(
    half_len, 0,
    "a no-output centered row must not reserve the de-interleave scratch"
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
  // overflow checks. With no outputs attached, `process` must return `Ok(())`
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
  let uv = std::vec![128u8; w]; // w/2 interleaved U,V pairs
  let mut sink = MixedSinker::<Nv16>::new(w, h).with_chroma_location(ChromaLocation::Center);
  let row = crate::source::Nv16Row::new(&y, &uv, idx, KernelMatrix::Bt601, false);
  crate::PixelSink::process(&mut sink, row).unwrap();
  let chroma_len = sink.chroma_full.len();
  drop(sink);
  assert_eq!(
    chroma_len, 0,
    "a no-output large-geometry row must allocate nothing"
  );
}

// ---- mid-frame siting-flip rejection (identity path freeze) -----------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn direct_path_mid_frame_siting_flip_is_rejected() {
  // The identity (no-resample) 4:2:2 Nv16 decode freezes the horizontal siting on
  // its first output-bearing row, mirroring the planar `Yuv422p` twin. Flipping
  // co-sited ⇆ centered mid-frame must reject the next in-sequence row with
  // `ChromaSitingChanged` (leaving the centered chroma scratch untouched); flipping
  // back and retrying then matches a clean single-phase decode byte-for-byte.
  use super::super::MixedSinkerError;
  let (yp, up, vp) = ramp_planes();
  let uvp = interleave_nv16(&up, &vp);
  let w = W as usize;
  let h = H as usize;
  for (loc1, loc2) in [
    (ChromaLocation::Left, ChromaLocation::Center),
    (ChromaLocation::Center, ChromaLocation::Left),
  ] {
    let want = convert_rgb(loc1.clone(), true);
    let mut rgb = std::vec![0u8; w * h * 3];
    let mut sink = MixedSinker::<Nv16>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc1.clone())
      .with_simd(true);
    crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
    let row0 = Nv16Row::new(&yp[0..w], &uvp[0..w], 0, KernelMatrix::Bt601, false);
    crate::PixelSink::process(&mut sink, row0).unwrap();
    let scratch_len = sink.chroma_full.len();

    sink.set_chroma_location(loc2.clone());
    let row1 = Nv16Row::new(&yp[w..2 * w], &uvp[w..2 * w], 1, KernelMatrix::Bt601, false);
    let err = crate::PixelSink::process(&mut sink, row1).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "direct path {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
    assert_eq!(
      sink.chroma_full.len(),
      scratch_len,
      "{loc1:?}->{loc2:?}: a rejected flip must not grow the chroma scratch"
    );

    sink.set_chroma_location(loc1.clone());
    for r in 1..h {
      let row = Nv16Row::new(
        &yp[r * w..(r + 1) * w],
        &uvp[r * w..(r + 1) * w],
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
