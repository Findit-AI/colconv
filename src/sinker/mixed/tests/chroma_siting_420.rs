//! Chroma-siting-aware 4:2:0 upsampling for `Yuv420p` (#302).
//!
//! Covers: the centered-siting horizontal upsample kernel against a
//! hand-computed oracle; the default / co-sited path staying byte-identical
//! to the pre-#302 nearest-neighbor decode (the regression guard); the
//! centered RGB / RGBA / HSV identity decodes matching an independent
//! "upsample-then-4:4:4" reference; SIMD-vs-scalar parity of the centered
//! path; and that the centered phase actually shifts chroma horizontally.

use super::*;
use crate::ChromaLocation;

const W: u32 = 16;
const H: u32 = 8;

/// A `Yuv420p` frame with flat luma and a per-column chroma ramp, so the
/// horizontal chroma phase is observable (a solid chroma frame would make
/// every siting identical).
fn ramp_yuv420p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let y = std::vec![128u8; w * h];
  let mut u = std::vec![0u8; cw * ch];
  let mut v = std::vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      // Distinct per-column ramps for U and V; the `+ r` keeps chroma rows
      // from being identical so a (hypothetical) vertical mistake would show.
      u[r * cw + c] = (16 + c * 24 + r * 3).min(240) as u8;
      v[r * cw + c] = (240 - c * 24).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp. Mirrors the
/// documented formula but is written separately from the production kernel so
/// it is a real oracle.
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

/// Builds the full-resolution U / V planes a centered-siting `Yuv420p`
/// decode should reconstruct: each luma row `r` takes chroma row `r / 2`
/// (the walker's vertical replication, unchanged by #302) horizontally
/// upsampled with the centered weights. Feeding these to a `Yuv444p`
/// conversion is the end-to-end oracle for the centered path.
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

fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(loc)
    .with_simd(simd);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  rgb
}

// ---- kernel oracle ---------------------------------------------------------

#[test]
fn center_upsample_kernel_matches_hand_computed() {
  // c = [0, 0, 100, 100] (half = 4, width = 8). Centered reconstruction
  // ramps the step and shifts it right of the co-sited boundary:
  //   even 2j = (c[j-1] + 3·c[j] + 2) >> 2, odd 2j+1 = (3·c[j] + c[j+1] + 2) >> 2.
  let c_half = [0u8, 0, 100, 100];
  let mut out = [0u8; 8];
  crate::row::scalar::chroma_upsample_2to1_center_h(&c_half, &mut out, 8);
  assert_eq!(out, [0, 0, 0, 25, 75, 100, 100, 100]);
}

#[test]
fn center_upsample_kernel_clamps_edges() {
  // Width 4: left edge even = c[0] exactly, right edge odd = c[last] exactly.
  let c_half = [10u8, 20];
  let mut out = [0u8; 4];
  crate::row::scalar::chroma_upsample_2to1_center_h(&c_half, &mut out, 4);
  assert_eq!(out, [10, 13, 18, 20]);
  assert_eq!(out[0], c_half[0], "left edge even column is co-sited");
  assert_eq!(out[3], c_half[1], "right edge odd column is co-sited");
}

// ---- default / co-sited path is byte-identical (regression guard) ----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn default_and_cosited_sitings_are_byte_identical() {
  // The pre-#302 baseline: a sink that never sets a chroma location.
  let baseline = convert_rgb(ChromaLocation::Unspecified, true);

  // Unspecified, Unknown, and the fully co-sited default (`Left`, `v = 0.5`)
  // keep the exact nearest-neighbor decode — bit-for-bit equal to the baseline
  // even though the chroma plane is a non-trivial ramp. `BottomLeft` (`v = 1`)
  // and `TopLeft` (`v = 0`) are EXCLUDED: both are co-sited horizontally
  // (`h = 0`) but VERTICALLY sited, so they fold a vertical box blend (backward
  // for BottomLeft, forward for TopLeft) and are covered by their own tests.
  for loc in [
    ChromaLocation::Unspecified,
    ChromaLocation::Unknown(99),
    ChromaLocation::Left,
  ] {
    assert_eq!(
      convert_rgb(loc, true),
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
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Left);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  let chroma_len = sink.chroma_full.len();
  drop(sink);
  assert_eq!(
    chroma_len, 0,
    "co-sited path must not grow the chroma scratch"
  );
}

// ---- centered path correctness ---------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_rgb_matches_upsample_then_444_reference() {
  let (yp, up, vp) = ramp_yuv420p();

  // Reference: horizontally upsample chroma (centered weights) to full
  // resolution, then run the ordinary 4:4:4 decode.
  let (u444, v444) = ref_full_chroma(&up, &vp);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();

  assert_eq!(
    convert_rgb(ChromaLocation::Center, true),
    rgb_ref,
    "centered 4:2:0 RGB must equal upsample-then-4:4:4"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_grows_chroma_scratch_to_full_width() {
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  let chroma_len = sink.chroma_full.len();
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
fn top_forward_vblend_differs_from_center_and_bottom() {
  // Top shares Center's horizontal (centered) phase and now consumes its
  // vertical phase through the FORWARD one-row delay: the ODD output row is the
  // box average of its chroma row and the NEXT one. On a fixture whose chroma
  // varies across rows Top therefore DIFFERS from Center (horizontal-only) AND
  // from Bottom (which box-blends the EVEN row with the PREVIOUS chroma row).
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
fn centered_phase_differs_from_default() {
  // The whole point: on a chroma ramp the centered phase must move chroma
  // relative to the co-sited / nearest-neighbor default.
  assert_ne!(
    convert_rgb(ChromaLocation::Center, true),
    convert_rgb(ChromaLocation::Left, true),
    "centered siting must shift chroma vs the co-sited default"
  );
}

// ---- SIMD vs scalar parity -------------------------------------------------

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

// ---- RGBA / HSV identity outputs also honor siting -------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_rgba_and_hsv_match_444_reference() {
  let (yp, up, vp) = ramp_yuv420p();
  let (u444, v444) = ref_full_chroma(&up, &vp);

  // RGBA-only path.
  {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      rgba, rgba_ref,
      "centered RGBA must equal upsample-then-4:4:4"
    );
  }

  // HSV-direct path (no RGB / RGBA attached).
  {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let (mut h, mut s, mut v) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let (mut hr, mut sr, mut vr) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_hsv(&mut hr, &mut sr, &mut vr)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      (h, s, v),
      (hr, sr, vr),
      "centered HSV must equal upsample-then-4:4:4"
    );
  }
}

// ---- end-to-end ColorSpec flow (no manual with_chroma_location) ------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn color_spec_center_drives_decode_without_manual_chroma_call() {
  use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer, YuvOptions};

  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);

  // Drive the decode from a ColorSpec carrying ChromaLocation::Center via the
  // NORMAL path: YuvOptions::from_color_spec(spec) for the walk (matrix +
  // range) and the sink's ColorSpec entry point for the siting — with NO
  // manual `with_chroma_location` call.
  let info = ColorInfo::new(
    Primaries::Bt709,
    Transfer::Bt709,
    ColorMatrix::Bt601,
    DynamicRange::Limited,
    ChromaLocation::Center,
  );
  let spec = ColorSpec::from_info(PixelFormat::Yuv420p, info);
  let opts = YuvOptions::from_color_spec(spec);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_color_spec(spec);
  yuv420p_to(&src, opts.full_range(), opts.matrix(), &mut sink).unwrap();
  drop(sink);

  // `opts` mirrors `convert_rgb`'s limited-range Bt601 walk, so the
  // spec-driven output must (a) differ from the default (no siting) and
  // (b) match the explicit centered path bit-for-bit.
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

// ---- centered siting + ChromaDerivedNcl (#303 cross-feature seam) -----------

/// The centered chroma-siting decode must honour
/// [`ColorMatrix::ChromaDerivedNcl`] just like the non-centered path: its
/// Kr/Kb are derived from the ColorSpec's primaries, NOT the BT.709 fallback.
/// Before the fix the centered branch called the 4:4:4 kernels with only the
/// matrix tag, so a non-BT709 ChromaDerivedNcl decoded with BT.709
/// coefficients (wrong colors) and could even run on SIMD — this test fails
/// against that bug. Covers RGB and RGBA, and the scalar-routing invariant.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_chroma_derived_ncl_uses_primaries_not_bt709() {
  use crate::{
    ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer,
    row::{
      yuv_444_to_rgb_row, yuv_444_to_rgb_row_primaries, yuv_444_to_rgba_row,
      yuv_444_to_rgba_row_primaries,
    },
  };

  let w = W as usize;
  let h = H as usize;
  let (yp, up, vp) = ramp_yuv420p();
  // The centered full-width chroma the decode reconstructs per row.
  let (u444, v444) = ref_full_chroma(&up, &vp);
  // A non-BT709 primary set: ChromaDerivedNcl here must diverge from BT.709.
  let prim = Primaries::Bt2020;
  let info = ColorInfo::new(
    prim,
    Transfer::Bt709,
    ColorMatrix::ChromaDerivedNcl,
    DynamicRange::Limited,
    ChromaLocation::Center,
  );
  let spec = ColorSpec::from_info(PixelFormat::Yuv420p, info);

  // Drive the centered Yuv420p decode via the ColorSpec (limited-range).
  let decode_rgb = |simd: bool| {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgb = std::vec![0u8; w * h * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(spec)
      .with_simd(simd);
    yuv420p_to(&src, false, ColorMatrix::ChromaDerivedNcl, &mut sink).unwrap();
    rgb
  };

  // Exact reference: same centered-upsampled chroma, decoded 4:4:4 with the
  // primaries-derived coefficients (vs the BT.709 fallback the bug produced).
  let mut rgb_derived = std::vec![0u8; w * h * 3];
  let mut rgb_bt709 = std::vec![0u8; w * h * 3];
  for r in 0..h {
    let (ys, us, vs) = (
      &yp[r * w..r * w + w],
      &u444[r * w..r * w + w],
      &v444[r * w..r * w + w],
    );
    yuv_444_to_rgb_row_primaries(
      ys,
      us,
      vs,
      &mut rgb_derived[r * w * 3..(r + 1) * w * 3],
      w,
      ColorMatrix::ChromaDerivedNcl,
      prim,
      false,
      true,
    );
    yuv_444_to_rgb_row(
      ys,
      us,
      vs,
      &mut rgb_bt709[r * w * 3..(r + 1) * w * 3],
      w,
      ColorMatrix::Bt709,
      false,
      true,
    );
  }
  let rgb = decode_rgb(true);
  assert_eq!(
    rgb, rgb_derived,
    "centered ChromaDerivedNcl RGB must use the primaries-derived coefficients"
  );
  assert_ne!(
    rgb, rgb_bt709,
    "centered ChromaDerivedNcl RGB must NOT decode as the BT.709 fallback (the bug)"
  );
  assert_ne!(
    rgb_derived, rgb_bt709,
    "sanity: Bt2020-derived and BT.709 coefficients differ on the chroma ramp"
  );
  // Scalar-routing invariant: ChromaDerivedNcl is deterministic across tiers
  // (it is forced onto the scalar kernel), so SIMD == scalar.
  assert_eq!(
    rgb,
    decode_rgb(false),
    "centered ChromaDerivedNcl must be scalar-routed (no SIMD/scalar split)"
  );

  // ---- RGBA twin ----
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgba(&mut rgba)
    .unwrap()
    .with_color_spec(spec);
  yuv420p_to(&src, false, ColorMatrix::ChromaDerivedNcl, &mut sink).unwrap();
  drop(sink);

  let mut rgba_derived = std::vec![0u8; w * h * 4];
  let mut rgba_bt709 = std::vec![0u8; w * h * 4];
  for r in 0..h {
    let (ys, us, vs) = (
      &yp[r * w..r * w + w],
      &u444[r * w..r * w + w],
      &v444[r * w..r * w + w],
    );
    yuv_444_to_rgba_row_primaries(
      ys,
      us,
      vs,
      &mut rgba_derived[r * w * 4..(r + 1) * w * 4],
      w,
      ColorMatrix::ChromaDerivedNcl,
      prim,
      false,
      true,
    );
    yuv_444_to_rgba_row(
      ys,
      us,
      vs,
      &mut rgba_bt709[r * w * 4..(r + 1) * w * 4],
      w,
      ColorMatrix::Bt709,
      false,
      true,
    );
  }
  assert_eq!(
    rgba, rgba_derived,
    "centered ChromaDerivedNcl RGBA must use the primaries-derived coefficients"
  );
  assert_ne!(
    rgba, rgba_bt709,
    "centered ChromaDerivedNcl RGBA must NOT decode as the BT.709 fallback (the bug)"
  );
}

// ---- preflight-ordering atomicity (#302, cf. #180) -------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // A sink requesting luma PLUS a centered RGB decode whose chroma-scratch
  // allocation fails must leave EVERY output buffer — luma included —
  // untouched on the error path: the centered scratch is reserved (fallibly)
  // BEFORE any output row is written, so a refusal can't half-update the
  // frame.
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut luma = std::vec![0xABu8; (W * H) as usize];
  let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);

  super::super::arm_chroma_full_alloc_failure();
  let err = yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
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

// Gated on `yuva`: reuses the crate's existing RGB-scratch allocation
// failpoint (`arm_rgb_scratch_alloc_failure`, itself `yuva`-gated). Under
// `--all-features` both `yuv-planar` and `yuva` are on, so it runs.
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // Codex R3's exact case: luma + RGBA + HSV with NO rgb output. That is
  // `want_hsv && want_rgba && !want_rgb` → `need_rgb_kernel` with no caller
  // RGB buffer, so the decode grows the RGB row scratch
  // (`rgb_row_buf_or_scratch`'s scratch arm). With that allocation armed to
  // fail, the up-front preflight returns AllocationFailed BEFORE any output
  // row — luma included — is written.
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut luma = std::vec![0xABu8; (W * H) as usize];
  let mut rgba = std::vec![0xCDu8; (W * H * 4) as usize];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; (W * H) as usize],
    std::vec![0xCDu8; (W * H) as usize],
    std::vec![0xCDu8; (W * H) as usize],
  );
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}

// ---- bottom-sited VERTICAL phase (#302 deferred vertical part) --------------

/// A `Yuv420p` frame with flat luma and a per-ROW chroma ramp, so the vertical
/// chroma phase is observable in isolation (constant within each chroma row,
/// stepped between rows — a horizontal-only siting leaves it untouched, while a
/// vertical blend visibly moves it).
fn vramp_yuv420p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let y = std::vec![128u8; w * h];
  let mut u = std::vec![0u8; cw * ch];
  let mut v = std::vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      // Flat across columns, a strong step between chroma rows: only a VERTICAL
      // reconstruction changes these, so Bottom != the horizontal-only sitings.
      u[r * cw + c] = (20 + r * 40).min(240) as u8;
      v[r * cw + c] = (220 - r * 40).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Independent reference for the bottom-sited (`v = 1`) full reconstruction: per
/// luma row `r`, the EVEN rows take the vertical box average of chroma rows
/// `r/2 - 1` (clamped to `r/2` at the top edge) and `r/2`; the ODD rows take
/// chroma row `r/2` directly. Each resulting half-row is then horizontally
/// upsampled with the SAME centered `1/4`–`3/4` weights as Center (Bottom is
/// `h = 0.5`). Written separately from the production kernels so it is a true
/// oracle.
fn ref_full_chroma_bottom(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  let vblend = |plane: &[u8], cr: usize, prev: usize| -> Vec<u8> {
    (0..cw)
      .map(|c| {
        let a = plane[prev * cw + c] as u32;
        let b = plane[cr * cw + c] as u32;
        ((a + b + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  for r in 0..h {
    let cr = r / 2;
    let (uhalf, vhalf) = if r & 1 == 0 {
      // Even row: vertical box blend of chroma rows cr-1 (clamp to cr) and cr.
      let prev = cr.saturating_sub(1);
      (vblend(u420, cr, prev), vblend(v420, cr, prev))
    } else {
      // Odd row: co-sited with chroma row cr (no vertical blend).
      let _ = ch;
      (
        u420[cr * cw..cr * cw + cw].to_vec(),
        v420[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    let urow = ref_upsample_center_h(&uhalf, w);
    let vrow = ref_upsample_center_h(&vhalf, w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

fn convert_rgb_with(loc: ChromaLocation, simd: bool, yp: &[u8], up: &[u8], vp: &[u8]) -> Vec<u8> {
  let src = Yuv420pFrame::new(yp, up, vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(loc)
    .with_simd(simd);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  rgb
}

// ---- bottom-sited vertical kernel oracle -----------------------------------

#[test]
fn bottom_even_kernel_matches_hand_computed() {
  // prev = [0, 0, 100, 100], cur = [40, 40, 60, 60] (half = 4, width = 8).
  // Vertical box blend e = (prev + cur + 1) >> 1 = [20, 20, 80, 80], then the
  // centered horizontal 1/4-3/4 reconstruction with edge clamp:
  //   2j   = (e[j-1] + 3 e[j] + 2) >> 2,  2j+1 = (3 e[j] + e[j+1] + 2) >> 2.
  let prev = [0u8, 0, 100, 100];
  let cur = [40u8, 40, 60, 60];
  let mut out = [0u8; 8];
  crate::row::scalar::chroma_upsample_420_bottom_even_h(&prev, &cur, &mut out, 8);
  // e = [20,20,80,80] -> [20, 20, 20, 35, 65, 80, 80, 80].
  assert_eq!(out, [20, 20, 20, 35, 65, 80, 80, 80]);
}

#[test]
fn bottom_even_kernel_equals_center_when_rows_match() {
  // When prev == cur the vertical box blend is a no-op, so the bottom-even
  // kernel must reproduce the plain horizontal centered upsample exactly.
  let cur = [10u8, 40, 90, 30];
  let mut bottom = [0u8; 8];
  let mut center = [0u8; 8];
  crate::row::scalar::chroma_upsample_420_bottom_even_h(&cur, &cur, &mut bottom, 8);
  crate::row::scalar::chroma_upsample_2to1_center_h(&cur, &mut center, 8);
  assert_eq!(
    bottom, center,
    "prev == cur must collapse the vertical blend to the horizontal centered path"
  );
}

// ---- bottom-sited end-to-end correctness -----------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_rgb_matches_vblend_then_444_reference() {
  let (yp, up, vp) = vramp_yuv420p();

  // Reference: vertical-box-blend (even rows) + horizontal centered upsample to
  // full resolution, then the ordinary 4:4:4 decode.
  let (u444, v444) = ref_full_chroma_bottom(&up, &vp);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();

  assert_eq!(
    convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
    rgb_ref,
    "bottom-sited 4:2:0 RGB must equal vertical-blend + horizontal-upsample then 4:4:4"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_differs_from_center_and_default_vertically() {
  // On a purely-vertical chroma ramp, Bottom's even-row vertical blend must
  // move chroma both vs the horizontal-only Center (same horizontal phase, no
  // vertical) and vs the replicate default.
  let (yp, up, vp) = vramp_yuv420p();
  let bottom = convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp);
  assert_ne!(
    bottom,
    convert_rgb_with(ChromaLocation::Center, true, &yp, &up, &vp),
    "Bottom (v=1) must differ from Center (horizontal-only) on a vertical chroma ramp"
  );
  assert_ne!(
    bottom,
    convert_rgb_with(ChromaLocation::Left, true, &yp, &up, &vp),
    "Bottom must differ from the vertical-replicate default"
  );
  // Top box-blends the ODD row FORWARD (with the next chroma row) where Bottom
  // blends the EVEN row backward, so the two differ on this ramp.
  assert_ne!(
    bottom,
    convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
    "Top (forward, odd rows) must differ from Bottom's backward even-row blend"
  );
}

// ---- top-sited VERTICAL phase (RFC #238 forward one-row delay) --------------

/// Independent reference for the top-sited (`v = 0`) full reconstruction: per
/// luma row `r`, the EVEN rows take chroma row `r/2` directly (co-sited with the
/// top luma row); the ODD rows take the vertical box average of chroma rows
/// `r/2` and `r/2 + 1` (the FORWARD neighbour, clamped to `ch - 1` at the bottom
/// edge — so the trailing odd row collapses back to co-sited). Each half-row is
/// then horizontally upsampled — centered (`Top`, `h = 0.5`) or replicated
/// (`TopLeft`, `h = 0`). Written separately from the production kernels so it is
/// a true oracle for the forward one-row-delay reconstruction.
fn ref_full_chroma_top(u420: &[u8], v420: &[u8], centered: bool) -> (Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let up_h = |half: &[u8]| -> Vec<u8> {
    if centered {
      ref_upsample_center_h(half, w)
    } else {
      ref_upsample_cosited_h(half, w)
    }
  };
  let vblend = |plane: &[u8], cr: usize, next: usize| -> Vec<u8> {
    (0..cw)
      .map(|c| {
        let a = plane[cr * cw + c] as u32;
        let b = plane[next * cw + c] as u32;
        ((a + b + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  for r in 0..h {
    let cr = r / 2;
    let (uhalf, vhalf) = if r & 1 == 0 {
      (
        u420[cr * cw..cr * cw + cw].to_vec(),
        v420[cr * cw..cr * cw + cw].to_vec(),
      )
    } else {
      let next = (cr + 1).min(ch - 1);
      (vblend(u420, cr, next), vblend(v420, cr, next))
    };
    u444[r * w..r * w + w].copy_from_slice(&up_h(&uhalf));
    v444[r * w..r * w + w].copy_from_slice(&up_h(&vhalf));
  }
  (u444, v444)
}

/// End-to-end RGB reference for a top-sited siting: forward-reconstruct full
/// chroma, then the ordinary 4:4:4 decode.
fn top_ref_rgb(centered: bool, yp: &[u8], up: &[u8], vp: &[u8]) -> Vec<u8> {
  let (u444, v444) = ref_full_chroma_top(up, vp, centered);
  let ref_src = Yuv444pFrame::new(yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
  rgb_ref
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_rgb_matches_forward_reconstruct_reference() {
  // The core Top correctness proof: the streaming forward one-row-delay decode
  // must reproduce "reconstruct full chroma forward, then 4:4:4" bit-for-bit,
  // INCLUDING the trailing-odd-row bottom-edge clamp. Both SIMD tiers match.
  let (yp, up, vp) = vramp_yuv420p();
  let rgb_ref = top_ref_rgb(true, &yp, &up, &vp);
  assert_eq!(
    convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
    rgb_ref,
    "top-sited 4:2:0 RGB (SIMD) must equal forward-blend + centered-upsample then 4:4:4"
  );
  assert_eq!(
    convert_rgb_with(ChromaLocation::Top, false, &yp, &up, &vp),
    rgb_ref,
    "top-sited 4:2:0 RGB (scalar) must match the same reference"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn topleft_rgb_matches_cosited_forward_reference() {
  // TopLeft is the co-sited-horizontal (`h = 0`) twin of Top: same forward
  // vertical blend, plain 2x horizontal replicate instead of the centered phase.
  let (yp, up, vp) = vramp_yuv420p();
  let rgb_ref = top_ref_rgb(false, &yp, &up, &vp);
  assert_eq!(
    convert_rgb_with(ChromaLocation::TopLeft, true, &yp, &up, &vp),
    rgb_ref,
    "topleft-sited RGB (SIMD) must equal forward-blend + co-sited-replicate then 4:4:4"
  );
  assert_eq!(
    convert_rgb_with(ChromaLocation::TopLeft, false, &yp, &up, &vp),
    rgb_ref,
    "topleft-sited RGB (scalar) must match the same reference"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_differs_from_topleft_horizontally() {
  // On a purely-HORIZONTAL chroma ramp the shared forward vertical blend is a
  // no-op, so only the horizontal phase separates Top (`h = 0.5` centered) from
  // TopLeft (`h = 0` co-sited) — they must differ.
  let (yp, up, vp) = ramp_yuv420p();
  assert_ne!(
    convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::TopLeft, true, &yp, &up, &vp),
    "Top (h=0.5) must differ from TopLeft (h=0) on a horizontal chroma ramp"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_rgba_and_hsv_match_forward_reference() {
  // Exercise the RGBA-only and HSV-direct branches of the delayed colour decode:
  // both must reproduce the forward-reconstruct-then-4:4:4 reference.
  let (yp, up, vp) = vramp_yuv420p();
  let (u444, v444) = ref_full_chroma_top(&up, &vp, true);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);

  // RGBA-only.
  {
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut rs = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut rs).unwrap();

    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Top);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_eq!(rgba, rgba_ref, "Top RGBA must match the forward reference");
  }

  // HSV-direct (no RGB / RGBA attached).
  {
    let (mut hh, mut ss, mut vv) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut rs = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut rs).unwrap();

    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let (mut h2, mut s2, mut v2) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_hsv(&mut h2, &mut s2, &mut v2)
      .unwrap()
      .with_chroma_location(ChromaLocation::Top);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_eq!(
      (h2, s2, v2),
      (hh, ss, vv),
      "Top HSV-direct must match the forward reference"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_maintains_forward_delay_buffers() {
  // A top-sited colour decode grows BOTH the one-row chroma lookback (`c[i]` for
  // the odd flush) and the forward-delay Y buffer (the held odd row's luma).
  let (yp, up, vp) = vramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Top);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  assert_eq!(
    sink.chroma_prev.len(),
    W as usize,
    "Top maintains the width-byte (half U + half V) chroma lookback"
  );
  assert_eq!(
    sink.chroma_top_y.len(),
    W as usize,
    "Top allocates the forward-delay Y buffer"
  );
  drop(sink);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_survives_frame_reuse() {
  // Two frames through the SAME sink: the second frame's decode must NOT be
  // polluted by the first frame's held odd row or lookback (both reset in
  // begin_frame), so it equals a fresh forward-reconstruct reference.
  let (yp, up, vp) = vramp_yuv420p();
  let rgb_ref = top_ref_rgb(true, &yp, &up, &vp);
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Top);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  drop(sink);
  assert_eq!(
    rgb, rgb_ref,
    "second frame must match a fresh Top reconstruct"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_rgba_and_hsv_match_vblend_then_444_reference() {
  let (yp, up, vp) = vramp_yuv420p();
  let (u444, v444) = ref_full_chroma_bottom(&up, &vp);

  // RGBA-only path.
  {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Bottom);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(rgba, rgba_ref, "bottom RGBA must equal vblend-then-4:4:4");
  }

  // HSV-direct path (no RGB / RGBA attached).
  {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let (mut h, mut s, mut v) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap()
      .with_chroma_location(ChromaLocation::Bottom);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let (mut hr, mut sr, mut vr) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_hsv(&mut hr, &mut sr, &mut vr)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      (h, s, v),
      (hr, sr, vr),
      "bottom HSV must equal vblend-then-4:4:4"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_path_simd_matches_scalar() {
  let (yp, up, vp) = vramp_yuv420p();
  assert_eq!(
    convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::Bottom, false, &yp, &up, &vp),
    "bottom path must be bit-identical across the SIMD and scalar tiers"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_grows_chroma_prev_lookback() {
  let (yp, up, vp) = vramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Bottom);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  let prev_len = sink.chroma_prev.len();
  drop(sink);
  assert_eq!(
    prev_len, W as usize,
    "bottom-sited path stages a width-byte (half-width U+V) chroma lookback"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn non_bottom_sitings_do_not_grow_chroma_prev() {
  // Center / Left / Unspecified never touch the vertical lookback (the `Top`
  // forward blend now DOES maintain it, so it is covered by its own tests).
  for loc in [
    ChromaLocation::Center,
    ChromaLocation::Left,
    ChromaLocation::Unspecified,
  ] {
    let (yp, up, vp) = vramp_yuv420p();
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgb = std::vec![0u8; (W * H * 3) as usize];
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
    let prev_len = sink.chroma_prev.len();
    drop(sink);
    assert_eq!(
      prev_len, 0,
      "siting {loc:?} must not grow the vertical chroma lookback"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_chroma_prev_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // A bottom-sited decode whose vertical-lookback allocation fails must leave
  // EVERY output buffer — luma included — untouched: the lookback is reserved
  // (fallibly) BEFORE any output row is written.
  let (yp, up, vp) = vramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut luma = std::vec![0xABu8; (W * H) as usize];
  let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Bottom);

  super::super::arm_chroma_prev_alloc_failure();
  let err = yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "bottom chroma-lookback refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the bottom lookback alloc-failure path"
  );
  assert!(
    rgb.iter().all(|&b| b == 0xCD),
    "rgb must be untouched on the bottom lookback alloc-failure path"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_luma_only_chroma_prev_alloc_failure_leaves_luma_untouched() {
  use crate::resample::ResampleError;

  // A LUMA-ONLY bottom-sited row also stages the vertical lookback (the
  // always-maintain path), and its `reserve_420_chroma_prev` must run BEFORE the
  // luma write — so a lookback alloc refusal on a luma-only frame leaves luma
  // untouched (no partial write), exactly like the colour path.
  let (yp, up, vp) = vramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut luma = std::vec![0xABu8; (W * H) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_chroma_location(ChromaLocation::Bottom);

  super::super::arm_chroma_prev_alloc_failure();
  let err = yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "luma-only bottom lookback refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched when the luma-only bottom lookback reservation fails"
  );
}

// ---- bottom-sited lookback is row-order-safe (no STALE blend) ---------------

/// The RGB a `Yuv444p` decode produces for ONE row built from the centered
/// (horizontal-only, NO vertical) upsample of a given chroma row — the
/// well-defined "clamp" result a bottom-sited EVEN row must fall back to when
/// its vertical predecessor is not provably available. A genuine vertical blend
/// (with whatever stale data) would diverge from this on a vertically-varying
/// ramp.
fn ref_centered_row_rgb(
  yp: &[u8],
  up: &[u8],
  vp: &[u8],
  chroma_row: usize,
  out_row: usize,
) -> Vec<u8> {
  let w = W as usize;
  let cw = w / 2;
  let urow = ref_upsample_center_h(&up[chroma_row * cw..chroma_row * cw + cw], w);
  let vrow = ref_upsample_center_h(&vp[chroma_row * cw..chroma_row * cw + cw], w);
  // Decode just that one full-width 4:4:4 row.
  let mut rgb = std::vec![0u8; w * 3];
  crate::row::yuv_444_to_rgb_row(
    &yp[out_row * w..out_row * w + w],
    &urow,
    &vrow,
    &mut rgb,
    w,
    ColorMatrix::Bt601,
    false,
    true,
  );
  rgb
}

/// Drives a bottom-sited `Yuv420p` RGB decode by feeding an EXPLICIT sequence of
/// row indices through the public `process` API (no walker), so out-of-order /
/// skipped / replayed delivery can be exercised. Returns the full RGB plane;
/// rows never fed stay zero. `begin_frame` is called first (resetting the
/// vertical lookback).
fn drive_bottom_rows(yp: &[u8], up: &[u8], vp: &[u8], rows: &[usize]) -> Vec<u8> {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut rgb = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Bottom);
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  for &r in rows {
    let cr = r / 2;
    let row = Yuv420pRow::new(
      &yp[r * w..r * w + w],
      &up[cr * cw..cr * cw + cw],
      &vp[cr * cw..cr * cw + cw],
      r,
      ColorMatrix::Bt601,
      false,
    );
    crate::PixelSink::process(&mut sink, row).unwrap();
  }
  drop(sink);
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_even_row_without_valid_prev_clamps_not_stale() {
  // Feed ONLY row 2 (an even row, pair 1) — its vertical predecessor (chroma row
  // 0) was never staged, so the blend must CLAMP to the centered upsample of the
  // current chroma row (row 1), not blend with whatever happens to be in the
  // lookback buffer.
  let (yp, up, vp) = vramp_yuv420p();
  let w = W as usize;
  let rgb = drive_bottom_rows(&yp, &up, &vp, &[2]);
  let got_row2 = &rgb[2 * w * 3..3 * w * 3];
  let want = ref_centered_row_rgb(&yp, &up, &vp, 1, 2);
  assert_eq!(
    got_row2,
    &want[..],
    "an even row with no valid vertical predecessor must clamp to the centered \
     (horizontal-only) decode, never blend stale data"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_even_row_after_skip_clamps_not_stale() {
  // Feed rows 0, 1 (pair 0 → lookback now holds chroma row 0), then JUMP to row
  // 4 (even, pair 2) skipping pair 1. Row 4's predecessor is chroma row 1, but
  // the lookback holds the STALE chroma row 0; the validity tag (Some(0) !=
  // Some(2-1)) must force the clamp to the centered upsample of chroma row 2.
  let (yp, up, vp) = vramp_yuv420p();
  let w = W as usize;
  let rgb = drive_bottom_rows(&yp, &up, &vp, &[0, 1, 4]);
  let got_row4 = &rgb[4 * w * 3..5 * w * 3];
  let want = ref_centered_row_rgb(&yp, &up, &vp, 2, 4);
  assert_eq!(
    got_row4,
    &want[..],
    "an even row whose lookback holds a non-adjacent (stale) chroma row must \
     clamp, never blend the stale row"
  );
  // Sanity: the stale blend (chroma row 0 vs row 2) would be a DIFFERENT value,
  // so the clamp is observably not a coincidence on this vertical ramp.
  let stale = {
    let cw = w / 2;
    let mut e_u = std::vec![0u8; cw];
    let mut e_v = std::vec![0u8; cw];
    for c in 0..cw {
      e_u[c] = (((up[c] as u32) + (up[2 * cw + c] as u32) + 1) >> 1) as u8;
      e_v[c] = (((vp[c] as u32) + (vp[2 * cw + c] as u32) + 1) >> 1) as u8;
    }
    let urow = ref_upsample_center_h(&e_u, w);
    let vrow = ref_upsample_center_h(&e_v, w);
    let mut rgb = std::vec![0u8; w * 3];
    crate::row::yuv_444_to_rgb_row(
      &yp[4 * w..4 * w + w],
      &urow,
      &vrow,
      &mut rgb,
      w,
      ColorMatrix::Bt601,
      false,
      true,
    );
    rgb
  };
  assert_ne!(
    got_row4,
    &stale[..],
    "guard: the clamp must NOT coincide with the stale (row0xrow2) blend"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_two_frames_no_cross_frame_stale_blend() {
  // Frame 1 decoded fully in order (chroma plane A) leaves chroma row A[last] in
  // the lookback. After `begin_frame`, frame 2's row 0 (even, pair 0) has NO
  // predecessor and must clamp to the centered upsample of plane B's row 0 — it
  // must NOT box-blend B's row 0 with frame 1's last chroma row.
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let (ya, ua, va) = vramp_yuv420p();
  // Plane B: a distinct vertical ramp so a cross-frame blend would be visible.
  let yb = std::vec![128u8; w * h];
  let mut ub = std::vec![0u8; cw * ch];
  let mut vb = std::vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      ub[r * cw + c] = (200 - r * 30).max(16) as u8;
      vb[r * cw + c] = (30 + r * 30).min(240) as u8;
    }
  }

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Bottom);
  // Frame 1, in order.
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  for r in 0..h {
    let cr = r / 2;
    let row = Yuv420pRow::new(
      &ya[r * w..r * w + w],
      &ua[cr * cw..cr * cw + cw],
      &va[cr * cw..cr * cw + cw],
      r,
      ColorMatrix::Bt601,
      false,
    );
    crate::PixelSink::process(&mut sink, row).unwrap();
  }
  // Frame 2: begin_frame resets the lookback; feed only row 0 (plane B).
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  let row0 = Yuv420pRow::new(
    &yb[0..w],
    &ub[0..cw],
    &vb[0..cw],
    0,
    ColorMatrix::Bt601,
    false,
  );
  crate::PixelSink::process(&mut sink, row0).unwrap();
  drop(sink);

  let got_row0 = &rgb[0..w * 3];
  let want = ref_centered_row_rgb(&yb, &ub, &vb, 0, 0);
  assert_eq!(
    got_row0,
    &want[..],
    "frame 2 row 0 must clamp to plane B's centered decode, never blend frame 1's chroma"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_luma_only_then_late_color_box_blends() {
  // The always-maintain lookback: rows 0, 1 are processed LUMA-ONLY (no colour
  // attached), so the colour upsample helper never runs — yet the bottom-sited
  // lookback must still be staged through them. After attaching RGB via
  // `set_rgb`, row 2 (Bottom even, pair 1) must correctly box-blend chroma rows
  // 0 and 1 (== the all-output reference), NOT clamp to chroma row 1.
  let (yp, up, vp) = vramp_yuv420p();
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;

  // All-output reference: the full Bottom reconstruction decoded via Yuv444p.
  let (u444, v444) = ref_full_chroma_bottom(&up, &vp);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; w * h * 3];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(w, h)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
  drop(ref_sink);

  // Luma-only rows 0, 1, then attach RGB and decode rows 2, 3.
  let mut luma = std::vec![0u8; w * h];
  let mut rgb = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_chroma_location(ChromaLocation::Bottom);
    crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
    let feed = |sink: &mut MixedSinker<'_, Yuv420p>, r: usize| {
      let cr = r / 2;
      let row = Yuv420pRow::new(
        &yp[r * w..r * w + w],
        &up[cr * cw..cr * cw + cw],
        &vp[cr * cw..cr * cw + cw],
        r,
        ColorMatrix::Bt601,
        false,
      );
      crate::PixelSink::process(sink, row).unwrap();
    };
    feed(&mut sink, 0);
    feed(&mut sink, 1);
    // Late colour attach.
    sink.set_rgb(&mut rgb).unwrap();
    feed(&mut sink, 2);
    feed(&mut sink, 3);
  }

  // Row 2 (Bottom even) must equal the all-output reference's row 2 — proving
  // the lookback (staged through the luma-only rows 0, 1) drove the box blend.
  let got_row2 = &rgb[2 * w * 3..3 * w * 3];
  assert_eq!(
    got_row2,
    &rgb_ref[2 * w * 3..3 * w * 3],
    "a luma-only-then-late-colour row 2 must box-blend chroma rows 0,1 (all-output reference)"
  );
  // And it must NOT be the clamp (centered upsample of chroma row 1 only) — the
  // clamp is what an unmaintained lookback would have produced.
  let clamp = ref_centered_row_rgb(&yp, &up, &vp, 1, 2);
  assert_ne!(
    got_row2,
    &clamp[..],
    "row 2 must NOT clamp — the lookback was maintained through the luma-only rows"
  );
}

// ---- no-output rows are invisible to the lookback (no-output invariant) ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_no_output_row_does_not_grow_chroma_prev() {
  // A bottom-sited sink with NO outputs attached must honour the repo-wide
  // no-output invariant: every `process` call returns before the preflight, so
  // the vertical lookback is NEVER reserved (a no-output row must allocate
  // nothing — not even the lookback).
  let (yp, up, vp) = vramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_chroma_location(ChromaLocation::Bottom);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  let prev_len = sink.chroma_prev.len();
  drop(sink);
  assert_eq!(
    prev_len, 0,
    "a no-output bottom-sited row must not reserve the vertical lookback"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_no_output_rows_do_not_enable_late_color_blend() {
  // The no-output invariant's correctness twin: rows 0, 1 are processed with NO
  // outputs (invisible — they must not prime the lookback). After attaching RGB
  // via `set_rgb`, row 2 (Bottom even, pair 1) must NOT box-blend through those
  // invisible rows; with no legitimately-output predecessor staged, it CLAMPS to
  // the centered upsample of the current chroma row. (Contrast
  // `bottom_luma_only_then_late_color_box_blends`, where rows 0,1 ARE outputs —
  // luma — and so DO prime the lookback and the later row box-blends.)
  let (yp, up, vp) = vramp_yuv420p();
  let w = W as usize;
  let cw = w / 2;

  let mut rgb = std::vec![0u8; w * H as usize * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p>::new(w, H as usize).with_chroma_location(ChromaLocation::Bottom);
    crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
    let feed = |sink: &mut MixedSinker<'_, Yuv420p>, r: usize| {
      let cr = r / 2;
      let row = Yuv420pRow::new(
        &yp[r * w..r * w + w],
        &up[cr * cw..cr * cw + cw],
        &vp[cr * cw..cr * cw + cw],
        r,
        ColorMatrix::Bt601,
        false,
      );
      crate::PixelSink::process(sink, row).unwrap();
    };
    // Rows 0, 1: NO output attached — invisible, must not prime the lookback.
    feed(&mut sink, 0);
    feed(&mut sink, 1);
    // Late colour attach, then decode rows 2, 3.
    sink.set_rgb(&mut rgb).unwrap();
    feed(&mut sink, 2);
    feed(&mut sink, 3);
  }

  // Row 2 must CLAMP (centered upsample of chroma row 1), NOT box-blend chroma
  // rows 0,1 — because the predecessor (chroma row 0) only ever arrived through
  // invisible no-output rows that left the lookback unprimed.
  let got_row2 = &rgb[2 * w * 3..3 * w * 3];
  let clamp = ref_centered_row_rgb(&yp, &up, &vp, 1, 2);
  assert_eq!(
    got_row2,
    &clamp[..],
    "a no-output predecessor row must not enable a later colour even row to box-blend through it"
  );
  // Guard: the box-blend (the all-output reconstruction) is a DIFFERENT value,
  // so the clamp is observably the no-output-invisible behaviour, not a fluke.
  let (u444, v444) = ref_full_chroma_bottom(&up, &vp);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; w * H as usize * 3];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(w, H as usize)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
  drop(ref_sink);
  assert_ne!(
    got_row2,
    &rgb_ref[2 * w * 3..3 * w * 3],
    "guard: the clamp must differ from the box-blend, so the no-output rows were truly invisible"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "constructs an absurd geometry; the no-op contract is the point, not Miri"
)]
fn bottom_no_output_row_large_geometry_does_not_overflow() {
  // The no-output guard must run BEFORE the `idx * w` single-plane offset
  // arithmetic. A no-output `process` call never ran an attach-time
  // `w x h x 1` validation (nothing was attached), so on a 32-bit target
  // (`usize == u32`) an absurd geometry where `idx * w` exceeds `u32::MAX`
  // would overflow that offset and panic under overflow checks. With no outputs
  // attached, `process` must return `Ok(())` having done NO row math and NO
  // allocation. On 64-bit (`idx * w` fits `u64`) this documents the
  // no-output-is-a-pure-no-op contract and trivially passes; on the i686 /
  // overflow-check CI it catches a regression if the guard ever moves back below
  // the arithmetic.
  //
  // w = 4, idx = 2^30 → idx * w = 2^32 = u32::MAX + 1 (overflows u32).
  let w: usize = 4;
  let idx: usize = 1 << 30; // 1_073_741_824
  let h: usize = idx + 1; // idx < height so the row-index check passes
  assert!(
    (idx as u64) * (w as u64) > u32::MAX as u64,
    "test geometry must exceed u32::MAX to exercise the 32-bit offset overflow"
  );

  let y = std::vec![128u8; w];
  let c = std::vec![128u8; w / 2];
  let mut sink = MixedSinker::<Yuv420p>::new(w, h).with_chroma_location(ChromaLocation::Bottom);
  // No outputs attached: the guard returns before `idx * w` (no overflow panic)
  // and before the bottom preflight (no allocation).
  let row = Yuv420pRow::new(&y, &c, &c, idx, ColorMatrix::Bt601, false);
  crate::PixelSink::process(&mut sink, row).unwrap();
  let prev_len = sink.chroma_prev.len();
  drop(sink);
  assert_eq!(
    prev_len, 0,
    "a no-output large-geometry row must allocate nothing (lookback included)"
  );
}

// ---- bottom-LEFT-sited phase (co-sited h = 0 + bottom v = 1) ----------------

/// Independent reference for the co-sited (`h = 0`) horizontal upsample — a plain
/// 2× nearest-neighbor replicate. Written separately from the production kernel so
/// it is a real oracle.
fn ref_upsample_cosited_h(c_half: &[u8], width: usize) -> Vec<u8> {
  let half = width / 2;
  let mut out = std::vec![0u8; width];
  for j in 0..half {
    out[2 * j] = c_half[j];
    out[2 * j + 1] = c_half[j];
  }
  out
}

/// Independent reference for the bottom-left-sited (`h = 0`, `v = 1`) full
/// reconstruction: the [`ref_full_chroma_bottom`] vertical box blend, but fed to
/// the CO-SITED horizontal replicate ([`ref_upsample_cosited_h`]) instead of the
/// centered `1/4`–`3/4` weights. Written separately from the production kernels so
/// it is a true oracle.
fn ref_full_chroma_bottomleft(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  let vblend = |plane: &[u8], cr: usize, prev: usize| -> Vec<u8> {
    (0..cw)
      .map(|c| {
        let a = plane[prev * cw + c] as u32;
        let b = plane[cr * cw + c] as u32;
        ((a + b + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  for r in 0..h {
    let cr = r / 2;
    let (uhalf, vhalf) = if r & 1 == 0 {
      let prev = cr.saturating_sub(1);
      (vblend(u420, cr, prev), vblend(v420, cr, prev))
    } else {
      (
        u420[cr * cw..cr * cw + cw].to_vec(),
        v420[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    let urow = ref_upsample_cosited_h(&uhalf, w);
    let vrow = ref_upsample_cosited_h(&vhalf, w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

#[test]
fn cosited_upsample_kernel_matches_hand_computed() {
  // Co-sited horizontal reconstruction is a plain 2× replicate.
  let c_half = [10u8, 20, 30, 40];
  let mut out = [0u8; 8];
  crate::row::scalar::chroma_upsample_2to1_cosited_h(&c_half, &mut out, 8);
  assert_eq!(out, [10, 10, 20, 20, 30, 30, 40, 40]);
}

#[test]
fn bottomleft_even_kernel_matches_hand_computed() {
  // prev = [0, 0, 100, 100], cur = [40, 40, 60, 60]; vertical box blend
  // e = (prev + cur + 1) >> 1 = [20, 20, 80, 80], then the co-sited 2× replicate.
  let prev = [0u8, 0, 100, 100];
  let cur = [40u8, 40, 60, 60];
  let mut out = [0u8; 8];
  crate::row::scalar::chroma_upsample_420_bottomleft_even_h(&prev, &cur, &mut out, 8);
  assert_eq!(out, [20, 20, 20, 20, 80, 80, 80, 80]);
}

#[test]
fn bottomleft_even_kernel_equals_cosited_when_rows_match() {
  // When prev == cur the vertical box blend is a no-op, so the bottom-left-even
  // kernel must reproduce the plain co-sited replicate exactly.
  let cur = [10u8, 40, 90, 30];
  let mut bl = [0u8; 8];
  let mut cosited = [0u8; 8];
  crate::row::scalar::chroma_upsample_420_bottomleft_even_h(&cur, &cur, &mut bl, 8);
  crate::row::scalar::chroma_upsample_2to1_cosited_h(&cur, &mut cosited, 8);
  assert_eq!(
    bl, cosited,
    "prev == cur must collapse the vertical blend to the co-sited replicate"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_rgb_matches_cosited_vblend_then_444_reference() {
  let (yp, up, vp) = vramp_yuv420p();

  // Reference: co-sited-horizontal + vertical-box-blend (even rows) to full
  // resolution, then the ordinary 4:4:4 decode.
  let (u444, v444) = ref_full_chroma_bottomleft(&up, &vp);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();

  assert_eq!(
    convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp),
    rgb_ref,
    "bottom-left 4:2:0 RGB must equal co-sited-replicate + vertical-blend then 4:4:4"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_differs_from_bottom_horizontally() {
  // On a purely-HORIZONTAL chroma ramp, BottomLeft (co-sited h) must differ from
  // Bottom (centered h) — the vertical fold is identical, only the horizontal
  // phase differs.
  let (yp, up, vp) = ramp_yuv420p();
  assert_ne!(
    convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
    "BottomLeft (h=0) must differ from Bottom (h=0.5) on a horizontal chroma ramp"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_differs_from_topleft_vertically() {
  // On a purely-VERTICAL chroma ramp, BottomLeft (`v = 1`, backward even-row
  // blend) must differ from TopLeft (`v = 0`, forward odd-row blend) — same
  // co-sited horizontal phase, only the vertical fold DIRECTION differs — and
  // from the vertical-replicate default.
  let (yp, up, vp) = vramp_yuv420p();
  let bl = convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp);
  assert_ne!(
    bl,
    convert_rgb_with(ChromaLocation::TopLeft, true, &yp, &up, &vp),
    "BottomLeft (v=1 backward) must differ from TopLeft (v=0 forward) on a vertical ramp"
  );
  assert_ne!(
    bl,
    convert_rgb_with(ChromaLocation::Left, true, &yp, &up, &vp),
    "BottomLeft must differ from the vertical-replicate default"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_rgba_and_hsv_match_cosited_vblend_then_444_reference() {
  let (yp, up, vp) = vramp_yuv420p();
  let (u444, v444) = ref_full_chroma_bottomleft(&up, &vp);

  // RGBA-only path.
  {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::BottomLeft);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      rgba, rgba_ref,
      "bottom-left RGBA must equal cosited-vblend-then-4:4:4"
    );
  }

  // HSV-direct path (no RGB / RGBA attached).
  {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let (mut h, mut s, mut v) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap()
      .with_chroma_location(ChromaLocation::BottomLeft);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let (mut hr, mut sr, mut vr) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_hsv(&mut hr, &mut sr, &mut vr)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      (h, s, v),
      (hr, sr, vr),
      "bottom-left HSV must equal cosited-vblend-then-4:4:4"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_path_simd_matches_scalar() {
  let (yp, up, vp) = vramp_yuv420p();
  assert_eq!(
    convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::BottomLeft, false, &yp, &up, &vp),
    "bottom-left path must be bit-identical across the SIMD and scalar tiers"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_equals_cosited_on_flat_chroma() {
  // On spatially FLAT chroma the vertical blend is a no-op and the co-sited
  // horizontal phase is the default nearest-neighbor decode, so BottomLeft
  // collapses to the byte-identical co-sited decode.
  let w = W as usize;
  let h = H as usize;
  let yp = std::vec![128u8; w * h];
  let up = std::vec![90u8; (w / 2) * (h / 2)];
  let vp = std::vec![150u8; (w / 2) * (h / 2)];
  assert_eq!(
    convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::Left, true, &yp, &up, &vp),
    "flat-chroma BottomLeft must collapse to the co-sited decode"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottomleft_grows_chroma_prev_lookback() {
  let (yp, up, vp) = vramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::BottomLeft);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  let prev_len = sink.chroma_prev.len();
  drop(sink);
  assert_eq!(
    prev_len, W as usize,
    "bottom-left-sited path stages a width-byte (half-width U+V) chroma lookback"
  );
}
