//! Chroma-siting-aware 4:4:0 identity decode for `Yuv440p` (RFC #238 S8b).
//!
//! 4:4:0 keeps FULL-width chroma, subsampled 2:1 **vertically only**, so the
//! siting reduces to its vertical axis alone: `Bottom`
//! ([`chroma_440_bottom_sited_v`](super::super::chroma_440_bottom_sited_v),
//! `v = 1`) box-blends the even output row's full-width chroma with the previous
//! chroma row, while every co-sited / horizontal siting keeps the byte-identical
//! vertical-replicate decode. There is NO horizontal phase (chroma is already
//! full width), so — unlike 4:2:0 — nothing here reconstructs horizontally.
//!
//! Covers: the full-width vertical kernel against a hand-computed oracle; the
//! default / co-sited path staying byte-identical (the regression guard — only
//! `Bottom` is new); the `Bottom` RGB / RGBA / HSV identity decodes matching an
//! independent "vertical-blend-then-4:4:4" reference; SIMD-vs-scalar parity; and
//! the row-order safety of the one-row chroma lookback (no stale / cross-frame
//! blend).

use super::*;
use crate::ChromaLocation;

const W: u32 = 8;
const H: u32 = 8;

/// A `Yuv440p` frame (full-width chroma, `ch = h / 2` rows) with flat luma and a
/// per-ROW chroma step (flat across columns), so the VERTICAL chroma phase is
/// observable in isolation: a co-sited decode leaves it untouched, the `v = 1`
/// blend visibly moves it.
fn vramp_yuv440p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let ch = h / 2;
  let y = std::vec![128u8; w * h];
  let mut u = std::vec![0u8; w * ch];
  let mut v = std::vec![0u8; w * ch];
  for r in 0..ch {
    for c in 0..w {
      u[r * w + c] = (20 + r * 40).min(240) as u8;
      v[r * w + c] = (220u32.saturating_sub((r * 40) as u32)).max(16) as u8;
    }
  }
  (y, u, v)
}

/// A `Yuv440p` frame with BOTH a horizontal and vertical chroma ramp — a general
/// correctness fixture that exercises the full-width copy on odd rows and the
/// full-width blend on even rows.
fn ramp_yuv440p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let ch = h / 2;
  let mut y = std::vec![0u8; w * h];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i as u32 * 3) % 160) as u8;
  }
  let mut u = std::vec![0u8; w * ch];
  let mut v = std::vec![0u8; w * ch];
  for r in 0..ch {
    for c in 0..w {
      u[r * w + c] = (16 + c * 8 + r * 40).min(240) as u8;
      v[r * w + c] = (240u32.saturating_sub((c * 6 + r * 30) as u32)).max(16) as u8;
    }
  }
  (y, u, v)
}

/// A flat-chroma fixture: the vertical blend of a constant is that constant, so
/// `Bottom` must equal co-sited byte-for-byte. Luma still varies.
fn flat_chroma_yuv440p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let ch = h / 2;
  let mut y = std::vec![0u8; w * h];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i as u32 * 7) % 170) as u8;
  }
  (y, std::vec![110u8; w * ch], std::vec![140u8; w * ch])
}

/// Independent reference for the bottom-sited (`v = 1`) full-height chroma
/// reconstruction: per luma row `r`, the EVEN rows take the full-width vertical
/// box average of chroma rows `r/2 - 1` (clamped to `r/2` at the top edge) and
/// `r/2`; the ODD rows take chroma row `r/2` directly. NO horizontal
/// reconstruction (4:4:0 chroma is already full width). Written separately from
/// the production kernel so it is a true oracle. Feeding these full-resolution
/// planes to a `Yuv444p` conversion is the end-to-end oracle for `Bottom`.
fn ref_full_chroma_bottom(u440: &[u8], v440: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  let vblend = |plane: &[u8], cr: usize, prev: usize| -> Vec<u8> {
    (0..w)
      .map(|c| {
        let a = plane[prev * w + c] as u32;
        let b = plane[cr * w + c] as u32;
        ((a + b + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  for r in 0..h {
    let cr = r / 2;
    let (urow, vrow) = if r & 1 == 0 {
      let prev = cr.saturating_sub(1);
      (vblend(u440, cr, prev), vblend(v440, cr, prev))
    } else {
      (
        u440[cr * w..cr * w + w].to_vec(),
        v440[cr * w..cr * w + w].to_vec(),
      )
    };
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

fn convert_rgb_with(loc: ChromaLocation, simd: bool, yp: &[u8], up: &[u8], vp: &[u8]) -> Vec<u8> {
  let src = Yuv440pFrame::new(yp, up, vp, W, H, W, W, W);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(loc.clone())
    .with_simd(simd);
  yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  rgb
}

// ---- full-width vertical kernel oracle -------------------------------------

#[test]
fn bottom_v_kernel_matches_hand_computed() {
  // prev = [0, 40, 200, 100], cur = [40, 80, 100, 100] (width 4). Full-width
  // vertical box blend out[j] = (prev[j] + cur[j] + 1) >> 1.
  let prev = [0u8, 40, 200, 100];
  let cur = [40u8, 80, 100, 100];
  let mut out = [0u8; 4];
  crate::row::scalar::chroma_upsample_440_bottom_v(&prev, &cur, &mut out, 4);
  assert_eq!(out, [20, 60, 150, 100]);
}

#[test]
fn bottom_v_kernel_equals_passthrough_when_rows_match() {
  // When prev == cur the vertical box blend is a no-op, so the kernel must
  // reproduce a straight copy of the current chroma row exactly.
  let cur = [10u8, 40, 90, 30];
  let mut out = [0u8; 4];
  crate::row::scalar::chroma_upsample_440_bottom_v(&cur, &cur, &mut out, 4);
  assert_eq!(
    out, cur,
    "prev == cur must collapse the vertical blend to a straight copy"
  );
}

// ---- default / co-sited path is byte-identical (regression guard) ----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn default_and_cosited_and_horizontal_sitings_are_byte_identical() {
  // 4:4:0 has no horizontal phase and folds only its VERTICAL axis, so `Bottom`
  // (v=1) and `Top` (v=0) reconstruct while the co-sited (`Left`) and
  // horizontally-centered but vertically-central (`Center`, v=0.5) sitings keep the
  // exact vertical-replicate decode, bit-for-bit equal to the Unspecified baseline
  // even though the chroma plane is a non-trivial ramp. (`Top` / `TopLeft` now fold
  // the forward `v=0` triangle and left this group — covered by the Top tests.)
  let (yp, up, vp) = ramp_yuv440p();
  let baseline = convert_rgb_with(ChromaLocation::Unspecified, true, &yp, &up, &vp);
  for loc in [
    ChromaLocation::Unspecified,
    ChromaLocation::other("unassigned-99"),
    ChromaLocation::Left,
    ChromaLocation::Center,
  ] {
    assert_eq!(
      convert_rgb_with(loc.clone(), true, &yp, &up, &vp),
      baseline,
      "siting {loc:?} must keep the byte-identical co-sited 4:4:0 decode"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_left_decodes_as_bottom() {
  // 4:4:0 has no horizontal phase, so `BottomLeft` (h=0, v=1) folds the SAME
  // vertical phase as `Bottom` (h=0.5, v=1) and must decode byte-for-byte
  // identically. On a vertical ramp BOTH must differ from the co-sited decode, so
  // this cannot be vacuously satisfied by a `BottomLeft` that stayed co-sited.
  let (yp, up, vp) = vramp_yuv440p();
  let bottom = convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp);
  let bottom_left = convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp);
  let cosited = convert_rgb_with(ChromaLocation::Left, true, &yp, &up, &vp);
  assert_eq!(
    bottom_left, bottom,
    "4:4:0 BottomLeft must fold the identical vertical phase as Bottom"
  );
  assert_ne!(
    bottom_left, cosited,
    "BottomLeft must vertically fold, not take the co-sited decode"
  );
}

// ---- bottom-sited end-to-end correctness -----------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_rgb_matches_vblend_then_444_reference() {
  let (yp, up, vp) = ramp_yuv440p();

  // Reference: vertical-box-blend (even rows) / passthrough (odd rows) to full
  // resolution, then the ordinary 4:4:4 decode.
  let (u444, v444) = ref_full_chroma_bottom(&up, &vp);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(
    &ref_src,
    false,
    ref_sink.set_kernel_matrix(KernelMatrix::Bt601),
  )
  .unwrap();

  assert_eq!(
    convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
    rgb_ref,
    "bottom-sited 4:4:0 RGB must equal vertical-blend then 4:4:4"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_differs_from_cosited_on_vertical_ramp() {
  // On a purely-vertical chroma ramp, Bottom's even-row vertical blend must move
  // chroma vs the co-sited / replicate default and vs the horizontal-only
  // sitings (which are all co-sited vertically for 4:4:0).
  let (yp, up, vp) = vramp_yuv440p();
  let bottom = convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp);
  for loc in [
    ChromaLocation::Left,
    ChromaLocation::Center,
    ChromaLocation::Top,
    ChromaLocation::Unspecified,
  ] {
    assert_ne!(
      bottom,
      convert_rgb_with(loc.clone(), true, &yp, &up, &vp),
      "Bottom (v=1) must differ from {loc:?} on a vertical chroma ramp"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_equals_cosited_on_flat_chroma() {
  // On constant chroma the vertical blend is a no-op, so Bottom collapses to the
  // co-sited decode byte-for-byte (the phase machinery corrupts nothing).
  let (yp, up, vp) = flat_chroma_yuv440p();
  assert_eq!(
    convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::Left, true, &yp, &up, &vp),
    "flat-chroma Bottom must equal co-sited"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_rgba_and_hsv_match_vblend_then_444_reference() {
  let (yp, up, vp) = ramp_yuv440p();
  let (u444, v444) = ref_full_chroma_bottom(&up, &vp);

  // RGBA-only path.
  {
    let src = Yuv440pFrame::new(&yp, &up, &vp, W, H, W, W, W);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Bottom);
    yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuv444p_to(
      &ref_src,
      false,
      ref_sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
    assert_eq!(rgba, rgba_ref, "bottom RGBA must equal vblend-then-4:4:4");
  }

  // HSV-direct path (no RGB / RGBA attached).
  {
    let src = Yuv440pFrame::new(&yp, &up, &vp, W, H, W, W, W);
    let (mut h, mut s, mut v) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap()
      .with_chroma_location(ChromaLocation::Bottom);
    yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let (mut hr, mut sr, mut vr) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_hsv(&mut hr, &mut sr, &mut vr)
      .unwrap();
    yuv444p_to(
      &ref_src,
      false,
      ref_sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
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
  let (yp, up, vp) = ramp_yuv440p();
  assert_eq!(
    convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::Bottom, false, &yp, &up, &vp),
    "bottom path must be bit-identical across the SIMD and scalar tiers"
  );
}

// ---- lookback growth / no-output invariant ---------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_grows_full_width_chroma_prev_lookback() {
  let (yp, up, vp) = vramp_yuv440p();
  let src = Yuv440pFrame::new(&yp, &up, &vp, W, H, W, W, W);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Bottom);
  yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  let prev_len = sink.chroma_prev.len();
  drop(sink);
  assert_eq!(
    prev_len,
    2 * W as usize,
    "bottom-sited 4:4:0 stages a full-width (2·w) U+V chroma lookback"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn non_bottom_sitings_do_not_grow_chroma_prev() {
  // Center / Left never touch the vertical lookback (no vertical fold). `Top`
  // (v=0) now DOES maintain the lookback for its forward one-row delay, so it is
  // excluded here and covered by the Top tests.
  for loc in [
    ChromaLocation::Center,
    ChromaLocation::Left,
    ChromaLocation::Unspecified,
  ] {
    let (yp, up, vp) = vramp_yuv440p();
    let src = Yuv440pFrame::new(&yp, &up, &vp, W, H, W, W, W);
    let mut rgb = std::vec![0u8; (W * H * 3) as usize];
    let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc.clone());
    yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn bottom_no_output_row_does_not_grow_chroma_prev() {
  // A bottom-sited sink with NO outputs attached must honour the repo-wide
  // no-output invariant: every `process` call returns before the preflight, so
  // the vertical lookback is NEVER reserved.
  let (yp, up, vp) = vramp_yuv440p();
  let src = Yuv440pFrame::new(&yp, &up, &vp, W, H, W, W, W);
  let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
    .with_chroma_location(ChromaLocation::Bottom);
  yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  let prev_len = sink.chroma_prev.len();
  drop(sink);
  assert_eq!(
    prev_len, 0,
    "a no-output bottom-sited row must not reserve the vertical lookback"
  );
}

// ---- bottom-sited lookback is row-order-safe (no STALE blend) ---------------

/// The RGB a `Yuv444p` decode produces for ONE row built from the co-sited
/// (straight-copy, NO vertical blend) full-width chroma of a given chroma row —
/// the "clamp" result a bottom-sited EVEN row must fall back to when its vertical
/// predecessor is not provably available. A genuine vertical blend (with stale
/// data) would diverge from this on a vertically-varying ramp.
fn ref_cosited_row_rgb(
  yp: &[u8],
  up: &[u8],
  vp: &[u8],
  chroma_row: usize,
  out_row: usize,
) -> Vec<u8> {
  let w = W as usize;
  let urow = &up[chroma_row * w..chroma_row * w + w];
  let vrow = &vp[chroma_row * w..chroma_row * w + w];
  let mut rgb = std::vec![0u8; w * 3];
  crate::row::yuv_444_to_rgb_row(
    &yp[out_row * w..out_row * w + w],
    urow,
    vrow,
    &mut rgb,
    w,
    KernelMatrix::Bt601,
    false,
    true,
  );
  rgb
}

/// Drives a bottom-sited `Yuv440p` RGB decode over an EXPLICIT sequence of row
/// indices through the public `process` API (no walker), so out-of-order /
/// skipped / replayed delivery can be exercised. `begin_frame` runs first
/// (resetting the vertical lookback).
fn drive_bottom_rows(yp: &[u8], up: &[u8], vp: &[u8], rows: &[usize]) -> Vec<u8> {
  let w = W as usize;
  let h = H as usize;
  let mut rgb = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Yuv440p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Bottom);
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  for &r in rows {
    let cr = r / 2;
    let row = Yuv440pRow::for_tests(
      &yp[r * w..r * w + w],
      &up[cr * w..cr * w + w],
      &vp[cr * w..cr * w + w],
      r,
      KernelMatrix::Bt601,
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
  // 0) was never staged, so the blend must CLAMP to a straight copy of the
  // current chroma row (row 1), not blend with whatever is in the lookback.
  let (yp, up, vp) = vramp_yuv440p();
  let w = W as usize;
  let rgb = drive_bottom_rows(&yp, &up, &vp, &[2]);
  let got_row2 = &rgb[2 * w * 3..3 * w * 3];
  let want = ref_cosited_row_rgb(&yp, &up, &vp, 1, 2);
  assert_eq!(
    got_row2,
    &want[..],
    "an even row with no valid vertical predecessor must clamp to the co-sited decode"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bottom_two_frames_no_cross_frame_stale_blend() {
  // Frame 1 decoded fully in order leaves plane A's last chroma row in the
  // lookback. After `begin_frame`, frame 2's row 0 (even, pair 0) has NO
  // predecessor and must clamp to plane B's row 0 — NOT box-blend B's row 0 with
  // frame 1's last chroma row.
  let w = W as usize;
  let h = H as usize;
  let ch = h / 2;
  let (ya, ua, va) = vramp_yuv440p();
  // Plane B: a distinct vertical ramp so a cross-frame blend would be visible.
  let yb = std::vec![128u8; w * h];
  let mut ub = std::vec![0u8; w * ch];
  let mut vb = std::vec![0u8; w * ch];
  for r in 0..ch {
    for c in 0..w {
      ub[r * w + c] = (200u32.saturating_sub((r * 30) as u32)).max(16) as u8;
      vb[r * w + c] = (30 + r * 30).min(240) as u8;
    }
  }

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Yuv440p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Bottom);
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  for r in 0..h {
    let cr = r / 2;
    let row = Yuv440pRow::for_tests(
      &ya[r * w..r * w + w],
      &ua[cr * w..cr * w + w],
      &va[cr * w..cr * w + w],
      r,
      KernelMatrix::Bt601,
      false,
    );
    crate::PixelSink::process(&mut sink, row).unwrap();
  }
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  let row0 = Yuv440pRow::for_tests(
    &yb[0..w],
    &ub[0..w],
    &vb[0..w],
    0,
    KernelMatrix::Bt601,
    false,
  );
  crate::PixelSink::process(&mut sink, row0).unwrap();
  drop(sink);

  let got_row0 = &rgb[0..w * 3];
  let want = ref_cosited_row_rgb(&yb, &ub, &vb, 0, 0);
  assert_eq!(
    got_row0,
    &want[..],
    "frame 2 row 0 must clamp to plane B's co-sited decode, never blend frame 1's chroma"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn direct_path_mid_frame_siting_flip_is_rejected() {
  // The identity (no-resample) decode freezes the vertical siting on its first
  // output-bearing row, exactly like the resample tiers. `set_chroma_location` is
  // public, so flipping co-sited ⇆ Bottom mid-frame must reject the second
  // in-sequence row with `ChromaSitingChanged` rather than silently emit a mixture
  // of co-sited and bottom-folded chroma (and never leave the vertical lookback in
  // a stale, gap-advanced state).
  use super::super::MixedSinkerError;
  let (yp, up, vp) = vramp_yuv440p();
  let w = W as usize;
  let h = H as usize;
  for (loc1, loc2) in [
    (ChromaLocation::Bottom, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Center),
    (ChromaLocation::Center, ChromaLocation::Bottom),
    // `BottomLeft` is also v=1 for 4:4:0 — flipping it against a co-sited siting
    // changes the vertical fold and must reject too.
    (ChromaLocation::BottomLeft, ChromaLocation::Left),
    (ChromaLocation::Top, ChromaLocation::BottomLeft),
    // `Top` / `TopLeft` (v=0) now fold the forward triangle, so flipping to / from
    // a co-sited or `Bottom` siting must reject too.
    (ChromaLocation::Top, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Top),
    (ChromaLocation::Top, ChromaLocation::Center),
    (ChromaLocation::Top, ChromaLocation::Bottom),
    (ChromaLocation::TopLeft, ChromaLocation::Left),
  ] {
    let mut rgb = std::vec![0u8; w * h * 3];
    let mut sink = MixedSinker::<Yuv440p>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc1.clone());
    crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
    // Row 0 with the first siting freezes the vertical phase.
    let row0 = Yuv440pRow::for_tests(
      &yp[0..w],
      &up[0..w],
      &vp[0..w],
      0,
      KernelMatrix::Bt601,
      false,
    );
    crate::PixelSink::process(&mut sink, row0).unwrap();
    // Flip the siting and deliver the next in-sequence row.
    sink.set_chroma_location(loc2.clone());
    let row1 = Yuv440pRow::for_tests(
      &yp[w..2 * w],
      &up[0..w],
      &vp[0..w],
      1,
      KernelMatrix::Bt601,
      false,
    );
    let err = crate::PixelSink::process(&mut sink, row1).unwrap_err();
    drop(sink);
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "direct path {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
  }
}

// ============ Top-sited (v = 0) FORWARD one-row delay =======================

/// Independent reference for the top-sited (`v = 0`) full-height chroma
/// reconstruction — the FORWARD mirror of [`ref_full_chroma_bottom`]: per luma row
/// `r`, the EVEN rows take chroma row `r/2` directly (co-sited with the pair's TOP
/// luma); the ODD rows take the full-width vertical box average of chroma rows
/// `r/2` and `r/2 + 1` (clamped to `r/2` at the bottom edge). `ch = ceil(h / 2)`.
/// Feeding these full-resolution planes to a `Yuv444p` conversion is the
/// end-to-end oracle for `Top`.
fn ref_full_chroma_top_hw(
  u440: &[u8],
  v440: &[u8],
  w: usize,
  h: usize,
  ch: usize,
) -> (Vec<u8>, Vec<u8>) {
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  let vblend = |plane: &[u8], a: usize, b: usize| -> Vec<u8> {
    (0..w)
      .map(|c| {
        let x = plane[a * w + c] as u32;
        let y = plane[b * w + c] as u32;
        ((x + y + 1) >> 1) as u8
      })
      .collect::<Vec<u8>>()
  };
  for r in 0..h {
    let cr = r / 2;
    let (urow, vrow) = if r & 1 == 0 {
      (
        u440[cr * w..cr * w + w].to_vec(),
        v440[cr * w..cr * w + w].to_vec(),
      )
    } else {
      let next = (cr + 1).min(ch - 1);
      (vblend(u440, cr, next), vblend(v440, cr, next))
    };
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

fn ref_full_chroma_top(u440: &[u8], v440: &[u8]) -> (Vec<u8>, Vec<u8>) {
  ref_full_chroma_top_hw(u440, v440, W as usize, H as usize, (H / 2) as usize)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_rgb_matches_vfold_then_444_reference() {
  let (yp, up, vp) = ramp_yuv440p();
  let (u444, v444) = ref_full_chroma_top(&up, &vp);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(
    &ref_src,
    false,
    ref_sink.set_kernel_matrix(KernelMatrix::Bt601),
  )
  .unwrap();
  assert_eq!(
    convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
    rgb_ref,
    "top-sited 4:4:0 RGB must equal forward-vfold then 4:4:4"
  );
  // `TopLeft` is v=0 for 4:4:0 (no horizontal phase), so it decodes identically.
  assert_eq!(
    convert_rgb_with(ChromaLocation::TopLeft, true, &yp, &up, &vp),
    rgb_ref,
    "TopLeft must fold the identical vertical phase as Top for 4:4:0"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_rgba_and_hsv_match_vfold_then_444_reference() {
  let (yp, up, vp) = ramp_yuv440p();
  let (u444, v444) = ref_full_chroma_top(&up, &vp);

  // RGBA-only path.
  {
    let src = Yuv440pFrame::new(&yp, &up, &vp, W, H, W, W, W);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Top);
    yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuv444p_to(
      &ref_src,
      false,
      ref_sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
    assert_eq!(
      rgba, rgba_ref,
      "top RGBA must equal forward-vfold-then-4:4:4"
    );
  }

  // HSV-direct path (no RGB / RGBA attached).
  {
    let src = Yuv440pFrame::new(&yp, &up, &vp, W, H, W, W, W);
    let (mut h, mut s, mut v) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap()
      .with_chroma_location(ChromaLocation::Top);
    yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let (mut hr, mut sr, mut vr) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_hsv(&mut hr, &mut sr, &mut vr)
      .unwrap();
    yuv444p_to(
      &ref_src,
      false,
      ref_sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
    assert_eq!(
      (h, s, v),
      (hr, sr, vr),
      "top HSV must equal forward-vfold-then-4:4:4"
    );
  }
}

/// Drives a Top RGB decode over a full in-order frame of arbitrary `w x h` through
/// the walker, and the independent forward-vfold-then-4:4:4 reference for the same
/// planes, returning `(got, want)`. Exercises the two-row FINAL flush (odd height,
/// last row even) and the trailing-odd clamp (even height, last row odd).
fn top_rgb_and_ref_hw(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
  let ch = h.div_ceil(2);
  let mut y = std::vec![0u8; w * h];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i as u32 * 3) % 160) as u8;
  }
  let mut u = std::vec![0u8; w * ch];
  let mut v = std::vec![0u8; w * ch];
  for r in 0..ch {
    for c in 0..w {
      u[r * w + c] = (16 + c * 8 + r * 40).min(240) as u8;
      v[r * w + c] = (240u32.saturating_sub((c * 6 + r * 30) as u32)).max(16) as u8;
    }
  }
  let src = Yuv440pFrame::new(&y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32);
  let mut rgb = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Yuv440p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Top);
  yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  drop(sink);

  let (u444, v444) = ref_full_chroma_top_hw(&u, &v, w, h, ch);
  let ref_src = Yuv444pFrame::new(
    &y, &u444, &v444, w as u32, h as u32, w as u32, w as u32, w as u32,
  );
  let mut rgb_ref = std::vec![0u8; w * h * 3];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(w, h)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(
    &ref_src,
    false,
    ref_sink.set_kernel_matrix(KernelMatrix::Bt601),
  )
  .unwrap();
  (rgb, rgb_ref)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_even_and_odd_height_two_row_flush_match_reference() {
  // Even heights end on the trailing-odd clamp; odd heights end on the final even
  // row's TWO-row flush (held odd + current even). Both must equal the oracle.
  for (w, h) in [(8, 8), (8, 7), (6, 5), (4, 3), (4, 1)] {
    let (got, want) = top_rgb_and_ref_hw(w, h);
    assert_eq!(
      got, want,
      "Top {w}x{h} must equal the forward-vfold-then-4:4:4 oracle"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_differs_from_bottom_and_cosited_on_vertical_ramp() {
  // On a purely-vertical chroma ramp, Top's odd-row forward blend must move chroma
  // vs the co-sited default, vs the horizontal-only co-sited sitings, and vs the
  // BACKWARD-folded Bottom (Top != Bottom).
  let (yp, up, vp) = vramp_yuv440p();
  let top = convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp);
  for loc in [
    ChromaLocation::Left,
    ChromaLocation::Center,
    ChromaLocation::Unspecified,
    ChromaLocation::Bottom,
  ] {
    assert_ne!(
      top,
      convert_rgb_with(loc.clone(), true, &yp, &up, &vp),
      "Top (v=0) must differ from {loc:?} on a vertical chroma ramp"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_equals_cosited_on_flat_chroma() {
  // On constant chroma the forward vertical blend is a no-op, so Top collapses to
  // the co-sited decode byte-for-byte.
  let (yp, up, vp) = flat_chroma_yuv440p();
  assert_eq!(
    convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::Left, true, &yp, &up, &vp),
    "flat-chroma Top must equal co-sited"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_path_simd_matches_scalar() {
  let (yp, up, vp) = ramp_yuv440p();
  assert_eq!(
    convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
    convert_rgb_with(ChromaLocation::Top, false, &yp, &up, &vp),
    "top path must be bit-identical across the SIMD and scalar tiers"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_grows_full_width_chroma_prev_lookback() {
  // The forward one-row delay maintains the same full-width (2·w) U+V lookback as
  // Bottom, so a later even row can forward-blend the held odd row's chroma.
  let (yp, up, vp) = vramp_yuv440p();
  let src = Yuv440pFrame::new(&yp, &up, &vp, W, H, W, W, W);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv440p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Top);
  yuv440p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  let prev_len = sink.chroma_prev.len();
  drop(sink);
  assert_eq!(
    prev_len,
    2 * W as usize,
    "top-sited 4:4:0 stages a full-width (2·w) U+V chroma lookback"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_begin_frame_after_held_odd_row_clears_state() {
  // The Nv21 cross-frame-corruption class: a deferred odd Top row is HELD (no
  // output); `begin_frame` MUST drop it so it never flushes into the next frame. A
  // fresh in-order frame then decodes byte-identically to a clean Top decode.
  let (yp, up, vp) = vramp_yuv440p();
  let w = W as usize;
  let h = H as usize;
  let mut rgb = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Yuv440p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Top);
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  // Deliver ONLY row 1 (odd) — it is HELD in the forward delay, producing no output.
  let cr = 1 / 2;
  let row1 = Yuv440pRow::for_tests(
    &yp[w..2 * w],
    &up[cr * w..cr * w + w],
    &vp[cr * w..cr * w + w],
    1,
    KernelMatrix::Bt601,
    false,
  );
  crate::PixelSink::process(&mut sink, row1).unwrap();
  assert!(
    sink.chroma_top_pending.is_some(),
    "an odd Top row must be HELD in the forward delay"
  );
  // New frame: the held odd row must be dropped.
  crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
  assert!(
    sink.chroma_top_pending.is_none(),
    "begin_frame must clear the held Top odd row (Nv21 class)"
  );
  // A full in-order decode now matches a clean decode — no stale held row leaked.
  for r in 0..h {
    let cr = r / 2;
    let row = Yuv440pRow::for_tests(
      &yp[r * w..r * w + w],
      &up[cr * w..cr * w + w],
      &vp[cr * w..cr * w + w],
      r,
      KernelMatrix::Bt601,
      false,
    );
    crate::PixelSink::process(&mut sink, row).unwrap();
  }
  drop(sink);
  let clean = convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp);
  assert_eq!(
    rgb, clean,
    "post-clear frame must decode identically to a clean Top decode"
  );
}
