//! Horizontal chroma-siting for the `1→4`-subsampled 4:1:1 (`Yuv411p`,
//! `Uyyvyy411`) and 4:1:0 (`Yuv410p`) layouts (#302, RFC #238).
//!
//! A centered (`Center` / `Top` / `Bottom`) 4:1:x chroma sample sits at the
//! CENTER of the four luma columns it covers (luma `4j + 1.5` for `c[j]`), so the
//! reconstruction is the `1→4` triangle
//! ([`chroma_upsample_4to1_center_h`](crate::row::scalar::chroma_upsample_4to1_center_h))
//! rather than the co-sited nearest decode. Covers: the `1→4` kernel matching
//! hand-computed sub-phase weights; the identity centered RGB / HSV decodes
//! matching an independent "reconstruct-then-4:4:4" oracle; the native / HSV-only
//! resample tiers matching the folded single-rounding YUV-domain oracle; the RGB
//! row-stage tier matching the reconstruct-then-bin oracle; the default /
//! co-sited path staying byte-identical; SIMD-vs-scalar parity; the packed
//! `Uyyvyy411` decode matching the planar `Yuv411p` decode of the de-packed
//! planes; and the mid-frame siting-flip rejection.

use crate::{
  ChromaLocation, KernelMatrix, PixelSink,
  resample::AreaResampler,
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Uyyvyy411, Uyyvyy411Row, Yuv410p, Yuv410pRow, Yuv411p, Yuv411pRow, Yuv444p, uyyvyy411_to,
    yuv410p_to, yuv411p_to, yuv444p_to,
  },
};
use mediaframe::frame::{Uyyvyy411Frame, Yuv410pFrame, Yuv411pFrame, Yuv444pFrame};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

// ---- shared oracles --------------------------------------------------------

/// Round `a / d` half-up — the production `round_div_half_up`, replicated so the
/// oracle is independent.
fn rdhu(a: u64, d: u64) -> u64 {
  let q = a / d;
  let r = a % d;
  q + u64::from(r >= d - d / 2)
}

/// Exact box-overlap area weights for `src -> out` (mirrors `AxisSpans::area`):
/// per output `(first source cell, overlaps)`.
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

/// Factor-`f` subsampled area weights for `src_full -> out` (mirrors
/// `AxisSpans::area_subsampled`): each cell is `f` full-grid rows/cols, weights
/// live on the `src_full` grid, `start` is a CELL index.
fn area_subsampled_weights(src_full: usize, out: usize, f: usize) -> Vec<(usize, Vec<u64>)> {
  let (src64, out64, f64_) = (src_full as u64, out as u64, f as u64);
  let cells = src_full.div_ceil(f);
  (0..out)
    .map(|o| {
      let lo = o as u64 * src64;
      let hi = lo + src64;
      let start = ((lo / out64) / f64_) as usize;
      let mut w = Vec::new();
      let mut c = start as u64;
      loop {
        let clo = (f64_ * c) * out64;
        let chi = ((f64_ * c + f64_).min(src64)) * out64;
        if clo >= hi {
          break;
        }
        let ov = chi.min(hi) - clo.max(lo);
        if ov == 0 {
          break;
        }
        w.push(ov);
        if chi >= hi || c as usize + 1 >= cells {
          break;
        }
        c += 1;
      }
      (start, w)
    })
    .collect()
}

/// Co-sited box-average of a full-resolution `sw x sh` u8 plane to `ow x oh` —
/// the reference for a phase-free plane (luma).
fn bin_cosited(plane: &[u8], sw: usize, sh: usize, ow: usize, oh: usize) -> Vec<u8> {
  let hw = area_weights(sw, ow);
  let vw = area_weights(sh, oh);
  let denom = (sw * sh) as u64;
  let mut out = vec![0u8; ow * oh];
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
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// Independent `1→4` centered horizontal reconstruction (`×8` weights, edge
/// clamp) — written separately from the production kernel so it is a real
/// oracle. `quarter = width.div_ceil(4)` samples in, `width` samples out.
fn ref_upsample_4to1_center_h(cq: &[u8], width: usize) -> Vec<u8> {
  let quarter = width.div_ceil(4);
  let mut out = vec![0u8; width];
  for j in 0..quarter {
    let l = i32::from(cq[j.saturating_sub(1)]);
    let m = i32::from(cq[j]);
    let r = i32::from(cq[if j + 1 < quarter { j + 1 } else { j }]);
    let base = 4 * j;
    if base < width {
      out[base] = ((3 * l + 5 * m + 4) >> 3) as u8;
    }
    if base + 1 < width {
      out[base + 1] = ((l + 7 * m + 4) >> 3) as u8;
    }
    if base + 2 < width {
      out[base + 2] = ((7 * m + r + 4) >> 3) as u8;
    }
    if base + 3 < width {
      out[base + 3] = ((5 * m + 3 * r + 4) >> 3) as u8;
    }
  }
  out
}

/// The EXACT centered chroma oracle for the folded native / HSV-only resample:
/// reconstruct each `cw`-wide chroma row to `luma_w` full width with the `1→4`
/// triangle kept UNROUNDED (scaled ×8), then box-average to `ow x oh` with a
/// SINGLE round-half-up over `8·luma_w·<v-denom>`. `vw` is the vertical binning
/// (full-height for 4:1:1, factor-4 subsampled for 4:1:0) and `v_denom` its
/// normalization (`ch` for 4:1:1, `luma_h` for 4:1:0).
fn bin_chroma_411x_centered(
  cq: &[u8],
  luma_w: usize,
  ch: usize,
  vw: &[(usize, Vec<u64>)],
  v_denom: usize,
  ow: usize,
  oh: usize,
) -> Vec<u8> {
  let cw = luma_w.div_ceil(4);
  let mut r8 = vec![0u32; luma_w * ch];
  for r in 0..ch {
    let row = &cq[r * cw..r * cw + cw];
    for j in 0..cw {
      let l = u32::from(row[j.saturating_sub(1)]);
      let m = u32::from(row[j]);
      let rt = u32::from(row[if j + 1 < cw { j + 1 } else { j }]);
      let base = 4 * j;
      if base < luma_w {
        r8[r * luma_w + base] = 3 * l + 5 * m;
      }
      if base + 1 < luma_w {
        r8[r * luma_w + base + 1] = l + 7 * m;
      }
      if base + 2 < luma_w {
        r8[r * luma_w + base + 2] = 7 * m + rt;
      }
      if base + 3 < luma_w {
        r8[r * luma_w + base + 3] = 5 * m + 3 * rt;
      }
    }
  }
  let hw = area_weights(luma_w, ow);
  let denom = (8 * luma_w * v_denom) as u64;
  let mut out = vec![0u8; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * u64::from(r8[(vs + dy) * luma_w + hs + dx]);
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u8;
    }
  }
  out
}

/// Co-sited area-bin of a packed `sw x sh` RGB plane (3 bytes/pixel) to
/// `ow x oh` — the reference for the RGB row-stage tier, which bins the
/// full-resolution converted RGB rows (`Rgb24` area resample of the
/// identity-decoded frame).
fn bin_rgb(rgb: &[u8], sw: usize, sh: usize, ow: usize, oh: usize) -> Vec<u8> {
  let hw = area_weights(sw, ow);
  let vw = area_weights(sh, oh);
  let denom = (sw * sh) as u64;
  let mut out = vec![0u8; ow * oh * 3];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      for ch in 0..3 {
        let mut s = 0u64;
        for (dy, &vwt) in vwin.iter().enumerate() {
          let mut hsum = 0u64;
          for (dx, &hwt) in hwin.iter().enumerate() {
            hsum += hwt * u64::from(rgb[((vs + dy) * sw + hs + dx) * 3 + ch]);
          }
          s += vwt * hsum;
        }
        out[(oy * ow + ox) * 3 + ch] = rdhu(s, denom) as u8;
      }
    }
  }
  out
}

// ---- 4:1:1 planar fixtures -------------------------------------------------

/// A `Yuv411p` frame with a per-column chroma ramp on the quarter-width,
/// full-height chroma planes. `+ r` keeps chroma rows distinct.
fn ramp_411(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = w.div_ceil(4);
  let mut y = vec![0u8; w * h];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (32 + (i % 96)) as u8;
  }
  let mut u = vec![0u8; cw * h];
  let mut v = vec![0u8; cw * h];
  for r in 0..h {
    for c in 0..cw {
      u[r * cw + c] = (16 + c * 40 + r * 5).min(240) as u8;
      v[r * cw + c] = (240 - c * 40).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Identity `Yuv411p` decode at `loc` siting → `(rgb, (h,s,v), luma)`.
fn id_411(w: usize, h: usize, loc: ChromaLocation, simd: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = w.div_ceil(4);
  let (yp, up, vp) = ramp_411(w, h);
  let mut rgb = vec![0u8; w * h * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
  let mut luma = vec![0u8; w * h];
  {
    let f = Yuv411pFrame::new(
      &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
    );
    let mut sink = MixedSinker::<Yuv411p>::new(w, h)
      .with_chroma_location(loc.clone())
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    yuv411p_to(&f, FR, M, &mut sink).unwrap();
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  (rgb, hsv, luma)
}

/// Resample `Yuv411p` decode at `loc` siting → `(rgb, (h,s,v), luma)`.
#[allow(clippy::too_many_arguments)]
fn rs_411(
  w: usize,
  h: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  simd: bool,
  color: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = w.div_ceil(4);
  let (yp, up, vp) = ramp_411(w, h);
  let mut rgb = vec![0u8; ow * oh * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  {
    let f = Yuv411pFrame::new(
      &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
    );
    let mut sink =
      MixedSinker::<Yuv411p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_chroma_location(loc.clone())
        .with_simd(simd);
    if color {
      sink = sink.with_rgb(&mut rgb).unwrap();
    }
    let mut sink = sink
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    yuv411p_to(&f, FR, M, &mut sink).unwrap();
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  (rgb, hsv, luma)
}

/// End-to-end identity oracle: reconstruct each luma row's chroma with the
/// independent `1→4` kernel, then decode as `Yuv444p` → `(rgb, hsv)`.
fn id_411_oracle(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
  let cw = w.div_ceil(4);
  let (yp, up, vp) = ramp_411(w, h);
  let mut u444 = vec![0u8; w * h];
  let mut v444 = vec![0u8; w * h];
  for r in 0..h {
    let ur = ref_upsample_4to1_center_h(&up[r * cw..r * cw + cw], w);
    let vr = ref_upsample_4to1_center_h(&vp[r * cw..r * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&ur);
    v444[r * w..r * w + w].copy_from_slice(&vr);
  }
  yuv444_decode(&yp, &u444, &v444, w, h)
}

/// Decodes full-resolution `Yuv444p` planes → `(rgb, hsv)` via an identity sink.
fn yuv444_decode(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
  let mut rgb = vec![0u8; w * h * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
  {
    let f = Yuv444pFrame::new(y, u, v, w as u32, h as u32, w as u32, w as u32, w as u32);
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuv444p_to(&f, FR, M, &mut sink).unwrap();
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  (rgb, hsv)
}

const CENTERED: [ChromaLocation; 3] = [
  ChromaLocation::Center,
  ChromaLocation::Top,
  ChromaLocation::Bottom,
];
/// The cosited sitings, plus the open escape: a siting this build does not
/// name cosites like `Unspecified`. Not a `const` because `other` allocates
/// its slug (mediaframe 0.3 struck the numeric `Unknown(u32)` escape).
fn cosited() -> [ChromaLocation; 5] {
  [
    ChromaLocation::Unspecified,
    ChromaLocation::Left,
    ChromaLocation::TopLeft,
    ChromaLocation::BottomLeft,
    ChromaLocation::other("unassigned-99"),
  ]
}

// ---- 1→4 kernel oracle -----------------------------------------------------

#[test]
fn kernel_4to1_matches_hand_computed() {
  // c = [0, 0, 100, 100], width 16. Hand-computed ×8 sub-phase weights with the
  // left/right edge clamp: the interior columns 6,7,8,9 ramp the step.
  let cq = [0u8, 0, 100, 100];
  let mut out = [0u8; 16];
  crate::row::scalar::chroma_upsample_4to1_center_h(&cq, &mut out, 16);
  assert_eq!(
    out,
    [
      0, 0, 0, 0, 0, 0, 13, 38, 63, 88, 100, 100, 100, 100, 100, 100
    ]
  );
}

#[test]
fn kernel_4to1_flat_input_is_flat() {
  // A flat quarter row reconstructs flat (every sub-phase sums to 8·c).
  let cq = [77u8; 5];
  let mut out = [0u8; 20];
  crate::row::scalar::chroma_upsample_4to1_center_h(&cq, &mut out, 20);
  assert!(out.iter().all(|&x| x == 77));
}

#[test]
fn kernel_4to1_non_multiple_of_4_width_writes_only_real_columns() {
  // width 6 (quarter = 2): only columns 0..6 written, last group partial.
  let cq = [10u8, 200];
  let mut out = [255u8; 6];
  crate::row::scalar::chroma_upsample_4to1_center_h(&cq, &mut out, 6);
  // Column 4 = (3·c0 + 5·c1)/8, column 5 = (c0 + 7·c1)/8 (right clamp unused).
  assert_eq!(out[4], ((3 * 10 + 5 * 200 + 4) >> 3) as u8);
  assert_eq!(out[5], ((10 + 7 * 200 + 4) >> 3) as u8);
}

// ---- 4:1:1 identity: co-sited byte-identical + centered oracle -------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_411_cosited_is_byte_identical() {
  let base = id_411(16, 6, ChromaLocation::Unspecified, true);
  for loc in cosited() {
    assert_eq!(id_411(16, 6, loc.clone(), true), base, "siting {loc:?}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_411_centered_differs_from_cosited() {
  let cos = id_411(16, 6, ChromaLocation::Left, true);
  for loc in CENTERED {
    let cen = id_411(16, 6, loc.clone(), true);
    assert_ne!(
      cen.0, cos.0,
      "centered rgb {loc:?} must differ from co-sited"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_411_centered_equals_reconstruct_444_oracle() {
  // Every centered siting shares the SAME horizontal phase, so all three match
  // the single `1→4` reconstruct-then-4:4:4 oracle. Widths mult-of-4 and not.
  for (w, h) in [(16usize, 6usize), (12, 4), (20, 5)] {
    let (rgb_o, hsv_o) = id_411_oracle(w, h);
    for loc in CENTERED {
      let (rgb, hsv, _) = id_411(w, h, loc.clone(), true);
      assert_eq!(rgb, rgb_o, "rgb {loc:?} {w}x{h}");
      assert_eq!(hsv, hsv_o, "hsv {loc:?} {w}x{h}");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_411_centered_simd_matches_scalar() {
  // RGB + luma are bit-exact SIMD vs scalar. HSV derives via `rgb_to_hsv_row`,
  // whose hue is numerically unstable near gray and can diverge SIMD vs scalar
  // independently of siting, so it is not asserted here (siting correctness of
  // HSV is pinned by the reconstruct-then-4:4:4 oracle test).
  let s = id_411(16, 6, ChromaLocation::Center, false);
  let d = id_411(16, 6, ChromaLocation::Center, true);
  assert_eq!(s.0, d.0, "rgb simd vs scalar");
  assert_eq!(s.2, d.2, "luma simd vs scalar");
}

// ---- 4:1:1 resample: co-sited + centered folded oracle ---------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_411_cosited_is_byte_identical() {
  let base = rs_411(16, 8, 4, 4, ChromaLocation::Unspecified, true, true);
  for loc in cosited() {
    assert_eq!(
      rs_411(16, 8, 4, 4, loc.clone(), true, true),
      base,
      "resample siting {loc:?}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_411_centered_differs_from_cosited() {
  let cos = rs_411(16, 8, 4, 4, ChromaLocation::Left, true, true);
  let cen = rs_411(16, 8, 4, 4, ChromaLocation::Center, true, true);
  assert_ne!(cen.0, cos.0, "centered resample rgb must differ");
  // Luma is siting-independent (Y is binned co-sited regardless of chroma phase).
  assert_eq!(cen.2, cos.2, "luma must be siting-independent");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_411_hsv_direct_centered_equals_folded_oracle() {
  // The HSV-only arm bins the SUBSAMPLED chroma through the folded
  // `area_chroma_411` plan (single rounding). Its output is the exact
  // reconstruct-unrounded-then-bin YUV-domain oracle.
  for (w, h, ow, oh) in [(16usize, 8usize, 4usize, 4usize), (16, 10, 7, 6)] {
    let cw = w.div_ceil(4);
    let (yp, up, vp) = ramp_411(w, h);
    // HSV-only resample (no RGB attached → the hsv_direct join).
    let (_, hsv, luma) = rs_411(w, h, ow, oh, ChromaLocation::Center, true, false);
    // Oracle: bin Y co-sited, bin U/V through the folded centered plan (V
    // full-height), then convert once at output width as Yuv444p.
    let vw = area_weights(h, oh);
    let yb = bin_cosited(&yp, w, h, ow, oh);
    let ub = bin_chroma_411x_centered(&up, w, h, &vw, h, ow, oh);
    let vb = bin_chroma_411x_centered(&vp, w, h, &vw, h, ow, oh);
    let (_, hsv_o) = yuv444_decode(&yb, &ub, &vb, ow, oh);
    assert_eq!(luma, yb, "luma {w}x{h}->{ow}x{oh}");
    assert_eq!(hsv, hsv_o, "hsv-direct {w}x{h}->{ow}x{oh}");
    let _ = cw;
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_411_centered_rgb_equals_reconstruct_then_bin() {
  // The RGB row-stage tier converts each source row to full-width RGB (via the
  // centered `1→4` reconstruction + 4:4:4 decode) then area-bins those RGB rows.
  // So it is byte-identical to area-binning the identity centered decode.
  for (w, h, ow, oh) in [(16usize, 8usize, 4usize, 4usize), (16, 10, 7, 6)] {
    let full = id_411(w, h, ChromaLocation::Center, true).0;
    let oracle = bin_rgb(&full, w, h, ow, oh);
    let rgb = rs_411(w, h, ow, oh, ChromaLocation::Center, true, true).0;
    assert_eq!(
      rgb, oracle,
      "row-stage rgb == reconstruct-then-bin {w}x{h}->{ow}x{oh}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_411_centered_simd_matches_scalar() {
  let s = rs_411(16, 8, 4, 4, ChromaLocation::Center, false, true);
  let d = rs_411(16, 8, 4, 4, ChromaLocation::Center, true, true);
  assert_eq!(s.0, d.0, "rgb simd vs scalar");
  assert_eq!(s.1, d.1, "hsv simd vs scalar");
  assert_eq!(s.2, d.2, "luma simd vs scalar");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_411_mid_frame_siting_flip_is_rejected() {
  let (yp, up, vp) = ramp_411(16, 8);
  let cw = 4usize;
  let (mut hh, mut ss, mut vv) = (vec![0u8; 16], vec![0u8; 16], vec![0u8; 16]);
  let mut sink =
    MixedSinker::<Yuv411p, AreaResampler>::with_resampler(16, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_chroma_location(ChromaLocation::Center)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 16, 8).unwrap();
  for r in 0..2 {
    let row = Yuv411pRow::new(
      &yp[r * 16..r * 16 + 16],
      &up[r * cw..r * cw + cw],
      &vp[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  sink.set_chroma_location(ChromaLocation::Left);
  let bad = Yuv411pRow::new(
    &yp[2 * 16..3 * 16],
    &up[2 * cw..3 * cw],
    &vp[2 * cw..3 * cw],
    2,
    M,
    FR,
  );
  let err = PixelSink::process(&mut sink, bad).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
    "mid-frame siting flip must be ChromaSitingChanged, got {err:?}"
  );
  // The rejected row mutated no state: a fresh frame at the new siting works.
  PixelSink::begin_frame(&mut sink, 16, 8).unwrap();
  for r in 0..8 {
    let row = Yuv411pRow::new(
      &yp[r * 16..r * 16 + 16],
      &up[r * cw..r * cw + cw],
      &vp[r * cw..r * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_411_mid_frame_siting_flip_is_rejected() {
  let (w, h) = (16usize, 8usize);
  let cw = w.div_ceil(4);
  let (yp, up, vp) = ramp_411(w, h);
  let mut rgb = vec![0u8; w * h * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
  let mut luma = vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Yuv411p>::new(w, h)
      .with_chroma_location(ChromaLocation::Center)
      .with_simd(true)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    PixelSink::begin_frame(&mut sink, w as u32, h as u32).unwrap();
    // Rows 0,1 decode centered and freeze the phase for the frame.
    for r in 0..2 {
      let row = Yuv411pRow::new(
        &yp[r * w..r * w + w],
        &up[r * cw..r * cw + cw],
        &vp[r * cw..r * cw + cw],
        r,
        M,
        FR,
      );
      PixelSink::process(&mut sink, row).unwrap();
    }
    // A mid-frame flip to a co-sited phase is rejected before any reservation or
    // output write on the identity path (not silently mixed with the frozen
    // centered rows).
    sink.set_chroma_location(ChromaLocation::Left);
    let bad = Yuv411pRow::new(
      &yp[2 * w..3 * w],
      &up[2 * cw..3 * cw],
      &vp[2 * cw..3 * cw],
      2,
      M,
      FR,
    );
    let err = PixelSink::process(&mut sink, bad).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "411 identity mid-frame flip must be ChromaSitingChanged, got {err:?}"
    );
    // The rejected row mutated nothing: flip back and the same row retries, and the
    // frame completes at the frozen centered phase.
    sink.set_chroma_location(ChromaLocation::Center);
    for r in 2..h {
      let row = Yuv411pRow::new(
        &yp[r * w..r * w + w],
        &up[r * cw..r * cw + cw],
        &vp[r * cw..r * cw + cw],
        r,
        M,
        FR,
      );
      PixelSink::process(&mut sink, row).unwrap();
    }
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  // Byte-identical to a clean single-phase centered identity decode: the frame
  // that survived the rejected flip carries no mixed centered / co-sited output.
  let (want_rgb, want_hsv, want_luma) = id_411(w, h, ChromaLocation::Center, true);
  assert_eq!(rgb, want_rgb, "411 identity rgb after rejected flip");
  assert_eq!(hsv, want_hsv, "411 identity hsv after rejected flip");
  assert_eq!(luma, want_luma, "411 identity luma after rejected flip");
}

// ---- 4:1:0 planar ----------------------------------------------------------

/// A `Yuv410p` frame (chroma quarter-width AND quarter-height).
fn ramp_410(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = w / 4;
  let ch = h.div_ceil(4);
  let mut y = vec![0u8; w * h];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i % 100)) as u8;
  }
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (20 + c * 44 + r * 7).min(240) as u8;
      v[r * cw + c] = (230 - c * 44).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Identity `Yuv410p` decode at `loc` → `(rgb, hsv, luma)`.
fn id_410(w: usize, h: usize, loc: ChromaLocation, simd: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = (w / 4) as u32;
  let (yp, up, vp) = ramp_410(w, h);
  let mut rgb = vec![0u8; w * h * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
  let mut luma = vec![0u8; w * h];
  {
    let f = Yuv410pFrame::new(&yp, &up, &vp, w as u32, h as u32, w as u32, cw, cw);
    let mut sink = MixedSinker::<Yuv410p>::new(w, h)
      .with_chroma_location(loc.clone())
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    yuv410p_to(&f, FR, M, &mut sink).unwrap();
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  (rgb, hsv, luma)
}

/// Identity 4:1:0 oracle: each luma row `r` uses chroma row `r/4` reconstructed
/// horizontally with the independent `1→4` kernel, decoded as `Yuv444p`.
fn id_410_oracle(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
  let cw = w / 4;
  let (yp, up, vp) = ramp_410(w, h);
  let mut u444 = vec![0u8; w * h];
  let mut v444 = vec![0u8; w * h];
  for r in 0..h {
    let cr = r / 4;
    let ur = ref_upsample_4to1_center_h(&up[cr * cw..cr * cw + cw], w);
    let vr = ref_upsample_4to1_center_h(&vp[cr * cw..cr * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&ur);
    v444[r * w..r * w + w].copy_from_slice(&vr);
  }
  yuv444_decode(&yp, &u444, &v444, w, h)
}

/// Resample `Yuv410p` HSV-only / RGB decode at `loc` → `(rgb, hsv, luma)`.
#[allow(clippy::too_many_arguments)]
fn rs_410(
  w: usize,
  h: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  simd: bool,
  color: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = (w / 4) as u32;
  let (yp, up, vp) = ramp_410(w, h);
  let mut rgb = vec![0u8; ow * oh * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut luma = vec![0u8; ow * oh];
  {
    let f = Yuv410pFrame::new(&yp, &up, &vp, w as u32, h as u32, w as u32, cw, cw);
    let mut sink =
      MixedSinker::<Yuv410p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_chroma_location(loc.clone())
        .with_simd(simd);
    if color {
      sink = sink.with_rgb(&mut rgb).unwrap();
    }
    let mut sink = sink
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    yuv410p_to(&f, FR, M, &mut sink).unwrap();
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  (rgb, hsv, luma)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_410_cosited_is_byte_identical() {
  let base = id_410(16, 8, ChromaLocation::Unspecified, true);
  for loc in cosited() {
    assert_eq!(id_410(16, 8, loc.clone(), true), base, "410 siting {loc:?}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_410_centered_equals_reconstruct_444_oracle() {
  for (w, h) in [(16usize, 8usize), (12, 8), (20, 12)] {
    let (rgb_o, hsv_o) = id_410_oracle(w, h);
    for loc in CENTERED {
      let (rgb, hsv, _) = id_410(w, h, loc.clone(), true);
      assert_eq!(rgb, rgb_o, "410 rgb {loc:?} {w}x{h}");
      assert_eq!(hsv, hsv_o, "410 hsv {loc:?} {w}x{h}");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_410_centered_simd_matches_scalar() {
  // RGB + luma are bit-exact; HSV via `rgb_to_hsv_row` is not (see the 4:1:1 twin).
  let s = id_410(16, 8, ChromaLocation::Center, false);
  let d = id_410(16, 8, ChromaLocation::Center, true);
  assert_eq!(s.0, d.0, "410 rgb simd vs scalar");
  assert_eq!(s.2, d.2, "410 luma simd vs scalar");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_410_cosited_is_byte_identical() {
  let base = rs_410(16, 8, 4, 2, ChromaLocation::Unspecified, true, true);
  for loc in cosited() {
    assert_eq!(
      rs_410(16, 8, 4, 2, loc.clone(), true, true),
      base,
      "410 resample siting {loc:?}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_410_hsv_direct_centered_equals_folded_oracle() {
  for (w, h, ow, oh) in [(16usize, 8usize, 4usize, 2usize), (16, 12, 3, 2)] {
    let (yp, up, vp) = ramp_410(w, h);
    let ch = h.div_ceil(4);
    let (_, hsv, luma) = rs_410(w, h, ow, oh, ChromaLocation::Center, true, false);
    // Oracle: bin Y co-sited; bin U/V folded-H over the quarter-height chroma
    // with a factor-4 subsampled V binning (denominator luma_h).
    let vw = area_subsampled_weights(h, oh, 4);
    let yb = bin_cosited(&yp, w, h, ow, oh);
    let ub = bin_chroma_411x_centered(&up, w, ch, &vw, h, ow, oh);
    let vb = bin_chroma_411x_centered(&vp, w, ch, &vw, h, ow, oh);
    let (_, hsv_o) = yuv444_decode(&yb, &ub, &vb, ow, oh);
    assert_eq!(luma, yb, "410 luma {w}x{h}->{ow}x{oh}");
    assert_eq!(hsv, hsv_o, "410 hsv-direct {w}x{h}->{ow}x{oh}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_410_centered_differs_and_simd_matches_scalar() {
  let cos = rs_410(16, 8, 4, 2, ChromaLocation::Left, true, true);
  let cen = rs_410(16, 8, 4, 2, ChromaLocation::Center, true, true);
  assert_ne!(cen.0, cos.0, "410 centered resample must differ");
  let cen_s = rs_410(16, 8, 4, 2, ChromaLocation::Center, false, true);
  assert_eq!(cen.0, cen_s.0, "410 centered resample simd vs scalar");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rs_410_mid_frame_siting_flip_is_rejected() {
  let (yp, up, vp) = ramp_410(16, 8);
  let cw = 4usize;
  let (mut hh, mut ss, mut vv) = (vec![0u8; 8], vec![0u8; 8], vec![0u8; 8]);
  let mut sink =
    MixedSinker::<Yuv410p, AreaResampler>::with_resampler(16, 8, AreaResampler::to(4, 2))
      .unwrap()
      .with_chroma_location(ChromaLocation::Center)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 16, 8).unwrap();
  for r in 0..2 {
    let cr = r / 4;
    let row = Yuv410pRow::new(
      &yp[r * 16..r * 16 + 16],
      &up[cr * cw..cr * cw + cw],
      &vp[cr * cw..cr * cw + cw],
      r,
      M,
      FR,
    );
    PixelSink::process(&mut sink, row).unwrap();
  }
  sink.set_chroma_location(ChromaLocation::Left);
  let bad = Yuv410pRow::new(&yp[2 * 16..3 * 16], &up[0..cw], &vp[0..cw], 2, M, FR);
  let err = PixelSink::process(&mut sink, bad).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
    "410 mid-frame flip must be ChromaSitingChanged, got {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_410_mid_frame_siting_flip_is_rejected() {
  let (w, h) = (16usize, 8usize);
  let cw = w / 4;
  let (yp, up, vp) = ramp_410(w, h);
  let mut rgb = vec![0u8; w * h * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
  let mut luma = vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Yuv410p>::new(w, h)
      .with_chroma_location(ChromaLocation::Center)
      .with_simd(true)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    PixelSink::begin_frame(&mut sink, w as u32, h as u32).unwrap();
    // Rows 0,1 decode centered (chroma row 0) and freeze the phase for the frame.
    for r in 0..2 {
      let cr = r / 4;
      let row = Yuv410pRow::new(
        &yp[r * w..r * w + w],
        &up[cr * cw..cr * cw + cw],
        &vp[cr * cw..cr * cw + cw],
        r,
        M,
        FR,
      );
      PixelSink::process(&mut sink, row).unwrap();
    }
    // A mid-frame flip to a co-sited phase is rejected before any reservation or
    // output write on the identity path.
    sink.set_chroma_location(ChromaLocation::Left);
    let bad = Yuv410pRow::new(&yp[2 * w..3 * w], &up[0..cw], &vp[0..cw], 2, M, FR);
    let err = PixelSink::process(&mut sink, bad).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "410 identity mid-frame flip must be ChromaSitingChanged, got {err:?}"
    );
    // The rejected row mutated nothing: flip back and the same row retries, and the
    // frame completes at the frozen centered phase.
    sink.set_chroma_location(ChromaLocation::Center);
    for r in 2..h {
      let cr = r / 4;
      let row = Yuv410pRow::new(
        &yp[r * w..r * w + w],
        &up[cr * cw..cr * cw + cw],
        &vp[cr * cw..cr * cw + cw],
        r,
        M,
        FR,
      );
      PixelSink::process(&mut sink, row).unwrap();
    }
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  // Byte-identical to a clean single-phase centered identity decode: no mixed
  // centered / co-sited output survives the rejected flip.
  let (want_rgb, want_hsv, want_luma) = id_410(w, h, ChromaLocation::Center, true);
  assert_eq!(rgb, want_rgb, "410 identity rgb after rejected flip");
  assert_eq!(hsv, want_hsv, "410 identity hsv after rejected flip");
  assert_eq!(luma, want_luma, "410 identity luma after rejected flip");
}

// ---- packed Uyyvyy411 ------------------------------------------------------

fn uyyvyy411_from(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  assert_eq!(w & 3, 0, "uyyvyy411 width must be a multiple of 4");
  let cw = w / 4;
  let mut buf = vec![0u8; w * 3 / 2 * h];
  for row in 0..h {
    let base = row * w * 3 / 2;
    for cx in 0..cw {
      let blk = base + cx * 6;
      buf[blk] = u[row * cw + cx];
      buf[blk + 1] = y[row * w + cx * 4];
      buf[blk + 2] = y[row * w + cx * 4 + 1];
      buf[blk + 3] = v[row * cw + cx];
      buf[blk + 4] = y[row * w + cx * 4 + 2];
      buf[blk + 5] = y[row * w + cx * 4 + 3];
    }
  }
  buf
}

/// Identity packed `Uyyvyy411` decode at `loc` → `(rgb, hsv, luma)`.
fn id_packed(w: usize, h: usize, loc: ChromaLocation, simd: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let (yp, up, vp) = ramp_411(w, h);
  let packed = uyyvyy411_from(&yp, &up, &vp, w, h);
  let mut rgb = vec![0u8; w * h * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
  let mut luma = vec![0u8; w * h];
  {
    let f = Uyyvyy411Frame::new(&packed, w as u32, h as u32, (w * 3 / 2) as u32);
    let mut sink = MixedSinker::<Uyyvyy411>::new(w, h)
      .with_chroma_location(loc.clone())
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    uyyvyy411_to(&f, FR, M, &mut sink).unwrap();
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  (rgb, hsv, luma)
}

/// Resample packed `Uyyvyy411` decode at `loc` (native or row-stage) → rgb+luma.
#[allow(clippy::too_many_arguments)]
fn rs_packed(
  w: usize,
  h: usize,
  ow: usize,
  oh: usize,
  loc: ChromaLocation,
  native: bool,
  simd: bool,
) -> (Vec<u8>, Vec<u8>) {
  let (yp, up, vp) = ramp_411(w, h);
  let packed = uyyvyy411_from(&yp, &up, &vp, w, h);
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut luma = vec![0u8; ow * oh];
  {
    let f = Uyyvyy411Frame::new(&packed, w as u32, h as u32, (w * 3 / 2) as u32);
    let mut sink =
      MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(native)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    uyyvyy411_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, luma)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_cosited_is_byte_identical() {
  let base = id_packed(16, 6, ChromaLocation::Unspecified, true);
  for loc in cosited() {
    assert_eq!(
      id_packed(16, 6, loc.clone(), true),
      base,
      "packed siting {loc:?}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_centered_equals_planar_411_decode() {
  // The packed centered decode de-packs to Y / U / V then runs the SAME `1→4`
  // reconstruction + 4:4:4 kernels as planar Yuv411p, so it is byte-identical to
  // a planar Yuv411p centered decode of the same logical planes.
  for loc in CENTERED {
    let packed = id_packed(16, 6, loc.clone(), true);
    let planar = id_411(16, 6, loc.clone(), true);
    assert_eq!(packed, planar, "packed vs planar {loc:?}");
  }
  // Negative control: centered differs from co-sited.
  let cos = id_packed(16, 6, ChromaLocation::Left, true);
  let cen = id_packed(16, 6, ChromaLocation::Center, true);
  assert_ne!(cen.0, cos.0, "packed centered must differ from co-sited");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_centered_simd_matches_scalar() {
  // RGB + luma are bit-exact; HSV via `rgb_to_hsv_row` is not (see the 4:1:1 twin).
  let s = id_packed(16, 6, ChromaLocation::Center, false);
  let d = id_packed(16, 6, ChromaLocation::Center, true);
  assert_eq!(s.0, d.0, "packed rgb simd vs scalar");
  assert_eq!(s.2, d.2, "packed luma simd vs scalar");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_resample_cosited_is_byte_identical() {
  for native in [true, false] {
    let base = rs_packed(16, 8, 4, 4, ChromaLocation::Unspecified, native, true);
    for loc in cosited() {
      assert_eq!(
        rs_packed(16, 8, 4, 4, loc.clone(), native, true),
        base,
        "packed resample siting {loc:?} native={native}"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_native_centered_equals_planar_411_resample() {
  // Both the packed native tier and the planar Yuv411p HSV-only tier fold the
  // SAME `area_chroma_411` centered plan, so their binned luma matches exactly;
  // the packed native RGB is the folded-chroma 4:4:4 decode. Cross-check the
  // packed native luma against the planar folded luma.
  let (_, _, planar_luma) = rs_411(16, 8, 4, 4, ChromaLocation::Center, true, false);
  let (_, native_luma) = rs_packed(16, 8, 4, 4, ChromaLocation::Center, true, true);
  assert_eq!(
    native_luma, planar_luma,
    "packed native luma vs planar folded"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_native_centered_equals_folded_rgb_oracle() {
  // The packed native tier bins Y co-sited + folded chroma, then converts ONCE at
  // output width — byte-identical to a Yuv444p decode of the exact single-round
  // folded oracle planes. The row-stage tier (double round) then lands within a
  // small tolerance of the same oracle.
  let w = 16;
  let h = 8;
  let (ow, oh) = (4, 4);
  let (yp, up, vp) = ramp_411(w, h);
  let vw = area_weights(h, oh);
  let yb = bin_cosited(&yp, w, h, ow, oh);
  let ub = bin_chroma_411x_centered(&up, w, h, &vw, h, ow, oh);
  let vb = bin_chroma_411x_centered(&vp, w, h, &vw, h, ow, oh);
  let (rgb_o, _) = yuv444_decode(&yb, &ub, &vb, ow, oh);
  let native = rs_packed(w, h, ow, oh, ChromaLocation::Center, true, true);
  let row_stage = rs_packed(w, h, ow, oh, ChromaLocation::Center, false, true);
  assert_eq!(native.1, yb, "native luma equals the co-sited binned Y");
  assert_eq!(
    native.0, rgb_o,
    "packed native rgb equals the folded oracle"
  );
  assert_eq!(native.1, row_stage.1, "native / row-stage luma identical");
  // The row-stage tier reconstructs to u8 then bins — byte-identical to
  // area-binning the identity centered packed decode.
  let full = id_packed(w, h, ChromaLocation::Center, true).0;
  assert_eq!(
    row_stage.0,
    bin_rgb(&full, w, h, ow, oh),
    "packed row-stage rgb == reconstruct-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_resample_mid_frame_flip_is_rejected() {
  let (yp, up, vp) = ramp_411(16, 8);
  let packed = uyyvyy411_from(&yp, &up, &vp, 16, 8);
  let stride = 16 * 3 / 2;
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(16, 8, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_chroma_location(ChromaLocation::Center)
      .with_rgb(&mut rgb)
      .unwrap();
  PixelSink::begin_frame(&mut sink, 16, 8).unwrap();
  for r in 0..2 {
    let row = Uyyvyy411Row::new(&packed[r * stride..r * stride + stride], r, M, FR);
    PixelSink::process(&mut sink, row).unwrap();
  }
  sink.set_chroma_location(ChromaLocation::Left);
  let bad = Uyyvyy411Row::new(&packed[2 * stride..3 * stride], 2, M, FR);
  let err = PixelSink::process(&mut sink, bad).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
    "packed mid-frame flip must be ChromaSitingChanged, got {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn id_packed_mid_frame_siting_flip_is_rejected() {
  let (w, h) = (16usize, 8usize);
  let stride = w * 3 / 2;
  let (yp, up, vp) = ramp_411(w, h);
  let packed = uyyvyy411_from(&yp, &up, &vp, w, h);
  let mut rgb = vec![0u8; w * h * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
  let mut luma = vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Uyyvyy411>::new(w, h)
      .with_chroma_location(ChromaLocation::Center)
      .with_simd(true)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    PixelSink::begin_frame(&mut sink, w as u32, h as u32).unwrap();
    // Rows 0,1 decode centered and freeze the phase for the frame.
    for r in 0..2 {
      let row = Uyyvyy411Row::new(&packed[r * stride..r * stride + stride], r, M, FR);
      PixelSink::process(&mut sink, row).unwrap();
    }
    // A mid-frame flip to a co-sited phase is rejected before any reservation or
    // output write on the packed identity path.
    sink.set_chroma_location(ChromaLocation::Left);
    let bad = Uyyvyy411Row::new(&packed[2 * stride..3 * stride], 2, M, FR);
    let err = PixelSink::process(&mut sink, bad).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "packed identity mid-frame flip must be ChromaSitingChanged, got {err:?}"
    );
    // The rejected row mutated nothing: flip back and the same row retries, and the
    // frame completes at the frozen centered phase.
    sink.set_chroma_location(ChromaLocation::Center);
    for r in 2..h {
      let row = Uyyvyy411Row::new(&packed[r * stride..r * stride + stride], r, M, FR);
      PixelSink::process(&mut sink, row).unwrap();
    }
  }
  let hsv: Vec<u8> = hh.iter().chain(&ss).chain(&vv).copied().collect();
  // Byte-identical to a clean single-phase centered identity decode: no mixed
  // centered / co-sited output survives the rejected flip.
  let (want_rgb, want_hsv, want_luma) = id_packed(w, h, ChromaLocation::Center, true);
  assert_eq!(rgb, want_rgb, "packed identity rgb after rejected flip");
  assert_eq!(hsv, want_hsv, "packed identity hsv after rejected flip");
  assert_eq!(luma, want_luma, "packed identity luma after rejected flip");
}
