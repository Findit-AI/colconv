//! #263 follow-up — RGB-free YUV-domain HSV-only **area** row-stage
//! resample for the 8-bit planar YUV families.
//!
//! Three concerns:
//!
//! 1. **Parity vs the native fast tier** (THE correctness anchor): for a
//!    format with a native tier (`Yuv420p` / `Yuv422p` / `Yuv444p` /
//!    `Yuv440p`), an HSV-only area-resample with `set_native(false)` (the
//!    row-stage tier) must produce HSV **bit-identical** to the same sink
//!    with the native tier enabled — both now bin Y / U / V then convert
//!    via the SAME `yuv_444_to_hsv_row` kernel at output width. Swept over
//!    matrices, ranges, and output ratios.
//! 2. **YUV-domain reference** for the formats WITHOUT a native tier
//!    (`Yuv410p` / `Yuv411p`): the new HSV equals a hand-built YUV-domain
//!    bin-then-convert reference (exact area-mean of Y / U / V to output
//!    resolution, then `yuv_444_to_hsv_row`).
//! 3. **Structural** — an HSV-only row-stage resample grows NO source-width
//!    RGB scratch (`rgb_scratch.len() == 0`), and the RGB / RGBA / luma
//!    resample paths still run the RGB-staged path (a sink with RGB + HSV
//!    DOES grow the scratch, unchanged).

// `Yuv420p` / `Yuv410p` / `Yuv411p` and their frames / walkers are used
// directly below; the `Yuv422p` / `Yuv444p` / `Yuv440p` markers, frames, and
// walkers are referenced only as `parity_vs_native!` macro arguments (a
// `:ty` / `:path` macro fragment from the call site does not mark a call-site
// import as used), so they are passed by full path there instead of imported.
use crate::{
  KernelMatrix,
  resample::{AreaResampler, AreaStream, ResamplePlan},
  row::yuv_444_to_hsv_row,
  sinker::MixedSinker,
  source::{Yuv410p, Yuv411p, Yuv420p, yuv410p_to, yuv411p_to, yuv420p_to},
};
use mediaframe::frame::{Yuv410pFrame, Yuv411pFrame, Yuv420pFrame};

const MATRICES: [KernelMatrix; 3] = [
  KernelMatrix::Bt601,
  KernelMatrix::Bt709,
  KernelMatrix::Bt2020Ncl,
];

/// A non-gray pseudo-random byte so the HSV hue / saturation branches are
/// exercised (not the degenerate gray fast-path).
fn pat(i: usize, salt: usize) -> u8 {
  ((i
    .wrapping_mul(37)
    .wrapping_add(salt.wrapping_mul(101))
    .wrapping_add(11))
    & 0xFF) as u8
}

fn plane(w: usize, h: usize, salt: usize) -> Vec<u8> {
  (0..w * h).map(|i| pat(i, salt)).collect()
}

/// Exact integer-ratio area mean (round-half-up) of an `in_w x in_h` plane
/// to `out_w x out_h`. Valid only when `in_w % out_w == 0 &&
/// in_h % out_h == 0` (the downscale / identity case the references use), so
/// it reproduces the box-coverage `AreaStream` output exactly.
fn area_bin(plane: &[u8], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
  let (rx, ry) = (in_w / out_w, in_h / out_h);
  let denom = (rx * ry) as u32;
  let mut out = vec![0u8; out_w * out_h];
  for oy in 0..out_h {
    for ox in 0..out_w {
      let mut s = 0u32;
      for dy in 0..ry {
        for dx in 0..rx {
          s += plane[(oy * ry + dy) * in_w + ox * rx + dx] as u32;
        }
      }
      out[oy * out_w + ox] = ((s + denom / 2) / denom) as u8;
    }
  }
  out
}

/// YUV-domain HSV reference: bin Y / U / V to output resolution (exact area
/// mean) then convert each output row through `yuv_444_to_hsv_row` — the
/// ground truth the RGB-free row-stage path reproduces. Chroma is supplied
/// at its own resolution (`cw x ch`) and binned to the SAME output grid.
#[allow(clippy::too_many_arguments)]
fn yuv_domain_hsv_reference(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  src_w: usize,
  src_h: usize,
  cw: usize,
  ch: usize,
  out_w: usize,
  out_h: usize,
  matrix: KernelMatrix,
  full_range: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let yb = area_bin(y, src_w, src_h, out_w, out_h);
  let ub = area_bin(u, cw, ch, out_w, out_h);
  let vb = area_bin(v, cw, ch, out_w, out_h);
  let mut h = vec![0u8; out_w * out_h];
  let mut s = vec![0u8; out_w * out_h];
  let mut val = vec![0u8; out_w * out_h];
  for oy in 0..out_h {
    yuv_444_to_hsv_row(
      &yb[oy * out_w..(oy + 1) * out_w],
      &ub[oy * out_w..(oy + 1) * out_w],
      &vb[oy * out_w..(oy + 1) * out_w],
      &mut h[oy * out_w..(oy + 1) * out_w],
      &mut s[oy * out_w..(oy + 1) * out_w],
      &mut val[oy * out_w..(oy + 1) * out_w],
      out_w,
      matrix,
      full_range,
      true,
    );
  }
  (h, s, val)
}

/// Reference single-plane area resample driving the engine's own
/// [`AreaStream`] row-by-row — the oracle for the HSV join's 3-plane
/// COORDINATION. It resamples one plane in isolation (no staging ring), so a
/// ring that mis-pairs a chroma output row with the wrong luma row diverges
/// from it. Unlike the integer-ratio `area_bin`, it handles vertical chroma
/// upsampling (`out_h` greater than `src_h`), where one feed finalises
/// several output rows at once.
fn area_resample_plane(plan: &ResamplePlan, src: &[u8], src_w: usize, src_h: usize) -> Vec<u8> {
  let (ow, oh) = (plan.out_w(), plan.out_h());
  let mut out = vec![0u8; ow * oh];
  let mut stream = AreaStream::<u8>::new(plan.h(), plan.v(), src_w, src_h, 1).unwrap();
  for sy in 0..src_h {
    stream
      .feed_row(sy, &src[sy * src_w..(sy + 1) * src_w], true, |oy, row| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(row);
      })
      .unwrap();
  }
  out
}

// ===== Parity vs native: HSV-only set_native(false) == default ===========

macro_rules! parity_vs_native {
  ($name:ident, $marker:path, $frame:path, $walker:path, $cw:expr, $ch:expr) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    #[allow(elided_lifetimes_in_paths)]
    fn $name() {
      // Square + non-square downscales over a 12x12 source.
      let (w, h) = (12usize, 12usize);
      let (cw, ch) = ($cw(w, h), $ch(w, h));
      let yp = plane(w, h, 1);
      let up = plane(cw, ch, 2);
      let vp = plane(cw, ch, 3);
      for &(ow, oh) in &[(6usize, 6usize), (4, 4), (6, 4), (4, 6), (3, 3)] {
        for &full_range in &[false, true] {
          for &matrix in &MATRICES {
            let run = |native: bool| {
              let (mut hh, mut ss, mut vv) =
                (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
              let scratch_len = {
                let frame = <$frame>::new(
                  &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
                );
                let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
                  w,
                  h,
                  AreaResampler::to(ow, oh),
                )
                .unwrap();
                sink.set_native(native);
                let mut sink = sink.with_hsv(&mut hh, &mut ss, &mut vv).unwrap();
                $walker(&frame, full_range, matrix, &mut sink).unwrap();
                sink.rgb_scratch.len()
              };
              ((hh, ss, vv), scratch_len)
            };
            let (native, native_scratch) = run(true);
            let (row, row_scratch) = run(false);
            assert_eq!(
              native, row,
              "HSV must be bit-identical native-vs-rowstage {ow}x{oh} \
               fr={full_range} {matrix:?}"
            );
            // RGB-free on BOTH tiers.
            assert_eq!(native_scratch, 0, "native HSV-only RGB-free");
            assert_eq!(row_scratch, 0, "row-stage HSV-only RGB-free");
          }
        }
      }
    }
  };
}

parity_vs_native!(
  yuv420p_hsv_only_rowstage_matches_native,
  Yuv420p,
  Yuv420pFrame,
  yuv420p_to,
  |w: usize, _h| w / 2,
  |_w, h: usize| h / 2
);
parity_vs_native!(
  yuv422p_hsv_only_rowstage_matches_native,
  crate::source::Yuv422p,
  mediaframe::frame::Yuv422pFrame,
  crate::source::yuv422p_to,
  |w: usize, _h| w / 2,
  |_w, h: usize| h
);
parity_vs_native!(
  yuv444p_hsv_only_rowstage_matches_native,
  crate::source::Yuv444p,
  mediaframe::frame::Yuv444pFrame,
  crate::source::yuv444p_to,
  |w: usize, _h| w,
  |_w, h: usize| h
);
parity_vs_native!(
  yuv440p_hsv_only_rowstage_matches_native,
  crate::source::Yuv440p,
  mediaframe::frame::Yuv440pFrame,
  crate::source::yuv440p_to,
  |w: usize, _h| w,
  |_w, h: usize| h / 2
);

// ===== YUV-domain reference: formats WITHOUT a native tier ================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv410p_hsv_only_matches_yuv_domain_reference() {
  // 4:1:0 chroma is quarter-width AND quarter-height. Use SRC=16 / OUT=4 so
  // chroma (4x4) bins to (4x4) — identity per axis, an exact area reference.
  let (w, h) = (16usize, 16usize);
  let (cw, ch) = (w / 4, h / 4);
  let yp = plane(w, h, 1);
  let up = plane(cw, ch, 2);
  let vp = plane(cw, ch, 3);
  let (ow, oh) = (4usize, 4usize);
  for &full_range in &[false, true] {
    for &matrix in &MATRICES {
      let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
      let scratch_len = {
        let frame = Yuv410pFrame::new(
          &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
        );
        let mut sink =
          MixedSinker::<Yuv410p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
            .unwrap()
            .with_hsv(&mut hh, &mut ss, &mut vv)
            .unwrap();
        yuv410p_to(&frame, full_range, matrix, &mut sink).unwrap();
        sink.rgb_scratch.len()
      };
      let want = yuv_domain_hsv_reference(&yp, &up, &vp, w, h, cw, ch, ow, oh, matrix, full_range);
      assert_eq!((hh, ss, vv), want, "410p HSV fr={full_range} {matrix:?}");
      assert_eq!(scratch_len, 0, "410p HSV-only RGB-free");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_hsv_only_matches_yuv_domain_reference() {
  // 4:1:1 chroma is quarter-width, full height. SRC=16 / OUT=4: chroma
  // (4x16) bins to (4x4) — horizontal identity, vertical 4:1, exact area.
  let (w, h) = (16usize, 16usize);
  let cw = w.div_ceil(4);
  let ch = h;
  let yp = plane(w, h, 1);
  let up = plane(cw, ch, 2);
  let vp = plane(cw, ch, 3);
  let (ow, oh) = (4usize, 4usize);
  for &full_range in &[false, true] {
    for &matrix in &MATRICES {
      let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
      let scratch_len = {
        let frame = Yuv411pFrame::new(
          &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
        );
        let mut sink =
          MixedSinker::<Yuv411p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
            .unwrap()
            .with_hsv(&mut hh, &mut ss, &mut vv)
            .unwrap();
        yuv411p_to(&frame, full_range, matrix, &mut sink).unwrap();
        sink.rgb_scratch.len()
      };
      let want = yuv_domain_hsv_reference(&yp, &up, &vp, w, h, cw, ch, ow, oh, matrix, full_range);
      assert_eq!((hh, ss, vv), want, "411p HSV fr={full_range} {matrix:?}");
      assert_eq!(scratch_len, 0, "411p HSV-only RGB-free");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv410p_hsv_only_vertical_chroma_upsample() {
  // A height-preserving (or lightly height-reduced) downscale upsamples the
  // 4:1:0 chroma plane vertically: one chroma feed finalises up to
  // chroma_vsub = 4 output rows before the matching luma rows arrive, so the
  // join's staging ring must hold 4 rows. out_h up to 4 * chroma_src_h (16);
  // out_h greater than 2 * ceil(h/4) = 8 is the lead the old 2-slot ring
  // mis-paired. Oracle: each plane area-resampled independently (no ring),
  // then converted per row.
  let (w, h) = (16usize, 16usize);
  let (cw, ch) = (w / 4, h / 4);
  let yp = plane(w, h, 1);
  let up = plane(cw, ch, 2);
  let vp = plane(cw, ch, 3);
  for &(ow, oh) in &[(8usize, 16usize), (4, 16), (16, 16), (8, 12), (12, 16)] {
    for &full_range in &[false, true] {
      for &matrix in &MATRICES {
        let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
        let scratch_len = {
          let frame = Yuv410pFrame::new(
            &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
          );
          let mut sink =
            MixedSinker::<Yuv410p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
              .unwrap()
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
          yuv410p_to(&frame, full_range, matrix, &mut sink).unwrap();
          sink.rgb_scratch.len()
        };
        let yb = area_resample_plane(&ResamplePlan::area(w, h, ow, oh).unwrap(), &yp, w, h);
        let chroma_plan = ResamplePlan::area(cw, h.div_ceil(4), ow, oh).unwrap();
        let ub = area_resample_plane(&chroma_plan, &up, cw, ch);
        let vb = area_resample_plane(&chroma_plan, &vp, cw, ch);
        let (mut rh, mut rs, mut rv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
        for oy in 0..oh {
          yuv_444_to_hsv_row(
            &yb[oy * ow..(oy + 1) * ow],
            &ub[oy * ow..(oy + 1) * ow],
            &vb[oy * ow..(oy + 1) * ow],
            &mut rh[oy * ow..(oy + 1) * ow],
            &mut rs[oy * ow..(oy + 1) * ow],
            &mut rv[oy * ow..(oy + 1) * ow],
            ow,
            matrix,
            full_range,
            true,
          );
        }
        assert_eq!(
          (hh, ss, vv),
          (rh, rs, rv),
          "410p HSV vertical-chroma-upsample {ow}x{oh} fr={full_range} {matrix:?}"
        );
        assert_eq!(scratch_len, 0, "410p HSV-only RGB-free {ow}x{oh}");
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv410p_hsv_only_odd_height_chroma_weighting() {
  // 4:1:0 height may be non-multiple-of-4: the final chroma row then covers
  // only 1..=3 trailing luma rows and must be weighted by that coverage, not
  // as a full 4-row cell (the naive equal-height chroma plan over-weights it).
  // Oracle: expand each chroma sample to its [4c, 4c+4) ∩ [0, h) luma rows (and
  // 4 luma cols — 4:1:0 width is a multiple of 4), then area-resample the
  // luma-resolution plane exactly like Y — independent of area_chroma_410.
  let expand = |c: &[u8], cw: usize, w: usize, h: usize| -> Vec<u8> {
    (0..w * h)
      .map(|i| c[(i / w / 4) * cw + (i % w) / 4])
      .collect()
  };
  for &(w, h) in &[(16usize, 5usize), (16, 6), (16, 7), (8, 5)] {
    let (cw, ch) = (w / 4, h.div_ceil(4));
    let yp = plane(w, h, 1);
    let up = plane(cw, ch, 2);
    let vp = plane(cw, ch, 3);
    let ue = expand(&up, cw, w, h);
    let ve = expand(&vp, cw, w, h);
    for &(ow, oh) in &[(w, h), (w / 2, h), (w / 2, 2), (4, 1)] {
      for &full_range in &[false, true] {
        for &matrix in &MATRICES {
          let (mut hh, mut ss, mut vv) =
            (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
          let scratch_len = {
            let frame = Yuv410pFrame::new(
              &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
            );
            let mut sink = MixedSinker::<Yuv410p, AreaResampler>::with_resampler(
              w,
              h,
              AreaResampler::to(ow, oh),
            )
            .unwrap()
            .with_hsv(&mut hh, &mut ss, &mut vv)
            .unwrap();
            yuv410p_to(&frame, full_range, matrix, &mut sink).unwrap();
            sink.rgb_scratch.len()
          };
          let lp = ResamplePlan::area(w, h, ow, oh).unwrap();
          let yb = area_resample_plane(&lp, &yp, w, h);
          let ub = area_resample_plane(&lp, &ue, w, h);
          let vb = area_resample_plane(&lp, &ve, w, h);
          let (mut rh, mut rs, mut rv) =
            (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
          for oy in 0..oh {
            yuv_444_to_hsv_row(
              &yb[oy * ow..(oy + 1) * ow],
              &ub[oy * ow..(oy + 1) * ow],
              &vb[oy * ow..(oy + 1) * ow],
              &mut rh[oy * ow..(oy + 1) * ow],
              &mut rs[oy * ow..(oy + 1) * ow],
              &mut rv[oy * ow..(oy + 1) * ow],
              ow,
              matrix,
              full_range,
              true,
            );
          }
          assert_eq!(
            (hh, ss, vv),
            (rh, rs, rv),
            "410p HSV odd-height {w}x{h}->{ow}x{oh} fr={full_range} {matrix:?}"
          );
          assert_eq!(scratch_len, 0, "410p HSV-only RGB-free {w}x{h}->{ow}x{oh}");
        }
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_hsv_only_odd_width_chroma_weighting() {
  // 4:1:1 width may be non-multiple-of-4: the final chroma column then covers
  // only 1..=3 trailing luma columns and must be weighted by that coverage,
  // not as a full 4-column cell (the horizontal analog of the 4:1:0 partial
  // row). Oracle: expand each chroma sample to its [4c, 4c+4) ∩ [0, w) luma
  // columns (full height), then area-resample like Y — independent of
  // area_chroma_411.
  let expand = |c: &[u8], cw: usize, w: usize, h: usize| -> Vec<u8> {
    (0..w * h).map(|i| c[(i / w) * cw + (i % w) / 4]).collect()
  };
  for &(w, h) in &[(13usize, 8usize), (14, 8), (15, 8), (5, 4)] {
    let (cw, ch) = (w.div_ceil(4), h);
    let yp = plane(w, h, 1);
    let up = plane(cw, ch, 2);
    let vp = plane(cw, ch, 3);
    let ue = expand(&up, cw, w, h);
    let ve = expand(&vp, cw, w, h);
    for &(ow, oh) in &[(w, h), (w, h / 2), (3, h), (2, 1)] {
      for &full_range in &[false, true] {
        for &matrix in &MATRICES {
          let (mut hh, mut ss, mut vv) =
            (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
          let scratch_len = {
            let frame = Yuv411pFrame::new(
              &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
            );
            let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
              w,
              h,
              AreaResampler::to(ow, oh),
            )
            .unwrap()
            .with_hsv(&mut hh, &mut ss, &mut vv)
            .unwrap();
            yuv411p_to(&frame, full_range, matrix, &mut sink).unwrap();
            sink.rgb_scratch.len()
          };
          let lp = ResamplePlan::area(w, h, ow, oh).unwrap();
          let yb = area_resample_plane(&lp, &yp, w, h);
          let ub = area_resample_plane(&lp, &ue, w, h);
          let vb = area_resample_plane(&lp, &ve, w, h);
          let (mut rh, mut rs, mut rv) =
            (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
          for oy in 0..oh {
            yuv_444_to_hsv_row(
              &yb[oy * ow..(oy + 1) * ow],
              &ub[oy * ow..(oy + 1) * ow],
              &vb[oy * ow..(oy + 1) * ow],
              &mut rh[oy * ow..(oy + 1) * ow],
              &mut rs[oy * ow..(oy + 1) * ow],
              &mut rv[oy * ow..(oy + 1) * ow],
              ow,
              matrix,
              full_range,
              true,
            );
          }
          assert_eq!(
            (hh, ss, vv),
            (rh, rs, rv),
            "411p HSV odd-width {w}x{h}->{ow}x{oh} fr={full_range} {matrix:?}"
          );
          assert_eq!(scratch_len, 0, "411p HSV-only RGB-free {w}x{h}->{ow}x{oh}");
        }
      }
    }
  }
}

// ===== Structural: luma + HSV-only is RGB-free; RGB + HSV still stages =====

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_plus_hsv_only_rowstage_is_rgb_free_and_matches_native() {
  // luma + HSV (no RGB): native Y luma AND RGB-free YUV-domain HSV, with no
  // source-width RGB scratch — and bit-identical to the native tier.
  let (w, h) = (12usize, 12usize);
  let (cw, ch) = (w / 2, h / 2);
  let yp = plane(w, h, 7);
  let up = plane(cw, ch, 9);
  let vp = plane(cw, ch, 11);
  let (ow, oh) = (6usize, 6usize);
  let run = |native: bool| {
    let mut luma = vec![0u8; ow * oh];
    let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
    let scratch_len = {
      let frame = Yuv420pFrame::new(
        &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
      );
      let mut sink =
        MixedSinker::<Yuv420p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
          .unwrap();
      sink.set_native(native);
      let mut sink = sink
        .with_luma(&mut luma)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv420p_to(&frame, true, KernelMatrix::Bt709, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    (luma, (hh, ss, vv), scratch_len)
  };
  let (n_luma, n_hsv, n_scratch) = run(true);
  let (r_luma, r_hsv, r_scratch) = run(false);
  assert_eq!(n_luma, r_luma, "luma bit-identical native-vs-rowstage");
  assert_eq!(n_hsv, r_hsv, "hsv bit-identical native-vs-rowstage");
  assert_eq!(n_scratch, 0, "native luma+HSV RGB-free");
  assert_eq!(r_scratch, 0, "row-stage luma+HSV RGB-free");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_plus_hsv_rowstage_still_stages_rgb_scratch() {
  // The RGB-staged path is unchanged: a sink with RGB + HSV (row-stage)
  // DOES grow the source-width RGB scratch (HSV derives off the binned RGB).
  let (w, h) = (12usize, 12usize);
  let (cw, ch) = (w / 2, h / 2);
  let yp = plane(w, h, 1);
  let up = plane(cw, ch, 2);
  let vp = plane(cw, ch, 3);
  let (ow, oh) = (6usize, 6usize);
  let mut rgb = vec![0u8; ow * oh * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let scratch_len = {
    let frame = Yuv420pFrame::new(
      &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
    );
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    yuv420p_to(&frame, true, KernelMatrix::Bt601, &mut sink).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len,
    w * 3,
    "RGB + HSV row-stage stages the source-width RGB scratch (unchanged)"
  );
}

// ===== Reuse across frames: the join resets cleanly =======================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_only_rowstage_reused_sink_resets_between_frames() {
  let (w, h) = (12usize, 12usize);
  let (cw, ch) = (w / 2, h / 2);
  let yp = plane(w, h, 5);
  let up = plane(cw, ch, 6);
  let vp = plane(cw, ch, 8);
  let (ow, oh) = (6usize, 6usize);
  let frame = Yuv420pFrame::new(
    &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
  );

  // Reference: a FRESH single-frame sink's output.
  let (mut rh, mut rs, mut rv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_hsv(&mut rh, &mut rs, &mut rv)
        .unwrap();
    yuv420p_to(&frame, true, KernelMatrix::Bt601, &mut sink).unwrap();
  }

  // Two frames in a row through the same sink; the walker calls begin_frame
  // each time, so the second frame must not trip an out-of-sequence row (the
  // join's Y stream is reset in begin_frame) and must reproduce the
  // fresh-sink output byte-for-byte.
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
      .unwrap()
      .with_native(false)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
  yuv420p_to(&frame, true, KernelMatrix::Bt601, &mut sink).unwrap();
  yuv420p_to(&frame, true, KernelMatrix::Bt601, &mut sink).unwrap();
  drop(sink);
  assert_eq!(
    (hh, ss, vv),
    (rh, rs, rv),
    "reused-sink second frame matches a fresh sink (reset worked)"
  );
}
