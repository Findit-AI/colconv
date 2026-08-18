//! #263 PR 1 — direct planar 8-bit YUV→HSV kernels + RGB-free routing.
//!
//! Two concerns:
//!
//! 1. **Row-kernel parity** — each `yuv_*_to_hsv_row` dispatcher must be
//!    byte-identical to `rgb_to_hsv_row(yuv_*_to_rgb_row(...))` within a
//!    tier (they share the RGB intermediate and the same per-tier HSV),
//!    for both the scalar path (`use_simd = false`) and the host SIMD
//!    path (`use_simd = true`).
//! 2. **Structural** — a planar-YUV sink with ONLY `with_hsv()` (no RGB /
//!    RGBA) must not grow the source-width RGB scratch.

use super::*;
use crate::row::{
  rgb_to_hsv_row, yuv_410_to_hsv_row, yuv_410_to_rgb_row, yuv_411_to_hsv_row, yuv_411_to_rgb_row,
  yuv_420_to_hsv_row, yuv_420_to_rgb_row, yuv_444_to_hsv_row, yuv_444_to_rgb_row,
};

const MATRICES: [KernelMatrix; 3] = [
  KernelMatrix::Bt601,
  KernelMatrix::Bt709,
  KernelMatrix::Bt2020Ncl,
];

/// A non-trivial, non-gray pseudo-random byte at a given position so the
/// HSV hue / saturation branches are all exercised (delta != 0, every
/// `v == r / g / b` arm) rather than the degenerate gray fast-path.
fn pat(i: usize, salt: usize) -> u8 {
  ((i
    .wrapping_mul(37)
    .wrapping_add(salt.wrapping_mul(101))
    .wrapping_add(11))
    & 0xFF) as u8
}

/// Reference HSV: the explicit `YUV → RGB → HSV` via the same dispatcher
/// tier, into a freshly-staged RGB row. The direct kernel must reproduce
/// this bit-for-bit.
fn ref_hsv_from_rgb(rgb: &[u8], w: usize, use_simd: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut h = std::vec![0u8; w];
  let mut s = std::vec![0u8; w];
  let mut v = std::vec![0u8; w];
  rgb_to_hsv_row(rgb, &mut h, &mut s, &mut v, w, use_simd);
  (h, s, v)
}

macro_rules! assert_planes_eq {
  ($got:expr, $want:expr, $ctx:expr) => {{
    let (gh, gs, gv) = &$got;
    let (wh, ws, wv) = &$want;
    assert_eq!(gh, wh, "H mismatch ({})", $ctx);
    assert_eq!(gs, ws, "S mismatch ({})", $ctx);
    assert_eq!(gv, wv, "V mismatch ({})", $ctx);
  }};
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv_420_hsv_row_matches_rgb_then_hsv() {
  // 4:2:0 / 4:2:2 share this kernel; even widths incl. an odd-half tail.
  for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
    let y: Vec<u8> = (0..w).map(|i| pat(i, 1)).collect();
    let uh: Vec<u8> = (0..w / 2).map(|i| pat(i, 2)).collect();
    let vh: Vec<u8> = (0..w / 2).map(|i| pat(i, 3)).collect();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          yuv_420_to_rgb_row(&y, &uh, &vh, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          yuv_420_to_hsv_row(
            &y, &uh, &vh, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("420 w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
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
fn yuv_444_hsv_row_matches_rgb_then_hsv() {
  // 4:4:4 / 4:4:0 share this kernel; arbitrary widths incl. odd.
  for &w in &[1usize, 3, 5, 15, 16, 31, 64, 65] {
    let y: Vec<u8> = (0..w).map(|i| pat(i, 1)).collect();
    let u: Vec<u8> = (0..w).map(|i| pat(i, 2)).collect();
    let v_in: Vec<u8> = (0..w).map(|i| pat(i, 3)).collect();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          yuv_444_to_rgb_row(&y, &u, &v_in, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          yuv_444_to_hsv_row(
            &y, &u, &v_in, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("444 w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
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
fn yuv_410_hsv_row_matches_rgb_then_hsv() {
  // 4:1:0 — quarter-width chroma; widths are multiples of 4.
  for &w in &[4usize, 8, 12, 16, 64, 68] {
    let y: Vec<u8> = (0..w).map(|i| pat(i, 1)).collect();
    let uq: Vec<u8> = (0..w / 4).map(|i| pat(i, 2)).collect();
    let vq: Vec<u8> = (0..w / 4).map(|i| pat(i, 3)).collect();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          yuv_410_to_rgb_row(&y, &uq, &vq, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          yuv_410_to_hsv_row(
            &y, &uq, &vq, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("410 w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
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
fn yuv_411_hsv_row_matches_rgb_then_hsv() {
  // 4:1:1 — quarter-width chroma with FFmpeg ceil-shift tails; include
  // widths NOT divisible by 4 to exercise the partial-group path.
  for &w in &[4usize, 5, 7, 8, 13, 16, 64, 67] {
    let y: Vec<u8> = (0..w).map(|i| pat(i, 1)).collect();
    let cw = w.div_ceil(4);
    let uq: Vec<u8> = (0..cw).map(|i| pat(i, 2)).collect();
    let vq: Vec<u8> = (0..cw).map(|i| pat(i, 3)).collect();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          yuv_411_to_rgb_row(&y, &uq, &vq, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          yuv_411_to_hsv_row(
            &y, &uq, &vq, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("411 w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
        }
      }
    }
  }
}

// ---- Structural: HSV-only attaches no source-width RGB scratch ----------

/// Planar plane bytes for one format (a varied, non-gray pattern).
fn plane(width: usize, height: usize, salt: usize) -> Vec<u8> {
  (0..width * height).map(|i| pat(i, salt)).collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_hsv_only_grows_no_rgb_scratch() {
  let (w, h) = (16usize, 8usize);
  let yp = plane(w, h, 1);
  let up = plane(w / 2, h / 2, 2);
  let vp = plane(w / 2, h / 2, 3);
  let src = Yuv420pFrame::new(
    &yp,
    &up,
    &vp,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );

  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuv420p_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "Yuv420p HSV-only must not grow the source-width RGB scratch"
  );

  // Cross-check row 0 against the explicit YUV→RGB→HSV reference.
  let mut rgb0 = std::vec![0u8; w * 3];
  yuv_420_to_rgb_row(
    &yp[..w],
    &up[..w / 2],
    &vp[..w / 2],
    &mut rgb0,
    w,
    KernelMatrix::Bt601,
    true,
    true,
  );
  let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
  assert_eq!(&hh[..w], &rh[..], "row 0 H");
  assert_eq!(&ss[..w], &rs[..], "row 0 S");
  assert_eq!(&vv[..w], &rv[..], "row 0 V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_luma_plus_hsv_only_is_correct_and_rgb_free() {
  // with_luma() + with_hsv(), no RGB: native Y luma AND direct HSV, with
  // no source-width RGB scratch.
  let (w, h) = (16usize, 8usize);
  let yp = plane(w, h, 7);
  let up = plane(w / 2, h / 2, 9);
  let vp = plane(w / 2, h / 2, 11);
  let src = Yuv420pFrame::new(
    &yp,
    &up,
    &vp,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );

  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuv420p_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    sink.rgb_scratch.len()
  };

  assert_eq!(
    scratch_len, 0,
    "luma + HSV (no RGB) must not grow the RGB scratch"
  );
  // Luma is the Y plane verbatim.
  assert_eq!(luma, yp, "luma must equal the native Y plane");
  // HSV row 0 matches the two-step reference.
  let mut rgb0 = std::vec![0u8; w * 3];
  yuv_420_to_rgb_row(
    &yp[..w],
    &up[..w / 2],
    &vp[..w / 2],
    &mut rgb0,
    w,
    KernelMatrix::Bt709,
    true,
    true,
  );
  let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
  assert_eq!(&hh[..w], &rh[..], "row 0 H");
  assert_eq!(&ss[..w], &rs[..], "row 0 S");
  assert_eq!(&vv[..w], &rv[..], "row 0 V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn planar_yuv_hsv_only_grows_no_rgb_scratch_all_formats() {
  let (w, h) = (16usize, 8usize);

  // 4:2:2 — half-width chroma, full height.
  {
    let yp = plane(w, h, 1);
    let up = plane(w / 2, h, 2);
    let vp = plane(w / 2, h, 3);
    let src = Yuv422pFrame::new(
      &yp,
      &up,
      &vp,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
    );
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Yuv422p>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv422p_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv422p HSV-only RGB-free");
  }

  // 4:4:4 — full-width chroma.
  {
    let yp = plane(w, h, 1);
    let up = plane(w, h, 2);
    let vp = plane(w, h, 3);
    let src = Yuv444pFrame::new(
      &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
    );
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Yuv444p>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv444p_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv444p HSV-only RGB-free");
  }

  // 4:4:0 — full-width chroma, half height.
  {
    let yp = plane(w, h, 1);
    let up = plane(w, h / 2, 2);
    let vp = plane(w, h / 2, 3);
    let src = Yuv440pFrame::new(
      &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
    );
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Yuv440p>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv440p_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv440p HSV-only RGB-free");
  }

  // 4:1:0 — quarter-width chroma, quarter height.
  {
    let yp = plane(w, h, 1);
    let up = plane(w / 4, h / 4, 2);
    let vp = plane(w / 4, h / 4, 3);
    let src = Yuv410pFrame::new(
      &yp,
      &up,
      &vp,
      w as u32,
      h as u32,
      w as u32,
      (w / 4) as u32,
      (w / 4) as u32,
    );
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Yuv410p>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv410p_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv410p HSV-only RGB-free");
  }

  // 4:1:1 — quarter-width chroma, full height.
  {
    let yp = plane(w, h, 1);
    let up = plane(w / 4, h, 2);
    let vp = plane(w / 4, h, 3);
    let src = Yuv411pFrame::new(
      &yp,
      &up,
      &vp,
      w as u32,
      h as u32,
      w as u32,
      (w / 4) as u32,
      (w / 4) as u32,
    );
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Yuv411p>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv411p_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv411p HSV-only RGB-free");
  }
}
