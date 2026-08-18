//! #263 PR 5 — direct packed YUV (yuyv422 / uyvy422 / yvyu422 +
//! uyyvyy411) YUV→HSV kernels + RGB-free routing.
//!
//! Two concerns:
//!
//! 1. **Row-kernel parity** — each `*_to_hsv_row` dispatcher must be
//!    byte-identical to `rgb_to_hsv_row(*_to_rgb_row(...))` within a tier
//!    (they share the RGB intermediate and the same per-tier HSV), for
//!    both the scalar path (`use_simd = false`) and the host SIMD path
//!    (`use_simd = true`), across BT.601 / 709 / 2020 × full / limited ×
//!    a range of widths honoring each format's width constraint (4:2:2
//!    even, 4:1:1 multiple of 4).
//! 2. **Structural** — a packed-YUV sink with ONLY `with_hsv()` (no RGB /
//!    RGBA) must not grow the source-width RGB scratch
//!    (`rgb_scratch.len() == 0`), and its HSV output equals the explicit
//!    two-step `*_to_rgb_row` → `rgb_to_hsv_row` reference.

use super::*;
use crate::row::{
  rgb_to_hsv_row, uyvy422_to_hsv_row, uyvy422_to_rgb_row, uyyvyy411_to_hsv_row,
  uyyvyy411_to_rgb_row, yuyv422_to_hsv_row, yuyv422_to_rgb_row, yvyu422_to_hsv_row,
  yvyu422_to_rgb_row,
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

/// Reference HSV: the explicit `packed → RGB → HSV` via the same
/// dispatcher tier, into a freshly-staged RGB row. The direct kernel must
/// reproduce this bit-for-bit.
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

// ---- Row-kernel parity (packed 4:2:2) ----------------------------------
//
// The parity tests are layout-agnostic: both the direct `*_to_hsv_row`
// kernel and the `*_to_rgb_row` → `rgb_to_hsv_row` reference read the
// SAME packed buffer through the SAME format kernel, so a pseudo-random
// fill exercises the full pipeline regardless of which byte is Y vs
// chroma. Widths are even (the 4:2:2 constraint) and include an odd-half
// tail (`w / 2` odd) plus widths above / below the SIMD block.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_hsv_row_matches_rgb_then_hsv() {
  for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
    let packed: Vec<u8> = (0..w * 2).map(|i| pat(i, 1)).collect();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          yuyv422_to_rgb_row(&packed, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          yuyv422_to_hsv_row(
            &packed, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("yuyv422 w={w} {matrix:?} full={full_range} simd={use_simd}")
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
fn uyvy422_hsv_row_matches_rgb_then_hsv() {
  for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
    let packed: Vec<u8> = (0..w * 2).map(|i| pat(i, 2)).collect();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          uyvy422_to_rgb_row(&packed, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          uyvy422_to_hsv_row(
            &packed, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("uyvy422 w={w} {matrix:?} full={full_range} simd={use_simd}")
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
fn yvyu422_hsv_row_matches_rgb_then_hsv() {
  for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
    let packed: Vec<u8> = (0..w * 2).map(|i| pat(i, 3)).collect();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          yvyu422_to_rgb_row(&packed, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          yvyu422_to_hsv_row(
            &packed, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("yvyu422 w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
        }
      }
    }
  }
}

// ---- Row-kernel parity (packed 4:1:1) ----------------------------------
//
// UYYVYY411 packs 6 bytes per 4-pixel block, so the packed row is
// `width * 3 / 2` bytes. Widths are multiples of 4 (the 4:1:1 constraint)
// and include widths above / below the SIMD block.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_hsv_row_matches_rgb_then_hsv() {
  for &w in &[4usize, 8, 12, 28, 64, 68] {
    let packed: Vec<u8> = (0..w * 3 / 2).map(|i| pat(i, 4)).collect();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          uyyvyy411_to_rgb_row(&packed, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          uyyvyy411_to_hsv_row(
            &packed, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("uyyvyy411 w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
        }
      }
    }
  }
}

// ---- Structural: HSV-only attaches no source-width RGB scratch ----------

/// Pseudo-random packed bytes (a varied, non-gray pattern).
fn packed_bytes(len: usize, salt: usize) -> Vec<u8> {
  (0..len).map(|i| pat(i, salt)).collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_hsv_only_grows_no_rgb_scratch() {
  let (w, h) = (16usize, 8usize);
  let buf = packed_bytes(2 * w * h, 1);
  let src = Yuyv422Frame::new(&buf, w as u32, h as u32, (2 * w) as u32);

  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Yuyv422>::new(w, h)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuyv422_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "Yuyv422 HSV-only must not grow the source-width RGB scratch"
  );

  // Cross-check row 0 against the explicit packed→RGB→HSV reference.
  let mut rgb0 = std::vec![0u8; w * 3];
  yuyv422_to_rgb_row(&buf[..2 * w], &mut rgb0, w, KernelMatrix::Bt601, true, true);
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
fn yuyv422_luma_plus_hsv_only_is_correct_and_rgb_free() {
  // with_luma() + with_hsv(), no RGB: native-Y luma AND direct HSV, with
  // no source-width RGB scratch.
  let (w, h) = (16usize, 8usize);
  let buf = packed_bytes(2 * w * h, 7);
  let src = Yuyv422Frame::new(&buf, w as u32, h as u32, (2 * w) as u32);

  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Yuyv422>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuyv422_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "luma + HSV (no RGB) must not grow the RGB scratch"
  );
  // HSV row 0 matches the two-step reference.
  let mut rgb0 = std::vec![0u8; w * 3];
  yuyv422_to_rgb_row(&buf[..2 * w], &mut rgb0, w, KernelMatrix::Bt709, true, true);
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
fn packed_yuv_hsv_only_grows_no_rgb_scratch_all_formats() {
  let (w, h) = (16usize, 8usize);

  // UYVY422 — 4:2:2, `2 * width` bytes/row.
  {
    let buf = packed_bytes(2 * w * h, 2);
    let src = Uyvy422Frame::new(&buf, w as u32, h as u32, (2 * w) as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Uyvy422>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      uyvy422_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Uyvy422 HSV-only RGB-free");
    let mut rgb0 = std::vec![0u8; w * 3];
    uyvy422_to_rgb_row(&buf[..2 * w], &mut rgb0, w, KernelMatrix::Bt601, true, true);
    let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
    assert_eq!(&hh[..w], &rh[..], "uyvy422 row 0 H");
    assert_eq!(&ss[..w], &rs[..], "uyvy422 row 0 S");
    assert_eq!(&vv[..w], &rv[..], "uyvy422 row 0 V");
  }

  // YVYU422 — 4:2:2, UV-swapped, `2 * width` bytes/row.
  {
    let buf = packed_bytes(2 * w * h, 3);
    let src = Yvyu422Frame::new(&buf, w as u32, h as u32, (2 * w) as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Yvyu422>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yvyu422_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yvyu422 HSV-only RGB-free");
    let mut rgb0 = std::vec![0u8; w * 3];
    yvyu422_to_rgb_row(&buf[..2 * w], &mut rgb0, w, KernelMatrix::Bt601, true, true);
    let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
    assert_eq!(&hh[..w], &rh[..], "yvyu422 row 0 H");
    assert_eq!(&ss[..w], &rs[..], "yvyu422 row 0 S");
    assert_eq!(&vv[..w], &rv[..], "yvyu422 row 0 V");
  }

  // UYYVYY411 — 4:1:1, `width * 3 / 2` bytes/row.
  {
    let row_bytes = w * 3 / 2;
    let buf = packed_bytes(row_bytes * h, 4);
    let src = Uyyvyy411Frame::new(&buf, w as u32, h as u32, row_bytes as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Uyyvyy411>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      uyyvyy411_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Uyyvyy411 HSV-only RGB-free");
    let mut rgb0 = std::vec![0u8; w * 3];
    uyyvyy411_to_rgb_row(
      &buf[..row_bytes],
      &mut rgb0,
      w,
      KernelMatrix::Bt601,
      true,
      true,
    );
    let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
    assert_eq!(&hh[..w], &rh[..], "uyyvyy411 row 0 H");
    assert_eq!(&ss[..w], &rs[..], "uyyvyy411 row 0 S");
    assert_eq!(&vv[..w], &rv[..], "uyyvyy411 row 0 V");
  }
}
