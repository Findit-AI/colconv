//! #263 PR 8 — route the YUVA sinks' `with_hsv()` RGB-free, reusing the
//! existing planar YUV→HSV kernels.
//!
//! YUVA is planar Y/U/V + an alpha plane; HSV is colour-only (alpha is
//! dropped), so a YUVA→HSV is byte-identical to the no-alpha planar
//! YUV→HSV on the same Y/U/V. The YUVA sinks' alpha-drop RGB path already
//! reuses the planar `yuv_*_to_rgb_row` / `yuv*p*_to_rgb_row_endian`
//! dispatchers, so HSV reuses the MATCHING planar HSV kernel — no new
//! kernels. This PR wires that routing in the six sink direct paths
//! (8-bit + high-bit, for 4:2:0 / 4:2:2 / 4:4:4).
//!
//! Two concerns:
//!
//! 1. **Sink parity** — a YUVA sink with ONLY `with_hsv()` must produce
//!    output byte-identical to the SAME YUVA sink with `with_rgb()`
//!    followed by `rgb_to_hsv_row` on that RGB, for both tiers
//!    (`use_simd` on/off), every matrix, range, and width.
//! 2. **Structural** — a YUVA sink with ONLY `with_hsv()` (and
//!    `with_luma()` + `with_hsv()`) must not grow the source-width RGB
//!    scratch.

use super::*;
use crate::row::{rgb_to_hsv_row, yuv_420_to_rgb_row, yuv_444_to_rgb_row};

const MATRICES: [KernelMatrix; 3] = [
  KernelMatrix::Bt601,
  KernelMatrix::Bt709,
  KernelMatrix::Bt2020Ncl,
];

/// A non-trivial, non-gray pseudo-random byte so the HSV hue /
/// saturation branches are all exercised (delta != 0, every `v == r /
/// g / b` arm) rather than the degenerate gray fast-path.
fn pat(i: usize, salt: usize) -> u8 {
  ((i
    .wrapping_mul(37)
    .wrapping_add(salt.wrapping_mul(101))
    .wrapping_add(11))
    & 0xFF) as u8
}

/// HSV reference for one frame: derive it from the YUVA sink's *RGB*
/// output by running `rgb_to_hsv_row` per row at the SAME tier. The
/// HSV-direct sink path must reproduce this bit-for-bit.
fn ref_hsv_from_rgb_frame(
  rgb: &[u8],
  w: usize,
  h: usize,
  use_simd: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  for y in 0..h {
    rgb_to_hsv_row(
      &rgb[y * w * 3..(y + 1) * w * 3],
      &mut hh[y * w..(y + 1) * w],
      &mut ss[y * w..(y + 1) * w],
      &mut vv[y * w..(y + 1) * w],
      w,
      use_simd,
    );
  }
  (hh, ss, vv)
}

macro_rules! assert_hsv_eq {
  ($got:expr, $want:expr, $ctx:expr) => {{
    let (gh, gs, gv) = &$got;
    let (wh, ws, wv) = &$want;
    assert_eq!(gh, wh, "H mismatch ({})", $ctx);
    assert_eq!(gs, ws, "S mismatch ({})", $ctx);
    assert_eq!(gv, wv, "V mismatch ({})", $ctx);
  }};
}

// ---- 8-bit sink parity (HSV-only == with_rgb then rgb_to_hsv) ----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_sink_hsv_only_matches_rgb_then_hsv() {
  // 4:2:0 — half-width, half-height chroma; even widths incl. an odd half.
  for &(w, h) in &[(2usize, 2usize), (6, 4), (16, 8), (30, 6)] {
    let yp: Vec<u8> = (0..w * h).map(|i| pat(i, 1)).collect();
    let up: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| pat(i, 2)).collect();
    let vp: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| pat(i, 3)).collect();
    let ap: Vec<u8> = (0..w * h).map(|i| pat(i, 4)).collect();
    let src = Yuva420pFrame::try_new(
      &yp,
      &up,
      &vp,
      &ap,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
      w as u32,
    )
    .unwrap();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          // Reference: with_rgb on the YUVA sink, then rgb_to_hsv_row.
          let mut rgb = std::vec![0u8; w * h * 3];
          {
            let mut s = MixedSinker::<Yuva420p>::new(w, h)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap();
            yuva420p_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          let want = ref_hsv_from_rgb_frame(&rgb, w, h, use_simd);

          // HSV-direct: with_hsv only on the YUVA sink.
          let mut hh = std::vec![0u8; w * h];
          let mut ss = std::vec![0u8; w * h];
          let mut vv = std::vec![0u8; w * h];
          {
            let mut s = MixedSinker::<Yuva420p>::new(w, h)
              .with_simd(use_simd)
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
            yuva420p_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          assert_hsv_eq!(
            (hh, ss, vv),
            want,
            std::format!("yuva420p w={w} h={h} {matrix:?} full={full_range} simd={use_simd}")
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
fn yuva422p_sink_hsv_only_matches_rgb_then_hsv() {
  // 4:2:2 — half-width chroma, full height.
  for &(w, h) in &[(2usize, 3usize), (6, 4), (16, 8), (30, 5)] {
    let yp: Vec<u8> = (0..w * h).map(|i| pat(i, 1)).collect();
    let up: Vec<u8> = (0..(w / 2) * h).map(|i| pat(i, 2)).collect();
    let vp: Vec<u8> = (0..(w / 2) * h).map(|i| pat(i, 3)).collect();
    let ap: Vec<u8> = (0..w * h).map(|i| pat(i, 4)).collect();
    let src = Yuva422pFrame::try_new(
      &yp,
      &up,
      &vp,
      &ap,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
      w as u32,
    )
    .unwrap();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * h * 3];
          {
            let mut s = MixedSinker::<Yuva422p>::new(w, h)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap();
            yuva422p_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          let want = ref_hsv_from_rgb_frame(&rgb, w, h, use_simd);

          let mut hh = std::vec![0u8; w * h];
          let mut ss = std::vec![0u8; w * h];
          let mut vv = std::vec![0u8; w * h];
          {
            let mut s = MixedSinker::<Yuva422p>::new(w, h)
              .with_simd(use_simd)
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
            yuva422p_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          assert_hsv_eq!(
            (hh, ss, vv),
            want,
            std::format!("yuva422p w={w} h={h} {matrix:?} full={full_range} simd={use_simd}")
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
fn yuva444p_sink_hsv_only_matches_rgb_then_hsv() {
  // 4:4:4 — full-width chroma; arbitrary widths incl. odd.
  for &(w, h) in &[(1usize, 1usize), (5, 4), (16, 8), (31, 5)] {
    let yp: Vec<u8> = (0..w * h).map(|i| pat(i, 1)).collect();
    let up: Vec<u8> = (0..w * h).map(|i| pat(i, 2)).collect();
    let vp: Vec<u8> = (0..w * h).map(|i| pat(i, 3)).collect();
    let ap: Vec<u8> = (0..w * h).map(|i| pat(i, 4)).collect();
    let src = Yuva444pFrame::try_new(
      &yp, &up, &vp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
    )
    .unwrap();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * h * 3];
          {
            let mut s = MixedSinker::<Yuva444p>::new(w, h)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap();
            yuva444p_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          let want = ref_hsv_from_rgb_frame(&rgb, w, h, use_simd);

          let mut hh = std::vec![0u8; w * h];
          let mut ss = std::vec![0u8; w * h];
          let mut vv = std::vec![0u8; w * h];
          {
            let mut s = MixedSinker::<Yuva444p>::new(w, h)
              .with_simd(use_simd)
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
            yuva444p_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          assert_hsv_eq!(
            (hh, ss, vv),
            want,
            std::format!("yuva444p w={w} h={h} {matrix:?} full={full_range} simd={use_simd}")
          );
        }
      }
    }
  }
}

// ---- High-bit sink parity (one depth per family) -----------------------
//
// High-bit YUVA stores LOGICAL u16 values directly (the default LE
// `BE = false`). Each family routes the HSV-direct path through the
// matching `yuv*p*_to_hsv_row_endian` kernel; the reference is the same
// sink's `with_rgb` output run through `rgb_to_hsv_row`.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_sink_hsv_only_matches_rgb_then_hsv() {
  // 4:2:0 high-bit (10-bit, max logical 1023).
  for &(w, h) in &[(2usize, 2usize), (16, 8), (30, 6)] {
    let yp: Vec<u16> = (0..w * h).map(|i| (pat(i, 1) as u16) << 2).collect();
    let up: Vec<u16> = (0..(w / 2) * (h / 2))
      .map(|i| (pat(i, 2) as u16) << 2)
      .collect();
    let vp: Vec<u16> = (0..(w / 2) * (h / 2))
      .map(|i| (pat(i, 3) as u16) << 2)
      .collect();
    let ap: Vec<u16> = (0..w * h).map(|i| (pat(i, 4) as u16) << 2).collect();
    let src = Yuva420p10Frame::try_new(
      &yp,
      &up,
      &vp,
      &ap,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
      w as u32,
    )
    .unwrap();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * h * 3];
          {
            let mut s = MixedSinker::<Yuva420p10>::new(w, h)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap();
            yuva420p10_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          let want = ref_hsv_from_rgb_frame(&rgb, w, h, use_simd);

          let mut hh = std::vec![0u8; w * h];
          let mut ss = std::vec![0u8; w * h];
          let mut vv = std::vec![0u8; w * h];
          {
            let mut s = MixedSinker::<Yuva420p10>::new(w, h)
              .with_simd(use_simd)
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
            yuva420p10_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          assert_hsv_eq!(
            (hh, ss, vv),
            want,
            std::format!("yuva420p10 w={w} h={h} {matrix:?} full={full_range} simd={use_simd}")
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
fn yuva422p12_sink_hsv_only_matches_rgb_then_hsv() {
  // 4:2:2 high-bit (12-bit, max logical 4095).
  for &(w, h) in &[(2usize, 3usize), (16, 8), (30, 5)] {
    let yp: Vec<u16> = (0..w * h).map(|i| (pat(i, 1) as u16) << 4).collect();
    let up: Vec<u16> = (0..(w / 2) * h).map(|i| (pat(i, 2) as u16) << 4).collect();
    let vp: Vec<u16> = (0..(w / 2) * h).map(|i| (pat(i, 3) as u16) << 4).collect();
    let ap: Vec<u16> = (0..w * h).map(|i| (pat(i, 4) as u16) << 4).collect();
    let src = Yuva422p12Frame::try_new(
      &yp,
      &up,
      &vp,
      &ap,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
      w as u32,
    )
    .unwrap();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * h * 3];
          {
            let mut s = MixedSinker::<Yuva422p12>::new(w, h)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap();
            yuva422p12_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          let want = ref_hsv_from_rgb_frame(&rgb, w, h, use_simd);

          let mut hh = std::vec![0u8; w * h];
          let mut ss = std::vec![0u8; w * h];
          let mut vv = std::vec![0u8; w * h];
          {
            let mut s = MixedSinker::<Yuva422p12>::new(w, h)
              .with_simd(use_simd)
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
            yuva422p12_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          assert_hsv_eq!(
            (hh, ss, vv),
            want,
            std::format!("yuva422p12 w={w} h={h} {matrix:?} full={full_range} simd={use_simd}")
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
fn yuva444p14_sink_hsv_only_matches_rgb_then_hsv() {
  // 4:4:4 high-bit (14-bit, max logical 16383).
  for &(w, h) in &[(1usize, 1usize), (16, 8), (31, 5)] {
    let yp: Vec<u16> = (0..w * h).map(|i| (pat(i, 1) as u16) << 6).collect();
    let up: Vec<u16> = (0..w * h).map(|i| (pat(i, 2) as u16) << 6).collect();
    let vp: Vec<u16> = (0..w * h).map(|i| (pat(i, 3) as u16) << 6).collect();
    let ap: Vec<u16> = (0..w * h).map(|i| (pat(i, 4) as u16) << 6).collect();
    let src = Yuva444p14Frame::try_new(
      &yp, &up, &vp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
    )
    .unwrap();
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * h * 3];
          {
            let mut s = MixedSinker::<Yuva444p14>::new(w, h)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap();
            yuva444p14_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          let want = ref_hsv_from_rgb_frame(&rgb, w, h, use_simd);

          let mut hh = std::vec![0u8; w * h];
          let mut ss = std::vec![0u8; w * h];
          let mut vv = std::vec![0u8; w * h];
          {
            let mut s = MixedSinker::<Yuva444p14>::new(w, h)
              .with_simd(use_simd)
              .with_hsv(&mut hh, &mut ss, &mut vv)
              .unwrap();
            yuva444p14_to(&src, full_range, s.set_kernel_matrix(matrix)).unwrap();
          }
          assert_hsv_eq!(
            (hh, ss, vv),
            want,
            std::format!("yuva444p14 w={w} h={h} {matrix:?} full={full_range} simd={use_simd}")
          );
        }
      }
    }
  }
}

// ---- HSV is colour-only: drops alpha (independent of the A plane) ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_hsv_only_is_independent_of_alpha() {
  // Two frames with identical Y/U/V but different alpha planes must yield
  // byte-identical HSV — confirming the HSV-direct path drops alpha.
  let (w, h) = (16usize, 8usize);
  let yp: Vec<u8> = (0..w * h).map(|i| pat(i, 1)).collect();
  let up: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| pat(i, 2)).collect();
  let vp: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| pat(i, 3)).collect();
  let a_zero = std::vec![0u8; w * h];
  let a_full = std::vec![0xFFu8; w * h];

  let mut got = [(); 2].map(|_| {
    (
      std::vec![0u8; w * h],
      std::vec![0u8; w * h],
      std::vec![0u8; w * h],
    )
  });
  for (slot, ap) in [&a_zero, &a_full].into_iter().enumerate() {
    let src = Yuva420pFrame::try_new(
      &yp,
      &up,
      &vp,
      ap,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
      w as u32,
    )
    .unwrap();
    let (hh, ss, vv) = &mut got[slot];
    let mut s = MixedSinker::<Yuva420p>::new(w, h)
      .with_hsv(hh, ss, vv)
      .unwrap();
    yuva420p_to(&src, true, s.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert_eq!(got[0].0, got[1].0, "H must ignore alpha");
  assert_eq!(got[0].1, got[1].1, "S must ignore alpha");
  assert_eq!(got[0].2, got[1].2, "V must ignore alpha");
}

// ---- Structural: HSV-only attaches no source-width RGB scratch ---------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva_sink_hsv_only_grows_no_rgb_scratch_all_families() {
  let (w, h) = (16usize, 8usize);
  let yp: Vec<u8> = (0..w * h).map(|i| pat(i, 1)).collect();
  let ap: Vec<u8> = (0..w * h).map(|i| pat(i, 4)).collect();

  // 4:2:0 — half-width, half-height chroma.
  {
    let up: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| pat(i, 2)).collect();
    let vp: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| pat(i, 3)).collect();
    let src = Yuva420pFrame::try_new(
      &yp,
      &up,
      &vp,
      &ap,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
      w as u32,
    )
    .unwrap();
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut s = MixedSinker::<Yuva420p>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuva420p_to(&src, true, s.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      s.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuva420p HSV-only RGB-free");
  }

  // 4:2:2 — half-width chroma, full height.
  {
    let up: Vec<u8> = (0..(w / 2) * h).map(|i| pat(i, 2)).collect();
    let vp: Vec<u8> = (0..(w / 2) * h).map(|i| pat(i, 3)).collect();
    let src = Yuva422pFrame::try_new(
      &yp,
      &up,
      &vp,
      &ap,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
      w as u32,
    )
    .unwrap();
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut s = MixedSinker::<Yuva422p>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuva422p_to(&src, true, s.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      s.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuva422p HSV-only RGB-free");
  }

  // 4:4:4 — full-width chroma.
  {
    let up: Vec<u8> = (0..w * h).map(|i| pat(i, 2)).collect();
    let vp: Vec<u8> = (0..w * h).map(|i| pat(i, 3)).collect();
    let src = Yuva444pFrame::try_new(
      &yp, &up, &vp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
    )
    .unwrap();
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut s = MixedSinker::<Yuva444p>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuva444p_to(&src, true, s.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      s.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuva444p HSV-only RGB-free");
  }

  // High-bit representative (4:4:4 14-bit).
  {
    let yp16: Vec<u16> = (0..w * h).map(|i| (pat(i, 1) as u16) << 6).collect();
    let up16: Vec<u16> = (0..w * h).map(|i| (pat(i, 2) as u16) << 6).collect();
    let vp16: Vec<u16> = (0..w * h).map(|i| (pat(i, 3) as u16) << 6).collect();
    let ap16: Vec<u16> = (0..w * h).map(|i| (pat(i, 4) as u16) << 6).collect();
    let src = Yuva444p14Frame::try_new(
      &yp16, &up16, &vp16, &ap16, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
    )
    .unwrap();
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut s = MixedSinker::<Yuva444p14>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuva444p14_to(&src, true, s.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      s.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuva444p14 HSV-only RGB-free");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_luma_plus_hsv_only_is_correct_and_rgb_free() {
  // with_luma() + with_hsv(), no RGB: native Y luma AND direct HSV, with
  // no source-width RGB scratch.
  let (w, h) = (16usize, 8usize);
  let yp: Vec<u8> = (0..w * h).map(|i| pat(i, 7)).collect();
  let up: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| pat(i, 9)).collect();
  let vp: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| pat(i, 11)).collect();
  let ap: Vec<u8> = (0..w * h).map(|i| pat(i, 13)).collect();
  let src = Yuva420pFrame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
    w as u32,
  )
  .unwrap();

  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut s = MixedSinker::<Yuva420p>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuva420p_to(&src, true, s.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    s.rgb_scratch.len()
  };

  assert_eq!(
    scratch_len, 0,
    "luma + HSV (no RGB) must not grow the RGB scratch"
  );
  // Luma is the Y plane verbatim (8-bit YUVA Y).
  assert_eq!(luma, yp, "luma must equal the native Y plane");
  // HSV row 0 matches the two-step reference (alpha dropped).
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
  let (rh, rs, rv) = {
    let mut h0 = std::vec![0u8; w];
    let mut s0 = std::vec![0u8; w];
    let mut v0 = std::vec![0u8; w];
    rgb_to_hsv_row(&rgb0, &mut h0, &mut s0, &mut v0, w, true);
    (h0, s0, v0)
  };
  assert_eq!(&hh[..w], &rh[..], "row 0 H");
  assert_eq!(&ss[..w], &rs[..], "row 0 S");
  assert_eq!(&vv[..w], &rv[..], "row 0 V");

  // Cross-check 4:4:4 row 0 too (full-width chroma kernel).
  let up4: Vec<u8> = (0..w * h).map(|i| pat(i, 9)).collect();
  let vp4: Vec<u8> = (0..w * h).map(|i| pat(i, 11)).collect();
  let src4 = Yuva444pFrame::try_new(
    &yp, &up4, &vp4, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();
  let mut hh4 = std::vec![0u8; w * h];
  let mut ss4 = std::vec![0u8; w * h];
  let mut vv4 = std::vec![0u8; w * h];
  {
    let mut s = MixedSinker::<Yuva444p>::new(w, h)
      .with_hsv(&mut hh4, &mut ss4, &mut vv4)
      .unwrap();
    yuva444p_to(&src4, true, s.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  let mut rgb04 = std::vec![0u8; w * 3];
  yuv_444_to_rgb_row(
    &yp[..w],
    &up4[..w],
    &vp4[..w],
    &mut rgb04,
    w,
    KernelMatrix::Bt709,
    true,
    true,
  );
  let mut h04 = std::vec![0u8; w];
  let mut s04 = std::vec![0u8; w];
  let mut v04 = std::vec![0u8; w];
  rgb_to_hsv_row(&rgb04, &mut h04, &mut s04, &mut v04, w, true);
  assert_eq!(&hh4[..w], &h04[..], "444 row 0 H");
  assert_eq!(&ss4[..w], &s04[..], "444 row 0 S");
  assert_eq!(&vv4[..w], &v04[..], "444 row 0 V");
}
