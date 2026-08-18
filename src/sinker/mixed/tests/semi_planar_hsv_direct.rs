//! #263 PR 2 — direct semi-planar NV YUV→HSV kernels + native-Y luma
//! kernel + RGB-free routing.
//!
//! Three concerns:
//!
//! 1. **Row-kernel parity** — each `nv*_to_hsv_row` dispatcher must be
//!    byte-identical to `rgb_to_hsv_row(nv*_to_rgb_row(...))` within a
//!    tier (they share the RGB intermediate and the same per-tier HSV),
//!    for both the scalar path (`use_simd = false`) and the host SIMD
//!    path (`use_simd = true`).
//! 2. **Native luma** — `nv_to_luma_row` reproduces the Y plane verbatim
//!    (the same bytes the sink's former inline `copy_from_slice`
//!    produced).
//! 3. **Structural** — an NV sink with ONLY `with_hsv()` (no RGB / RGBA)
//!    must not grow the source-width RGB scratch, `with_luma()` +
//!    `with_hsv()` stays RGB-free, and the luma output equals the native
//!    Y plane.

use super::*;
use crate::row::{
  nv_to_luma_row, nv12_to_hsv_row, nv12_to_rgb_row, nv21_to_hsv_row, nv21_to_rgb_row,
  nv24_to_hsv_row, nv24_to_rgb_row, nv42_to_hsv_row, nv42_to_rgb_row, rgb_to_hsv_row,
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

/// Reference HSV: the explicit `NV → RGB → HSV` via the same dispatcher
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

/// Interleaves half-width `u` / `v` chroma planes into a single NV-style
/// row. `swap = false` writes `U0 V0 U1 V1 …` (NV12 / NV16 / NV24);
/// `swap = true` writes `V0 U0 …` (NV21 / NV42).
fn interleave(u: &[u8], v: &[u8], swap: bool) -> Vec<u8> {
  let mut out = std::vec![0u8; u.len() * 2];
  for i in 0..u.len() {
    if swap {
      out[i * 2] = v[i];
      out[i * 2 + 1] = u[i];
    } else {
      out[i * 2] = u[i];
      out[i * 2 + 1] = v[i];
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_hsv_row_matches_rgb_then_hsv() {
  // NV12 (4:2:0, UV-ordered) — even widths incl. an odd-half tail. The
  // chroma row is `width` interleaved bytes (one UV pair per two pixels).
  for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
    let y: Vec<u8> = (0..w).map(|i| pat(i, 1)).collect();
    let u: Vec<u8> = (0..w / 2).map(|i| pat(i, 2)).collect();
    let v_in: Vec<u8> = (0..w / 2).map(|i| pat(i, 3)).collect();
    let uv = interleave(&u, &v_in, false);
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          nv12_to_rgb_row(&y, &uv, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          nv12_to_hsv_row(
            &y, &uv, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("nv12 w={w} {matrix:?} full={full_range} simd={use_simd}")
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
fn nv21_hsv_row_matches_rgb_then_hsv() {
  // NV21 (4:2:0, VU-ordered) — the swapped-chroma twin of NV12.
  for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
    let y: Vec<u8> = (0..w).map(|i| pat(i, 1)).collect();
    let u: Vec<u8> = (0..w / 2).map(|i| pat(i, 2)).collect();
    let v_in: Vec<u8> = (0..w / 2).map(|i| pat(i, 3)).collect();
    let vu = interleave(&u, &v_in, true);
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          nv21_to_rgb_row(&y, &vu, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          nv21_to_hsv_row(
            &y, &vu, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("nv21 w={w} {matrix:?} full={full_range} simd={use_simd}")
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
fn nv24_hsv_row_matches_rgb_then_hsv() {
  // NV24 (4:4:4, UV-ordered) — full-width chroma (`2 * width` interleaved
  // bytes); arbitrary widths incl. odd.
  for &w in &[1usize, 3, 5, 15, 16, 31, 64, 65] {
    let y: Vec<u8> = (0..w).map(|i| pat(i, 1)).collect();
    let u: Vec<u8> = (0..w).map(|i| pat(i, 2)).collect();
    let v_in: Vec<u8> = (0..w).map(|i| pat(i, 3)).collect();
    let uv = interleave(&u, &v_in, false);
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          nv24_to_rgb_row(&y, &uv, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          nv24_to_hsv_row(
            &y, &uv, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("nv24 w={w} {matrix:?} full={full_range} simd={use_simd}")
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
fn nv42_hsv_row_matches_rgb_then_hsv() {
  // NV42 (4:4:4, VU-ordered) — the swapped-chroma twin of NV24.
  for &w in &[1usize, 3, 5, 15, 16, 31, 64, 65] {
    let y: Vec<u8> = (0..w).map(|i| pat(i, 1)).collect();
    let u: Vec<u8> = (0..w).map(|i| pat(i, 2)).collect();
    let v_in: Vec<u8> = (0..w).map(|i| pat(i, 3)).collect();
    let vu = interleave(&u, &v_in, true);
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          nv42_to_rgb_row(&y, &vu, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          nv42_to_hsv_row(
            &y, &vu, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("nv42 w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
        }
      }
    }
  }
}

// ---- Native luma: the kernel equals the verbatim Y plane copy ----------

#[test]
fn nv_luma_row_equals_verbatim_y_copy() {
  // The native-Y contract: `nv_to_luma_row` is a straight copy of the
  // first `width` Y samples — bit-identical to the sink's former inline
  // `luma.copy_from_slice(&row.y()[..w])`.
  for &w in &[1usize, 2, 7, 16, 31, 64] {
    let y: Vec<u8> = (0..w + 5).map(|i| pat(i, 4)).collect();
    let mut got = std::vec![0u8; w];
    nv_to_luma_row(&y, &mut got, w);
    let want = &y[..w];
    assert_eq!(&got[..], want, "nv_to_luma_row w={w}");
  }
}

// ---- Structural: HSV-only attaches no source-width RGB scratch ----------

/// Plane bytes (a varied, non-gray pattern).
fn plane(width: usize, height: usize, salt: usize) -> Vec<u8> {
  (0..width * height).map(|i| pat(i, salt)).collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_hsv_only_grows_no_rgb_scratch() {
  let (w, h) = (16usize, 8usize);
  let yp = plane(w, h, 1);
  let up = plane(w / 2, h / 2, 2);
  let vp = plane(w / 2, h / 2, 3);
  // NV12 chroma is `width`-byte interleaved UV per row, half height.
  let mut uv = std::vec![0u8; w * (h / 2)];
  for row in 0..h / 2 {
    let u_row = &up[row * (w / 2)..(row + 1) * (w / 2)];
    let v_row = &vp[row * (w / 2)..(row + 1) * (w / 2)];
    uv[row * w..(row + 1) * w].copy_from_slice(&interleave(u_row, v_row, false));
  }
  let src = Nv12Frame::new(&yp, &uv, w as u32, h as u32, w as u32, w as u32);

  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Nv12>::new(w, h)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    nv12_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "Nv12 HSV-only must not grow the source-width RGB scratch"
  );

  // Cross-check row 0 against the explicit NV→RGB→HSV reference.
  let mut rgb0 = std::vec![0u8; w * 3];
  nv12_to_rgb_row(
    &yp[..w],
    &uv[..w],
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
fn nv12_luma_plus_hsv_only_is_correct_and_rgb_free() {
  // with_luma() + with_hsv(), no RGB: native-Y luma (via `nv_to_luma_row`)
  // AND direct HSV, with no source-width RGB scratch.
  let (w, h) = (16usize, 8usize);
  let yp = plane(w, h, 7);
  let up = plane(w / 2, h / 2, 9);
  let vp = plane(w / 2, h / 2, 11);
  let mut uv = std::vec![0u8; w * (h / 2)];
  for row in 0..h / 2 {
    let u_row = &up[row * (w / 2)..(row + 1) * (w / 2)];
    let v_row = &vp[row * (w / 2)..(row + 1) * (w / 2)];
    uv[row * w..(row + 1) * w].copy_from_slice(&interleave(u_row, v_row, false));
  }
  let src = Nv12Frame::new(&yp, &uv, w as u32, h as u32, w as u32, w as u32);

  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Nv12>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    nv12_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    sink.rgb_scratch.len()
  };

  assert_eq!(
    scratch_len, 0,
    "luma + HSV (no RGB) must not grow the RGB scratch"
  );
  // Luma is the Y plane verbatim (native-Y contract).
  assert_eq!(luma, yp, "luma must equal the native Y plane");
  // HSV row 0 matches the two-step reference.
  let mut rgb0 = std::vec![0u8; w * 3];
  nv12_to_rgb_row(
    &yp[..w],
    &uv[..w],
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
fn nv_hsv_only_grows_no_rgb_scratch_all_formats() {
  let (w, h) = (16usize, 8usize);

  // NV16 (4:2:2, UV) — half-width chroma, full height (`width` bytes/row).
  {
    let yp = plane(w, h, 1);
    let up = plane(w / 2, h, 2);
    let vp = plane(w / 2, h, 3);
    let mut uv = std::vec![0u8; w * h];
    for row in 0..h {
      let u_row = &up[row * (w / 2)..(row + 1) * (w / 2)];
      let v_row = &vp[row * (w / 2)..(row + 1) * (w / 2)];
      uv[row * w..(row + 1) * w].copy_from_slice(&interleave(u_row, v_row, false));
    }
    let src = Nv16Frame::new(&yp, &uv, w as u32, h as u32, w as u32, w as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Nv16>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      nv16_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Nv16 HSV-only RGB-free");
    // Row 0 matches the NV12-kernel reference (4:2:2 reuses it).
    let mut rgb0 = std::vec![0u8; w * 3];
    nv12_to_rgb_row(
      &yp[..w],
      &uv[..w],
      &mut rgb0,
      w,
      KernelMatrix::Bt601,
      true,
      true,
    );
    let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
    assert_eq!(&hh[..w], &rh[..], "nv16 row 0 H");
    assert_eq!(&ss[..w], &rs[..], "nv16 row 0 S");
    assert_eq!(&vv[..w], &rv[..], "nv16 row 0 V");
  }

  // NV21 (4:2:0, VU) — swapped-chroma twin of NV12.
  {
    let yp = plane(w, h, 1);
    let up = plane(w / 2, h / 2, 2);
    let vp = plane(w / 2, h / 2, 3);
    let mut vu = std::vec![0u8; w * (h / 2)];
    for row in 0..h / 2 {
      let u_row = &up[row * (w / 2)..(row + 1) * (w / 2)];
      let v_row = &vp[row * (w / 2)..(row + 1) * (w / 2)];
      vu[row * w..(row + 1) * w].copy_from_slice(&interleave(u_row, v_row, true));
    }
    let src = Nv21Frame::new(&yp, &vu, w as u32, h as u32, w as u32, w as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Nv21>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      nv21_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Nv21 HSV-only RGB-free");
  }

  // NV24 (4:4:4, UV) — full-width chroma (`2 * width` bytes/row).
  {
    let yp = plane(w, h, 1);
    let up = plane(w, h, 2);
    let vp = plane(w, h, 3);
    let mut uv = std::vec![0u8; 2 * w * h];
    for row in 0..h {
      let u_row = &up[row * w..(row + 1) * w];
      let v_row = &vp[row * w..(row + 1) * w];
      uv[row * 2 * w..(row + 1) * 2 * w].copy_from_slice(&interleave(u_row, v_row, false));
    }
    let src = Nv24Frame::new(&yp, &uv, w as u32, h as u32, w as u32, (2 * w) as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Nv24>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      nv24_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Nv24 HSV-only RGB-free");
    let mut rgb0 = std::vec![0u8; w * 3];
    nv24_to_rgb_row(
      &yp[..w],
      &uv[..2 * w],
      &mut rgb0,
      w,
      KernelMatrix::Bt601,
      true,
      true,
    );
    let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
    assert_eq!(&hh[..w], &rh[..], "nv24 row 0 H");
    assert_eq!(&ss[..w], &rs[..], "nv24 row 0 S");
    assert_eq!(&vv[..w], &rv[..], "nv24 row 0 V");
  }

  // NV42 (4:4:4, VU) — swapped-chroma twin of NV24.
  {
    let yp = plane(w, h, 1);
    let up = plane(w, h, 2);
    let vp = plane(w, h, 3);
    let mut vu = std::vec![0u8; 2 * w * h];
    for row in 0..h {
      let u_row = &up[row * w..(row + 1) * w];
      let v_row = &vp[row * w..(row + 1) * w];
      vu[row * 2 * w..(row + 1) * 2 * w].copy_from_slice(&interleave(u_row, v_row, true));
    }
    let src = Nv42Frame::new(&yp, &vu, w as u32, h as u32, w as u32, (2 * w) as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Nv42>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      nv42_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Nv42 HSV-only RGB-free");
  }
}
