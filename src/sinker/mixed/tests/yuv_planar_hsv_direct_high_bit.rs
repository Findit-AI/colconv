//! #263 PR 3 — direct planar **high-bit** YUV→HSV kernels + RGB-free
//! routing (yuv420p / yuv422p / yuv440p / yuv444p at 9/10/12/14/16-bit).
//!
//! Two concerns, mirroring the 8-bit PR-1 suite:
//!
//! 1. **Row-kernel parity** — each `yuv{420,444}pN_to_hsv_row_endian`
//!    dispatcher must be byte-identical to
//!    `rgb_to_hsv_row(yuv{420,444}pN_to_rgb_row_endian(...))` within a
//!    tier (they share the same **8-bit** RGB intermediate the existing
//!    high-bit→RGB→HSV path uses, and the same per-tier HSV), across
//!    every `BITS ∈ {9, 10, 12, 14, 16}`, both endiannesses
//!    (`BE ∈ {false, true}`), the three colour matrices, full/limited
//!    range, and a spread of widths — for both the scalar path
//!    (`use_simd = false`) and the host SIMD path (`use_simd = true`).
//! 2. **Structural** — a high-bit planar-YUV sink with ONLY `with_hsv()`
//!    (no RGB / RGBA) must not grow the source-width RGB scratch.

use super::*;
use crate::row::{
  rgb_to_hsv_row, yuv420p9_to_hsv_row_endian, yuv420p9_to_rgb_row_endian,
  yuv420p10_to_hsv_row_endian, yuv420p10_to_rgb_row_endian, yuv420p12_to_hsv_row_endian,
  yuv420p12_to_rgb_row_endian, yuv420p14_to_hsv_row_endian, yuv420p14_to_rgb_row_endian,
  yuv420p16_to_hsv_row_endian, yuv420p16_to_rgb_row_endian, yuv444p9_to_hsv_row_endian,
  yuv444p9_to_rgb_row_endian, yuv444p10_to_hsv_row_endian, yuv444p10_to_rgb_row_endian,
  yuv444p12_to_hsv_row_endian, yuv444p12_to_rgb_row_endian, yuv444p14_to_hsv_row_endian,
  yuv444p14_to_rgb_row_endian, yuv444p16_to_hsv_row_endian, yuv444p16_to_rgb_row_endian,
};

const MATRICES: [KernelMatrix; 3] = [
  KernelMatrix::Bt601,
  KernelMatrix::Bt709,
  KernelMatrix::Bt2020Ncl,
];

/// A non-trivial, non-gray pseudo-random sample masked to `bits` (16-bit
/// uses the full range) at a given position, so the HSV hue / saturation
/// branches are all exercised rather than the degenerate gray fast-path.
fn samp(i: usize, salt: usize, bits: u32) -> u16 {
  let v = (i
    .wrapping_mul(2_654_435_761)
    .wrapping_add(salt.wrapping_mul(40_503))
    .wrapping_add(0x9E37)) as u32;
  if bits >= 16 {
    (v & 0xFFFF) as u16
  } else {
    (v & ((1u32 << bits) - 1)) as u16
  }
}

/// Build host-native LE/BE `u16` buffers from a slice of intended
/// low-bit-packed samples. Returns `(le, be)` where each element, when
/// serialized via `to_ne_bytes`, reproduces the LE/BE wire bytes — so the
/// `BE = false` / `BE = true` kernel paths are exercised regardless of
/// host endianness.
fn split_le_be(intended: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let le: Vec<u16> = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect();
  let be: Vec<u16> = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect();
  (le, be)
}

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

/// One BITS' worth of the 4:2:0 (also 4:2:2) parity sweep: the
/// `*_to_hsv_row_endian` kernel vs `rgb_to_hsv_row(*_to_rgb_row_endian)`
/// for both endiannesses × matrices × range × widths × tiers.
macro_rules! parity_420 {
  ($name:ident, $bits:expr, $rgb_fn:path, $hsv_fn:path) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $name() {
      let bits: u32 = $bits;
      for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
        let y_i: Vec<u16> = (0..w).map(|i| samp(i, 1, bits)).collect();
        let u_i: Vec<u16> = (0..w / 2).map(|i| samp(i, 2, bits)).collect();
        let v_i: Vec<u16> = (0..w / 2).map(|i| samp(i, 3, bits)).collect();
        let (y_le, y_be) = split_le_be(&y_i);
        let (u_le, u_be) = split_le_be(&u_i);
        let (v_le, v_be) = split_le_be(&v_i);
        for (be, (yy, uu, vv)) in
          [(false, (&y_le, &u_le, &v_le)), (true, (&y_be, &u_be, &v_be))]
        {
          for &matrix in &MATRICES {
            for &full_range in &[true, false] {
              for &use_simd in &[false, true] {
                let mut rgb = std::vec![0u8; w * 3];
                $rgb_fn(yy, uu, vv, &mut rgb, w, matrix, full_range, use_simd, be);
                let want = ref_hsv_from_rgb(&rgb, w, use_simd);

                let mut h = std::vec![0u8; w];
                let mut s = std::vec![0u8; w];
                let mut v = std::vec![0u8; w];
                $hsv_fn(
                  yy, uu, vv, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd, be,
                );
                assert_planes_eq!(
                  (h, s, v),
                  want,
                  std::format!(
                    "420 bits={bits} be={be} w={w} {matrix:?} full={full_range} simd={use_simd}"
                  )
                );
              }
            }
          }
        }
      }
    }
  };
}

/// One BITS' worth of the 4:4:4 (also 4:4:0) parity sweep.
macro_rules! parity_444 {
  ($name:ident, $bits:expr, $rgb_fn:path, $hsv_fn:path) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $name() {
      let bits: u32 = $bits;
      for &w in &[1usize, 3, 5, 15, 16, 31, 64, 65] {
        let y_i: Vec<u16> = (0..w).map(|i| samp(i, 1, bits)).collect();
        let u_i: Vec<u16> = (0..w).map(|i| samp(i, 2, bits)).collect();
        let v_i: Vec<u16> = (0..w).map(|i| samp(i, 3, bits)).collect();
        let (y_le, y_be) = split_le_be(&y_i);
        let (u_le, u_be) = split_le_be(&u_i);
        let (v_le, v_be) = split_le_be(&v_i);
        for (be, (yy, uu, vv)) in
          [(false, (&y_le, &u_le, &v_le)), (true, (&y_be, &u_be, &v_be))]
        {
          for &matrix in &MATRICES {
            for &full_range in &[true, false] {
              for &use_simd in &[false, true] {
                let mut rgb = std::vec![0u8; w * 3];
                $rgb_fn(yy, uu, vv, &mut rgb, w, matrix, full_range, use_simd, be);
                let want = ref_hsv_from_rgb(&rgb, w, use_simd);

                let mut h = std::vec![0u8; w];
                let mut s = std::vec![0u8; w];
                let mut v = std::vec![0u8; w];
                $hsv_fn(
                  yy, uu, vv, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd, be,
                );
                assert_planes_eq!(
                  (h, s, v),
                  want,
                  std::format!(
                    "444 bits={bits} be={be} w={w} {matrix:?} full={full_range} simd={use_simd}"
                  )
                );
              }
            }
          }
        }
      }
    }
  };
}

parity_420!(
  yuv420p9_hsv_row_matches_rgb_then_hsv,
  9,
  yuv420p9_to_rgb_row_endian,
  yuv420p9_to_hsv_row_endian
);
parity_420!(
  yuv420p10_hsv_row_matches_rgb_then_hsv,
  10,
  yuv420p10_to_rgb_row_endian,
  yuv420p10_to_hsv_row_endian
);
parity_420!(
  yuv420p12_hsv_row_matches_rgb_then_hsv,
  12,
  yuv420p12_to_rgb_row_endian,
  yuv420p12_to_hsv_row_endian
);
parity_420!(
  yuv420p14_hsv_row_matches_rgb_then_hsv,
  14,
  yuv420p14_to_rgb_row_endian,
  yuv420p14_to_hsv_row_endian
);
parity_420!(
  yuv420p16_hsv_row_matches_rgb_then_hsv,
  16,
  yuv420p16_to_rgb_row_endian,
  yuv420p16_to_hsv_row_endian
);

parity_444!(
  yuv444p9_hsv_row_matches_rgb_then_hsv,
  9,
  yuv444p9_to_rgb_row_endian,
  yuv444p9_to_hsv_row_endian
);
parity_444!(
  yuv444p10_hsv_row_matches_rgb_then_hsv,
  10,
  yuv444p10_to_rgb_row_endian,
  yuv444p10_to_hsv_row_endian
);
parity_444!(
  yuv444p12_hsv_row_matches_rgb_then_hsv,
  12,
  yuv444p12_to_rgb_row_endian,
  yuv444p12_to_hsv_row_endian
);
parity_444!(
  yuv444p14_hsv_row_matches_rgb_then_hsv,
  14,
  yuv444p14_to_rgb_row_endian,
  yuv444p14_to_hsv_row_endian
);
parity_444!(
  yuv444p16_hsv_row_matches_rgb_then_hsv,
  16,
  yuv444p16_to_rgb_row_endian,
  yuv444p16_to_hsv_row_endian
);

// ---- Structural: HSV-only attaches no source-width RGB scratch ----------

/// Plane of varied (non-gray) low-bit-packed `u16` samples.
fn plane_u16(width: usize, height: usize, salt: usize, bits: u32) -> Vec<u16> {
  (0..width * height).map(|i| samp(i, salt, bits)).collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_hsv_only_grows_no_rgb_scratch() {
  const BITS: u32 = 10;
  let (w, h) = (16usize, 8usize);
  let yp = plane_u16(w, h, 1, BITS);
  let up = plane_u16(w / 2, h / 2, 2, BITS);
  let vp = plane_u16(w / 2, h / 2, 3, BITS);
  let src = Yuv420p10Frame::new(
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
    let mut sink = MixedSinker::<Yuv420p10>::new(w, h)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuv420p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "Yuv420p10 HSV-only must not grow the source-width RGB scratch"
  );

  // Cross-check row 0 against the explicit YUV→RGB→HSV reference (LE).
  let mut rgb0 = std::vec![0u8; w * 3];
  yuv420p10_to_rgb_row_endian(
    &yp[..w],
    &up[..w / 2],
    &vp[..w / 2],
    &mut rgb0,
    w,
    KernelMatrix::Bt601,
    true,
    true,
    false,
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
fn planar_high_bit_hsv_only_grows_no_rgb_scratch_all_formats() {
  let (w, h) = (16usize, 8usize);

  // 4:2:0 16-bit — half-width, half-height chroma.
  {
    const BITS: u32 = 16;
    let yp = plane_u16(w, h, 1, BITS);
    let up = plane_u16(w / 2, h / 2, 2, BITS);
    let vp = plane_u16(w / 2, h / 2, 3, BITS);
    let src = Yuv420p16Frame::new(
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
      let mut sink = MixedSinker::<Yuv420p16>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv420p16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv420p16 HSV-only RGB-free");
  }

  // 4:2:2 12-bit — half-width chroma, full height.
  {
    const BITS: u32 = 12;
    let yp = plane_u16(w, h, 1, BITS);
    let up = plane_u16(w / 2, h, 2, BITS);
    let vp = plane_u16(w / 2, h, 3, BITS);
    let src = Yuv422p12Frame::new(
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
      let mut sink = MixedSinker::<Yuv422p12>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv422p12_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv422p12 HSV-only RGB-free");
  }

  // 4:4:4 14-bit — full-width chroma.
  {
    const BITS: u32 = 14;
    let yp = plane_u16(w, h, 1, BITS);
    let up = plane_u16(w, h, 2, BITS);
    let vp = plane_u16(w, h, 3, BITS);
    let src = Yuv444p14Frame::new(
      &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
    );
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Yuv444p14>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv444p14_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv444p14 HSV-only RGB-free");
  }

  // 4:4:0 10-bit — full-width chroma, half height.
  {
    const BITS: u32 = 10;
    let yp = plane_u16(w, h, 1, BITS);
    let up = plane_u16(w, h / 2, 2, BITS);
    let vp = plane_u16(w, h / 2, 3, BITS);
    let src = Yuv440p10Frame::new(
      &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
    );
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Yuv440p10>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      yuv440p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Yuv440p10 HSV-only RGB-free");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p12_luma_plus_hsv_only_is_correct_and_rgb_free() {
  // with_luma() + with_hsv(), no RGB: native Y luma AND direct HSV with
  // no source-width RGB scratch.
  const BITS: u32 = 12;
  let (w, h) = (16usize, 8usize);
  let yp = plane_u16(w, h, 7, BITS);
  let up = plane_u16(w, h, 9, BITS);
  let vp = plane_u16(w, h, 11, BITS);
  let src = Yuv444p12Frame::new(
    &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
  );

  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Yuv444p12>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuv444p12_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    sink.rgb_scratch.len()
  };

  assert_eq!(
    scratch_len, 0,
    "luma + HSV (no RGB) must not grow the RGB scratch"
  );
  // HSV row 0 matches the two-step reference (LE).
  let mut rgb0 = std::vec![0u8; w * 3];
  yuv444p12_to_rgb_row_endian(
    &yp[..w],
    &up[..w],
    &vp[..w],
    &mut rgb0,
    w,
    KernelMatrix::Bt709,
    true,
    true,
    false,
  );
  let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
  assert_eq!(&hh[..w], &rh[..], "row 0 H");
  assert_eq!(&ss[..w], &rs[..], "row 0 S");
  assert_eq!(&vv[..w], &rv[..], "row 0 V");
}
