//! #263 PR 7 — direct Y2xx packed YUV (Y210 / Y212 / Y216, packed 4:2:2,
//! 10 / 12 / 16-bit) YUV→HSV kernels + RGB-free routing.
//!
//! Two concerns, mirroring the packed-YUV (PR 5) and high-bit
//! semi-planar (PR 4) suites:
//!
//! 1. **Row-kernel parity** — each `*_to_hsv_row_endian` dispatcher must
//!    be byte-identical to `rgb_to_hsv_row(*_to_rgb_row_endian(...))`
//!    within a tier (they share the same **8-bit** RGB intermediate the
//!    existing Y2xx→RGB→HSV path uses, and the same per-tier HSV), across
//!    `BITS ∈ {10, 12, 16}`, both endiannesses (`BE ∈ {false, true}`), the
//!    three colour matrices, full / limited range, and a spread of even
//!    (4:2:2) widths — for both the scalar path (`use_simd = false`) and
//!    the host SIMD path (`use_simd = true`).
//! 2. **Structural** — a Y2xx sink with ONLY `with_hsv()` (no RGB / RGBA)
//!    must not grow the source-width RGB scratch (`rgb_scratch.len() ==
//!    0`), and `with_luma()` + `with_hsv()` stays RGB-free; the HSV output
//!    equals the explicit two-step `*_to_rgb_row` → `rgb_to_hsv_row`
//!    reference.

use super::*;
use crate::row::{
  rgb_to_hsv_row, y210_to_hsv_row_endian, y210_to_rgb_row_endian, y212_to_hsv_row_endian,
  y212_to_rgb_row_endian, y216_to_hsv_row_endian, y216_to_rgb_row_endian,
};

const MATRICES: [KernelMatrix; 3] = [
  KernelMatrix::Bt601,
  KernelMatrix::Bt709,
  KernelMatrix::Bt2020Ncl,
];

/// A non-trivial, non-gray pseudo-random logical sample masked to `bits`
/// (16-bit uses the full range) at a given position, so the HSV hue /
/// saturation branches are all exercised rather than the degenerate gray
/// fast-path.
fn logical_samp(i: usize, salt: usize, bits: u32) -> u16 {
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

/// Build a Y2xx packed row of `width` pixels: `width / 2` quadruples
/// `(Y₀, U, Y₁, V)`, each sample MSB-aligned (`logical << (16 - bits)`;
/// `<< 0` at 16-bit), then split into host-native LE / BE wire `u16`
/// buffers. Returns `(packed_le, packed_be)` where each element,
/// serialized via `to_ne_bytes`, reproduces the LE / BE wire bytes — so
/// the `BE = false` / `BE = true` kernel paths are exercised regardless of
/// host endianness.
fn packed_le_be(width: usize, bits: u32) -> (Vec<u16>, Vec<u16>) {
  let shift = 16 - bits;
  let pairs = width / 2;
  let mut intended = Vec::with_capacity(width * 2);
  for p in 0..pairs {
    intended.push(logical_samp(p, 1, bits) << shift); // Y0
    intended.push(logical_samp(p, 2, bits) << shift); // U
    intended.push(logical_samp(p, 3, bits) << shift); // Y1
    intended.push(logical_samp(p, 4, bits) << shift); // V
  }
  let to_le: Vec<u16> = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect();
  let to_be: Vec<u16> = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect();
  (to_le, to_be)
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

// ---- Row-kernel parity (packed 4:2:2, both tiers × BITS × BE) ----------
//
// Both the direct `*_to_hsv_row_endian` kernel and the `*_to_rgb_row_endian`
// → `rgb_to_hsv_row` reference read the SAME packed buffer through the SAME
// format kernel at the same `(matrix, full_range, use_simd, big_endian)`,
// so a pseudo-random fill exercises the full pipeline. Widths are even (the
// 4:2:2 constraint) and span above / below every SIMD block size, plus an
// odd-half tail (`w / 2` odd).

macro_rules! parity_y2xx {
  ($name:ident, $bits:expr, $rgb_fn:path, $hsv_fn:path) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $name() {
      let bits: u32 = $bits;
      for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
        let (le, be) = packed_le_be(w, bits);
        for (big_endian, packed) in [(false, &le), (true, &be)] {
          for &matrix in &MATRICES {
            for &full_range in &[true, false] {
              for &use_simd in &[false, true] {
                let mut rgb = std::vec![0u8; w * 3];
                $rgb_fn(packed, &mut rgb, w, matrix, full_range, use_simd, big_endian);
                let want = ref_hsv_from_rgb(&rgb, w, use_simd);

                let mut h = std::vec![0u8; w];
                let mut s = std::vec![0u8; w];
                let mut v = std::vec![0u8; w];
                $hsv_fn(
                  packed, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd, big_endian,
                );
                assert_planes_eq!(
                  (h, s, v),
                  want,
                  std::format!(
                    "y2xx bits={bits} be={big_endian} w={w} {matrix:?} \
                     full={full_range} simd={use_simd}"
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

parity_y2xx!(
  y210_hsv_row_matches_rgb_then_hsv,
  10,
  y210_to_rgb_row_endian,
  y210_to_hsv_row_endian
);
parity_y2xx!(
  y212_hsv_row_matches_rgb_then_hsv,
  12,
  y212_to_rgb_row_endian,
  y212_to_hsv_row_endian
);
parity_y2xx!(
  y216_hsv_row_matches_rgb_then_hsv,
  16,
  y216_to_rgb_row_endian,
  y216_to_hsv_row_endian
);

// ---- Structural: HSV-only attaches no source-width RGB scratch ----------

/// Build a Y2xx packed frame (`width * 2` u16 per row, MSB-aligned), as a
/// host-native LE-wire `u16` buffer.
fn packed_le_frame(width: usize, height: usize, bits: u32) -> Vec<u16> {
  let shift = 16 - bits;
  let pairs = width / 2;
  let mut buf = Vec::with_capacity(width * 2 * height);
  for row in 0..height {
    for p in 0..pairs {
      let salt = row * pairs + p;
      buf.push(u16::from_ne_bytes(
        (logical_samp(salt, 1, bits) << shift).to_le_bytes(),
      )); // Y0
      buf.push(u16::from_ne_bytes(
        (logical_samp(salt, 2, bits) << shift).to_le_bytes(),
      )); // U
      buf.push(u16::from_ne_bytes(
        (logical_samp(salt, 3, bits) << shift).to_le_bytes(),
      )); // Y1
      buf.push(u16::from_ne_bytes(
        (logical_samp(salt, 4, bits) << shift).to_le_bytes(),
      )); // V
    }
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y210_hsv_only_grows_no_rgb_scratch() {
  const BITS: u32 = 10;
  let (w, h) = (16usize, 8usize);
  let buf = packed_le_frame(w, h, BITS);
  // Y210 row stride = `width * 2` u16.
  let src = Y210Frame::new(&buf, w as u32, h as u32, (w * 2) as u32);

  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Y210>::new(w, h)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    y210_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "Y210 HSV-only must not grow the source-width RGB scratch"
  );

  // Cross-check row 0 against the explicit packed→RGB→HSV reference (LE).
  let mut rgb0 = std::vec![0u8; w * 3];
  y210_to_rgb_row_endian(
    &buf[..w * 2],
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
fn y2xx_hsv_only_grows_no_rgb_scratch_all_formats() {
  let (w, h) = (16usize, 8usize);

  // Y212 — 12-bit MSB-aligned.
  {
    const BITS: u32 = 12;
    let buf = packed_le_frame(w, h, BITS);
    let src = Y212Frame::new(&buf, w as u32, h as u32, (w * 2) as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Y212>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      y212_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Y212 HSV-only RGB-free");
    let mut rgb0 = std::vec![0u8; w * 3];
    y212_to_rgb_row_endian(
      &buf[..w * 2],
      &mut rgb0,
      w,
      KernelMatrix::Bt709,
      true,
      true,
      false,
    );
    let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
    assert_eq!(&hh[..w], &rh[..], "y212 row 0 H");
    assert_eq!(&ss[..w], &rs[..], "y212 row 0 S");
    assert_eq!(&vv[..w], &rv[..], "y212 row 0 V");
  }

  // Y216 — full 16-bit.
  {
    const BITS: u32 = 16;
    let buf = packed_le_frame(w, h, BITS);
    let src = Y216Frame::new(&buf, w as u32, h as u32, (w * 2) as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Y216>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      y216_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt2020Ncl)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "Y216 HSV-only RGB-free");
    let mut rgb0 = std::vec![0u8; w * 3];
    y216_to_rgb_row_endian(
      &buf[..w * 2],
      &mut rgb0,
      w,
      KernelMatrix::Bt2020Ncl,
      true,
      true,
      false,
    );
    let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
    assert_eq!(&hh[..w], &rh[..], "y216 row 0 H");
    assert_eq!(&ss[..w], &rs[..], "y216 row 0 S");
    assert_eq!(&vv[..w], &rv[..], "y216 row 0 V");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_luma_plus_hsv_only_is_correct_and_rgb_free() {
  // with_luma() + with_hsv(), no RGB: native-Y luma AND direct HSV, with
  // no source-width RGB scratch.
  const BITS: u32 = 12;
  let (w, h) = (16usize, 8usize);
  let buf = packed_le_frame(w, h, BITS);
  let src = Y212Frame::new(&buf, w as u32, h as u32, (w * 2) as u32);

  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Y212>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    y212_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "luma + HSV (no RGB) must not grow the RGB scratch"
  );
  // HSV row 0 matches the two-step reference.
  let mut rgb0 = std::vec![0u8; w * 3];
  y212_to_rgb_row_endian(
    &buf[..w * 2],
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
