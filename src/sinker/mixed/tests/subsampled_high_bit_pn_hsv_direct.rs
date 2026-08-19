//! #263 PR 4 — direct semi-planar **high-bit** P-format YUV→HSV kernels
//! + native-Y luma kernel + RGB-free routing (P010 / P012 / P016, 4:2:0).
//!
//! Three concerns, mirroring the PR-3 high-bit planar suite:
//!
//! 1. **Row-kernel parity** — each `p0xx_to_hsv_row_endian` dispatcher
//!    must be byte-identical to `rgb_to_hsv_row(p0xx_to_rgb_row_endian
//!    (...))` within a tier (they share the same **8-bit** RGB
//!    intermediate the existing high-bit→RGB→HSV path uses, and the same
//!    per-tier HSV), across `BITS ∈ {10, 12, 16}`, both endiannesses
//!    (`BE ∈ {false, true}`), the three colour matrices, full/limited
//!    range, and a spread of widths — for both the scalar path
//!    (`use_simd = false`) and the host SIMD path (`use_simd = true`).
//! 2. **Native luma** — `p0xx_to_luma_row_endian` reproduces the Y
//!    plane's high byte (`>> 8` of the high-bit-packed wire `u16` after
//!    host-native normalization) — the exact bytes the sink's former
//!    inline native-Y loop produced.
//! 3. **Structural** — a P-format sink with ONLY `with_hsv()` (no RGB /
//!    RGBA) must not grow the source-width RGB scratch, and
//!    `with_luma()` + `with_hsv()` stays RGB-free.

use super::*;
use crate::row::{
  p010_to_hsv_row_endian, p010_to_luma_row_endian, p010_to_rgb_row_endian, p012_to_hsv_row_endian,
  p012_to_luma_row_endian, p012_to_rgb_row_endian, p016_to_hsv_row_endian, p016_to_luma_row_endian,
  p016_to_rgb_row_endian, rgb_to_hsv_row,
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

/// Build a high-bit-packed (`logical << (16 - bits)`) Y plane and a
/// half-width interleaved `U, V, U, V…` chroma plane (4:2:0), then split
/// each into host-native LE / BE wire `u16` buffers. Returns
/// `(y_le, y_be, uv_le, uv_be)` where each element, serialized via
/// `to_ne_bytes`, reproduces the LE / BE wire bytes — so the
/// `BE = false` / `BE = true` kernel paths are exercised regardless of
/// host endianness.
#[allow(clippy::type_complexity)]
fn packed_le_be(width: usize, bits: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let shift = 16 - bits;
  let y: Vec<u16> = (0..width)
    .map(|i| logical_samp(i, 1, bits) << shift)
    .collect();
  // Half-width chroma: `width / 2` pairs = `width` interleaved u16.
  let uv: Vec<u16> = (0..width / 2)
    .flat_map(|i| {
      [
        logical_samp(i, 2, bits) << shift,
        logical_samp(i, 3, bits) << shift,
      ]
    })
    .collect();
  let to_le = |s: &[u16]| -> Vec<u16> {
    s.iter()
      .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
      .collect()
  };
  let to_be = |s: &[u16]| -> Vec<u16> {
    s.iter()
      .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
      .collect()
  };
  (to_le(&y), to_be(&y), to_le(&uv), to_be(&uv))
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

/// One BITS' worth of the semi-planar 4:2:0 parity sweep: the
/// `*_to_hsv_row_endian` kernel vs `rgb_to_hsv_row(*_to_rgb_row_endian)`
/// for both endiannesses × matrices × range × widths × tiers.
macro_rules! parity_p0xx {
  ($name:ident, $bits:expr, $rgb_fn:path, $hsv_fn:path) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $name() {
      let bits: u32 = $bits;
      for &w in &[2usize, 4, 6, 14, 16, 30, 64, 66] {
        let (y_le, y_be, uv_le, uv_be) = packed_le_be(w, bits);
        for (be, (yy, uu)) in [(false, (&y_le, &uv_le)), (true, (&y_be, &uv_be))] {
          for &matrix in &MATRICES {
            for &full_range in &[true, false] {
              for &use_simd in &[false, true] {
                let mut rgb = std::vec![0u8; w * 3];
                $rgb_fn(yy, uu, &mut rgb, w, matrix, full_range, use_simd, be);
                let want = ref_hsv_from_rgb(&rgb, w, use_simd);

                let mut h = std::vec![0u8; w];
                let mut s = std::vec![0u8; w];
                let mut v = std::vec![0u8; w];
                $hsv_fn(yy, uu, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd, be);
                assert_planes_eq!(
                  (h, s, v),
                  want,
                  std::format!(
                    "p0xx bits={bits} be={be} w={w} {matrix:?} full={full_range} simd={use_simd}"
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

parity_p0xx!(
  p010_hsv_row_matches_rgb_then_hsv,
  10,
  p010_to_rgb_row_endian,
  p010_to_hsv_row_endian
);
parity_p0xx!(
  p012_hsv_row_matches_rgb_then_hsv,
  12,
  p012_to_rgb_row_endian,
  p012_to_hsv_row_endian
);
parity_p0xx!(
  p016_hsv_row_matches_rgb_then_hsv,
  16,
  p016_to_rgb_row_endian,
  p016_to_hsv_row_endian
);

// ---- Native luma: kernel == the sink's former inline `>> 8` loop ------

/// Reference for the P-format sink's former inline native-Y luma: the Y
/// plane's high byte (`>> 8`) after BE/LE host-native normalization.
fn ref_inline_luma(y: &[u16], width: usize, be: bool) -> Vec<u8> {
  y[..width]
    .iter()
    .map(|&s| {
      let logical = if be { u16::from_be(s) } else { u16::from_le(s) };
      (logical >> 8) as u8
    })
    .collect()
}

macro_rules! luma_p0xx {
  ($name:ident, $bits:expr, $luma_fn:path) => {
    #[test]
    fn $name() {
      let bits: u32 = $bits;
      for &w in &[2usize, 4, 8, 16, 30, 64] {
        let (y_le, y_be, _uv_le, _uv_be) = packed_le_be(w, bits);
        for (be, yy) in [(false, &y_le), (true, &y_be)] {
          let want = ref_inline_luma(yy, w, be);
          let mut got = std::vec![0u8; w];
          $luma_fn(yy, &mut got, w, be);
          assert_eq!(got, want, "p0xx luma bits={bits} be={be} w={w}");
        }
      }
    }
  };
}

luma_p0xx!(
  p010_luma_matches_inline_native_y,
  10,
  p010_to_luma_row_endian
);
luma_p0xx!(
  p012_luma_matches_inline_native_y,
  12,
  p012_to_luma_row_endian
);
luma_p0xx!(
  p016_luma_matches_inline_native_y,
  16,
  p016_to_luma_row_endian
);

// ---- Structural: HSV-only attaches no source-width RGB scratch --------

/// Build a high-bit-packed LE Y plane + half-width/half-height
/// interleaved UV plane (4:2:0) of varied (non-gray) samples, as a
/// host-native LE-wire `u16` buffer pair (the `*LeFrame` plane
/// contract).
fn packed_le_frame_planes(width: usize, height: usize, bits: u32) -> (Vec<u16>, Vec<u16>) {
  let shift = 16 - bits;
  let y: Vec<u16> = (0..width * height)
    .map(|i| u16::from_ne_bytes((logical_samp(i, 1, bits) << shift).to_le_bytes()))
    .collect();
  // 4:2:0 chroma: half-width, half-height, interleaved U,V (= `width`
  // u16 per chroma row).
  let uv: Vec<u16> = (0..(width / 2) * (height / 2))
    .flat_map(|i| {
      [
        u16::from_ne_bytes((logical_samp(i, 2, bits) << shift).to_le_bytes()),
        u16::from_ne_bytes((logical_samp(i, 3, bits) << shift).to_le_bytes()),
      ]
    })
    .collect();
  (y, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_hsv_only_grows_no_rgb_scratch() {
  const BITS: u32 = 10;
  let (w, h) = (16usize, 8usize);
  let (yp, uvp) = packed_le_frame_planes(w, h, BITS);
  // 4:2:0 interleaved UV row stride = `width` u16.
  let src = P010LeFrame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<P010>::new(w, h)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    p010_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "P010 HSV-only must not grow the source-width RGB scratch"
  );

  // Cross-check row 0 against the explicit YUV→RGB→HSV reference (LE).
  let mut rgb0 = std::vec![0u8; w * 3];
  p010_to_rgb_row_endian(
    &yp[..w],
    &uvp[..w],
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
fn p0xx_hsv_only_grows_no_rgb_scratch_all_formats() {
  let (w, h) = (16usize, 8usize);

  // P012.
  {
    const BITS: u32 = 12;
    let (yp, uvp) = packed_le_frame_planes(w, h, BITS);
    let src = P012LeFrame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<P012>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      p012_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "P012 HSV-only RGB-free");
  }

  // P016.
  {
    const BITS: u32 = 16;
    let (yp, uvp) = packed_le_frame_planes(w, h, BITS);
    let src = P016LeFrame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<P016>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      p016_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt2020Ncl)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(scratch_len, 0, "P016 HSV-only RGB-free");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_luma_plus_hsv_only_is_correct_and_rgb_free() {
  // with_luma() + with_hsv(), no RGB: native-Y luma (via the luma
  // kernel) AND direct HSV with no source-width RGB scratch.
  const BITS: u32 = 12;
  let (w, h) = (16usize, 8usize);
  let (yp, uvp) = packed_le_frame_planes(w, h, BITS);
  let src = P012LeFrame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<P012>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    p012_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "luma + HSV (no RGB) must not grow the RGB scratch"
  );

  // Luma matches the kernel reference (the sink's former inline loop).
  let want_luma = ref_inline_luma(&yp[..w], w, false);
  assert_eq!(&luma[..w], &want_luma[..], "row 0 luma");

  // HSV row 0 matches the two-step reference (LE).
  let mut rgb0 = std::vec![0u8; w * 3];
  p012_to_rgb_row_endian(
    &yp[..w],
    &uvp[..w],
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
