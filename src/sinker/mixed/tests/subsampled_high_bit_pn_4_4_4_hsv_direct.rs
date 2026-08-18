//! #263 — direct semi-planar **high-bit 4:4:4** P-format YUV→HSV kernels
//! + RGB-free routing (P410 / P412 / P416).
//!
//! Two concerns, mirroring the 4:2:0 (p0xx) HSV-direct suite:
//!
//! 1. **Row-kernel parity** — each `p4NN_to_hsv_row_endian` dispatcher
//!    must be byte-identical to `rgb_to_hsv_row(p4NN_to_rgb_row_endian
//!    (...))` within a tier (they share the same **8-bit** RGB
//!    intermediate the existing high-bit→RGB→HSV path uses), across
//!    `BITS ∈ {10, 12, 16}`, both endiannesses (`BE ∈ {false, true}`), the
//!    three colour matrices, full/limited range, and a spread of widths
//!    (including odd widths — 4:4:4 has no even-width constraint) — for
//!    both the scalar path (`use_simd = false`) and the host SIMD path
//!    (`use_simd = true`). This is the new-kernel SIMD-vs-scalar parity.
//! 2. **Structural** — a P4NN sink with ONLY `with_hsv()` (no RGB / RGBA)
//!    must not grow the source-width RGB scratch, and its HSV output
//!    matches the per-row two-step YUV→RGB→HSV reference.

use super::*;
use crate::row::{
  p410_to_hsv_row_endian, p410_to_rgb_row_endian, p412_to_hsv_row_endian, p412_to_rgb_row_endian,
  p416_to_hsv_row_endian, p416_to_rgb_row_endian, rgb_to_hsv_row,
};

const MATRICES: [KernelMatrix; 3] = [
  KernelMatrix::Bt601,
  KernelMatrix::Bt709,
  KernelMatrix::Bt2020Ncl,
];

/// A non-trivial, non-gray pseudo-random logical sample masked to `bits`
/// (16-bit uses the full range), so the HSV hue / saturation branches are
/// all exercised rather than the degenerate gray fast-path.
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
/// **full-width** interleaved `U, V, U, V…` chroma plane (4:4:4 — one
/// pair per pixel = `2 * width` u16), then split each into host-native
/// LE / BE wire `u16` buffers. Returns `(y_le, y_be, uv_le, uv_be)`.
#[allow(clippy::type_complexity)]
fn packed_le_be(width: usize, bits: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let shift = 16 - bits;
  let y: Vec<u16> = (0..width)
    .map(|i| logical_samp(i, 1, bits) << shift)
    .collect();
  let uv: Vec<u16> = (0..width)
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

/// One BITS' worth of the semi-planar 4:4:4 parity sweep: the
/// `p4NN_to_hsv_row_endian` kernel vs `rgb_to_hsv_row(p4NN_to_rgb_row_endian)`
/// for both endiannesses × matrices × range × widths × tiers.
macro_rules! parity_p4xx {
  ($name:ident, $bits:expr, $rgb_fn:path, $hsv_fn:path) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $name() {
      let bits: u32 = $bits;
      for &w in &[1usize, 2, 3, 7, 8, 15, 16, 31, 64, 65] {
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
                    "p4xx bits={bits} be={be} w={w} {matrix:?} full={full_range} simd={use_simd}"
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

parity_p4xx!(
  p410_hsv_row_matches_rgb_then_hsv,
  10,
  p410_to_rgb_row_endian,
  p410_to_hsv_row_endian
);
parity_p4xx!(
  p412_hsv_row_matches_rgb_then_hsv,
  12,
  p412_to_rgb_row_endian,
  p412_to_hsv_row_endian
);
parity_p4xx!(
  p416_hsv_row_matches_rgb_then_hsv,
  16,
  p416_to_rgb_row_endian,
  p416_to_hsv_row_endian
);

// ---- Structural: HSV-only attaches no source-width RGB scratch --------

/// Build a high-bit-packed LE Y plane + full-width interleaved UV plane
/// (4:4:4, one `U, V` pair per pixel = `2 * width` u16 per row) of varied
/// (non-gray) samples, as host-native LE-wire `u16` buffers (the
/// `*LeFrame` plane contract).
fn packed_le_frame_planes(width: usize, height: usize, bits: u32) -> (Vec<u16>, Vec<u16>) {
  let shift = 16 - bits;
  let y: Vec<u16> = (0..width * height)
    .map(|i| u16::from_ne_bytes((logical_samp(i, 1, bits) << shift).to_le_bytes()))
    .collect();
  let uv: Vec<u16> = (0..width * height)
    .flat_map(|i| {
      [
        u16::from_ne_bytes((logical_samp(i, 2, bits) << shift).to_le_bytes()),
        u16::from_ne_bytes((logical_samp(i, 3, bits) << shift).to_le_bytes()),
      ]
    })
    .collect();
  (y, uv)
}

/// Drives a P4NN sink (HSV-only, identity plan, LE wire) and asserts: the
/// source-width RGB scratch is never grown, and every HSV row equals the
/// two-step `rgb_to_hsv_row(p4NN_to_rgb_row_endian(...))` reference for the
/// driven tier.
macro_rules! structural_p4xx {
  ($name:ident, $bits:expr, $marker:ident, $frame:ident, $walker:path, $rgb_fn:path) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $name() {
      const BITS: u32 = $bits;
      let (w, h) = (16usize, 8usize);
      let m = KernelMatrix::Bt709;
      let (yp, uvp) = packed_le_frame_planes(w, h, BITS);
      // 4:4:4 interleaved UV stride = `2 * width` u16.
      let src = $frame::new(&yp, &uvp, w as u32, h as u32, w as u32, 2 * w as u32);

      let mut hh = std::vec![0u8; w * h];
      let mut ss = std::vec![0u8; w * h];
      let mut vv = std::vec![0u8; w * h];
      let scratch_len = {
        let mut sink = MixedSinker::<$marker>::new(w, h)
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap();
        $walker(&src, true, m, &mut sink).unwrap();
        sink.rgb_scratch.len()
      };
      assert_eq!(
        scratch_len, 0,
        "P4NN HSV-only must not grow the source-width RGB scratch"
      );

      // Every row matches the explicit YUV→RGB→HSV reference (SIMD tier).
      for r in 0..h {
        let mut rgb = std::vec![0u8; w * 3];
        $rgb_fn(
          &yp[r * w..r * w + w],
          &uvp[r * 2 * w..r * 2 * w + 2 * w],
          &mut rgb,
          w,
          m,
          true,
          true,
          false,
        );
        let (rh, rs, rv) = ref_hsv_from_rgb(&rgb, w, true);
        assert_eq!(&hh[r * w..r * w + w], &rh[..], "row {r} H");
        assert_eq!(&ss[r * w..r * w + w], &rs[..], "row {r} S");
        assert_eq!(&vv[r * w..r * w + w], &rv[..], "row {r} V");
      }
    }
  };
}

structural_p4xx!(
  p410_hsv_only_rgb_free_and_correct,
  10,
  P410,
  P410LeFrame,
  p410_to,
  p410_to_rgb_row_endian
);
structural_p4xx!(
  p412_hsv_only_rgb_free_and_correct,
  12,
  P412,
  P412LeFrame,
  p412_to,
  p412_to_rgb_row_endian
);
structural_p4xx!(
  p416_hsv_only_rgb_free_and_correct,
  16,
  P416,
  P416LeFrame,
  p416_to,
  p416_to_rgb_row_endian
);
