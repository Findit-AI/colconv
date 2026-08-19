//! #263 — direct semi-planar **high-bit 4:2:2** P-format YUV→HSV routing
//! (P210 / P212 / P216).
//!
//! 4:2:2 reuses the 4:2:0 row kernels: the per-row chroma contract is
//! identical (half-width interleaved `U, V`, horizontally 1→2 upsampled),
//! and the 4:2:0-vs-4:2:2 difference is purely vertical (resolved by the
//! walker feeding one chroma row per luma row). So the P210/P212/P216 HSV
//! row kernel IS the P010/P012/P016 one — whose per-tier byte-identity to
//! `rgb_to_hsv_row(pNNN_to_rgb_row_endian(...))` is already covered by the
//! 4:2:0 (p0xx) HSV-direct suite. This suite asserts the **sink routing**:
//! a P2NN sink with ONLY `with_hsv()` must take the direct kernel (no
//! source-width RGB scratch) and produce the same HSV planes as the
//! per-row two-step YUV→RGB→HSV reference.

use super::*;
use crate::row::{
  p010_to_rgb_row_endian, p012_to_rgb_row_endian, p016_to_rgb_row_endian, rgb_to_hsv_row,
};

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

fn ref_hsv_from_rgb(rgb: &[u8], w: usize, use_simd: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut h = std::vec![0u8; w];
  let mut s = std::vec![0u8; w];
  let mut v = std::vec![0u8; w];
  rgb_to_hsv_row(rgb, &mut h, &mut s, &mut v, w, use_simd);
  (h, s, v)
}

/// Build a high-bit-packed LE Y plane + half-width / **full-height**
/// interleaved UV plane (4:2:2, `width` u16 per chroma row = `width / 2`
/// pairs) of varied (non-gray) samples, as host-native LE-wire `u16`
/// buffers (the `*LeFrame` plane contract).
fn packed_le_frame_planes(width: usize, height: usize, bits: u32) -> (Vec<u16>, Vec<u16>) {
  let shift = 16 - bits;
  let y: Vec<u16> = (0..width * height)
    .map(|i| u16::from_ne_bytes((logical_samp(i, 1, bits) << shift).to_le_bytes()))
    .collect();
  // 4:2:2 chroma: half-width, full-height, interleaved U,V (= `width` u16
  // per row).
  let uv: Vec<u16> = (0..(width / 2) * height)
    .flat_map(|i| {
      [
        u16::from_ne_bytes((logical_samp(i, 2, bits) << shift).to_le_bytes()),
        u16::from_ne_bytes((logical_samp(i, 3, bits) << shift).to_le_bytes()),
      ]
    })
    .collect();
  (y, uv)
}

/// Drives a P2NN sink (HSV-only, identity plan, LE wire) and asserts: the
/// source-width RGB scratch is never grown (the direct kernel ran), and
/// every HSV row equals the two-step `rgb_to_hsv_row(pNNN_to_rgb_row_endian
/// (...))` reference (the same row kernel the 4:2:2 sink reuses).
macro_rules! structural_p2xx {
  ($name:ident, $bits:expr, $marker:ident, $frame:ident, $walker:path, $rgb_fn:path) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $name() {
      const BITS: u32 = $bits;
      let (w, h) = (16usize, 8usize);
      let m = KernelMatrix::Bt2020Ncl;
      let (yp, uvp) = packed_le_frame_planes(w, h, BITS);
      // 4:2:2 interleaved UV stride = `width` u16 (half-width pairs, full
      // height — one chroma row per luma row).
      let src = $frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

      let mut hh = std::vec![0u8; w * h];
      let mut ss = std::vec![0u8; w * h];
      let mut vv = std::vec![0u8; w * h];
      let scratch_len = {
        let mut sink = MixedSinker::<$marker>::new(w, h)
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap();
        $walker(&src, true, sink.set_kernel_matrix(m)).unwrap();
        sink.rgb_scratch.len()
      };
      assert_eq!(
        scratch_len, 0,
        "P2NN HSV-only must not grow the source-width RGB scratch"
      );

      // Every row matches the explicit YUV→RGB→HSV reference (host tier).
      // 4:2:2 feeds one chroma row per luma row (UV stride = width).
      for r in 0..h {
        let mut rgb = std::vec![0u8; w * 3];
        $rgb_fn(
          &yp[r * w..r * w + w],
          &uvp[r * w..r * w + w],
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

structural_p2xx!(
  p210_hsv_only_rgb_free_and_correct,
  10,
  P210,
  P210LeFrame,
  p210_to,
  p010_to_rgb_row_endian
);
structural_p2xx!(
  p212_hsv_only_rgb_free_and_correct,
  12,
  P212,
  P212LeFrame,
  p212_to,
  p012_to_rgb_row_endian
);
structural_p2xx!(
  p216_hsv_only_rgb_free_and_correct,
  16,
  P216,
  P216LeFrame,
  p216_to,
  p016_to_rgb_row_endian
);
