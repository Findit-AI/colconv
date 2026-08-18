//! Integration tests for the MSB-aligned high-bit planar GBR sinker impls
//! (`Gbrp10Msb` / `Gbrp12Msb`).
//!
//! The oracle is the already-tested low-bit `Gbrp10` / `Gbrp12` family: an
//! MSB frame whose samples are `s << (16 - BITS)` must produce byte-identical
//! output to a low-bit frame whose samples are `s`, for every attached output
//! (rgb / rgb_u16 / rgba / rgba_u16 / luma / luma_u16 / hsv) and on both the
//! direct and the fused-resample paths. Endianness and SIMD-vs-scalar parity
//! are checked directly.

use crate::{KernelMatrix, sinker::MixedSinker};

const MATRIX: KernelMatrix = KernelMatrix::Bt709;

/// Deterministic native-depth samples masked to `BITS` (legal native codes).
fn samples<const BITS: u32>(seed: u64, n: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let mut s = seed;
  (0..n)
    .map(|_| {
      s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      ((s >> 33) as u32 & mask) as u16
    })
    .collect()
}

/// MSB-align a native sample slice into the high `BITS` bits of each `u16`.
fn msb_align<const BITS: u32>(low: &[u16]) -> std::vec::Vec<u16> {
  low.iter().map(|&s| s << (16 - BITS)).collect()
}

/// Byte-swap a u16 slice's storage so a `BE = true` kernel recovers the same
/// logical values a `BE = false` kernel sees on the LE-stored twin.
fn to_be_storage(le: &[u16]) -> std::vec::Vec<u16> {
  le.iter()
    .map(|&v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Run a low-bit `GbrpN` sink with every output attached → flattened bytes.
struct AllOutputs {
  rgb: std::vec::Vec<u8>,
  rgb_u16: std::vec::Vec<u16>,
  rgba: std::vec::Vec<u8>,
  rgba_u16: std::vec::Vec<u16>,
  luma: std::vec::Vec<u8>,
  luma_u16: std::vec::Vec<u16>,
  h: std::vec::Vec<u8>,
  s: std::vec::Vec<u8>,
  v: std::vec::Vec<u8>,
}

impl AllOutputs {
  fn alloc(n: usize) -> Self {
    Self {
      rgb: std::vec![0u8; n * 3],
      rgb_u16: std::vec![0u16; n * 3],
      rgba: std::vec![0u8; n * 4],
      rgba_u16: std::vec![0u16; n * 4],
      luma: std::vec![0u8; n],
      luma_u16: std::vec![0u16; n],
      h: std::vec![0u8; n],
      s: std::vec![0u8; n],
      v: std::vec![0u8; n],
    }
  }
}

macro_rules! msb_parity_suite {
  ($mod:ident, $bits:literal, $msb_marker:ident, $msb_walker:ident, $msb_endian:ident, $lo_marker:ident, $lo_walker:ident) => {
    mod $mod {
      use super::*;

      // ---- direct-path parity vs the low-bit oracle --------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
      fn direct_all_outputs_match_low_bit() {
        let w = 37usize;
        let h = 5usize;
        let n = w * h;
        for full_range in [true, false] {
          let g = samples::<$bits>(0x1111, n);
          let b = samples::<$bits>(0x2222, n);
          let r = samples::<$bits>(0x3333, n);
          let (gm, bm, rm) = (msb_align::<$bits>(&g), msb_align::<$bits>(&b), msb_align::<$bits>(&r));

          // MSB sink.
          let mut o_msb = AllOutputs::alloc(n);
          {
            let src = crate::frame::GbrpMsbFrame::<$bits>::new(
              &gm, &bm, &rm, w as u32, h as u32, w as u32, w as u32, w as u32,
            );
            let mut sink = MixedSinker::<crate::source::$msb_marker>::new(w, h)
              .with_rgb(&mut o_msb.rgb).unwrap()
              .with_rgb_u16(&mut o_msb.rgb_u16).unwrap()
              .with_rgba(&mut o_msb.rgba).unwrap()
              .with_rgba_u16(&mut o_msb.rgba_u16).unwrap()
              .with_luma(&mut o_msb.luma).unwrap()
              .with_luma_u16(&mut o_msb.luma_u16).unwrap()
              .with_hsv(&mut o_msb.h, &mut o_msb.s, &mut o_msb.v).unwrap();
            crate::source::$msb_walker(&src, full_range, MATRIX, &mut sink).unwrap();
          }

          // Low-bit oracle.
          let mut o_lo = AllOutputs::alloc(n);
          {
            let src = crate::frame::GbrpHighBitFrame::<$bits>::new(
              &g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32,
            );
            let mut sink = MixedSinker::<crate::source::$lo_marker>::new(w, h)
              .with_rgb(&mut o_lo.rgb).unwrap()
              .with_rgb_u16(&mut o_lo.rgb_u16).unwrap()
              .with_rgba(&mut o_lo.rgba).unwrap()
              .with_rgba_u16(&mut o_lo.rgba_u16).unwrap()
              .with_luma(&mut o_lo.luma).unwrap()
              .with_luma_u16(&mut o_lo.luma_u16).unwrap()
              .with_hsv(&mut o_lo.h, &mut o_lo.s, &mut o_lo.v).unwrap();
            crate::source::$lo_walker(&src, full_range, MATRIX, &mut sink).unwrap();
          }

          assert_eq!(o_msb.rgb, o_lo.rgb, "rgb full_range={full_range}");
          assert_eq!(o_msb.rgb_u16, o_lo.rgb_u16, "rgb_u16 full_range={full_range}");
          assert_eq!(o_msb.rgba, o_lo.rgba, "rgba full_range={full_range}");
          assert_eq!(o_msb.rgba_u16, o_lo.rgba_u16, "rgba_u16 full_range={full_range}");
          assert_eq!(o_msb.luma, o_lo.luma, "luma full_range={full_range}");
          assert_eq!(o_msb.luma_u16, o_lo.luma_u16, "luma_u16 full_range={full_range}");
          assert_eq!(o_msb.h, o_lo.h, "hsv.h full_range={full_range}");
          assert_eq!(o_msb.s, o_lo.s, "hsv.s full_range={full_range}");
          assert_eq!(o_msb.v, o_lo.v, "hsv.v full_range={full_range}");
        }
      }

      // ---- endian parity: BE storage recovers the same logical output --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
      fn be_matches_le() {
        let w = 20usize;
        let h = 3usize;
        let n = w * h;
        let g = msb_align::<$bits>(&samples::<$bits>(0xAA, n));
        let b = msb_align::<$bits>(&samples::<$bits>(0xBB, n));
        let r = msb_align::<$bits>(&samples::<$bits>(0xCC, n));
        let (gb, bb, rb) = (to_be_storage(&g), to_be_storage(&b), to_be_storage(&r));

        let mut le = std::vec![0u16; n * 3];
        {
          let src = crate::frame::GbrpMsbFrame::<$bits, false>::new(
            &g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32,
          );
          let mut sink = MixedSinker::<crate::source::$msb_marker<false>>::new(w, h)
            .with_rgb_u16(&mut le).unwrap();
          crate::source::$msb_walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        let mut be = std::vec![0u16; n * 3];
        {
          let src = crate::frame::GbrpMsbFrame::<$bits, true>::new(
            &gb, &bb, &rb, w as u32, h as u32, w as u32, w as u32, w as u32,
          );
          let mut sink = MixedSinker::<crate::source::$msb_marker<true>>::new(w, h)
            .with_rgb_u16(&mut be).unwrap();
          crate::source::$msb_endian::<_, true>(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(le, be, "BE storage must recover the same logical RGB as LE");
      }

      // ---- SIMD-vs-scalar parity (main loop + tail widths) -------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
      fn simd_matches_scalar() {
        for w in [128usize, 130usize] {
          let h = 4usize;
          let n = w * h;
          let g = msb_align::<$bits>(&samples::<$bits>(0x51, n));
          let b = msb_align::<$bits>(&samples::<$bits>(0x52, n));
          let r = msb_align::<$bits>(&samples::<$bits>(0x53, n));

          let mut s_out = AllOutputs::alloc(n);
          let mut c_out = AllOutputs::alloc(n);
          for (simd, o) in [(true, &mut s_out), (false, &mut c_out)] {
            let src = crate::frame::GbrpMsbFrame::<$bits>::new(
              &g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32,
            );
            let mut sink = MixedSinker::<crate::source::$msb_marker>::new(w, h)
              .with_simd(simd)
              .with_rgb(&mut o.rgb).unwrap()
              .with_rgb_u16(&mut o.rgb_u16).unwrap()
              .with_rgba(&mut o.rgba).unwrap()
              .with_rgba_u16(&mut o.rgba_u16).unwrap()
              .with_luma_u16(&mut o.luma_u16).unwrap();
            crate::source::$msb_walker(&src, true, MATRIX, &mut sink).unwrap();
          }
          assert_eq!(s_out.rgb, c_out.rgb, "rgb simd≠scalar w={w}");
          assert_eq!(s_out.rgb_u16, c_out.rgb_u16, "rgb_u16 simd≠scalar w={w}");
          assert_eq!(s_out.rgba, c_out.rgba, "rgba simd≠scalar w={w}");
          assert_eq!(s_out.rgba_u16, c_out.rgba_u16, "rgba_u16 simd≠scalar w={w}");
          assert_eq!(s_out.luma_u16, c_out.luma_u16, "luma_u16 simd≠scalar w={w}");
        }
      }

      // ---- fused downscale: every output matches the low-bit oracle ----

      #[test]
      #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
      fn downscale_all_outputs_match_low_bit() {
        let src_w = 8usize;
        let src_h = 8usize;
        let out_w = 4usize;
        let out_h = 4usize;
        let n = src_w * src_h;
        let g = samples::<$bits>(0x71, n);
        let b = samples::<$bits>(0x72, n);
        let r = samples::<$bits>(0x73, n);
        let (gm, bm, rm) = (msb_align::<$bits>(&g), msb_align::<$bits>(&b), msb_align::<$bits>(&r));

        let mut o_msb = AllOutputs::alloc(out_w * out_h);
        {
          let src = crate::frame::GbrpMsbFrame::<$bits>::new(
            &gm, &bm, &rm, src_w as u32, src_h as u32, src_w as u32, src_w as u32, src_w as u32,
          );
          let mut sink = MixedSinker::<crate::source::$msb_marker, crate::resample::AreaResampler>::with_resampler(
            src_w, src_h, crate::resample::AreaResampler::to(out_w, out_h),
          )
          .unwrap()
          .with_rgb(&mut o_msb.rgb).unwrap()
          .with_rgb_u16(&mut o_msb.rgb_u16).unwrap()
          .with_rgba(&mut o_msb.rgba).unwrap()
          .with_rgba_u16(&mut o_msb.rgba_u16).unwrap()
          .with_luma(&mut o_msb.luma).unwrap()
          .with_luma_u16(&mut o_msb.luma_u16).unwrap()
          .with_hsv(&mut o_msb.h, &mut o_msb.s, &mut o_msb.v).unwrap();
          crate::source::$msb_walker(&src, true, MATRIX, &mut sink).unwrap();
        }

        let mut o_lo = AllOutputs::alloc(out_w * out_h);
        {
          let src = crate::frame::GbrpHighBitFrame::<$bits>::new(
            &g, &b, &r, src_w as u32, src_h as u32, src_w as u32, src_w as u32, src_w as u32,
          );
          let mut sink = MixedSinker::<crate::source::$lo_marker, crate::resample::AreaResampler>::with_resampler(
            src_w, src_h, crate::resample::AreaResampler::to(out_w, out_h),
          )
          .unwrap()
          .with_rgb(&mut o_lo.rgb).unwrap()
          .with_rgb_u16(&mut o_lo.rgb_u16).unwrap()
          .with_rgba(&mut o_lo.rgba).unwrap()
          .with_rgba_u16(&mut o_lo.rgba_u16).unwrap()
          .with_luma(&mut o_lo.luma).unwrap()
          .with_luma_u16(&mut o_lo.luma_u16).unwrap()
          .with_hsv(&mut o_lo.h, &mut o_lo.s, &mut o_lo.v).unwrap();
          crate::source::$lo_walker(&src, true, MATRIX, &mut sink).unwrap();
        }

        assert_eq!(o_msb.rgb, o_lo.rgb, "resampled rgb");
        assert_eq!(o_msb.rgb_u16, o_lo.rgb_u16, "resampled rgb_u16");
        assert_eq!(o_msb.rgba, o_lo.rgba, "resampled rgba");
        assert_eq!(o_msb.rgba_u16, o_lo.rgba_u16, "resampled rgba_u16");
        assert_eq!(o_msb.luma, o_lo.luma, "resampled luma");
        assert_eq!(o_msb.luma_u16, o_lo.luma_u16, "resampled luma_u16");
        assert_eq!(o_msb.h, o_lo.h, "resampled hsv.h");
        assert_eq!(o_msb.s, o_lo.s, "resampled hsv.s");
        assert_eq!(o_msb.v, o_lo.v, "resampled hsv.v");
      }
    }
  };
}

msb_parity_suite!(
  bits10,
  10,
  Gbrp10Msb,
  gbrp10_msb_to,
  gbrp10_msb_to_endian,
  Gbrp10,
  gbrp10_to
);
msb_parity_suite!(
  bits12,
  12,
  Gbrp12Msb,
  gbrp12_msb_to,
  gbrp12_msb_to_endian,
  Gbrp12,
  gbrp12_to
);
