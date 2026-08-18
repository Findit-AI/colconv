//! Integration tests for the MSB-aligned high-bit planar YUV 4:4:4 sinker
//! impls (`Yuv444p10Msb` / `Yuv444p12Msb`).
//!
//! The oracle is the already-tested low-bit `Yuv444p10` / `Yuv444p12` family:
//! an MSB frame whose samples are `s << (16 - BITS)` must produce byte-identical
//! output to a low-bit frame whose samples are `s`, for every attached output
//! (rgb / rgb_u16 / rgba / rgba_u16 / luma / hsv) and on both the direct and
//! the fused-resample paths (native + with_native(false) row-stage). Endianness
//! and SIMD-vs-scalar parity are checked directly. (Yuv444p exposes no
//! `luma_u16` output, so it is absent here.)

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

/// Every output a `Yuv444p*` sink can emit (no `luma_u16`).
struct AllOutputs {
  rgb: std::vec::Vec<u8>,
  rgb_u16: std::vec::Vec<u16>,
  rgba: std::vec::Vec<u8>,
  rgba_u16: std::vec::Vec<u16>,
  luma: std::vec::Vec<u8>,
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
          let y = samples::<$bits>(0x1111, n);
          let u = samples::<$bits>(0x2222, n);
          let v = samples::<$bits>(0x3333, n);
          let (ym, um, vm) = (msb_align::<$bits>(&y), msb_align::<$bits>(&u), msb_align::<$bits>(&v));

          // MSB sink.
          let mut o_msb = AllOutputs::alloc(n);
          {
            let src = crate::frame::Yuv444pMsbFrame::<$bits>::new(
              &ym, &um, &vm, w as u32, h as u32, w as u32, w as u32, w as u32,
            );
            let mut sink = MixedSinker::<crate::source::$msb_marker>::new(w, h)
              .with_rgb(&mut o_msb.rgb).unwrap()
              .with_rgb_u16(&mut o_msb.rgb_u16).unwrap()
              .with_rgba(&mut o_msb.rgba).unwrap()
              .with_rgba_u16(&mut o_msb.rgba_u16).unwrap()
              .with_luma(&mut o_msb.luma).unwrap()
              .with_hsv(&mut o_msb.h, &mut o_msb.s, &mut o_msb.v).unwrap();
            crate::source::$msb_walker(&src, full_range, MATRIX, &mut sink).unwrap();
          }

          // Low-bit oracle.
          let mut o_lo = AllOutputs::alloc(n);
          {
            let src = crate::frame::Yuv444pFrame16::<$bits>::new(
              &y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
            );
            let mut sink = MixedSinker::<crate::source::$lo_marker>::new(w, h)
              .with_rgb(&mut o_lo.rgb).unwrap()
              .with_rgb_u16(&mut o_lo.rgb_u16).unwrap()
              .with_rgba(&mut o_lo.rgba).unwrap()
              .with_rgba_u16(&mut o_lo.rgba_u16).unwrap()
              .with_luma(&mut o_lo.luma).unwrap()
              .with_hsv(&mut o_lo.h, &mut o_lo.s, &mut o_lo.v).unwrap();
            crate::source::$lo_walker(&src, full_range, MATRIX, &mut sink).unwrap();
          }

          assert_eq!(o_msb.rgb, o_lo.rgb, "rgb full_range={full_range}");
          assert_eq!(o_msb.rgb_u16, o_lo.rgb_u16, "rgb_u16 full_range={full_range}");
          assert_eq!(o_msb.rgba, o_lo.rgba, "rgba full_range={full_range}");
          assert_eq!(o_msb.rgba_u16, o_lo.rgba_u16, "rgba_u16 full_range={full_range}");
          assert_eq!(o_msb.luma, o_lo.luma, "luma full_range={full_range}");
          assert_eq!(o_msb.h, o_lo.h, "hsv.h full_range={full_range}");
          assert_eq!(o_msb.s, o_lo.s, "hsv.s full_range={full_range}");
          assert_eq!(o_msb.v, o_lo.v, "hsv.v full_range={full_range}");
        }
      }

      // ---- HSV-only direct path (no RGB scratch) -----------------------

      #[test]
      #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
      fn hsv_only_matches_low_bit() {
        let w = 19usize;
        let h = 3usize;
        let n = w * h;
        let y = samples::<$bits>(0x9001, n);
        let u = samples::<$bits>(0x9002, n);
        let v = samples::<$bits>(0x9003, n);
        let (ym, um, vm) = (msb_align::<$bits>(&y), msb_align::<$bits>(&u), msb_align::<$bits>(&v));

        let (mut hm, mut sm, mut vm_out) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
        {
          let src = crate::frame::Yuv444pMsbFrame::<$bits>::new(
            &ym, &um, &vm, w as u32, h as u32, w as u32, w as u32, w as u32,
          );
          let mut sink = MixedSinker::<crate::source::$msb_marker>::new(w, h)
            .with_hsv(&mut hm, &mut sm, &mut vm_out).unwrap();
          crate::source::$msb_walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        let (mut hl, mut sl, mut vl) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
        {
          let src = crate::frame::Yuv444pFrame16::<$bits>::new(
            &y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
          );
          let mut sink = MixedSinker::<crate::source::$lo_marker>::new(w, h)
            .with_hsv(&mut hl, &mut sl, &mut vl).unwrap();
          crate::source::$lo_walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(hm, hl, "hsv-only h");
        assert_eq!(sm, sl, "hsv-only s");
        assert_eq!(vm_out, vl, "hsv-only v");
      }

      // ---- endian parity: BE storage recovers the same logical output --

      #[test]
      #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
      fn be_matches_le() {
        let w = 20usize;
        let h = 3usize;
        let n = w * h;
        let y = msb_align::<$bits>(&samples::<$bits>(0xAA, n));
        let u = msb_align::<$bits>(&samples::<$bits>(0xBB, n));
        let v = msb_align::<$bits>(&samples::<$bits>(0xCC, n));
        let (yb, ub, vb) = (to_be_storage(&y), to_be_storage(&u), to_be_storage(&v));

        let mut le = std::vec![0u16; n * 3];
        {
          let src = crate::frame::Yuv444pMsbFrame::<$bits, false>::new(
            &y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
          );
          let mut sink = MixedSinker::<crate::source::$msb_marker<false>>::new(w, h)
            .with_rgb_u16(&mut le).unwrap();
          crate::source::$msb_walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        let mut be = std::vec![0u16; n * 3];
        {
          let src = crate::frame::Yuv444pMsbFrame::<$bits, true>::new(
            &yb, &ub, &vb, w as u32, h as u32, w as u32, w as u32, w as u32,
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
          let y = msb_align::<$bits>(&samples::<$bits>(0x51, n));
          let u = msb_align::<$bits>(&samples::<$bits>(0x52, n));
          let v = msb_align::<$bits>(&samples::<$bits>(0x53, n));

          let mut s_out = AllOutputs::alloc(n);
          let mut c_out = AllOutputs::alloc(n);
          for (simd, o) in [(true, &mut s_out), (false, &mut c_out)] {
            let src = crate::frame::Yuv444pMsbFrame::<$bits>::new(
              &y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
            );
            let mut sink = MixedSinker::<crate::source::$msb_marker>::new(w, h)
              .with_simd(simd)
              .with_rgb(&mut o.rgb).unwrap()
              .with_rgb_u16(&mut o.rgb_u16).unwrap()
              .with_rgba(&mut o.rgba).unwrap()
              .with_rgba_u16(&mut o.rgba_u16).unwrap()
              .with_luma(&mut o.luma).unwrap();
            crate::source::$msb_walker(&src, true, MATRIX, &mut sink).unwrap();
          }
          // HSV is intentionally omitted: the SIMD HSV path
          // (`rgb_to_hsv_row`) and the scalar path (`rgb_to_hsv_pixel`) are
          // byte-identical only WITHIN a tier, not across — a pre-existing
          // property of the low-bit family (the `_to_hsv_row` dispatch
          // contract). HSV correctness vs the low-bit oracle is covered by
          // `direct_all_outputs_match_low_bit` / `hsv_only_matches_low_bit`,
          // which compare at the SAME tier so the tier difference cancels.
          assert_eq!(s_out.rgb, c_out.rgb, "rgb simd≠scalar w={w}");
          assert_eq!(s_out.rgb_u16, c_out.rgb_u16, "rgb_u16 simd≠scalar w={w}");
          assert_eq!(s_out.rgba, c_out.rgba, "rgba simd≠scalar w={w}");
          assert_eq!(s_out.rgba_u16, c_out.rgba_u16, "rgba_u16 simd≠scalar w={w}");
          assert_eq!(s_out.luma, c_out.luma, "luma simd≠scalar w={w}");
        }
      }

      // ---- fused downscale: every output matches the low-bit oracle ----
      //
      // Runs both the native fast tier (default) and the row-stage tail
      // (`with_native(false)`) — each must match the low-bit oracle on the
      // matching tier.

      #[test]
      #[cfg_attr(miri, ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri")]
      fn downscale_all_outputs_match_low_bit() {
        let src_w = 8usize;
        let src_h = 8usize;
        let out_w = 4usize;
        let out_h = 4usize;
        let n = src_w * src_h;
        let y = samples::<$bits>(0x71, n);
        let u = samples::<$bits>(0x72, n);
        let v = samples::<$bits>(0x73, n);
        let (ym, um, vm) = (msb_align::<$bits>(&y), msb_align::<$bits>(&u), msb_align::<$bits>(&v));

        for native in [true, false] {
          let mut o_msb = AllOutputs::alloc(out_w * out_h);
          {
            let src = crate::frame::Yuv444pMsbFrame::<$bits>::new(
              &ym, &um, &vm, src_w as u32, src_h as u32, src_w as u32, src_w as u32, src_w as u32,
            );
            let mut sink = MixedSinker::<crate::source::$msb_marker, crate::resample::AreaResampler>::with_resampler(
              src_w, src_h, crate::resample::AreaResampler::to(out_w, out_h),
            )
            .unwrap()
            .with_native(native)
            .with_rgb(&mut o_msb.rgb).unwrap()
            .with_rgb_u16(&mut o_msb.rgb_u16).unwrap()
            .with_rgba(&mut o_msb.rgba).unwrap()
            .with_rgba_u16(&mut o_msb.rgba_u16).unwrap()
            .with_luma(&mut o_msb.luma).unwrap()
            .with_hsv(&mut o_msb.h, &mut o_msb.s, &mut o_msb.v).unwrap();
            crate::source::$msb_walker(&src, true, MATRIX, &mut sink).unwrap();
          }

          let mut o_lo = AllOutputs::alloc(out_w * out_h);
          {
            let src = crate::frame::Yuv444pFrame16::<$bits>::new(
              &y, &u, &v, src_w as u32, src_h as u32, src_w as u32, src_w as u32, src_w as u32,
            );
            let mut sink = MixedSinker::<crate::source::$lo_marker, crate::resample::AreaResampler>::with_resampler(
              src_w, src_h, crate::resample::AreaResampler::to(out_w, out_h),
            )
            .unwrap()
            .with_native(native)
            .with_rgb(&mut o_lo.rgb).unwrap()
            .with_rgb_u16(&mut o_lo.rgb_u16).unwrap()
            .with_rgba(&mut o_lo.rgba).unwrap()
            .with_rgba_u16(&mut o_lo.rgba_u16).unwrap()
            .with_luma(&mut o_lo.luma).unwrap()
            .with_hsv(&mut o_lo.h, &mut o_lo.s, &mut o_lo.v).unwrap();
            crate::source::$lo_walker(&src, true, MATRIX, &mut sink).unwrap();
          }

          assert_eq!(o_msb.rgb, o_lo.rgb, "resampled rgb native={native}");
          assert_eq!(o_msb.rgb_u16, o_lo.rgb_u16, "resampled rgb_u16 native={native}");
          assert_eq!(o_msb.rgba, o_lo.rgba, "resampled rgba native={native}");
          assert_eq!(o_msb.rgba_u16, o_lo.rgba_u16, "resampled rgba_u16 native={native}");
          assert_eq!(o_msb.luma, o_lo.luma, "resampled luma native={native}");
          assert_eq!(o_msb.h, o_lo.h, "resampled hsv.h native={native}");
          assert_eq!(o_msb.s, o_lo.s, "resampled hsv.s native={native}");
          assert_eq!(o_msb.v, o_lo.v, "resampled hsv.v native={native}");
        }
      }
    }
  };
}

msb_parity_suite!(
  bits10,
  10,
  Yuv444p10Msb,
  yuv444p10_msb_to,
  yuv444p10_msb_to_endian,
  Yuv444p10,
  yuv444p10_to
);
msb_parity_suite!(
  bits12,
  12,
  Yuv444p12Msb,
  yuv444p12_msb_to,
  yuv444p12_msb_to_endian,
  Yuv444p12,
  yuv444p12_to
);

// ---- Atomicity (#308): MSB-aligned high-bit 4:4:4 planar ---------------
//
// The MSB recovery shares the identity-path `process` shape of the low-bit
// `Yuv444p10` / `Yuv444p12` family, so it inherits the same up-front
// RGB-scratch preflight: it must return `AllocationFailed` BEFORE any output
// row — luma included — is written, leaving the output frame untouched on an
// allocator refusal. Triggering set: luma + RGBA + HSV with NO rgb output —
// `want_hsv && want_rgba && !want_rgb` would grow `rgb_row_buf_or_scratch`'s
// scratch arm (the only growable scratch on the identity path; the u16 RGB /
// RGBA outputs write straight into their caller buffers). Reuses the crate's
// `yuva`-gated RGB-scratch failpoint.

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_msb_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;

  let n = 16 * 8;
  let y = msb_align::<10>(&std::vec![512u16; n]);
  let u = msb_align::<10>(&std::vec![512u16; n]);
  let v = msb_align::<10>(&std::vec![512u16; n]);
  let src = crate::frame::Yuv444pMsbFrame::<10>::new(&y, &u, &v, 16, 8, 16, 16, 16);
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<crate::source::Yuv444p10Msb>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = crate::source::yuv444p10_msb_to(&src, false, MATRIX, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p12_msb_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;

  let n = 16 * 8;
  let y = msb_align::<12>(&std::vec![2048u16; n]);
  let u = msb_align::<12>(&std::vec![2048u16; n]);
  let v = msb_align::<12>(&std::vec![2048u16; n]);
  let src = crate::frame::Yuv444pMsbFrame::<12>::new(&y, &u, &v, 16, 8, 16, 16, 16);
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<crate::source::Yuv444p12Msb>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = crate::source::yuv444p12_msb_to(&src, false, MATRIX, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}
