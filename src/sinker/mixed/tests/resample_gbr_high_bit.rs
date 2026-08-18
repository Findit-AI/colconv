//! Fused-downscale coverage for the high-bit planar GBR family
//! (`Gbrp9` / `Gbrp10` / `Gbrp12` / `Gbrp14` / `Gbrp16`).
//!
//! Each `GbrpN` scatters its native-depth G/B/R planes into a source-width
//! packed `u16` RGB row and feeds the shared high-bit packed-RGB resample
//! tail (the same one `Rgb48` / `Bgr48` use, parameterized by the source
//! depth `BITS`). Binning runs at native depth, so:
//! - `rgb_u16` is the exact native 2x2 block mean,
//! - every output (rgb / rgba / rgb_u16 / rgba_u16 / luma / luma_u16 / hsv)
//!   matches a **direct** full-resolution `GbrpN` conversion of the
//!   pre-binned frame — `luma_u16` at native precision, full parity.
//!
//! The out-of-sequence / mid-frame contract is exercised by the shared
//! tail's `resample_rgb48` suite against the exact same stream/preflight
//! functions; `GbrpNRow::new` is `pub(crate)` in `mediaframe`, so a high-bit
//! GBR row can only reach `process` through the in-order walker and a direct
//! out-of-order `process` call cannot be constructed here (mirrors the 8-bit
//! `resample_gbrp` suite).

use crate::{KernelMatrix, sinker::MixedSinker};

const SRC: usize = 8;
const OUT: usize = 4;
const MATRIX: KernelMatrix = KernelMatrix::Bt709;

/// Native-depth `(r, g, b)` ramp for source pixel `i`, masked to `BITS` so
/// every sample is a legal native code; interior values so the derived luma
/// / HSV kernels see real math and the wide accumulator carries bits a u8
/// path would drop.
fn rgb_px<const BITS: u32>(i: usize) -> [u16; 3] {
  let mask = (1u32 << BITS) - 1;
  let r = (40u32 * BITS + (i as u32) * 173) & mask;
  let g = mask.wrapping_sub((i as u32) * 211) & mask;
  let b = (1000u32 + (i as u32 % 8) * 4099) & mask;
  [r as u16, g as u16, b as u16]
}

/// Source-width packed native-u16 RGB ramp (`SRC * SRC * 3` elements).
fn rgb_ramp<const BITS: u32>() -> Vec<u16> {
  let mut buf = std::vec![0u16; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px.copy_from_slice(&rgb_px::<BITS>(i));
  }
  buf
}

/// Scatter a packed-RGB u16 buffer into `(g, b, r)` planes — the inverse of
/// `gbr_to_rgb_u16_high_bit_row`. Each plane has `width * height` elements.
fn planes_from_packed_rgb(rgb: &[u16], n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let (mut g, mut b, mut r) = (std::vec![0u16; n], std::vec![0u16; n], std::vec![0u16; n]);
  for i in 0..n {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  (g, b, r)
}

/// Exact 2x2 block mean with round-half-up over native u16 values — the
/// integer-area-mean contract for a 2:1 downscale at native depth.
fn expected_block_mean(rgb: &[u16], ox: usize, oy: usize, c: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

macro_rules! gbr_high_bit_resample_tests {
  ($mod:ident, $marker:ident, $walker:ident, $bits:literal) => {
    mod $mod {
      use super::*;

      fn frame<'a>(
        g: &'a [u16],
        b: &'a [u16],
        r: &'a [u16],
        w: usize,
        h: usize,
      ) -> crate::frame::GbrpHighBitFrame<'a, $bits> {
        crate::frame::GbrpHighBitFrame::try_new(g, b, r, w as u32, h as u32, w as u32, w as u32, w as u32)
          .unwrap()
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn downscale_rgb_u16_is_exact_native_area_mean() {
        let rgb = rgb_ramp::<$bits>();
        let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
        let src = frame(&g, &b, &r, SRC, SRC);

        let mut out = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
            SRC,
            SRC,
            crate::resample::AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb_u16(&mut out)
          .unwrap();
          crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        for oy in 0..OUT {
          for ox in 0..OUT {
            for c in 0..3 {
              assert_eq!(
                out[(oy * OUT + ox) * 3 + c],
                expected_block_mean(&rgb, ox, oy, c),
                "({ox},{oy}) c{c}"
              );
            }
          }
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn all_outputs_match_direct_conversion_of_prebinned_frame() {
        // Resample SRC->OUT with every output attached, then compare against
        // a full-resolution direct GbrpN conversion of the pre-binned
        // (native block-mean) frame — the parity oracle. Every output,
        // luma_u16 included at native precision, matches the direct path.
        let rgb = rgb_ramp::<$bits>();
        let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
        let src = frame(&g, &b, &r, SRC, SRC);

        let mut rgb_o = std::vec![0u8; OUT * OUT * 3];
        let mut rgb_u16_o = std::vec![0u16; OUT * OUT * 3];
        let mut rgba_o = std::vec![0u8; OUT * OUT * 4];
        let mut rgba_u16_o = std::vec![0u16; OUT * OUT * 4];
        let mut luma_o = std::vec![0u8; OUT * OUT];
        let mut lu16_o = std::vec![0u16; OUT * OUT];
        let mut h_o = std::vec![0u8; OUT * OUT];
        let mut s_o = std::vec![0u8; OUT * OUT];
        let mut v_o = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
            SRC,
            SRC,
            crate::resample::AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb(&mut rgb_o)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16_o)
          .unwrap()
          .with_rgba(&mut rgba_o)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16_o)
          .unwrap()
          .with_luma(&mut luma_o)
          .unwrap()
          .with_luma_u16(&mut lu16_o)
          .unwrap()
          .with_hsv(&mut h_o, &mut s_o, &mut v_o)
          .unwrap();
          crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
        }

        // The resampled rgb_u16 IS the exact native block mean; assert that
        // link explicitly, then drive the oracle from the same samples.
        let mut binned = std::vec![0u16; OUT * OUT * 3];
        for oy in 0..OUT {
          for ox in 0..OUT {
            for c in 0..3 {
              binned[(oy * OUT + ox) * 3 + c] = expected_block_mean(&rgb, ox, oy, c);
            }
          }
        }
        assert_eq!(rgb_u16_o, binned, "resample rgb_u16 == exact native block-mean");

        let (bg, bb, br) = planes_from_packed_rgb(&binned, OUT * OUT);
        let binned_src = frame(&bg, &bb, &br, OUT, OUT);
        let mut rgb_ref = std::vec![0u8; OUT * OUT * 3];
        let mut rgba_ref = std::vec![0u8; OUT * OUT * 4];
        let mut rgba_u16_ref = std::vec![0u16; OUT * OUT * 4];
        let mut luma_ref = std::vec![0u8; OUT * OUT];
        let mut lu16_ref = std::vec![0u16; OUT * OUT];
        let mut h_ref = std::vec![0u8; OUT * OUT];
        let mut s_ref = std::vec![0u8; OUT * OUT];
        let mut v_ref = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::$marker>::new(OUT, OUT)
            .with_rgb(&mut rgb_ref)
            .unwrap()
            .with_rgba(&mut rgba_ref)
            .unwrap()
            .with_rgba_u16(&mut rgba_u16_ref)
            .unwrap()
            .with_luma(&mut luma_ref)
            .unwrap()
            .with_luma_u16(&mut lu16_ref)
            .unwrap()
            .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
            .unwrap();
          crate::source::$walker(&binned_src, true, MATRIX, &mut sink).unwrap();
        }

        assert_eq!(rgb_o, rgb_ref, "rgb (narrowed)");
        assert_eq!(rgba_o, rgba_ref, "rgba (narrowed, alpha forced max)");
        assert_eq!(rgba_u16_o, rgba_u16_ref, "rgba_u16 (native, alpha forced max)");
        assert_eq!(luma_o, luma_ref, "luma (narrowed)");
        assert_eq!(h_o, h_ref, "hsv H");
        assert_eq!(s_o, s_ref, "hsv S");
        assert_eq!(v_o, v_ref, "hsv V");
        // luma_u16 on the fused path is native-precision — byte-identical
        // to the direct GbrpN with_luma_u16 of the binned frame.
        assert_eq!(lu16_o, lu16_ref, "luma_u16 (native, full parity)");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_new_sink() {
        let rgb = rgb_ramp::<$bits>();
        let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
        let src = frame(&g, &b, &r, SRC, SRC);

        let mut direct = std::vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<crate::source::$marker>::new(SRC, SRC)
            .with_rgb_u16(&mut direct)
            .unwrap();
          crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        let mut via_area = std::vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
            SRC,
            SRC,
            crate::resample::AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgb_u16(&mut via_area)
          .unwrap();
          crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(direct, via_area, "identity-plan resample == direct sink");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resample_no_outputs_is_a_no_op() {
        // A resampling sink with no attached outputs is the documented legal
        // no-op: it walks every row and returns Ok without touching any
        // caller buffer (there is none to touch).
        let rgb = rgb_ramp::<$bits>();
        let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
        let src = frame(&g, &b, &r, SRC, SRC);

        let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
          SRC,
          SRC,
          crate::resample::AreaResampler::to(OUT, OUT),
        )
        .unwrap();
        crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resample_reuses_stream_across_frames() {
        // begin_frame resets the u16 area stream + frozen output set, so
        // frame 2's row 0 is accepted (not rejected as out-of-sequence) and
        // the output reflects frame 2's input — without the reset it would
        // still show frame 1. Both frames share one output buffer; only the
        // input data changes.
        let mask = ((1u32 << $bits) - 1) as u16;
        let rgb1 = rgb_ramp::<$bits>();
        let rgb2: Vec<u16> = rgb1.iter().map(|&p| mask - p).collect();
        let (g1, b1, r1) = planes_from_packed_rgb(&rgb1, SRC * SRC);
        let (g2, b2, r2) = planes_from_packed_rgb(&rgb2, SRC * SRC);

        let mut out = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
            SRC,
            SRC,
            crate::resample::AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb_u16(&mut out)
          .unwrap();
          crate::source::$walker(&frame(&g1, &b1, &r1, SRC, SRC), true, MATRIX, &mut sink).unwrap();
          crate::source::$walker(&frame(&g2, &b2, &r2, SRC, SRC), true, MATRIX, &mut sink).unwrap();
        }

        let mut expected = std::vec![0u16; OUT * OUT * 3];
        for oy in 0..OUT {
          for ox in 0..OUT {
            for c in 0..3 {
              expected[(oy * OUT + ox) * 3 + c] = expected_block_mean(&rgb2, ox, oy, c);
            }
          }
        }
        assert_eq!(out, expected, "frame 2 output must area-downscale frame 2");
      }
    }
  };
}

gbr_high_bit_resample_tests!(gbrp9, Gbrp9, gbrp9_to, 9);
gbr_high_bit_resample_tests!(gbrp10, Gbrp10, gbrp10_to, 10);
gbr_high_bit_resample_tests!(gbrp12, Gbrp12, gbrp12_to, 12);
gbr_high_bit_resample_tests!(gbrp14, Gbrp14, gbrp14_to, 14);
gbr_high_bit_resample_tests!(gbrp16, Gbrp16, gbrp16_to, 16);

// ---- Filter (separable, PIL-parity) routing -----------------------------
//
// The `GbrpN` filter arm scatters the native-depth G/B/R planes into the
// same source-width packed `u16` RGB row the area arm builds, then bins it
// through the signed-coefficient filter stream — the `u16` twin of the
// `Rgb48` filter routing. The native-depth `rgb_u16` output is the binned
// row verbatim (no narrowing), so a `GbrpN` filter `rgb_u16` is
// byte-identical to the `Rgb48` filter `rgb_u16` of the same logical
// pixels, for every kernel and every (down/up) ratio.

/// A filter plan must be accepted by the high-bit GBR sink — feature-
/// independent (no `Rgb48` reference needed), so it also guards the
/// `gbr`-solo build. Before this routing the filter plan was rejected with
/// `UnsupportedFilter`; now it produces a real (non-sentinel) output.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_filter_plan_is_accepted() {
  use crate::resample::{FilteredResampler, Triangle};

  const BITS: u32 = 10;
  let rgb = rgb_ramp::<BITS>();
  let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
  let src = crate::frame::GbrpHighBitFrame::<BITS>::try_new(
    &g, &b, &r, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  )
  .unwrap();

  let mut out = std::vec![0xABCDu16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<crate::source::Gbrp10, FilteredResampler<Triangle>>::with_resampler(
        SRC,
        SRC,
        FilteredResampler::new(OUT, OUT, Triangle),
      )
      .unwrap()
      .with_rgb_u16(&mut out)
      .unwrap();
    // Accepted: a filter plan no longer raises `UnsupportedFilter`.
    crate::source::gbrp10_to(&src, true, MATRIX, &mut sink).unwrap();
  }
  // The filter pass wrote every output element (the sentinel is gone).
  assert!(
    out.iter().all(|&v| v != 0xABCD),
    "filter resample must populate rgb_u16"
  );
}

/// Native-range clamping for a sub-16-bit signed-kernel filter. A
/// `CatmullRom` / `Lanczos3` negative lobe overshoots a near-max edge, so
/// a finalized binned sample can exceed the 10-bit native max (1023) even
/// though the `FilterStream` only clamps to the full u16 range. For
/// `Gbrp10` that overshoot must be clipped to 1023 before any output is
/// derived: the native `rgb_u16` / `rgba_u16` must stay `<= 1023`, and the
/// u8 narrowing of a clipped-high edge must be `255` — not the wrapped
/// small value `(overshoot >> 2) as u8` the un-clamped binned row produced
/// (e.g. `1062 >> 2 = 265`, which casts to `9`). The `rgb_u16`-only
/// `Rgb48`-parity test in `filter_parity` below cannot catch this: both
/// sources overshoot identically in u16, so only the narrowed u8 (and the
/// native-range contract) diverge. Feature-independent — no `Rgb48`
/// oracle — so it also guards the `gbr`-solo build.
mod filter_native_range {
  use super::*;
  use crate::resample::{CatmullRom, FilteredResampler, Lanczos3};

  const BITS: u32 = 10;
  const NATIVE_MAX: u16 = (1 << BITS) - 1; // 1023

  /// A sharp 0 -> native-max horizontal step (the prompt's `[0, 0, 1023,
  /// 1023]` per row), uniform vertically, all three channels equal. A
  /// signed kernel enlarging this overshoots above `NATIVE_MAX` on the
  /// high side of the step.
  fn step_edge_packed_rgb(w: usize, h: usize) -> Vec<u16> {
    let mut buf = std::vec![0u16; w * h * 3];
    for (i, px) in buf.chunks_exact_mut(3).enumerate() {
      let x = i % w;
      let v = if x >= w / 2 { NATIVE_MAX } else { 0 };
      px.copy_from_slice(&[v, v, v]);
    }
    buf
  }

  /// Run the `Gbrp10` filter sink over a host-native packed-RGB source,
  /// attaching `rgb_u16`, `rgba_u16` **and** the narrowed u8 `rgb` at once,
  /// and return all three outputs.
  fn gbrp10_filter_outputs<K>(
    rgb: &[u16],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> (Vec<u16>, Vec<u16>, Vec<u8>)
  where
    K: crate::resample::FilterKernel,
  {
    let (g, b, r) = planes_from_packed_rgb(rgb, sw * sh);
    let src = crate::frame::GbrpHighBitFrame::<BITS>::try_new(
      &g, &b, &r, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    )
    .unwrap();
    let mut rgb_u16_o = std::vec![0u16; ow * oh * 3];
    let mut rgba_u16_o = std::vec![0u16; ow * oh * 4];
    let mut rgb_u8_o = std::vec![0u8; ow * oh * 3];
    {
      let mut sink = MixedSinker::<crate::source::Gbrp10, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_o)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16_o)
      .unwrap()
      .with_rgb(&mut rgb_u8_o)
      .unwrap();
      crate::source::gbrp10_to(&src, true, MATRIX, &mut sink).unwrap();
    }
    (rgb_u16_o, rgba_u16_o, rgb_u8_o)
  }

  fn assert_clamped_and_narrowed<K>(name: &str, kernel: K)
  where
    K: crate::resample::FilterKernel + Copy,
  {
    // 4 -> 7 enlargement of a 0 -> 1023 step (the prompt's 4->7 case).
    const SW: usize = 4;
    const SD: usize = 7;
    let rgb = step_edge_packed_rgb(SW, SW);
    let (rgb_u16, rgba_u16, rgb_u8) = gbrp10_filter_outputs(&rgb, SW, SW, SD, SD, kernel);

    // (a) Every native-depth sample is within the 10-bit native range.
    assert!(
      rgb_u16.iter().all(|&v| v <= NATIVE_MAX),
      "{name}: rgb_u16 must stay <= {NATIVE_MAX}; max was {}",
      rgb_u16.iter().copied().max().unwrap()
    );
    for px in rgba_u16.chunks_exact(4) {
      assert!(
        px[..3].iter().all(|&v| v <= NATIVE_MAX),
        "{name}: rgba_u16 color must stay <= {NATIVE_MAX}; px = {px:?}"
      );
      assert_eq!(px[3], NATIVE_MAX, "{name}: opaque alpha is the native max");
    }

    // The step overshoots, so a clipped-high edge (a sample pinned to the
    // native ceiling) must exist — otherwise the test is not exercising the
    // overshoot it claims to.
    assert!(
      rgb_u16.contains(&NATIVE_MAX),
      "{name}: expected a clipped-high (== {NATIVE_MAX}) edge in rgb_u16"
    );

    // (b) The u8 narrowing of a clipped-high edge is the correctly-clipped
    // 255 — never the wrapped small value an un-clamped overshoot produces
    // (`1062 >> 2 = 265 as u8 = 9`). Each u8 is the `>> 2` of the *clamped*
    // native sample, so at a ceiling pixel it is `1023 >> 2 = 255`.
    for (&hi, &lo) in rgb_u16.iter().zip(rgb_u8.iter()) {
      assert_eq!(
        lo,
        (hi >> (BITS - 8)) as u8,
        "{name}: u8 must be the narrowing of the clamped native sample"
      );
      if hi == NATIVE_MAX {
        assert_eq!(
          lo, 255,
          "{name}: clipped-high edge must narrow to 255, not a wrap"
        );
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn catmullrom_gbrp10_overshoot_is_clamped_to_native_max() {
    assert_clamped_and_narrowed("catmullrom", CatmullRom);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn lanczos3_gbrp10_overshoot_is_clamped_to_native_max() {
    assert_clamped_and_narrowed("lanczos3", Lanczos3);
  }
}

/// `Rgb48`-parity oracle: a `GbrpN` filter `rgb_u16` is byte-identical to
/// the `Rgb48` filter `rgb_u16` of the same logical pixels. Gated on `rgb`
/// (the oracle source); the `gbr`-solo build relies on
/// `gbrp_filter_plan_is_accepted` above for filter coverage.
#[cfg(feature = "rgb")]
mod filter_parity {
  use super::*;
  use crate::resample::{CatmullRom, FilteredResampler, Lanczos3, Triangle};

  /// Re-encode a host-native u16 slice as LE-wire byte storage so an
  /// `Rgb48` fixture reads back identically on LE (no-op) and BE
  /// (byte-swap) hosts.
  fn as_le_wire(host: &[u16]) -> Vec<u16> {
    host
      .iter()
      .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
      .collect()
  }

  /// Run the `Rgb48` filter sink over `rgb` (host-native) at `out_w` x
  /// `out_h` under `kernel`, returning the native `rgb_u16` output.
  fn rgb48_filter_rgb_u16<K>(
    rgb: &[u16],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> Vec<u16>
  where
    K: crate::resample::FilterKernel,
  {
    let wire = as_le_wire(rgb);
    let src = crate::frame::Rgb48Frame::new(&wire, sw as u32, sh as u32, (sw * 3) as u32);
    let mut out = std::vec![0u16; ow * oh * 3];
    {
      let mut sink = MixedSinker::<crate::source::Rgb48, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_rgb_u16(&mut out)
      .unwrap();
      crate::source::rgb48_to(&src, true, MATRIX, &mut sink).unwrap();
    }
    out
  }

  macro_rules! gbrp_filter_parity_tests {
    ($mod:ident, $marker:ident, $walker:ident, $bits:literal) => {
      mod $mod {
        use super::*;

        /// Run the `GbrpN` filter sink over `rgb` (host-native, scattered to
        /// G/B/R planes) at `ow` x `oh` under `kernel`, returning the native
        /// `rgb_u16` output — the value compared against the `Rgb48` oracle.
        fn gbrp_filter_rgb_u16<K>(
          rgb: &[u16],
          sw: usize,
          sh: usize,
          ow: usize,
          oh: usize,
          kernel: K,
        ) -> Vec<u16>
        where
          K: crate::resample::FilterKernel,
        {
          let (g, b, r) = planes_from_packed_rgb(rgb, sw * sh);
          let src = crate::frame::GbrpHighBitFrame::<$bits>::try_new(
            &g, &b, &r, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
          )
          .unwrap();
          let mut out = std::vec![0u16; ow * oh * 3];
          {
            let mut sink = MixedSinker::<
              crate::source::$marker,
              FilteredResampler<K>,
            >::with_resampler(sw, sh, FilteredResampler::new(ow, oh, kernel))
            .unwrap()
            .with_rgb_u16(&mut out)
            .unwrap();
            crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
          }
          out
        }

        #[test]
        #[cfg_attr(
          miri,
          ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
        )]
        fn downscale_filter_rgb_u16_matches_rgb48() {
          // 8x8 -> 4x4 (OUT). Values masked to BITS, so each is a legal
          // native code for both the GBR planes and the Rgb48 wire. The
          // Rgb48 oracle is 16-bit, so a signed-kernel overshoot survives
          // unclamped there; the sub-16-bit GBR output clamps it to the
          // native max, so the reference is clamped to `(1 << BITS) - 1`
          // before comparison (a no-op for the 16-bit `Gbrp16` instance,
          // which stays byte-identical to `Rgb48`).
          let rgb = rgb_ramp::<$bits>();
          for (name, gbr, reference) in [
            (
              "triangle",
              gbrp_filter_rgb_u16(&rgb, SRC, SRC, OUT, OUT, Triangle),
              rgb48_filter_rgb_u16(&rgb, SRC, SRC, OUT, OUT, Triangle),
            ),
            (
              "catmullrom",
              gbrp_filter_rgb_u16(&rgb, SRC, SRC, OUT, OUT, CatmullRom),
              rgb48_filter_rgb_u16(&rgb, SRC, SRC, OUT, OUT, CatmullRom),
            ),
            (
              "lanczos3",
              gbrp_filter_rgb_u16(&rgb, SRC, SRC, OUT, OUT, Lanczos3),
              rgb48_filter_rgb_u16(&rgb, SRC, SRC, OUT, OUT, Lanczos3),
            ),
          ] {
            let native_max = ((1u32 << $bits) - 1) as u16;
            let reference: Vec<u16> = reference.iter().map(|&v| v.min(native_max)).collect();
            assert_eq!(gbr, reference, "downscale {name}: GbrpN filter == clamped Rgb48 filter");
          }
        }

        #[test]
        #[cfg_attr(
          miri,
          ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
        )]
        fn upscale_filter_rgb_u16_matches_rgb48() {
          // 4x4 -> 7x7: a non-integer enlargement exercising the native-
          // support (no-widen) filter branch on both axes. As in the
          // downscale case the 16-bit Rgb48 oracle is clamped to the GBR
          // native max before comparison: a `CatmullRom` / `Lanczos3`
          // overshoot is unclamped at 16-bit but clipped at sub-16-bit
          // native depth (`Gbrp16` stays an exact match).
          const UPS: usize = 4;
          const UPD: usize = 7;
          let mut rgb = std::vec![0u16; UPS * UPS * 3];
          for (i, px) in rgb.chunks_exact_mut(3).enumerate() {
            px.copy_from_slice(&rgb_px::<$bits>(i));
          }
          for (name, gbr, reference) in [
            (
              "triangle",
              gbrp_filter_rgb_u16(&rgb, UPS, UPS, UPD, UPD, Triangle),
              rgb48_filter_rgb_u16(&rgb, UPS, UPS, UPD, UPD, Triangle),
            ),
            (
              "catmullrom",
              gbrp_filter_rgb_u16(&rgb, UPS, UPS, UPD, UPD, CatmullRom),
              rgb48_filter_rgb_u16(&rgb, UPS, UPS, UPD, UPD, CatmullRom),
            ),
            (
              "lanczos3",
              gbrp_filter_rgb_u16(&rgb, UPS, UPS, UPD, UPD, Lanczos3),
              rgb48_filter_rgb_u16(&rgb, UPS, UPS, UPD, UPD, Lanczos3),
            ),
          ] {
            let native_max = ((1u32 << $bits) - 1) as u16;
            let reference: Vec<u16> = reference.iter().map(|&v| v.min(native_max)).collect();
            assert_eq!(gbr, reference, "upscale {name}: GbrpN filter == clamped Rgb48 filter");
          }
        }
      }
    };
  }

  // Per the task: a low-bit (Gbrp10) and the full-depth (Gbrp16) format.
  gbrp_filter_parity_tests!(gbrp10, Gbrp10, gbrp10_to, 10);
  gbrp_filter_parity_tests!(gbrp16, Gbrp16, gbrp16_to, 16);
}
