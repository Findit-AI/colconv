//! Fused-resize coverage for the high-bit `GrayN` sources (`Gray9` /
//! `Gray10` / `Gray12` / `Gray14`, low-bit-packed `u16`) — routed through a
//! single 1-channel u16 stream that resamples the source luma plane at u16
//! precision (`GrayN` *is* a u16 luma plane). The wire row converts to a
//! host-native u16 luma plane first (the same kernel the direct `luma_u16`
//! path uses, masking to the low `BITS` bits), then every attached output
//! derives from each finalized resampled u16 luma row exactly as the direct
//! path does: `luma_u16` masks/passes through, `luma` is `>> (BITS - 8)`,
//! `rgb` / `rgba` broadcast the narrowed byte (α = 0xFF), `rgb_u16` /
//! `rgba_u16` broadcast the native u16 (α = `(1 << BITS) - 1`), and `hsv`
//! is `H=0 / S=0 / V=Y8`. The `Area` span bins; the `Filter` span runs the
//! signed-coefficient `FilterStream<u16>`.
//!
//! ★ Native-depth clamp: `GrayN` is sub-16-bit, so a signed filter kernel
//! (`CatmullRom` / `Lanczos3`) can overshoot above the native max
//! `(1 << BITS) - 1` even though the `FilterStream` only clamps to the full
//! u16 range. The `gray_n_to_*` derive kernels finish with `raw & mask`,
//! which *wraps* an over-range sample instead of clipping it. So the
//! resample emit clamps every resampled sample to the native max before any
//! derive — and the oracle here is the single-channel `FilterStream<u16>`
//! resample **clamped to the native max**, a discriminating reference (NOT
//! the unclamped engine output). The `*_overshoot_clamped_not_wrapped`
//! tests construct an input whose filter genuinely overshoots and assert
//! the output saturates to the native max rather than wrapping to a small
//! value — they FAIL without the clamp.

use crate::{
  KernelMatrix, PixelSink,
  frame::GrayNFrame,
  resample::{
    AreaResampler, CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3,
    ResampleError, Resampler, Triangle,
  },
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Gray9, Gray10, Gray12, Gray12Row, Gray14, gray9_to, gray9_to_endian, gray10_to,
    gray10_to_endian, gray12_to, gray12_to_endian, gray14_to, gray14_to_endian,
  },
};

const SRC: usize = 8;
const OUT: usize = 4;
// Gray is luma-only; the walker still threads a matrix / range through.
const FR: bool = true;
const M: KernelMatrix = KernelMatrix::Bt709;

/// Re-encode a host-native u16 slice as LE-encoded byte storage (the
/// `grayNle` plane contract). The loader recovers the logical values via
/// `u16::from_le` — a no-op on LE hosts, a byte-swap on BE.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as BE-encoded byte storage (the
/// `grayNbe` plane contract), recovered via `u16::from_be`.
fn as_be_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// `(1 << BITS) - 1` — the native max a `BITS`-deep `GrayN` sample carries.
const fn native_max(bits: u32) -> u16 {
  ((1u32 << bits) - 1) as u16
}

/// An interior `BITS`-deep luma ramp (masked to the native range) so the
/// area mean / filter windows see real variation per block, exercising the
/// low byte the u16 stream must preserve.
fn ramp(bits: u32) -> Vec<u16> {
  let max = native_max(bits) as u32 + 1;
  let mut y = vec![0u16; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = ((i * 1031) as u32 % max) as u16;
  }
  y
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid `u16` plane to
/// the `OUT` grid — the integer-ratio (2:1) area-downscale reference at u16
/// precision (sum of four native-range u16 fits in u32; in-range so the
/// native clamp is a value no-op for the area path).
fn block_mean_2x2(plane: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u16;
    }
  }
  out
}

/// Per-`BITS` test body for the area path. `$marker` is the source marker,
/// `$walker` the LE loader, `$bits` the depth.
macro_rules! gray_n_area_tests {
  ($mod:ident, $marker:ident, $walker:ident, $bits:expr) => {
    mod $mod {
      use super::*;

      const BITS: u32 = $bits;

      fn frame(pix: &[u16], w: usize, h: usize) -> GrayNFrame<'_, BITS> {
        GrayNFrame::new(pix, w as u32, h as u32, w as u32)
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn area_luma_u16_is_exact_block_mean() {
        let plane = ramp(BITS);
        let pix = as_le_u16(&plane);
        let mut luma_u16 = vec![0u16; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walker(&frame(&pix, SRC, SRC), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(
          luma_u16,
          block_mean_2x2(&plane),
          "luma_u16 must be the exact 2x2 block mean of the native-depth luma plane"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn area_all_outputs_match_direct_over_binned_luma() {
        let plane = ramp(BITS);
        let pix = as_le_u16(&plane);

        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
        let mut luma = vec![0u8; OUT * OUT];
        let mut luma_u16 = vec![0u16; OUT * OUT];
        let mut h = vec![0u8; OUT * OUT];
        let mut s_ = vec![0u8; OUT * OUT];
        let mut v_ = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap()
          .with_hsv(&mut h, &mut s_, &mut v_)
          .unwrap();
          $walker(&frame(&pix, SRC, SRC), FR, sink.set_kernel_matrix(M)).unwrap();
        }

        // Reference: the direct sink over the exact binned luma plane (an
        // OUT-grid GrayN frame holding the 2x2 area mean).
        let binned = block_mean_2x2(&plane);
        let binned_pix = as_le_u16(&binned);
        let mut ref_rgb = vec![0u8; OUT * OUT * 3];
        let mut ref_rgba = vec![0u8; OUT * OUT * 4];
        let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
        let mut ref_luma = vec![0u8; OUT * OUT];
        let mut ref_luma_u16 = vec![0u16; OUT * OUT];
        let mut ref_h = vec![0u8; OUT * OUT];
        let mut ref_s = vec![0u8; OUT * OUT];
        let mut ref_v = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker>::new(OUT, OUT)
            .with_rgb(&mut ref_rgb)
            .unwrap()
            .with_rgba(&mut ref_rgba)
            .unwrap()
            .with_rgb_u16(&mut ref_rgb_u16)
            .unwrap()
            .with_rgba_u16(&mut ref_rgba_u16)
            .unwrap()
            .with_luma(&mut ref_luma)
            .unwrap()
            .with_luma_u16(&mut ref_luma_u16)
            .unwrap()
            .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
            .unwrap();
          $walker(&frame(&binned_pix, OUT, OUT), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(luma, ref_luma, "luma");
        assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
        assert_eq!(rgb, ref_rgb, "rgb");
        assert_eq!(rgba, ref_rgba, "rgba");
        assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
        assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
        assert_eq!(h, ref_h, "h");
        assert_eq!(s_, ref_s, "s");
        assert_eq!(v_, ref_v, "v");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn area_identity_plan_matches_new_sink() {
        let plane = ramp(BITS);
        let pix = as_le_u16(&plane);

        let mut direct = vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb_u16(&mut direct)
            .unwrap();
          $walker(&frame(&pix, SRC, SRC), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let mut via_area = vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgb_u16(&mut via_area)
          .unwrap();
          $walker(&frame(&pix, SRC, SRC), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(direct, via_area, "identity plan must match the direct sink");
      }

      #[test]
      fn area_no_outputs_is_a_no_op() {
        let plane = ramp(BITS);
        let pix = as_le_u16(&plane);
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap();
        // No outputs: a legal no-op, accepted without allocation.
        $walker(&frame(&pix, SRC, SRC), FR, sink.set_kernel_matrix(M)).unwrap();
        assert!(
          !sink.luma_stream_u16_allocated(),
          "no-output sink allocated a u16 luma stream"
        );
      }
    }
  };
}

gray_n_area_tests!(gray9, Gray9, gray9_to, 9);
gray_n_area_tests!(gray10, Gray10, gray10_to, 10);
gray_n_area_tests!(gray12, Gray12, gray12_to, 12);
gray_n_area_tests!(gray14, Gray14, gray14_to, 14);

// ---- Out-of-sequence / atomicity (shared shape, one representative depth)----
//
// The freeze / sequence / staging ordering is depth-independent (the const
// `BITS` only changes the per-sample math), so one representative depth
// (Gray12) exercises the atomicity contract.

#[test]
fn area_out_of_sequence_first_row_rejected_before_allocation() {
  let plane = ramp(12);
  let pix = as_le_u16(&plane);
  let row3 = &pix[3 * SRC..4 * SRC];

  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Gray12Row::for_tests(row3, 3, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  assert!(
    !sink.luma_stream_u16_allocated(),
    "stream allocated for a rejected row"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0),
    "rejected row mutated output"
  );
}

#[test]
fn area_rejected_first_row_does_not_poison_output_retry() {
  let plane = ramp(12);
  let pix = as_le_u16(&plane);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Gray12Row::for_tests(&pix[3 * SRC..4 * SRC], 3, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  sink.set_rgb_u16(&mut rgb_u16).unwrap();
  sink
    .process(Gray12Row::for_tests(&pix[..SRC], 0, M, FR))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

#[test]
fn area_rejects_mid_frame_output_change() {
  let plane = ramp(12);
  let pix = as_le_u16(&plane);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Gray12Row::for_tests(&pix[..SRC], 0, M, FR))
    .unwrap();
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(Gray12Row::for_tests(&pix[SRC..2 * SRC], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0),
    "rejected row mutated the new output"
  );
}

#[test]
fn area_reuses_luma_stream_across_frames() {
  // A reused sink must reset the u16 luma stream each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence.
  let y1 = ramp(12);
  let mut y2 = y1.clone();
  let max = native_max(12);
  for p in y2.iter_mut() {
    *p = max - *p;
  }
  let pix1 = as_le_u16(&y1);
  let pix2 = as_le_u16(&y2);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    let f1 = GrayNFrame::<12>::new(&pix1, SRC as u32, SRC as u32, SRC as u32);
    let f2 = GrayNFrame::<12>::new(&pix2, SRC as u32, SRC as u32, SRC as u32);
    gray12_to(&f1, FR, sink.set_kernel_matrix(M)).unwrap();
    gray12_to(&f2, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(
    luma_u16,
    block_mean_2x2(&y2),
    "frame 2 luma_u16 must area-downscale frame 2's luma"
  );
}

// ---- Filter-plan routing ----------------------------------------------------
//
// GrayN *is* a u16 luma plane: the filter path converts the wire row to a
// host-native u16 luma plane and resamples it through the signed-coefficient
// single-channel `FilterStream<u16>`. Because GrayN is sub-16-bit, the
// resampled luma is clamped to the native max `(1 << BITS) - 1` before any
// derive (the engine's `0..=65535` clamp is NOT the native clamp), so the
// oracle is the single-channel `FilterStream<u16>` resample CLAMPED to the
// native max — value for value.

/// A larger, native-range-wide grid than the 2:1 area fixture so a downscale
/// (8->4) and an upscale (4->7) both run real, non-trivial windows. The hard
/// mid-column edge makes filter windows straddling it produce real
/// intermediate values (and a positive overshoot near the high edge).
const FW: usize = 8;
const FH: usize = 8;
const FOUT_DOWN: usize = 4;
const FUP: usize = 7;

/// A host-native u16 Y ramp (masked to `BITS`) with a hard mid-column edge
/// from low to near-max per row plus two textured interior rows.
fn filter_ramp(bits: u32) -> Vec<u16> {
  let max = native_max(bits);
  let lo = max / 10;
  let hi = max - max / 16;
  let mut y = vec![0u16; FW * FH];
  for row in 0..FH {
    for col in 0..FW {
      y[row * FW + col] = if col < FW / 2 { lo } else { hi };
    }
  }
  for col in 0..FW {
    y[4 * FW + col] = ((col as u32 * max as u32) / FW as u32) as u16;
    y[5 * FW + col] = (max as u32 - (col as u32 * max as u32) / FW as u32) as u16;
  }
  y
}

/// Single-channel filter resample of a host-native u16 luma plane via the
/// merged engine's [`FilterStream<u16>`] (channels = 1) — the raw engine
/// output (clamped to `0..=65535` only).
fn native_luma_filter<K: FilterKernel>(
  kernel: K,
  luma_plane: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u16> {
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u16>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0u16; ow * oh];
  for row in 0..sh {
    stream
      .feed_row(
        row,
        &luma_plane[row * sw..(row + 1) * sw],
        true,
        |oy, fin| {
          out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
        },
      )
      .expect("rows in order");
  }
  out
}

/// The DISCRIMINATING GrayN luma oracle: the single-channel `FilterStream`
/// output CLAMPED to the native max `(1 << BITS) - 1`. Distinct from the raw
/// engine output wherever a signed kernel overshoots above the native max.
fn native_luma_filter_clamped<K: FilterKernel>(
  kernel: K,
  bits: u32,
  luma_plane: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u16> {
  let max = native_max(bits);
  native_luma_filter(kernel, luma_plane, sw, sh, ow, oh)
    .into_iter()
    .map(|v| v.min(max))
    .collect()
}

/// Per-`BITS` filter-path test body.
macro_rules! gray_n_filter_tests {
  ($mod:ident, $marker:ident, $walker:ident, $bits:expr) => {
    mod $mod {
      use super::*;

      const BITS: u32 = $bits;

      fn frame(pix: &[u16], w: usize, h: usize) -> GrayNFrame<'_, BITS> {
        GrayNFrame::new(pix, w as u32, h as u32, w as u32)
      }

      /// Run a filter sink at `ow x oh` under `kernel`, attaching every
      /// output the equivalence asserts on, and return them.
      #[allow(clippy::type_complexity)]
      fn outputs<K: FilterKernel + Copy>(
        ow: usize,
        oh: usize,
        kernel: K,
      ) -> (
        Vec<u8>,
        Vec<u16>,
        Vec<u8>,
        Vec<u8>,
        Vec<u16>,
        Vec<u16>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
      ) {
        let plane = filter_ramp(BITS);
        let pix = as_le_u16(&plane);
        let mut luma = vec![0u8; ow * oh];
        let mut luma_u16 = vec![0u16; ow * oh];
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgba = vec![0u8; ow * oh * 4];
        let mut rgb_u16 = vec![0u16; ow * oh * 3];
        let mut rgba_u16 = vec![0u16; ow * oh * 4];
        let mut hp = vec![0u8; ow * oh];
        let mut sp = vec![0u8; ow * oh];
        let mut vp = vec![0u8; ow * oh];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<K>>::with_resampler(
            FW,
            FH,
            FilteredResampler::new(ow, oh, kernel),
          )
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_hsv(&mut hp, &mut sp, &mut vp)
          .unwrap();
          $walker(&frame(&pix, FW, FH), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        (luma, luma_u16, rgb, rgba, rgb_u16, rgba_u16, hp, sp, vp)
      }

      /// Assert a filter resample's every output derives from the clamping
      /// oracle exactly as the area emit derives from its binned luma;
      /// returns the max per-sample `luma_u16` diff (exactly 0).
      fn assert_matches_oracle<K: FilterKernel + Copy>(
        kernel: K,
        ow: usize,
        oh: usize,
        ctx: &str,
      ) -> u16 {
        let plane = filter_ramp(BITS);
        let (luma, luma_u16, rgb, rgba, rgb_u16, rgba_u16, hp, sp, vp) = outputs(ow, oh, kernel);
        let y_ref = native_luma_filter_clamped(kernel, BITS, &plane, FW, FH, ow, oh);

        let mut max_diff = 0u16;
        for (i, (&g, &w)) in luma_u16.iter().zip(y_ref.iter()).enumerate() {
          max_diff = max_diff.max(g.abs_diff(w));
          assert_eq!(
            g, w,
            "{ctx} luma_u16[{i}]: {g} vs clamped single-channel native-luma filter {w}"
          );
        }
        // Every derived output mirrors the area emit applied to the clamped
        // resampled luma: the reference is the direct GrayN sink run over
        // the clamped-resampled-Y frame (re-encoded LE).
        let ref_pix = as_le_u16(&y_ref);
        let mut ref_luma = vec![0u8; ow * oh];
        let mut ref_luma_u16 = vec![0u16; ow * oh];
        let mut ref_rgb = vec![0u8; ow * oh * 3];
        let mut ref_rgba = vec![0u8; ow * oh * 4];
        let mut ref_rgb_u16 = vec![0u16; ow * oh * 3];
        let mut ref_rgba_u16 = vec![0u16; ow * oh * 4];
        let mut ref_h = vec![0u8; ow * oh];
        let mut ref_s = vec![0u8; ow * oh];
        let mut ref_v = vec![0u8; ow * oh];
        {
          let mut sink = MixedSinker::<$marker>::new(ow, oh)
            .with_luma(&mut ref_luma)
            .unwrap()
            .with_luma_u16(&mut ref_luma_u16)
            .unwrap()
            .with_rgb(&mut ref_rgb)
            .unwrap()
            .with_rgba(&mut ref_rgba)
            .unwrap()
            .with_rgb_u16(&mut ref_rgb_u16)
            .unwrap()
            .with_rgba_u16(&mut ref_rgba_u16)
            .unwrap()
            .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
            .unwrap();
          $walker(&frame(&ref_pix, ow, oh), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(luma, ref_luma, "{ctx} luma (>> BITS-8)");
        assert_eq!(rgb, ref_rgb, "{ctx} rgb");
        assert_eq!(rgba, ref_rgba, "{ctx} rgba");
        assert_eq!(rgb_u16, ref_rgb_u16, "{ctx} rgb_u16");
        assert_eq!(rgba_u16, ref_rgba_u16, "{ctx} rgba_u16");
        assert_eq!(hp, ref_h, "{ctx} hsv H");
        assert_eq!(sp, ref_s, "{ctx} hsv S");
        assert_eq!(vp, ref_v, "{ctx} hsv V");
        max_diff
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_luma_is_clamped_single_channel_native_luma() {
        for (ow, oh, tag) in [(FOUT_DOWN, FOUT_DOWN, "down"), (FUP, FUP, "up")] {
          assert_eq!(
            assert_matches_oracle(Triangle, ow, oh, &format!("triangle {tag}")),
            0,
            "triangle {tag} luma_u16 diff must be 0"
          );
          assert_eq!(
            assert_matches_oracle(CatmullRom, ow, oh, &format!("catmullrom {tag}")),
            0,
            "catmullrom {tag} luma_u16 diff must be 0"
          );
          assert_eq!(
            assert_matches_oracle(Lanczos3, ow, oh, &format!("lanczos3 {tag}")),
            0,
            "lanczos3 {tag} luma_u16 diff must be 0"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_plan_is_accepted() {
        // A FilteredResampler plan must be ACCEPTED (GrayN is routed to the
        // filter path) — not UnsupportedFilter — and must write the output.
        let plane = filter_ramp(BITS);
        let pix = as_le_u16(&plane);
        let mut luma_u16 = vec![0xA5A5u16; FOUT_DOWN * FOUT_DOWN];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<Triangle>>::with_resampler(
            FW,
            FH,
            FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
          )
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walker(&frame(&pix, FW, FH), FR, sink.set_kernel_matrix(M))
            .expect("filter plan must be accepted");
        }
        let y_ref =
          native_luma_filter_clamped(Triangle, BITS, &plane, FW, FH, FOUT_DOWN, FOUT_DOWN);
        assert_eq!(
          luma_u16, y_ref,
          "accepted filter luma_u16 = clamped single-channel oracle"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_overshoot_clamped_not_wrapped() {
        // ★ The discriminating native-clamp test. A hard 0 → native-max edge
        // makes a signed kernel (CatmullRom / Lanczos3) overshoot ABOVE the
        // native max near the high side. The `gray_n_to_luma_u16_row` derive
        // finishes with `& mask`, which would WRAP such an over-range sample
        // to a small value; the resample emit clamps it to the native max
        // first. So:
        //   * every output sample must be <= native_max (no value escapes
        //     the documented range), and
        //   * at the overshoot the output must SATURATE to native_max — which
        //     also proves the engine genuinely overshot there (the raw,
        //     unclamped engine value exceeds the native max), so this is a
        //     real discriminator, not a vacuous bound. Without the clamp the
        //     output would wrap to a small value != native_max and the
        //     saturation assert fails.
        let max = native_max(BITS);
        // A sharp horizontal step: a block of 0 then a block of native-max.
        let mut plane = vec![0u16; FW * FH];
        for row in 0..FH {
          for col in 0..FW {
            plane[row * FW + col] = if col >= FW / 2 { max } else { 0 };
          }
        }
        let pix = as_le_u16(&plane);

        for kernel_name in ["catmullrom", "lanczos3"] {
          let (luma_u16, raw) = match kernel_name {
            "catmullrom" => {
              let mut out = vec![0u16; FUP * FUP];
              {
                let mut sink =
                  MixedSinker::<$marker, FilteredResampler<CatmullRom>>::with_resampler(
                    FW,
                    FH,
                    FilteredResampler::new(FUP, FUP, CatmullRom),
                  )
                  .unwrap()
                  .with_luma_u16(&mut out)
                  .unwrap();
                $walker(&frame(&pix, FW, FH), FR, sink.set_kernel_matrix(M)).unwrap();
              }
              (
                out,
                native_luma_filter(CatmullRom, &plane, FW, FH, FUP, FUP),
              )
            }
            _ => {
              let mut out = vec![0u16; FUP * FUP];
              {
                let mut sink = MixedSinker::<$marker, FilteredResampler<Lanczos3>>::with_resampler(
                  FW,
                  FH,
                  FilteredResampler::new(FUP, FUP, Lanczos3),
                )
                .unwrap()
                .with_luma_u16(&mut out)
                .unwrap();
                $walker(&frame(&pix, FW, FH), FR, sink.set_kernel_matrix(M)).unwrap();
              }
              (out, native_luma_filter(Lanczos3, &plane, FW, FH, FUP, FUP))
            }
          };

          // The fixture must genuinely overshoot, else the test is vacuous.
          let overshoot_idx = raw
            .iter()
            .position(|&v| v > max)
            .unwrap_or_else(|| panic!("{kernel_name}: fixture did not overshoot native max"));

          // No sample escapes the native range.
          for (i, &v) in luma_u16.iter().enumerate() {
            assert!(
              v <= max,
              "{kernel_name}: luma_u16[{i}] = {v} exceeds native max {max}"
            );
          }
          // At the overshoot the clamp saturates to native_max; the wrapped
          // (`raw & mask`) value would be a small number != native_max.
          let wrapped = raw[overshoot_idx] & max;
          assert_ne!(
            wrapped, max,
            "{kernel_name}: fixture overshoot does not actually wrap to a distinct value"
          );
          assert_eq!(
            luma_u16[overshoot_idx], max,
            "{kernel_name}: overshoot must saturate to native max {max}, not wrap to {wrapped}"
          );
        }
      }
    }
  };
}

gray_n_filter_tests!(filter_gray9, Gray9, gray9_to, 9);
gray_n_filter_tests!(filter_gray10, Gray10, gray10_to, 10);
gray_n_filter_tests!(filter_gray12, Gray12, gray12_to, 12);
gray_n_filter_tests!(filter_gray14, Gray14, gray14_to, 14);

// ---- LE/BE parity (area + filter) -------------------------------------------
//
// The resampled u16 luma is host-native, so the derive kernels must run with
// `HOST_NATIVE_BE`, not `<false>`. On an LE dev/CI host `<false>` masks the
// bug; the LE-vs-BE parity check catches a wrong const on either host (LE and
// BE wire encodings of the same logical plane must resample identically).

/// Per-`BITS` LE/BE parity for both the area and the filter path. `$walker`
/// is the LE loader, `$walker_be` the const-generic BE entry point.
macro_rules! gray_n_le_be_tests {
  ($mod:ident, $marker:ident, $walker:ident, $walker_be:ident, $bits:expr) => {
    mod $mod {
      use super::*;

      const BITS: u32 = $bits;

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn area_le_be_outputs_identical() {
        let plane = ramp(BITS);
        let pix_le = as_le_u16(&plane);
        let pix_be = as_be_u16(&plane);

        let mut le_luma_u16 = vec![0u16; OUT * OUT];
        let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut le_rgba = vec![0u8; OUT * OUT * 4];
        {
          let frame = GrayNFrame::<BITS>::new(&pix_le, SRC as u32, SRC as u32, SRC as u32);
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_luma_u16(&mut le_luma_u16)
          .unwrap()
          .with_rgb_u16(&mut le_rgb_u16)
          .unwrap()
          .with_rgba(&mut le_rgba)
          .unwrap();
          $walker(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let mut be_luma_u16 = vec![0u16; OUT * OUT];
        let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut be_rgba = vec![0u8; OUT * OUT * 4];
        {
          let frame = GrayNFrame::<BITS, true>::new(&pix_be, SRC as u32, SRC as u32, SRC as u32);
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_luma_u16(&mut be_luma_u16)
          .unwrap()
          .with_rgb_u16(&mut be_rgb_u16)
          .unwrap()
          .with_rgba(&mut be_rgba)
          .unwrap();
          $walker_be::<_, true>(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(le_luma_u16, be_luma_u16, "area luma_u16 LE/BE diverge");
        assert_eq!(le_rgb_u16, be_rgb_u16, "area rgb_u16 LE/BE diverge");
        assert_eq!(le_rgba, be_rgba, "area rgba LE/BE diverge");
        assert_eq!(
          le_luma_u16,
          block_mean_2x2(&plane),
          "area luma_u16 not block mean"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_le_be_outputs_identical() {
        let plane = filter_ramp(BITS);
        let pix_le = as_le_u16(&plane);
        let pix_be = as_be_u16(&plane);

        let mut le_luma_u16 = vec![0u16; FOUT_DOWN * FOUT_DOWN];
        let mut le_rgb_u16 = vec![0u16; FOUT_DOWN * FOUT_DOWN * 3];
        {
          let frame = GrayNFrame::<BITS>::new(&pix_le, FW as u32, FH as u32, FW as u32);
          let mut sink = MixedSinker::<$marker, FilteredResampler<Triangle>>::with_resampler(
            FW,
            FH,
            FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
          )
          .unwrap()
          .with_luma_u16(&mut le_luma_u16)
          .unwrap()
          .with_rgb_u16(&mut le_rgb_u16)
          .unwrap();
          $walker(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        let mut be_luma_u16 = vec![0u16; FOUT_DOWN * FOUT_DOWN];
        let mut be_rgb_u16 = vec![0u16; FOUT_DOWN * FOUT_DOWN * 3];
        {
          let frame = GrayNFrame::<BITS, true>::new(&pix_be, FW as u32, FH as u32, FW as u32);
          let mut sink = MixedSinker::<$marker<true>, FilteredResampler<Triangle>>::with_resampler(
            FW,
            FH,
            FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
          )
          .unwrap()
          .with_luma_u16(&mut be_luma_u16)
          .unwrap()
          .with_rgb_u16(&mut be_rgb_u16)
          .unwrap();
          $walker_be::<_, true>(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        assert_eq!(le_luma_u16, be_luma_u16, "filter luma_u16 LE/BE diverge");
        assert_eq!(le_rgb_u16, be_rgb_u16, "filter rgb_u16 LE/BE diverge");
      }
    }
  };
}

gray_n_le_be_tests!(parity_gray9, Gray9, gray9_to, gray9_to_endian, 9);
gray_n_le_be_tests!(parity_gray10, Gray10, gray10_to, gray10_to_endian, 10);
gray_n_le_be_tests!(parity_gray12, Gray12, gray12_to, gray12_to_endian, 12);
gray_n_le_be_tests!(parity_gray14, Gray14, gray14_to, gray14_to_endian, 14);
