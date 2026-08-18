//! Fused-downscale coverage for `Grayf32` — routed through a single
//! 1-channel `AreaStream<f32>` that bins the source `f32` luma plane at
//! f32 precision (Grayf32 *is* an f32 luma plane). The wire row converts
//! to a host-native f32 luma plane first (the same kernel the direct
//! `luma_f32` path uses), then every attached output derives from each
//! finalized binned f32 luma row exactly as the direct path does:
//! `luma_f32` is a host-native pass-through, `rgb_f32` replicates Y to
//! R=G=B losslessly, `luma` / `luma_u16` clamp-and-scale to u8 / u16,
//! `rgb` / `rgba` broadcast the clamped u8 (α = 0xFF), `rgb_u16` /
//! `rgba_u16` broadcast the clamped u16 (α = 0xFFFF), and `hsv` is
//! `H=0 / S=0 / V=clamp(Y)x255`. So every resampled output equals the
//! direct Grayf32 sink run over a frame that already holds the binned
//! f32 luma plane.
//!
//! The binning is at full f32 precision: a u8 / u16 luma stream would
//! quantize every sample before averaging, and an HDR (> 1.0) or
//! negative sample would be clamped before it could contribute to the
//! mean.

use crate::{
  KernelMatrix, PixelSink,
  frame::Grayf32Frame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Grayf32, Grayf32Row, grayf32_to, grayf32_to_endian},
};

const SRC: usize = 8;
const OUT: usize = 4;
// Gray is luma-only; the walker still threads a matrix / range through.
const FR: bool = true;
const M: KernelMatrix = KernelMatrix::Bt709;

/// Re-encode a host-native f32 slice as LE-encoded byte storage (the
/// `grayf32le` plane contract). The `Grayf32` loader recovers the
/// logical values via `u32::from_le` over the element bits — a no-op on
/// LE hosts, a byte-swap on BE. Mirrors `bytemuck::cast_slice` over
/// `f32::to_le_bytes` output without needing a `&[u8]` → `&[f32]` cast.
fn as_le_f32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Re-encode a host-native f32 slice as BE-encoded byte storage (the
/// `grayf32be` plane contract), recovered via `u32::from_be`.
fn as_be_f32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_be()))
    .collect()
}

/// Interior f32 luma ramp mixing in-range, HDR (> 1.0), and negative
/// values so the area mean sees real variation per 2x2 block and the
/// f32-precision binning is exercised against the integer streams'
/// pre-average clamp.
fn ramp() -> Vec<f32> {
  let mut y = vec![0.0f32; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = match i % 5 {
      0 => i as f32 / 64.0,      // in-range ramp [0, 1)
      1 => 1.0 + i as f32 / 8.0, // HDR > 1.0
      2 => -(i as f32) / 100.0,  // negative
      3 => 0.5,
      _ => 2.5,
    };
  }
  y
}

/// Exact 2x2-block area mean of an `SRC`-grid `f32` plane to the `OUT`
/// grid — the integer-ratio (2:1) area-downscale reference. The engine
/// accumulates in `f64` and finalizes `(acc / denom) as f32`; for the
/// uniform 2:1 box the per-output value is `(a+b+c+d)/4` evaluated in
/// `f64` (each term exact, the divide by a power of two exact) then cast
/// to f32 — bit-identical to the stream.
fn block_mean_2x2(plane: &[f32]) -> Vec<f32> {
  let mut out = vec![0.0f32; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0.0f64;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as f64;
        }
      }
      out[oy * OUT + ox] = (s / 4.0) as f32;
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_downscale_luma_f32_is_exact_area_mean() {
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma_f32,
    block_mean_2x2(&plane),
    "luma_f32 must be the exact 2x2 f32 block mean (no pre-average clamp)"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_all_outputs_match_direct_over_binned_luma() {
  // Every attached output — luma_f32 / rgb_f32 / luma / luma_u16 / rgb /
  // rgba / rgb_u16 / rgba_u16 / hsv — must be exactly what the direct
  // Grayf32 sink produces over the (exact) binned f32 luma plane. The
  // binned luma is the f32 area mean, so we feed that mean as a
  // full-resolution `OUT`-grid Grayf32 frame to the reference sink.
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
  }

  // Reference: the direct sink over the exact binned f32 luma plane.
  let binned = block_mean_2x2(&plane);
  let binned_pix = as_le_f32(&binned);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut ref_luma_f32 = vec![0.0f32; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned_frame = Grayf32Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Grayf32>::new(OUT, OUT)
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
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_luma_f32(&mut ref_luma_f32)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    grayf32_to(&binned_frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(luma_f32, ref_luma_f32, "luma_f32");
  assert_eq!(rgb_f32, ref_rgb_f32, "rgb_f32");
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
fn grayf32_fractional_outputs_match_direct_over_binned_luma() {
  // Non-integer (8->3) downscale with values straddling u8/u16 rounding
  // thresholds. The f32 area stream's SIMD parity is a small tolerance,
  // not 0-ULP (resample/mod.rs: f32 adds are non-associative) — the
  // documented engine contract every f32 route inherits, covered at the
  // stream level by `stream_f32_matches_direct_2d_reference_fractional`.
  // This checks the ROUTE's derivation for a fractional ratio: every
  // derived output must be exactly what the direct Grayf32 sink produces
  // over the binned `luma_f32` plane (the emit applies the identical
  // kernels per finalized row, regardless of ratio).
  const S: usize = 8;
  const O: usize = 3;
  let plane: std::vec::Vec<f32> = (0..S * S).map(|i| i as f32 / (S * S - 1) as f32).collect();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, S as u32, S as u32, S as u32);

  let mut rgb = vec![0u8; O * O * 3];
  let mut rgba = vec![0u8; O * O * 4];
  let mut rgb_u16 = vec![0u16; O * O * 3];
  let mut rgba_u16 = vec![0u16; O * O * 4];
  let mut luma = vec![0u8; O * O];
  let mut luma_u16 = vec![0u16; O * O];
  let mut rgb_f32 = vec![0.0f32; O * O * 3];
  let mut luma_f32 = vec![0.0f32; O * O];
  let mut h = vec![0u8; O * O];
  let mut s_ = vec![0u8; O * O];
  let mut v_ = vec![0u8; O * O];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(S, S, AreaResampler::to(O, O))
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
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
  }

  // Reference: the direct sink over the route's own binned f32 luma plane.
  // `luma_f32` is the engine's f32 area mean (validated at the stream
  // level); every other output must equal a direct conversion of it.
  let binned_pix = as_le_f32(&luma_f32);
  let mut ref_rgb = vec![0u8; O * O * 3];
  let mut ref_rgba = vec![0u8; O * O * 4];
  let mut ref_rgb_u16 = vec![0u16; O * O * 3];
  let mut ref_rgba_u16 = vec![0u16; O * O * 4];
  let mut ref_luma = vec![0u8; O * O];
  let mut ref_luma_u16 = vec![0u16; O * O];
  let mut ref_rgb_f32 = vec![0.0f32; O * O * 3];
  let mut ref_luma_f32 = vec![0.0f32; O * O];
  let mut ref_h = vec![0u8; O * O];
  let mut ref_s = vec![0u8; O * O];
  let mut ref_v = vec![0u8; O * O];
  {
    let binned_frame = Grayf32Frame::new(&binned_pix, O as u32, O as u32, O as u32);
    let mut sink = MixedSinker::<Grayf32>::new(O, O)
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
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_luma_f32(&mut ref_luma_f32)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    grayf32_to(&binned_frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(luma_f32, ref_luma_f32, "fractional luma_f32");
  assert_eq!(rgb_f32, ref_rgb_f32, "fractional rgb_f32");
  assert_eq!(luma, ref_luma, "fractional luma");
  assert_eq!(luma_u16, ref_luma_u16, "fractional luma_u16");
  assert_eq!(rgb, ref_rgb, "fractional rgb");
  assert_eq!(rgba, ref_rgba, "fractional rgba");
  assert_eq!(rgb_u16, ref_rgb_u16, "fractional rgb_u16");
  assert_eq!(rgba_u16, ref_rgba_u16, "fractional rgba_u16");
  assert_eq!(h, ref_h, "fractional h");
  assert_eq!(s_, ref_s, "fractional s");
  assert_eq!(v_, ref_v, "fractional v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_le_be_resample_outputs_identical() {
  // The binned f32 luma is host-native, so the derive kernels must run
  // with `HOST_NATIVE_BE`, not `<false>`. On an LE dev/CI host `<false>`
  // masks the bug; the LE-vs-BE parity check catches a wrong const on
  // either host (LE and BE wire encodings of the same logical plane must
  // resample to identical outputs). Covers a lossless float output
  // (`luma_f32`), a clamp-scale integer output (`rgb_u16`), and a u8
  // RGBA broadcast.
  let plane = ramp();
  let pix_le = as_le_f32(&plane);
  let pix_be = as_be_f32(&plane);

  let mut le_luma_f32 = vec![0.0f32; OUT * OUT];
  let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut le_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Grayf32Frame::new(&pix_le, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_f32(&mut le_luma_f32)
        .unwrap()
        .with_rgb_u16(&mut le_rgb_u16)
        .unwrap()
        .with_rgba(&mut le_rgba)
        .unwrap();
    grayf32_to(&frame, FR, M, &mut sink).unwrap();
  }

  let mut be_luma_f32 = vec![0.0f32; OUT * OUT];
  let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut be_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Grayf32Frame::<true>::new(&pix_be, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Grayf32<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_luma_f32(&mut be_luma_f32)
    .unwrap()
    .with_rgb_u16(&mut be_rgb_u16)
    .unwrap()
    .with_rgba(&mut be_rgba)
    .unwrap();
    grayf32_to_endian::<_, true>(&frame, FR, M, &mut sink).unwrap();
  }

  assert_eq!(le_luma_f32, be_luma_f32, "luma_f32 LE/BE diverge");
  assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
  assert_eq!(le_rgba, be_rgba, "rgba LE/BE diverge");
  // And the binned f32 luma is the exact area mean regardless of wire.
  assert_eq!(
    le_luma_f32,
    block_mean_2x2(&plane),
    "luma_f32 not area mean"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_standalone_rgba_matches_direct_over_binned_luma() {
  // u8 RGBA-only exercises the dedicated fast path (no RGB scratch).
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_pix = as_le_f32(&block_mean_2x2(&plane));
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  {
    let binned = Grayf32Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Grayf32>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap();
    grayf32_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "standalone rgba");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_standalone_rgba_u16_matches_direct_over_binned_luma() {
  // u16 RGBA-only exercises the native rgba_u16 fast path.
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_pix = as_le_f32(&block_mean_2x2(&plane));
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let binned = Grayf32Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Grayf32>::new(OUT, OUT)
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap();
    grayf32_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba_u16, ref_rgba_u16, "standalone rgba_u16");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_hsv_plus_rgba_matches_direct_over_binned_luma() {
  // HSV + u8 RGBA without RGB: both derive directly from the binned luma
  // (no RGB kernel, no RGB scratch — asserted below), byte-identical to
  // the direct path's output.
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
    // Regression: this case must not reserve RGB scratch — it derives
    // HSV+RGBA from luma and never reads the scratch, so reserving it
    // could spuriously AllocationFail under memory pressure.
    assert_eq!(
      sink.rgb_scratch_capacity(),
      0,
      "grayf32 HSV+RGBA must not reserve RGB scratch"
    );
  }
  let binned_pix = as_le_f32(&block_mean_2x2(&plane));
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Grayf32Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Grayf32>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    grayf32_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "hsv+rgba: rgba");
  assert_eq!(h, ref_h, "hsv+rgba: h");
  assert_eq!(s_, ref_s, "hsv+rgba: s");
  assert_eq!(v_, ref_v, "hsv+rgba: v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_rgb_f32_only_does_not_grow_luma_scratch_beyond_width() {
  // An `rgb_f32`-only sink still bins through the f32 luma stream (rgb_f32
  // is derived from the binned luma), so it stages exactly `width` f32 of
  // source luma — never the `3 * width` an RGB scratch would need.
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
    // Source-luma staging is exactly source width; no RGB scratch grows.
    assert_eq!(
      sink.luma_scratch_f32_capacity(),
      SRC,
      "luma scratch must be source-width f32"
    );
    assert_eq!(
      sink.rgb_scratch_capacity(),
      0,
      "rgb_f32-only sink must not reserve RGB scratch"
    );
  }
  // And rgb_f32 is the lossless broadcast of the exact f32 area mean.
  let binned = block_mean_2x2(&plane);
  for (x, &y) in binned.iter().enumerate() {
    assert_eq!(rgb_f32[x * 3], y, "px {x} R");
    assert_eq!(rgb_f32[x * 3 + 1], y, "px {x} G");
    assert_eq!(rgb_f32[x * 3 + 2], y, "px {x} B");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_identity_plan_matches_new_sink() {
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut direct = vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Grayf32>::new(SRC, SRC)
      .with_rgb_f32(&mut direct)
      .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
  }
  let mut via_area = vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_f32(&mut via_area)
        .unwrap();
    grayf32_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_resample_reuses_luma_stream_across_frames() {
  // A reused sink must reset the f32 luma stream each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence.
  let y1 = ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 3.0 - *p;
  }
  let pix1 = as_le_f32(&y1);
  let pix2 = as_le_f32(&y2);
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap();
    grayf32_to(
      &Grayf32Frame::new(&pix1, SRC as u32, SRC as u32, SRC as u32),
      FR,
      M,
      &mut sink,
    )
    .unwrap();
    grayf32_to(
      &Grayf32Frame::new(&pix2, SRC as u32, SRC as u32, SRC as u32),
      FR,
      M,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(
    luma_f32,
    block_mean_2x2(&y2),
    "frame 2 luma_f32 must area-downscale frame 2's luma"
  );
}

#[test]
fn grayf32_resample_no_outputs_is_a_no_op() {
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let src = Grayf32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);
  let mut sink =
    MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  // No outputs attached: a legal no-op, accepted without error.
  grayf32_to(&src, FR, M, &mut sink).unwrap();
  // A no-output call has no stream to sequence and never allocates.
  assert!(
    !sink.luma_stream_f32_allocated(),
    "no-output sink allocated an f32 luma stream"
  );
}

#[test]
fn grayf32_out_of_sequence_first_row_rejected_before_allocation() {
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let row3 = &pix[3 * SRC..4 * SRC];

  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  let mut sink =
    MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_f32(&mut luma_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Feed row 3 first — the stream expects strict sequencing from 0.
  let err = sink.process(Grayf32Row::new(row3, 3, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  // The out-of-sequence first row must be rejected before the f32 luma
  // stream is allocated and before the source-luma staging grows.
  assert!(
    !sink.luma_stream_f32_allocated(),
    "stream allocated for a rejected row"
  );
  assert_eq!(
    sink.luma_scratch_f32_capacity(),
    0,
    "source-luma staging grown for a rejected row"
  );
  assert!(
    luma_f32.iter().all(|&b| b == 0.0),
    "rejected row mutated output"
  );
}

#[test]
fn grayf32_rejected_first_row_does_not_poison_output_retry() {
  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot, so retrying row 0 after reconfiguring the output set
  // succeeds instead of tripping ResampleOutputsChanged.
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  let mut sink =
    MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_f32(&mut luma_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Grayf32Row::new(&pix[3 * SRC..4 * SRC], 3, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  sink.set_rgb_f32(&mut rgb_f32).unwrap();
  sink
    .process(Grayf32Row::new(&pix[..SRC], 0, M, FR))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

#[test]
fn grayf32_resample_rejects_mid_frame_out_of_sequence() {
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  let mut sink =
    MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_f32(&mut luma_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Grayf32Row::new(&pix[..SRC], 0, M, FR))
    .unwrap();
  // Skip row 1 — feeding row 2 next is out of sequence.
  let err = sink
    .process(Grayf32Row::new(&pix[2 * SRC..3 * SRC], 2, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
}

#[test]
fn grayf32_resample_rejects_mid_frame_output_change() {
  let plane = ramp();
  let pix = as_le_f32(&plane);
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  let mut sink =
    MixedSinker::<Grayf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Grayf32Row::new(&pix[..SRC], 0, M, FR))
    .unwrap();
  // Attaching a new output mid-frame trips the frozen-output check — and
  // `luma_f32` is the freshly-added FrozenOutputs slot, so this also
  // guards that slot against a mid-frame reattachment.
  sink.set_luma_f32(&mut luma_f32).unwrap();
  let err = sink
    .process(Grayf32Row::new(&pix[SRC..2 * SRC], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma_f32.iter().all(|&b| b == 0.0),
    "rejected row mutated the new output"
  );
}
