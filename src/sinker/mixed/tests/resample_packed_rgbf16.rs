//! Filter-resample coverage for the packed-half-float-RGB family
//! ([`Rgbf16`](crate::source::Rgbf16)) routed through the separable
//! filter engine.
//!
//! `Rgbf16` has no f16 stream — it widens its packed `R, G, B`
//! `half::f16` wire row to source-width host-native f32 RGB and bins in
//! **float**, reusing the same `f32` filter stream as
//! [`Rgbf32`](crate::source::Rgbf32). Per finalized output row the tail
//! **rounds the binned packed f32 row to `half::f16`** and runs the exact
//! direct `rgbf16_*` kernels (the area path's emit, now fed the filter
//! stream). So there is no per-channel PIL golden to assert against (the
//! engine's `f32` filter lands inside the +-1-LSB PIL budget, not 0-ULP,
//! and the f16 narrow is on top).
//!
//! Instead this mirrors the per-channel equivalence the packed-RGBA / the
//! [`Rgbf32`] filter tests use, with the f16 narrow folded into the
//! oracle: a 3-channel `Rgbf16` filter resample's R / G / B `rgb_f32`
//! output must each equal **bit-for-bit** the single-channel
//! [`FilterStream<f32>`] resample of that plane's *f16-widened* samples,
//! with each finalized element **rounded to `half::f16` then widened back
//! to f32** — exactly the narrow the area f16 emit applies to `rgb_f32`
//! (`half::f16::from_f32` then `rgbf16_to_rgb_f32_row`). The engine
//! filters each channel independently, so the per-plane oracle is exact.
//! Covered for `Triangle` / `CatmullRom` / `Lanczos3` across a downscale
//! (8 -> 4) and an upscale (4 -> 7), plus a filter-plan-accepted
//! regression (no `UnsupportedFilter`).

use crate::{
  KernelMatrix,
  frame::Rgbf16LeFrame,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{Rgbf16, rgbf16_to},
};

/// LE-encode a host-native `half::f16` slice as the `*LE` Frame contract
/// requires, so a fixture reads back identically on LE (no-op) and BE
/// (byte-swap) hosts. Mirrors `as_le_f16` in the area Rgbf16 tests.
fn as_le_f16(host: &[half::f16]) -> Vec<half::f16> {
  host
    .iter()
    .map(|&v| half::f16::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Per-channel packed f16 ramp spanning HDR (> 1.0) and negative values —
/// the filter path must carry both through R / G / B with no clamp (f16 is
/// a float narrow of the unclamped f32 bin). Distinct per channel so a
/// channel mix-up diverges immediately. Returns a host-native packed
/// `R, G, B` f16 buffer.
fn packed_frame_f16(w: usize, h: usize) -> Vec<half::f16> {
  let mut buf = vec![half::f16::ZERO; w * h * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    let i = i as f32;
    px[0] = half::f16::from_f32(0.1 + i * 0.05); // R: climbs through and past 1.0 (HDR)
    px[1] = half::f16::from_f32(2.5 - i * 0.07); // G: large, crosses 1.0 and 0.0
    px[2] = half::f16::from_f32(-0.4 + i * 0.03); // B: starts negative, climbs positive
  }
  buf
}

/// Single-channel filter resample of channel `c` of a packed `R, G, B`
/// f16 plane, via the merged engine's [`FilterStream<f32>`]
/// (channels = 1), with the f16 narrow the area emit applies to `rgb_f32`
/// folded in: each source f16 sample is **widened to f32** before
/// filtering (the `Rgbf16` wire->bin widen), and each finalized f32
/// element is **rounded to `half::f16` then widened back to f32** (the
/// `rgb_f32` emit path: `half::f16::from_f32` then `rgbf16_to_rgb_f32_row`,
/// which is a lossless f16->f32 widen). The 3-channel `Rgbf16` filter
/// resample's `rgb_f32` channel `c` must equal this **bit-for-bit**: same
/// engine, same coefficients, same narrow, run independently per plane.
#[allow(clippy::too_many_arguments)]
fn channel_plane_filter_f16<K: FilterKernel>(
  kernel: K,
  packed: &[half::f16],
  c: usize,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  use_simd: bool,
) -> Vec<f32> {
  // Widen the f16 channel to f32 — the same lossless widen the Rgbf16
  // wire->source-row conversion performs before binning.
  let mut plane = vec![0.0f32; sw * sh];
  for (dst, px) in plane.iter_mut().zip(packed.chunks_exact(3)) {
    *dst = px[c].to_f32();
  }
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<f32>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0.0f32; ow * oh];
  for y in 0..sh {
    stream
      .feed_row(y, &plane[y * sw..(y + 1) * sw], use_simd, |oy, fin| {
        for (dst, &src) in out[oy * ow..(oy + 1) * ow].iter_mut().zip(fin.iter()) {
          // Round to f16, then widen back to f32 — exactly the emit's
          // `rgb_f32` path (`from_f32` into the packed f16 scratch, then
          // `rgbf16_to_rgb_f32_row` widens the rounded row). Both narrows
          // are the canonical `half::f16` conversions, so this matches the
          // 3-channel emit element-for-element.
          *dst = half::f16::from_f32(src).to_f32();
        }
      })
      .expect("rows in order");
  }
  out
}

/// Runs the `Rgbf16` filter sink over a host-native packed `R, G, B` f16
/// source and returns the interleaved `rgb_f32` output (the f16-rounded
/// binned row widened back to f32).
fn rgbf16_filter_rgb_f32<K: FilterKernel + Copy>(
  packed_host: &[half::f16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> Vec<f32> {
  let wire = as_le_f16(packed_host);
  let src = Rgbf16LeFrame::try_new(&wire, sw as u32, sh as u32, (sw * 3) as u32).unwrap();
  let mut rgb_f32 = vec![0.0f32; ow * oh * 3];
  {
    let mut sink = MixedSinker::<Rgbf16, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_rgb_f32(&mut rgb_f32)
    .unwrap();
    rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  rgb_f32
}

/// Asserts the 3-channel `Rgbf16` filter `rgb_f32` output equals, for each
/// of R / G / B, the per-channel single-plane [`FilterStream<f32>`]
/// resample of the f16-widened source narrowed back through f16 — exact
/// (same engine, same narrow).
fn assert_rgb_f32_is_per_channel_filter<K: FilterKernel + Copy>(
  kernel: K,
  packed_host: &[half::f16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) {
  let resampled = rgbf16_filter_rgb_f32(packed_host, sw, sh, ow, oh, kernel);
  let mut max_diff = 0.0f32;
  for c in 0..3 {
    let plane = channel_plane_filter_f16(kernel, packed_host, c, sw, sh, ow, oh, true);
    for (i, &want) in plane.iter().enumerate() {
      let got = resampled[i * 3 + c];
      let diff = (got - want).abs();
      if diff > max_diff {
        max_diff = diff;
      }
      assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{ctx} channel {c} px {i}: rgb_f32 {got} vs per-plane f16-narrowed filter {want}",
      );
    }
  }
  assert_eq!(max_diff, 0.0, "{ctx}: per-channel diff must be exactly 0");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_downscale_rgb_f32_is_per_channel_filter() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let packed = packed_frame_f16(SW, SH);
  assert_rgb_f32_is_per_channel_filter(Triangle, &packed, SW, SH, OW, OH, "triangle down");
  assert_rgb_f32_is_per_channel_filter(CatmullRom, &packed, SW, SH, OW, OH, "catmullrom down");
  assert_rgb_f32_is_per_channel_filter(Lanczos3, &packed, SW, SH, OW, OH, "lanczos3 down");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_upscale_rgb_f32_is_per_channel_filter() {
  const SW: usize = 4;
  const SH: usize = 4;
  const OW: usize = 7;
  const OH: usize = 7;
  let packed = packed_frame_f16(SW, SH);
  assert_rgb_f32_is_per_channel_filter(Triangle, &packed, SW, SH, OW, OH, "triangle up");
  assert_rgb_f32_is_per_channel_filter(CatmullRom, &packed, SW, SH, OW, OH, "catmullrom up");
  assert_rgb_f32_is_per_channel_filter(Lanczos3, &packed, SW, SH, OW, OH, "lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_filter_plan_is_accepted() {
  // Regression: a filter plan must no longer raise `UnsupportedFilter`
  // at the Rgbf16 fence — the routed Filter arm runs the engine and
  // populates every attached output (the sentinel is gone).
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let packed = packed_frame_f16(SW, SH);
  let wire = as_le_f16(&packed);
  let src = Rgbf16LeFrame::try_new(&wire, SW as u32, SH as u32, (SW * 3) as u32).unwrap();

  let sentinel = f32::from_bits(0x7FC0_1234); // a quiet-NaN sentinel
  let mut rgb_f32 = vec![sentinel; OW * OH * 3];
  {
    let mut sink = MixedSinker::<Rgbf16, FilteredResampler<Triangle>>::with_resampler(
      SW,
      SH,
      FilteredResampler::new(OW, OH, Triangle),
    )
    .unwrap()
    .with_rgb_f32(&mut rgb_f32)
    .unwrap();
    rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert!(
    rgb_f32.iter().all(|&v| v.to_bits() != sentinel.to_bits()),
    "filter resample must populate rgb_f32 (no UnsupportedFilter)"
  );
}
