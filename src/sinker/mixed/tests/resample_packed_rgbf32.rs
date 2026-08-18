//! Filter-resample coverage for the packed-float-RGB family
//! ([`Rgbf32`](crate::source::Rgbf32)) routed through the separable
//! filter engine.
//!
//! `Rgbf32` is full-range float, so there is no per-channel PIL golden
//! to assert against (the engine's `f32` filter lands inside the +-1-LSB
//! PIL budget, not 0-ULP). Instead this mirrors the per-channel
//! equivalence the packed RGBA tests use: a 3-channel `Rgbf32` filter
//! resample's R / G / B output must each equal **bit-for-bit** the
//! single-channel [`FilterStream<f32>`] resample of that plane — the
//! *same engine*, run per plane — because the merged engine filters each
//! channel independently. Covered for `Triangle` / `CatmullRom` /
//! `Lanczos3` across a downscale (8 -> 4) and an upscale (4 -> 7), plus a
//! filter-plan-accepted regression (no `UnsupportedFilter`).

use crate::{
  KernelMatrix,
  frame::Rgbf32LeFrame,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{Rgbf32, rgbf32_to},
};

/// Re-encode a host-native f32 slice as **LE-encoded** byte storage, so
/// a fixture reads back identically on LE (no-op) and BE (byte-swap)
/// hosts. Mirrors `as_le_rgbf32` in the area Rgbf32 tests.
fn as_le_rgbf32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Per-channel f32 ramp spanning HDR (> 1.0) and negative values — the
/// filter path must carry both through R / G / B with no clamp (the
/// engine's `f32` finalize is the identity, PIL `F`-mode). Distinct per
/// channel so a channel mix-up diverges immediately.
fn packed_frame_f32(w: usize, h: usize) -> Vec<f32> {
  let mut buf = vec![0.0f32; w * h * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    let i = i as f32;
    px[0] = 0.1 + i * 0.05; // R: climbs through and past 1.0 (HDR)
    px[1] = 2.5 - i * 0.07; // G: large, crosses 1.0 and 0.0
    px[2] = -0.4 + i * 0.03; // B: starts negative, climbs positive
  }
  buf
}

/// Single-channel filter resample of channel `c` of a packed `R, G, B`
/// `f32` plane, via the merged engine's [`FilterStream<f32>`]
/// (channels = 1) — the per-channel oracle. The 3-channel `Rgbf32`
/// filter resample's channel `c` must equal this **bit-for-bit**: same
/// engine, same coefficients, run independently per plane.
#[allow(clippy::too_many_arguments)]
fn channel_plane_filter<K: FilterKernel>(
  kernel: K,
  packed: &[f32],
  c: usize,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  use_simd: bool,
) -> Vec<f32> {
  let mut plane = vec![0.0f32; sw * sh];
  for (dst, px) in plane.iter_mut().zip(packed.chunks_exact(3)) {
    *dst = px[c];
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
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// Runs the `Rgbf32` filter sink over a host-native packed `R, G, B`
/// f32 source and returns the interleaved `rgb_f32` output.
fn rgbf32_filter_rgb_f32<K: FilterKernel + Copy>(
  packed_host: &[f32],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> Vec<f32> {
  let wire = as_le_rgbf32(packed_host);
  let src = Rgbf32LeFrame::try_new(&wire, sw as u32, sh as u32, (sw * 3) as u32).unwrap();
  let mut rgb_f32 = vec![0.0f32; ow * oh * 3];
  {
    let mut sink = MixedSinker::<Rgbf32, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_rgb_f32(&mut rgb_f32)
    .unwrap();
    rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  rgb_f32
}

/// Asserts the 3-channel `Rgbf32` filter `rgb_f32` output equals, for
/// each of R / G / B, the per-channel single-plane [`FilterStream<f32>`]
/// resample of the source — exact (same engine).
fn assert_rgb_f32_is_per_channel_filter<K: FilterKernel + Copy>(
  kernel: K,
  packed_host: &[f32],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) {
  let resampled = rgbf32_filter_rgb_f32(packed_host, sw, sh, ow, oh, kernel);
  let mut max_diff = 0.0f32;
  for c in 0..3 {
    let plane = channel_plane_filter(kernel, packed_host, c, sw, sh, ow, oh, true);
    for (i, &want) in plane.iter().enumerate() {
      let got = resampled[i * 3 + c];
      let diff = (got - want).abs();
      if diff > max_diff {
        max_diff = diff;
      }
      assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{ctx} channel {c} px {i}: rgb_f32 {got} vs per-plane filter {want}",
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
fn rgbf32_downscale_rgb_f32_is_per_channel_filter() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let packed = packed_frame_f32(SW, SH);
  assert_rgb_f32_is_per_channel_filter(Triangle, &packed, SW, SH, OW, OH, "triangle down");
  assert_rgb_f32_is_per_channel_filter(CatmullRom, &packed, SW, SH, OW, OH, "catmullrom down");
  assert_rgb_f32_is_per_channel_filter(Lanczos3, &packed, SW, SH, OW, OH, "lanczos3 down");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_upscale_rgb_f32_is_per_channel_filter() {
  const SW: usize = 4;
  const SH: usize = 4;
  const OW: usize = 7;
  const OH: usize = 7;
  let packed = packed_frame_f32(SW, SH);
  assert_rgb_f32_is_per_channel_filter(Triangle, &packed, SW, SH, OW, OH, "triangle up");
  assert_rgb_f32_is_per_channel_filter(CatmullRom, &packed, SW, SH, OW, OH, "catmullrom up");
  assert_rgb_f32_is_per_channel_filter(Lanczos3, &packed, SW, SH, OW, OH, "lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_filter_plan_is_accepted() {
  // Regression: a filter plan must no longer raise `UnsupportedFilter`
  // at the Rgbf32 fence — the routed Filter arm runs the engine and
  // populates every attached output (the sentinel is gone).
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let packed = packed_frame_f32(SW, SH);
  let wire = as_le_rgbf32(&packed);
  let src = Rgbf32LeFrame::try_new(&wire, SW as u32, SH as u32, (SW * 3) as u32).unwrap();

  let sentinel = f32::from_bits(0x7FC0_1234); // a quiet-NaN sentinel
  let mut rgb_f32 = vec![sentinel; OW * OH * 3];
  {
    let mut sink = MixedSinker::<Rgbf32, FilteredResampler<Triangle>>::with_resampler(
      SW,
      SH,
      FilteredResampler::new(OW, OH, Triangle),
    )
    .unwrap()
    .with_rgb_f32(&mut rgb_f32)
    .unwrap();
    rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert!(
    rgb_f32.iter().all(|&v| v.to_bits() != sentinel.to_bits()),
    "filter resample must populate rgb_f32 (no UnsupportedFilter)"
  );
}
