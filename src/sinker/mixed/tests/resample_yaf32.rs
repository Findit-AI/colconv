//! Fused-downscale coverage for `Yaf32` (f32 gray + alpha) — the float twin of
//! the `Ya16` resample. The wire `[Y, A]` row stages as a host-native
//! 3-channel `[Yc, A, Y]` plane (colour luma, alpha, independent native luma),
//! all binned in f32; every output derives from each finalized binned row via
//! the direct `grayf32_*` kernels (colour Y broadcast for RGB, native Y for
//! luma, source α patched into the RGBA α channel).
//!
//! The fixtures use uniform 2x2 blocks so the 2:1 area mean of each block is the
//! block's own (exact) value: `mean(Y) = Y`, `mean(A) = A`, and the
//! premultiplied colour `mean(Y*A)/mean(A) = Y` (or `0` where `A == 0`). That
//! makes every oracle a closed form and isolates the resample wiring (native-Y
//! independence, straight vs premultiplied colour, the α patch).

use crate::{
  KernelMatrix,
  frame::Yaf32Frame,
  resample::{AreaResampler, FilteredResampler, ResampleError, Triangle},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Yaf32, yaf32_to},
};

const SRC: usize = 8;
const OUT: usize = 4;
const FR: bool = true;
const M: KernelMatrix = KernelMatrix::Bt709;

/// Per-output-block luma in `[0, 15/16]`.
fn y_block(oy: usize, ox: usize) -> f32 {
  (oy * OUT + ox) as f32 / 16.0
}

/// Per-output-block alpha in `{0, 0.25, 0.5, 0.75}` — includes the fully
/// transparent `A == 0` block that distinguishes straight from premultiplied.
fn a_block(oy: usize, ox: usize) -> f32 {
  ((oy + ox) % 4) as f32 / 4.0
}

/// Build the `SRC x SRC` packed `[Y, A]` plane with uniform 2x2 blocks, encoded
/// LE (the `yaf32le` contract; the loader recovers via `u32::from_le`).
fn source() -> Vec<f32> {
  let mut pix = vec![0.0f32; SRC * SRC * 2];
  for y in 0..SRC {
    for x in 0..SRC {
      let yv = y_block(y / 2, x / 2);
      let av = a_block(y / 2, x / 2);
      pix[(y * SRC + x) * 2] = f32::from_bits(yv.to_bits().to_le());
      pix[(y * SRC + x) * 2 + 1] = f32::from_bits(av.to_bits().to_le());
    }
  }
  pix
}

fn f32_to_u8(v: f32) -> u8 {
  (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn frame(pix: &[f32]) -> Yaf32Frame<'_> {
  Yaf32Frame::new(pix, SRC as u32, SRC as u32, (SRC * 2) as u32)
}

// ---- native-Y luma is the area mean, independent of alpha mode --------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yaf32_downscale_luma_f32_is_native_block_mean() {
  let pix = source();
  let src = frame(&pix);
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap();
    yaf32_to(&src, FR, M, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      assert_eq!(
        luma_f32[oy * OUT + ox],
        y_block(oy, ox),
        "luma_f32 must be the native Y block mean at ({oy},{ox})"
      );
    }
  }
}

// ---- straight alpha: colour Y broadcast, α = mean(A) ------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yaf32_downscale_rgba_straight() {
  let pix = source();
  let src = frame(&pix);
  let mut rgba = vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    yaf32_to(&src, FR, M, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      let y8 = f32_to_u8(y_block(oy, ox));
      let a8 = f32_to_u8(a_block(oy, ox));
      let p = (oy * OUT + ox) * 4;
      assert_eq!(
        &rgba[p..p + 4],
        &[y8, y8, y8, a8],
        "straight rgba at ({oy},{ox})"
      );
    }
  }
}

// ---- premultiplied: colour un-premultiplied, native luma unchanged ----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yaf32_downscale_rgba_premultiplied_unpremultiplies_and_keeps_native_luma() {
  let pix = source();
  let src = frame(&pix);
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap();
    yaf32_to(&src, FR, M, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      let a = a_block(oy, ox);
      // Uniform block: mean(Y*A)/mean(A) = Y (A != 0) else 0.
      let color_y = if a == 0.0 { 0.0 } else { y_block(oy, ox) };
      let y8 = f32_to_u8(color_y);
      let a8 = f32_to_u8(a);
      let p = (oy * OUT + ox) * 4;
      assert_eq!(
        &rgba[p..p + 4],
        &[y8, y8, y8, a8],
        "premultiplied rgba at ({oy},{ox})"
      );
      // Native Y luma is mode-independent: still mean(Y), even where the
      // premultiplied colour collapsed to black.
      assert_eq!(
        luma_f32[oy * OUT + ox],
        y_block(oy, ox),
        "premultiplied native luma must stay mean(Y) at ({oy},{ox})"
      );
    }
  }
}

// ---- premultiplied + filter has no analogue → UnsupportedFilter -------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yaf32_premultiplied_filter_is_unsupported() {
  let pix = source();
  let src = frame(&pix);
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut sink = MixedSinker::<Yaf32, FilteredResampler<Triangle>>::with_resampler(
    SRC,
    SRC,
    FilteredResampler::new(OUT, OUT, Triangle),
  )
  .unwrap()
  .with_alpha_mode(AlphaMode::Premultiplied)
  .with_rgba(&mut rgba)
  .unwrap();
  let err = yaf32_to(&src, FR, M, &mut sink).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
    ),
    "premultiplied filter must surface UnsupportedFilter, got {err:?}"
  );
}

// ---- straight filter runs (no UnsupportedFilter) ----------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yaf32_straight_filter_populates_rgba_and_luma() {
  let pix = source();
  let src = frame(&pix);
  let mut rgba = vec![0xABu8; OUT * OUT * 4];
  let mut luma = vec![0xCDu8; OUT * OUT];
  {
    let mut sink = MixedSinker::<Yaf32, FilteredResampler<Triangle>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(OUT, OUT, Triangle),
    )
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    yaf32_to(&src, FR, M, &mut sink).unwrap();
  }
  // The straight filter path must have written every output (no sentinel left).
  assert!(rgba.iter().any(|&b| b != 0xAB), "rgba must be filtered");
  assert!(luma.iter().any(|&b| b != 0xCD), "luma must be filtered");
}
