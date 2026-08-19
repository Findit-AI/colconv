//! Separable-filter resample coverage for the 8-bit semi-planar YUV family —
//! `Nv12` (4:2:0), `Nv16` (4:2:2), `Nv21` (4:2:0, VU), `Nv24` (4:4:4),
//! `Nv42` (4:4:4, VU) — routed through the merged filter engine.
//!
//! Each format routes a `Filter` plan to
//! [`planar_dual_filter_resample`](super::super::planar_resample::planar_dual_filter_resample):
//! the interleaved chroma is de-interleaved + upsampled by the **same**
//! `nv*_to_rgb_row` kernel the area path (and the identity path) uses into a
//! source-width RGB row, then the RGB is resampled by the signed-coefficient
//! filter stream (the filter twin of the area bin). Luma stays native Y: the
//! Y plane is filter-resampled as a 1-channel `u8` stream, never
//! colour-derived. So:
//!
//! - **`rgb` / `rgba`** equal the equivalent `Rgb24` filter resample of the
//!   source converted to u8 RGB (the exact source-width RGB the filter path
//!   bins). `rgba` is that RGB expanded with opaque (`0xFF`) alpha.
//! - **`luma`** equals a single-channel [`FilterStream<u8>`] resample of the
//!   Y plane; **`luma_u16`** is that resampled Y zero-extended.
//!
//! These are 8-bit sources, so there is **no native-depth clamp** (the `u8`
//! stream finalizes to the full `u8` range, which *is* the native range) —
//! and the 4:2:0 native fast tier is area-only, so the filter sink bypasses
//! it (the filter plan branches before the native-route machinery). The
//! `Rgb24` oracle is gated on `rgb` (the oracle source). The native-Y luma
//! equivalence and the filter-plan-accepted regression are
//! feature-independent, so they also guard the `yuv-semi-planar`-solo build
//! (where the routing exists but no packed-RGB oracle does).

use crate::{
  KernelMatrix,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{Nv12, Nv16, Nv21, Nv24, Nv42, nv12_to, nv16_to, nv21_to, nv24_to, nv42_to},
};
use mediaframe::frame::{Nv12Frame, Nv16Frame, Nv21Frame, Nv24Frame, Nv42Frame};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  rgb: Vec<u8>,
  /// Only the `rgb`-gated equivalence module reads the RGBA colour output,
  /// so it is dead in a `yuv-semi-planar`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgba: Vec<u8>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

/// The bits a filter test needs to drive one 8-bit semi-planar YUV format:
/// how to build its planes (Y full-res, interleaved UV at the format's
/// chroma subsampling and byte order), how to run its filter sink, and the
/// direct full-res u8 RGB conversion that produces the exact rows the filter
/// path consumes (the colour equivalence oracle's input).
trait SemiPlanarYuvFilter {
  /// Per-axis chroma divisors `(horizontal, vertical)` — e.g. 4:2:0 is
  /// `(2, 2)`, 4:2:2 is `(2, 1)`, 4:4:4 is `(1, 1)`. The test geometries
  /// (8x8 down, 4x4 up) divide evenly under all three and satisfy every
  /// width-alignment rule (4:2:x need width % 2 == 0).
  const CW_DIV: usize;
  const CH_DIV: usize;
  /// `true` writes the chroma plane `V0 U0 …` (Nv21 / Nv42); `false` writes
  /// `U0 V0 …` (Nv12 / Nv16 / Nv24).
  const SWAP_UV: bool;

  /// Run the format's filter sink over the planes (`sw x sh`) at `ow x oh`
  /// under `kernel`, attaching every output the equivalence asserts on.
  #[allow(clippy::too_many_arguments)]
  fn filter_outputs<K: FilterKernel + Copy>(
    y: &[u8],
    uv: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs;

  /// Direct full-res u8 RGB conversion of the planes (`w x h`) — the exact
  /// source-width u8 RGB the filter path bins, so it is the `Rgb24`
  /// oracle's input. Only the `rgb`-gated equivalence module consumes it, so
  /// it is dead in a `yuv-semi-planar`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  fn direct_rgb_u8(y: &[u8], uv: &[u8], w: usize, h: usize) -> Vec<u8>;
}

/// A per-channel ramp for the planes: Y varies per pixel and U/V vary per
/// chroma sample so every filter window sees distinct neighbours (a channel
/// mix-up or a row/column transpose diverges immediately). All samples
/// interior so the conversions see real math. Returns the Y plane and the
/// interleaved chroma plane (byte order per `F::SWAP_UV`).
fn semi_planar_ramp<F: SemiPlanarYuvFilter>(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>) {
  let cw = sw / F::CW_DIV;
  let ch = sh / F::CH_DIV;
  let mut y = vec![0u8; sw * sh];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i % 100) * 2) as u8;
  }
  let mut uv = vec![0u8; cw * ch * 2];
  for i in 0..cw * ch {
    let u = (70 + (i % 30) * 5) as u8;
    let v = (200u8).wrapping_sub(((i % 40) * 4) as u8);
    let (a, b) = if F::SWAP_UV { (v, u) } else { (u, v) };
    uv[i * 2] = a;
    uv[i * 2 + 1] = b;
  }
  (y, uv)
}

// The frame constructors take `(y, uv, width, height, y_stride, uv_stride)`.
// 4:2:0 / 4:2:2 chroma is half-width-interleaved so the chroma plane stride
// is `width` bytes (`$uv_stride_mul = 1`); 4:4:4 chroma is
// full-width-interleaved so the stride is `2 * width`
// (`$uv_stride_mul = 2`).
macro_rules! semi_planar_format {
  (
    $name:ident, $src:ty, $frame:ident, $walk:path,
    $cw_div:expr, $ch_div:expr, $swap_uv:expr, $uv_stride_mul:expr
  ) => {
    struct $name;

    impl SemiPlanarYuvFilter for $name {
      const CW_DIV: usize = $cw_div;
      const CH_DIV: usize = $ch_div;
      const SWAP_UV: bool = $swap_uv;

      #[allow(clippy::too_many_arguments)]
      fn filter_outputs<K: FilterKernel + Copy>(
        y: &[u8],
        uv: &[u8],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        kernel: K,
      ) -> FilterOutputs {
        let src = $frame::new(
          y,
          uv,
          sw as u32,
          sh as u32,
          sw as u32,
          (sw * $uv_stride_mul) as u32,
        );
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgba = vec![0u8; ow * oh * 4];
        let mut luma = vec![0u8; ow * oh];
        let mut luma_u16 = vec![0u16; ow * oh];
        {
          let mut sink = MixedSinker::<$src, FilteredResampler<K>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, kernel),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walk(&src, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        FilterOutputs {
          rgb,
          rgba,
          luma,
          luma_u16,
        }
      }

      fn direct_rgb_u8(y: &[u8], uv: &[u8], w: usize, h: usize) -> Vec<u8> {
        let src = $frame::new(
          y,
          uv,
          w as u32,
          h as u32,
          w as u32,
          (w * $uv_stride_mul) as u32,
        );
        let mut rgb = vec![0u8; w * h * 3];
        {
          let mut sink = MixedSinker::<$src>::new(w, h).with_rgb(&mut rgb).unwrap();
          $walk(&src, FR, sink.set_kernel_matrix(M)).unwrap();
        }
        rgb
      }
    }
  };
}

semi_planar_format!(Nv12F, Nv12, Nv12Frame, nv12_to, 2, 2, false, 1);
semi_planar_format!(Nv16F, Nv16, Nv16Frame, nv16_to, 2, 1, false, 1);
semi_planar_format!(Nv21F, Nv21, Nv21Frame, nv21_to, 2, 2, true, 1);
semi_planar_format!(Nv24F, Nv24, Nv24Frame, nv24_to, 1, 1, false, 2);
semi_planar_format!(Nv42F, Nv42, Nv42Frame, nv42_to, 1, 1, true, 2);

// ---- Single-channel native-Y luma oracle (feature-independent) --------

/// Single-channel filter resample of a u8 Y plane via the merged engine's
/// [`FilterStream<u8>`] (channels = 1) — the luma oracle. A semi-planar 8-bit
/// filter path's `luma` must equal this **byte-for-bit** (same engine, same
/// coefficients, the Y plane resampled directly). 8-bit, so no native-depth
/// clamp: the full `u8` range *is* the native range.
fn native_y_filter<K: FilterKernel>(
  kernel: K,
  y: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u8> {
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u8>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0u8; ow * oh];
  for row in 0..sh {
    stream
      .feed_row(row, &y[row * sw..(row + 1) * sw], true, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// Asserts a format's filter `luma` equals the single-channel native-Y
/// oracle, and `luma_u16` is that resampled Y zero-extended. Returns the max
/// per-sample `luma` diff (exactly 0 — same engine, no clamp on either).
fn assert_native_y_luma<F: SemiPlanarYuvFilter, K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u8 {
  let (y, uv) = semi_planar_ramp::<F>(sw, sh);
  let got = F::filter_outputs(&y, &uv, sw, sh, ow, oh, kernel);
  let want = native_y_filter(kernel, &y, sw, sh, ow, oh);

  let mut max_diff = 0u8;
  for (i, (&g, &w)) in got.luma.iter().zip(want.iter()).enumerate() {
    max_diff = max_diff.max(g.abs_diff(w));
    assert_eq!(
      g, w,
      "{ctx} luma[{i}]: {g} vs single-channel native-Y filter {w}"
    );
  }
  for (&lo, &hi) in got.luma.iter().zip(got.luma_u16.iter()) {
    assert_eq!(
      hi, lo as u16,
      "{ctx} luma_u16: must be the resampled native Y zero-extended"
    );
  }
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_luma_filter_is_single_channel_native_y() {
  // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma must be the
  // native-Y single-channel filter (max diff 0), luma_u16 its zero-extend.
  assert_native_y_luma::<Nv12F, _>(Triangle, 8, 8, 4, 4, "nv12 triangle down");
  assert_native_y_luma::<Nv12F, _>(CatmullRom, 8, 8, 4, 4, "nv12 catmullrom down");
  assert_native_y_luma::<Nv12F, _>(Lanczos3, 8, 8, 4, 4, "nv12 lanczos3 down");
  assert_native_y_luma::<Nv12F, _>(Triangle, 4, 4, 7, 7, "nv12 triangle up");
  assert_native_y_luma::<Nv12F, _>(CatmullRom, 4, 4, 7, 7, "nv12 catmullrom up");
  assert_native_y_luma::<Nv12F, _>(Lanczos3, 4, 4, 7, 7, "nv12 lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Nv16F, _>(Triangle, 8, 8, 4, 4, "nv16 triangle down");
  assert_native_y_luma::<Nv16F, _>(CatmullRom, 8, 8, 4, 4, "nv16 catmullrom down");
  assert_native_y_luma::<Nv16F, _>(Lanczos3, 8, 8, 4, 4, "nv16 lanczos3 down");
  assert_native_y_luma::<Nv16F, _>(Triangle, 4, 4, 7, 7, "nv16 triangle up");
  assert_native_y_luma::<Nv16F, _>(CatmullRom, 4, 4, 7, 7, "nv16 catmullrom up");
  assert_native_y_luma::<Nv16F, _>(Lanczos3, 4, 4, 7, 7, "nv16 lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv21_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Nv21F, _>(Triangle, 8, 8, 4, 4, "nv21 triangle down");
  assert_native_y_luma::<Nv21F, _>(CatmullRom, 8, 8, 4, 4, "nv21 catmullrom down");
  assert_native_y_luma::<Nv21F, _>(Lanczos3, 8, 8, 4, 4, "nv21 lanczos3 down");
  assert_native_y_luma::<Nv21F, _>(Triangle, 4, 4, 7, 7, "nv21 triangle up");
  assert_native_y_luma::<Nv21F, _>(CatmullRom, 4, 4, 7, 7, "nv21 catmullrom up");
  assert_native_y_luma::<Nv21F, _>(Lanczos3, 4, 4, 7, 7, "nv21 lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Nv24F, _>(Triangle, 8, 8, 4, 4, "nv24 triangle down");
  assert_native_y_luma::<Nv24F, _>(CatmullRom, 8, 8, 4, 4, "nv24 catmullrom down");
  assert_native_y_luma::<Nv24F, _>(Lanczos3, 8, 8, 4, 4, "nv24 lanczos3 down");
  assert_native_y_luma::<Nv24F, _>(Triangle, 4, 4, 7, 7, "nv24 triangle up");
  assert_native_y_luma::<Nv24F, _>(CatmullRom, 4, 4, 7, 7, "nv24 catmullrom up");
  assert_native_y_luma::<Nv24F, _>(Lanczos3, 4, 4, 7, 7, "nv24 lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Nv42F, _>(Triangle, 8, 8, 4, 4, "nv42 triangle down");
  assert_native_y_luma::<Nv42F, _>(CatmullRom, 8, 8, 4, 4, "nv42 catmullrom down");
  assert_native_y_luma::<Nv42F, _>(Lanczos3, 8, 8, 4, 4, "nv42 lanczos3 down");
  assert_native_y_luma::<Nv42F, _>(Triangle, 4, 4, 7, 7, "nv42 triangle up");
  assert_native_y_luma::<Nv42F, _>(CatmullRom, 4, 4, 7, 7, "nv42 catmullrom up");
  assert_native_y_luma::<Nv42F, _>(Lanczos3, 4, 4, 7, 7, "nv42 lanczos3 up");
}

// ---- Cross-frame stream reset (feature-independent) -------------------

/// A reused filter sink must reset its filter streams each frame, else
/// frame 2 row 0 is rejected as out-of-sequence (the filter twin of the
/// area cross-frame coverage). Drives two frames through one `Nv24` filter
/// sink and asserts frame 2's `luma` is the single-channel native-Y filter
/// of frame 2's Y — confirming the `begin_frame` reset of
/// `luma_filter_stream` / `rgb_filter_stream`.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_filter_reuses_streams_across_frames() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y1, uv) = semi_planar_ramp::<Nv24F>(SW, SH);
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let (sw, sh) = (SW as u32, SH as u32);
  let frame1 = Nv24Frame::new(&y1, &uv, sw, sh, sw, (SW * 2) as u32);
  let frame2 = Nv24Frame::new(&y2, &uv, sw, sh, sw, (SW * 2) as u32);
  let mut luma = vec![0u8; OW * OH];
  {
    let mut sink = MixedSinker::<Nv24, FilteredResampler<Triangle>>::with_resampler(
      SW,
      SH,
      FilteredResampler::new(OW, OH, Triangle),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    nv24_to(&frame1, FR, sink.set_kernel_matrix(M)).unwrap();
    nv24_to(&frame2, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let want = native_y_filter(Triangle, &y2, SW, SH, OW, OH);
  assert_eq!(
    luma, want,
    "frame 2 luma must be the native-Y filter of frame 2's Y (streams reset each frame)"
  );
}

// ---- Filter-plan-accepted regression (feature-independent) ------------

/// A filter plan must be accepted by the semi-planar 8-bit YUV sink — before
/// this routing it was rejected with `UnsupportedFilter`; now it produces a
/// real (non-sentinel) output. Feature-independent, so it guards the
/// `yuv-semi-planar`-solo build.
fn assert_filter_plan_accepted<F: SemiPlanarYuvFilter>(ctx: &str) {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, uv) = semi_planar_ramp::<F>(SW, SH);
  // A filter plan no longer raises `UnsupportedFilter`; the resampled
  // outputs are populated (the rgb colour and the luma are non-zero for
  // this ramp).
  let got = F::filter_outputs(&y, &uv, SW, SH, OW, OH, Triangle);
  assert!(
    got.rgb.iter().any(|&v| v != 0),
    "{ctx}: filter resample must populate rgb (no UnsupportedFilter)"
  );
  assert!(
    got.luma.iter().any(|&v| v != 0),
    "{ctx}: filter resample must populate luma (no UnsupportedFilter)"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv_semi_planar_8bit_filter_plans_are_accepted() {
  assert_filter_plan_accepted::<Nv12F>("nv12");
  assert_filter_plan_accepted::<Nv16F>("nv16");
  assert_filter_plan_accepted::<Nv21F>("nv21");
  assert_filter_plan_accepted::<Nv24F>("nv24");
  assert_filter_plan_accepted::<Nv42F>("nv42");
}

// ---- Packed-RGB equivalence oracle (gated on `rgb`) -------------------
//
// The filter path converts the YUV to RGB with the same `nv*_to_rgb_row`
// kernel the direct sink uses, then filters the RGB. So a semi-planar filter
// colour output equals the equivalent `Rgb24` filter resample of those
// exact converted pixels.

#[cfg(feature = "rgb")]
mod packed_rgb_equivalence {
  use super::*;
  use crate::source::{Rgb24, rgb24_to};

  /// `Rgb24` filter resample of a u8 RGB frame at `ow x oh` under `kernel`,
  /// returning the `rgb` output.
  fn rgb24_filter_rgb<K: FilterKernel>(
    rgb: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> Vec<u8> {
    let src = crate::frame::Rgb24Frame::new(rgb, sw as u32, sh as u32, (sw * 3) as u32);
    let mut out = vec![0u8; ow * oh * 3];
    {
      let mut sink = MixedSinker::<Rgb24, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_rgb(&mut out)
      .unwrap();
      rgb24_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    out
  }

  /// Asserts a format's filter colour outputs equal the equivalent `Rgb24`
  /// filter of the YUV->RGB-converted source pixels: `rgb` == `Rgb24`
  /// filter, `rgba` == that RGB with opaque alpha. Returns the max
  /// per-channel `rgb` diff (0 — same engine, same converted pixels).
  fn assert_color_equals_rgb24<F: SemiPlanarYuvFilter, K: FilterKernel + Copy>(
    kernel: K,
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    ctx: &str,
  ) -> u8 {
    let (y, uv) = semi_planar_ramp::<F>(sw, sh);
    let got = F::filter_outputs(&y, &uv, sw, sh, ow, oh, kernel);

    let src_rgb = F::direct_rgb_u8(&y, &uv, sw, sh);
    let want = rgb24_filter_rgb(&src_rgb, sw, sh, ow, oh, kernel);

    let mut max_diff = 0u8;
    for (i, (&g, &w)) in got.rgb.iter().zip(want.iter()).enumerate() {
      max_diff = max_diff.max(g.abs_diff(w));
      assert_eq!(g, w, "{ctx} rgb[{i}]: {g} vs Rgb24 filter {w}");
    }
    // rgba colour == rgb, opaque alpha (0xFF).
    for (px, c) in got.rgba.chunks_exact(4).zip(want.chunks_exact(3)) {
      assert_eq!(&px[..3], c, "{ctx} rgba colour == rgb");
      assert_eq!(px[3], 0xFF, "{ctx} rgba opaque alpha");
    }

    max_diff
  }

  macro_rules! color_equiv_tests {
    ($down:ident, $up:ident, $fmt:ty, $label:literal) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $down() {
        assert_color_equals_rgb24::<$fmt, _>(
          Triangle,
          8,
          8,
          4,
          4,
          concat!($label, " triangle down"),
        );
        assert_color_equals_rgb24::<$fmt, _>(
          CatmullRom,
          8,
          8,
          4,
          4,
          concat!($label, " catmullrom down"),
        );
        assert_color_equals_rgb24::<$fmt, _>(
          Lanczos3,
          8,
          8,
          4,
          4,
          concat!($label, " lanczos3 down"),
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $up() {
        assert_color_equals_rgb24::<$fmt, _>(Triangle, 4, 4, 7, 7, concat!($label, " triangle up"));
        assert_color_equals_rgb24::<$fmt, _>(
          CatmullRom,
          4,
          4,
          7,
          7,
          concat!($label, " catmullrom up"),
        );
        assert_color_equals_rgb24::<$fmt, _>(Lanczos3, 4, 4, 7, 7, concat!($label, " lanczos3 up"));
      }
    };
  }

  color_equiv_tests!(
    nv12_downscale_color_filter_equals_rgb24,
    nv12_upscale_color_filter_equals_rgb24,
    Nv12F,
    "nv12"
  );
  color_equiv_tests!(
    nv16_downscale_color_filter_equals_rgb24,
    nv16_upscale_color_filter_equals_rgb24,
    Nv16F,
    "nv16"
  );
  color_equiv_tests!(
    nv21_downscale_color_filter_equals_rgb24,
    nv21_upscale_color_filter_equals_rgb24,
    Nv21F,
    "nv21"
  );
  color_equiv_tests!(
    nv24_downscale_color_filter_equals_rgb24,
    nv24_upscale_color_filter_equals_rgb24,
    Nv24F,
    "nv24"
  );
  color_equiv_tests!(
    nv42_downscale_color_filter_equals_rgb24,
    nv42_upscale_color_filter_equals_rgb24,
    Nv42F,
    "nv42"
  );
}
