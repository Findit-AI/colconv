//! Separable-filter resample coverage for the 8-bit planar YUV family —
//! `Yuv410p` (4:1:0), `Yuv420p` (4:2:0), `Yuv422p` (4:2:2), `Yuv444p`
//! (4:4:4), `Yuv440p` (4:4:0) — routed through the merged filter engine.
//!
//! Each format routes a `Filter` plan to
//! [`planar_dual_filter_resample`](super::super::planar_resample::planar_dual_filter_resample):
//! the separate Y/U/V planes are converted to a source-width RGB row with
//! the **same** `*_to_rgb_row` kernel the area path (and the identity path)
//! uses, then the RGB is resampled by the signed-coefficient filter stream
//! (the filter twin of the area bin). Luma stays native Y: the Y plane is
//! filter-resampled as a 1-channel `u8` stream, never colour-derived. So:
//!
//! - **`rgb` / `rgba`** equal the equivalent `Rgb24` filter resample of the
//!   source converted to u8 RGB (the exact source-width RGB the filter path
//!   bins). `rgba` is that RGB expanded with opaque (`0xFF`) alpha.
//! - **`luma`** equals a single-channel [`FilterStream<u8>`] resample of the
//!   Y plane; **`luma_u16`** is that resampled Y zero-extended.
//!
//! These are 8-bit sources, so there is **no native-depth clamp** (the `u8`
//! stream finalizes to the full `u8` range, which *is* the native range) —
//! unlike the high-bit planar / packed YUV filter routes, which clamp to
//! `(1 << BITS) - 1`. The `Rgb24` oracle is gated on `rgb` (the oracle
//! source). The native-Y luma equivalence and the filter-plan-accepted
//! regression are feature-independent, so they also guard the
//! `yuv-planar`-solo build (where the routing exists but no packed-RGB
//! oracle does).

use crate::{
  KernelMatrix,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{
    Yuv410p, Yuv420p, Yuv422p, Yuv440p, Yuv444p, yuv410p_to, yuv420p_to, yuv422p_to, yuv440p_to,
    yuv444p_to,
  },
};
use mediaframe::frame::{Yuv410pFrame, Yuv420pFrame, Yuv422pFrame, Yuv440pFrame, Yuv444pFrame};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  rgb: Vec<u8>,
  /// Only the `rgb`-gated equivalence module reads the RGBA colour output,
  /// so it is dead in a `yuv-planar`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgba: Vec<u8>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

/// The bits a filter test needs to drive one 8-bit planar YUV format: how
/// to build its planes (Y full-res, U/V at the format's chroma
/// subsampling), how to run its filter sink, and the direct full-res u8 RGB
/// and native-Y conversions that produce the exact rows the filter path
/// consumes (the equivalence oracles' inputs).
trait PlanarYuvFilter {
  /// Per-axis chroma divisors `(horizontal, vertical)` — e.g. 4:2:0 is
  /// `(2, 2)`, 4:4:0 is `(1, 2)`, 4:4:4 is `(1, 1)`. The test geometries
  /// (8x8 down, 4x4 up) divide evenly under all five and satisfy every
  /// width-alignment rule (4:1:0 needs width % 4 == 0, 4:2:x width % 2).
  const CW_DIV: usize;
  const CH_DIV: usize;

  /// Run the format's filter sink over the planes (`sw x sh`) at `ow x oh`
  /// under `kernel`, attaching every output the equivalence asserts on.
  #[allow(clippy::too_many_arguments)]
  fn filter_outputs<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs;

  /// Direct full-res u8 RGB conversion of the planes (`w x h`) — the exact
  /// source-width u8 RGB the filter path bins, so it is the `Rgb24`
  /// oracle's input. Only the `rgb`-gated equivalence module consumes it,
  /// so it is dead in a `yuv-planar`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  fn direct_rgb_u8(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8>;
}

/// A per-channel ramp for the planes: Y varies per pixel and U/V vary per
/// chroma sample so every filter window sees distinct neighbours (a channel
/// mix-up or a row/column transpose diverges immediately). All samples
/// interior so the conversions see real math.
fn planar_ramp<F: PlanarYuvFilter>(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = sw / F::CW_DIV;
  let ch = sh / F::CH_DIV;
  let mut y = vec![0u8; sw * sh];
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i % 100) * 2) as u8;
  }
  for (i, p) in u.iter_mut().enumerate() {
    *p = (70 + (i % 30) * 5) as u8;
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = (200u8).wrapping_sub(((i % 40) * 4) as u8);
  }
  (y, u, v)
}

macro_rules! planar_format {
  (
    $name:ident, $src:ty, $frame:ident, $walk:path,
    $cw_div:expr, $ch_div:expr
  ) => {
    struct $name;

    impl PlanarYuvFilter for $name {
      const CW_DIV: usize = $cw_div;
      const CH_DIV: usize = $ch_div;

      #[allow(clippy::too_many_arguments)]
      fn filter_outputs<K: FilterKernel + Copy>(
        y: &[u8],
        u: &[u8],
        v: &[u8],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        kernel: K,
      ) -> FilterOutputs {
        // The frame constructor takes `(y, u, v, width, height, y_stride,
        // u_stride, v_stride)`; the U / V strides are the chroma plane width
        // `cw` (the chroma height is derived from the format's subsampling).
        let cw = (sw / $cw_div) as u32;
        let src = $frame::new(y, u, v, sw as u32, sh as u32, sw as u32, cw, cw);
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
          $walk(&src, FR, M, &mut sink).unwrap();
        }
        FilterOutputs {
          rgb,
          rgba,
          luma,
          luma_u16,
        }
      }

      fn direct_rgb_u8(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
        let cw = (w / $cw_div) as u32;
        let src = $frame::new(y, u, v, w as u32, h as u32, w as u32, cw, cw);
        let mut rgb = vec![0u8; w * h * 3];
        {
          let mut sink = MixedSinker::<$src>::new(w, h).with_rgb(&mut rgb).unwrap();
          $walk(&src, FR, M, &mut sink).unwrap();
        }
        rgb
      }
    }
  };
}

// 4:1:0 — quarter-width, quarter-height chroma (width multiple of 4).
planar_format!(Yuv410F, Yuv410p, Yuv410pFrame, yuv410p_to, 4, 4);
// 4:2:0 — half-width, half-height chroma (width even). Routes via the
// row-stage-equivalent filter path (the native fast tier is area-only).
planar_format!(Yuv420F, Yuv420p, Yuv420pFrame, yuv420p_to, 2, 2);
// 4:2:2 — half-width, full-height chroma (width even).
planar_format!(Yuv422F, Yuv422p, Yuv422pFrame, yuv422p_to, 2, 1);
// 4:4:4 — full-width, full-height chroma.
planar_format!(Yuv444F, Yuv444p, Yuv444pFrame, yuv444p_to, 1, 1);
// 4:4:0 — full-width, half-height chroma.
planar_format!(Yuv440F, Yuv440p, Yuv440pFrame, yuv440p_to, 1, 2);

// ---- Single-channel native-Y luma oracle (feature-independent) --------

/// Single-channel filter resample of a u8 Y plane via the merged engine's
/// [`FilterStream<u8>`] (channels = 1) — the luma oracle. A planar 8-bit
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
fn assert_native_y_luma<F: PlanarYuvFilter, K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u8 {
  let (y, u, v) = planar_ramp::<F>(sw, sh);
  let got = F::filter_outputs(&y, &u, &v, sw, sh, ow, oh, kernel);
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
fn yuv410p_luma_filter_is_single_channel_native_y() {
  // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma must be the
  // native-Y single-channel filter (max diff 0), luma_u16 its zero-extend.
  assert_native_y_luma::<Yuv410F, _>(Triangle, 8, 8, 4, 4, "yuv410p triangle down");
  assert_native_y_luma::<Yuv410F, _>(CatmullRom, 8, 8, 4, 4, "yuv410p catmullrom down");
  assert_native_y_luma::<Yuv410F, _>(Lanczos3, 8, 8, 4, 4, "yuv410p lanczos3 down");
  assert_native_y_luma::<Yuv410F, _>(Triangle, 4, 4, 7, 7, "yuv410p triangle up");
  assert_native_y_luma::<Yuv410F, _>(CatmullRom, 4, 4, 7, 7, "yuv410p catmullrom up");
  assert_native_y_luma::<Yuv410F, _>(Lanczos3, 4, 4, 7, 7, "yuv410p lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Yuv420F, _>(Triangle, 8, 8, 4, 4, "yuv420p triangle down");
  assert_native_y_luma::<Yuv420F, _>(CatmullRom, 8, 8, 4, 4, "yuv420p catmullrom down");
  assert_native_y_luma::<Yuv420F, _>(Lanczos3, 8, 8, 4, 4, "yuv420p lanczos3 down");
  assert_native_y_luma::<Yuv420F, _>(Triangle, 4, 4, 7, 7, "yuv420p triangle up");
  assert_native_y_luma::<Yuv420F, _>(CatmullRom, 4, 4, 7, 7, "yuv420p catmullrom up");
  assert_native_y_luma::<Yuv420F, _>(Lanczos3, 4, 4, 7, 7, "yuv420p lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Yuv422F, _>(Triangle, 8, 8, 4, 4, "yuv422p triangle down");
  assert_native_y_luma::<Yuv422F, _>(CatmullRom, 8, 8, 4, 4, "yuv422p catmullrom down");
  assert_native_y_luma::<Yuv422F, _>(Lanczos3, 8, 8, 4, 4, "yuv422p lanczos3 down");
  assert_native_y_luma::<Yuv422F, _>(Triangle, 4, 4, 7, 7, "yuv422p triangle up");
  assert_native_y_luma::<Yuv422F, _>(CatmullRom, 4, 4, 7, 7, "yuv422p catmullrom up");
  assert_native_y_luma::<Yuv422F, _>(Lanczos3, 4, 4, 7, 7, "yuv422p lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Yuv444F, _>(Triangle, 8, 8, 4, 4, "yuv444p triangle down");
  assert_native_y_luma::<Yuv444F, _>(CatmullRom, 8, 8, 4, 4, "yuv444p catmullrom down");
  assert_native_y_luma::<Yuv444F, _>(Lanczos3, 8, 8, 4, 4, "yuv444p lanczos3 down");
  assert_native_y_luma::<Yuv444F, _>(Triangle, 4, 4, 7, 7, "yuv444p triangle up");
  assert_native_y_luma::<Yuv444F, _>(CatmullRom, 4, 4, 7, 7, "yuv444p catmullrom up");
  assert_native_y_luma::<Yuv444F, _>(Lanczos3, 4, 4, 7, 7, "yuv444p lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Yuv440F, _>(Triangle, 8, 8, 4, 4, "yuv440p triangle down");
  assert_native_y_luma::<Yuv440F, _>(CatmullRom, 8, 8, 4, 4, "yuv440p catmullrom down");
  assert_native_y_luma::<Yuv440F, _>(Lanczos3, 8, 8, 4, 4, "yuv440p lanczos3 down");
  assert_native_y_luma::<Yuv440F, _>(Triangle, 4, 4, 7, 7, "yuv440p triangle up");
  assert_native_y_luma::<Yuv440F, _>(CatmullRom, 4, 4, 7, 7, "yuv440p catmullrom up");
  assert_native_y_luma::<Yuv440F, _>(Lanczos3, 4, 4, 7, 7, "yuv440p lanczos3 up");
}

// ---- Cross-frame stream reset (feature-independent) -------------------

/// A reused filter sink must reset its filter streams each frame, else
/// frame 2 row 0 is rejected as out-of-sequence (the filter twin of the
/// area `*_reuses_luma_stream_across_frames` coverage). Drives two frames
/// through one `Yuv444p` filter sink and asserts frame 2's `luma` is the
/// single-channel native-Y filter of frame 2's Y — confirming the
/// `begin_frame` reset of `luma_filter_stream` / `rgb_filter_stream`.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_filter_reuses_streams_across_frames() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y1, u, v) = planar_ramp::<Yuv444F>(SW, SH);
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let (sw, sh) = (SW as u32, SH as u32);
  let frame1 = Yuv444pFrame::new(&y1, &u, &v, sw, sh, sw, sw, sw);
  let frame2 = Yuv444pFrame::new(&y2, &u, &v, sw, sh, sw, sw, sw);
  let mut luma = vec![0u8; OW * OH];
  {
    let mut sink = MixedSinker::<Yuv444p, FilteredResampler<Triangle>>::with_resampler(
      SW,
      SH,
      FilteredResampler::new(OW, OH, Triangle),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    yuv444p_to(&frame1, FR, M, &mut sink).unwrap();
    yuv444p_to(&frame2, FR, M, &mut sink).unwrap();
  }
  let want = native_y_filter(Triangle, &y2, SW, SH, OW, OH);
  assert_eq!(
    luma, want,
    "frame 2 luma must be the native-Y filter of frame 2's Y (streams reset each frame)"
  );
}

// ---- Filter-plan-accepted regression (feature-independent) ------------

/// A filter plan must be accepted by the planar 8-bit YUV sink — before
/// this routing it was rejected with `UnsupportedFilter`; now it produces a
/// real (non-sentinel) output. Feature-independent, so it guards the
/// `yuv-planar`-solo build.
fn assert_filter_plan_accepted<F: PlanarYuvFilter>(ctx: &str) {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, u, v) = planar_ramp::<F>(SW, SH);
  // A filter plan no longer raises `UnsupportedFilter`; the resampled
  // outputs are populated (the rgb colour and the luma are non-zero for
  // this ramp).
  let got = F::filter_outputs(&y, &u, &v, SW, SH, OW, OH, Triangle);
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
fn yuv_planar_8bit_filter_plans_are_accepted() {
  assert_filter_plan_accepted::<Yuv410F>("yuv410p");
  assert_filter_plan_accepted::<Yuv420F>("yuv420p");
  assert_filter_plan_accepted::<Yuv422F>("yuv422p");
  assert_filter_plan_accepted::<Yuv444F>("yuv444p");
  assert_filter_plan_accepted::<Yuv440F>("yuv440p");
}

// ---- Packed-RGB equivalence oracle (gated on `rgb`) -------------------
//
// The filter path converts the YUV to RGB with the same `*_to_rgb_row`
// kernel the direct sink uses, then filters the RGB. So a planar filter
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
      rgb24_to(&src, FR, M, &mut sink).unwrap();
    }
    out
  }

  /// Asserts a format's filter colour outputs equal the equivalent `Rgb24`
  /// filter of the YUV->RGB-converted source pixels: `rgb` == `Rgb24`
  /// filter, `rgba` == that RGB with opaque alpha. Returns the max
  /// per-channel `rgb` diff (0 — same engine, same converted pixels).
  fn assert_color_equals_rgb24<F: PlanarYuvFilter, K: FilterKernel + Copy>(
    kernel: K,
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    ctx: &str,
  ) -> u8 {
    let (y, u, v) = planar_ramp::<F>(sw, sh);
    let got = F::filter_outputs(&y, &u, &v, sw, sh, ow, oh, kernel);

    let src_rgb = F::direct_rgb_u8(&y, &u, &v, sw, sh);
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
    yuv410p_downscale_color_filter_equals_rgb24,
    yuv410p_upscale_color_filter_equals_rgb24,
    Yuv410F,
    "yuv410p"
  );
  color_equiv_tests!(
    yuv420p_downscale_color_filter_equals_rgb24,
    yuv420p_upscale_color_filter_equals_rgb24,
    Yuv420F,
    "yuv420p"
  );
  color_equiv_tests!(
    yuv422p_downscale_color_filter_equals_rgb24,
    yuv422p_upscale_color_filter_equals_rgb24,
    Yuv422F,
    "yuv422p"
  );
  color_equiv_tests!(
    yuv444p_downscale_color_filter_equals_rgb24,
    yuv444p_upscale_color_filter_equals_rgb24,
    Yuv444F,
    "yuv444p"
  );
  color_equiv_tests!(
    yuv440p_downscale_color_filter_equals_rgb24,
    yuv440p_upscale_color_filter_equals_rgb24,
    Yuv440F,
    "yuv440p"
  );
}
