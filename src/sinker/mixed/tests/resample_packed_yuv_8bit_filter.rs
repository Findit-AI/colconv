//! Separable-filter resample coverage for the 8-bit packed YUV sources —
//! `Yuyv422` (YUY2), `Uyvy422` (UYVY), `Yvyu422` (YVYU) (4:2:2) and
//! `Uyyvyy411` (4:1:1), routed through the merged filter engine.
//!
//! Each format routes a `Filter` plan to
//! [`packed_yuv422_dual_filter_resample`](super::super::packed_yuv422_dual_filter_resample):
//! the YUV is converted to a source-width u8 RGB row with the **same**
//! `*_to_rgb_row` closure the area path uses (chroma de-interleave +
//! horizontal upsample in-register), then the RGB is resampled by the
//! signed-coefficient `FilterStream<u8>` (the filter twin of the area bin).
//! Luma stays native Y: the de-interleaved Y bytes are filter-resampled by a
//! 1-channel `FilterStream<u8>`, never colour-derived. So:
//!
//! - **`rgb`** equals the equivalent `Rgb24` filter resample of the source
//!   converted to u8 RGB (max diff 0 — same engine, same converted pixels).
//! - **`rgba`** equals that `rgb` with a constant `0xFF` alpha pad.
//! - **`luma`** equals a single-channel [`FilterStream<u8>`] resample of the
//!   de-interleaved native Y; **`luma_u16`** is that binned Y zero-extended.
//!
//! There is no native-depth clamp — these are 8-bit sources, so the source's
//! native range *is* the full `u8` range and the stream's own `clip8` (a
//! signed-kernel overshoot clamped to `[0, 255]`) keeps every binned sample
//! in range. The `Rgb24` oracle is gated on `rgb` (the oracle source); the
//! native-Y luma equivalence and the filter-plan-accepted regression are
//! feature-independent, so they also guard the `yuv-packed`-solo build (where
//! the routing exists but no packed-RGB oracle does).

use crate::{
  KernelMatrix,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{
    Uyvy422, Uyyvyy411, Yuyv422, Yvyu422, uyvy422_to, uyyvyy411_to, yuyv422_to, yvyu422_to,
  },
};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// A per-pixel `(Y, U, V)` ramp varying per pixel so every filter window
/// sees distinct neighbours (a channel mix-up or a row/column transpose
/// diverges immediately). Chroma is sampled per chroma column (4:2:2 uses
/// `w / 2` columns, 4:1:1 uses `w / 4`); the per-format builders index it
/// at their own subsampling. Returns the Y plane (`w x h`) plus a `(u, v)`
/// pair indexed `[row * cols + col]` at the format's chroma column count.
fn yuv_ramp(w: usize, h: usize, cols: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = vec![0u8; w * h];
  let mut u = vec![0u8; cols * h];
  let mut v = vec![0u8; cols * h];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i as u32) * 3).min(235) as u8;
  }
  for row in 0..h {
    for cx in 0..cols {
      u[row * cols + cx] = (70 + (cx as u32) * 9 + (row as u32) * 2).min(240) as u8;
      v[row * cols + cx] = 200u8
        .saturating_sub((cx as u8) * 7)
        .saturating_sub(row as u8);
    }
  }
  (y, u, v)
}

// ---- Per-format hooks (the four packings + their walkers) --------------

/// The bits a filter test needs to drive one packed 8-bit YUV format: its
/// chroma column count, how to pack a `(Y, U, V)` plane, and the full-res
/// direct conversions that produce the exact RGB / Y rows the filter path
/// consumes.
trait PackedYuv8Filter {
  /// Number of chroma columns per row (`w / 2` for 4:2:2, `w / 4` for
  /// 4:1:1) — drives the shared ramp's chroma indexing.
  fn chroma_cols(w: usize) -> usize;

  /// Pack a logical `(Y, U, V)` plane into the format's bytes.
  fn pack(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8>;

  /// Run the format's filter sink over `packed` (`sw x sh`) at `ow x oh`
  /// under `kernel`, attaching every output the equivalence asserts on.
  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs;

  /// Direct full-res u8 RGB conversion of `packed` (`w x h`) — the exact
  /// source-width u8 RGB the filter path bins, so it is the `Rgb24`
  /// oracle's input. Only the `rgb`-gated equivalence module consumes it,
  /// so it is dead in a `yuv-packed`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  fn direct_rgb_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8>;

  /// Direct full-res native Y of `packed` (`w x h`) — the exact
  /// de-interleaved Y plane the filter path bins, so it is the
  /// single-channel luma oracle's input.
  fn direct_luma_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8>;

  /// Walks `packed` (`sw x sh`) through a **no-output** filter sink (a
  /// `Filter` plan to `ow x oh`) and returns `(luma_filter_allocated,
  /// rgb_filter_allocated)` — both must be `false` (a no-output call is a
  /// no-op that allocates no stream).
  fn no_output_streams_allocated(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
  ) -> (bool, bool);
}

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  rgb: Vec<u8>,
  rgba: Vec<u8>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

struct Yuyv;
struct Uyvy;
struct Yvyu;
struct Uyyvyy;

impl PackedYuv8Filter for Yuyv {
  fn chroma_cols(w: usize) -> usize {
    w / 2
  }

  fn pack(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
    // `Y0, U0, Y1, V0` per 2-pixel pair.
    let cols = w / 2;
    let mut buf = vec![0u8; 2 * w * h];
    for row in 0..h {
      for cx in 0..cols {
        let base = row * 2 * w + cx * 4;
        buf[base] = y[row * w + cx * 2];
        buf[base + 1] = u[row * cols + cx];
        buf[base + 2] = y[row * w + cx * 2 + 1];
        buf[base + 3] = v[row * cols + cx];
      }
    }
    buf
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = crate::frame::Yuyv422Frame::new(packed, sw as u32, sh as u32, (2 * sw) as u32);
    let (mut rgb, mut rgba) = (vec![0u8; ow * oh * 3], vec![0u8; ow * oh * 4]);
    let (mut luma, mut luma_u16) = (vec![0u8; ow * oh], vec![0u16; ow * oh]);
    {
      let mut sink = MixedSinker::<Yuyv422, FilteredResampler<K>>::with_resampler(
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
      yuyv422_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  fn direct_rgb_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::Yuyv422Frame::new(packed, w as u32, h as u32, (2 * w) as u32);
    let mut rgb = vec![0u8; w * h * 3];
    {
      let mut sink = MixedSinker::<Yuyv422>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap();
      yuyv422_to(&src, FR, M, &mut sink).unwrap();
    }
    rgb
  }

  fn direct_luma_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::Yuyv422Frame::new(packed, w as u32, h as u32, (2 * w) as u32);
    let mut y = vec![0u8; w * h];
    {
      let mut sink = MixedSinker::<Yuyv422>::new(w, h).with_luma(&mut y).unwrap();
      yuyv422_to(&src, FR, M, &mut sink).unwrap();
    }
    y
  }

  fn no_output_streams_allocated(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
  ) -> (bool, bool) {
    let src = crate::frame::Yuyv422Frame::new(packed, sw as u32, sh as u32, (2 * sw) as u32);
    let mut sink = MixedSinker::<Yuyv422, FilteredResampler<Triangle>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, Triangle),
    )
    .unwrap();
    yuyv422_to(&src, FR, M, &mut sink).unwrap();
    (
      sink.luma_filter_stream_allocated(),
      sink.rgb_filter_stream_allocated(),
    )
  }
}

impl PackedYuv8Filter for Uyvy {
  fn chroma_cols(w: usize) -> usize {
    w / 2
  }

  fn pack(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
    // `U0, Y0, V0, Y1` per 2-pixel pair.
    let cols = w / 2;
    let mut buf = vec![0u8; 2 * w * h];
    for row in 0..h {
      for cx in 0..cols {
        let base = row * 2 * w + cx * 4;
        buf[base] = u[row * cols + cx];
        buf[base + 1] = y[row * w + cx * 2];
        buf[base + 2] = v[row * cols + cx];
        buf[base + 3] = y[row * w + cx * 2 + 1];
      }
    }
    buf
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = crate::frame::Uyvy422Frame::new(packed, sw as u32, sh as u32, (2 * sw) as u32);
    let (mut rgb, mut rgba) = (vec![0u8; ow * oh * 3], vec![0u8; ow * oh * 4]);
    let (mut luma, mut luma_u16) = (vec![0u8; ow * oh], vec![0u16; ow * oh]);
    {
      let mut sink = MixedSinker::<Uyvy422, FilteredResampler<K>>::with_resampler(
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
      uyvy422_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  fn direct_rgb_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::Uyvy422Frame::new(packed, w as u32, h as u32, (2 * w) as u32);
    let mut rgb = vec![0u8; w * h * 3];
    {
      let mut sink = MixedSinker::<Uyvy422>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap();
      uyvy422_to(&src, FR, M, &mut sink).unwrap();
    }
    rgb
  }

  fn direct_luma_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::Uyvy422Frame::new(packed, w as u32, h as u32, (2 * w) as u32);
    let mut y = vec![0u8; w * h];
    {
      let mut sink = MixedSinker::<Uyvy422>::new(w, h).with_luma(&mut y).unwrap();
      uyvy422_to(&src, FR, M, &mut sink).unwrap();
    }
    y
  }

  fn no_output_streams_allocated(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
  ) -> (bool, bool) {
    let src = crate::frame::Uyvy422Frame::new(packed, sw as u32, sh as u32, (2 * sw) as u32);
    let mut sink = MixedSinker::<Uyvy422, FilteredResampler<Triangle>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, Triangle),
    )
    .unwrap();
    uyvy422_to(&src, FR, M, &mut sink).unwrap();
    (
      sink.luma_filter_stream_allocated(),
      sink.rgb_filter_stream_allocated(),
    )
  }
}

impl PackedYuv8Filter for Yvyu {
  fn chroma_cols(w: usize) -> usize {
    w / 2
  }

  fn pack(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
    // `Y0, V0, Y1, U0` per 2-pixel pair.
    let cols = w / 2;
    let mut buf = vec![0u8; 2 * w * h];
    for row in 0..h {
      for cx in 0..cols {
        let base = row * 2 * w + cx * 4;
        buf[base] = y[row * w + cx * 2];
        buf[base + 1] = v[row * cols + cx];
        buf[base + 2] = y[row * w + cx * 2 + 1];
        buf[base + 3] = u[row * cols + cx];
      }
    }
    buf
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = crate::frame::Yvyu422Frame::new(packed, sw as u32, sh as u32, (2 * sw) as u32);
    let (mut rgb, mut rgba) = (vec![0u8; ow * oh * 3], vec![0u8; ow * oh * 4]);
    let (mut luma, mut luma_u16) = (vec![0u8; ow * oh], vec![0u16; ow * oh]);
    {
      let mut sink = MixedSinker::<Yvyu422, FilteredResampler<K>>::with_resampler(
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
      yvyu422_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  fn direct_rgb_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::Yvyu422Frame::new(packed, w as u32, h as u32, (2 * w) as u32);
    let mut rgb = vec![0u8; w * h * 3];
    {
      let mut sink = MixedSinker::<Yvyu422>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap();
      yvyu422_to(&src, FR, M, &mut sink).unwrap();
    }
    rgb
  }

  fn direct_luma_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::Yvyu422Frame::new(packed, w as u32, h as u32, (2 * w) as u32);
    let mut y = vec![0u8; w * h];
    {
      let mut sink = MixedSinker::<Yvyu422>::new(w, h).with_luma(&mut y).unwrap();
      yvyu422_to(&src, FR, M, &mut sink).unwrap();
    }
    y
  }

  fn no_output_streams_allocated(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
  ) -> (bool, bool) {
    let src = crate::frame::Yvyu422Frame::new(packed, sw as u32, sh as u32, (2 * sw) as u32);
    let mut sink = MixedSinker::<Yvyu422, FilteredResampler<Triangle>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, Triangle),
    )
    .unwrap();
    yvyu422_to(&src, FR, M, &mut sink).unwrap();
    (
      sink.luma_filter_stream_allocated(),
      sink.rgb_filter_stream_allocated(),
    )
  }
}

impl PackedYuv8Filter for Uyyvyy {
  fn chroma_cols(w: usize) -> usize {
    w / 4
  }

  fn pack(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
    // `U, Y0, Y1, V, Y2, Y3` per 6-byte / 4-pixel block (12 bpp).
    let cols = w / 4;
    let mut buf = vec![0u8; w * h * 3 / 2];
    for row in 0..h {
      for cx in 0..cols {
        let base = row * (w * 3 / 2) + cx * 6;
        buf[base] = u[row * cols + cx];
        buf[base + 1] = y[row * w + cx * 4];
        buf[base + 2] = y[row * w + cx * 4 + 1];
        buf[base + 3] = v[row * cols + cx];
        buf[base + 4] = y[row * w + cx * 4 + 2];
        buf[base + 5] = y[row * w + cx * 4 + 3];
      }
    }
    buf
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = crate::frame::Uyyvyy411Frame::new(packed, sw as u32, sh as u32, (sw * 3 / 2) as u32);
    let (mut rgb, mut rgba) = (vec![0u8; ow * oh * 3], vec![0u8; ow * oh * 4]);
    let (mut luma, mut luma_u16) = (vec![0u8; ow * oh], vec![0u16; ow * oh]);
    {
      let mut sink = MixedSinker::<Uyyvyy411, FilteredResampler<K>>::with_resampler(
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
      uyyvyy411_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  fn direct_rgb_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::Uyyvyy411Frame::new(packed, w as u32, h as u32, (w * 3 / 2) as u32);
    let mut rgb = vec![0u8; w * h * 3];
    {
      let mut sink = MixedSinker::<Uyyvyy411>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap();
      uyyvyy411_to(&src, FR, M, &mut sink).unwrap();
    }
    rgb
  }

  fn direct_luma_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::Uyyvyy411Frame::new(packed, w as u32, h as u32, (w * 3 / 2) as u32);
    let mut y = vec![0u8; w * h];
    {
      let mut sink = MixedSinker::<Uyyvyy411>::new(w, h)
        .with_luma(&mut y)
        .unwrap();
      uyyvyy411_to(&src, FR, M, &mut sink).unwrap();
    }
    y
  }

  fn no_output_streams_allocated(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
  ) -> (bool, bool) {
    let src = crate::frame::Uyyvyy411Frame::new(packed, sw as u32, sh as u32, (sw * 3 / 2) as u32);
    let mut sink = MixedSinker::<Uyyvyy411, FilteredResampler<Triangle>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, Triangle),
    )
    .unwrap();
    uyyvyy411_to(&src, FR, M, &mut sink).unwrap();
    (
      sink.luma_filter_stream_allocated(),
      sink.rgb_filter_stream_allocated(),
    )
  }
}

// ---- Single-channel native-Y luma oracle (feature-independent) --------

/// Single-channel filter resample of a native-u8 Y plane via the merged
/// engine's [`FilterStream<u8>`] (channels = 1) — the luma oracle. The
/// packed-YUV filter path's `luma` must equal this **byte-for-byte** (same
/// engine, same coefficients, the de-interleaved native Y resampled), and
/// `luma_u16` is it zero-extended. No native clamp: an 8-bit source's range
/// is the full `u8` range, so the stream's `clip8` is the only bound (no
/// `>> SHIFT` and no `min(native_max)` like the high-bit 4:4:4 path).
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
/// oracle, and `luma_u16` is that binned Y zero-extended. Returns the max
/// per-sample `luma` diff (exactly 0 — same engine over the same Y plane).
fn assert_native_y_luma<F: PackedYuv8Filter, K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u8 {
  let (y, u, v) = yuv_ramp(sw, sh, F::chroma_cols(sw));
  let packed = F::pack(&y, &u, &v, sw, sh);
  let got = F::filter_outputs(&packed, sw, sh, ow, oh, kernel);
  let native_y = F::direct_luma_u8(&packed, sw, sh);
  let want = native_y_filter(kernel, &native_y, sw, sh, ow, oh);

  let mut max_diff = 0u8;
  for (i, (&g, &w)) in got.luma.iter().zip(want.iter()).enumerate() {
    max_diff = max_diff.max(g.abs_diff(w));
    assert_eq!(
      g, w,
      "{ctx} luma[{i}]: {g} vs single-channel native-Y filter {w}"
    );
  }
  for (i, (&hi, &lo)) in got.luma_u16.iter().zip(got.luma.iter()).enumerate() {
    assert_eq!(
      hi, lo as u16,
      "{ctx} luma_u16[{i}]: must be the binned native Y zero-extended"
    );
  }
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_luma_filter_is_single_channel_native_y() {
  // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma must be the
  // native-Y single-channel filter (max diff 0), luma_u16 its zero-extension.
  assert_native_y_luma::<Yuyv, _>(Triangle, 8, 8, 4, 4, "yuyv triangle down");
  assert_native_y_luma::<Yuyv, _>(CatmullRom, 8, 8, 4, 4, "yuyv catmullrom down");
  assert_native_y_luma::<Yuyv, _>(Lanczos3, 8, 8, 4, 4, "yuyv lanczos3 down");
  assert_native_y_luma::<Yuyv, _>(Triangle, 4, 4, 7, 7, "yuyv triangle up");
  assert_native_y_luma::<Yuyv, _>(CatmullRom, 4, 4, 7, 7, "yuyv catmullrom up");
  assert_native_y_luma::<Yuyv, _>(Lanczos3, 4, 4, 7, 7, "yuyv lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyvy422_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Uyvy, _>(Triangle, 8, 8, 4, 4, "uyvy triangle down");
  assert_native_y_luma::<Uyvy, _>(CatmullRom, 8, 8, 4, 4, "uyvy catmullrom down");
  assert_native_y_luma::<Uyvy, _>(Lanczos3, 8, 8, 4, 4, "uyvy lanczos3 down");
  assert_native_y_luma::<Uyvy, _>(Triangle, 4, 4, 7, 7, "uyvy triangle up");
  assert_native_y_luma::<Uyvy, _>(CatmullRom, 4, 4, 7, 7, "uyvy catmullrom up");
  assert_native_y_luma::<Uyvy, _>(Lanczos3, 4, 4, 7, 7, "uyvy lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yvyu422_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<Yvyu, _>(Triangle, 8, 8, 4, 4, "yvyu triangle down");
  assert_native_y_luma::<Yvyu, _>(CatmullRom, 8, 8, 4, 4, "yvyu catmullrom down");
  assert_native_y_luma::<Yvyu, _>(Lanczos3, 8, 8, 4, 4, "yvyu lanczos3 down");
  assert_native_y_luma::<Yvyu, _>(Triangle, 4, 4, 7, 7, "yvyu triangle up");
  assert_native_y_luma::<Yvyu, _>(CatmullRom, 4, 4, 7, 7, "yvyu catmullrom up");
  assert_native_y_luma::<Yvyu, _>(Lanczos3, 4, 4, 7, 7, "yvyu lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_luma_filter_is_single_channel_native_y() {
  // 4:1:1 width must be a multiple of 4: 8 -> 4 and 4 -> 8 (the 4->7 case
  // is not 4-aligned at the source, so use 4->8 for the upscale leg).
  assert_native_y_luma::<Uyyvyy, _>(Triangle, 8, 8, 4, 4, "uyyvyy triangle down");
  assert_native_y_luma::<Uyyvyy, _>(CatmullRom, 8, 8, 4, 4, "uyyvyy catmullrom down");
  assert_native_y_luma::<Uyyvyy, _>(Lanczos3, 8, 8, 4, 4, "uyyvyy lanczos3 down");
  assert_native_y_luma::<Uyyvyy, _>(Triangle, 4, 4, 8, 7, "uyyvyy triangle up");
  assert_native_y_luma::<Uyyvyy, _>(CatmullRom, 4, 4, 8, 7, "uyyvyy catmullrom up");
  assert_native_y_luma::<Uyyvyy, _>(Lanczos3, 4, 4, 8, 7, "uyyvyy lanczos3 up");
}

// ---- Filter-plan-accepted regression (feature-independent) ------------

/// A filter plan must be accepted by the packed 8-bit YUV sink — before
/// this routing it was rejected with `UnsupportedFilter`; now it produces a
/// real (non-sentinel) output. Feature-independent, so it guards the
/// `yuv-packed`-solo build.
fn assert_filter_plan_accepted<F: PackedYuv8Filter>(sw: usize, ctx: &str) {
  let sh = sw;
  let (ow, oh) = (sw / 2, sw / 2);
  let (y, u, v) = yuv_ramp(sw, sh, F::chroma_cols(sw));
  let packed = F::pack(&y, &u, &v, sw, sh);
  // A filter plan no longer raises `UnsupportedFilter`; the resampled
  // outputs are populated (the rgb colour and the luma are non-zero for
  // this ramp).
  let got = F::filter_outputs(&packed, sw, sh, ow, oh, Triangle);
  assert!(
    got.rgb.iter().any(|&v| v != 0),
    "{ctx}: filter resample must populate rgb (no UnsupportedFilter)"
  );
  assert!(
    got.luma.iter().any(|&v| v != 0),
    "{ctx}: filter resample must populate luma (no UnsupportedFilter)"
  );
  // rgba is the rgb row with a constant 0xFF alpha pad (Strategy A, the
  // same derivation the area path uses) — a feature-independent
  // self-consistency check that does not need the `rgb`-gated oracle.
  for (px, rgb_px) in got.rgba.chunks_exact(4).zip(got.rgb.chunks_exact(3)) {
    assert_eq!(&px[..3], rgb_px, "{ctx}: rgba colour must equal rgb");
    assert_eq!(px[3], 0xFF, "{ctx}: rgba alpha must be 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_yuv_8bit_filter_plan_is_accepted() {
  assert_filter_plan_accepted::<Yuyv>(8, "yuyv");
  assert_filter_plan_accepted::<Uyvy>(8, "uyvy");
  assert_filter_plan_accepted::<Yvyu>(8, "yvyu");
  assert_filter_plan_accepted::<Uyyvyy>(8, "uyyvyy411");
}

// ---- No-output is a no-op (feature-independent) -----------------------

/// A no-output filter sink drives every row but allocates neither filter
/// stream and stays a no-op. Feature-independent — the white-box
/// `*_allocated()` probes are gated on `yuv-packed`, which guards this file.
fn assert_no_output_no_op<F: PackedYuv8Filter>(ctx: &str) {
  const SW: usize = 8;
  const SH: usize = 8;
  let (y, u, v) = yuv_ramp(SW, SH, F::chroma_cols(SW));
  let packed = F::pack(&y, &u, &v, SW, SH);
  let (luma_alloc, rgb_alloc) = F::no_output_streams_allocated(&packed, SW, SH, SW / 2, SH / 2);
  assert!(
    !luma_alloc,
    "{ctx}: no-output sink allocated a luma filter stream"
  );
  assert!(
    !rgb_alloc,
    "{ctx}: no-output sink allocated an rgb filter stream"
  );
}

#[test]
fn packed_yuv_8bit_filter_no_outputs_is_a_no_op() {
  assert_no_output_no_op::<Yuyv>("yuyv");
  assert_no_output_no_op::<Uyvy>("uyvy");
  assert_no_output_no_op::<Yvyu>("yvyu");
  assert_no_output_no_op::<Uyyvyy>("uyyvyy411");
}

// ---- Packed-RGB equivalence oracle (gated on `rgb`) -------------------
//
// The filter path converts the YUV to u8 RGB with the same `*_to_rgb_row`
// closure the direct sink uses, then filters the RGB. So a packed-YUV
// filter colour output equals the equivalent `Rgb24` filter resample of
// those exact converted pixels: `rgb` == `Rgb24` filter; `rgba` == that rgb
// with a 0xFF alpha pad.

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
  /// filter of the YUV→RGB-converted source pixels (max diff 0 — same
  /// engine, same converted pixels). Returns that max per-channel diff.
  fn assert_color_equals_rgb24<F: PackedYuv8Filter, K: FilterKernel + Copy>(
    kernel: K,
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    ctx: &str,
  ) -> u8 {
    let (y, u, v) = yuv_ramp(sw, sh, F::chroma_cols(sw));
    let packed = F::pack(&y, &u, &v, sw, sh);
    let got = F::filter_outputs(&packed, sw, sh, ow, oh, kernel);

    let src_rgb = F::direct_rgb_u8(&packed, sw, sh);
    let want = rgb24_filter_rgb(&src_rgb, sw, sh, ow, oh, kernel);

    let mut max_diff = 0u8;
    for (i, (&g, &w)) in got.rgb.iter().zip(want.iter()).enumerate() {
      max_diff = max_diff.max(g.abs_diff(w));
      assert_eq!(g, w, "{ctx} rgb[{i}]: {g} vs Rgb24 filter {w}");
    }
    // rgba colour == rgb, opaque alpha == 0xFF.
    for (px, c) in got.rgba.chunks_exact(4).zip(want.chunks_exact(3)) {
      assert_eq!(&px[..3], c, "{ctx} rgba colour");
      assert_eq!(px[3], 0xFF, "{ctx} rgba alpha");
    }
    max_diff
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuyv422_color_filter_equals_rgb24() {
    assert_color_equals_rgb24::<Yuyv, _>(Triangle, 8, 8, 4, 4, "yuyv triangle down");
    assert_color_equals_rgb24::<Yuyv, _>(CatmullRom, 8, 8, 4, 4, "yuyv catmullrom down");
    assert_color_equals_rgb24::<Yuyv, _>(Lanczos3, 8, 8, 4, 4, "yuyv lanczos3 down");
    assert_color_equals_rgb24::<Yuyv, _>(Triangle, 4, 4, 7, 7, "yuyv triangle up");
    assert_color_equals_rgb24::<Yuyv, _>(CatmullRom, 4, 4, 7, 7, "yuyv catmullrom up");
    assert_color_equals_rgb24::<Yuyv, _>(Lanczos3, 4, 4, 7, 7, "yuyv lanczos3 up");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn uyvy422_color_filter_equals_rgb24() {
    assert_color_equals_rgb24::<Uyvy, _>(Triangle, 8, 8, 4, 4, "uyvy triangle down");
    assert_color_equals_rgb24::<Uyvy, _>(CatmullRom, 8, 8, 4, 4, "uyvy catmullrom down");
    assert_color_equals_rgb24::<Uyvy, _>(Lanczos3, 8, 8, 4, 4, "uyvy lanczos3 down");
    assert_color_equals_rgb24::<Uyvy, _>(Triangle, 4, 4, 7, 7, "uyvy triangle up");
    assert_color_equals_rgb24::<Uyvy, _>(CatmullRom, 4, 4, 7, 7, "uyvy catmullrom up");
    assert_color_equals_rgb24::<Uyvy, _>(Lanczos3, 4, 4, 7, 7, "uyvy lanczos3 up");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yvyu422_color_filter_equals_rgb24() {
    assert_color_equals_rgb24::<Yvyu, _>(Triangle, 8, 8, 4, 4, "yvyu triangle down");
    assert_color_equals_rgb24::<Yvyu, _>(CatmullRom, 8, 8, 4, 4, "yvyu catmullrom down");
    assert_color_equals_rgb24::<Yvyu, _>(Lanczos3, 8, 8, 4, 4, "yvyu lanczos3 down");
    assert_color_equals_rgb24::<Yvyu, _>(Triangle, 4, 4, 7, 7, "yvyu triangle up");
    assert_color_equals_rgb24::<Yvyu, _>(CatmullRom, 4, 4, 7, 7, "yvyu catmullrom up");
    assert_color_equals_rgb24::<Yvyu, _>(Lanczos3, 4, 4, 7, 7, "yvyu lanczos3 up");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn uyyvyy411_color_filter_equals_rgb24() {
    // 4:1:1 source width must be a multiple of 4: 8 -> 4 down, 4 -> 8 up.
    assert_color_equals_rgb24::<Uyyvyy, _>(Triangle, 8, 8, 4, 4, "uyyvyy triangle down");
    assert_color_equals_rgb24::<Uyyvyy, _>(CatmullRom, 8, 8, 4, 4, "uyyvyy catmullrom down");
    assert_color_equals_rgb24::<Uyyvyy, _>(Lanczos3, 8, 8, 4, 4, "uyyvyy lanczos3 down");
    assert_color_equals_rgb24::<Uyyvyy, _>(Triangle, 4, 4, 8, 7, "uyyvyy triangle up");
    assert_color_equals_rgb24::<Uyyvyy, _>(CatmullRom, 4, 4, 8, 7, "uyyvyy catmullrom up");
    assert_color_equals_rgb24::<Uyyvyy, _>(Lanczos3, 4, 4, 8, 7, "uyyvyy lanczos3 up");
  }
}
