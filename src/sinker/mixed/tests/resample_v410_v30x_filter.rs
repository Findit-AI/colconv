//! Separable-filter resample coverage for the high-bit packed 4:4:4 YUV
//! sources `V410` (10-bit, MSB padding, `<const BE>`) and `V30X` (10-bit,
//! LSB padding, LE-only), routed through the merged filter engine.
//!
//! Both formats route a `Filter` plan to
//! [`packed_yuv444_triple_filter_resample`](super::super::packed_yuv444_triple_filter_resample):
//! the YUV is converted to RGB with the **same** closures the area path
//! uses (`*_to_rgb_row` / `*_to_rgb_u16_row`), then the RGB is resampled
//! by the signed-coefficient filter stream (the filter twin of the area
//! bin). Luma stays native Y: the de-interleaved Y is filter-resampled at
//! native depth, never colour-derived. So:
//!
//! - **`rgb_u16` / `rgba_u16`** equal the equivalent `Rgb48` filter
//!   resample of the source converted to native-u16 RGB, clamped to the
//!   10-bit native max (the sub-16-bit clamp the area path also applies).
//! - **`rgb` / `rgba`** equal the equivalent `Rgb24` filter resample of
//!   the source converted to u8 RGB (the u8 conversion is binned
//!   independently of the u16 one — narrowing the u16 bin would diverge,
//!   the same independence the area path's uniform-gray counterexample
//!   pins).
//! - **`luma_u16`** equals a single-channel [`FilterStream<u16>`] resample
//!   of the de-interleaved native Y; `luma` is that binned Y narrowed
//!   `>> 2`.
//!
//! The `Rgb48` / `Rgb24` oracles are gated on `rgb` (the oracle source).
//! The native-range overshoot/no-wrap contract, the native-Y luma
//! equivalence, and the filter-plan-accepted regression are
//! feature-independent, so they also guard the `yuv-444-packed`-solo
//! build (where the routing exists but no packed-RGB oracle does).

use crate::{
  KernelMatrix,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{V30X, V410, v30x_to, v410_to},
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const BITS: u32 = 10;
const NATIVE_MAX: u16 = (1 << BITS) - 1; // 1023
const SHIFT: u32 = BITS - 8; // 10-bit native Y → u8

/// Packs a logical `(U, Y, V)` 10-bit plane into V410 words:
/// `(V << 20) | (Y << 10) | U` (2-bit MSB padding).
fn pack_v410(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u32> {
  (0..u.len())
    .map(|i| {
      let u = (u[i] & 0x3FF) as u32;
      let y = (y[i] & 0x3FF) as u32;
      let v = (v[i] & 0x3FF) as u32;
      (v << 20) | (y << 10) | u
    })
    .collect()
}

/// Packs a logical `(U, Y, V)` 10-bit plane into V30X words:
/// `(V << 22) | (Y << 12) | (U << 2)` (2-bit LSB padding).
fn pack_v30x(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u32> {
  (0..u.len())
    .map(|i| {
      let u = (u[i] & 0x3FF) as u32;
      let y = (y[i] & 0x3FF) as u32;
      let v = (v[i] & 0x3FF) as u32;
      (v << 22) | (y << 12) | (u << 2)
    })
    .collect()
}

/// A per-channel 10-bit `(U, Y, V)` ramp varying per pixel so every
/// filter window sees distinct neighbours (a channel mix-up or a
/// row/column transpose diverges immediately). All samples interior so
/// the conversions see real math and the wide accumulator carries low
/// bits a u8 path would drop.
fn yuv_ramp(n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let mut u = vec![0u16; n];
  let mut y = vec![0u16; n];
  let mut v = vec![0u16; n];
  for i in 0..n {
    y[i] = (180 + (i as u32) * 11).min(1000) as u16;
    u[i] = (320 + (i as u32) * 6).min(1000) as u16;
    v[i] = 820u16.saturating_sub((i as u16) * 5);
  }
  (u, y, v)
}

/// Re-encode a host-native u16 slice as LE-wire byte storage so an
/// `Rgb48` / `Rgb24`-equivalent fixture reads back identically on LE
/// (no-op) and BE (byte-swap) hosts.
#[cfg(feature = "rgb")]
fn as_le_wire(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

// ---- Per-format hooks (the two packings + their walkers) --------------

/// The bits a filter test needs to drive one packed 4:4:4 YUV format: how
/// to pack a `(U, Y, V)` plane, and the full-res direct conversions that
/// produce the exact RGB / Y rows the filter path consumes.
trait Yuv444Filter {
  /// Pack a logical `(U, Y, V)` 10-bit plane into the format's words.
  fn pack(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u32>;

  /// Run the format's filter sink over `packed` (`sw x sh`) at `ow x oh`
  /// under `kernel`, attaching every output the equivalence asserts on.
  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u32],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs;

  /// Direct full-res native-u16 RGB conversion of `packed` (`w x h`) —
  /// the exact source-width u16 RGB the filter path bins, so it is the
  /// `Rgb48` oracle's input. Only the `rgb`-gated equivalence module
  /// consumes it, so it is dead in a `yuv-444-packed`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  fn direct_rgb_u16(packed: &[u32], w: usize, h: usize) -> Vec<u16>;

  /// Direct full-res u8 RGB conversion of `packed` (`w x h`) — the exact
  /// source-width u8 RGB the filter path bins, so it is the `Rgb24`
  /// oracle's input. Only the `rgb`-gated equivalence module consumes it,
  /// so it is dead in a `yuv-444-packed`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  fn direct_rgb_u8(packed: &[u32], w: usize, h: usize) -> Vec<u8>;

  /// Direct full-res native Y of `packed` (`w x h`) — the exact
  /// de-interleaved Y plane the filter path bins, so it is the
  /// single-channel luma oracle's input.
  fn direct_luma_u16(packed: &[u32], w: usize, h: usize) -> Vec<u16>;
}

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  /// Only the `rgb`-gated equivalence module reads the u8 colour output,
  /// so it is dead in a `yuv-444-packed`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgb: Vec<u8>,
  rgb_u16: Vec<u16>,
  rgba_u16: Vec<u16>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

struct V410F;
struct V30XF;

impl Yuv444Filter for V410F {
  fn pack(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u32> {
    pack_v410(u, y, v)
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u32],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = crate::frame::V410Frame::new(packed, sw as u32, sh as u32, sw as u32);
    let mut rgb = vec![0u8; ow * oh * 3];
    let mut rgb_u16 = vec![0u16; ow * oh * 3];
    let mut rgba_u16 = vec![0u16; ow * oh * 4];
    let mut luma = vec![0u8; ow * oh];
    let mut luma_u16 = vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<V410, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
      v410_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    FilterOutputs {
      rgb,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
    }
  }

  fn direct_rgb_u16(packed: &[u32], w: usize, h: usize) -> Vec<u16> {
    let src = crate::frame::V410Frame::new(packed, w as u32, h as u32, w as u32);
    let mut rgb_u16 = vec![0u16; w * h * 3];
    {
      let mut sink = MixedSinker::<V410>::new(w, h)
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
      v410_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    rgb_u16
  }

  fn direct_rgb_u8(packed: &[u32], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::V410Frame::new(packed, w as u32, h as u32, w as u32);
    let mut rgb = vec![0u8; w * h * 3];
    {
      let mut sink = MixedSinker::<V410>::new(w, h).with_rgb(&mut rgb).unwrap();
      v410_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    rgb
  }

  fn direct_luma_u16(packed: &[u32], w: usize, h: usize) -> Vec<u16> {
    let src = crate::frame::V410Frame::new(packed, w as u32, h as u32, w as u32);
    let mut y = vec![0u16; w * h];
    {
      let mut sink = MixedSinker::<V410>::new(w, h)
        .with_luma_u16(&mut y)
        .unwrap();
      v410_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    y
  }
}

impl Yuv444Filter for V30XF {
  fn pack(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u32> {
    pack_v30x(u, y, v)
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u32],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = crate::frame::V30XFrame::new(packed, sw as u32, sh as u32, sw as u32);
    let mut rgb = vec![0u8; ow * oh * 3];
    let mut rgb_u16 = vec![0u16; ow * oh * 3];
    let mut rgba_u16 = vec![0u16; ow * oh * 4];
    let mut luma = vec![0u8; ow * oh];
    let mut luma_u16 = vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<V30X, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
      v30x_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    FilterOutputs {
      rgb,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
    }
  }

  fn direct_rgb_u16(packed: &[u32], w: usize, h: usize) -> Vec<u16> {
    let src = crate::frame::V30XFrame::new(packed, w as u32, h as u32, w as u32);
    let mut rgb_u16 = vec![0u16; w * h * 3];
    {
      let mut sink = MixedSinker::<V30X>::new(w, h)
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
      v30x_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    rgb_u16
  }

  fn direct_rgb_u8(packed: &[u32], w: usize, h: usize) -> Vec<u8> {
    let src = crate::frame::V30XFrame::new(packed, w as u32, h as u32, w as u32);
    let mut rgb = vec![0u8; w * h * 3];
    {
      let mut sink = MixedSinker::<V30X>::new(w, h).with_rgb(&mut rgb).unwrap();
      v30x_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    rgb
  }

  fn direct_luma_u16(packed: &[u32], w: usize, h: usize) -> Vec<u16> {
    let src = crate::frame::V30XFrame::new(packed, w as u32, h as u32, w as u32);
    let mut y = vec![0u16; w * h];
    {
      let mut sink = MixedSinker::<V30X>::new(w, h)
        .with_luma_u16(&mut y)
        .unwrap();
      v30x_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    y
  }
}

// ---- Single-channel native-Y luma oracle (feature-independent) --------

/// Single-channel filter resample of a native-u16 Y plane via the merged
/// engine's [`FilterStream<u16>`] (channels = 1) — the luma oracle. The
/// 4:4:4 filter path's `luma_u16` must equal this **byte-for-bit** (same
/// engine, same coefficients, the de-interleaved native Y resampled at
/// native depth).
fn native_y_filter<K: FilterKernel>(
  kernel: K,
  y: &[u16],
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
      .feed_row(row, &y[row * sw..(row + 1) * sw], true, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// Asserts a format's filter `luma_u16` equals the single-channel native-Y
/// oracle **clamped to the 10-bit native max**, and `luma` is that clamped
/// binned Y narrowed `>> 2`. The raw [`FilterStream<u16>`] finalizes to the
/// full `u16` range, so a signed kernel can overshoot a legal 10-bit edge;
/// the 4:4:4 path clips the binned native Y to 1023 before publishing it,
/// so the oracle clamps too (`min(.., NATIVE_MAX)`). Returns the max
/// per-sample `luma_u16` diff (exactly 0 — same engine, clamp on both).
fn assert_native_y_luma<F: Yuv444Filter, K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u16 {
  let (u, y, v) = yuv_ramp(sw * sh);
  let packed = F::pack(&u, &y, &v);
  let got = F::filter_outputs(&packed, sw, sh, ow, oh, kernel);
  let native_y = F::direct_luma_u16(&packed, sw, sh);
  let raw = native_y_filter(kernel, &native_y, sw, sh, ow, oh);

  let mut max_diff = 0u16;
  for (i, (&g, &r)) in got.luma_u16.iter().zip(raw.iter()).enumerate() {
    let w = r.min(NATIVE_MAX);
    max_diff = max_diff.max(g.abs_diff(w));
    assert_eq!(
      g, w,
      "{ctx} luma_u16[{i}]: {g} vs clamped single-channel native-Y filter {w} (raw {r})"
    );
  }
  for (&hi, &lo) in got.luma_u16.iter().zip(got.luma.iter()) {
    assert_eq!(
      lo,
      (hi.min(NATIVE_MAX) >> SHIFT) as u8,
      "{ctx} luma: must be the clamped binned native Y narrowed >> 2"
    );
  }
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_luma_filter_is_single_channel_native_y() {
  // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma_u16 must be
  // the native-Y single-channel filter (max diff 0), luma its >> 2.
  assert_native_y_luma::<V410F, _>(Triangle, 8, 8, 4, 4, "v410 triangle down");
  assert_native_y_luma::<V410F, _>(CatmullRom, 8, 8, 4, 4, "v410 catmullrom down");
  assert_native_y_luma::<V410F, _>(Lanczos3, 8, 8, 4, 4, "v410 lanczos3 down");
  assert_native_y_luma::<V410F, _>(Triangle, 4, 4, 7, 7, "v410 triangle up");
  assert_native_y_luma::<V410F, _>(CatmullRom, 4, 4, 7, 7, "v410 catmullrom up");
  assert_native_y_luma::<V410F, _>(Lanczos3, 4, 4, 7, 7, "v410 lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<V30XF, _>(Triangle, 8, 8, 4, 4, "v30x triangle down");
  assert_native_y_luma::<V30XF, _>(CatmullRom, 8, 8, 4, 4, "v30x catmullrom down");
  assert_native_y_luma::<V30XF, _>(Lanczos3, 8, 8, 4, 4, "v30x lanczos3 down");
  assert_native_y_luma::<V30XF, _>(Triangle, 4, 4, 7, 7, "v30x triangle up");
  assert_native_y_luma::<V30XF, _>(CatmullRom, 4, 4, 7, 7, "v30x catmullrom up");
  assert_native_y_luma::<V30XF, _>(Lanczos3, 4, 4, 7, 7, "v30x lanczos3 up");
}

// ---- Native-range clamp / no-wrap (feature-independent) ---------------
//
// A `CatmullRom` / `Lanczos3` negative lobe overshoots a near-max colour
// edge, so a finalized binned colour sample can exceed the 10-bit native
// max even though the `FilterStream` only clamps to the full `u16` range.
// `packed_rgb_u16_resample_emit::<10, false>` clips every colour sample to
// 1023 before any native-u16 output is published: the native `rgb_u16` /
// `rgba_u16` must stay `<= 1023` (no value wraps above the documented
// 10-bit range). Feature-independent — no `Rgb48` oracle — so it also
// guards the `yuv-444-packed`-solo build; the `rgb`-gated
// `clamp_is_load_bearing` test below proves the clamp clips a *real*
// overshoot rather than passing vacuously. (The u8 `rgb` output is a
// separate filtered stream — the u8 RGB conversion is binned
// independently of the u16 one — so it does NOT equal `rgb_u16 >> 2` here;
// its own clamp is the [0, 255] of `packed_rgb_resample_emit`.)

/// A sharp white -> black horizontal step (left half max-Y neutral chroma
/// → RGB pinned at the 10-bit ceiling, right half min-Y → RGB near 0),
/// uniform vertically. A signed kernel enlarging this overshoots above the
/// native max on the bright side of the step.
fn step_edge_yuv(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let (mut u, mut y, mut v) = (vec![0u16; w * h], vec![0u16; w * h], vec![0u16; w * h]);
  for i in 0..w * h {
    let x = i % w;
    if x >= w / 2 {
      // White: max luma, neutral chroma (512) → RGB pinned at 1023.
      y[i] = 1023;
      u[i] = 512;
      v[i] = 512;
    } else {
      // Black: min luma, neutral chroma → RGB near 0.
      y[i] = 0;
      u[i] = 512;
      v[i] = 512;
    }
  }
  (u, y, v)
}

fn assert_color_clamped_to_native_max<F: Yuv444Filter, K: FilterKernel + Copy>(
  kernel: K,
  ctx: &str,
) {
  // 4 -> 7 enlargement of the white/black step (the prompt's 4->7 case).
  const SW: usize = 4;
  const SD: usize = 7;
  let (u, y, v) = step_edge_yuv(SW, SW);
  let packed = F::pack(&u, &y, &v);
  let got = F::filter_outputs(&packed, SW, SW, SD, SD, kernel);

  // (a) Every native-depth colour sample is within the 10-bit native
  //     range, and the opaque alpha is the native max.
  assert!(
    got.rgb_u16.iter().all(|&v| v <= NATIVE_MAX),
    "{ctx}: rgb_u16 must stay <= {NATIVE_MAX}; max was {}",
    got.rgb_u16.iter().copied().max().unwrap()
  );
  for px in got.rgba_u16.chunks_exact(4) {
    assert!(
      px[..3].iter().all(|&v| v <= NATIVE_MAX),
      "{ctx}: rgba_u16 colour must stay <= {NATIVE_MAX}; px = {px:?}"
    );
    assert_eq!(px[3], NATIVE_MAX, "{ctx}: opaque alpha is the native max");
  }

  // The bright plateau pins RGB at the ceiling, so a clipped-high edge
  // (a sample == native max) must exist — otherwise the overshoot the
  // clamp targets is not being exercised.
  assert!(
    got.rgb_u16.contains(&NATIVE_MAX),
    "{ctx}: expected a clipped-high (== {NATIVE_MAX}) edge in rgb_u16"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_catmullrom_color_overshoot_is_clamped_to_native_max() {
  assert_color_clamped_to_native_max::<V410F, _>(CatmullRom, "v410 catmullrom");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_lanczos3_color_overshoot_is_clamped_to_native_max() {
  assert_color_clamped_to_native_max::<V410F, _>(Lanczos3, "v410 lanczos3");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_catmullrom_color_overshoot_is_clamped_to_native_max() {
  assert_color_clamped_to_native_max::<V30XF, _>(CatmullRom, "v30x catmullrom");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_lanczos3_color_overshoot_is_clamped_to_native_max() {
  assert_color_clamped_to_native_max::<V30XF, _>(Lanczos3, "v30x lanczos3");
}

// ---- Native-Y luma overshoot / no-wrap (feature-independent) -----------
//
// The native-Y luma path (de-interleaved Y → 1-channel `FilterStream<u16>`)
// bypasses the colour helper's clamp: the stream finalizes to the full
// `u16` range, so a `CatmullRom` / `Lanczos3` negative lobe overshoots the
// near-max bright Y edge above the 10-bit native max. Before the fix the
// path copied that raw binned Y straight to `luma_u16` (publishing > 1023)
// and narrowed it `>> 2` to `luma` (wrapping a clipped-high edge to a small
// value instead of `255`). `packed_yuv444_triple_feed_emit` now clamps the
// binned Y to 1023 first, so `luma_u16` stays `<= 1023` and a clipped-high
// Y edge gives `luma_u16 == 1023` / `luma == 255` (no wrap). The bright
// plateau of `step_edge_yuv` pins Y at the ceiling, so such an edge must
// exist — without the clamp these asserts FAIL (the raw binned Y exceeds
// 1023 and its `>> 2` is < 255). Feature-independent — no `Rgb48` oracle —
// so it also guards the `yuv-444-packed`-solo build.

/// Drives `step_edge_yuv` enlarged 4 -> 7 (a near-ceiling bright Y plateau)
/// through a format's filter sink and asserts the native-Y luma stays in
/// the 10-bit range and never wraps: every `luma_u16 <= 1023`, a clipped-
/// high (`== 1023`) edge exists, and wherever `luma_u16 == 1023` the u8
/// `luma == 255`. Without the clamp the raw overshoot publishes a
/// `luma_u16 > 1023` whose `>> 2` wraps below 255 — so this discriminates.
fn assert_native_y_luma_clamped_no_wrap<F: Yuv444Filter, K: FilterKernel + Copy>(
  kernel: K,
  ctx: &str,
) {
  const SW: usize = 4;
  const SD: usize = 7;
  let (u, y, v) = step_edge_yuv(SW, SW);
  let packed = F::pack(&u, &y, &v);
  let got = F::filter_outputs(&packed, SW, SW, SD, SD, kernel);

  // (a) Every native-depth luma sample is within the 10-bit native range.
  assert!(
    got.luma_u16.iter().all(|&v| v <= NATIVE_MAX),
    "{ctx}: luma_u16 must stay <= {NATIVE_MAX}; max was {}",
    got.luma_u16.iter().copied().max().unwrap()
  );
  // (b) The bright plateau pins Y at the ceiling, so a clipped-high edge
  //     (`== NATIVE_MAX`) must exist — otherwise the overshoot the clamp
  //     targets is not exercised.
  assert!(
    got.luma_u16.contains(&NATIVE_MAX),
    "{ctx}: expected a clipped-high (== {NATIVE_MAX}) edge in luma_u16"
  );
  // (c) A clipped-high Y edge narrows to 255 (no wrap): `1023 >> 2 == 255`.
  for (i, (&hi, &lo)) in got.luma_u16.iter().zip(got.luma.iter()).enumerate() {
    assert_eq!(
      lo,
      (hi.min(NATIVE_MAX) >> SHIFT) as u8,
      "{ctx}: luma[{i}] must be the clamped binned native Y narrowed >> 2"
    );
    if hi == NATIVE_MAX {
      assert_eq!(
        lo, 255,
        "{ctx}: a clipped-high Y edge (luma_u16 == {NATIVE_MAX}) must give luma == 255, not wrap"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_catmullrom_luma_overshoot_is_clamped_no_wrap() {
  assert_native_y_luma_clamped_no_wrap::<V410F, _>(CatmullRom, "v410 catmullrom");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_lanczos3_luma_overshoot_is_clamped_no_wrap() {
  assert_native_y_luma_clamped_no_wrap::<V410F, _>(Lanczos3, "v410 lanczos3");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_catmullrom_luma_overshoot_is_clamped_no_wrap() {
  assert_native_y_luma_clamped_no_wrap::<V30XF, _>(CatmullRom, "v30x catmullrom");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_lanczos3_luma_overshoot_is_clamped_no_wrap() {
  assert_native_y_luma_clamped_no_wrap::<V30XF, _>(Lanczos3, "v30x lanczos3");
}

// ---- Filter-plan-accepted regression (feature-independent) ------------

/// A filter plan must be accepted by the packed 4:4:4 YUV sink — before
/// this routing it was rejected with `UnsupportedFilter`; now it produces
/// a real (non-sentinel) output. Feature-independent, so it guards the
/// `yuv-444-packed`-solo build.
fn assert_filter_plan_accepted<F: Yuv444Filter>(ctx: &str) {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (u, y, v) = yuv_ramp(SW * SH);
  let packed = F::pack(&u, &y, &v);
  // A filter plan no longer raises `UnsupportedFilter`; the resampled
  // outputs are populated (the rgb_u16 colour and the luma are non-zero
  // for this ramp).
  let got = F::filter_outputs(&packed, SW, SH, OW, OH, Triangle);
  assert!(
    got.rgb_u16.iter().any(|&v| v != 0),
    "{ctx}: filter resample must populate rgb_u16 (no UnsupportedFilter)"
  );
  assert!(
    got.luma_u16.iter().any(|&v| v != 0),
    "{ctx}: filter resample must populate luma_u16 (no UnsupportedFilter)"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_filter_plan_is_accepted() {
  assert_filter_plan_accepted::<V410F>("v410");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_filter_plan_is_accepted() {
  assert_filter_plan_accepted::<V30XF>("v30x");
}

// ---- Packed-RGB equivalence oracles (gated on `rgb`) ------------------
//
// The filter path converts the YUV to RGB (u8 and native-u16) with the
// same closures the direct sink uses, then filters the RGB. So a 4:4:4
// filter colour output equals the equivalent packed-RGB filter resample
// of those exact converted pixels: `rgb_u16` == `Rgb48` filter (clamped
// to the 10-bit native max), `rgb` == `Rgb24` filter.

#[cfg(feature = "rgb")]
mod packed_rgb_equivalence {
  use super::*;
  use crate::source::{Rgb24, Rgb48, rgb24_to, rgb48_to};

  /// `Rgb48` filter resample of a host-native u16 RGB frame at `ow x oh`
  /// under `kernel`, returning the native `rgb_u16` output.
  fn rgb48_filter_rgb_u16<K: FilterKernel>(
    rgb: &[u16],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> Vec<u16> {
    let wire = as_le_wire(rgb);
    let src = crate::frame::Rgb48Frame::new(&wire, sw as u32, sh as u32, (sw * 3) as u32);
    let mut out = vec![0u16; ow * oh * 3];
    {
      let mut sink = MixedSinker::<Rgb48, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_rgb_u16(&mut out)
      .unwrap();
      rgb48_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    out
  }

  /// `Rgb24` filter resample of a u8 RGB frame at `ow x oh` under
  /// `kernel`, returning the `rgb` output.
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

  /// Asserts a format's filter colour outputs equal the equivalent
  /// packed-RGB filter of the YUV→RGB-converted source pixels. The
  /// 16-bit `Rgb48` oracle is clamped to the 10-bit native max before
  /// comparison (its unclamped overshoot is what the 4:4:4 path clips).
  /// Returns the max per-channel `rgb_u16` diff (0 — same engine, same
  /// converted pixels, clamp applied to both).
  fn assert_color_equals_packed_rgb<F: Yuv444Filter, K: FilterKernel + Copy>(
    kernel: K,
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    ctx: &str,
  ) -> u16 {
    let (u, y, v) = yuv_ramp(sw * sh);
    let packed = F::pack(&u, &y, &v);
    let got = F::filter_outputs(&packed, sw, sh, ow, oh, kernel);

    // u16 colour: == Rgb48 filter of the converted native-u16 RGB,
    // clamped to the native max.
    let src_rgb_u16 = F::direct_rgb_u16(&packed, sw, sh);
    let rgb48 = rgb48_filter_rgb_u16(&src_rgb_u16, sw, sh, ow, oh, kernel);
    let want_u16: Vec<u16> = rgb48.iter().map(|&v| v.min(NATIVE_MAX)).collect();
    let mut max_diff = 0u16;
    for (i, (&g, &w)) in got.rgb_u16.iter().zip(want_u16.iter()).enumerate() {
      max_diff = max_diff.max(g.abs_diff(w));
      assert_eq!(g, w, "{ctx} rgb_u16[{i}]: {g} vs clamped Rgb48 filter {w}");
    }
    // rgba_u16 colour == rgb_u16, opaque alpha == native max.
    for (px, c) in got.rgba_u16.chunks_exact(4).zip(want_u16.chunks_exact(3)) {
      assert_eq!(&px[..3], c, "{ctx} rgba_u16 colour");
      assert_eq!(px[3], NATIVE_MAX, "{ctx} rgba_u16 alpha");
    }

    // u8 colour: == Rgb24 filter of the converted u8 RGB (binned
    // independently of the u16 path).
    let src_rgb_u8 = F::direct_rgb_u8(&packed, sw, sh);
    let want_u8 = rgb24_filter_rgb(&src_rgb_u8, sw, sh, ow, oh, kernel);
    assert_eq!(got.rgb, want_u8, "{ctx} rgb (u8) == Rgb24 filter");

    max_diff
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn v410_downscale_color_filter_equals_packed_rgb() {
    assert_color_equals_packed_rgb::<V410F, _>(Triangle, 8, 8, 4, 4, "v410 triangle down");
    assert_color_equals_packed_rgb::<V410F, _>(CatmullRom, 8, 8, 4, 4, "v410 catmullrom down");
    assert_color_equals_packed_rgb::<V410F, _>(Lanczos3, 8, 8, 4, 4, "v410 lanczos3 down");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn v410_upscale_color_filter_equals_packed_rgb() {
    assert_color_equals_packed_rgb::<V410F, _>(Triangle, 4, 4, 7, 7, "v410 triangle up");
    assert_color_equals_packed_rgb::<V410F, _>(CatmullRom, 4, 4, 7, 7, "v410 catmullrom up");
    assert_color_equals_packed_rgb::<V410F, _>(Lanczos3, 4, 4, 7, 7, "v410 lanczos3 up");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn v30x_downscale_color_filter_equals_packed_rgb() {
    assert_color_equals_packed_rgb::<V30XF, _>(Triangle, 8, 8, 4, 4, "v30x triangle down");
    assert_color_equals_packed_rgb::<V30XF, _>(CatmullRom, 8, 8, 4, 4, "v30x catmullrom down");
    assert_color_equals_packed_rgb::<V30XF, _>(Lanczos3, 8, 8, 4, 4, "v30x lanczos3 down");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn v30x_upscale_color_filter_equals_packed_rgb() {
    assert_color_equals_packed_rgb::<V30XF, _>(Triangle, 4, 4, 7, 7, "v30x triangle up");
    assert_color_equals_packed_rgb::<V30XF, _>(CatmullRom, 4, 4, 7, 7, "v30x catmullrom up");
    assert_color_equals_packed_rgb::<V30XF, _>(Lanczos3, 4, 4, 7, 7, "v30x lanczos3 up");
  }

  /// Proves the native-depth clamp is *load-bearing*, not vacuous: the
  /// **unclamped** 16-bit `Rgb48` filter of the same white/black step
  /// converted RGB overshoots above the 10-bit native max, and at every
  /// position the 4:4:4 path's `rgb_u16` equals that raw filter clipped to
  /// 1023. So a real signed-kernel overshoot exists and the clamp clips it
  /// — the `rgb_u16 <= 1023` invariant in the feature-independent test is
  /// not passing by accident.
  fn assert_clamp_is_load_bearing<F: Yuv444Filter, K: FilterKernel + Copy>(kernel: K, ctx: &str) {
    const SW: usize = 4;
    const SD: usize = 7;
    let (u, y, v) = step_edge_yuv(SW, SW);
    let packed = F::pack(&u, &y, &v);
    let got = F::filter_outputs(&packed, SW, SW, SD, SD, kernel);

    // The unclamped 16-bit oracle over the SAME converted native-u16 RGB.
    let src_rgb_u16 = F::direct_rgb_u16(&packed, SW, SW);
    let raw = rgb48_filter_rgb_u16(&src_rgb_u16, SW, SW, SD, SD, kernel);

    // A real overshoot above the native max occurs in the unclamped path.
    assert!(
      raw.iter().any(|&v| v > NATIVE_MAX),
      "{ctx}: the unclamped Rgb48 filter never overshoots {NATIVE_MAX} — the clamp test is vacuous"
    );
    // The 4:4:4 path is exactly that raw filter clipped to the native max.
    for (i, (&g, &r)) in got.rgb_u16.iter().zip(raw.iter()).enumerate() {
      assert_eq!(
        g,
        r.min(NATIVE_MAX),
        "{ctx} rgb_u16[{i}]: {g} vs clamped unclamped-oracle {} (raw {r})",
        r.min(NATIVE_MAX)
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn v410_clamp_is_load_bearing() {
    assert_clamp_is_load_bearing::<V410F, _>(CatmullRom, "v410 catmullrom");
    assert_clamp_is_load_bearing::<V410F, _>(Lanczos3, "v410 lanczos3");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn v30x_clamp_is_load_bearing() {
    assert_clamp_is_load_bearing::<V30XF, _>(CatmullRom, "v30x catmullrom");
    assert_clamp_is_load_bearing::<V30XF, _>(Lanczos3, "v30x lanczos3");
  }
}
