//! Separable-filter resample coverage for the 8-bit packed 4:4:4 YUV
//! sources `Vuya` (`[V, U, Y, A]`, the A byte real source alpha) and `Vuyx`
//! (`[V, U, Y, X]`, the X byte padding forced opaque), routed through the
//! merged filter engine.
//!
//! Both formats route a `Filter` plan to
//! [`packed_yuva444_filter_resample`](super::super::packed_yuva444_filter_resample):
//! the YUVA is converted to a canonical u8 `R, G, B, A` row with the **same**
//! `vuya_to_rgba_row` / `vuyx_to_rgba_row` kernel the area / direct paths use,
//! then the four interleaved channels are resampled by the signed-coefficient
//! filter stream (the filter twin of the area bin). Straight alpha only
//! (`Vuya` is straight; `Vuyx` writes a constant `0xFF` α plane that filters
//! to itself). Luma stays native Y: the de-interleaved Y is filter-resampled
//! at native depth, never colour-derived. So:
//!
//! - **`rgba` / `rgb`** equal the equivalent 8-bit `Rgba` filter resample of
//!   the source converted to u8 RGBA (alpha is a real filtered channel, NOT
//!   forced opaque — for `Vuyx` it filters a constant `0xFF` to `0xFF`).
//! - **`luma_u16`** equals a single-channel [`FilterStream<u16>`] resample of
//!   the de-interleaved native Y **clamped to the 8-bit native max** `255`
//!   (the raw stream finalizes to the full `u16` range, so a signed kernel
//!   overshoots a legal Y edge above `255`; the 4:4:4 path clips it); `luma`
//!   is that clamped binned Y (a `>> 0` no-op at 8-bit).
//!
//! `Vuya` / `Vuyx` are 8-bit and expose no u16 colour outputs, so the colour
//! native-max clamp in the shared emit is exercised only by the native-Y luma
//! here; the u8 colour outputs cannot overshoot (the `FilterStream<u8>` clamps
//! to `[0, 255]`), so they have no separate clamp test. The `Rgba` equivalence
//! oracle is gated on `rgb` (the oracle source). The native-Y luma equivalence,
//! the luma overshoot/no-wrap contract, and the filter-plan-accepted regression
//! are feature-independent, so they also guard the `yuv-444-packed`-solo build.

use crate::{
  KernelMatrix,
  frame::{VuyaFrame, VuyxFrame},
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{Vuya, Vuyx, vuya_to, vuyx_to},
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const NATIVE_MAX: u16 = 255; // 8-bit native Y / colour max
const SHIFT: u32 = 0; // 8-bit native Y → u8 is a zero-shift identity

/// A per-channel `[V, U, Y, A]` ramp varying per pixel so every filter window
/// sees distinct neighbours (a channel mix-up or a row/column transpose
/// diverges immediately). Alpha varies (not all-opaque) so the real-alpha
/// filter is genuinely exercised.
fn vuya_ramp(sw: usize, sh: usize) -> Vec<u8> {
  let n = sw * sh;
  let mut packed = std::vec![0u8; n * 4];
  for (i, px) in packed.chunks_exact_mut(4).enumerate() {
    px[0] = (40 + i * 5).min(235) as u8; // V
    px[1] = (200u32.wrapping_sub((i as u32) * 3) & 0xFF) as u8; // U
    px[2] = (24 + i * 7).min(235) as u8; // Y
    px[3] = (16 + i * 9).min(250) as u8; // A (varies)
  }
  packed
}

/// A sharp black -> white horizontal step (left half min-Y, right half max-Y,
/// neutral chroma, opaque), uniform vertically. A signed kernel enlarging the
/// near-max bright Y plateau overshoots above the 8-bit native max.
fn step_edge_vuya(w: usize, h: usize) -> Vec<u8> {
  let mut packed = std::vec![0u8; w * h * 4];
  for (i, px) in packed.chunks_exact_mut(4).enumerate() {
    let x = i % w;
    px[0] = 128; // V neutral
    px[1] = 128; // U neutral
    px[2] = if x >= w / 2 { 255 } else { 0 }; // Y: white / black plateau
    px[3] = 255; // opaque
  }
  packed
}

// ---- Per-format hooks (the two packings + their walkers) --------------

/// The bits a filter test needs to drive one packed 4:4:4 YUVA format: run the
/// format's filter sink, and the full-res direct conversions that produce the
/// exact RGBA / Y rows the filter path consumes.
trait Yuva444Filter {
  /// Run the format's filter sink over `packed` (`sw x sh`) at `ow x oh` under
  /// `kernel`, attaching every output the equivalence asserts on.
  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs;

  /// Direct full-res u8 RGBA conversion of `packed` (`w x h`) — the exact
  /// canonical source-width RGBA the filter path resamples, so it is the
  /// `Rgba` oracle's input. (`Vuya` passes source α; `Vuyx` forces `0xFF`.)
  /// Only the `rgb`-gated equivalence module consumes it, so it is dead in a
  /// `yuv-444-packed`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  fn direct_rgba_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8>;

  /// Direct full-res native Y of `packed` (`w x h`) — the exact de-interleaved
  /// Y plane the filter path resamples, so it is the single-channel luma
  /// oracle's input.
  fn direct_luma_u16(packed: &[u8], w: usize, h: usize) -> Vec<u16>;
}

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  rgb: Vec<u8>,
  rgba: Vec<u8>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

struct VuyaF;
struct VuyxF;

impl Yuva444Filter for VuyaF {
  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = VuyaFrame::try_new(packed, sw as u32, sh as u32, (sw * 4) as u32).unwrap();
    let mut rgb = std::vec![0u8; ow * oh * 3];
    let mut rgba = std::vec![0u8; ow * oh * 4];
    let mut luma = std::vec![0u8; ow * oh];
    let mut luma_u16 = std::vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<Vuya, FilteredResampler<K>>::with_resampler(
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
      vuya_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  fn direct_rgba_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = VuyaFrame::try_new(packed, w as u32, h as u32, (w * 4) as u32).unwrap();
    let mut rgba = std::vec![0u8; w * h * 4];
    {
      let mut sink = MixedSinker::<Vuya>::new(w, h).with_rgba(&mut rgba).unwrap();
      vuya_to(&src, FR, M, &mut sink).unwrap();
    }
    rgba
  }

  fn direct_luma_u16(packed: &[u8], w: usize, h: usize) -> Vec<u16> {
    let src = VuyaFrame::try_new(packed, w as u32, h as u32, (w * 4) as u32).unwrap();
    let mut y = std::vec![0u16; w * h];
    {
      let mut sink = MixedSinker::<Vuya>::new(w, h)
        .with_luma_u16(&mut y)
        .unwrap();
      vuya_to(&src, FR, M, &mut sink).unwrap();
    }
    y
  }
}

impl Yuva444Filter for VuyxF {
  fn filter_outputs<K: FilterKernel + Copy>(
    packed: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = VuyxFrame::try_new(packed, sw as u32, sh as u32, (sw * 4) as u32).unwrap();
    let mut rgb = std::vec![0u8; ow * oh * 3];
    let mut rgba = std::vec![0u8; ow * oh * 4];
    let mut luma = std::vec![0u8; ow * oh];
    let mut luma_u16 = std::vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<Vuyx, FilteredResampler<K>>::with_resampler(
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
      vuyx_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  fn direct_rgba_u8(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = VuyxFrame::try_new(packed, w as u32, h as u32, (w * 4) as u32).unwrap();
    let mut rgba = std::vec![0u8; w * h * 4];
    {
      let mut sink = MixedSinker::<Vuyx>::new(w, h).with_rgba(&mut rgba).unwrap();
      vuyx_to(&src, FR, M, &mut sink).unwrap();
    }
    rgba
  }

  fn direct_luma_u16(packed: &[u8], w: usize, h: usize) -> Vec<u16> {
    let src = VuyxFrame::try_new(packed, w as u32, h as u32, (w * 4) as u32).unwrap();
    let mut y = std::vec![0u16; w * h];
    {
      let mut sink = MixedSinker::<Vuyx>::new(w, h)
        .with_luma_u16(&mut y)
        .unwrap();
      vuyx_to(&src, FR, M, &mut sink).unwrap();
    }
    y
  }
}

// ---- Single-channel native-Y luma oracle (feature-independent) --------

/// Single-channel filter resample of a native-u16 Y plane via the merged
/// engine's [`FilterStream<u16>`] (channels = 1) — the luma oracle. The
/// 4:4:4 filter path's `luma_u16` must equal this **clamped to the 8-bit
/// native max** (same engine, same coefficients, the de-interleaved native Y
/// resampled at native depth, then clipped to `255`).
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
  let mut out = std::vec![0u16; ow * oh];
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
/// oracle **clamped to the 8-bit native max**, and `luma` is that clamped
/// binned Y (`>> 0`). The raw [`FilterStream<u16>`] finalizes to the full
/// `u16` range, so a signed kernel can overshoot a legal 8-bit Y edge; the
/// 4:4:4 path clips the binned native Y to `255` before publishing it, so the
/// oracle clamps too (`min(.., NATIVE_MAX)`). Returns the max per-sample
/// `luma_u16` diff (exactly 0 — same engine, clamp on both).
fn assert_native_y_luma<F: Yuva444Filter, K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u16 {
  let packed = vuya_ramp(sw, sh);
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
      "{ctx} luma: must be the clamped binned native Y (>> 0)"
    );
  }
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_luma_filter_is_single_channel_native_y() {
  // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma_u16 must be the
  // clamped native-Y single-channel filter (max diff 0), luma its >> 0.
  assert_native_y_luma::<VuyaF, _>(Triangle, 8, 8, 4, 4, "vuya triangle down");
  assert_native_y_luma::<VuyaF, _>(CatmullRom, 8, 8, 4, 4, "vuya catmullrom down");
  assert_native_y_luma::<VuyaF, _>(Lanczos3, 8, 8, 4, 4, "vuya lanczos3 down");
  assert_native_y_luma::<VuyaF, _>(Triangle, 4, 4, 7, 7, "vuya triangle up");
  assert_native_y_luma::<VuyaF, _>(CatmullRom, 4, 4, 7, 7, "vuya catmullrom up");
  assert_native_y_luma::<VuyaF, _>(Lanczos3, 4, 4, 7, 7, "vuya lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_luma_filter_is_single_channel_native_y() {
  assert_native_y_luma::<VuyxF, _>(Triangle, 8, 8, 4, 4, "vuyx triangle down");
  assert_native_y_luma::<VuyxF, _>(CatmullRom, 8, 8, 4, 4, "vuyx catmullrom down");
  assert_native_y_luma::<VuyxF, _>(Lanczos3, 8, 8, 4, 4, "vuyx lanczos3 down");
  assert_native_y_luma::<VuyxF, _>(Triangle, 4, 4, 7, 7, "vuyx triangle up");
  assert_native_y_luma::<VuyxF, _>(CatmullRom, 4, 4, 7, 7, "vuyx catmullrom up");
  assert_native_y_luma::<VuyxF, _>(Lanczos3, 4, 4, 7, 7, "vuyx lanczos3 up");
}

// ---- Native-Y luma overshoot / no-wrap (feature-independent) -----------
//
// The native-Y luma path (de-interleaved Y → 1-channel `FilterStream<u16>`)
// finalizes to the full `u16` range, so a `CatmullRom` / `Lanczos3` negative
// lobe overshoots the near-max bright Y edge above the 8-bit native max.
// Before the clamp the path copied that raw binned Y straight to `luma_u16`
// (publishing > 255) and cast it `>> 0` to `luma` (wrapping a clipped-high
// edge — e.g. `260 as u8 == 4` — instead of `255`).
// `packed_yuva444_feed_emit` now clamps the binned Y to `255` first, so
// `luma_u16` stays `<= 255` and a clipped-high Y edge gives `luma_u16 == 255`
// / `luma == 255` (no wrap). The bright plateau of `step_edge_vuya` pins Y at
// the ceiling, so such an edge must exist — without the clamp these asserts
// FAIL. Feature-independent — no `Rgba` oracle — so it also guards the
// `yuv-444-packed`-solo build.

/// Drives `step_edge_vuya` enlarged 4 -> 7 (a near-ceiling bright Y plateau)
/// through a format's filter sink and asserts the native-Y luma stays in the
/// 8-bit range and never wraps: every `luma_u16 <= 255`, a clipped-high
/// (`== 255`) edge exists, and wherever `luma_u16 == 255` the u8 `luma == 255`.
fn assert_native_y_luma_clamped_no_wrap<F: Yuva444Filter, K: FilterKernel + Copy>(
  kernel: K,
  ctx: &str,
) {
  const SW: usize = 4;
  const SD: usize = 7;
  let packed = step_edge_vuya(SW, SW);
  let got = F::filter_outputs(&packed, SW, SW, SD, SD, kernel);

  // (a) Every native-depth luma sample is within the 8-bit native range.
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
  // (c) A clipped-high Y edge maps to 255 (no wrap): `255 >> 0 == 255`.
  for (i, (&hi, &lo)) in got.luma_u16.iter().zip(got.luma.iter()).enumerate() {
    assert_eq!(
      lo,
      (hi.min(NATIVE_MAX) >> SHIFT) as u8,
      "{ctx}: luma[{i}] must be the clamped binned native Y (>> 0)"
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
fn vuya_catmullrom_luma_overshoot_is_clamped_no_wrap() {
  assert_native_y_luma_clamped_no_wrap::<VuyaF, _>(CatmullRom, "vuya catmullrom");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_lanczos3_luma_overshoot_is_clamped_no_wrap() {
  assert_native_y_luma_clamped_no_wrap::<VuyaF, _>(Lanczos3, "vuya lanczos3");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_catmullrom_luma_overshoot_is_clamped_no_wrap() {
  assert_native_y_luma_clamped_no_wrap::<VuyxF, _>(CatmullRom, "vuyx catmullrom");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_lanczos3_luma_overshoot_is_clamped_no_wrap() {
  assert_native_y_luma_clamped_no_wrap::<VuyxF, _>(Lanczos3, "vuyx lanczos3");
}

/// Proves the native-Y luma clamp is *load-bearing*, not vacuous: the
/// **unclamped** single-channel `FilterStream<u16>` of the same step-edge
/// native Y overshoots above the 8-bit native max, and at every position the
/// 4:4:4 path's `luma_u16` equals that raw filter clipped to `255`. So a real
/// signed-kernel overshoot exists and the clamp clips it — the `luma_u16 <=
/// 255` / no-wrap invariants above are not passing by accident. Feature-
/// independent, so it guards the `yuv-444-packed`-solo build.
fn assert_luma_clamp_is_load_bearing<F: Yuva444Filter, K: FilterKernel + Copy>(
  kernel: K,
  ctx: &str,
) {
  const SW: usize = 4;
  const SD: usize = 7;
  let packed = step_edge_vuya(SW, SW);
  let got = F::filter_outputs(&packed, SW, SW, SD, SD, kernel);

  // The unclamped single-channel oracle over the SAME de-interleaved native Y.
  let native_y = F::direct_luma_u16(&packed, SW, SW);
  let raw = native_y_filter(kernel, &native_y, SW, SW, SD, SD);

  // A real overshoot above the native max occurs in the unclamped path.
  assert!(
    raw.iter().any(|&v| v > NATIVE_MAX),
    "{ctx}: the unclamped native-Y filter never overshoots {NATIVE_MAX} — the clamp test is vacuous"
  );
  // The 4:4:4 path is exactly that raw filter clipped to the native max, and
  // its u8 luma is the clipped value (a clipped-high edge → 255, not a wrap).
  for (i, (&g, &r)) in got.luma_u16.iter().zip(raw.iter()).enumerate() {
    assert_eq!(
      g,
      r.min(NATIVE_MAX),
      "{ctx} luma_u16[{i}]: {g} vs clamped unclamped-oracle {} (raw {r})",
      r.min(NATIVE_MAX)
    );
  }
  for (&hi, &lo) in got.luma_u16.iter().zip(got.luma.iter()) {
    assert_eq!(lo, hi as u8, "{ctx}: u8 luma == clamped binned Y (>> 0)");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_luma_clamp_is_load_bearing() {
  assert_luma_clamp_is_load_bearing::<VuyaF, _>(CatmullRom, "vuya catmullrom");
  assert_luma_clamp_is_load_bearing::<VuyaF, _>(Lanczos3, "vuya lanczos3");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_luma_clamp_is_load_bearing() {
  assert_luma_clamp_is_load_bearing::<VuyxF, _>(CatmullRom, "vuyx catmullrom");
  assert_luma_clamp_is_load_bearing::<VuyxF, _>(Lanczos3, "vuyx lanczos3");
}

// ---- u8 colour in-range (feature-independent) -------------------------
//
// `Vuya` / `Vuyx` expose no u16 colour outputs, so the only colour the filter
// produces is u8 (rgb / rgba). The `FilterStream<u8>` clamps every channel to
// `[0, 255]`, so a signed-kernel overshoot cannot wrap the u8 colour (no
// native-max clamp needed on this path — unlike the native-Y luma above). This
// pins that contract: enlarging the white/black step keeps every u8 colour
// channel — alpha too — within range, and a saturated (== 255) edge exists.

fn assert_u8_color_in_range<F: Yuva444Filter, K: FilterKernel + Copy>(kernel: K, ctx: &str) {
  const SW: usize = 4;
  const SD: usize = 7;
  let packed = step_edge_vuya(SW, SW);
  let got = F::filter_outputs(&packed, SW, SW, SD, SD, kernel);
  // The `FilterStream<u8>` clamps every channel into the `u8` range, so the
  // overshoot the native-Y luma path must clip can never wrap the u8 colour —
  // it saturates instead. The bright plateau pins colour at the ceiling, so a
  // saturated (== 255) edge must exist (the kernel really pushes against the
  // clamp).
  assert!(
    got.rgb.contains(&255),
    "{ctx}: expected a saturated (== 255) colour edge in rgb"
  );
  // Opaque-α step → every filtered α stays 255 (a constant channel filters to
  // itself, partition of unity).
  assert!(
    got.rgba.chunks_exact(4).all(|px| px[3] == 255),
    "{ctx}: opaque-α step must keep filtered α == 255"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_u8_color_in_range_at_step_edge() {
  assert_u8_color_in_range::<VuyaF, _>(CatmullRom, "vuya catmullrom");
  assert_u8_color_in_range::<VuyaF, _>(Lanczos3, "vuya lanczos3");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_u8_color_in_range_at_step_edge() {
  assert_u8_color_in_range::<VuyxF, _>(CatmullRom, "vuyx catmullrom");
  assert_u8_color_in_range::<VuyxF, _>(Lanczos3, "vuyx lanczos3");
}

// ---- Filter-plan-accepted regression (feature-independent) ------------

/// A filter plan must be accepted by the packed 4:4:4 YUVA sink — before this
/// routing it was rejected with `UnsupportedFilter`; now it produces a real
/// (non-sentinel) output. Feature-independent, so it guards the
/// `yuv-444-packed`-solo build.
fn assert_filter_plan_accepted<F: Yuva444Filter>(ctx: &str) {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let packed = vuya_ramp(SW, SH);
  // A filter plan no longer raises `UnsupportedFilter`; the resampled outputs
  // are populated (the rgba colour and the luma are non-zero for this ramp).
  let got = F::filter_outputs(&packed, SW, SH, OW, OH, Triangle);
  assert!(
    got.rgba.iter().any(|&v| v != 0),
    "{ctx}: filter resample must populate rgba (no UnsupportedFilter)"
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
fn vuya_filter_plan_is_accepted() {
  assert_filter_plan_accepted::<VuyaF>("vuya");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_filter_plan_is_accepted() {
  assert_filter_plan_accepted::<VuyxF>("vuyx");
}

// ---- Packed-RGBA (8-bit) equivalence oracle (gated on `rgb`) ----------
//
// The filter path converts the YUVA to a canonical u8 `R, G, B, A` row with
// the same `*_to_rgba_row` kernel the direct sink uses, then filters the four
// channels independently. So a 4:4:4 filter colour output equals the
// equivalent 8-bit `Rgba` filter resample of those exact converted pixels:
// `rgba` == the `Rgba` filter (per-channel, alpha a real filtered channel),
// `rgb` == its alpha drop. The `FilterStream<u8>` is byte-exact per channel,
// so the max diff is 0.

#[cfg(feature = "rgb")]
mod packed_rgba_equivalence {
  use super::*;
  use crate::source::{Rgba, rgba_to};
  use mediaframe::frame::RgbaFrame;

  /// `Rgba` (8-bit) filter resample of a canonical u8 RGBA frame at `ow x oh`
  /// under `kernel`, returning the `rgba` output (per-channel filter, no
  /// premultiplication — straight alpha).
  fn rgba_filter_rgba<K: FilterKernel>(
    rgba: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> Vec<u8> {
    let src = RgbaFrame::try_new(rgba, sw as u32, sh as u32, (sw * 4) as u32).unwrap();
    let mut out = std::vec![0u8; ow * oh * 4];
    {
      let mut sink = MixedSinker::<Rgba, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_rgba(&mut out)
      .unwrap();
      rgba_to(&src, FR, M, &mut sink).unwrap();
    }
    out
  }

  /// Asserts a format's filter colour outputs equal the equivalent 8-bit
  /// `Rgba` filter of the YUVA→RGBA-converted source pixels. Returns the max
  /// per-channel `rgba` diff (0 — same engine, same converted pixels).
  fn assert_color_equals_packed_rgba<F: Yuva444Filter, K: FilterKernel + Copy>(
    kernel: K,
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    ctx: &str,
  ) -> u8 {
    let packed = vuya_ramp(sw, sh);
    let got = F::filter_outputs(&packed, sw, sh, ow, oh, kernel);

    // rgba == the 8-bit Rgba filter of the converted canonical RGBA.
    let canonical = F::direct_rgba_u8(&packed, sw, sh);
    let want = rgba_filter_rgba(&canonical, sw, sh, ow, oh, kernel);
    let mut max_diff = 0u8;
    for (i, (&g, &w)) in got.rgba.iter().zip(want.iter()).enumerate() {
      max_diff = max_diff.max(g.abs_diff(w));
      assert_eq!(g, w, "{ctx} rgba[{i}]: {g} vs Rgba filter {w}");
    }
    // rgb == the alpha drop of the filtered RGBA.
    for (rgb_px, rgba_px) in got.rgb.chunks_exact(3).zip(want.chunks_exact(4)) {
      assert_eq!(
        rgb_px,
        &rgba_px[..3],
        "{ctx} rgb == drop-alpha(filtered rgba)"
      );
    }
    max_diff
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn vuya_downscale_color_filter_equals_packed_rgba() {
    assert_color_equals_packed_rgba::<VuyaF, _>(Triangle, 8, 8, 4, 4, "vuya triangle down");
    assert_color_equals_packed_rgba::<VuyaF, _>(CatmullRom, 8, 8, 4, 4, "vuya catmullrom down");
    assert_color_equals_packed_rgba::<VuyaF, _>(Lanczos3, 8, 8, 4, 4, "vuya lanczos3 down");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn vuya_upscale_color_filter_equals_packed_rgba() {
    assert_color_equals_packed_rgba::<VuyaF, _>(Triangle, 4, 4, 7, 7, "vuya triangle up");
    assert_color_equals_packed_rgba::<VuyaF, _>(CatmullRom, 4, 4, 7, 7, "vuya catmullrom up");
    assert_color_equals_packed_rgba::<VuyaF, _>(Lanczos3, 4, 4, 7, 7, "vuya lanczos3 up");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn vuyx_downscale_color_filter_equals_packed_rgba() {
    assert_color_equals_packed_rgba::<VuyxF, _>(Triangle, 8, 8, 4, 4, "vuyx triangle down");
    assert_color_equals_packed_rgba::<VuyxF, _>(CatmullRom, 8, 8, 4, 4, "vuyx catmullrom down");
    assert_color_equals_packed_rgba::<VuyxF, _>(Lanczos3, 8, 8, 4, 4, "vuyx lanczos3 down");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn vuyx_upscale_color_filter_equals_packed_rgba() {
    assert_color_equals_packed_rgba::<VuyxF, _>(Triangle, 4, 4, 7, 7, "vuyx triangle up");
    assert_color_equals_packed_rgba::<VuyxF, _>(CatmullRom, 4, 4, 7, 7, "vuyx catmullrom up");
    assert_color_equals_packed_rgba::<VuyxF, _>(Lanczos3, 4, 4, 7, 7, "vuyx lanczos3 up");
  }

  /// `Vuyx` forces α opaque: its filtered RGBA must carry α == 255 everywhere
  /// (a constant `0xFF` α plane filters to itself), and dropping that α gives
  /// the same RGB as the equivalent `Rgba` filter. Proves the padding byte is
  /// never a real filtered alpha.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn vuyx_filter_alpha_is_opaque() {
    let packed = vuya_ramp(8, 8);
    let got = VuyxF::filter_outputs(&packed, 8, 8, 4, 4, CatmullRom);
    assert!(
      got.rgba.chunks_exact(4).all(|px| px[3] == 255),
      "vuyx filtered α must be opaque (255) everywhere"
    );
  }
}
