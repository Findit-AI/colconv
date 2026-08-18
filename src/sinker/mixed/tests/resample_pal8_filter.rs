//! Separable-filter resample coverage for the 8-bit palette-indexed source
//! `Pal8`, routed through the merged filter engine.
//!
//! Filtering palette *indices* is meaningless (index 5 and index 200 filtered
//! is an unrelated color), exactly as averaging them is — so the only sensible
//! filter-resample is to expand each pixel to its palette color and filter
//! THAT. `Pal8` therefore routes a `Filter` plan like any real-alpha
//! RGBA-producing format: each index is looked up to its `[R, G, B, A]` (real
//! per-entry STRAIGHT alpha, FFmpeg `[B, G, R, A]` palette order) and the
//! canonical RGBA is fed through the 4-channel filter tail
//! (`pal8_rgba_filter_resample`), which filters R, G, B, A independently with
//! no premultiplication (the PIL RGBA convention). A resampled frame is
//! byte-identical to a direct full-res `Pal8` -> RGBA conversion followed by
//! the same 4-channel `FilterStream<u8>` resample of that color.
//!
//! THE per-channel equivalence (the parity goal): the `Pal8` filter output R,
//! G, B, A each equals the packed-RGBA `Rgba` source's filter resample of the
//! canonical RGBA (the direct full-res `Pal8` lookup) at the same plan — max
//! diff 0, because both feed the *same* 4-channel `FilterStream<u8>` the same
//! pixels. Asserted for Triangle / CatmullRom / Lanczos3 across a downscale
//! (8 -> 4) and an upscale (4 -> 7). The packed-RGBA oracle lives under `rgb`,
//! so the equivalence + derived-output suites are `rgb`-gated; the
//! filter-plan-accepted / no-output-noop / premult-reject / sequencing
//! regressions are feature-independent and also guard the `mono`-solo build
//! (where the routing exists but no packed-RGBA oracle does).
//!
//! The non-color outputs keep the **direct `Pal8` derivations** (NOT the
//! matrix-based ones the chromatic packed-RGBA path uses), so they are
//! validated against a direct `Pal8` conversion of the filtered color: rgb
//! drops alpha; rgb_u16 / rgba_u16 are the `(x << 8) | x` full-range widening;
//! luma / luma_u16 use the Q8 BT.709 coefficients; hsv is OpenCV HSV. There is
//! no native-depth clamp — `Pal8` is 8-bit, so the stream's own `[0, 255]`
//! clip (a signed-kernel overshoot) is exactly the source's native range.

use crate::{
  PixelSink,
  frame::Pal8Frame,
  resample::{FilteredResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Pal8, Pal8Row, pal8_to},
};
// `FilterKernel` bounds only the `rgb`-gated equivalence/oracle helpers; the
// feature-independent regressions (plan-accepted / no-output / premult-reject /
// sequencing) use concrete kernels, so a no-`rgb` build leaves it unused.
#[cfg(feature = "rgb")]
use crate::resample::FilterKernel;

const SRC: usize = 8;

/// Local LCG (the shared `pseudo_random_u8` helper is gated to feature sets
/// that exclude `mono`, so a `mono`-solo build needs its own).
fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u8;
  }
}

/// A palette whose every entry maps to a DISTINCT, non-trivial `[R, G, B, A]`
/// — so the expand-then-filter color genuinely depends on the lookup.
fn varied_palette(seed: u32) -> [[u8; 4]; 256] {
  let mut p = [[0u8; 4]; 256];
  for (i, entry) in p.iter_mut().enumerate() {
    let i = i as u32;
    // FFmpeg PAL8 byte order is [B, G, R, A].
    entry[0] = ((i.wrapping_mul(97).wrapping_add(seed)) ^ 0x5A) as u8; // B
    entry[1] = ((i.wrapping_mul(57).wrapping_add(seed)) ^ 0x3C) as u8; // G
    entry[2] = ((i.wrapping_mul(193).wrapping_add(seed)) ^ 0xA5) as u8; // R
    entry[3] = ((i.wrapping_mul(151).wrapping_add(seed)) ^ 0xF0) as u8; // A
  }
  p
}

/// A pseudo-random `n` index plane.
fn index_plane(n: usize, seed: u32) -> Vec<u8> {
  let mut buf = std::vec![0u8; n];
  fill_pseudo_random(&mut buf, seed);
  buf
}

/// Full-resolution canonical RGBA of the source — a DIRECT (identity) `Pal8`
/// conversion (palette lookup, `[B, G, R, A]` -> `[R, G, B, A]`). The oracle
/// filters this. Only the `rgb`-gated equivalence suite consumes it (the
/// packed-RGBA oracle it feeds lives under `rgb`).
#[cfg(feature = "rgb")]
fn direct_rgba(indices: &[u8], palette: &[[u8; 4]; 256], w: usize, h: usize) -> Vec<u8> {
  let frame = Pal8Frame::new(indices, palette, w as u32, h as u32, w as u32);
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Pal8>::new(w, h).with_rgba(&mut rgba).unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  rgba
}

/// Run the `Pal8` filter sink over an index plane at `ow x oh` under `kernel`,
/// attaching every output the equivalence asserts on.
#[cfg(feature = "rgb")]
#[allow(clippy::type_complexity)]
fn pal8_filter_outputs<K: FilterKernel + Copy>(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> (
  Vec<u8>,
  Vec<u8>,
  Vec<u16>,
  Vec<u16>,
  Vec<u8>,
  Vec<u16>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
) {
  let frame = Pal8Frame::new(indices, palette, sw as u32, sh as u32, sw as u32);
  let mut rgb = std::vec![0u8; ow * oh * 3];
  let mut rgba = std::vec![0u8; ow * oh * 4];
  let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut luma = std::vec![0u8; ow * oh];
  let mut lu16 = std::vec![0u16; ow * oh];
  let mut h = std::vec![0u8; ow * oh];
  let mut s = std::vec![0u8; ow * oh];
  let mut v = std::vec![0u8; ow * oh];
  {
    let mut sink = MixedSinker::<Pal8, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
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
    .with_luma_u16(&mut lu16)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }
  (rgb, rgba, rgb_u16, rgba_u16, luma, lu16, (h, s, v))
}

/// Filter-resample a canonical RGBA plane through the packed-RGBA `Rgba`
/// source at `ow x oh` under `kernel` — the trusted 4-channel
/// `FilterStream<u8>` oracle for the `Pal8` filter RGBA output.
#[cfg(feature = "rgb")]
fn rgba_filter_oracle<K: FilterKernel + Copy>(
  canonical: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> Vec<u8> {
  use crate::{KernelMatrix, frame::RgbaFrame, source::rgba_to};
  let src = RgbaFrame::new(canonical, sw as u32, sh as u32, (sw * 4) as u32);
  let mut rgba = std::vec![0u8; ow * oh * 4];
  {
    let mut sink = MixedSinker::<crate::source::Rgba, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
    rgba_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  rgba
}

/// Drop alpha from a canonical RGBA plane -> packed RGB.
#[cfg(feature = "rgb")]
fn drop_alpha(rgba: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Full-range `(x << 8) | x` widening of a u8 plane to u16 — the `Pal8` u16
/// output convention (NOT the `Ya8` zero-extension).
#[cfg(feature = "rgb")]
fn expand_u16(plane: &[u8]) -> Vec<u16> {
  plane
    .iter()
    .map(|&v| {
      let v = v as u16;
      (v << 8) | v
    })
    .collect()
}

/// Builds a 1-row `Pal8Frame` (one palette entry per pixel) reproducing a
/// canonical RGBA plane exactly, so a DIRECT `Pal8` conversion runs the
/// identity-path kernels (Q8 BT.709 luma, `(x << 8) | x` luma_u16, OpenCV HSV)
/// byte-for-byte over the filtered color — the parity source of truth for the
/// derived outputs. Returns `(indices, palette)`.
#[cfg(feature = "rgb")]
fn per_pixel_palette(filtered_rgba: &[u8]) -> (Vec<u8>, [[u8; 4]; 256]) {
  let n = filtered_rgba.len() / 4;
  let mut palette = [[0u8; 4]; 256];
  let mut indices = std::vec![0u8; n];
  for (i, px) in filtered_rgba.chunks_exact(4).enumerate() {
    // px = [R, G, B, A]; palette entry is [B, G, R, A].
    palette[i] = [px[2], px[1], px[0], px[3]];
    indices[i] = i as u8;
  }
  (indices, palette)
}

/// Direct `Pal8` luma / luma_u16 of a filtered canonical RGBA plane.
#[cfg(feature = "rgb")]
fn direct_luma_of_filtered(filtered_rgba: &[u8]) -> (Vec<u8>, Vec<u16>) {
  let n = filtered_rgba.len() / 4;
  let (indices, palette) = per_pixel_palette(filtered_rgba);
  let frame = Pal8Frame::new(&indices, &palette, n as u32, 1, n as u32);
  let mut luma = std::vec![0u8; n];
  let mut lu16 = std::vec![0u16; n];
  let mut sink = MixedSinker::<Pal8>::new(n, 1)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut lu16)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  (luma, lu16)
}

/// Direct `Pal8` HSV of a filtered canonical RGBA plane.
#[cfg(feature = "rgb")]
fn direct_hsv_of_filtered(filtered_rgba: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let n = filtered_rgba.len() / 4;
  let (indices, palette) = per_pixel_palette(filtered_rgba);
  let frame = Pal8Frame::new(&indices, &palette, n as u32, 1, n as u32);
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut v = std::vec![0u8; n];
  let mut sink = MixedSinker::<Pal8>::new(n, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  (h, s, v)
}

/// Max abs per-channel diff between two equal-length u8 planes.
#[cfg(feature = "rgb")]
fn max_diff(a: &[u8], b: &[u8]) -> u32 {
  assert_eq!(a.len(), b.len(), "length mismatch");
  a.iter()
    .zip(b.iter())
    .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
    .max()
    .unwrap_or(0)
}

/// Max **circular** per-element diff between two OpenCV hue planes (`H` is in
/// `[0, 180)`, so it wraps: `0` and `179` are 1 apart, not 179). Used to bound
/// the x86 SIMD HSV kernel's documented `+/-1` LSB against the scalar oracle
/// without a spurious failure when a near-red hue straddles the wrap.
#[cfg(feature = "rgb")]
fn max_hue_diff(a: &[u8], b: &[u8]) -> u32 {
  assert_eq!(a.len(), b.len(), "length mismatch");
  a.iter()
    .zip(b.iter())
    .map(|(&x, &y)| {
      let d = (x as i32 - y as i32).unsigned_abs();
      d.min(180 - d)
    })
    .max()
    .unwrap_or(0)
}

// ---- Per-channel RGBA equivalence vs the packed-RGBA filter oracle ---------

/// `Pal8` filter RGBA (R, G, B, A each) == the `Rgba` source's filter resample
/// of the canonical RGBA at the same plan, across every kernel and both a
/// downscale (8 -> 4) and an upscale (4 -> 7). Max diff MUST be 0.
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_rgba_equals_packed_rgba_filter_of_canonical() {
  use crate::resample::{CatmullRom, Lanczos3, Triangle};

  fn one<K: FilterKernel + Copy>(kernel: K, label: &str) {
    // Downscale 8 -> 4 and upscale 4 -> 7.
    for &(sw, sh, ow, oh, seed) in &[
      (8usize, 8usize, 4usize, 4usize, 0x51A1u32),
      (4, 4, 7, 7, 0x7C0F),
    ] {
      let palette = varied_palette(seed ^ 0x11);
      let indices = index_plane(sw * sh, seed);
      let (_, rgba, ..) = pal8_filter_outputs(&indices, &palette, sw, sh, ow, oh, kernel);
      let canonical = direct_rgba(&indices, &palette, sw, sh);
      let oracle = rgba_filter_oracle(&canonical, sw, sh, ow, oh, kernel);
      assert_eq!(
        max_diff(&rgba, &oracle),
        0,
        "{label}: Pal8 filter RGBA != packed-RGBA filter of canonical ({sw}x{sh}->{ow}x{oh})"
      );
    }
  }

  one(Triangle, "Triangle");
  one(CatmullRom, "CatmullRom");
  one(Lanczos3, "Lanczos3");
}

/// Every derived output (rgb / rgb_u16 / rgba_u16 / luma / luma_u16 / hsv) of
/// the `Pal8` filter path equals a direct full-res `Pal8` conversion of the
/// filtered color (RGBA taken from the trusted packed-RGBA filter oracle) —
/// pinning that luma stays Q8 BT.709 and the u16 outputs the `(x << 8) | x`
/// full-range widening (NOT the matrix-luma / zero-extension the chromatic
/// packed-RGBA tail uses).
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_all_outputs_derive_from_filtered_color() {
  use crate::resample::{CatmullRom, Lanczos3, Triangle};

  fn one<K: FilterKernel + Copy>(kernel: K, label: &str) {
    for &(sw, sh, ow, oh, seed) in &[
      (8usize, 8usize, 4usize, 4usize, 0xBEEFu32),
      (4, 4, 7, 7, 0x42AC),
    ] {
      let palette = varied_palette(seed ^ 0x42);
      let indices = index_plane(sw * sh, seed);
      let (rgb, rgba, rgb_u16, rgba_u16, luma, lu16, (h, s, v)) =
        pal8_filter_outputs(&indices, &palette, sw, sh, ow, oh, kernel);

      // The filtered color is the trusted packed-RGBA filter oracle's RGBA.
      let canonical = direct_rgba(&indices, &palette, sw, sh);
      let filtered = rgba_filter_oracle(&canonical, sw, sh, ow, oh, kernel);
      assert_eq!(rgba, filtered, "{label}: rgba == packed-RGBA filter oracle");

      assert_eq!(rgb, drop_alpha(&filtered), "{label}: rgb == drop-alpha");
      assert_eq!(
        rgba_u16,
        expand_u16(&filtered),
        "{label}: rgba_u16 == (x<<8)|x"
      );
      assert_eq!(
        rgb_u16,
        expand_u16(&drop_alpha(&filtered)),
        "{label}: rgb_u16 == (x<<8)|x of drop-alpha"
      );
      let (luma_ref, lu16_ref) = direct_luma_of_filtered(&filtered);
      let (h_ref, s_ref, v_ref) = direct_hsv_of_filtered(&filtered);
      assert_eq!(luma, luma_ref, "{label}: luma (Q8 BT.709 of filtered RGB)");
      assert_eq!(lu16, lu16_ref, "{label}: luma_u16 ((y<<8)|y)");
      // H and S divide by `delta` / `v`; the x86 SIMD HSV kernel
      // (`x86_common::rgb_to_hsv_16_pixels`) computes those reciprocals with
      // `_mm_rcp_ps` + one Newton-Raphson step, which is documented to land
      // within +/-1 LSB of the scalar reference. Under the SSE4.1-only tier
      // that approximation can differ from the scalar oracle by 1 in a lane,
      // so H and S are checked within that tolerance. V is `max(r,g,b)` (no
      // division) and stays exact. H is OpenCV `[0, 180)` and wraps, so it is
      // compared with a circular distance (a near-red 1-LSB drift is `0` vs
      // `179`, a circular distance of 1, not 179).
      assert!(max_hue_diff(&h, &h_ref) <= 1, "{label}: hsv H within 1 LSB");
      assert!(max_diff(&s, &s_ref) <= 1, "{label}: hsv S within 1 LSB");
      assert_eq!(v, v_ref, "{label}: hsv V");
    }
  }

  one(Triangle, "Triangle");
  one(CatmullRom, "CatmullRom");
  one(Lanczos3, "Lanczos3");
}

/// Cross-frame reuse: the lazily-created filter stream is reset in
/// `begin_frame`, so a second frame produces the same output as the first.
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_cross_frame_reset_reuses_stream() {
  use crate::resample::Triangle;
  let palette = varied_palette(0x51);
  let indices = index_plane(SRC * SRC, 0x5151);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; 4 * 4 * 4];
  {
    let mut sink = MixedSinker::<Pal8, FilteredResampler<Triangle>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(4, 4, Triangle),
    )
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }
  let canonical = direct_rgba(&indices, &palette, SRC, SRC);
  let oracle = rgba_filter_oracle(&canonical, SRC, SRC, 4, 4, Triangle);
  assert_eq!(rgba, oracle, "second frame != filter oracle");
}

// ---- Feature-independent regressions (guard the `mono`-solo build) ---------

/// A `Filter` plan is ACCEPTED on `Pal8` (the routing exists) — regression
/// against the old area-only fence that rejected every filter plan. A prior
/// orphaned-test bug shipped a module that was never registered; this lives in
/// the registered `resample_pal8_filter` module (see `tests/mod.rs`).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_plan_is_accepted() {
  use crate::resample::Lanczos3;
  let palette = varied_palette(0x24);
  let indices = index_plane(SRC * SRC, 0x2424);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);
  let mut rgba = std::vec![0u8; 4 * 4 * 4];
  let mut sink = MixedSinker::<Pal8, FilteredResampler<Lanczos3>>::with_resampler(
    SRC,
    SRC,
    FilteredResampler::new(4, 4, Lanczos3),
  )
  .unwrap()
  .with_rgba(&mut rgba)
  .unwrap();
  pal8_to(&frame, &mut sink).expect("Pal8 must accept a Filter plan (straight alpha)");
  // A genuine resample ran (the output is not all-zero).
  assert!(
    rgba.iter().any(|&b| b != 0),
    "filter resample produced no output"
  );
}

/// A no-output filter sink is a no-op that allocates no filter stream.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_no_output_sink_is_a_noop() {
  use crate::resample::Triangle;
  let palette = varied_palette(0x77);
  let indices = index_plane(SRC * SRC, 0x7777);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);
  let mut sink = MixedSinker::<Pal8, FilteredResampler<Triangle>>::with_resampler(
    SRC,
    SRC,
    FilteredResampler::new(4, 4, Triangle),
  )
  .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  assert!(
    !sink.rgba_filter_stream_allocated(),
    "a no-output filter sink allocated the 4-channel filter stream"
  );
}

/// Premultiplied alpha has no filter analogue — a `Filter` plan under
/// `AlphaMode::Premultiplied` is rejected with the typed `UnsupportedFilter`
/// (routed to the area tail's reject), never silently straight-filtered.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_filter_is_rejected() {
  use crate::resample::Triangle;
  let palette = varied_palette(0x33);
  let indices = index_plane(SRC * SRC, 0x3333);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);
  let mut rgba = std::vec![0u8; 4 * 4 * 4];
  let mut sink = MixedSinker::<Pal8, FilteredResampler<Triangle>>::with_resampler(
    SRC,
    SRC,
    FilteredResampler::new(4, 4, Triangle),
  )
  .unwrap()
  .with_alpha_mode(AlphaMode::Premultiplied)
  .with_rgba(&mut rgba)
  .unwrap();
  let err = pal8_to(&frame, &mut sink).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
    ),
    "premultiplied filter not rejected with UnsupportedFilter: {err:?}"
  );
}

/// A fresh filter-resampling sink expects row 0 first; feeding row 1 first
/// trips `OutOfSequenceRow` before any snapshot is stored (atomicity).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_out_of_sequence_first_row_is_rejected() {
  use crate::resample::Triangle;
  let palette = varied_palette(0x4B);
  let indices = index_plane(SRC * SRC, 0x44BB);
  let mut rgba = std::vec![0u8; 4 * 4 * 4];
  let mut sink = MixedSinker::<Pal8, FilteredResampler<Triangle>>::with_resampler(
    SRC,
    SRC,
    FilteredResampler::new(4, 4, Triangle),
  )
  .unwrap()
  .with_rgba(&mut rgba)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Pal8Row::new(&indices[SRC..2 * SRC], &palette, 1))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "out-of-sequence first row not rejected: {err:?}"
  );
  assert!(
    !sink.rgba_filter_stream_allocated(),
    "a rejected first row allocated the filter stream"
  );
}

// RFC #238 tail-collapse atomicity: the drop-alpha RGB scratch (`source_rgb_scratch`,
// grown here because hsv is attached) is grown AFTER the 4-channel filter stream
// builds into a LOCAL and BEFORE the field insert + freeze, so a scratch OOM leaves
// the stream field `None` AND `resample_outputs` uncommitted — a retryable first row.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  use crate::resample::Triangle;
  let palette = varied_palette(0x33);
  let indices = index_plane(SRC * SRC, 0x7A1C);
  let mut rgba = std::vec![0u8; 4 * 4 * 4];
  let mut h = std::vec![0u8; 4 * 4];
  let mut s = std::vec![0u8; 4 * 4];
  let mut v = std::vec![0u8; 4 * 4];
  let mut luma = std::vec![0u8; 4 * 4];
  {
    let mut sink = MixedSinker::<Pal8, FilteredResampler<Triangle>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(4, 4, Triangle),
    )
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
    // Attach hsv so the drop-alpha RGB scratch (`source_rgb_scratch`) is grown.
    sink.set_hsv(&mut h, &mut s, &mut v).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot source-RGB scratch grow failpoint: the row-0 filter stream
    // box builds OK (into a LOCAL), then the drop-alpha RGB scratch grow refuses —
    // surfacing AllocationFailed with BOTH the stream field `None` and the freeze
    // uncommitted (the commit-together atomic shape).
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = sink
      .process(Pal8Row::new(&indices[..SRC], &palette, 0))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row drop-alpha scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the filter stream field
    // must be `None` — the fix builds it into a LOCAL and inserts only after the
    // scratch grows. The insert-before-scratch shape would leave it `Some` here, so
    // this assertion FAILS against that shape and PASSES against the fix.
    assert!(
      !sink.rgba_filter_stream_allocated(),
      "a scratch OOM must leave the rgba_filter_stream field None (no partial commit)"
    );
    // DIRECT freeze-rollback proof: the output-set snapshot must NOT be committed. A
    // freeze-before-scratch regression would commit it here even though the stream
    // insertion is still delayed.
    assert!(
      !sink.resample_outputs_frozen(),
      "a first-row scratch OOM must leave resample_outputs uncommitted (no partial commit)"
    );
    // The retry drives the same frame ROW-LEVEL (no re-begin), so the freeze snapshot
    // stays frozen — attaching luma (a CHANGED output set) and replaying from row 0
    // is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Pal8Row::new(&indices[r * SRC..(r + 1) * SRC], &palette, r))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    rgba.iter().any(|&b| b != 0),
    "the recovered frame produced real RGBA output (scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the changed-output retry drove the newly-attached luma"
  );
}
