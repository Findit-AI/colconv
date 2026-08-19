//! Separable-filter resample coverage for the high-bit packed 4:2:2 YUV
//! family — `Y210` (10-bit), `Y212` (12-bit), `Y216` (16-bit). Packed
//! YUYV-order (`Y₀, U, Y₁, V`) u16 quadruples, MSB-aligned, routed through
//! the merged filter engine.
//!
//! Each format routes a `Filter` plan to
//! [`packed_yuv422_triple_filter_resample`](super::super::packed_yuv422_triple_filter_resample):
//! the YUV is converted to RGB with the **same** closures the area path
//! uses (`*_to_rgb_row` / `*_to_rgb_u16_row`, which upsample the 4:2:2
//! chroma), then the RGB is resampled by the signed-coefficient filter
//! stream (the filter twin of the area bin). The staged RGB feeds the same
//! emit as the 4:4:4 path
//! ([`packed_yuv444_triple_feed_emit`](super::super::packed_yuv444_triple_feed_emit)).
//! Luma stays native Y: the de-interleaved Y is filter-resampled at native
//! depth, never colour-derived. So:
//!
//! - **`rgb_u16` / `rgba_u16`** equal the equivalent `Rgb48` filter resample
//!   of the source converted to native-u16 RGB, clamped to the format's
//!   native max (the sub-16-bit clamp the area path also applies; a value
//!   no-op for 16-bit `Y216`).
//! - **`rgb` / `rgba`** equal the equivalent `Rgb24` filter resample of the
//!   source converted to u8 RGB (the u8 conversion is binned independently
//!   of the u16 one — narrowing the u16 bin would diverge).
//! - **`luma_u16`** equals a single-channel [`FilterStream<u16>`] resample
//!   of the de-interleaved native Y, clamped to the native max; `luma` is
//!   that clamped binned Y narrowed `>> (BITS - 8)`.
//!
//! The `Rgb48` / `Rgb24` oracles are gated on `rgb` (the oracle source).
//! The native-range overshoot/no-wrap contract, the native-Y luma
//! equivalence, and the filter-plan-accepted regression are
//! feature-independent, so they also guard the `y2xx`-solo build (where the
//! routing exists but no packed-RGB oracle does). The overshoot /
//! load-bearing discrimination is sub-16-bit only (`Y210` / `Y212`): a raw
//! `FilterStream<u16>` cannot exceed `u16::MAX`, so a 16-bit `Y216` has no
//! over-native-max overshoot to clip.

use crate::{
  KernelMatrix,
  frame::Y2xxFrame,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;

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

/// Single-channel filter resample of a native-u16 Y plane via the merged
/// engine's [`FilterStream<u16>`] (channels = 1) — the luma oracle. The
/// 4:2:2 filter path's `luma_u16` must equal this clamped to the native
/// max (same engine, same coefficients, the de-interleaved native Y
/// resampled at native depth).
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

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  /// Only the `rgb`-gated equivalence module reads the u8 colour output,
  /// so it is dead in a `y2xx`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgb: Vec<u8>,
  rgb_u16: Vec<u16>,
  rgba_u16: Vec<u16>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

// A per-format macro keeps the three near-identical suites in lockstep
// while naming each test after its format (so a failure points at the
// exact bit depth). `$marker` is the source marker, `$row` the row type,
// `$walker` the LE walker, `$bits` the active depth.
macro_rules! y2xx_filter_suite {
  (
    $mod:ident, $marker:ident, $row:ident, $walker:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $walker};

      const BITS: u32 = $bits;
      const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
      const SHIFT: u32 = BITS - 8; // native Y → u8
      const WIRE_SHIFT: u32 = 16 - BITS; // logical code → MSB-aligned wire

      fn frame(buf: &[u16], w: usize, h: usize) -> Y2xxFrame<'_, $bits, false> {
        Y2xxFrame::try_new(buf, w as u32, h as u32, (2 * w) as u32).unwrap()
      }

      /// Pack a logical `(Y, U, V)` plane (`w x h`, chroma sampled at the
      /// even column of each 2-pixel pair) into MSB-aligned Y2xx
      /// quadruples (`Y₀, U, Y₁, V`).
      fn pack(y: &[u16], u: &[u16], v: &[u16], w: usize, h: usize) -> Vec<u16> {
        let mut buf = vec![0u16; w * 2 * h];
        for row in 0..h {
          for cx in 0..w / 2 {
            let p0 = row * w + cx * 2;
            let p1 = p0 + 1;
            let base = row * 2 * w + cx * 4;
            buf[base] = (y[p0] & NATIVE_MAX) << WIRE_SHIFT;
            buf[base + 1] = (u[p0] & NATIVE_MAX) << WIRE_SHIFT;
            buf[base + 2] = (y[p1] & NATIVE_MAX) << WIRE_SHIFT;
            buf[base + 3] = (v[p0] & NATIVE_MAX) << WIRE_SHIFT;
          }
        }
        buf
      }

      /// A per-channel `(Y, U, V)` ramp varying per pixel so every filter
      /// window sees distinct neighbours (a channel mix-up or a row/column
      /// transpose diverges immediately). All samples interior so the
      /// conversions see real math.
      fn yuv_ramp(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let n = w * h;
        let mut y = vec![0u16; n];
        let mut u = vec![0u16; n];
        let mut v = vec![0u16; n];
        let hi = ((NATIVE_MAX as u32) * 39 / 40) as u16; // interior ceiling
        for i in 0..n {
          y[i] = ((NATIVE_MAX as u32 / 6 + (i as u32) * 11) as u16).min(hi);
          u[i] = ((NATIVE_MAX as u32 / 3 + (i as u32) * 6) as u16).min(hi);
          v[i] = ((NATIVE_MAX as u32 * 4 / 5) as u16).saturating_sub((i as u16) * 5);
        }
        (y, u, v)
      }

      /// Run the format's filter sink over `packed` (`sw x sh`) at `ow x oh`
      /// under `kernel`, attaching every output the equivalence asserts on.
      fn filter_outputs<K: FilterKernel + Copy>(
        packed: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        kernel: K,
      ) -> FilterOutputs {
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb_u16 = vec![0u16; ow * oh * 3];
        let mut rgba_u16 = vec![0u16; ow * oh * 4];
        let mut luma = vec![0u8; ow * oh];
        let mut luma_u16 = vec![0u16; ow * oh];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<K>>::with_resampler(
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
          $walker(&frame(packed, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        FilterOutputs {
          rgb,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
        }
      }

      /// Direct full-res native Y of the packed frame (`w x h`).
      fn direct_luma_u16(packed: &[u16], w: usize, h: usize) -> Vec<u16> {
        let mut y = vec![0u16; w * h];
        {
          let mut sink = MixedSinker::<$marker>::new(w, h)
            .with_luma_u16(&mut y)
            .unwrap();
          $walker(&frame(packed, w, h), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        y
      }

      // ---- Native-Y luma equivalence (CLAMPING oracle) -----------------

      /// `luma_u16` equals the single-channel native-Y oracle **clamped to
      /// the native max**, and `luma` is that clamped binned Y narrowed
      /// `>> (BITS - 8)`. The raw [`FilterStream<u16>`] finalizes to the full
      /// `u16` range, so a signed kernel can overshoot a legal sub-16-bit
      /// edge; the 4:2:2 path clips the binned native Y to the native max
      /// before publishing it, so the oracle clamps too (`min(.., NATIVE_MAX)`
      /// — a no-op for 16-bit `Y216`). Returns the max per-sample `luma_u16`
      /// diff (exactly 0 — same engine, clamp on both).
      fn assert_native_y_luma<K: FilterKernel + Copy>(
        kernel: K,
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        ctx: &str,
      ) -> u16 {
        let (y, u, v) = yuv_ramp(sw, sh);
        let packed = pack(&y, &u, &v, sw, sh);
        let got = filter_outputs(&packed, sw, sh, ow, oh, kernel);
        let native_y = direct_luma_u16(&packed, sw, sh);
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
            "{ctx} luma: must be the clamped binned native Y narrowed >> (BITS - 8)"
          );
        }
        max_diff
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_filter_is_single_channel_native_y() {
        // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma_u16 must
        // be the native-Y single-channel filter (max diff 0), luma its
        // narrow.
        assert_native_y_luma(Triangle, 8, 8, 4, 4, "triangle down");
        assert_native_y_luma(CatmullRom, 8, 8, 4, 4, "catmullrom down");
        assert_native_y_luma(Lanczos3, 8, 8, 4, 4, "lanczos3 down");
        assert_native_y_luma(Triangle, 4, 4, 7, 7, "triangle up");
        assert_native_y_luma(CatmullRom, 4, 4, 7, 7, "catmullrom up");
        assert_native_y_luma(Lanczos3, 4, 4, 7, 7, "lanczos3 up");
      }

      // ---- Filter-plan-accepted regression -----------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_plan_is_accepted() {
        // A filter plan must be accepted — before this routing it was
        // rejected with `UnsupportedFilter`; now it produces a real output.
        const SW: usize = 8;
        const SH: usize = 8;
        const OW: usize = 4;
        const OH: usize = 4;
        let (y, u, v) = yuv_ramp(SW, SH);
        let packed = pack(&y, &u, &v, SW, SH);
        let got = filter_outputs(&packed, SW, SH, OW, OH, Triangle);
        assert!(
          got.rgb_u16.iter().any(|&v| v != 0),
          "filter resample must populate rgb_u16 (no UnsupportedFilter)"
        );
        assert!(
          got.luma_u16.iter().any(|&v| v != 0),
          "filter resample must populate luma_u16 (no UnsupportedFilter)"
        );
      }
    }
  };
}

y2xx_filter_suite!(y210, Y210, Y210Row, y210_to, 10,);
y2xx_filter_suite!(y212, Y212, Y212Row, y212_to, 12,);
y2xx_filter_suite!(y216, Y216, Y216Row, y216_to, 16,);

// ---- Native-range clamp / no-wrap (sub-16-bit, feature-independent) ----
//
// A `CatmullRom` / `Lanczos3` negative lobe overshoots a near-max colour /
// Y edge, so a finalized binned sample can exceed the sub-16-bit native max
// even though the `FilterStream` only clamps to the full `u16` range. The
// filter path clips every colour sample (via
// `packed_rgb_u16_resample_emit`) and the de-interleaved native Y (via
// `packed_yuv444_triple_feed_emit`) to the native max before publishing, so
// no value wraps above the documented range. These are sub-16-bit only:
// `Y216` is full 16-bit, so a raw `FilterStream<u16>` cannot exceed
// `u16::MAX` — there is no over-native-max overshoot to clip. Feature-
// independent — no `Rgb48` oracle — so they guard the `y2xx`-solo build;
// the `rgb`-gated `clamp_is_load_bearing` test below proves the clamp clips
// a *real* overshoot rather than passing vacuously.

macro_rules! y2xx_overshoot_suite {
  (
    $mod:ident, $marker:ident, $walker:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::source::{$marker, $walker};

      const BITS: u32 = $bits;
      const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
      const SHIFT: u32 = BITS - 8;
      const WIRE_SHIFT: u32 = 16 - BITS;

      fn frame(buf: &[u16], w: usize, h: usize) -> Y2xxFrame<'_, $bits, false> {
        Y2xxFrame::try_new(buf, w as u32, h as u32, (2 * w) as u32).unwrap()
      }
      fn pack(y: &[u16], u: &[u16], v: &[u16], w: usize, h: usize) -> Vec<u16> {
        let mut buf = vec![0u16; w * 2 * h];
        for row in 0..h {
          for cx in 0..w / 2 {
            let p0 = row * w + cx * 2;
            let p1 = p0 + 1;
            let base = row * 2 * w + cx * 4;
            buf[base] = (y[p0] & NATIVE_MAX) << WIRE_SHIFT;
            buf[base + 1] = (u[p0] & NATIVE_MAX) << WIRE_SHIFT;
            buf[base + 2] = (y[p1] & NATIVE_MAX) << WIRE_SHIFT;
            buf[base + 3] = (v[p0] & NATIVE_MAX) << WIRE_SHIFT;
          }
        }
        buf
      }
      fn step_edge(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mid = 1u16 << (BITS - 1);
        let (mut y, mut u, mut v) = (vec![0u16; w * h], vec![0u16; w * h], vec![0u16; w * h]);
        for i in 0..w * h {
          let x = i % w;
          if x >= w / 2 {
            y[i] = NATIVE_MAX;
            u[i] = mid;
            v[i] = mid;
          } else {
            y[i] = 0;
            u[i] = mid;
            v[i] = mid;
          }
        }
        (y, u, v)
      }
      fn filter_outputs<K: FilterKernel + Copy>(
        packed: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        kernel: K,
      ) -> FilterOutputs {
        let mut rgb = vec![0u8; ow * oh * 3];
        let mut rgb_u16 = vec![0u16; ow * oh * 3];
        let mut rgba_u16 = vec![0u16; ow * oh * 4];
        let mut luma = vec![0u8; ow * oh];
        let mut luma_u16 = vec![0u16; ow * oh];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<K>>::with_resampler(
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
          $walker(&frame(packed, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        FilterOutputs {
          rgb,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
        }
      }

      /// Colour overshoot is clamped to the native max: every native-depth
      /// colour sample stays `<= NATIVE_MAX` (no wrap above the documented
      /// range), the opaque alpha is the native max, and a clipped-high
      /// (`== NATIVE_MAX`) edge exists (the bright plateau pins RGB at the
      /// ceiling, so the overshoot the clamp targets is exercised).
      fn assert_color_clamped<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
        const SW: usize = 4;
        const SD: usize = 7;
        let (y, u, v) = step_edge(SW, SW);
        let packed = pack(&y, &u, &v, SW, SW);
        let got = filter_outputs(&packed, SW, SW, SD, SD, kernel);

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
        assert!(
          got.rgb_u16.contains(&NATIVE_MAX),
          "{ctx}: expected a clipped-high (== {NATIVE_MAX}) edge in rgb_u16"
        );
      }

      /// Native-Y luma overshoot is clamped and never wraps: every
      /// `luma_u16 <= NATIVE_MAX`, a clipped-high (`== NATIVE_MAX`) edge
      /// exists, and wherever `luma_u16 == NATIVE_MAX` the u8 `luma` is its
      /// clamped narrow (`255`). Without the clamp the raw overshoot
      /// publishes a `luma_u16 > NATIVE_MAX` whose narrow wraps below 255 —
      /// so this discriminates.
      fn assert_luma_clamped_no_wrap<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
        const SW: usize = 4;
        const SD: usize = 7;
        let (y, u, v) = step_edge(SW, SW);
        let packed = pack(&y, &u, &v, SW, SW);
        let got = filter_outputs(&packed, SW, SW, SD, SD, kernel);

        assert!(
          got.luma_u16.iter().all(|&v| v <= NATIVE_MAX),
          "{ctx}: luma_u16 must stay <= {NATIVE_MAX}; max was {}",
          got.luma_u16.iter().copied().max().unwrap()
        );
        assert!(
          got.luma_u16.contains(&NATIVE_MAX),
          "{ctx}: expected a clipped-high (== {NATIVE_MAX}) edge in luma_u16"
        );
        for (i, (&hi, &lo)) in got.luma_u16.iter().zip(got.luma.iter()).enumerate() {
          assert_eq!(
            lo,
            (hi.min(NATIVE_MAX) >> SHIFT) as u8,
            "{ctx}: luma[{i}] must be the clamped binned native Y narrowed"
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
      fn catmullrom_color_overshoot_is_clamped_to_native_max() {
        assert_color_clamped(CatmullRom, "catmullrom");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn lanczos3_color_overshoot_is_clamped_to_native_max() {
        assert_color_clamped(Lanczos3, "lanczos3");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn catmullrom_luma_overshoot_is_clamped_no_wrap() {
        assert_luma_clamped_no_wrap(CatmullRom, "catmullrom");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn lanczos3_luma_overshoot_is_clamped_no_wrap() {
        assert_luma_clamped_no_wrap(Lanczos3, "lanczos3");
      }
    }
  };
}

// Sub-16-bit only — `Y216` (16-bit) has no over-native-max overshoot.
y2xx_overshoot_suite!(y210_overshoot, Y210, y210_to, 10,);
y2xx_overshoot_suite!(y212_overshoot, Y212, y212_to, 12,);

// ---- Packed-RGB equivalence oracles (gated on `rgb`) ------------------
//
// The filter path converts the YUV to RGB (u8 and native-u16) with the same
// closures the direct sink uses, then filters the RGB. So a 4:2:2 filter
// colour output equals the equivalent packed-RGB filter resample of those
// exact converted pixels: `rgb_u16` == `Rgb48` filter (clamped to the
// native max), `rgb` == `Rgb24` filter.

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

  macro_rules! y2xx_equiv_suite {
    (
      $mod:ident, $marker:ident, $walker:ident, $bits:literal,
    ) => {
      mod $mod {
        use super::*;
        use crate::source::{$marker, $walker};

        const BITS: u32 = $bits;
        const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
        const WIRE_SHIFT: u32 = 16 - BITS;

        fn frame(buf: &[u16], w: usize, h: usize) -> Y2xxFrame<'_, $bits, false> {
          Y2xxFrame::try_new(buf, w as u32, h as u32, (2 * w) as u32).unwrap()
        }
        fn pack(y: &[u16], u: &[u16], v: &[u16], w: usize, h: usize) -> Vec<u16> {
          let mut buf = vec![0u16; w * 2 * h];
          for row in 0..h {
            for cx in 0..w / 2 {
              let p0 = row * w + cx * 2;
              let p1 = p0 + 1;
              let base = row * 2 * w + cx * 4;
              buf[base] = (y[p0] & NATIVE_MAX) << WIRE_SHIFT;
              buf[base + 1] = (u[p0] & NATIVE_MAX) << WIRE_SHIFT;
              buf[base + 2] = (y[p1] & NATIVE_MAX) << WIRE_SHIFT;
              buf[base + 3] = (v[p0] & NATIVE_MAX) << WIRE_SHIFT;
            }
          }
          buf
        }
        fn yuv_ramp(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
          let n = w * h;
          let mut y = vec![0u16; n];
          let mut u = vec![0u16; n];
          let mut v = vec![0u16; n];
          let hi = ((NATIVE_MAX as u32) * 39 / 40) as u16;
          for i in 0..n {
            y[i] = ((NATIVE_MAX as u32 / 6 + (i as u32) * 11) as u16).min(hi);
            u[i] = ((NATIVE_MAX as u32 / 3 + (i as u32) * 6) as u16).min(hi);
            v[i] = ((NATIVE_MAX as u32 * 4 / 5) as u16).saturating_sub((i as u16) * 5);
          }
          (y, u, v)
        }
        fn filter_outputs<K: FilterKernel + Copy>(
          packed: &[u16],
          sw: usize,
          sh: usize,
          ow: usize,
          oh: usize,
          kernel: K,
        ) -> FilterOutputs {
          let mut rgb = vec![0u8; ow * oh * 3];
          let mut rgb_u16 = vec![0u16; ow * oh * 3];
          let mut rgba_u16 = vec![0u16; ow * oh * 4];
          let mut luma = vec![0u8; ow * oh];
          let mut luma_u16 = vec![0u16; ow * oh];
          {
            let mut sink = MixedSinker::<$marker, FilteredResampler<K>>::with_resampler(
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
            $walker(&frame(packed, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
          }
          FilterOutputs {
            rgb,
            rgb_u16,
            rgba_u16,
            luma,
            luma_u16,
          }
        }
        fn direct_rgb_u16(packed: &[u16], w: usize, h: usize) -> Vec<u16> {
          let mut rgb = vec![0u16; w * h * 3];
          {
            let mut sink = MixedSinker::<$marker>::new(w, h)
              .with_rgb_u16(&mut rgb)
              .unwrap();
            $walker(&frame(packed, w, h), FR, sink.set_kernel_matrix(M)).unwrap();
          }
          rgb
        }
        fn direct_rgb_u8(packed: &[u16], w: usize, h: usize) -> Vec<u8> {
          let mut rgb = vec![0u8; w * h * 3];
          {
            let mut sink = MixedSinker::<$marker>::new(w, h)
              .with_rgb(&mut rgb)
              .unwrap();
            $walker(&frame(packed, w, h), FR, sink.set_kernel_matrix(M)).unwrap();
          }
          rgb
        }

        /// Filter colour outputs equal the equivalent packed-RGB filter of
        /// the YUV→RGB-converted source pixels. The 16-bit `Rgb48` oracle is
        /// clamped to the native max before comparison (its unclamped
        /// overshoot is what the 4:2:2 path clips; a no-op for `Y216`).
        /// Returns the max per-channel `rgb_u16` diff (0).
        fn assert_color_equals_packed_rgb<K: FilterKernel + Copy>(
          kernel: K,
          sw: usize,
          sh: usize,
          ow: usize,
          oh: usize,
          ctx: &str,
        ) -> u16 {
          let (y, u, v) = yuv_ramp(sw, sh);
          let packed = pack(&y, &u, &v, sw, sh);
          let got = filter_outputs(&packed, sw, sh, ow, oh, kernel);

          let src_rgb_u16 = direct_rgb_u16(&packed, sw, sh);
          let rgb48 = rgb48_filter_rgb_u16(&src_rgb_u16, sw, sh, ow, oh, kernel);
          let want_u16: Vec<u16> = rgb48.iter().map(|&v| v.min(NATIVE_MAX)).collect();
          let mut max_diff = 0u16;
          for (i, (&g, &w)) in got.rgb_u16.iter().zip(want_u16.iter()).enumerate() {
            max_diff = max_diff.max(g.abs_diff(w));
            assert_eq!(g, w, "{ctx} rgb_u16[{i}]: {g} vs clamped Rgb48 filter {w}");
          }
          for (px, c) in got.rgba_u16.chunks_exact(4).zip(want_u16.chunks_exact(3)) {
            assert_eq!(&px[..3], c, "{ctx} rgba_u16 colour");
            assert_eq!(px[3], NATIVE_MAX, "{ctx} rgba_u16 alpha");
          }

          let src_rgb_u8 = direct_rgb_u8(&packed, sw, sh);
          let want_u8 = rgb24_filter_rgb(&src_rgb_u8, sw, sh, ow, oh, kernel);
          assert_eq!(got.rgb, want_u8, "{ctx} rgb (u8) == Rgb24 filter");

          max_diff
        }

        #[test]
        #[cfg_attr(
          miri,
          ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
        )]
        fn downscale_color_filter_equals_packed_rgb() {
          assert_color_equals_packed_rgb(Triangle, 8, 8, 4, 4, "triangle down");
          assert_color_equals_packed_rgb(CatmullRom, 8, 8, 4, 4, "catmullrom down");
          assert_color_equals_packed_rgb(Lanczos3, 8, 8, 4, 4, "lanczos3 down");
        }

        #[test]
        #[cfg_attr(
          miri,
          ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
        )]
        fn upscale_color_filter_equals_packed_rgb() {
          assert_color_equals_packed_rgb(Triangle, 4, 4, 7, 7, "triangle up");
          assert_color_equals_packed_rgb(CatmullRom, 4, 4, 7, 7, "catmullrom up");
          assert_color_equals_packed_rgb(Lanczos3, 4, 4, 7, 7, "lanczos3 up");
        }
      }
    };
  }

  y2xx_equiv_suite!(y210, Y210, y210_to, 10,);
  y2xx_equiv_suite!(y212, Y212, y212_to, 12,);
  y2xx_equiv_suite!(y216, Y216, y216_to, 16,);

  // ---- Load-bearing clamp (sub-16-bit only) ---------------------------
  //
  // Proves the native-depth clamp is *load-bearing*, not vacuous: the
  // **unclamped** 16-bit `Rgb48` filter of the same white/black step
  // converted RGB overshoots above the sub-16-bit native max, and at every
  // position the 4:2:2 path's `rgb_u16` equals that raw filter clipped to
  // the native max. (For `Y216` no overshoot above `u16::MAX` is possible,
  // so the load-bearing check is sub-16-bit only.)

  macro_rules! y2xx_load_bearing_suite {
    (
      $mod:ident, $marker:ident, $walker:ident, $bits:literal,
    ) => {
      mod $mod {
        use super::*;
        use crate::source::{$marker, $walker};

        const BITS: u32 = $bits;
        const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
        const WIRE_SHIFT: u32 = 16 - BITS;

        fn frame(buf: &[u16], w: usize, h: usize) -> Y2xxFrame<'_, $bits, false> {
          Y2xxFrame::try_new(buf, w as u32, h as u32, (2 * w) as u32).unwrap()
        }
        fn pack(y: &[u16], u: &[u16], v: &[u16], w: usize, h: usize) -> Vec<u16> {
          let mut buf = vec![0u16; w * 2 * h];
          for row in 0..h {
            for cx in 0..w / 2 {
              let p0 = row * w + cx * 2;
              let p1 = p0 + 1;
              let base = row * 2 * w + cx * 4;
              buf[base] = (y[p0] & NATIVE_MAX) << WIRE_SHIFT;
              buf[base + 1] = (u[p0] & NATIVE_MAX) << WIRE_SHIFT;
              buf[base + 2] = (y[p1] & NATIVE_MAX) << WIRE_SHIFT;
              buf[base + 3] = (v[p0] & NATIVE_MAX) << WIRE_SHIFT;
            }
          }
          buf
        }
        fn step_edge(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
          let mid = 1u16 << (BITS - 1);
          let (mut y, mut u, mut v) = (vec![0u16; w * h], vec![0u16; w * h], vec![0u16; w * h]);
          for i in 0..w * h {
            let x = i % w;
            if x >= w / 2 {
              y[i] = NATIVE_MAX;
              u[i] = mid;
              v[i] = mid;
            } else {
              y[i] = 0;
              u[i] = mid;
              v[i] = mid;
            }
          }
          (y, u, v)
        }
        fn filter_rgb_u16<K: FilterKernel + Copy>(
          packed: &[u16],
          sw: usize,
          sh: usize,
          ow: usize,
          oh: usize,
          kernel: K,
        ) -> Vec<u16> {
          let mut rgb_u16 = vec![0u16; ow * oh * 3];
          {
            let mut sink = MixedSinker::<$marker, FilteredResampler<K>>::with_resampler(
              sw,
              sh,
              FilteredResampler::new(ow, oh, kernel),
            )
            .unwrap()
            .with_rgb_u16(&mut rgb_u16)
            .unwrap();
            $walker(&frame(packed, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
          }
          rgb_u16
        }
        fn direct_rgb_u16(packed: &[u16], w: usize, h: usize) -> Vec<u16> {
          let mut rgb = vec![0u16; w * h * 3];
          {
            let mut sink = MixedSinker::<$marker>::new(w, h)
              .with_rgb_u16(&mut rgb)
              .unwrap();
            $walker(&frame(packed, w, h), FR, sink.set_kernel_matrix(M)).unwrap();
          }
          rgb
        }

        fn assert_clamp_is_load_bearing<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
          const SW: usize = 4;
          const SD: usize = 7;
          let (y, u, v) = step_edge(SW, SW);
          let packed = pack(&y, &u, &v, SW, SW);
          let got = filter_rgb_u16(&packed, SW, SW, SD, SD, kernel);

          // The unclamped 16-bit oracle over the SAME converted native-u16
          // RGB.
          let src_rgb_u16 = direct_rgb_u16(&packed, SW, SW);
          let raw = rgb48_filter_rgb_u16(&src_rgb_u16, SW, SW, SD, SD, kernel);

          assert!(
            raw.iter().any(|&v| v > NATIVE_MAX),
            "{ctx}: the unclamped Rgb48 filter never overshoots {NATIVE_MAX} — the clamp test is vacuous"
          );
          for (i, (&g, &r)) in got.iter().zip(raw.iter()).enumerate() {
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
        fn clamp_is_load_bearing() {
          assert_clamp_is_load_bearing(CatmullRom, "catmullrom");
          assert_clamp_is_load_bearing(Lanczos3, "lanczos3");
        }
      }
    };
  }

  y2xx_load_bearing_suite!(y210_load_bearing, Y210, y210_to, 10,);
  y2xx_load_bearing_suite!(y212_load_bearing, Y212, y212_to, 12,);
}
