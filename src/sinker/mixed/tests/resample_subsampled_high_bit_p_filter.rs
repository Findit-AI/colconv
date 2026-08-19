//! Separable-filter resample coverage for the high-bit **semi-planar** YUV
//! P-format family — `P010`/`P012`/`P016` (4:2:0), `P210`/`P212`/`P216`
//! (4:2:2), `P410`/`P412`/`P416` (4:4:4), at 10 / 12 / 16 bits. A
//! high-bit-packed `u16` Y plane plus one interleaved `U,V,U,V…` chroma
//! plane (half-width/half-height, half-width/full-height, or
//! full-width/full-height respectively), routed through the merged filter
//! engine.
//!
//! Each format routes a `Filter` plan to
//! [`packed_yuv422_triple_filter_resample`](super::super::packed_yuv422_triple_filter_resample)
//! (4:2:0 + 4:2:2) or
//! [`packed_yuv444_triple_filter_resample`](super::super::packed_yuv444_triple_filter_resample)
//! (4:4:4): the YUV is converted to RGB with the **same** `pNNN_to_rgb*`
//! closures the area path uses (which de-interleave + upsample the chroma),
//! then the RGB is resampled by the signed-coefficient filter stream (the
//! filter twin of the area bin). Luma stays native Y: the de-interleaved
//! native Y is filter-resampled at native depth, never colour-derived.
//!
//! The high-bit semi-planar P-formats expose **no** `luma_u16` output, so
//! the native-Y equivalence asserts on the u8 `luma`: it must equal the
//! single-channel [`FilterStream<u16>`] resample of the de-interleaved
//! native Y **clamped to the native max** then narrowed `>> (BITS - 8)`.
//! The clamp is load-bearing for that narrowing — without it a sub-16-bit
//! signed-kernel overshoot publishes a binned-Y above the native max whose
//! narrow wraps to a small value instead of saturating to `255`. So:
//!
//! - **`rgb_u16` / `rgba_u16`** equal the equivalent `Rgb48` filter resample
//!   of the source converted to native-u16 RGB, clamped to the format's
//!   native max (a value no-op for the 16-bit members).
//! - **`rgb` / `rgba`** equal the equivalent `Rgb24` filter resample of the
//!   source converted to u8 RGB.
//! - **`luma`** equals the clamped single-channel native-Y filter narrowed.
//!
//! For 4:2:0 the filter route must fire BEFORE the area-only native fast
//! tier (default-on); `filter_bypasses_native_route` proves it under the
//! default sink. The `Rgb48` / `Rgb24` oracles are gated on `rgb`; the
//! native-Y / overshoot / filter-plan-accepted contracts are
//! feature-independent.

use crate::{
  KernelMatrix,
  frame::*,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;

/// Single-channel filter resample of a native-u16 Y plane via the merged
/// engine's [`FilterStream<u16>`] (channels = 1) — the luma oracle. The
/// P-format filter path's `luma` must equal this **clamped to the native
/// max** then narrowed `>> (BITS - 8)` (the de-interleaved native Y
/// resampled at native depth, same engine + coefficients).
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

/// Re-encode a host-native u16 slice as LE-wire byte storage so an `Rgb48`
/// fixture reads back identically on LE (no-op) and BE (byte-swap) hosts.
#[cfg(feature = "rgb")]
fn as_le_wire(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// `Rgb48` filter resample of a host-native u16 RGB frame at `ow x oh`
/// under `kernel`, returning the native `rgb_u16` output. The colour oracle
/// for the native-u16 path.
#[cfg(feature = "rgb")]
fn rgb48_filter_rgb_u16<K: FilterKernel>(
  rgb: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> Vec<u16> {
  use crate::source::{Rgb48, rgb48_to};
  let wire = as_le_wire(rgb);
  let src = Rgb48Frame::new(&wire, sw as u32, sh as u32, (sw * 3) as u32);
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

/// `Rgb24` filter resample of a u8 RGB frame at `ow x oh` under `kernel` —
/// the colour oracle for the u8 path.
#[cfg(feature = "rgb")]
fn rgb24_filter_rgb<K: FilterKernel>(
  rgb: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> Vec<u8> {
  use crate::source::{Rgb24, rgb24_to};
  let src = Rgb24Frame::new(rgb, sw as u32, sh as u32, (sw * 3) as u32);
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

/// Every resampled output a P-format filter equivalence asserts on (no
/// `luma_u16` — the P-formats do not expose it).
struct FilterOutputs {
  /// Only the `rgb`-gated equivalence module reads the u8 colour output,
  /// so it is dead in a `yuv-semi-planar`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgb: Vec<u8>,
  rgb_u16: Vec<u16>,
  rgba_u16: Vec<u16>,
  luma: Vec<u8>,
}

// One macro per subsample family. The chroma plane geometry (`uv_len` +
// `uv_stride`) is shared across the family's bit depths, but each format has
// its own const-generic frame type / row / walker, so those are per-format.

macro_rules! p_filter_family {
  (
    $family:ident,
    uv_len = |$uw:ident, $uh:ident| $uvlen:expr,
    uv_stride = |$sw_:ident| $uvstride:expr,
    $( $mod:ident: $marker:ident, $frame:ident, $row:ident, $walker:ident, $bits:literal; )+
  ) => {
    mod $family {
      use super::*;

      $(
        mod $mod {
          use super::*;
          use crate::source::{$marker, $walker};

          const BITS: u32 = $bits;
          const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
          const SHIFT: u32 = BITS - 8; // native Y → u8
          const WIRE_SHIFT: u32 = 16 - BITS; // logical code → MSB-aligned wire

          fn uv_len(w: usize, h: usize) -> usize {
            let $uw = w;
            let $uh = h;
            let _ = ($uw, $uh);
            $uvlen
          }
          fn uv_stride(w: usize) -> usize {
            let $sw_ = w;
            $uvstride
          }

          /// Per-pixel logical `(Y)` ramp + per-chroma-sample logical
          /// `(U, V)` ramp, high-bit-packed into a full-width Y plane and an
          /// interleaved `U,V,U,V…` chroma plane (one pair per chroma site).
          fn ramp(w: usize, h: usize) -> (Vec<u16>, Vec<u16>) {
            let mut y = vec![0u16; w * h];
            let mut uv = vec![0u16; uv_len(w, h)];
            let hi = ((NATIVE_MAX as u32) * 39 / 40) as u16;
            for i in 0..w * h {
              y[i] = (((NATIVE_MAX as u32 / 6 + i as u32 * 11) as u16).min(hi)) << WIRE_SHIFT;
            }
            for c in 0..uv.len() / 2 {
              let u = ((NATIVE_MAX as u32 / 3 + c as u32 * 6) as u16).min(hi);
              let v = ((NATIVE_MAX as u32 * 4 / 5) as u16).saturating_sub((c as u16) * 5);
              uv[2 * c] = (u & NATIVE_MAX) << WIRE_SHIFT;
              uv[2 * c + 1] = (v & NATIVE_MAX) << WIRE_SHIFT;
            }
            (y, uv)
          }

          fn frame<'a>(y: &'a [u16], uv: &'a [u16], w: usize, h: usize) -> $frame<'a> {
            $frame::new(y, uv, w as u32, h as u32, w as u32, uv_stride(w) as u32)
          }

          /// Run the format's filter sink over `(y, uv)` (`sw x sh`) at
          /// `ow x oh` under `kernel` with the DEFAULT sink (native on for
          /// 4:2:0 — proving the filter route bypasses it), attaching every
          /// output the equivalence asserts on.
          fn filter_outputs<K: FilterKernel + Copy>(
            y: &[u16],
            uv: &[u16],
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
              .unwrap();
              $walker(&frame(y, uv, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
            }
            FilterOutputs {
              rgb,
              rgb_u16,
              rgba_u16,
              luma,
            }
          }

          /// De-packed logical native Y of the packed frame — the Y the luma
          /// oracle resamples (`wire >> WIRE_SHIFT`).
          fn logical_y(y: &[u16]) -> Vec<u16> {
            y.iter().map(|&s| s >> WIRE_SHIFT).collect()
          }

          // ---- Native-Y luma equivalence (CLAMPING oracle) ---------------

          /// `luma` equals the single-channel native-Y filter oracle
          /// **clamped to the native max** then narrowed `>> (BITS - 8)`.
          /// The raw [`FilterStream<u16>`] finalizes to the full `u16` range,
          /// so a signed kernel can overshoot a legal sub-16-bit edge; the
          /// P-format path clips the binned native Y to the native max before
          /// narrowing, so the oracle clamps too (a no-op for 16-bit
          /// members). Returns the max per-sample diff (exactly 0).
          fn assert_native_y_luma<K: FilterKernel + Copy>(
            kernel: K,
            sw: usize,
            sh: usize,
            ow: usize,
            oh: usize,
            ctx: &str,
          ) -> u8 {
            let (y, uv) = ramp(sw, sh);
            let got = filter_outputs(&y, &uv, sw, sh, ow, oh, kernel);
            let native_y = logical_y(&y);
            let raw = native_y_filter(kernel, &native_y, sw, sh, ow, oh);

            let mut max_diff = 0u8;
            for (i, (&g, &r)) in got.luma.iter().zip(raw.iter()).enumerate() {
              let want = (r.min(NATIVE_MAX) >> SHIFT) as u8;
              max_diff = max_diff.max(g.abs_diff(want));
              assert_eq!(
                g, want,
                "{ctx} luma[{i}]: {g} vs clamped single-channel native-Y filter narrowed {want} (raw {r})"
              );
            }
            max_diff
          }

          #[test]
          #[cfg_attr(
            miri,
            ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
          )]
          fn luma_filter_is_clamped_native_y() {
            // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma must be
            // the clamped native-Y single-channel filter narrowed (max diff 0).
            assert_native_y_luma(Triangle, 8, 8, 4, 4, "triangle down");
            assert_native_y_luma(CatmullRom, 8, 8, 4, 4, "catmullrom down");
            assert_native_y_luma(Lanczos3, 8, 8, 4, 4, "lanczos3 down");
            assert_native_y_luma(Triangle, 4, 4, 7, 7, "triangle up");
            assert_native_y_luma(CatmullRom, 4, 4, 7, 7, "catmullrom up");
            assert_native_y_luma(Lanczos3, 4, 4, 7, 7, "lanczos3 up");
          }

          // ---- Filter-plan-accepted regression ---------------------------

          #[test]
          #[cfg_attr(
            miri,
            ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
          )]
          fn filter_plan_is_accepted() {
            // A filter plan must be accepted — before this routing it was
            // rejected with `UnsupportedFilter`; now it produces real output.
            const SW: usize = 8;
            const SH: usize = 8;
            const OW: usize = 4;
            const OH: usize = 4;
            let (y, uv) = ramp(SW, SH);
            let got = filter_outputs(&y, &uv, SW, SH, OW, OH, Triangle);
            assert!(
              got.rgb_u16.iter().any(|&v| v != 0),
              "filter resample must populate rgb_u16 (no UnsupportedFilter)"
            );
            assert!(
              got.luma.iter().any(|&v| v != 0),
              "filter resample must populate luma (no UnsupportedFilter)"
            );
          }
        }
      )+
    }
  };
}

p_filter_family!(
  p0xx,
  uv_len = |w, h| (w / 2) * (h / 2) * 2,
  uv_stride = |w| w, // 4:2:0 interleaved half-width = 2 * (w / 2) = w
  p010: P010, P010LeFrame, P010Row, p010_to, 10;
  p012: P012, P012LeFrame, P012Row, p012_to, 12;
  p016: P016, P016LeFrame, P016Row, p016_to, 16;
);
p_filter_family!(
  p2xx,
  uv_len = |w, h| (w / 2) * h * 2,
  uv_stride = |w| w, // 4:2:2 interleaved half-width / full-height
  p210: P210, P210LeFrame, P210Row, p210_to, 10;
  p212: P212, P212LeFrame, P212Row, p212_to, 12;
  p216: P216, P216LeFrame, P216Row, p216_to, 16;
);
p_filter_family!(
  p4xx,
  uv_len = |w, h| w * h * 2,
  uv_stride = |w| 2 * w, // 4:4:4 interleaved full-width
  p410: P410, P410LeFrame, P410Row, p410_to, 10;
  p412: P412, P412LeFrame, P412Row, p412_to, 12;
  p416: P416, P416LeFrame, P416Row, p416_to, 16;
);

// ---- Native-range clamp / no-wrap (sub-16-bit, feature-independent) -----
//
// A `CatmullRom` / `Lanczos3` negative lobe overshoots a near-max colour /
// Y edge, so a finalized binned sample can exceed the sub-16-bit native max
// even though the `FilterStream` only clamps to the full `u16` range. The
// filter path clips every colour sample (via `packed_rgb_u16_resample_emit`)
// and the de-interleaved native Y (via `packed_yuv444_triple_feed_emit`) to
// the native max before publishing, so no value wraps above the documented
// range. Sub-16-bit only: the 16-bit members have no over-native-max
// overshoot. Feature-independent — no `Rgb48` oracle — so they guard the
// solo build; the `rgb`-gated `clamp_is_load_bearing` test below proves the
// clamp clips a *real* overshoot rather than passing vacuously.

macro_rules! p_overshoot_family {
  (
    $family:ident,
    uv_len = |$uw:ident, $uh:ident| $uvlen:expr,
    uv_stride = |$sw_:ident| $uvstride:expr,
    $( $mod:ident: $marker:ident, $frame:ident, $walker:ident, $bits:literal; )+
  ) => {
    mod $family {
      use super::*;

      $(
        mod $mod {
          use super::*;
          use crate::source::{$marker, $walker};

          const BITS: u32 = $bits;
          const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
          const SHIFT: u32 = BITS - 8;
          const WIRE_SHIFT: u32 = 16 - BITS;

          fn uv_len(w: usize, h: usize) -> usize {
            let $uw = w;
            let $uh = h;
            let _ = ($uw, $uh);
            $uvlen
          }
          fn uv_stride(w: usize) -> usize {
            let $sw_ = w;
            $uvstride
          }
          fn step_edge(w: usize, h: usize) -> (Vec<u16>, Vec<u16>) {
            let mid = 1u16 << (BITS - 1);
            let mut y = vec![0u16; w * h];
            let mut uv = vec![0u16; uv_len(w, h)];
            for i in 0..w * h {
              let logical = if i % w >= w / 2 { NATIVE_MAX } else { 0 };
              y[i] = logical << WIRE_SHIFT;
            }
            for c in 0..uv.len() / 2 {
              uv[2 * c] = (mid & NATIVE_MAX) << WIRE_SHIFT;
              uv[2 * c + 1] = (mid & NATIVE_MAX) << WIRE_SHIFT;
            }
            (y, uv)
          }
          fn frame<'a>(y: &'a [u16], uv: &'a [u16], w: usize, h: usize) -> $frame<'a> {
            $frame::new(y, uv, w as u32, h as u32, w as u32, uv_stride(w) as u32)
          }
          fn filter_outputs<K: FilterKernel + Copy>(
            y: &[u16],
            uv: &[u16],
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
              .unwrap();
              $walker(&frame(y, uv, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
            }
            FilterOutputs {
              rgb,
              rgb_u16,
              rgba_u16,
              luma,
            }
          }

          /// Colour overshoot is clamped to the native max: every
          /// native-depth colour sample stays `<= NATIVE_MAX`, the opaque
          /// alpha is the native max, and a clipped-high (`== NATIVE_MAX`)
          /// edge exists (the bright plateau pins RGB at the ceiling).
          fn assert_color_clamped<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
            const SW: usize = 4;
            const SD: usize = 7;
            let (y, uv) = step_edge(SW, SW);
            let got = filter_outputs(&y, &uv, SW, SW, SD, SD, kernel);

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

          /// Native-Y luma overshoot is clamped and never wraps: a
          /// clipped-high Y edge exists in the source step, and wherever the
          /// raw native-Y oracle overshoots the native max the u8 `luma`
          /// saturates to `255`. Without the clamp the raw overshoot's narrow
          /// would wrap below 255 — so this DISCRIMINATES.
          fn assert_luma_clamped_no_wrap<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
            const SW: usize = 4;
            const SD: usize = 7;
            let (y, uv) = step_edge(SW, SW);
            let got = filter_outputs(&y, &uv, SW, SW, SD, SD, kernel);

            // Oracle native-Y (de-packed logical), filter-resampled raw — its
            // unclamped overshoot is what the path clips. The wrapped
            // (no-clamp) narrow is `(raw >> SHIFT) as u8`; the clamped narrow
            // is `(raw.min(NATIVE_MAX) >> SHIFT) as u8`.
            let native_y: Vec<u16> = y.iter().map(|&s| s >> WIRE_SHIFT).collect();
            let raw = native_y_filter(kernel, &native_y, SW, SW, SD, SD);

            assert!(
              raw.iter().any(|&v| v > NATIVE_MAX),
              "{ctx}: the unclamped native-Y filter never overshoots {NATIVE_MAX} — the no-wrap test is vacuous"
            );
            let mut saw_wrap_divergence = false;
            for (i, (&got_lo, &r)) in got.luma.iter().zip(raw.iter()).enumerate() {
              let clamped = (r.min(NATIVE_MAX) >> SHIFT) as u8;
              let wrapped = (r >> SHIFT) as u8;
              assert_eq!(
                got_lo, clamped,
                "{ctx}: luma[{i}] must be the clamped binned native Y narrowed (raw {r})"
              );
              if r > NATIVE_MAX {
                assert_eq!(
                  got_lo, 255,
                  "{ctx}: a clipped-high Y edge must give luma == 255, not wrap (raw {r})"
                );
                if wrapped != clamped {
                  saw_wrap_divergence = true;
                }
              }
            }
            assert!(
              saw_wrap_divergence,
              "{ctx}: expected at least one overshoot whose UNCLAMPED narrow diverges from 255 (otherwise the clamp is not discriminated)"
            );
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
      )+
    }
  };
}

// Sub-16-bit only — the 16-bit members have no over-native-max overshoot.
p_overshoot_family!(
  p0xx_overshoot,
  uv_len = |w, h| (w / 2) * (h / 2) * 2,
  uv_stride = |w| w,
  p010: P010, P010LeFrame, p010_to, 10;
  p012: P012, P012LeFrame, p012_to, 12;
);
p_overshoot_family!(
  p2xx_overshoot,
  uv_len = |w, h| (w / 2) * h * 2,
  uv_stride = |w| w,
  p210: P210, P210LeFrame, p210_to, 10;
  p212: P212, P212LeFrame, p212_to, 12;
);
p_overshoot_family!(
  p4xx_overshoot,
  uv_len = |w, h| w * h * 2,
  uv_stride = |w| 2 * w,
  p410: P410, P410LeFrame, p410_to, 10;
  p412: P412, P412LeFrame, p412_to, 12;
);

// ---- Packed-RGB equivalence + load-bearing clamp (gated on `rgb`) -------
//
// The filter path converts the YUV to RGB (u8 and native-u16) with the same
// closures the direct sink uses, then filters the RGB. So a P-format filter
// colour output equals the equivalent packed-RGB filter resample of those
// exact converted pixels: `rgb_u16` == `Rgb48` filter (clamped to the native
// max), `rgb` == `Rgb24` filter. The load-bearing check proves the clamp is
// not vacuous: the **unclamped** 16-bit `Rgb48` filter of the same step
// overshoots above the sub-16-bit native max, and the path's `rgb_u16`
// equals that raw filter clipped.

#[cfg(feature = "rgb")]
mod packed_rgb_equivalence {
  use super::*;

  macro_rules! p_equiv_family {
    (
      $family:ident,
      uv_len = |$uw:ident, $uh:ident| $uvlen:expr,
      uv_stride = |$sw_:ident| $uvstride:expr,
      $( $mod:ident: $marker:ident, $frame:ident, $walker:ident, $bits:literal; )+
    ) => {
      mod $family {
        use super::*;

        $(
          mod $mod {
            use super::*;
            use crate::source::{$marker, $walker};

            const BITS: u32 = $bits;
            const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
            const WIRE_SHIFT: u32 = 16 - BITS;

            fn uv_len(w: usize, h: usize) -> usize {
              let $uw = w;
              let $uh = h;
              let _ = ($uw, $uh);
              $uvlen
            }
            fn uv_stride(w: usize) -> usize {
              let $sw_ = w;
              $uvstride
            }
            fn ramp(w: usize, h: usize) -> (Vec<u16>, Vec<u16>) {
              let mut y = vec![0u16; w * h];
              let mut uv = vec![0u16; uv_len(w, h)];
              let hi = ((NATIVE_MAX as u32) * 39 / 40) as u16;
              for i in 0..w * h {
                y[i] = (((NATIVE_MAX as u32 / 6 + i as u32 * 11) as u16).min(hi)) << WIRE_SHIFT;
              }
              for c in 0..uv.len() / 2 {
                let u = ((NATIVE_MAX as u32 / 3 + c as u32 * 6) as u16).min(hi);
                let v = ((NATIVE_MAX as u32 * 4 / 5) as u16).saturating_sub((c as u16) * 5);
                uv[2 * c] = (u & NATIVE_MAX) << WIRE_SHIFT;
                uv[2 * c + 1] = (v & NATIVE_MAX) << WIRE_SHIFT;
              }
              (y, uv)
            }
            fn frame<'a>(y: &'a [u16], uv: &'a [u16], w: usize, h: usize) -> $frame<'a> {
              $frame::new(y, uv, w as u32, h as u32, w as u32, uv_stride(w) as u32)
            }
            fn filter_outputs<K: FilterKernel + Copy>(
              y: &[u16],
              uv: &[u16],
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
                .unwrap();
                $walker(&frame(y, uv, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
              }
              FilterOutputs {
                rgb,
                rgb_u16,
                rgba_u16,
                luma,
              }
            }
            fn direct_rgb_u16(y: &[u16], uv: &[u16], w: usize, h: usize) -> Vec<u16> {
              let mut rgb = vec![0u16; w * h * 3];
              {
                let mut sink = MixedSinker::<$marker>::new(w, h)
                  .with_rgb_u16(&mut rgb)
                  .unwrap();
                $walker(&frame(y, uv, w, h), FR, sink.set_kernel_matrix(M)).unwrap();
              }
              rgb
            }
            fn direct_rgb_u8(y: &[u16], uv: &[u16], w: usize, h: usize) -> Vec<u8> {
              let mut rgb = vec![0u8; w * h * 3];
              {
                let mut sink = MixedSinker::<$marker>::new(w, h)
                  .with_rgb(&mut rgb)
                  .unwrap();
                $walker(&frame(y, uv, w, h), FR, sink.set_kernel_matrix(M)).unwrap();
              }
              rgb
            }

            /// Filter colour outputs equal the equivalent packed-RGB filter
            /// of the YUV→RGB-converted source pixels. The 16-bit `Rgb48`
            /// oracle is clamped to the native max before comparison.
            /// Returns the max per-channel `rgb_u16` diff (0).
            fn assert_color_equals_packed_rgb<K: FilterKernel + Copy>(
              kernel: K,
              sw: usize,
              sh: usize,
              ow: usize,
              oh: usize,
              ctx: &str,
            ) -> u16 {
              let (y, uv) = ramp(sw, sh);
              let got = filter_outputs(&y, &uv, sw, sh, ow, oh, kernel);

              let src_rgb_u16 = direct_rgb_u16(&y, &uv, sw, sh);
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

              let src_rgb_u8 = direct_rgb_u8(&y, &uv, sw, sh);
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
        )+
      }
    };
  }

  p_equiv_family!(
    p0xx,
    uv_len = |w, h| (w / 2) * (h / 2) * 2,
    uv_stride = |w| w,
    p010: P010, P010LeFrame, p010_to, 10;
    p012: P012, P012LeFrame, p012_to, 12;
    p016: P016, P016LeFrame, p016_to, 16;
  );
  p_equiv_family!(
    p2xx,
    uv_len = |w, h| (w / 2) * h * 2,
    uv_stride = |w| w,
    p210: P210, P210LeFrame, p210_to, 10;
    p212: P212, P212LeFrame, p212_to, 12;
    p216: P216, P216LeFrame, p216_to, 16;
  );
  p_equiv_family!(
    p4xx,
    uv_len = |w, h| w * h * 2,
    uv_stride = |w| 2 * w,
    p410: P410, P410LeFrame, p410_to, 10;
    p412: P412, P412LeFrame, p412_to, 12;
    p416: P416, P416LeFrame, p416_to, 16;
  );

  // ---- Load-bearing clamp (sub-16-bit only) -----------------------------

  macro_rules! p_load_bearing_family {
    (
      $family:ident,
      uv_len = |$uw:ident, $uh:ident| $uvlen:expr,
      uv_stride = |$sw_:ident| $uvstride:expr,
      $( $mod:ident: $marker:ident, $frame:ident, $walker:ident, $bits:literal; )+
    ) => {
      mod $family {
        use super::*;

        $(
          mod $mod {
            use super::*;
            use crate::source::{$marker, $walker};

            const BITS: u32 = $bits;
            const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
            const WIRE_SHIFT: u32 = 16 - BITS;

            fn uv_len(w: usize, h: usize) -> usize {
              let $uw = w;
              let $uh = h;
              let _ = ($uw, $uh);
              $uvlen
            }
            fn uv_stride(w: usize) -> usize {
              let $sw_ = w;
              $uvstride
            }
            fn step_edge(w: usize, h: usize) -> (Vec<u16>, Vec<u16>) {
              let mid = 1u16 << (BITS - 1);
              let mut y = vec![0u16; w * h];
              let mut uv = vec![0u16; uv_len(w, h)];
              for i in 0..w * h {
                let logical = if i % w >= w / 2 { NATIVE_MAX } else { 0 };
                y[i] = logical << WIRE_SHIFT;
              }
              for c in 0..uv.len() / 2 {
                uv[2 * c] = (mid & NATIVE_MAX) << WIRE_SHIFT;
                uv[2 * c + 1] = (mid & NATIVE_MAX) << WIRE_SHIFT;
              }
              (y, uv)
            }
            fn frame<'a>(y: &'a [u16], uv: &'a [u16], w: usize, h: usize) -> $frame<'a> {
              $frame::new(y, uv, w as u32, h as u32, w as u32, uv_stride(w) as u32)
            }
            fn filter_rgb_u16<K: FilterKernel + Copy>(
              y: &[u16],
              uv: &[u16],
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
                $walker(&frame(y, uv, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
              }
              rgb_u16
            }
            fn direct_rgb_u16(y: &[u16], uv: &[u16], w: usize, h: usize) -> Vec<u16> {
              let mut rgb = vec![0u16; w * h * 3];
              {
                let mut sink = MixedSinker::<$marker>::new(w, h)
                  .with_rgb_u16(&mut rgb)
                  .unwrap();
                $walker(&frame(y, uv, w, h), FR, sink.set_kernel_matrix(M)).unwrap();
              }
              rgb
            }

            fn assert_clamp_is_load_bearing<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
              const SW: usize = 4;
              const SD: usize = 7;
              let (y, uv) = step_edge(SW, SW);
              let got = filter_rgb_u16(&y, &uv, SW, SW, SD, SD, kernel);

              // The unclamped 16-bit oracle over the SAME converted native-u16
              // RGB.
              let src_rgb_u16 = direct_rgb_u16(&y, &uv, SW, SW);
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
        )+
      }
    };
  }

  p_load_bearing_family!(
    p0xx_load_bearing,
    uv_len = |w, h| (w / 2) * (h / 2) * 2,
    uv_stride = |w| w,
    p010: P010, P010LeFrame, p010_to, 10;
    p012: P012, P012LeFrame, p012_to, 12;
  );
  p_load_bearing_family!(
    p2xx_load_bearing,
    uv_len = |w, h| (w / 2) * h * 2,
    uv_stride = |w| w,
    p210: P210, P210LeFrame, p210_to, 10;
    p212: P212, P212LeFrame, p212_to, 12;
  );
  p_load_bearing_family!(
    p4xx_load_bearing,
    uv_len = |w, h| w * h * 2,
    uv_stride = |w| 2 * w,
    p410: P410, P410LeFrame, p410_to, 10;
    p412: P412, P412LeFrame, p412_to, 12;
  );
}

// ---- 4:2:0 filter route bypasses the native fast tier ------------------
//
// `P010` / `P012` / `P016` carry an area-only native decimator that is ON by
// default. A `Filter` plan MUST branch to the filter resampler BEFORE that
// machinery, so the default sink (native on) and an explicit
// `with_native(false)` sink produce byte-identical filter output — the
// filter route is native-route-invariant. (The native tier never runs for a
// filter plan; if it leaked through, the area-native output would diverge
// from the filter output.)

mod filter_bypasses_native_route {
  use super::*;

  macro_rules! p0xx_bypass_suite {
    ( $( $mod:ident: $marker:ident, $frame:ident, $walker:ident, $bits:literal; )+ ) => {
      $(
        mod $mod {
          use super::*;
          use crate::source::{$marker, $walker};

          const BITS: u32 = $bits;
          const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
          const WIRE_SHIFT: u32 = 16 - BITS;

          fn ramp(w: usize, h: usize) -> (Vec<u16>, Vec<u16>) {
            let cw = w / 2;
            let ch = h / 2;
            let mut y = vec![0u16; w * h];
            let mut uv = vec![0u16; cw * ch * 2];
            let hi = ((NATIVE_MAX as u32) * 39 / 40) as u16;
            for i in 0..w * h {
              y[i] = (((NATIVE_MAX as u32 / 6 + i as u32 * 11) as u16).min(hi)) << WIRE_SHIFT;
            }
            for c in 0..cw * ch {
              let u = ((NATIVE_MAX as u32 / 3 + c as u32 * 6) as u16).min(hi);
              let v = ((NATIVE_MAX as u32 * 4 / 5) as u16).saturating_sub((c as u16) * 5);
              uv[2 * c] = (u & NATIVE_MAX) << WIRE_SHIFT;
              uv[2 * c + 1] = (v & NATIVE_MAX) << WIRE_SHIFT;
            }
            (y, uv)
          }
          fn frame<'a>(y: &'a [u16], uv: &'a [u16], w: usize, h: usize) -> $frame<'a> {
            $frame::new(y, uv, w as u32, h as u32, w as u32, w as u32)
          }

          #[allow(clippy::too_many_arguments)]
          fn run<K: FilterKernel + Copy>(
            native: bool,
            y: &[u16],
            uv: &[u16],
            sw: usize,
            sh: usize,
            ow: usize,
            oh: usize,
            kernel: K,
          ) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
            let mut rgb = vec![0u8; ow * oh * 3];
            let mut rgb_u16 = vec![0u16; ow * oh * 3];
            let mut luma = vec![0u8; ow * oh];
            {
              let mut sink = MixedSinker::<$marker, FilteredResampler<K>>::with_resampler(
                sw,
                sh,
                FilteredResampler::new(ow, oh, kernel),
              )
              .unwrap()
              .with_native(native)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgb_u16(&mut rgb_u16)
              .unwrap()
              .with_luma(&mut luma)
              .unwrap();
              $walker(&frame(y, uv, sw, sh), FR, sink.set_kernel_matrix(M)).unwrap();
            }
            (rgb, rgb_u16, luma)
          }

          #[test]
          #[cfg_attr(
            miri,
            ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
          )]
          fn filter_is_native_route_invariant() {
            // Default sink has native ON; the `Filter` plan must bypass it,
            // so default (native) and `with_native(false)` are identical.
            for (sw, sh, ow, oh) in [(8usize, 8usize, 4usize, 4usize), (4, 4, 7, 7)] {
              let (y, uv) = ramp(sw, sh);
              let def = run(true, &y, &uv, sw, sh, ow, oh, CatmullRom);
              let off = run(false, &y, &uv, sw, sh, ow, oh, CatmullRom);
              assert_eq!(def.0, off.0, "rgb: native-on vs native-off diverge");
              assert_eq!(def.1, off.1, "rgb_u16: native-on vs native-off diverge");
              assert_eq!(def.2, off.2, "luma: native-on vs native-off diverge");
              // Non-trivial output, so the test can't pass vacuously on two
              // empty buffers.
              assert!(
                def.1.iter().any(|&v| v != 0),
                "filter output must be non-trivial"
              );
            }
          }
        }
      )+
    };
  }

  p0xx_bypass_suite!(
    p010: P010, P010LeFrame, p010_to, 10;
    p012: P012, P012LeFrame, p012_to, 12;
    p016: P016, P016LeFrame, p016_to, 16;
  );
}
