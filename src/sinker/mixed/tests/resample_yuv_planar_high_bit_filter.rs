//! Separable-filter resample coverage for the high-bit **planar** YUV
//! family — `Yuv420p{10,12,14,16}` (4:2:0), `Yuv422p{10,12,14,16}` (4:2:2),
//! `Yuv444p{10,12,14,16}` (4:4:4), and `Yuv440p{10,12}` (4:4:0). Low-packed
//! `u16` Y plane + (sub-sampled) U / V planes, routed through the merged
//! filter engine.
//!
//! Each format routes a `Filter` plan to the shared high-bit triple-filter
//! tail — 4:2:0 / 4:2:2 to
//! [`packed_yuv422_triple_filter_resample`](super::super::packed_yuv422_triple_filter_resample),
//! 4:4:4 / 4:4:0 to
//! [`packed_yuv444_triple_filter_resample`](super::super::packed_yuv444_triple_filter_resample)
//! — exactly the filter twin of the area path each family already uses
//! ([`packed_yuv422_triple_resample`] / [`packed_yuv444_triple_resample`]).
//! The separate Y/U/V planes are converted to RGB with the **same** closures
//! the area path uses (`yuv4{2,4}{0,4}pN_to_rgb_row_endian` /
//! `..._to_rgb_u16_row_endian`, which upsample sub-sampled chroma), then the
//! RGB is resampled by the signed-coefficient filter stream and the same emit
//! ([`packed_yuv444_triple_feed_emit`]) runs. Luma stays native Y: the
//! de-interleaved Y is filter-resampled at native depth, never colour-derived.
//! So:
//!
//! - **`rgb_u16` / `rgba_u16`** equal the equivalent `Rgb48` filter resample
//!   of the source converted to native-u16 RGB, clamped to the format's
//!   native max (the sub-16-bit clamp the area path also applies; a value
//!   no-op for the 16-bit formats).
//! - **`rgb` / `rgba`** equal the equivalent `Rgb24` filter resample of the
//!   source converted to u8 RGB (binned independently of the u16 one).
//! - **`luma`** equals a single-channel [`FilterStream<u16>`] resample of the
//!   de-interleaved native Y, clamped to the native max, then narrowed
//!   `>> (BITS - 8)`. (The high-bit planar family exposes no `luma_u16`.)
//!
//! The `Rgb48` / `Rgb24` oracles are gated on `rgb` (the oracle source). The
//! native-range overshoot/no-wrap contract (colour AND native-Y luma) and the
//! filter-plan-accepted regression are feature-independent, so they also guard
//! the `yuv-planar`-solo build (where the routing exists but no packed-RGB
//! oracle does). The over-native-max overshoot / load-bearing discrimination
//! is sub-16-bit only (10 / 12 / 14): a raw `FilterStream<u16>` cannot exceed
//! `u16::MAX`, so a 16-bit format has no over-native-max overshoot to clip.

use crate::{
  KernelMatrix,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const SRC: usize = 8;
const OUT: usize = 4;

/// Re-encode a host-native u16 slice as LE-wire byte storage so an
/// `Rgb48` / `Rgb24`-equivalent fixture reads back identically on LE
/// (no-op) and BE (byte-swap) hosts.
#[cfg(feature = "rgb")]
fn as_le_wire(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Single-channel filter resample of a native-u16 Y plane via the merged
/// engine's [`FilterStream<u16>`] (channels = 1) — the luma oracle. The
/// planar filter path's binned native Y must equal this clamped to the
/// native max (same engine, same coefficients, the de-interleaved native Y
/// resampled at native depth), and `luma` is that clamped binning narrowed.
fn native_y_filter<K: FilterKernel>(
  kernel: K,
  y: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> std::vec::Vec<u16> {
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

/// `Rgb48` filter resample of a host-native u16 RGB frame at `ow x oh`
/// under `kernel`, returning the native `rgb_u16` output (the colour
/// oracle). Gated on `rgb` (its source format).
#[cfg(feature = "rgb")]
fn rgb48_filter_rgb_u16<K: FilterKernel>(
  rgb: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> std::vec::Vec<u16> {
  use crate::source::{Rgb48, rgb48_to};
  let wire = as_le_wire(rgb);
  let src = crate::frame::Rgb48Frame::new(&wire, sw as u32, sh as u32, (sw * 3) as u32);
  let mut out = std::vec![0u16; ow * oh * 3];
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
/// returning the `rgb` output (the u8 colour oracle). Gated on `rgb`.
#[cfg(feature = "rgb")]
fn rgb24_filter_rgb<K: FilterKernel>(
  rgb: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> std::vec::Vec<u8> {
  use crate::source::{Rgb24, rgb24_to};
  let src = crate::frame::Rgb24Frame::new(rgb, sw as u32, sh as u32, (sw * 3) as u32);
  let mut out = std::vec![0u8; ow * oh * 3];
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

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  rgb: std::vec::Vec<u8>,
  rgb_u16: std::vec::Vec<u16>,
  rgba_u16: std::vec::Vec<u16>,
  luma: std::vec::Vec<u8>,
}

// A per-format macro keeps the near-identical suites in lockstep while
// naming each test after its format (so a failure points at the exact
// bit depth + sub-sampling). `$cw_div` / `$ch_div` are the chroma width /
// height divisors (4:2:0 → 2/2, 4:2:2 → 2/1, 4:4:4 → 1/1, 4:4:0 → 1/2).
macro_rules! planar_hb_filter_suite {
  (
    $mod:ident, $frame:ident, $marker:ident, $walker:ident,
    $bits:literal, $cw_div:literal, $ch_div:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::frame::$frame;
      use crate::source::{$marker, $walker};

      const BITS: u32 = $bits;
      const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;
      const SHIFT: u32 = BITS - 8; // native Y / colour → u8
      const CW: usize = SRC / $cw_div;
      const CH: usize = SRC / $ch_div;

      fn frame<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16]) -> $frame<'a> {
        $frame::try_new(
          y, u, v, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
        )
        .unwrap()
      }

      /// A per-channel `(Y, U, V)` ramp varying per sample so every filter
      /// window sees distinct neighbours (a channel mix-up or a row/column
      /// transpose diverges immediately); all samples interior so the
      /// conversions see real math.
      fn yuv_ramp() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
        let mut y = std::vec![0u16; SRC * SRC];
        let mut u = std::vec![0u16; CW * CH];
        let mut v = std::vec![0u16; CW * CH];
        let hi = ((NATIVE_MAX as u32) * 39 / 40) as u16; // interior ceiling
        for i in 0..SRC * SRC {
          y[i] = ((NATIVE_MAX as u32 / 6 + (i as u32) * 11) as u16).min(hi);
        }
        for i in 0..CW * CH {
          u[i] = ((NATIVE_MAX as u32 / 3 + (i as u32) * 6) as u16).min(hi);
          v[i] = ((NATIVE_MAX as u32 * 4 / 5) as u16).saturating_sub((i as u16) * 5);
        }
        (y, u, v)
      }

      /// A bright/dark vertical step (`x >= SRC/2` near-max Y, else 0; neutral
      /// chroma) so a `CatmullRom` / `Lanczos3` negative lobe overshoots the
      /// high colour / Y edge.
      fn step_edge() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
        let mid = 1u16 << (BITS - 1);
        let mut y = std::vec![0u16; SRC * SRC];
        let u = std::vec![mid; CW * CH];
        let v = std::vec![mid; CW * CH];
        for i in 0..SRC * SRC {
          if i % SRC >= SRC / 2 {
            y[i] = NATIVE_MAX;
          }
        }
        (y, u, v)
      }

      /// Run the format's filter sink over the planes at `OUT x OUT` (or `up x
      /// up`) under `kernel`, attaching every output the equivalence asserts
      /// on.
      fn filter_outputs<K: FilterKernel + Copy>(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        ow: usize,
        oh: usize,
        kernel: K,
      ) -> FilterOutputs {
        let mut rgb = std::vec![0u8; ow * oh * 3];
        let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
        let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
        let mut luma = std::vec![0u8; ow * oh];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<K>>::with_resampler(
            SRC,
            SRC,
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
          $walker(&frame(y, u, v), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        FilterOutputs {
          rgb,
          rgb_u16,
          rgba_u16,
          luma,
        }
      }

      /// Host-native Y of the source planes (`SRC x SRC`) — the exact input
      /// the filter path's `deinterleave_y_high_bit_masked::<BITS, false>`
      /// produces from the `*LeFrame` Y plane (`from_le` per element, then a
      /// depth-mask that is a no-op on these in-range samples), so applying
      /// `from_le` here makes the single-channel oracle byte-identical to the
      /// sink's native Y on any host (identity on a LE host).
      fn direct_luma_u16(y: &[u16], u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
        let _ = (u, v);
        y.iter().map(|&s| u16::from_le(s)).collect()
      }

      // ---- Native-Y luma equivalence (CLAMPING oracle) -----------------

      /// `luma` equals the single-channel native-Y filter **clamped to the
      /// native max**, narrowed `>> (BITS - 8)`. The raw [`FilterStream<u16>`]
      /// finalizes to the full `u16` range, so a signed kernel can overshoot a
      /// legal sub-16-bit edge; the planar path clips the binned native Y to
      /// the native max before narrowing, so the oracle clamps too
      /// (`min(.., NATIVE_MAX)` — a no-op for the 16-bit formats). The raw
      /// (unclamped) narrow would WRAP a clipped-high edge — this is the
      /// V410 / Y210 trap, so the oracle must clamp, not mirror the raw
      /// stream. Returns the max per-sample `luma` diff (exactly 0).
      fn assert_native_y_luma<K: FilterKernel + Copy>(
        kernel: K,
        ow: usize,
        oh: usize,
        ctx: &str,
      ) -> u8 {
        let (y, u, v) = yuv_ramp();
        let got = filter_outputs(&y, &u, &v, ow, oh, kernel);
        let native_y = direct_luma_u16(&y, &u, &v);
        let raw = native_y_filter(kernel, &native_y, SRC, SRC, ow, oh);

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
      fn luma_filter_is_single_channel_native_y() {
        // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma must be the
        // native-Y single-channel filter clamped then narrowed (max diff 0).
        assert_native_y_luma(Triangle, OUT, OUT, "triangle down");
        assert_native_y_luma(CatmullRom, OUT, OUT, "catmullrom down");
        assert_native_y_luma(Lanczos3, OUT, OUT, "lanczos3 down");
        assert_native_y_luma(Triangle, 7, 7, "triangle up");
        assert_native_y_luma(CatmullRom, 7, 7, "catmullrom up");
        assert_native_y_luma(Lanczos3, 7, 7, "lanczos3 up");
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
        let (y, u, v) = yuv_ramp();
        let got = filter_outputs(&y, &u, &v, OUT, OUT, Triangle);
        assert!(
          got.rgb_u16.iter().any(|&v| v != 0),
          "filter resample must populate rgb_u16 (no UnsupportedFilter)"
        );
        assert!(
          got.luma.iter().any(|&v| v != 0),
          "filter resample must populate luma (no UnsupportedFilter)"
        );
      }

      // ---- Native-range clamp / no-wrap (feature-independent) ----------
      //
      // A `CatmullRom` / `Lanczos3` negative lobe overshoots a near-max colour
      // / Y edge, so a finalized binned sample can exceed the sub-16-bit native
      // max even though the `FilterStream` only clamps to the full `u16` range.
      // The filter path clips every colour sample (via
      // `packed_rgb_u16_resample_emit`) and the de-interleaved native Y (via
      // `packed_yuv444_triple_feed_emit`) to the native max before publishing,
      // so no value wraps above the documented range. Sub-16-bit only (16-bit
      // formats: a raw `FilterStream<u16>` cannot exceed `u16::MAX`).

      /// Colour overshoot is clamped to the native max: every native-depth
      /// colour sample (`rgb_u16` and the colour of `rgba_u16`) stays
      /// `<= NATIVE_MAX` (no wrap above the documented range), the opaque alpha
      /// is the native max, and a clipped-high (`== NATIVE_MAX`) edge exists
      /// (the bright plateau pins RGB at the ceiling, so the overshoot the
      /// clamp targets is exercised). Without the clamp a finalized native u16
      /// sample exceeds NATIVE_MAX — so the `<= NATIVE_MAX` assertion FAILS;
      /// the load-bearing suite below pins the exact clipped value. The u8
      /// `rgb` path bins independently through a `u8` `FilterStream` (it cannot
      /// exceed `255` by construction), so its no-wrap discrimination lives in
      /// the native-Y `luma` no-wrap test, which narrows the clamped native Y.
      fn assert_color_clamped<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
        let (y, u, v) = step_edge();
        let got = filter_outputs(&y, &u, &v, 7, 7, kernel);

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
        // The u8 `rgb` is a real independent binning bounded to `0..=255` by
        // its `u8` stream — assert it is populated (no UnsupportedFilter), not
        // a narrowing of `rgb_u16` (the two kernels round independently).
        assert!(
          got.rgb.iter().any(|&b| b != 0),
          "{ctx}: u8 rgb must be populated"
        );
      }

      /// Native-Y luma overshoot is clamped and never wraps: wherever the
      /// clamped single-channel native-Y oracle is `NATIVE_MAX`, `luma` is its
      /// clamped narrow (`255`). The unclamped raw oracle overshoots above
      /// NATIVE_MAX there, whose narrow wraps below 255 — so without the clamp
      /// `luma` would differ. This genuinely discriminates.
      fn assert_luma_clamped_no_wrap<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
        let (y, u, v) = step_edge();
        let got = filter_outputs(&y, &u, &v, 7, 7, kernel);
        let native_y = direct_luma_u16(&y, &u, &v);
        let raw = native_y_filter(kernel, &native_y, SRC, SRC, 7, 7);

        assert!(
          raw.iter().any(|&v| v > NATIVE_MAX),
          "{ctx}: the unclamped single-channel native-Y filter never overshoots \
           {NATIVE_MAX} — the luma clamp test is vacuous"
        );
        for (i, (&lo, &r)) in got.luma.iter().zip(raw.iter()).enumerate() {
          let clamped = (r.min(NATIVE_MAX) >> SHIFT) as u8;
          assert_eq!(
            lo, clamped,
            "{ctx} luma[{i}]: {lo} vs clamped narrowed native-Y {clamped} (raw {r})"
          );
          if r >= NATIVE_MAX {
            assert_eq!(
              lo, 255,
              "{ctx}: a clipped-high Y edge must give luma == 255, not wrap"
            );
          }
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn catmullrom_color_overshoot_is_clamped_no_wrap() {
        if BITS < 16 {
          assert_color_clamped(CatmullRom, "catmullrom");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn lanczos3_color_overshoot_is_clamped_no_wrap() {
        if BITS < 16 {
          assert_color_clamped(Lanczos3, "lanczos3");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn catmullrom_luma_overshoot_is_clamped_no_wrap() {
        if BITS < 16 {
          assert_luma_clamped_no_wrap(CatmullRom, "catmullrom");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn lanczos3_luma_overshoot_is_clamped_no_wrap() {
        if BITS < 16 {
          assert_luma_clamped_no_wrap(Lanczos3, "lanczos3");
        }
      }

      // ---- Packed-RGB equivalence oracles (gated on `rgb`) -------------
      //
      // The filter path converts the YUV to RGB (u8 and native-u16) with the
      // same closures the direct sink uses, then filters the RGB. So a planar
      // filter colour output equals the equivalent packed-RGB filter resample
      // of those exact converted pixels: `rgb_u16` == `Rgb48` filter (clamped
      // to the native max), `rgb` == `Rgb24` filter.

      #[cfg(feature = "rgb")]
      fn direct_rgb_u16(y: &[u16], u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
        let mut rgb = std::vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb_u16(&mut rgb)
            .unwrap();
          $walker(&frame(y, u, v), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        rgb
      }

      #[cfg(feature = "rgb")]
      fn direct_rgb_u8(y: &[u16], u: &[u16], v: &[u16]) -> std::vec::Vec<u8> {
        let mut rgb = std::vec![0u8; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut rgb)
            .unwrap();
          $walker(&frame(y, u, v), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        rgb
      }

      /// Filter colour outputs equal the equivalent packed-RGB filter of the
      /// YUV→RGB-converted source pixels. The 16-bit `Rgb48` oracle is clamped
      /// to the native max before comparison (its unclamped overshoot is what
      /// the planar path clips; a no-op for the 16-bit formats). Returns the
      /// max per-channel `rgb_u16` diff (0).
      #[cfg(feature = "rgb")]
      fn assert_color_equals_packed_rgb<K: FilterKernel + Copy>(
        kernel: K,
        ow: usize,
        oh: usize,
        ctx: &str,
      ) -> u16 {
        let (y, u, v) = yuv_ramp();
        let got = filter_outputs(&y, &u, &v, ow, oh, kernel);

        let src_rgb_u16 = direct_rgb_u16(&y, &u, &v);
        let rgb48 = rgb48_filter_rgb_u16(&src_rgb_u16, SRC, SRC, ow, oh, kernel);
        let want_u16: std::vec::Vec<u16> = rgb48.iter().map(|&v| v.min(NATIVE_MAX)).collect();
        let mut max_diff = 0u16;
        for (i, (&g, &w)) in got.rgb_u16.iter().zip(want_u16.iter()).enumerate() {
          max_diff = max_diff.max(g.abs_diff(w));
          assert_eq!(g, w, "{ctx} rgb_u16[{i}]: {g} vs clamped Rgb48 filter {w}");
        }
        for (px, c) in got.rgba_u16.chunks_exact(4).zip(want_u16.chunks_exact(3)) {
          assert_eq!(&px[..3], c, "{ctx} rgba_u16 colour");
          assert_eq!(px[3], NATIVE_MAX, "{ctx} rgba_u16 alpha");
        }

        let src_rgb_u8 = direct_rgb_u8(&y, &u, &v);
        let want_u8 = rgb24_filter_rgb(&src_rgb_u8, SRC, SRC, ow, oh, kernel);
        assert_eq!(got.rgb, want_u8, "{ctx} rgb (u8) == Rgb24 filter");

        max_diff
      }

      #[cfg(feature = "rgb")]
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn downscale_color_filter_equals_packed_rgb() {
        assert_color_equals_packed_rgb(Triangle, OUT, OUT, "triangle down");
        assert_color_equals_packed_rgb(CatmullRom, OUT, OUT, "catmullrom down");
        assert_color_equals_packed_rgb(Lanczos3, OUT, OUT, "lanczos3 down");
      }

      #[cfg(feature = "rgb")]
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn upscale_color_filter_equals_packed_rgb() {
        assert_color_equals_packed_rgb(Triangle, 7, 7, "triangle up");
        assert_color_equals_packed_rgb(CatmullRom, 7, 7, "catmullrom up");
        assert_color_equals_packed_rgb(Lanczos3, 7, 7, "lanczos3 up");
      }

      // ---- Load-bearing clamp (sub-16-bit only, gated on `rgb`) --------
      //
      // Proves the colour native-depth clamp is *load-bearing*, not vacuous:
      // the **unclamped** 16-bit `Rgb48` filter of the same step's converted
      // RGB overshoots above the sub-16-bit native max, and at every position
      // the planar path's `rgb_u16` equals that raw filter clipped to the
      // native max. (For the 16-bit formats no overshoot above `u16::MAX` is
      // possible, so the load-bearing check is sub-16-bit only.)

      #[cfg(feature = "rgb")]
      fn assert_clamp_is_load_bearing<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
        let (y, u, v) = step_edge();
        let got = filter_outputs(&y, &u, &v, 7, 7, kernel);

        let src_rgb_u16 = direct_rgb_u16(&y, &u, &v);
        let raw = rgb48_filter_rgb_u16(&src_rgb_u16, SRC, SRC, 7, 7, kernel);

        assert!(
          raw.iter().any(|&v| v > NATIVE_MAX),
          "{ctx}: the unclamped Rgb48 filter never overshoots {NATIVE_MAX} — \
           the colour clamp test is vacuous"
        );
        for (i, (&g, &r)) in got.rgb_u16.iter().zip(raw.iter()).enumerate() {
          assert_eq!(
            g,
            r.min(NATIVE_MAX),
            "{ctx} rgb_u16[{i}]: {g} vs clamped unclamped-oracle {} (raw {r})",
            r.min(NATIVE_MAX)
          );
        }
      }

      #[cfg(feature = "rgb")]
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn clamp_is_load_bearing() {
        if BITS < 16 {
          assert_clamp_is_load_bearing(CatmullRom, "catmullrom");
          assert_clamp_is_load_bearing(Lanczos3, "lanczos3");
        }
      }
    }
  };
}

// 4:2:0 (half-width, half-height chroma).
planar_hb_filter_suite!(yuv420p9, Yuv420p9LeFrame, Yuv420p9, yuv420p9_to, 9, 2, 2,);
planar_hb_filter_suite!(
  yuv420p10,
  Yuv420p10LeFrame,
  Yuv420p10,
  yuv420p10_to,
  10,
  2,
  2,
);
planar_hb_filter_suite!(
  yuv420p12,
  Yuv420p12LeFrame,
  Yuv420p12,
  yuv420p12_to,
  12,
  2,
  2,
);
planar_hb_filter_suite!(
  yuv420p14,
  Yuv420p14LeFrame,
  Yuv420p14,
  yuv420p14_to,
  14,
  2,
  2,
);
planar_hb_filter_suite!(
  yuv420p16,
  Yuv420p16LeFrame,
  Yuv420p16,
  yuv420p16_to,
  16,
  2,
  2,
);

// 4:2:2 (half-width, full-height chroma).
planar_hb_filter_suite!(yuv422p9, Yuv422p9LeFrame, Yuv422p9, yuv422p9_to, 9, 2, 1,);
planar_hb_filter_suite!(
  yuv422p10,
  Yuv422p10LeFrame,
  Yuv422p10,
  yuv422p10_to,
  10,
  2,
  1,
);
planar_hb_filter_suite!(
  yuv422p12,
  Yuv422p12LeFrame,
  Yuv422p12,
  yuv422p12_to,
  12,
  2,
  1,
);
planar_hb_filter_suite!(
  yuv422p14,
  Yuv422p14LeFrame,
  Yuv422p14,
  yuv422p14_to,
  14,
  2,
  1,
);
planar_hb_filter_suite!(
  yuv422p16,
  Yuv422p16LeFrame,
  Yuv422p16,
  yuv422p16_to,
  16,
  2,
  1,
);

// 4:4:4 (full-width, full-height chroma).
planar_hb_filter_suite!(yuv444p9, Yuv444p9LeFrame, Yuv444p9, yuv444p9_to, 9, 1, 1,);
planar_hb_filter_suite!(
  yuv444p10,
  Yuv444p10LeFrame,
  Yuv444p10,
  yuv444p10_to,
  10,
  1,
  1,
);
planar_hb_filter_suite!(
  yuv444p12,
  Yuv444p12LeFrame,
  Yuv444p12,
  yuv444p12_to,
  12,
  1,
  1,
);
planar_hb_filter_suite!(
  yuv444p14,
  Yuv444p14LeFrame,
  Yuv444p14,
  yuv444p14_to,
  14,
  1,
  1,
);
planar_hb_filter_suite!(
  yuv444p16,
  Yuv444p16LeFrame,
  Yuv444p16,
  yuv444p16_to,
  16,
  1,
  1,
);

// 4:4:0 (full-width, half-height chroma).
planar_hb_filter_suite!(
  yuv440p10,
  Yuv440p10LeFrame,
  Yuv440p10,
  yuv440p10_to,
  10,
  1,
  2,
);
planar_hb_filter_suite!(
  yuv440p12,
  Yuv440p12LeFrame,
  Yuv440p12,
  yuv440p12_to,
  12,
  1,
  2,
);
