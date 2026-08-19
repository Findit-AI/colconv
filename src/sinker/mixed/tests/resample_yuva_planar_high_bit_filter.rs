//! Separable-filter resample coverage for the high-bit **planar** 4:x:x YUV
//! sources with a real full-resolution source alpha plane — `Yuva420p{9,10,16}`
//! (4:2:0), `Yuva422p{9,10,12,16}` (4:2:2), `Yuva444p{9,10,12,14,16}` (4:4:4).
//! Low-packed `u16` Y / A planes (full-res) + (sub-sampled) U / V planes,
//! routed through the merged filter engine.
//!
//! Each format routes a `Filter` plan to
//! [`packed_yuva444_filter_resample`](super::super::packed_yuva444_filter_resample)
//! at its native depth with `NATIVE_LUMA_U8 = false` — the SAME 4-channel
//! filter tail (and u16-luma branch) the packed `Vuya` / `Vuyx` use. Unlike
//! the 8-bit planar YUVA (whose native Y is `u8`), the high-bit native Y is
//! genuinely `u16`, so luma rides the `u16` filter stream over the
//! de-interleaved native Y (never colour-derived). The YUVA is converted to a
//! canonical native u16 `R, G, B, A` row with the **same**
//! `yuva{420,422,444}pN_to_rgba_u16_row_endian` kernel the area / direct paths
//! use, then the four interleaved channels are resampled by the
//! signed-coefficient filter stream (the filter twin of the area bin). Straight
//! alpha only (planar YUVA is not premultiplied; a premultiplied plan stays on
//! the area tail, which surfaces `UnsupportedFilter`). So:
//!
//! - **`rgba_u16` / `rgb_u16`** equal the equivalent `Rgba64` filter resample
//!   of the source converted to native-u16 RGBA, clamped to the format's
//!   native max (the sub-16-bit clamp the area path also applies — a value
//!   no-op for the 16-bit formats; alpha is a real filtered channel).
//! - **`rgb`** equals the alpha drop of the u8 colour, a real independent
//!   binning through a `u8` `FilterStream` (bounded `0..=255` by construction).
//! - **`luma` / `luma_u16`** equal a single-channel [`FilterStream<u16>`]
//!   resample of the de-interleaved native Y, clamped to the native max;
//!   `luma_u16` is that clamped binning, `luma` its `>> (BITS - 8)` narrow.
//!
//! The `Rgba64` colour oracle is gated on `rgb` (its source format). The
//! native-range overshoot/no-wrap contract (colour AND native-Y luma), the
//! premultiplied-filter rejection, and the filter-plan-accepted regression are
//! feature-independent, so they also guard the `yuva`-solo build. The
//! over-native-max overshoot / load-bearing discrimination is sub-16-bit only
//! (9 / 10 / 12 / 14): a raw `FilterStream<u16>` cannot exceed `u16::MAX`, so a
//! 16-bit format has no over-native-max overshoot to clip.

use crate::{
  KernelMatrix,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::{AlphaMode, MixedSinker},
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const SRC: usize = 8;
const OUT: usize = 4;

/// Re-encode a host-native u16 slice as LE-wire byte storage so an `Rgba64`
/// fixture reads back identically on LE (no-op) and BE (byte-swap) hosts.
#[cfg(feature = "rgb")]
fn as_le_wire(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Single-channel filter resample of a native-u16 Y plane via the merged
/// engine's [`FilterStream<u16>`] (channels = 1) — the luma oracle. The planar
/// YUVA filter path's binned native Y must equal this **clamped to the native
/// max** (same engine, same coefficients, the de-interleaved native Y
/// resampled at native depth); `luma_u16` is that clamped binning, `luma` its
/// narrow. The raw stream finalizes to the full `u16` range, so it is the
/// *unclamped* oracle — the clamping happens at the comparison site, never by
/// mirroring the stream (the V410 / Y210 native-max trap).
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

/// `Rgba64` (16-bit, native max 65535) filter resample of a host-native u16
/// RGBA frame at `ow x oh` under `kernel`, returning the native `rgba_u16`
/// output (the colour oracle — per-channel filter, no premultiplication, so
/// alpha is a real filtered channel). Gated on `rgb` (its source format).
#[cfg(feature = "rgb")]
fn rgba64_filter_rgba_u16<K: FilterKernel>(
  rgba: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  kernel: K,
) -> std::vec::Vec<u16> {
  use crate::source::{Rgba64, rgba64_to};
  let wire = as_le_wire(rgba);
  let src = crate::frame::Rgba64Frame::new(&wire, sw as u32, sh as u32, (sw * 4) as u32);
  let mut out = std::vec![0u16; ow * oh * 4];
  {
    let mut sink = MixedSinker::<Rgba64, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_rgba_u16(&mut out)
    .unwrap();
    rgba64_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  out
}

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  rgb: std::vec::Vec<u8>,
  rgb_u16: std::vec::Vec<u16>,
  rgba_u16: std::vec::Vec<u16>,
  luma: std::vec::Vec<u8>,
  luma_u16: std::vec::Vec<u16>,
}

// A per-format macro keeps the near-identical suites in lockstep while naming
// each test after its format (so a failure points at the exact bit depth +
// sub-sampling). `$cw_div` / `$ch_div` are the chroma width / height divisors
// (4:2:0 -> 2/2, 4:2:2 -> 2/1, 4:4:4 -> 1/1). Alpha is always full resolution.
macro_rules! planar_yuva_hb_filter_suite {
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
      const SHIFT: u32 = BITS - 8; // native Y / colour -> u8
      const CW: usize = SRC / $cw_div;
      const CH: usize = SRC / $ch_div;

      fn frame<'a>(
        y: &'a [u16],
        u: &'a [u16],
        v: &'a [u16],
        a: &'a [u16],
      ) -> $frame<'a> {
        $frame::try_new(
          y, u, v, a, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32, SRC as u32,
        )
        .unwrap()
      }

      /// A per-channel `(Y, U, V, A)` ramp varying per sample so every filter
      /// window sees distinct neighbours (a channel mix-up or a row/column
      /// transpose diverges immediately); alpha varies (not all-opaque) so the
      /// real-alpha filter is genuinely exercised. All samples interior so the
      /// conversions see real math.
      fn yuva_ramp() -> (
        std::vec::Vec<u16>,
        std::vec::Vec<u16>,
        std::vec::Vec<u16>,
        std::vec::Vec<u16>,
      ) {
        let mut y = std::vec![0u16; SRC * SRC];
        let mut u = std::vec![0u16; CW * CH];
        let mut v = std::vec![0u16; CW * CH];
        let mut a = std::vec![0u16; SRC * SRC];
        let hi = ((NATIVE_MAX as u32) * 39 / 40) as u16; // interior ceiling
        for i in 0..SRC * SRC {
          y[i] = ((NATIVE_MAX as u32 / 6 + (i as u32) * 11) as u16).min(hi);
          a[i] = ((NATIVE_MAX as u32 / 5 + (i as u32) * 9) as u16).min(hi);
        }
        for i in 0..CW * CH {
          u[i] = ((NATIVE_MAX as u32 / 3 + (i as u32) * 6) as u16).min(hi);
          v[i] = ((NATIVE_MAX as u32 * 4 / 5) as u16).saturating_sub((i as u16) * 5);
        }
        (y, u, v, a)
      }

      /// A bright/dark vertical step (`x >= SRC/2` near-max Y, else 0; neutral
      /// chroma, opaque alpha) so a `CatmullRom` / `Lanczos3` negative lobe
      /// overshoots the high colour / Y edge above the sub-16-bit native max.
      fn step_edge() -> (
        std::vec::Vec<u16>,
        std::vec::Vec<u16>,
        std::vec::Vec<u16>,
        std::vec::Vec<u16>,
      ) {
        let mid = 1u16 << (BITS - 1);
        let mut y = std::vec![0u16; SRC * SRC];
        let u = std::vec![mid; CW * CH];
        let v = std::vec![mid; CW * CH];
        let a = std::vec![NATIVE_MAX; SRC * SRC];
        for i in 0..SRC * SRC {
          if i % SRC >= SRC / 2 {
            y[i] = NATIVE_MAX;
          }
        }
        (y, u, v, a)
      }

      /// Run the format's filter sink over the planes at `ow x oh` under
      /// `kernel`, attaching every output the equivalence asserts on.
      fn filter_outputs<K: FilterKernel + Copy>(
        y: &[u16],
        u: &[u16],
        v: &[u16],
        a: &[u16],
        ow: usize,
        oh: usize,
        kernel: K,
      ) -> FilterOutputs {
        let mut rgb = std::vec![0u8; ow * oh * 3];
        let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
        let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
        let mut luma = std::vec![0u8; ow * oh];
        let mut luma_u16 = std::vec![0u16; ow * oh];
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
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
          $walker(&frame(y, u, v, a), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        FilterOutputs {
          rgb,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
        }
      }

      /// Host-native Y of the source planes (`SRC x SRC`) — the exact input the
      /// filter path's `deinterleave_y_high_bit_masked::<BITS, false>` produces
      /// from the `*LeFrame` Y plane (`from_le` per element, then a depth-mask
      /// that is a no-op on these in-range samples), so applying `from_le` here
      /// makes the single-channel oracle byte-identical to the sink's native Y
      /// on any host (identity on a LE host).
      fn direct_luma_u16(y: &[u16]) -> std::vec::Vec<u16> {
        y.iter().map(|&s| u16::from_le(s)).collect()
      }

      // ---- Native-Y luma equivalence (CLAMPING oracle) -----------------

      /// `luma_u16` equals the single-channel native-Y filter **clamped to the
      /// native max**, and `luma` is that clamped binning narrowed `>> SHIFT`.
      /// The raw [`FilterStream<u16>`] finalizes to the full `u16` range, so a
      /// signed kernel can overshoot a legal sub-16-bit edge; the YUVA filter
      /// path clips the binned native Y to the native max before publishing, so
      /// the oracle clamps too (`min(.., NATIVE_MAX)` — a no-op for the 16-bit
      /// formats). The raw (unclamped) value would WRAP a clipped-high edge —
      /// so the oracle must clamp, not mirror the raw stream. Returns the max
      /// per-sample `luma` diff (exactly 0).
      fn assert_native_y_luma<K: FilterKernel + Copy>(
        kernel: K,
        ow: usize,
        oh: usize,
        ctx: &str,
      ) -> u8 {
        let (y, u, v, a) = yuva_ramp();
        let got = filter_outputs(&y, &u, &v, &a, ow, oh, kernel);
        let native_y = direct_luma_u16(&y);
        let raw = native_y_filter(kernel, &native_y, SRC, SRC, ow, oh);

        let mut max_diff = 0u8;
        for (i, ((&lo, &hi), &r)) in
          got.luma.iter().zip(got.luma_u16.iter()).zip(raw.iter()).enumerate()
        {
          let want_u16 = r.min(NATIVE_MAX);
          let want_u8 = (want_u16 >> SHIFT) as u8;
          assert_eq!(
            hi, want_u16,
            "{ctx} luma_u16[{i}]: {hi} vs clamped single-channel native-Y filter {want_u16} (raw {r})"
          );
          max_diff = max_diff.max(lo.abs_diff(want_u8));
          assert_eq!(
            lo, want_u8,
            "{ctx} luma[{i}]: {lo} vs clamped native-Y narrowed {want_u8} (raw {r})"
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
        // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma / luma_u16
        // must be the native-Y single-channel filter clamped then narrowed
        // (max diff 0).
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
        // A filter plan must be accepted — before this routing it was rejected
        // with `UnsupportedFilter`; now it produces a real output.
        let (y, u, v, a) = yuva_ramp();
        let got = filter_outputs(&y, &u, &v, &a, OUT, OUT, Triangle);
        assert!(
          got.rgba_u16.iter().any(|&v| v != 0),
          "filter resample must populate rgba_u16 (no UnsupportedFilter)"
        );
        assert!(
          got.luma_u16.iter().any(|&v| v != 0),
          "filter resample must populate luma_u16 (no UnsupportedFilter)"
        );
      }

      // ---- Premultiplied has no filter analogue ------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_filter_is_rejected() {
        use crate::{resample::ResampleError, sinker::MixedSinkerError};
        // A premultiplied `Filter` plan has no analogue (the engine cannot
        // un-premultiply), so it routes to the area tail, which surfaces the
        // typed `UnsupportedFilter` rather than straight-filtering premultiplied
        // colour.
        let (y, u, v, a) = yuva_ramp();
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker, FilteredResampler<Triangle>>::with_resampler(
          SRC,
          SRC,
          FilteredResampler::new(OUT, OUT, Triangle),
        )
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
        let err = $walker(&frame(&y, &u, &v, &a), FR, sink.set_kernel_matrix(M)).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
          ),
          "premultiplied filter plan must reject with UnsupportedFilter, got {err:?}"
        );
      }

      // ---- Native-range clamp / no-wrap (feature-independent) ----------
      //
      // A `CatmullRom` / `Lanczos3` negative lobe overshoots a near-max colour
      // / Y edge, so a finalized binned sample can exceed the sub-16-bit native
      // max even though the `FilterStream` only clamps to the full `u16` range.
      // The filter path clips every colour sample (via
      // `packed_rgb_u16_resample_emit`) and the de-interleaved native Y (via
      // `packed_yuva444_feed_emit`) to the native max before publishing, so no
      // value wraps above the documented range. Sub-16-bit only (16-bit
      // formats: a raw `FilterStream<u16>` cannot exceed `u16::MAX`).

      /// Colour overshoot is clamped to the native max: every native-depth
      /// colour sample (`rgb_u16` and the colour of `rgba_u16`) stays
      /// `<= NATIVE_MAX` (no wrap above the documented range), the opaque alpha
      /// is the native max, and a clipped-high (`== NATIVE_MAX`) edge exists
      /// (the bright plateau pins RGB at the ceiling, so the overshoot the clamp
      /// targets is exercised). Without the clamp a finalized native u16 sample
      /// exceeds NATIVE_MAX — so the `<= NATIVE_MAX` assertion FAILS; the
      /// load-bearing suite below pins the exact clipped value.
      fn assert_color_clamped<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
        let (y, u, v, a) = step_edge();
        let got = filter_outputs(&y, &u, &v, &a, 7, 7, kernel);

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
        // The u8 `rgb` is a real independent binning bounded to `0..=255` by its
        // `u8` stream — assert it is populated (no UnsupportedFilter).
        assert!(
          got.rgb.iter().any(|&b| b != 0),
          "{ctx}: u8 rgb must be populated"
        );
      }

      /// Native-Y luma overshoot is clamped and never wraps: wherever the
      /// clamped single-channel native-Y oracle is `NATIVE_MAX`, `luma` is its
      /// clamped narrow (`255`) and `luma_u16` is `NATIVE_MAX`. The unclamped
      /// raw oracle overshoots above NATIVE_MAX there, whose narrow wraps below
      /// 255 (and whose value exceeds NATIVE_MAX) — so without the clamp `luma`
      /// / `luma_u16` would differ. This genuinely discriminates.
      fn assert_luma_clamped_no_wrap<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
        let (y, u, v, a) = step_edge();
        let got = filter_outputs(&y, &u, &v, &a, 7, 7, kernel);
        let native_y = direct_luma_u16(&y);
        let raw = native_y_filter(kernel, &native_y, SRC, SRC, 7, 7);

        assert!(
          raw.iter().any(|&v| v > NATIVE_MAX),
          "{ctx}: the unclamped single-channel native-Y filter never overshoots \
           {NATIVE_MAX} — the luma clamp test is vacuous"
        );
        for (i, ((&lo, &hi), &r)) in
          got.luma.iter().zip(got.luma_u16.iter()).zip(raw.iter()).enumerate()
        {
          let clamped_u16 = r.min(NATIVE_MAX);
          assert_eq!(
            hi, clamped_u16,
            "{ctx} luma_u16[{i}]: {hi} vs clamped native-Y {clamped_u16} (raw {r})"
          );
          assert_eq!(
            lo,
            (clamped_u16 >> SHIFT) as u8,
            "{ctx} luma[{i}]: {lo} vs clamped narrowed native-Y {} (raw {r})",
            (clamped_u16 >> SHIFT) as u8
          );
          if r >= NATIVE_MAX {
            assert_eq!(
              lo, 255,
              "{ctx}: a clipped-high Y edge must give luma == 255, not wrap"
            );
            assert_eq!(
              hi, NATIVE_MAX,
              "{ctx}: a clipped-high Y edge must give luma_u16 == NATIVE_MAX, not wrap"
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

      // ---- Packed-RGBA equivalence oracle (gated on `rgb`) -------------
      //
      // The filter path converts the YUVA to native-u16 RGBA with the same
      // closure the direct sink uses, then filters the four channels. So a YUVA
      // filter colour output equals the equivalent 16-bit `Rgba64` filter
      // resample of those exact converted pixels, clamped to the native max:
      // `rgba_u16` == the clamped `Rgba64` filter (per channel, alpha a real
      // filtered channel), `rgb_u16` == its alpha drop.

      #[cfg(feature = "rgb")]
      fn direct_rgba_u16(y: &[u16], u: &[u16], v: &[u16], a: &[u16]) -> std::vec::Vec<u16> {
        let mut rgba = std::vec![0u16; SRC * SRC * 4];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgba_u16(&mut rgba)
            .unwrap();
          $walker(&frame(y, u, v, a), FR, sink.set_kernel_matrix(M)).unwrap();
        }
        rgba
      }

      /// Filter colour outputs equal the equivalent `Rgba64` filter of the
      /// YUVA->RGBA-u16-converted source pixels, clamped to the native max (its
      /// unclamped overshoot is what the YUVA path clips; a no-op for the 16-bit
      /// formats). Returns the max per-channel `rgba_u16` diff (0).
      #[cfg(feature = "rgb")]
      fn assert_color_equals_packed_rgba<K: FilterKernel + Copy>(
        kernel: K,
        ow: usize,
        oh: usize,
        ctx: &str,
      ) -> u16 {
        let (y, u, v, a) = yuva_ramp();
        let got = filter_outputs(&y, &u, &v, &a, ow, oh, kernel);

        let src_rgba_u16 = direct_rgba_u16(&y, &u, &v, &a);
        let rgba64 = rgba64_filter_rgba_u16(&src_rgba_u16, SRC, SRC, ow, oh, kernel);
        let want: std::vec::Vec<u16> = rgba64.iter().map(|&v| v.min(NATIVE_MAX)).collect();

        let mut max_diff = 0u16;
        for (i, (&g, &w)) in got.rgba_u16.iter().zip(want.iter()).enumerate() {
          max_diff = max_diff.max(g.abs_diff(w));
          assert_eq!(g, w, "{ctx} rgba_u16[{i}]: {g} vs clamped Rgba64 filter {w}");
        }
        // rgb_u16 == the alpha drop of the filtered RGBA.
        for (rgb_px, rgba_px) in got.rgb_u16.chunks_exact(3).zip(want.chunks_exact(4)) {
          assert_eq!(
            rgb_px,
            &rgba_px[..3],
            "{ctx} rgb_u16 == drop-alpha(filtered rgba_u16)"
          );
        }
        max_diff
      }

      #[cfg(feature = "rgb")]
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn downscale_color_filter_equals_packed_rgba() {
        assert_color_equals_packed_rgba(Triangle, OUT, OUT, "triangle down");
        assert_color_equals_packed_rgba(CatmullRom, OUT, OUT, "catmullrom down");
        assert_color_equals_packed_rgba(Lanczos3, OUT, OUT, "lanczos3 down");
      }

      #[cfg(feature = "rgb")]
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn upscale_color_filter_equals_packed_rgba() {
        assert_color_equals_packed_rgba(Triangle, 7, 7, "triangle up");
        assert_color_equals_packed_rgba(CatmullRom, 7, 7, "catmullrom up");
        assert_color_equals_packed_rgba(Lanczos3, 7, 7, "lanczos3 up");
      }

      // ---- Load-bearing colour clamp (sub-16-bit only, gated on `rgb`) -
      //
      // Proves the colour native-depth clamp is *load-bearing*, not vacuous:
      // the **unclamped** 16-bit `Rgba64` filter of the same step's converted
      // RGBA overshoots above the sub-16-bit native max, and at every position
      // the YUVA path's `rgba_u16` equals that raw filter clipped to the native
      // max. (For the 16-bit formats no overshoot above `u16::MAX` is possible,
      // so the load-bearing check is sub-16-bit only.)

      #[cfg(feature = "rgb")]
      fn assert_clamp_is_load_bearing<K: FilterKernel + Copy>(kernel: K, ctx: &str) {
        let (y, u, v, a) = step_edge();
        let got = filter_outputs(&y, &u, &v, &a, 7, 7, kernel);

        let src_rgba_u16 = direct_rgba_u16(&y, &u, &v, &a);
        let raw = rgba64_filter_rgba_u16(&src_rgba_u16, SRC, SRC, 7, 7, kernel);

        // Only the colour channels (0..3) can overshoot the native max; the
        // opaque alpha plateau filters to itself (== NATIVE_MAX) and never
        // exceeds it, so check the colour channels for the over-max overshoot.
        assert!(
          raw.chunks_exact(4).any(|px| px[..3].iter().any(|&v| v > NATIVE_MAX)),
          "{ctx}: the unclamped Rgba64 filter colour never overshoots {NATIVE_MAX} — \
           the colour clamp test is vacuous"
        );
        for (i, (&g, &r)) in got.rgba_u16.iter().zip(raw.iter()).enumerate() {
          assert_eq!(
            g,
            r.min(NATIVE_MAX),
            "{ctx} rgba_u16[{i}]: {g} vs clamped unclamped-oracle {} (raw {r})",
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

// 4:2:0 (half-width, half-height chroma): 9 / 10 / 16-bit.
planar_yuva_hb_filter_suite!(
  yuva420p9,
  Yuva420p9LeFrame,
  Yuva420p9,
  yuva420p9_to,
  9,
  2,
  2,
);
planar_yuva_hb_filter_suite!(
  yuva420p10,
  Yuva420p10LeFrame,
  Yuva420p10,
  yuva420p10_to,
  10,
  2,
  2,
);
planar_yuva_hb_filter_suite!(
  yuva420p12,
  Yuva420p12LeFrame,
  Yuva420p12,
  yuva420p12_to,
  12,
  2,
  2,
);
planar_yuva_hb_filter_suite!(
  yuva420p16,
  Yuva420p16LeFrame,
  Yuva420p16,
  yuva420p16_to,
  16,
  2,
  2,
);

// 4:2:2 (half-width, full-height chroma): 9 / 10 / 12 / 16-bit.
planar_yuva_hb_filter_suite!(
  yuva422p9,
  Yuva422p9LeFrame,
  Yuva422p9,
  yuva422p9_to,
  9,
  2,
  1,
);
planar_yuva_hb_filter_suite!(
  yuva422p10,
  Yuva422p10LeFrame,
  Yuva422p10,
  yuva422p10_to,
  10,
  2,
  1,
);
planar_yuva_hb_filter_suite!(
  yuva422p12,
  Yuva422p12LeFrame,
  Yuva422p12,
  yuva422p12_to,
  12,
  2,
  1,
);
planar_yuva_hb_filter_suite!(
  yuva422p16,
  Yuva422p16LeFrame,
  Yuva422p16,
  yuva422p16_to,
  16,
  2,
  1,
);

// 4:4:4 (full-width, full-height chroma): 9 / 10 / 12 / 14 / 16-bit.
planar_yuva_hb_filter_suite!(
  yuva444p9,
  Yuva444p9LeFrame,
  Yuva444p9,
  yuva444p9_to,
  9,
  1,
  1,
);
planar_yuva_hb_filter_suite!(
  yuva444p10,
  Yuva444p10LeFrame,
  Yuva444p10,
  yuva444p10_to,
  10,
  1,
  1,
);
planar_yuva_hb_filter_suite!(
  yuva444p12,
  Yuva444p12LeFrame,
  Yuva444p12,
  yuva444p12_to,
  12,
  1,
  1,
);
planar_yuva_hb_filter_suite!(
  yuva444p14,
  Yuva444p14LeFrame,
  Yuva444p14,
  yuva444p14_to,
  14,
  1,
  1,
);
planar_yuva_hb_filter_suite!(
  yuva444p16,
  Yuva444p16LeFrame,
  Yuva444p16,
  yuva444p16_to,
  16,
  1,
  1,
);
