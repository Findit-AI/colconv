//! Separable-filter resample coverage for the 8-bit planar 4:x:x YUV sources
//! with a real full-resolution source alpha plane — `Yuva420p`, `Yuva422p`,
//! and `Yuva444p` — routed through the merged filter engine.
//!
//! Each format routes a `Filter` plan to
//! [`packed_yuva444_filter_resample`](super::super::packed_yuva444_filter_resample)
//! at `SRC_BITS = 8`, the same 4-channel filter tail the packed `Vuya` /
//! `Vuyx` formats use (the high-bit planar / semi-planar families reuse it
//! too): the YUVA is converted to a canonical u8 `R, G, B, A` row with the
//! **same** `yuva{420,444}p_to_rgba_row` kernel the area / direct paths use
//! (4:2:0 and 4:2:2 share the half-width-chroma `yuva420p_to_rgba_row`; the
//! 4:2:0-vs-4:2:2 vertical sampling is owned by the walker), then the four
//! interleaved channels are resampled by the signed-coefficient filter stream
//! (the filter twin of the area bin). Straight alpha only (planar YUVA is not
//! premultiplied; a premultiplied plan stays on the area tail, which surfaces
//! `UnsupportedFilter`). Luma stays native Y, never colour-derived. So:
//!
//! - **`rgba` / `rgb`** equal the equivalent 8-bit `Rgba` filter resample of
//!   the source converted to u8 RGBA (alpha is a real filtered channel).
//! - **`luma` / `luma_u16`** equal the **no-alpha** sibling YUV format's
//!   filter luma over the SAME Y plane (`Yuva420p` -> `Yuv420p`, etc.): the
//!   contiguous native Y is resampled through a single-channel
//!   [`FilterStream<u8>`] — the SAME stream the merged no-alpha planar YUV
//!   formats use — so merely attaching an alpha plane cannot change the luma
//!   (the consistency contract). `luma_u16` is that resampled Y zero-extended.
//!   8-bit, so no native-depth clamp (the `u8` stream finalizes to the full
//!   `u8` range, which *is* the native range) — unlike the packed `Vuya` /
//!   `Vuyx` filter callers (also 8-bit, but with no contiguous Y plane), which
//!   stay on the u16 luma stream.
//!
//! These sources expose no u16 colour outputs; the u8 colour outputs cannot
//! overshoot (the `FilterStream<u8>` clamps to `[0, 255]`), so they have no
//! separate clamp test. The `Rgba` equivalence oracle is gated on `rgb` (the
//! oracle source). The cross-format luma parity (and the test proving it
//! discriminates the old u16-luma stream) and the filter-plan-accepted
//! regression are feature-independent, so they also guard the `yuva`-solo
//! build.

use crate::{
  KernelMatrix,
  frame::{Yuv420pFrame, Yuv422pFrame, Yuv444pFrame, Yuva420pFrame, Yuva422pFrame, Yuva444pFrame},
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{
    Yuv420p, Yuv422p, Yuv444p, Yuva420p, Yuva422p, Yuva444p, yuv420p_to, yuv422p_to, yuv444p_to,
    yuva420p_to, yuva422p_to, yuva444p_to,
  },
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const NATIVE_MAX: u16 = 255; // 8-bit native Y / colour max (pre-fix u16 clamp)

// ---- Per-format plane ramps -------------------------------------------
//
// A per-pixel Y / A ramp (full resolution) and a per-chroma-sample U / V ramp
// so every filter window sees distinct neighbours (a channel mix-up or a
// row/column transpose diverges immediately). Alpha varies (not all-opaque)
// so the real-alpha filter is genuinely exercised.

fn luma_ramp(n: usize) -> Vec<u8> {
  (0..n).map(|i| (24 + i * 7).min(235) as u8).collect()
}
fn alpha_ramp(n: usize) -> Vec<u8> {
  (0..n).map(|i| (16 + i * 9).min(250) as u8).collect()
}
fn u_ramp(n: usize) -> Vec<u8> {
  (0..n)
    .map(|i| (200u32.wrapping_sub((i as u32) * 3) & 0xFF) as u8)
    .collect()
}
fn v_ramp(n: usize) -> Vec<u8> {
  (0..n).map(|i| (40 + i * 5).min(235) as u8).collect()
}

/// A sharp black -> white horizontal step in Y (left half min-Y, right half
/// max-Y, neutral chroma, opaque), uniform vertically. A signed kernel
/// enlarging the near-max bright Y plateau overshoots above the 8-bit native
/// max — the de-interleaved native Y the luma path resamples.
fn step_y(w: usize, h: usize) -> Vec<u8> {
  let mut y = std::vec![0u8; w * h];
  for (i, dst) in y.iter_mut().enumerate() {
    *dst = if (i % w) >= w / 2 { 255 } else { 0 };
  }
  y
}

// ---- Per-format hooks (the three subsamplings + their walkers) --------

/// The bits a filter test needs to drive one 8-bit planar YUVA format: run the
/// format's filter sink, and the full-res direct conversions that produce the
/// exact RGBA / Y rows the filter path consumes. Chroma dimensions follow the
/// subsampling; Y / A are always full resolution.
trait Yuva8Filter {
  /// `(chroma_w, chroma_h)` for a `w x h` luma frame under this subsampling.
  fn chroma_dims(w: usize, h: usize) -> (usize, usize);

  /// Run the format's filter sink over the planes (`sw x sh` luma) at
  /// `ow x oh` under `kernel`, attaching every output the equivalence asserts.
  #[allow(clippy::too_many_arguments)]
  fn filter_outputs<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    a: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs;

  /// Direct full-res u8 RGBA conversion of the planes (`w x h` luma) — the
  /// exact canonical source-width RGBA the filter path resamples, so it is the
  /// `Rgba` oracle's input (real source α from the alpha plane). Only the
  /// `rgb`-gated equivalence module consumes it, so it is gated to `rgb`.
  #[cfg(feature = "rgb")]
  fn direct_rgba_u8(y: &[u8], u: &[u8], v: &[u8], a: &[u8], w: usize, h: usize) -> Vec<u8>;

  /// Filter `(luma, luma_u16)` of the **no-alpha** sibling YUV format
  /// (`Yuva420p` -> `Yuv420p`, `Yuva422p` -> `Yuv422p`, `Yuva444p` ->
  /// `Yuv444p`) over the SAME Y/U/V planes at `ow x oh` under `kernel`. This
  /// is the cross-format parity oracle: the merged no-alpha planar YUV filter
  /// (in `main`) resamples native Y through a `FilterStream<u8>`, so the
  /// 8-bit YUVA filter luma must equal it byte-for-bit — merely attaching an
  /// alpha plane must not change the luma.
  #[allow(clippy::too_many_arguments)]
  fn no_alpha_yuv_luma<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> (Vec<u8>, Vec<u16>);
}

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  rgb: Vec<u8>,
  rgba: Vec<u8>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

struct Yuva420pF;
struct Yuva422pF;
struct Yuva444pF;

impl Yuva8Filter for Yuva420pF {
  fn chroma_dims(w: usize, h: usize) -> (usize, usize) {
    (w / 2, h / 2)
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    a: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let (cw, ch) = Self::chroma_dims(sw, sh);
    let src = Yuva420pFrame::try_new(
      y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
    )
    .unwrap();
    let _ = ch;
    let mut rgb = std::vec![0u8; ow * oh * 3];
    let mut rgba = std::vec![0u8; ow * oh * 4];
    let mut luma = std::vec![0u8; ow * oh];
    let mut luma_u16 = std::vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<Yuva420p, FilteredResampler<K>>::with_resampler(
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
      yuva420p_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  #[cfg(feature = "rgb")]
  fn direct_rgba_u8(y: &[u8], u: &[u8], v: &[u8], a: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (cw, _ch) = Self::chroma_dims(w, h);
    let src = Yuva420pFrame::try_new(
      y, u, v, a, w as u32, h as u32, w as u32, cw as u32, cw as u32, w as u32,
    )
    .unwrap();
    let mut rgba = std::vec![0u8; w * h * 4];
    {
      let mut sink = MixedSinker::<Yuva420p>::new(w, h)
        .with_rgba(&mut rgba)
        .unwrap();
      yuva420p_to(&src, FR, M, &mut sink).unwrap();
    }
    rgba
  }

  fn no_alpha_yuv_luma<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> (Vec<u8>, Vec<u16>) {
    let (cw, _ch) = Self::chroma_dims(sw, sh);
    let src = Yuv420pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    let mut luma = std::vec![0u8; ow * oh];
    let mut luma_u16 = std::vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<Yuv420p, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
      yuv420p_to(&src, FR, M, &mut sink).unwrap();
    }
    (luma, luma_u16)
  }
}

impl Yuva8Filter for Yuva422pF {
  fn chroma_dims(w: usize, h: usize) -> (usize, usize) {
    (w / 2, h)
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    a: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let (cw, _ch) = Self::chroma_dims(sw, sh);
    let src = Yuva422pFrame::try_new(
      y, u, v, a, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32, sw as u32,
    )
    .unwrap();
    let mut rgb = std::vec![0u8; ow * oh * 3];
    let mut rgba = std::vec![0u8; ow * oh * 4];
    let mut luma = std::vec![0u8; ow * oh];
    let mut luma_u16 = std::vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<Yuva422p, FilteredResampler<K>>::with_resampler(
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
      yuva422p_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  #[cfg(feature = "rgb")]
  fn direct_rgba_u8(y: &[u8], u: &[u8], v: &[u8], a: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (cw, _ch) = Self::chroma_dims(w, h);
    let src = Yuva422pFrame::try_new(
      y, u, v, a, w as u32, h as u32, w as u32, cw as u32, cw as u32, w as u32,
    )
    .unwrap();
    let mut rgba = std::vec![0u8; w * h * 4];
    {
      let mut sink = MixedSinker::<Yuva422p>::new(w, h)
        .with_rgba(&mut rgba)
        .unwrap();
      yuva422p_to(&src, FR, M, &mut sink).unwrap();
    }
    rgba
  }

  fn no_alpha_yuv_luma<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> (Vec<u8>, Vec<u16>) {
    let (cw, _ch) = Self::chroma_dims(sw, sh);
    let src = Yuv422pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, cw as u32, cw as u32,
    );
    let mut luma = std::vec![0u8; ow * oh];
    let mut luma_u16 = std::vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<Yuv422p, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
      yuv422p_to(&src, FR, M, &mut sink).unwrap();
    }
    (luma, luma_u16)
  }
}

impl Yuva8Filter for Yuva444pF {
  fn chroma_dims(w: usize, h: usize) -> (usize, usize) {
    (w, h)
  }

  fn filter_outputs<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    a: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> FilterOutputs {
    let src = Yuva444pFrame::try_new(
      y, u, v, a, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32, sw as u32,
    )
    .unwrap();
    let mut rgb = std::vec![0u8; ow * oh * 3];
    let mut rgba = std::vec![0u8; ow * oh * 4];
    let mut luma = std::vec![0u8; ow * oh];
    let mut luma_u16 = std::vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<Yuva444p, FilteredResampler<K>>::with_resampler(
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
      yuva444p_to(&src, FR, M, &mut sink).unwrap();
    }
    FilterOutputs {
      rgb,
      rgba,
      luma,
      luma_u16,
    }
  }

  #[cfg(feature = "rgb")]
  fn direct_rgba_u8(y: &[u8], u: &[u8], v: &[u8], a: &[u8], w: usize, h: usize) -> Vec<u8> {
    let src = Yuva444pFrame::try_new(
      y, u, v, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
    )
    .unwrap();
    let mut rgba = std::vec![0u8; w * h * 4];
    {
      let mut sink = MixedSinker::<Yuva444p>::new(w, h)
        .with_rgba(&mut rgba)
        .unwrap();
      yuva444p_to(&src, FR, M, &mut sink).unwrap();
    }
    rgba
  }

  fn no_alpha_yuv_luma<K: FilterKernel + Copy>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> (Vec<u8>, Vec<u16>) {
    let src = Yuv444pFrame::new(
      y, u, v, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32,
    );
    let mut luma = std::vec![0u8; ow * oh];
    let mut luma_u16 = std::vec![0u16; ow * oh];
    {
      let mut sink = MixedSinker::<Yuv444p, FilteredResampler<K>>::with_resampler(
        sw,
        sh,
        FilteredResampler::new(ow, oh, kernel),
      )
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
      yuv444p_to(&src, FR, M, &mut sink).unwrap();
    }
    (luma, luma_u16)
  }
}

/// Builds per-format ramp planes (`sw x sh` luma, subsampling-correct chroma).
fn ramp_planes<F: Yuva8Filter>(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let (cw, ch) = F::chroma_dims(sw, sh);
  (
    luma_ramp(sw * sh),
    u_ramp(cw * ch),
    v_ramp(cw * ch),
    alpha_ramp(sw * sh),
  )
}

// ---- Cross-format native-Y luma parity with the no-alpha YUV (the
// ---- consistency contract; feature-independent) -----------------------
//
// The consistency contract: the **same** Y plane must filter to the **same**
// luma whether or not an alpha plane is attached. The merged no-alpha planar
// YUV formats (`Yuv420p` / `Yuv422p` / `Yuv444p`, in `main`) resample native
// Y through a single-channel `FilterStream<u8>` (8bpc coefficient grid,
// horizontal intermediate clamped to `[0, 255]`), so the 8-bit planar YUVA
// `luma` / `luma_u16` MUST byte-match the no-alpha sibling's for the same
// Y/U/V — merely attaching an alpha plane cannot change the luma. (Before the
// `FilterStream<u8>` routing the YUVA luma rode a `FilterStream<u16>` over
// the zero-extended Y, which preserves signed-kernel overshoot at full
// precision and so diverged from the u8 path under CatmullRom / Lanczos3 —
// the bug this guards against.) No packed-RGB oracle, so feature-independent
// — it also guards the `yuva`-solo build.

/// Single-channel filter resample of a u8 Y plane via the merged engine's
/// [`FilterStream<u8>`] (channels = 1) — the same stream the no-alpha planar
/// YUV formats and (now) the 8-bit planar YUVA luma use, so a format's filter
/// `luma` must equal this byte-for-bit.
fn native_y_filter_u8<K: FilterKernel>(
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
  let mut out = std::vec![0u8; ow * oh];
  for row in 0..sh {
    stream
      .feed_row(row, &y[row * sw..(row + 1) * sw], true, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// **The cross-format parity assertion (the consistency contract).** A
/// format's filter `luma` AND `luma_u16` must equal the no-alpha sibling
/// YUV's filter `luma` / `luma_u16` over the SAME Y/U/V planes — byte-for-bit
/// (max diff 0). Also asserts both equal the standalone single-channel
/// `FilterStream<u8>` of the Y plane (`luma_u16` is the u8 luma
/// zero-extended). Returns the max per-sample `luma` diff (exactly 0).
fn assert_luma_matches_no_alpha_yuv<F: Yuva8Filter, K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u8 {
  let (y, u, v, a) = ramp_planes::<F>(sw, sh);
  let got = F::filter_outputs(&y, &u, &v, &a, sw, sh, ow, oh, kernel);
  let (yuv_luma, yuv_luma_u16) = F::no_alpha_yuv_luma(&y, &u, &v, sw, sh, ow, oh, kernel);
  // Independent oracle: the bare single-channel `FilterStream<u8>` over Y.
  let raw_u8 = native_y_filter_u8(kernel, &y, sw, sh, ow, oh);

  // The no-alpha YUV's own luma must equal the bare u8 stream (sanity on the
  // oracle itself), and YUVA must equal both.
  assert_eq!(
    yuv_luma, raw_u8,
    "{ctx}: no-alpha YUV luma must be the single-channel u8 Y filter (oracle sanity)"
  );

  let mut max_diff = 0u8;
  for (i, (&g, &w)) in got.luma.iter().zip(yuv_luma.iter()).enumerate() {
    max_diff = max_diff.max(g.abs_diff(w));
    assert_eq!(
      g, w,
      "{ctx} luma[{i}]: YUVA {g} vs no-alpha YUV {w} — attaching alpha must not change luma"
    );
  }
  for (i, (&g, &w)) in got.luma_u16.iter().zip(yuv_luma_u16.iter()).enumerate() {
    assert_eq!(
      g, w,
      "{ctx} luma_u16[{i}]: YUVA {g} vs no-alpha YUV {w} — attaching alpha must not change luma_u16"
    );
  }
  // luma_u16 is the u8 luma zero-extended.
  for (&hi, &lo) in got.luma_u16.iter().zip(got.luma.iter()) {
    assert_eq!(
      hi, lo as u16,
      "{ctx}: luma_u16 must be the u8 luma zero-extended (>> 0 / << 0)"
    );
  }
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_luma_filter_equals_yuv420p() {
  // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; YUVA luma / luma_u16
  // must byte-match the no-alpha `Yuv420p` filter (max diff 0).
  assert_luma_matches_no_alpha_yuv::<Yuva420pF, _>(Triangle, 8, 8, 4, 4, "yuva420p triangle down");
  assert_luma_matches_no_alpha_yuv::<Yuva420pF, _>(
    CatmullRom,
    8,
    8,
    4,
    4,
    "yuva420p catmullrom down",
  );
  assert_luma_matches_no_alpha_yuv::<Yuva420pF, _>(Lanczos3, 8, 8, 4, 4, "yuva420p lanczos3 down");
  assert_luma_matches_no_alpha_yuv::<Yuva420pF, _>(Triangle, 4, 4, 7, 7, "yuva420p triangle up");
  assert_luma_matches_no_alpha_yuv::<Yuva420pF, _>(
    CatmullRom,
    4,
    4,
    7,
    7,
    "yuva420p catmullrom up",
  );
  assert_luma_matches_no_alpha_yuv::<Yuva420pF, _>(Lanczos3, 4, 4, 7, 7, "yuva420p lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_luma_filter_equals_yuv422p() {
  assert_luma_matches_no_alpha_yuv::<Yuva422pF, _>(Triangle, 8, 8, 4, 4, "yuva422p triangle down");
  assert_luma_matches_no_alpha_yuv::<Yuva422pF, _>(
    CatmullRom,
    8,
    8,
    4,
    4,
    "yuva422p catmullrom down",
  );
  assert_luma_matches_no_alpha_yuv::<Yuva422pF, _>(Lanczos3, 8, 8, 4, 4, "yuva422p lanczos3 down");
  assert_luma_matches_no_alpha_yuv::<Yuva422pF, _>(Triangle, 4, 4, 7, 7, "yuva422p triangle up");
  assert_luma_matches_no_alpha_yuv::<Yuva422pF, _>(
    CatmullRom,
    4,
    4,
    7,
    7,
    "yuva422p catmullrom up",
  );
  assert_luma_matches_no_alpha_yuv::<Yuva422pF, _>(Lanczos3, 4, 4, 7, 7, "yuva422p lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_luma_filter_equals_yuv444p() {
  assert_luma_matches_no_alpha_yuv::<Yuva444pF, _>(Triangle, 8, 8, 4, 4, "yuva444p triangle down");
  assert_luma_matches_no_alpha_yuv::<Yuva444pF, _>(
    CatmullRom,
    8,
    8,
    4,
    4,
    "yuva444p catmullrom down",
  );
  assert_luma_matches_no_alpha_yuv::<Yuva444pF, _>(Lanczos3, 8, 8, 4, 4, "yuva444p lanczos3 down");
  assert_luma_matches_no_alpha_yuv::<Yuva444pF, _>(Triangle, 4, 4, 7, 7, "yuva444p triangle up");
  assert_luma_matches_no_alpha_yuv::<Yuva444pF, _>(
    CatmullRom,
    4,
    4,
    7,
    7,
    "yuva444p catmullrom up",
  );
  assert_luma_matches_no_alpha_yuv::<Yuva444pF, _>(Lanczos3, 4, 4, 7, 7, "yuva444p lanczos3 up");
}

// ---- The parity test discriminates the u8 vs u16 luma stream -----------
//
// Proves the parity assertion above is NOT vacuous: over a bright Y quadrant
// (a 255 block in the top-left, 0 elsewhere) resized under a signed kernel,
// the (pre-fix) u16-luma stream — a single-channel `FilterStream<u16>` over
// the zero-extended Y, clamped to the 8-bit native max — DIFFERS from the
// u8-luma stream the fix routes through. The two-axis transition makes both
// the horizontal and vertical passes non-trivial: the u8 stream clamps the
// horizontal intermediate to `[0, 255]` between passes, the u16 stream keeps
// the overshoot at full precision, so the final samples differ near the block
// corner. The YUVA `luma_u16` must equal the u8 path (== the no-alpha `Yuv*p`
// filter) and must NOT equal the divergent u16 path there — so oracling the
// old u16 behavior would FAIL on the fixed code, i.e. the same-Y consistency
// test genuinely catches the bug.

/// A bright Y quadrant: `255` in the top-left `(sh/2) x (sw/2)` block, `0`
/// elsewhere. The sharp transition in BOTH axes makes the separable filter's
/// horizontal and vertical passes both non-trivial, so the u8 stream's
/// between-pass `[0, 255]` clamp diverges from the u16 stream's full-precision
/// carry near the corner.
fn quadrant_y(sw: usize, sh: usize) -> Vec<u8> {
  let mut y = std::vec![0u8; sw * sh];
  for r in 0..sh / 2 {
    for c in 0..sw / 2 {
      y[r * sw + c] = 255;
    }
  }
  y
}

/// Single-channel `FilterStream<u16>` of the zero-extended Y, clamped to the
/// 8-bit native max — the EXACT pre-fix YUVA luma behavior, kept here only to
/// prove the new parity test discriminates it from the u8 path.
fn pre_fix_u16_luma<K: FilterKernel>(
  kernel: K,
  y: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u16> {
  let zext: Vec<u16> = y.iter().map(|&b| b as u16).collect();
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
      .feed_row(row, &zext[row * sw..(row + 1) * sw], true, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  // The pre-fix path clamped the binned native Y to the 8-bit native max.
  for v in out.iter_mut() {
    *v = (*v).min(NATIVE_MAX);
  }
  out
}

/// Drives a near-ceiling bright-Y step (neutral chroma, opaque) enlarged
/// 4 -> 7 and asserts: (1) the YUVA luma_u16 equals the no-alpha YUV / u8
/// path; (2) the pre-fix u16 path differs from it somewhere; (3) at every such
/// divergence the YUVA luma_u16 tracks the u8 path, NOT the u16 path. Returns
/// `(divergent_position, yuva_luma_u16, pre_fix_u16_luma)` for reporting.
fn assert_parity_discriminates_u16<F: Yuva8Filter, K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> (usize, u16, u16) {
  let (cw, ch) = F::chroma_dims(sw, sh);
  let y = quadrant_y(sw, sh);
  let u = std::vec![128u8; cw * ch];
  let v = std::vec![128u8; cw * ch];
  let a = std::vec![255u8; sw * sh];
  let got = F::filter_outputs(&y, &u, &v, &a, sw, sh, ow, oh, kernel);

  let (yuv_luma, yuv_luma_u16) = F::no_alpha_yuv_luma(&y, &u, &v, sw, sh, ow, oh, kernel);
  let pre_fix = pre_fix_u16_luma(kernel, &y, sw, sh, ow, oh);

  // (1) YUVA luma / luma_u16 == the no-alpha YUV (u8 path), byte-for-bit.
  assert_eq!(
    got.luma, yuv_luma,
    "{ctx}: YUVA luma must equal the no-alpha YUV luma (u8 path)"
  );
  assert_eq!(
    got.luma_u16, yuv_luma_u16,
    "{ctx}: YUVA luma_u16 must equal the no-alpha YUV luma_u16 (u8 path)"
  );

  // (2) The pre-fix u16 path must differ from the u8 path somewhere (else this
  //     discrimination test is vacuous — the two streams would be identical).
  let diverge = got
    .luma_u16
    .iter()
    .zip(pre_fix.iter())
    .position(|(&u8path, &u16path)| u8path != u16path)
    .unwrap_or_else(|| {
      panic!("{ctx}: the u8 and pre-fix u16 luma streams never diverge — test is vacuous")
    });

  // (3) At the divergence the YUVA luma_u16 tracks the u8 path, not the u16
  //     path (so oracling the u16 behavior would FAIL on the fixed code).
  assert_ne!(
    got.luma_u16[diverge], pre_fix[diverge],
    "{ctx}: at the divergence YUVA luma_u16 must NOT equal the pre-fix u16 stream"
  );
  assert_eq!(
    got.luma_u16[diverge], yuv_luma_u16[diverge],
    "{ctx}: at the divergence YUVA luma_u16 must equal the u8 no-alpha YUV luma_u16"
  );

  (diverge, got.luma_u16[diverge], pre_fix[diverge])
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_luma_parity_discriminates_u16_stream() {
  // Both 8 -> 4 (down) and 4 -> 7 (up) put the u8 and pre-fix u16 streams in
  // observable disagreement near the bright block corner.
  assert_parity_discriminates_u16::<Yuva420pF, _>(
    CatmullRom,
    8,
    8,
    4,
    4,
    "yuva420p catmullrom down",
  );
  assert_parity_discriminates_u16::<Yuva420pF, _>(Lanczos3, 8, 8, 4, 4, "yuva420p lanczos3 down");
  assert_parity_discriminates_u16::<Yuva420pF, _>(CatmullRom, 4, 4, 7, 7, "yuva420p catmullrom up");
  assert_parity_discriminates_u16::<Yuva420pF, _>(Lanczos3, 4, 4, 7, 7, "yuva420p lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_luma_parity_discriminates_u16_stream() {
  assert_parity_discriminates_u16::<Yuva422pF, _>(
    CatmullRom,
    8,
    8,
    4,
    4,
    "yuva422p catmullrom down",
  );
  assert_parity_discriminates_u16::<Yuva422pF, _>(Lanczos3, 8, 8, 4, 4, "yuva422p lanczos3 down");
  assert_parity_discriminates_u16::<Yuva422pF, _>(CatmullRom, 4, 4, 7, 7, "yuva422p catmullrom up");
  assert_parity_discriminates_u16::<Yuva422pF, _>(Lanczos3, 4, 4, 7, 7, "yuva422p lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_luma_parity_discriminates_u16_stream() {
  assert_parity_discriminates_u16::<Yuva444pF, _>(
    CatmullRom,
    8,
    8,
    4,
    4,
    "yuva444p catmullrom down",
  );
  assert_parity_discriminates_u16::<Yuva444pF, _>(Lanczos3, 8, 8, 4, 4, "yuva444p lanczos3 down");
  assert_parity_discriminates_u16::<Yuva444pF, _>(CatmullRom, 4, 4, 7, 7, "yuva444p catmullrom up");
  assert_parity_discriminates_u16::<Yuva444pF, _>(Lanczos3, 4, 4, 7, 7, "yuva444p lanczos3 up");
}

// ---- u8 colour in-range (feature-independent) -------------------------
//
// These sources expose no u16 colour outputs, so the only colour the filter
// produces is u8 (rgb / rgba). The `FilterStream<u8>` clamps every channel to
// `[0, 255]`, so a signed-kernel overshoot cannot wrap the u8 colour (no
// native-max clamp needed on this path — unlike the native-Y luma above).

fn assert_u8_color_in_range<F: Yuva8Filter, K: FilterKernel + Copy>(kernel: K, ctx: &str) {
  const SW: usize = 4;
  const SD: usize = 7;
  let (cw, ch) = F::chroma_dims(SW, SW);
  let y = step_y(SW, SW);
  let u = std::vec![128u8; cw * ch];
  let v = std::vec![128u8; cw * ch];
  let a = std::vec![255u8; SW * SW];
  let got = F::filter_outputs(&y, &u, &v, &a, SW, SW, SD, SD, kernel);
  // The bright plateau pins colour at the ceiling, so a saturated (== 255)
  // edge must exist (the kernel really pushes against the clamp).
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
fn yuva420p_u8_color_in_range_at_step_edge() {
  assert_u8_color_in_range::<Yuva420pF, _>(CatmullRom, "yuva420p catmullrom");
  assert_u8_color_in_range::<Yuva420pF, _>(Lanczos3, "yuva420p lanczos3");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_u8_color_in_range_at_step_edge() {
  assert_u8_color_in_range::<Yuva422pF, _>(CatmullRom, "yuva422p catmullrom");
  assert_u8_color_in_range::<Yuva422pF, _>(Lanczos3, "yuva422p lanczos3");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_u8_color_in_range_at_step_edge() {
  assert_u8_color_in_range::<Yuva444pF, _>(CatmullRom, "yuva444p catmullrom");
  assert_u8_color_in_range::<Yuva444pF, _>(Lanczos3, "yuva444p lanczos3");
}

// ---- Filter-plan-accepted regression (feature-independent) ------------

/// A filter plan must be accepted by the 8-bit planar YUVA sink — before this
/// routing it was rejected with `UnsupportedFilter`; now it produces a real
/// (non-sentinel) output. Feature-independent, so it guards the `yuva`-solo
/// build.
fn assert_filter_plan_accepted<F: Yuva8Filter>(ctx: &str) {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, u, v, a) = ramp_planes::<F>(SW, SH);
  let got = F::filter_outputs(&y, &u, &v, &a, SW, SH, OW, OH, Triangle);
  assert!(
    got.rgba.iter().any(|&val| val != 0),
    "{ctx}: filter resample must populate rgba (no UnsupportedFilter)"
  );
  assert!(
    got.luma_u16.iter().any(|&val| val != 0),
    "{ctx}: filter resample must populate luma_u16 (no UnsupportedFilter)"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_filter_plan_is_accepted() {
  assert_filter_plan_accepted::<Yuva420pF>("yuva420p");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_filter_plan_is_accepted() {
  assert_filter_plan_accepted::<Yuva422pF>("yuva422p");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_filter_plan_is_accepted() {
  assert_filter_plan_accepted::<Yuva444pF>("yuva444p");
}

// ---- Packed-RGBA (8-bit) equivalence oracle (gated on `rgb`) ----------
//
// The filter path converts the YUVA to a canonical u8 `R, G, B, A` row with
// the same `yuva{420,444}p_to_rgba_row` kernel the direct sink uses, then
// filters the four channels independently. So a YUVA filter colour output
// equals the equivalent 8-bit `Rgba` filter resample of those exact converted
// pixels: `rgba` == the `Rgba` filter (per-channel, alpha a real filtered
// channel), `rgb` == its alpha drop. The `FilterStream<u8>` is byte-exact per
// channel, so the max diff is 0.

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
  fn assert_color_equals_packed_rgba<F: Yuva8Filter, K: FilterKernel + Copy>(
    kernel: K,
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    ctx: &str,
  ) -> u8 {
    let (y, u, v, a) = ramp_planes::<F>(sw, sh);
    let got = F::filter_outputs(&y, &u, &v, &a, sw, sh, ow, oh, kernel);

    // rgba == the 8-bit Rgba filter of the converted canonical RGBA.
    let canonical = F::direct_rgba_u8(&y, &u, &v, &a, sw, sh);
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
  fn yuva420p_color_filter_equals_packed_rgba() {
    assert_color_equals_packed_rgba::<Yuva420pF, _>(Triangle, 8, 8, 4, 4, "yuva420p triangle down");
    assert_color_equals_packed_rgba::<Yuva420pF, _>(
      CatmullRom,
      8,
      8,
      4,
      4,
      "yuva420p catmullrom down",
    );
    assert_color_equals_packed_rgba::<Yuva420pF, _>(Lanczos3, 8, 8, 4, 4, "yuva420p lanczos3 down");
    assert_color_equals_packed_rgba::<Yuva420pF, _>(Triangle, 4, 4, 7, 7, "yuva420p triangle up");
    assert_color_equals_packed_rgba::<Yuva420pF, _>(
      CatmullRom,
      4,
      4,
      7,
      7,
      "yuva420p catmullrom up",
    );
    assert_color_equals_packed_rgba::<Yuva420pF, _>(Lanczos3, 4, 4, 7, 7, "yuva420p lanczos3 up");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuva422p_color_filter_equals_packed_rgba() {
    assert_color_equals_packed_rgba::<Yuva422pF, _>(Triangle, 8, 8, 4, 4, "yuva422p triangle down");
    assert_color_equals_packed_rgba::<Yuva422pF, _>(
      CatmullRom,
      8,
      8,
      4,
      4,
      "yuva422p catmullrom down",
    );
    assert_color_equals_packed_rgba::<Yuva422pF, _>(Lanczos3, 8, 8, 4, 4, "yuva422p lanczos3 down");
    assert_color_equals_packed_rgba::<Yuva422pF, _>(Triangle, 4, 4, 7, 7, "yuva422p triangle up");
    assert_color_equals_packed_rgba::<Yuva422pF, _>(
      CatmullRom,
      4,
      4,
      7,
      7,
      "yuva422p catmullrom up",
    );
    assert_color_equals_packed_rgba::<Yuva422pF, _>(Lanczos3, 4, 4, 7, 7, "yuva422p lanczos3 up");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuva444p_color_filter_equals_packed_rgba() {
    assert_color_equals_packed_rgba::<Yuva444pF, _>(Triangle, 8, 8, 4, 4, "yuva444p triangle down");
    assert_color_equals_packed_rgba::<Yuva444pF, _>(
      CatmullRom,
      8,
      8,
      4,
      4,
      "yuva444p catmullrom down",
    );
    assert_color_equals_packed_rgba::<Yuva444pF, _>(Lanczos3, 8, 8, 4, 4, "yuva444p lanczos3 down");
    assert_color_equals_packed_rgba::<Yuva444pF, _>(Triangle, 4, 4, 7, 7, "yuva444p triangle up");
    assert_color_equals_packed_rgba::<Yuva444pF, _>(
      CatmullRom,
      4,
      4,
      7,
      7,
      "yuva444p catmullrom up",
    );
    assert_color_equals_packed_rgba::<Yuva444pF, _>(Lanczos3, 4, 4, 7, 7, "yuva444p lanczos3 up");
  }
}
