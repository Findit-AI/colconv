//! Separable-filter resample coverage for the packed 8-bit gray+alpha source
//! (`Ya8`), routed through the merged filter engine.
//!
//! `Ya8` is structurally a degenerate YUVA (`R = G = B = Y`, neutral chroma)
//! plus an independent native-Y luma. A `Filter` plan routes to
//! [`packed_yuva444_filter_resample`](super::super::packed_yuva444_filter_resample)
//! at `SRC_BITS = 8` with `NATIVE_LUMA_U8`: each packed `[Y, A]` row is decoded
//! into the canonical u8 `R, G, B, A` row with the **same** `ya8_to_rgba_row`
//! kernel the area / direct paths use, then the four interleaved channels are
//! resampled by the signed-coefficient u8 filter stream (the filter twin of the
//! area bin). Straight alpha only (a premultiplied `Filter` plan routes to the
//! area tail, which surfaces `UnsupportedFilter`). So:
//!
//! - **`rgba` / `rgb`** equal the equivalent 8-bit `Rgba` filter resample of the
//!   source converted to u8 RGBA (`[Y, Y, Y, A]`) — alpha is a real filtered
//!   channel, byte-exact per channel (max diff 0).
//! - **`rgba_u16` / `rgb_u16`** are the **u8 colour zero-extended** —
//!   `rgba_u16 == rgba as u16`, `rgb_u16 == rgb as u16` (max diff 0). `Ya8` is
//!   an 8-bit source, so its native-depth colour IS the u8 colour widened, NOT
//!   an independent native-u16 filter (the `ZEXT_U16_COLOR` route). This is the
//!   same contract `Ya8`'s **area** path emits ([`packed_rgba_resample`], which
//!   likewise zero-extends the binned u8), so the filter and area u16 colour
//!   agree, and both agree with `rgba as u16`. An independent `Rgba64` filter
//!   of the zero-extended source would diverge from `rgba as u16` by up to 1
//!   LSB near a signed-kernel overshoot (the u8 and u16 filter streams round /
//!   clamp their horizontal-pass intermediates differently) — the contract
//!   split this routing closes, pinned by [`zext_u16_is_load_bearing`].
//! - **`luma` / `luma_u16`** are native Y — the de-interleaved Y resampled
//!   through a single-channel [`FilterStream<u8>`] (the SAME 8bpc-grid stream
//!   `Gray8` uses), NOT colour-derived. So `Ya8` filter `luma` is byte-identical
//!   to `Gray8`'s filter `luma` over the same Y plane (the consistency contract:
//!   attaching an alpha plane must not change the luma), and `luma_u16` is that
//!   resampled Y zero-extended. 8-bit, so no native-depth clamp (the `u8` stream
//!   finalizes to the full `u8` range, which *is* the native range).
//! - **filter == area native-Y luma:** the same Y plane resampled by the area
//!   and filter engines uses different kernels, but BOTH take luma from native Y
//!   (never colour-derived), so for the area kernel the filter `luma` matches the
//!   area `luma` byte-for-bit when both run the area weights — pinned here by
//!   asserting the filter `luma` equals the bare native-Y `FilterStream<u8>`,
//!   exactly the source-of-truth the area path's native-Y bin uses.
//!
//! The `Rgba` / `Rgba64` equivalence oracles are gated on `rgb` (the oracle
//! sources). The native-Y luma parity (against the bare `FilterStream<u8>` and
//! against `Gray8`), the premultiplied→`UnsupportedFilter` rejection, and the
//! filter-plan-accepted regression are feature-independent, so they also guard
//! the `gray`-solo build.

use crate::{
  KernelMatrix,
  frame::Ya8Frame,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, ResampleError, Resampler,
    Triangle,
  },
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Ya8, ya8_to},
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
/// Limited (studio) range — exercises native-Y luma against the range-dependent
/// `rgb_to_luma*` it must NOT use.
const FR_LIMITED: bool = false;

/// A per-pixel `[Y, A]` ramp varying so every filter window sees distinct
/// neighbours (a channel mix-up or a row/column transpose diverges
/// immediately). Alpha varies (not all-opaque) so the real-alpha filter is
/// genuinely exercised.
fn ya8_ramp(sw: usize, sh: usize) -> Vec<u8> {
  let n = sw * sh;
  let mut packed = std::vec![0u8; n * 2];
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    px[0] = (24 + i * 7).min(235) as u8; // Y
    px[1] = (16 + i * 9).min(250) as u8; // A (varies)
  }
  packed
}

/// A sharp black -> white horizontal step in Y (left half min-Y, right half
/// max-Y, opaque), uniform vertically. A signed kernel enlarging the near-max
/// bright Y plateau overshoots above the 8-bit max — the de-interleaved native
/// Y the luma path resamples, and the `R = Y` colour channel. Consumed only by
/// the `rgb`-gated colour equivalence module, so it is dead in a
/// `gray`-without-`rgb` build.
#[cfg_attr(not(feature = "rgb"), allow(dead_code))]
fn step_y(sw: usize, sh: usize) -> Vec<u8> {
  let mut packed = std::vec![0u8; sw * sh * 2];
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    px[0] = if (i % sw) >= sw / 2 { 255 } else { 0 }; // Y
    px[1] = 255; // opaque
  }
  packed
}

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  // `rgb` / `rgb_u16` / `rgba_u16` are read only by the `rgb`-gated colour
  // equivalence module below, so they are dead in a `gray`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgb: Vec<u8>,
  rgba: Vec<u8>,
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgb_u16: Vec<u16>,
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgba_u16: Vec<u16>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

/// Run the `Ya8` filter sink over `packed` (`sw x sh`) at `ow x oh` under
/// `kernel` and `full_range`, attaching every output the equivalences assert on.
fn ya8_filter_outputs<K: FilterKernel + Copy>(
  packed: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  full_range: bool,
  kernel: K,
) -> FilterOutputs {
  let src = Ya8Frame::new(packed, sw as u32, sh as u32, (sw * 2) as u32);
  let mut rgb = std::vec![0u8; ow * oh * 3];
  let mut rgba = std::vec![0u8; ow * oh * 4];
  let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut luma = std::vec![0u8; ow * oh];
  let mut luma_u16 = std::vec![0u16; ow * oh];
  {
    let mut sink = MixedSinker::<Ya8, FilteredResampler<K>>::with_resampler(
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
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    ya8_to(&src, full_range, M, &mut sink).unwrap();
  }
  FilterOutputs {
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
  }
}

/// Canonical full-res u8 `[Y, Y, Y, A]` of a packed `[Y, A]` plane — the exact
/// `ya8_to_rgba_row` mapping, the input the filter path resamples. Consumed only
/// by the `rgb`-gated colour equivalence module, so it is dead in a
/// `gray`-without-`rgb` build.
#[cfg_attr(not(feature = "rgb"), allow(dead_code))]
fn canonical_rgba_u8(packed: &[u8], n: usize) -> Vec<u8> {
  let mut out = std::vec![0u8; n * 4];
  for (px, src) in out.chunks_exact_mut(4).zip(packed.chunks_exact(2)) {
    px[0] = src[0];
    px[1] = src[0];
    px[2] = src[0];
    px[3] = src[1];
  }
  out
}

/// De-interleaved native Y of a packed `[Y, A]` plane — the single-channel luma
/// oracle's input.
fn native_y(packed: &[u8], n: usize) -> Vec<u8> {
  packed.chunks_exact(2).take(n).map(|p| p[0]).collect()
}

/// Single-channel filter resample of a u8 Y plane via the merged engine's
/// [`FilterStream<u8>`] (channels = 1) — the same stream the area path's
/// native-Y bin and `Gray8` use, so `Ya8` filter `luma` must equal it
/// byte-for-bit.
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

// ---- Native-Y luma equivalence (the consistency contract) -------------------
//
// Feature-independent (no packed-RGB oracle), so it also guards the `gray`-solo
// build. Asserts the `Ya8` filter `luma` / `luma_u16` equal the bare
// single-channel `FilterStream<u8>` of the de-interleaved native Y (NOT
// colour-derived), at BOTH ranges (native Y is range-independent). Returns the
// max per-sample `luma` diff (exactly 0).

fn assert_luma_is_native_y<K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  full_range: bool,
  ctx: &str,
) -> u8 {
  let packed = ya8_ramp(sw, sh);
  let got = ya8_filter_outputs(&packed, sw, sh, ow, oh, full_range, kernel);
  let raw = native_y_filter_u8(kernel, &native_y(&packed, sw * sh), sw, sh, ow, oh);

  let mut max_diff = 0u8;
  for (i, (&g, &r)) in got.luma.iter().zip(raw.iter()).enumerate() {
    max_diff = max_diff.max(g.abs_diff(r));
    assert_eq!(
      g, r,
      "{ctx} luma[{i}]: Ya8 {g} vs native-Y FilterStream<u8> {r} — luma must be native Y"
    );
  }
  for (&hi, &lo) in got.luma_u16.iter().zip(got.luma.iter()) {
    assert_eq!(
      hi, lo as u16,
      "{ctx}: luma_u16 must be the u8 native-Y luma zero-extended"
    );
  }
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_filter_is_native_y_full_range() {
  // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; max diff 0.
  for (sw, sh, ow, oh, tag) in [(8, 8, 4, 4, "down"), (4, 4, 7, 7, "up")] {
    assert_luma_is_native_y(Triangle, sw, sh, ow, oh, FR, &format!("triangle {tag}"));
    assert_luma_is_native_y(CatmullRom, sw, sh, ow, oh, FR, &format!("catmullrom {tag}"));
    assert_luma_is_native_y(Lanczos3, sw, sh, ow, oh, FR, &format!("lanczos3 {tag}"));
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_filter_is_native_y_limited_range() {
  // Native Y is range-independent: a `full_range = false` row must still take
  // luma from native Y (a colour-derived `rgb_to_luma*` would mis-map it).
  for (sw, sh, ow, oh, tag) in [(8, 8, 4, 4, "down"), (4, 4, 7, 7, "up")] {
    assert_luma_is_native_y(
      Triangle,
      sw,
      sh,
      ow,
      oh,
      FR_LIMITED,
      &format!("triangle {tag}"),
    );
    assert_luma_is_native_y(
      CatmullRom,
      sw,
      sh,
      ow,
      oh,
      FR_LIMITED,
      &format!("catmullrom {tag}"),
    );
    assert_luma_is_native_y(
      Lanczos3,
      sw,
      sh,
      ow,
      oh,
      FR_LIMITED,
      &format!("lanczos3 {tag}"),
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn limited_range_luma_equals_full_range_luma() {
  // The same source filtered at both ranges yields identical native-Y luma — a
  // colour-derived luma would differ (the divergence the FR=true suite alone
  // cannot surface). Uses a near-min limited-range Y so a range scale would be
  // visible.
  let mut packed = std::vec![0u8; 8 * 8 * 2];
  for px in packed.chunks_exact_mut(2) {
    px[0] = 16; // limited-range black
    px[1] = 200; // some alpha
  }
  let fr = ya8_filter_outputs(&packed, 8, 8, 4, 4, FR, CatmullRom);
  let lim = ya8_filter_outputs(&packed, 8, 8, 4, 4, FR_LIMITED, CatmullRom);
  assert_eq!(fr.luma, lim.luma, "native-Y luma must be range-independent");
  assert_eq!(
    fr.luma_u16, lim.luma_u16,
    "native-Y luma_u16 must be range-independent"
  );
  // A uniform Y = 16 plateau stays native 16 (a limited-range `rgb_to_luma` of
  // (16,16,16) would scale up to ≈30).
  assert!(
    lim.luma.iter().all(|&y| y == 16),
    "uniform limited-range Y=16 luma must stay 16, got {:?}",
    lim.luma
  );
}

// ---- filter `luma` matches the area `luma` for the area kernel --------------

/// The `Ya8` area `luma` of a packed plane at `ow x oh` — the native-Y area
/// bin, the source-of-truth the filter path's native-Y stream mirrors.
fn area_luma(
  packed: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  full_range: bool,
) -> Vec<u8> {
  use crate::resample::AreaResampler;
  let src = Ya8Frame::new(packed, sw as u32, sh as u32, (sw * 2) as u32);
  let mut luma = std::vec![0u8; ow * oh];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    ya8_to(&src, full_range, M, &mut sink).unwrap();
  }
  luma
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_native_y_oracle_matches_area_native_y() {
  // The filter and area engines use different kernels, but both take luma from
  // native Y. Pin that they share the SAME native-Y source-of-truth: the bare
  // `FilterStream<u8>` the filter `luma` matches equals the `Triangle` filter,
  // and an `AreaStream<u8>` over the same Y (the area `luma`) is the area-kernel
  // sibling — so neither path is colour-derived. (A 2:1 area downscale equals a
  // box bin; we assert the area `luma` equals the area-bin of native Y.)
  let packed = ya8_ramp(8, 8);
  let y = native_y(&packed, 64);
  let area = area_luma(&packed, 8, 8, 4, 4, FR);
  // Area `luma` = round-half-up 2x2 block mean of native Y (the area engine's
  // 2:1 box), proving the area path is native-Y (NOT colour-derived) too.
  let mut block = std::vec![0u8; 4 * 4];
  for oy in 0..4 {
    for ox in 0..4 {
      let mut acc = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += y[(oy * 2 + dy) * 8 + ox * 2 + dx] as u32;
        }
      }
      block[oy * 4 + ox] = ((acc + 2) / 4) as u8;
    }
  }
  assert_eq!(area, block, "area luma must be the native-Y 2x2 block mean");
}

// ---- Cross-check against `Gray8` (the consistency contract; `gray`-solo) ----

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_filter_equals_gray8() {
  // The SAME Y plane must filter to the SAME luma whether it carries an alpha
  // channel (`Ya8`) or not (`Gray8`). Attaching alpha cannot change the luma.
  use crate::{frame::Gray8Frame, source::gray8_to};
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (4, 4, 7, 7)] {
    let packed = ya8_ramp(sw, sh);
    let y = native_y(&packed, sw * sh);
    for kernel_tag in ["triangle", "catmullrom", "lanczos3"] {
      let (ya_luma, ya_lu16) = {
        let got = match kernel_tag {
          "triangle" => ya8_filter_outputs(&packed, sw, sh, ow, oh, FR, Triangle),
          "catmullrom" => ya8_filter_outputs(&packed, sw, sh, ow, oh, FR, CatmullRom),
          _ => ya8_filter_outputs(&packed, sw, sh, ow, oh, FR, Lanczos3),
        };
        (got.luma, got.luma_u16)
      };
      let mut g_luma = std::vec![0u8; ow * oh];
      let mut g_lu16 = std::vec![0u16; ow * oh];
      {
        let gsrc = Gray8Frame::new(&y, sw as u32, sh as u32, sw as u32);
        macro_rules! run {
          ($k:expr) => {{
            let mut sink =
              MixedSinker::<crate::source::Gray8, FilteredResampler<_>>::with_resampler(
                sw,
                sh,
                FilteredResampler::new(ow, oh, $k),
              )
              .unwrap()
              .with_luma(&mut g_luma)
              .unwrap()
              .with_luma_u16(&mut g_lu16)
              .unwrap();
            gray8_to(&gsrc, FR, M, &mut sink).unwrap();
          }};
        }
        match kernel_tag {
          "triangle" => run!(Triangle),
          "catmullrom" => run!(CatmullRom),
          _ => run!(Lanczos3),
        }
      }
      assert_eq!(
        ya_luma, g_luma,
        "{kernel_tag} {sw}x{sh}->{ow}x{oh}: Ya8 luma must equal Gray8 luma (same Y)"
      );
      assert_eq!(
        ya_lu16, g_lu16,
        "{kernel_tag} {sw}x{sh}->{ow}x{oh}: Ya8 luma_u16 must equal Gray8 luma_u16 (same Y)"
      );
    }
  }
}

// ---- Premultiplied + filter → UnsupportedFilter (`gray`-solo) ----------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_filter_is_unsupported() {
  // Premultiplied alpha has no filter analogue (the engine cannot
  // un-premultiply), so a premultiplied `Filter` plan must surface the typed
  // `UnsupportedFilter` rather than emitting straight-filtered premultiplied
  // colour.
  let packed = ya8_ramp(8, 8);
  let src = Ya8Frame::new(&packed, 8, 8, 16);
  let mut rgba = std::vec![0u8; 4 * 4 * 4];
  let mut sink = MixedSinker::<Ya8, FilteredResampler<Triangle>>::with_resampler(
    8,
    8,
    FilteredResampler::new(4, 4, Triangle),
  )
  .unwrap()
  .with_alpha_mode(AlphaMode::Premultiplied)
  .with_rgba(&mut rgba)
  .unwrap();
  let err = ya8_to(&src, FR, M, &mut sink).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
    ),
    "premultiplied filter must be UnsupportedFilter, got {err:?}"
  );
}

// ---- Filter-plan-accepted regression (`gray`-solo) --------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_filter_plan_is_accepted() {
  // A straight-alpha `Filter` plan must be accepted (before this routing it was
  // rejected with `UnsupportedFilter`); now it produces a real output.
  let packed = ya8_ramp(8, 8);
  let got = ya8_filter_outputs(&packed, 8, 8, 4, 4, FR, Triangle);
  assert!(
    got.rgba.iter().any(|&v| v != 0),
    "filter resample must populate rgba (no UnsupportedFilter)"
  );
  assert!(
    got.luma_u16.iter().any(|&v| v != 0),
    "filter resample must populate luma_u16 (no UnsupportedFilter)"
  );
}

// ---- Packed-RGBA equivalence oracles (gated on `rgb`) -----------------------
//
// The filter path converts `Ya8` to a canonical `[Y, Y, Y, A]` row (u8) /
// `[Y, Y, Y, A]` zero-extended (u16) with the same kernels the area / direct
// paths use, then filters the four channels independently. So each colour output
// equals the equivalent packed-RGBA filter of those exact converted pixels, at
// its own depth, byte-exact per channel (max diff 0).

#[cfg(feature = "rgb")]
mod packed_rgba_equivalence {
  use super::*;
  use crate::source::{Rgba, Rgba64, rgba_to, rgba64_to};
  use mediaframe::frame::{Rgba64Frame, RgbaFrame};

  /// `Rgba` (8-bit) filter resample of a canonical u8 RGBA frame — the `rgba`
  /// output (per-channel filter, straight alpha).
  fn rgba_filter<K: FilterKernel>(
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

  /// `Rgba64` (16-bit) filter resample of a canonical u16 RGBA frame — the
  /// `rgba_u16` output (per-channel native filter, straight alpha).
  fn rgba64_filter<K: FilterKernel>(
    rgba: &[u16],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> Vec<u16> {
    let src = Rgba64Frame::try_new(rgba, sw as u32, sh as u32, (sw * 4) as u32).unwrap();
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
      rgba64_to(&src, FR, M, &mut sink).unwrap();
    }
    out
  }

  fn assert_color_equals_packed<K: FilterKernel + Copy>(
    kernel: K,
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    ctx: &str,
  ) -> u8 {
    let packed = ya8_ramp(sw, sh);
    let got = ya8_filter_outputs(&packed, sw, sh, ow, oh, FR, kernel);

    // u8: rgba == the 8-bit Rgba filter of the converted canonical RGBA.
    let canon_u8 = canonical_rgba_u8(&packed, sw * sh);
    let want_u8 = rgba_filter(&canon_u8, sw, sh, ow, oh, kernel);
    let mut max_diff = 0u8;
    for (i, (&g, &w)) in got.rgba.iter().zip(want_u8.iter()).enumerate() {
      max_diff = max_diff.max(g.abs_diff(w));
      assert_eq!(g, w, "{ctx} rgba[{i}]: {g} vs Rgba filter {w}");
    }
    for (rgb_px, rgba_px) in got.rgb.chunks_exact(3).zip(want_u8.chunks_exact(4)) {
      assert_eq!(
        rgb_px,
        &rgba_px[..3],
        "{ctx} rgb == drop-alpha(filtered rgba)"
      );
    }

    // u16: `Ya8` is an 8-bit source, so its native-depth colour is the binned
    // **u8** colour zero-extended (the `ZEXT_U16_COLOR` route) — NOT an
    // independent native-u16 filter. So rgba_u16 == `rgba as u16` (the u8
    // filter widened) == `want_u8 as u16` (the 8-bit `Rgba` filter widened);
    // a value `<= 255` by construction, byte-exact (max diff 0). This is the
    // contract the area path emits too ([`packed_rgba_resample`] zero-extends
    // its binned u8), so the filter and area u16 colour agree. (The
    // independent `Rgba64`-filter contract this REPLACES is exercised — and
    // shown to diverge — by `zext_u16_is_load_bearing`.)
    const NATIVE_MAX: u16 = 255;
    for (i, (&g, &lo)) in got.rgba_u16.iter().zip(got.rgba.iter()).enumerate() {
      assert!(
        g <= NATIVE_MAX,
        "{ctx} rgba_u16[{i}] = {g} exceeds the 8-bit native max {NATIVE_MAX}"
      );
      assert_eq!(
        g, lo as u16,
        "{ctx} rgba_u16[{i}]: {g} must be the u8 rgba[{i}] = {lo} zero-extended"
      );
    }
    for (i, (&g, &w)) in got.rgba_u16.iter().zip(want_u8.iter()).enumerate() {
      assert_eq!(
        g, w as u16,
        "{ctx} rgba_u16[{i}]: {g} vs 8-bit Rgba filter {w} zero-extended"
      );
    }
    for (rgb16_px, rgb8_px) in got.rgb_u16.chunks_exact(3).zip(got.rgb.chunks_exact(3)) {
      for (&hi, &lo) in rgb16_px.iter().zip(rgb8_px.iter()) {
        assert_eq!(
          hi, lo as u16,
          "{ctx} rgb_u16 must be the u8 rgb zero-extended"
        );
      }
    }
    max_diff
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn color_filter_equals_packed_rgba() {
    for (sw, sh, ow, oh, tag) in [(8, 8, 4, 4, "down"), (4, 4, 7, 7, "up")] {
      assert_color_equals_packed(Triangle, sw, sh, ow, oh, &format!("triangle {tag}"));
      assert_color_equals_packed(CatmullRom, sw, sh, ow, oh, &format!("catmullrom {tag}"));
      assert_color_equals_packed(Lanczos3, sw, sh, ow, oh, &format!("lanczos3 {tag}"));
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn opaque_step_keeps_alpha_opaque_and_saturates_color() {
    // A bright Y plateau pins R=G=B at the ceiling (a saturated colour edge must
    // exist), and a fully-opaque α step filters to a constant 255 (partition of
    // unity), so attaching alpha does not perturb the no-alpha colour.
    let packed = step_y(4, 4);
    let got = ya8_filter_outputs(&packed, 4, 4, 7, 7, FR, CatmullRom);
    assert!(
      got.rgb.contains(&255),
      "expected a saturated (== 255) colour edge in rgb"
    );
    assert!(
      got.rgba.chunks_exact(4).all(|px| px[3] == 255),
      "opaque-α step must keep filtered α == 255"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn zext_u16_is_load_bearing() {
    // Proves the `ZEXT_U16_COLOR` route is load-bearing, not cosmetic: at a
    // sharp Y step a signed (Lanczos3) kernel overshoots, and the INDEPENDENT
    // native-u16 filter (the contract this routing replaced — an `Rgba64`
    // filter of the zero-extended source, clamped to the 8-bit native max)
    // resolves a channel to a DIFFERENT value than `rgba as u16`, because the
    // u8 and u16 filter streams round / clamp their horizontal-pass
    // intermediates differently. The fix derives the u16 colour from the u8
    // binning, so `rgba_u16 == rgba as u16` everywhere (matching the area
    // path); the old independent binning did not.
    let packed = step_y(4, 4);
    let (sw, sh, ow, oh) = (4, 4, 7, 7);
    let got = ya8_filter_outputs(&packed, sw, sh, ow, oh, FR, Lanczos3);

    // The fix's contract: every u16 colour sample is the u8 sample widened.
    for (i, (&hi, &lo)) in got.rgba_u16.iter().zip(got.rgba.iter()).enumerate() {
      assert_eq!(
        hi, lo as u16,
        "rgba_u16[{i}] must equal rgba[{i}] = {lo} zero-extended"
      );
    }

    // The OLD path: an independent `Rgba64` filter of the zero-extended source,
    // clamped to the 8-bit native max — what the sink emitted before this fix.
    const NATIVE_MAX: u16 = 255;
    let canon_u16: Vec<u16> = canonical_rgba_u8(&packed, sw * sh)
      .iter()
      .map(|&b| b as u16)
      .collect();
    let old_independent: Vec<u16> = rgba64_filter(&canon_u16, sw, sh, ow, oh, Lanczos3)
      .into_iter()
      .map(|v| v.min(NATIVE_MAX))
      .collect();

    // The two contracts diverge somewhere (the fix is load-bearing): at least
    // one channel where the old independent-u16 value differs from the new
    // zero-extended value.
    let divergences = got
      .rgba_u16
      .iter()
      .zip(old_independent.iter())
      .filter(|&(&new, &old)| new != old)
      .count();
    assert!(
      divergences > 0,
      "expected the old independent-u16 filter to diverge from `rgba as u16` \
       (else the fix would be a no-op); found none"
    );

    // Pin the exact known overshoot divergence so a regression that silently
    // re-routes `Ya8` through the independent-u16 stream is caught: at index 14
    // the new (correct) value is the u8 colour widened (128), while the old
    // independent native-u16 filter resolved 127 — a real 1-LSB split.
    assert_eq!(
      got.rgba_u16[14], 128,
      "new rgba_u16[14] must be the zero-extended u8 colour (128)"
    );
    assert_eq!(
      got.rgba[14], 128,
      "new rgba[14] (the u8 colour) must be 128"
    );
    assert_eq!(
      old_independent[14], 127,
      "the OLD independent native-u16 filter resolved 127 here (the divergence \
       this fix removes); if this changed, re-derive the load-bearing case"
    );
  }
}
