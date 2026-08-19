//! Separable-filter resample coverage for the packed 16-bit gray+alpha source
//! (`Ya16`, LE + BE), routed through the merged filter engine.
//!
//! `Ya16` is the high-bit analogue of `Ya8`: structurally a degenerate full-
//! 16-bit YUVA (`R = G = B = Y`, neutral chroma) plus an independent native-Y
//! luma. A `Filter` plan routes to
//! [`packed_yuva444_filter_resample`](super::super::packed_yuva444_filter_resample)
//! at `SRC_BITS = 16` with `NATIVE_LUMA_U8 = false`: each packed `[Y, A]` u16
//! row is decoded into the canonical host-native u16 `R, G, B, A` row with the
//! **same** `ya16_to_rgba_u16_row::<BE>` kernel the area / direct paths use,
//! then the four interleaved channels are resampled by the signed-coefficient
//! u16 filter stream (full 16-bit, so the `FilterStream`'s `0..=65535` clamp is
//! the native clamp). Straight alpha only (a premultiplied `Filter` plan routes
//! to the area tail, which surfaces `UnsupportedFilter`). So:
//!
//! - **`rgba_u16` / `rgb_u16`** equal the equivalent `Rgba64` filter resample of
//!   the source converted to native u16 RGBA (`[Y, Y, Y, A]`) — alpha is a real
//!   filtered channel, byte-exact per channel (max diff 0).
//! - **`rgba` / `rgb`** equal the equivalent 8-bit `Rgba` filter resample of the
//!   `>> 8` narrowed `[Y, Y, Y, A]` u8 RGBA (the filter tail's u8 colour binning
//!   filters the narrowed colour, like the no-alpha sibling).
//! - **`luma` / `luma_u16`** are native Y — the de-interleaved host-native Y
//!   resampled through a single-channel [`FilterStream<u16>`] (the SAME stream
//!   `Gray16` uses), NOT colour-derived. So `Ya16` filter `luma_u16` is
//!   byte-identical to `Gray16`'s over the same Y plane (the consistency
//!   contract), and `luma` is that resampled Y `>> 8`. Full 16-bit, so the u16
//!   stream's `0..=65535` clamp is the native clamp (no extra clamp).
//!
//! The `Rgba` / `Rgba64` equivalence oracles are gated on `rgb` (the oracle
//! sources). The native-Y luma parity (against the bare `FilterStream<u16>` and
//! against `Gray16`), the LE==BE equivalence, the
//! premultiplied→`UnsupportedFilter` rejection, and the filter-plan-accepted
//! regression are feature-independent, so they also guard the `gray`-solo build.

use crate::{
  KernelMatrix,
  frame::{Ya16BeFrame, Ya16Frame},
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, ResampleError, Resampler,
    Triangle,
  },
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Ya16, ya16_to, ya16_to_endian},
};

const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const FR_LIMITED: bool = false;

/// Re-encode a host-native u16 slice as LE-encoded wire byte storage (the
/// `Ya16Frame` plane contract).
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as BE-encoded wire byte storage.
fn as_be_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// A per-pixel host-native `[Y, A]` u16 ramp varying so every filter window sees
/// distinct neighbours. Alpha varies (not all-opaque) so the real-alpha filter
/// is genuinely exercised. Values span the high bits so `>> 8` narrowing is
/// non-trivial.
fn ya16_ramp(sw: usize, sh: usize) -> Vec<u16> {
  let n = sw * sh;
  let mut packed = std::vec![0u16; n * 2];
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    px[0] = (1500 + i * 900).min(60000) as u16; // Y
    px[1] = (4000 + i * 700).min(65000) as u16; // A (varies)
  }
  packed
}

/// Every resampled output a filter equivalence asserts on.
struct FilterOutputs {
  // `rgb` / `rgba` / `rgb_u16` are read only by the `rgb`-gated colour
  // equivalence module below, so they are dead in a `gray`-without-`rgb` build.
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgb: Vec<u8>,
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgba: Vec<u8>,
  #[cfg_attr(not(feature = "rgb"), allow(dead_code))]
  rgb_u16: Vec<u16>,
  rgba_u16: Vec<u16>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

/// Run the `Ya16` (LE) filter sink over the host-native `packed` (`sw x sh`) at
/// `ow x oh` under `kernel` and `full_range`, attaching every output.
fn ya16_filter_outputs<K: FilterKernel + Copy>(
  packed_host: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  full_range: bool,
  kernel: K,
) -> FilterOutputs {
  let plane = as_le_u16(packed_host);
  let src = Ya16Frame::new(&plane, sw as u32, sh as u32, (sw * 2) as u32);
  let mut rgb = std::vec![0u8; ow * oh * 3];
  let mut rgba = std::vec![0u8; ow * oh * 4];
  let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut luma = std::vec![0u8; ow * oh];
  let mut luma_u16 = std::vec![0u16; ow * oh];
  {
    let mut sink = MixedSinker::<Ya16, FilteredResampler<K>>::with_resampler(
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
    ya16_to(&src, full_range, sink.set_kernel_matrix(M)).unwrap();
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

/// Canonical full-res native u16 `[Y, Y, Y, A]` of a host-native packed `[Y, A]`
/// plane — the `ya16_to_rgba_u16_row` mapping, the u16 filter input. Consumed
/// only by the `rgb`-gated colour equivalence module, so it is dead in a
/// `gray`-without-`rgb` build.
#[cfg_attr(not(feature = "rgb"), allow(dead_code))]
fn canonical_rgba_u16(packed: &[u16], n: usize) -> Vec<u16> {
  let mut out = std::vec![0u16; n * 4];
  for (px, src) in out.chunks_exact_mut(4).zip(packed.chunks_exact(2)) {
    px[0] = src[0];
    px[1] = src[0];
    px[2] = src[0];
    px[3] = src[1];
  }
  out
}

/// Canonical full-res u8 `[Y>>8, Y>>8, Y>>8, A>>8]` — the `ya16_to_rgba_row`
/// mapping, the u8 filter input. Consumed only by the `rgb`-gated colour
/// equivalence module, so it is dead in a `gray`-without-`rgb` build.
#[cfg_attr(not(feature = "rgb"), allow(dead_code))]
fn canonical_rgba_u8(packed: &[u16], n: usize) -> Vec<u8> {
  let mut out = std::vec![0u8; n * 4];
  for (px, src) in out.chunks_exact_mut(4).zip(packed.chunks_exact(2)) {
    let y = (src[0] >> 8) as u8;
    px[0] = y;
    px[1] = y;
    px[2] = y;
    px[3] = (src[1] >> 8) as u8;
  }
  out
}

/// De-interleaved host-native Y of a packed `[Y, A]` plane — the single-channel
/// luma oracle's input.
fn native_y(packed: &[u16], n: usize) -> Vec<u16> {
  packed.chunks_exact(2).take(n).map(|p| p[0]).collect()
}

/// Single-channel filter resample of a host-native u16 Y plane via the merged
/// engine's [`FilterStream<u16>`] (channels = 1) — the same stream the area
/// path's native-Y bin and `Gray16` use.
fn native_y_filter_u16<K: FilterKernel>(
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

// ---- Native-Y luma equivalence (the consistency contract; `gray`-solo) ------

fn assert_luma_is_native_y<K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  full_range: bool,
  ctx: &str,
) -> u16 {
  let packed = ya16_ramp(sw, sh);
  let got = ya16_filter_outputs(&packed, sw, sh, ow, oh, full_range, kernel);
  let raw = native_y_filter_u16(kernel, &native_y(&packed, sw * sh), sw, sh, ow, oh);

  let mut max_diff = 0u16;
  for (i, (&g, &r)) in got.luma_u16.iter().zip(raw.iter()).enumerate() {
    max_diff = max_diff.max(g.abs_diff(r));
    assert_eq!(
      g, r,
      "{ctx} luma_u16[{i}]: Ya16 {g} vs native-Y FilterStream<u16> {r} — luma must be native Y"
    );
  }
  // luma (u8) is the resampled native Y narrowed `>> 8`.
  for (i, (&lo, &hi)) in got.luma.iter().zip(raw.iter()).enumerate() {
    assert_eq!(
      lo,
      (hi >> 8) as u8,
      "{ctx} luma[{i}]: must be native-Y luma_u16 >> 8"
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
  // colour-derived luma would differ. A near-min limited-range Y plateau stays
  // native (a range scale would be visible).
  let mut packed = std::vec![0u16; 8 * 8 * 2];
  for px in packed.chunks_exact_mut(2) {
    px[0] = 4096; // limited-range black-ish
    px[1] = 50000; // some alpha
  }
  let fr = ya16_filter_outputs(&packed, 8, 8, 4, 4, FR, CatmullRom);
  let lim = ya16_filter_outputs(&packed, 8, 8, 4, 4, FR_LIMITED, CatmullRom);
  assert_eq!(
    fr.luma_u16, lim.luma_u16,
    "native-Y luma_u16 must be range-independent"
  );
  assert_eq!(fr.luma, lim.luma, "native-Y luma must be range-independent");
  assert!(
    lim.luma_u16.iter().all(|&y| y == 4096),
    "uniform limited-range Y plateau luma_u16 must stay native, got {:?}",
    lim.luma_u16
  );
}

// ---- LE == BE wire-encoding equivalence (`gray`-solo) -----------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn le_be_filter_outputs_match() {
  // BE wire input is byte-swapped to the SAME host-native Y/A as LE, so every
  // filter output (colour + native-Y luma) must be byte-identical.
  let packed = ya16_ramp(8, 8);
  let le = ya16_filter_outputs(&packed, 8, 8, 4, 4, FR, CatmullRom);

  let be_plane = as_be_u16(&packed);
  let src = Ya16BeFrame::new(&be_plane, 8, 8, 16);
  let mut rgba_u16 = std::vec![0u16; 4 * 4 * 4];
  let mut luma_u16 = std::vec![0u16; 4 * 4];
  {
    let mut sink = MixedSinker::<Ya16<true>, FilteredResampler<CatmullRom>>::with_resampler(
      8,
      8,
      FilteredResampler::new(4, 4, CatmullRom),
    )
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    ya16_to_endian::<_, true>(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(le.rgba_u16, rgba_u16, "LE vs BE rgba_u16");
  assert_eq!(le.luma_u16, luma_u16, "LE vs BE luma_u16");
}

// ---- Cross-check against `Gray16` (the consistency contract; `gray`-solo) ---

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_filter_equals_gray16() {
  // The SAME Y plane must filter to the SAME luma whether it carries an alpha
  // channel (`Ya16`) or not (`Gray16`).
  use crate::{frame::Gray16Frame, source::gray16_to};
  for (sw, sh, ow, oh) in [(8, 8, 4, 4), (4, 4, 7, 7)] {
    let packed = ya16_ramp(sw, sh);
    let y_host = native_y(&packed, sw * sh);
    let y_le = as_le_u16(&y_host);
    for kernel_tag in ["triangle", "catmullrom", "lanczos3"] {
      let (ya_luma, ya_lu16) = {
        let got = match kernel_tag {
          "triangle" => ya16_filter_outputs(&packed, sw, sh, ow, oh, FR, Triangle),
          "catmullrom" => ya16_filter_outputs(&packed, sw, sh, ow, oh, FR, CatmullRom),
          _ => ya16_filter_outputs(&packed, sw, sh, ow, oh, FR, Lanczos3),
        };
        (got.luma, got.luma_u16)
      };
      let mut g_luma = std::vec![0u8; ow * oh];
      let mut g_lu16 = std::vec![0u16; ow * oh];
      {
        let gsrc = Gray16Frame::new(&y_le, sw as u32, sh as u32, sw as u32);
        macro_rules! run {
          ($k:expr) => {{
            let mut sink =
              MixedSinker::<crate::source::Gray16, FilteredResampler<_>>::with_resampler(
                sw,
                sh,
                FilteredResampler::new(ow, oh, $k),
              )
              .unwrap()
              .with_luma(&mut g_luma)
              .unwrap()
              .with_luma_u16(&mut g_lu16)
              .unwrap();
            gray16_to(&gsrc, FR, sink.set_kernel_matrix(M)).unwrap();
          }};
        }
        match kernel_tag {
          "triangle" => run!(Triangle),
          "catmullrom" => run!(CatmullRom),
          _ => run!(Lanczos3),
        }
      }
      assert_eq!(
        ya_lu16, g_lu16,
        "{kernel_tag} {sw}x{sh}->{ow}x{oh}: Ya16 luma_u16 must equal Gray16 luma_u16 (same Y)"
      );
      assert_eq!(
        ya_luma, g_luma,
        "{kernel_tag} {sw}x{sh}->{ow}x{oh}: Ya16 luma must equal Gray16 luma (same Y)"
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
  let packed = ya16_ramp(8, 8);
  let plane = as_le_u16(&packed);
  let src = Ya16Frame::new(&plane, 8, 8, 16);
  let mut rgba_u16 = std::vec![0u16; 4 * 4 * 4];
  let mut sink = MixedSinker::<Ya16, FilteredResampler<Triangle>>::with_resampler(
    8,
    8,
    FilteredResampler::new(4, 4, Triangle),
  )
  .unwrap()
  .with_alpha_mode(AlphaMode::Premultiplied)
  .with_rgba_u16(&mut rgba_u16)
  .unwrap();
  let err = ya16_to(&src, FR, sink.set_kernel_matrix(M)).unwrap_err();
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
  let packed = ya16_ramp(8, 8);
  let got = ya16_filter_outputs(&packed, 8, 8, 4, 4, FR, Triangle);
  assert!(
    got.rgba_u16.iter().any(|&v| v != 0),
    "filter resample must populate rgba_u16 (no UnsupportedFilter)"
  );
  assert!(
    got.luma_u16.iter().any(|&v| v != 0),
    "filter resample must populate luma_u16 (no UnsupportedFilter)"
  );
}

// ---- Packed-RGBA equivalence oracles (gated on `rgb`) -----------------------

#[cfg(feature = "rgb")]
mod packed_rgba_equivalence {
  use super::*;
  use crate::source::{Rgba, Rgba64, rgba_to, rgba64_to};
  use mediaframe::frame::{Rgba64Frame, RgbaFrame};

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
      rgba_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    out
  }

  fn rgba64_filter<K: FilterKernel>(
    rgba: &[u16],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
  ) -> Vec<u16> {
    // `Rgba64Frame` expects LE-wire u16; re-encode the host-native canonical.
    let wire = as_le_u16(rgba);
    let src = Rgba64Frame::try_new(&wire, sw as u32, sh as u32, (sw * 4) as u32).unwrap();
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

  fn assert_color_equals_packed<K: FilterKernel + Copy>(
    kernel: K,
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    ctx: &str,
  ) -> u16 {
    let packed = ya16_ramp(sw, sh);
    let got = ya16_filter_outputs(&packed, sw, sh, ow, oh, FR, kernel);

    // u16: rgba_u16 == the Rgba64 filter of the native canonical u16 RGBA.
    let canon_u16 = canonical_rgba_u16(&packed, sw * sh);
    let want_u16 = rgba64_filter(&canon_u16, sw, sh, ow, oh, kernel);
    let mut max_diff = 0u16;
    for (i, (&g, &w)) in got.rgba_u16.iter().zip(want_u16.iter()).enumerate() {
      max_diff = max_diff.max(g.abs_diff(w));
      assert_eq!(g, w, "{ctx} rgba_u16[{i}]: {g} vs Rgba64 filter {w}");
    }
    for (rgb_px, rgba_px) in got.rgb_u16.chunks_exact(3).zip(want_u16.chunks_exact(4)) {
      assert_eq!(
        rgb_px,
        &rgba_px[..3],
        "{ctx} rgb_u16 == drop-alpha(filtered rgba_u16)"
      );
    }

    // u8: rgba == the 8-bit Rgba filter of the `>> 8` narrowed canonical RGBA.
    let canon_u8 = canonical_rgba_u8(&packed, sw * sh);
    let want_u8 = rgba_filter(&canon_u8, sw, sh, ow, oh, kernel);
    for (i, (&g, &w)) in got.rgba.iter().zip(want_u8.iter()).enumerate() {
      assert_eq!(g, w, "{ctx} rgba[{i}]: {g} vs Rgba (>>8) filter {w}");
    }
    for (rgb_px, rgba_px) in got.rgb.chunks_exact(3).zip(want_u8.chunks_exact(4)) {
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
  fn color_filter_equals_packed_rgba() {
    for (sw, sh, ow, oh, tag) in [(8, 8, 4, 4, "down"), (4, 4, 7, 7, "up")] {
      assert_color_equals_packed(Triangle, sw, sh, ow, oh, &format!("triangle {tag}"));
      assert_color_equals_packed(CatmullRom, sw, sh, ow, oh, &format!("catmullrom {tag}"));
      assert_color_equals_packed(Lanczos3, sw, sh, ow, oh, &format!("lanczos3 {tag}"));
    }
  }
}
