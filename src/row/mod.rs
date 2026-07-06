//! Crate-internal row-level primitives.
//!
//! These are the composable units that Sinks call on each row handed
//! to them by a source kernel. Source kernels are pure row walkers;
//! the actual arithmetic lives here.
//!
//! Backends (all crate‑private modules):
//! - `scalar` — always compiled, reference implementation.
//! - `arch::neon` — aarch64 NEON.
//! - `arch::x86_sse41`, `arch::x86_avx2`, `arch::x86_avx512` — x86_64
//!   tiers.
//! - `arch::wasm_simd128` — wasm32 simd128.
//!
//! Each is gated on the appropriate `target_arch` / `target_feature`
//! cfg.
//!
//! Dispatch model: every backend is selected at call time by runtime
//! CPU feature detection — `is_aarch64_feature_detected!` /
//! `is_x86_feature_detected!` under `feature = "std"`, or compile‑time
//! `cfg!(target_feature = ...)` in no‑std builds. `std`'s runtime
//! detection caches the result in an atomic, so per‑call overhead is a
//! single relaxed load plus a branch. Each SIMD kernel itself carries
//! `#[target_feature(enable = "...")]` so its intrinsics execute in an
//! explicitly feature‑enabled context, not one inherited from the
//! target's default features.
//!
//! Output guarantees: every backend is either byte‑identical to
//! `scalar` or differs by at most 1 LSB per channel (documented per
//! backend). Tests in `arch` enforce this contract.
//!
//! Dispatcher `cfg_select!` requires Rust 1.95+ (stable, in the core
//! prelude — no import needed). The crate's MSRV matches.
//!
//! # Submodule layout
//!
//! Public dispatchers are split across `dispatch::*` submodules by
//! source format family for readability — `yuv420` / `yuv444` / `nv` /
//! `pn` / `yuva` / `rgb_ops` / `bayer`. They are re-exported as
//! `pub use dispatch::*::*` here so the public API stays at
//! `crate::row::*` (e.g. `crate::row::yuv_420_to_rgb_row`). Callers
//! see no API change from the split.

pub(crate) mod arch;
pub(crate) mod dispatch;
pub(crate) mod scalar;

// Re-exported only when a caller is compiled. The `MixedSinker` Strategy A
// fan-out is the sole consumer, and it lives in `crate::sinker::mixed` which
// is gated on `feature = "std"` / `feature = "alloc"` (needs `Vec`). Without
// either feature both this re-export and the underlying scalar function would
// be unused, which is a hard error under `cargo clippy -- -D warnings`.
//
// Consumer source families — every YUV family except `bayer` / `mono` /
// `rgb-float` / `rgb-legacy` / `xyz` expands RGB → RGBA via the Strategy A
// helpers.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "gbr",
    feature = "gray",
    feature = "rgb",
    feature = "v210",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ),
))]
pub(crate) use scalar::expand_rgb_to_rgba_row;
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "gbr",
    feature = "gray",
    feature = "rgb",
    feature = "v210",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ),
))]
pub(crate) use scalar::expand_rgb_u16_to_rgba_u16_row;
// Native-precision `u16` luma from packed host-order `u16` RGB, used by
// the shared high-bit packed-RGB resample tail
// (`packed_rgb_u16_resample_emit`) when its `NATIVE_LUMA16` parameter is
// set — i.e. the fused high-bit planar GBR path, whose `luma_u16` stays
// byte-identical to the direct `gbr_to_luma_u16_high_bit_row`. The
// `Rgb48` / `Bgr48` callers — and the high-bit packed YUV color groups
// (4:4:4 and 4:2:2) and the high-bit semi-planar P-format groups — pass
// `NATIVE_LUMA16 = false` and never reach it at runtime, but the generic
// tail still references the symbol, so the re-export is gated on the same
// consumer set as that tail.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  ),
))]
pub(crate) use scalar::rgb_to_luma_u16_native_row;

// Strategy A+ α-extract dispatcher — re-exported at `crate::row::alpha_extract`
// so source-α sinkers don't have to reach into `dispatch::` internals. Same
// `feature = "std" | "alloc"` gating as the expand helpers above.
//
// Consumer source families with a source-α channel: `gbr` (Gbrap), `yuv-444-packed`
// (AYUV64), and `yuva` (yuva planar) only.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(feature = "gbr", feature = "yuv-444-packed", feature = "yuva"),
))]
pub(crate) use dispatch::alpha_extract;
// Fused-downscale H-pass reduction; consumed by `crate::resample`'s
// `AreaStream`.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  )
))]
pub(crate) use dispatch::area_reduce::{
  PaddedSpans, area_h_reduce_row, area_h_reduce_row_f32, area_h_reduce_row_u16, area_v_accumulate,
  area_v_accumulate_f32, area_v_accumulate_u16,
};
// The `u32` area H/V passes have no SIMD tier (their `u128` accumulators
// exceed every lane width the integer kernels handle), so they re-export
// straight from the scalar reference rather than through a `dispatch`
// wrapper. Consumed by `AreaSample<u32>` for the `u32` source formats.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  )
))]
pub(crate) use scalar::area_reduce::{area_h_reduce_row_u32, area_v_accumulate_u32};
// Separable filter H/V passes; consumed by `crate::resample::filter`'s
// `FilterStream` (the signed twin of `AreaStream`).
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  )
))]
pub(crate) use dispatch::filter_reduce::{
  FilterPaddedSpans, FilterSimdElem, filter_h_reduce_row, filter_v_accumulate,
};
// `y_plane_to_luma_u16_row` is consumed by every source family that exposes
// a luma plane to the MixedSinker.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "gray",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ),
))]
pub(crate) use dispatch::y_plane_to_luma_u16::y_plane_to_luma_u16_row;

// Task 3 — packed YUV 4:2:2 luma_u16 dispatchers (pub(crate) because they are
// consumed only by the MixedSinker impls, not the public API).
#[cfg(all(feature = "yuv-packed", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::packed_yuv422::{
  uyvy422_to_luma_u16_row, yuyv422_to_luma_u16_row, yvyu422_to_luma_u16_row,
};

// Tier 5.25 — packed YUV 4:1:1 luma_u16 dispatcher (pub(crate); MixedSinker only).
#[cfg(all(feature = "yuv-packed", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::packed_yuv411::uyyvyy411_to_luma_u16_row;

// Task 4 — packed YUV 4:4:4 (VUYA / VUYX) luma_u16 dispatchers.
#[cfg(all(feature = "yuv-444-packed", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::vuya::vuya_to_luma_u16_row;
#[cfg(all(feature = "yuv-444-packed", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::vuyx::vuyx_to_luma_u16_row;
// Packed YUV 4:4:4 channel re-orderings (AYUV / UYVA / VYU444) luma_u16
// dispatchers — `pub(crate)`, MixedSinker only.
#[cfg(all(feature = "yuv-444-packed", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::{
  ayuv::ayuv_to_luma_u16_row, uyva::uyva_to_luma_u16_row, vyu444::vyu444_to_luma_u16_row,
};

#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
pub use dispatch::ayuv64::*;
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
pub use dispatch::bayer::*;
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
pub use dispatch::legacy_rgb::*;
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
pub use dispatch::nv::*;
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
pub use dispatch::nv20::*;
#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
pub use dispatch::pal8::*;
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
pub use dispatch::pn::*;
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
pub use dispatch::{packed_yuv411::*, packed_yuv422::*};
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
pub use dispatch::{planar_gbr::*, planar_gbr_32bit::*, planar_gbr_high_bit::*, planar_gbr_msb::*};
#[cfg(feature = "rgb-float")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-float")))]
pub use dispatch::{rgb_f16_ops::*, rgb_float_ops::*};
// rgb_ops contains the cross-format `rgb_to_hsv_row` / `rgb_to_luma_row`
// / `rgb_to_luma_u16_row` helpers used by every sinker, as well as the
// `rgb`-family packed RGB/RGBA dispatchers. Always re-exported so HSV /
// luma derivations stay reachable when the `rgb` family is disabled.
#[cfg(all(feature = "mono", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::mono1bit::*;
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
pub use dispatch::packed_rgb_16bit::{
  bgr48_to_rgb_row, bgr48_to_rgb_row_endian, bgr48_to_rgb_u16_row, bgr48_to_rgb_u16_row_endian,
  bgr48_to_rgba_row, bgr48_to_rgba_row_endian, bgr48_to_rgba_u16_row, bgr48_to_rgba_u16_row_endian,
  bgra64_to_rgb_row, bgra64_to_rgb_row_endian, bgra64_to_rgb_u16_row, bgra64_to_rgb_u16_row_endian,
  bgra64_to_rgba_row, bgra64_to_rgba_row_endian, bgra64_to_rgba_u16_row,
  bgra64_to_rgba_u16_row_endian, rgb48_to_rgb_row, rgb48_to_rgb_row_endian, rgb48_to_rgb_u16_row,
  rgb48_to_rgb_u16_row_endian, rgb48_to_rgba_row, rgb48_to_rgba_row_endian, rgb48_to_rgba_u16_row,
  rgb48_to_rgba_u16_row_endian, rgba64_to_rgb_row, rgba64_to_rgb_row_endian, rgba64_to_rgb_u16_row,
  rgba64_to_rgb_u16_row_endian, rgba64_to_rgba_row, rgba64_to_rgba_row_endian,
  rgba64_to_rgba_u16_row, rgba64_to_rgba_u16_row_endian,
};
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
pub use dispatch::packed_rgb_32bit::{
  rgb96_to_rgb_row, rgb96_to_rgb_row_endian, rgb96_to_rgb_u16_row, rgb96_to_rgb_u16_row_endian,
  rgb96_to_rgba_row, rgb96_to_rgba_row_endian, rgb96_to_rgba_u16_row, rgb96_to_rgba_u16_row_endian,
  rgba128_to_rgb_row, rgba128_to_rgb_row_endian, rgba128_to_rgb_u16_row,
  rgba128_to_rgb_u16_row_endian, rgba128_to_rgba_row, rgba128_to_rgba_row_endian,
  rgba128_to_rgba_u16_row, rgba128_to_rgba_u16_row_endian,
};
pub use dispatch::rgb_ops::*;
#[cfg(feature = "v210")]
#[cfg_attr(docsrs, doc(cfg(feature = "v210")))]
pub use dispatch::v210::*;
#[cfg(all(feature = "xyz", any(feature = "std", feature = "alloc")))]
#[cfg_attr(
  docsrs,
  doc(cfg(all(feature = "xyz", any(feature = "std", feature = "alloc"))))
)]
pub use dispatch::xyz12::*;
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
pub use dispatch::yuv411p::*;
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
pub use dispatch::yuva::*;
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
pub use dispatch::{
  ayuv::*, uyva::*, v30x::*, v410::*, vuya::*, vuyx::*, vyu444::*, xv36::*, xv48::*,
};
#[cfg(feature = "y2xx")]
#[cfg_attr(docsrs, doc(cfg(feature = "y2xx")))]
pub use dispatch::{y210::*, y212::*, y216::*};
// `yuv420` carries both the planar 4:2:0 dispatchers and the
// semi-planar P010/P012/P016 dispatchers, so it re-exports under the
// union; the glob only carries whichever family's items are compiled.
// `yuv444` is planar-only.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
pub use dispatch::yuv420::*;
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
pub use dispatch::yuv444::*;
// luma + HSV variants take an extra rgb_scratch parameter
#[cfg(feature = "rgb")]
#[allow(unused_imports)]
pub(crate) use dispatch::packed_rgb_16bit::{
  bgr48_to_hsv_row, bgr48_to_hsv_row_endian, bgr48_to_luma_row, bgr48_to_luma_row_endian,
  bgr48_to_luma_u16_row, bgr48_to_luma_u16_row_endian, bgra64_to_hsv_row, bgra64_to_hsv_row_endian,
  bgra64_to_luma_row, bgra64_to_luma_row_endian, bgra64_to_luma_u16_row,
  bgra64_to_luma_u16_row_endian, rgb48_to_hsv_row, rgb48_to_hsv_row_endian, rgb48_to_luma_row,
  rgb48_to_luma_row_endian, rgb48_to_luma_u16_row, rgb48_to_luma_u16_row_endian, rgba64_to_hsv_row,
  rgba64_to_hsv_row_endian, rgba64_to_luma_row, rgba64_to_luma_row_endian, rgba64_to_luma_u16_row,
  rgba64_to_luma_u16_row_endian,
};
#[cfg(feature = "rgb")]
#[allow(unused_imports)]
pub(crate) use dispatch::packed_rgb_32bit::{
  rgb96_to_hsv_row, rgb96_to_hsv_row_endian, rgb96_to_luma_row, rgb96_to_luma_row_endian,
  rgb96_to_luma_u16_row, rgb96_to_luma_u16_row_endian, rgba128_to_hsv_row,
  rgba128_to_hsv_row_endian, rgba128_to_luma_row, rgba128_to_luma_row_endian,
  rgba128_to_luma_u16_row, rgba128_to_luma_u16_row_endian,
};
// Gray dispatchers are pub(crate) — sinker code uses them via crate::row::gray*_row.
#[cfg(all(feature = "gray", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::gray::*;
// Grayf16 / Grayf32 / Ya8 / Ya16 dispatchers — pub(crate) for sinker use.
#[cfg(all(feature = "gray", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::grayf16::*;
#[cfg(all(feature = "gray", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::grayf32::*;
#[cfg(all(feature = "gray", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::ya8::*;
#[cfg(all(feature = "gray", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::ya16::*;
#[cfg(all(feature = "gray", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::yaf16::*;
#[cfg(all(feature = "gray", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::yaf32::*;
// Planar GBR float dispatchers — pub(crate) for sinker use (MixedSinker<Gbrpf32> etc.).
#[cfg(all(feature = "gbr", any(feature = "std", feature = "alloc")))]
pub(crate) use dispatch::planar_gbr_float::*;

// `yuv_444p_n_to_rgb_u16_row` is consumed by the 32-bit overflow test
// `yuv_444p_n_u16_dispatcher_rejects_width_times_3_overflow` below —
// the dispatch submodule keeps it as `pub(crate)`, so glob `pub use`
// doesn't pick it up. Gated on the same cfg the test uses to avoid
// `unused_imports` on builds that don't compile the test. (The planar
// 4:2:0 high-bit native fast tier converts its output-width binned rows
// through the per-format `yuv444pN_to_rgb_*_row_endian` wrappers — which
// the glob `pub use dispatch::yuv444::*` above already re-exports — so it
// does not need this generic kernel widened.)
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  target_pointer_width = "32"
))]
pub(crate) use dispatch::yuv444::yuv_444p_n_to_rgb_u16_row;

// ---- shared dispatcher helpers ---------------------------------------

/// Computes the byte length of one packed‑RGB row with overflow
/// checking. Panics if `width x 3` cannot be represented as `usize`
/// (only reachable on 32‑bit targets — wasm32, i686 — with extreme
/// widths). Callers use the returned length as the minimum buffer
/// size they hand to unsafe SIMD kernels, so an unchecked
/// multiplication here could admit an undersized buffer and trigger
/// out‑of‑bounds writes downstream.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb_row_bytes(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n,
    None => panic!("width ({width}) x 3 overflows usize"),
  }
}

/// Byte length of one packed‑RGBA row (`width x 4`) with overflow
/// checking. Same purpose as [`rgb_row_bytes`] for the 4-channel
/// path used by the RGBA dispatchers.
///
/// Used by every non-Bayer dispatcher family that emits packed RGBA
/// output (Bayer is RGB-only). The 14-way `any(feature)` cfg
/// enumerates every consumer family explicitly so dead-code analysis
/// stays strict under non-`frame` feature subsets.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgba_row_bytes(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) x 4 overflows usize"),
  }
}

/// Element count of one packed `u16`‑RGB row (`width x 3`). Identical
/// math to [`rgb_row_bytes`] — the returned value is in `u16`
/// elements, not bytes. Callers use it to size `&mut [u16]` buffers
/// for the `u16` output path. `width x 3` overflow still matters on
/// 32‑bit targets: the product names the number of elements the
/// caller allocates, and downstream SIMD kernels index with it
/// directly without re‑multiplying.
///
/// Used by every dispatcher family that emits a packed u16-RGB row.
/// Packed YUV 4:2:2 / 4:1:1 (`yuv-packed`) emits u8 only and does
/// not consume this helper.
#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgb_row_elems(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n,
    None => panic!("width ({width}) x 3 overflows usize"),
  }
}

/// Element count of one packed `u16`-RGBA row (`width x 4`). Identical
/// math to [`rgba_row_bytes`] — the returned value is in `u16`
/// elements, not bytes. Callers use it to size `&mut [u16]` buffers
/// for the high-bit-depth `u16` RGBA output path.
///
/// Bayer is RGB-only and packed YUV 4:2:2 / 4:1:1 emit u8 only, so
/// neither consume this helper.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgba_row_elems(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) x 4 overflows usize"),
  }
}

/// Element count of one packed YA (luma + alpha) row (`width x 2`)
/// with overflow checking. Same purpose as [`rgb_row_bytes`] for the
/// 2-element `[Y, A, ...]` interleaved layout used by Ya8 (`&[u8]`),
/// Ya16 (`&[u16]`), Yaf16 (`&[half::f16]`) and Yaf32 (`&[f32]`) packed
/// inputs — all index `width x 2` elements regardless of element width,
/// so this helper covers them all.
#[cfg(feature = "gray")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya_row_elems(width: usize) -> usize {
  match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) x 2 overflows usize"),
  }
}

/// Maximum permitted magnitude of any element of a fused color
/// transform handed to a Bayer row dispatcher.
///
/// Set to `WhiteBalance::MAX_GAIN x ColorCorrectionMatrix::MAX_COEFFICIENT_ABS
/// = 1e6 x 1e6 = 1e12`, which is the largest absolute value any
/// fused entry can take when the upstream WB / CCM were
/// validated through [`crate::raw::WhiteBalance::try_new`] /
/// [`crate::raw::ColorCorrectionMatrix::try_new`]. The overflow
/// analysis (see those constructor docs) shows that with `|m[i][j]|
/// ≤ 1e12` and 16-bit samples, the largest per-channel sum stays
/// `~21` orders of magnitude under `f32::MAX`. So bounding here
/// at 1e12 closes the door on direct-row-API callers passing
/// extreme finite matrices that would silently overflow during
/// the matmul.
#[cfg(feature = "bayer")]
pub(crate) const MAX_FUSED_TRANSFORM_ABS: f32 = 1.0e12;

/// Asserts every element of a 3x3 fused color transform is
/// **finite and within the magnitude bound**
/// ([`MAX_FUSED_TRANSFORM_ABS`]).
///
/// Used by the Bayer row dispatchers in release builds before
/// invoking the kernel — once SIMD backends land they will rely on
/// this guarantee for branchless f32 arithmetic. A single Inf or
/// NaN would otherwise propagate through every pixel of the row
/// (Inf clamps to saturated white, NaN casts to 0, both producing
/// silently-wrong frames); finite-but-extreme entries (e.g. mixed
/// `±f32::MAX` from a direct row-API caller) likewise produce
/// `Inf + -Inf == NaN` during the matmul.
///
/// Validating WB / CCM upstream via
/// [`crate::raw::WhiteBalance::try_new`] /
/// [`crate::raw::ColorCorrectionMatrix::try_new`] catches the
/// common case; this is the kernel-boundary backstop for direct
/// row-API callers and the dispatcher-level guarantee that
/// matches what validated upstream inputs can produce.
#[cfg(feature = "bayer")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn assert_color_transform_well_formed(m: &[[f32; 3]; 3]) {
  let mut row = 0;
  while row < 3 {
    let mut col = 0;
    while col < 3 {
      let v = m[row][col];
      assert!(
        v.is_finite(),
        "color transform m[{row}][{col}] is non-finite (NaN or ±∞)"
      );
      assert!(
        v.abs() <= MAX_FUSED_TRANSFORM_ABS,
        "color transform m[{row}][{col}] = {v} exceeds magnitude bound \
         (|coeff| ≤ {MAX_FUSED_TRANSFORM_ABS}); validated WB x CCM cannot \
         produce values past this bound"
      );
      col += 1;
    }
    row += 1;
  }
}

/// Element count of one full-width interleaved-UV row (`width x 2`)
/// for semi-planar 4:4:4 sources (`P410` / `P412` / `P416`). One
/// `(U, V)` pair per pixel = `2 * width` `u16` elements per row.
/// Same `checked_mul` rationale as [`rgb_row_bytes`] — the returned
/// length feeds into unsafe SIMD kernels' bounds via the dispatcher's
/// `assert!`, so an unchecked multiplication on 32-bit targets could
/// silently admit an undersized buffer.
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uv_full_row_elems(width: usize) -> usize {
  match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) x 2 overflows usize (UV row)"),
  }
}

/// Byte length of one packed YUV 4:2:2 row (`width x 2`) for
/// `Yuyv422` / `Uyvy422` / `Yvyu422` sources. Two bytes per pixel
/// (one `Y` + one half of an interleaved `U`/`V` pair). Same
/// `checked_mul` rationale as [`rgb_row_bytes`] — the returned byte
/// count feeds into the packed dispatchers' input-side `assert!`,
/// which gates entry into unsafe SIMD loads. An unchecked
/// multiplication on 32-bit targets could silently admit an
/// undersized `packed` slice.
#[cfg(any(feature = "rgb-legacy", feature = "yuv-packed"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn packed_yuv422_row_bytes(width: usize) -> usize {
  match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) x 2 overflows usize (packed YUV 4:2:2 row)"),
  }
}

/// Byte length of one packed YUV 4:1:1 row (`width x 3 / 2`) for
/// the `Uyyvyy411` source. 6 bytes per 4-pixel block (12 bpp). Width
/// must be a multiple of 4 — callers assert this separately. Same
/// `checked_mul` rationale as [`rgb_row_bytes`]: the returned byte
/// count feeds into the dispatcher's input-side `assert!`, which
/// gates entry into unsafe SIMD loads. Computed as
/// `(width x 3) / 2` so the intermediate `width x 3` is the only
/// product that can overflow on 32-bit targets at extreme widths.
#[cfg(feature = "yuv-packed")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn packed_yuv411_row_bytes(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n / 2,
    None => panic!("width ({width}) x 3 / 2 overflows usize (packed YUV 4:1:1 row)"),
  }
}

/// Byte length of one packed `v210` row (`ceil(width / 6) * 16`) with
/// overflow checking. v210 packs 6 pixels per 16-byte word; widths
/// that don't end on a complete-word boundary (e.g. 1280 for 720p)
/// round up to the next word, with the final word emitting only its
/// 2 or 4 valid pixels.
///
/// Same `checked_mul` rationale as [`rgb_row_bytes`] — the returned
/// byte count gates entry into unsafe SIMD loads. Panics if the
/// multiplication overflows.
#[cfg(feature = "v210")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v210_row_bytes(width: usize) -> usize {
  let words = width.div_ceil(6);
  match words.checked_mul(16) {
    Some(n) => n,
    None => panic!("width ({width}) / 6 x 16 overflows usize (v210 row)"),
  }
}

/// Element count of one packed `Y2xx` row (`width x 2` u16
/// elements) with overflow checking. Used by the Y210 / Y212 / Y216
/// dispatchers to gate entry into unsafe SIMD loads. Same
/// `checked_mul` rationale as [`rgb_row_bytes`].
#[cfg(feature = "y2xx")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y2xx_row_elems(width: usize) -> usize {
  match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) x 2 overflows usize (Y2xx row)"),
  }
}

// ---- runtime CPU feature detection -----------------------------------
//
// Each `*_available` helper returns `true` iff the named feature is
// present. `feature = "std"` branches use std's cached
// `is_*_feature_detected!` macros (atomic load + branch after the
// first call). No‑std branches fall back to `cfg!(target_feature = ...)`
// which is resolved at compile time. Helpers are only compiled for
// targets where the corresponding feature exists.

// The `colconv_force_scalar` cfg, when set, short‑circuits every
// `*_available()` helper to `false` so the dispatcher always falls
// through to the scalar reference path. CI uses this via
// `RUSTFLAGS='--cfg colconv_force_scalar'` to benchmark / measure
// coverage of the scalar baseline. `colconv_disable_avx512` /
// `colconv_disable_avx2` similarly force lower‑tier x86 paths for
// per‑tier coverage on runners that would otherwise always pick
// AVX‑512.

/// NEON availability on aarch64.
#[cfg(all(target_arch = "aarch64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn neon_available() -> bool {
  if cfg!(colconv_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

/// NEON availability on aarch64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn neon_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "neon")
}

/// FP16 conversion-instruction availability on aarch64. Required for
/// `vcvt_f32_f16` / FCVTL — a separate CPU feature from NEON. Older
/// AArch64 cores (e.g. Cortex-A53/A57 base, some embedded SoCs) ship
/// NEON without `fp16`; calling `vcvt_f32_f16` there raises SIGILL.
/// The Rgbf16 NEON dispatchers gate on `neon_available() &&
/// fp16_available()` and fall back to scalar when this returns false.
///
/// Consumers: `rgb-float` (`dispatch::rgb_f16_ops`), `gbr`
/// (`dispatch::planar_gbr_float`), and `gray` (`dispatch::grayf16`).
/// The cfg matches those families exactly.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(feature = "gbr", feature = "rgb-float", feature = "gray"),
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn fp16_available() -> bool {
  if cfg!(colconv_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("fp16")
}

/// FP16 availability on aarch64 — no‑std variant (compile‑time).
#[cfg(all(
  target_arch = "aarch64",
  not(feature = "std"),
  any(feature = "gbr", feature = "rgb-float", feature = "gray"),
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn fp16_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "fp16")
}

/// AVX2 availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn avx2_available() -> bool {
  if cfg!(colconv_force_scalar) || cfg!(colconv_disable_avx2) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx2")
}

/// AVX2 availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn avx2_available() -> bool {
  !cfg!(colconv_force_scalar) && !cfg!(colconv_disable_avx2) && cfg!(target_feature = "avx2")
}

/// SSE4.1 availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn sse41_available() -> bool {
  if cfg!(colconv_force_scalar) {
    return false;
  }
  std::arch::is_x86_feature_detected!("sse4.1")
}

/// SSE4.1 availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn sse41_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "sse4.1")
}

/// AVX‑512 (F + BW) availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn avx512_available() -> bool {
  if cfg!(colconv_force_scalar) || cfg!(colconv_disable_avx512) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx512bw")
}

/// AVX‑512 (F + BW) availability on x86_64 — no‑std variant
/// (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn avx512_available() -> bool {
  !cfg!(colconv_force_scalar) && !cfg!(colconv_disable_avx512) && cfg!(target_feature = "avx512bw")
}

/// F16C availability on x86_64. Used by the `Rgbf16` and `Grayf16` dispatchers to gate the
/// hardware-accelerated f16→f32 widening path. F16C is checked *in addition*
/// to the SIMD tier (AVX-512 / AVX2 / SSE4.1) because it is an independent
/// feature bit that can be absent even on AVX2 machines.
#[cfg(all(
  target_arch = "x86_64",
  feature = "std",
  any(feature = "rgb-float", feature = "gbr", feature = "gray"),
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn f16c_available() -> bool {
  if cfg!(colconv_force_scalar) {
    return false;
  }
  std::arch::is_x86_feature_detected!("f16c")
}

/// F16C availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(
  target_arch = "x86_64",
  not(feature = "std"),
  any(feature = "rgb-float", feature = "gbr", feature = "gray"),
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn f16c_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "f16c")
}

/// simd128 availability on wasm32. WASM has no runtime CPU detection
/// (SIMD support is fixed at module produce time), so this is always
/// a compile‑time check regardless of the `std` feature.
#[cfg(target_arch = "wasm32")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn simd128_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "simd128")
}
#[cfg(all(test, feature = "std"))]
mod overflow_tests {
  //! 32-bit RGB-row-bytes overflow regressions for the public
  //! dispatchers. `width x 3` can wrap `usize` on wasm32 / i686 for
  //! extreme widths; the shared [`rgb_row_bytes`] helper rejects
  //! these before any unsafe kernel sees them. Tests are gated on
  //! 32-bit because `u32 x 3` never wraps 64-bit `usize`.

  #[cfg(target_pointer_width = "32")]
  use super::*;
  #[cfg(all(
    target_pointer_width = "32",
    any(
      feature = "yuv-planar",
      feature = "yuv-semi-planar",
      feature = "yuv-packed",
      feature = "v210",
      feature = "y2xx"
    )
  ))]
  use crate::ColorMatrix;

  /// The smallest even width greater than `usize::MAX / 3`, so
  /// `width * 3` overflows 32-bit `usize` without tripping the
  /// dispatchers' even-width precondition first. `(usize::MAX / 3)`
  /// is always odd on 32-bit (`(2^32 - 1) / 3 == 1431655765`), so
  /// `+ 1` produces an even number — the `+ (candidate & 1)` keeps
  /// this correct on hypothetical platforms where the parity
  /// differs.
  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH: usize = {
    let candidate = (usize::MAX / 3) + 1;
    candidate + (candidate & 1)
  };

  #[cfg(all(target_pointer_width = "32", feature = "yuv-planar"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuv_420_dispatcher_rejects_width_times_3_overflow() {
    let y: [u8; 0] = [];
    let u: [u8; 0] = [];
    let v: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    yuv_420_to_rgb_row(
      &y,
      &u,
      &v,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-planar"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuv_444_dispatcher_rejects_width_times_3_overflow() {
    let y: [u8; 0] = [];
    let u: [u8; 0] = [];
    let v: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    yuv_444_to_rgb_row(
      &y,
      &u,
      &v,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-semi-planar"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn nv12_dispatcher_rejects_width_times_3_overflow() {
    let y: [u8; 0] = [];
    let uv: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    nv12_to_rgb_row(
      &y,
      &uv,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn rgb_to_hsv_dispatcher_rejects_width_times_3_overflow() {
    let rgb: [u8; 0] = [];
    let mut h: [u8; 0] = [];
    let mut s: [u8; 0] = [];
    let mut v: [u8; 0] = [];
    rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, OVERFLOW_WIDTH, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "rgb"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn bgr_to_rgb_dispatcher_rejects_width_times_3_overflow() {
    let input: [u8; 0] = [];
    let mut output: [u8; 0] = [];
    bgr_to_rgb_row(&input, &mut output, OVERFLOW_WIDTH, false);
  }

  // ---- Tier 10 planar GBR dispatchers — `width x {3, 4}` overflow ----
  //
  // The Tier 10 GBR sources interleave G/B/R (and optional A) planes
  // into packed RGB / RGBA. The dispatcher uses [`rgb_row_bytes`] /
  // [`rgba_row_bytes`] to compute the minimum output buffer length;
  // an unchecked multiplication on 32-bit could admit an undersized
  // buffer to unsafe SIMD. Each public entry point gets a regression
  // so a future drop of either guard surfaces independently.

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbr_to_rgb_dispatcher_rejects_width_times_3_overflow() {
    let g: [u8; 0] = [];
    let b: [u8; 0] = [];
    let r: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    gbr_to_rgb_row(&g, &b, &r, &mut rgb, OVERFLOW_WIDTH, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbr_to_rgba_opaque_dispatcher_rejects_width_times_4_overflow() {
    let g: [u8; 0] = [];
    let b: [u8; 0] = [];
    let r: [u8; 0] = [];
    let mut rgba: [u8; 0] = [];
    gbr_to_rgba_opaque_row(&g, &b, &r, &mut rgba, OVERFLOW_WIDTH, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbra_to_rgba_dispatcher_rejects_width_times_4_overflow() {
    let g: [u8; 0] = [];
    let b: [u8; 0] = [];
    let r: [u8; 0] = [];
    let a: [u8; 0] = [];
    let mut rgba: [u8; 0] = [];
    gbra_to_rgba_row(&g, &b, &r, &a, &mut rgba, OVERFLOW_WIDTH, false);
  }

  // ---- Tier 10b planar GBR high-bit dispatchers — `width x {3,4}` overflow
  //
  // The high-bit (`GbrpN` / `GbrapN`) dispatchers must guard their output
  // buffer sizes against `width * {3, 4}` wrapping on 32-bit targets.
  // Each {3, 4}-channel-output entry point gets an independent regression
  // test so future guard removals surface individually. The native-depth
  // luma dispatcher (`gbr_to_luma_u16_high_bit_row`) is omitted because
  // it doesn't perform a width x N multiply — output length is `width`
  // exactly, so there is no wrapping path to guard.

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbr_to_rgb_high_bit_dispatcher_rejects_width_times_3_overflow() {
    let g: [u16; 0] = [];
    let b: [u16; 0] = [];
    let r: [u16; 0] = [];
    let mut rgb: [u8; 0] = [];
    gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut rgb, OVERFLOW_WIDTH, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbr_to_rgb_u16_high_bit_dispatcher_rejects_width_times_3_overflow() {
    let g: [u16; 0] = [];
    let b: [u16; 0] = [];
    let r: [u16; 0] = [];
    let mut rgb: [u16; 0] = [];
    gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut rgb, OVERFLOW_WIDTH, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbr_to_rgba_opaque_high_bit_dispatcher_rejects_width_times_4_overflow() {
    let g: [u16; 0] = [];
    let b: [u16; 0] = [];
    let r: [u16; 0] = [];
    let mut rgba: [u8; 0] = [];
    gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut rgba, OVERFLOW_WIDTH, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbr_to_rgba_opaque_u16_high_bit_dispatcher_rejects_width_times_4_overflow() {
    let g: [u16; 0] = [];
    let b: [u16; 0] = [];
    let r: [u16; 0] = [];
    let mut rgba: [u16; 0] = [];
    gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut rgba, OVERFLOW_WIDTH, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbra_to_rgba_high_bit_dispatcher_rejects_width_times_4_overflow() {
    let g: [u16; 0] = [];
    let b: [u16; 0] = [];
    let r: [u16; 0] = [];
    let a: [u16; 0] = [];
    let mut rgba: [u8; 0] = [];
    gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut rgba, OVERFLOW_WIDTH, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gbr"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbra_to_rgba_u16_high_bit_dispatcher_rejects_width_times_4_overflow() {
    let g: [u16; 0] = [];
    let b: [u16; 0] = [];
    let r: [u16; 0] = [];
    let a: [u16; 0] = [];
    let mut rgba: [u16; 0] = [];
    gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut rgba, OVERFLOW_WIDTH, false);
  }

  // ---- Tier 11 gray dispatchers — `width x {3, 4}` overflow ----
  //
  // The gray RGB / RGBA / RGB-u16 / RGBA-u16 dispatchers route through
  // [`rgb_row_bytes`] / [`rgba_row_bytes`] / [`rgb_row_elems`] /
  // [`rgba_row_elems`] for output bounds-checking. Without these
  // helpers an `out.len() >= width * N` assert could pass on 32-bit
  // for wrapped multiplications and admit an undersized buffer to
  // unsafe SIMD. Each format family gets a regression for the *3
  // (RGB) and *4 (RGBA) paths so a future regression on any one of
  // them surfaces independently.

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray8_to_rgb_dispatcher_rejects_width_times_3_overflow() {
    let y: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    gray8_to_rgb_row(&y, &mut rgb, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray8_to_rgba_dispatcher_rejects_width_times_4_overflow() {
    let y: [u8; 0] = [];
    let mut rgba: [u8; 0] = [];
    gray8_to_rgba_row(&y, &mut rgba, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray_n_to_rgb_dispatcher_rejects_width_times_3_overflow() {
    let y: [u16; 0] = [];
    let mut rgb: [u8; 0] = [];
    gray_n_to_rgb_row::<10, false>(&y, &mut rgb, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray_n_to_rgba_dispatcher_rejects_width_times_4_overflow() {
    let y: [u16; 0] = [];
    let mut rgba: [u8; 0] = [];
    gray_n_to_rgba_row::<10, false>(&y, &mut rgba, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray_n_to_rgb_u16_dispatcher_rejects_width_times_3_overflow() {
    let y: [u16; 0] = [];
    let mut rgb: [u16; 0] = [];
    gray_n_to_rgb_u16_row::<10, false>(&y, &mut rgb, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray_n_to_rgba_u16_dispatcher_rejects_width_times_4_overflow() {
    let y: [u16; 0] = [];
    let mut rgba: [u16; 0] = [];
    gray_n_to_rgba_u16_row::<10, false>(&y, &mut rgba, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray16_to_rgb_dispatcher_rejects_width_times_3_overflow() {
    let y: [u16; 0] = [];
    let mut rgb: [u8; 0] = [];
    gray16_to_rgb_row::<false>(&y, &mut rgb, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray16_to_rgba_dispatcher_rejects_width_times_4_overflow() {
    let y: [u16; 0] = [];
    let mut rgba: [u8; 0] = [];
    gray16_to_rgba_row::<false>(&y, &mut rgba, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray16_to_rgb_u16_dispatcher_rejects_width_times_3_overflow() {
    let y: [u16; 0] = [];
    let mut rgb: [u16; 0] = [];
    gray16_to_rgb_u16_row::<false>(&y, &mut rgb, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "gray"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gray16_to_rgba_u16_dispatcher_rejects_width_times_4_overflow() {
    let y: [u16; 0] = [];
    let mut rgba: [u16; 0] = [];
    gray16_to_rgba_u16_row::<false>(&y, &mut rgba, OVERFLOW_WIDTH, false, true);
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-planar"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuv_444p_n_u16_dispatcher_rejects_width_times_3_overflow() {
    let y: [u16; 0] = [];
    let u: [u16; 0] = [];
    let v: [u16; 0] = [];
    let mut rgb: [u16; 0] = [];
    yuv_444p_n_to_rgb_u16_row::<10, false>(
      &y,
      &u,
      &v,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-planar"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuv444p16_u16_dispatcher_rejects_width_times_3_overflow() {
    let y: [u16; 0] = [];
    let u: [u16; 0] = [];
    let v: [u16; 0] = [];
    let mut rgb: [u16; 0] = [];
    yuv444p16_to_rgb_u16_row(
      &y,
      &u,
      &v,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  // ---- Packed YUV 4:2:2 dispatchers — `width x 2` overflow ----
  //
  // The packed Tier 3 sources (Yuyv422 / Uyvy422 / Yvyu422) consume
  // `2 * width` bytes per row. Without the [`packed_yuv422_row_bytes`]
  // helper a 32-bit caller could overflow `width * 2` to a small
  // value, pass the input-side `assert!` with an undersized slice,
  // and reach unsafe SIMD loads. Each packed RGB / RGBA / luma
  // entry point gets its own regression so a future regression on
  // any one of them surfaces independently.

  /// Smallest even width whose `width x 2` overflows 32-bit `usize`
  /// without first failing the `width x 3` overflow guard the
  /// 3-channel-output dispatchers also enforce. On 32-bit
  /// `usize::MAX / 2 == 2^31 - 1` is odd, so `+ 1` produces an
  /// even value (`2^31`); the `+ (candidate & 1)` is a parity
  /// safety on hypothetical platforms where this differs.
  #[cfg(all(
    target_pointer_width = "32",
    any(feature = "yuv-packed", feature = "y2xx")
  ))]
  const OVERFLOW_WIDTH_TIMES_2: usize = {
    let candidate = (usize::MAX / 2) + 1;
    candidate + (candidate & 1)
  };

  #[cfg(all(target_pointer_width = "32", feature = "yuv-packed"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuyv422_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    yuyv422_to_rgb_row(
      &p,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_2,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-packed"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn uyvy422_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut rgba: [u8; 0] = [];
    uyvy422_to_rgba_row(
      &p,
      &mut rgba,
      OVERFLOW_WIDTH_TIMES_2,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-packed"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yvyu422_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    yvyu422_to_rgb_row(
      &p,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_2,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-packed"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuyv422_luma_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut luma: [u8; 0] = [];
    yuyv422_to_luma_row(&p, &mut luma, OVERFLOW_WIDTH_TIMES_2, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-packed"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn uyvy422_luma_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut luma: [u8; 0] = [];
    uyvy422_to_luma_row(&p, &mut luma, OVERFLOW_WIDTH_TIMES_2, false);
  }

  #[cfg(all(target_pointer_width = "32", feature = "yuv-packed"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yvyu422_luma_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut luma: [u8; 0] = [];
    yvyu422_to_luma_row(&p, &mut luma, OVERFLOW_WIDTH_TIMES_2, false);
  }

  // ---- v210 dispatcher — `(width / 6) x 16` overflow ----
  //
  // The v210 source (Tier 4) packs 6 pixels per 16-byte word, so the
  // row's byte count is `(width / 6) * 16`. Without the
  // [`v210_row_bytes`] helper, a 32-bit caller could overflow this
  // multiplication to a small value, pass the input-side `assert!`
  // with an undersized slice, and reach unsafe SIMD loads.

  #[cfg(all(target_pointer_width = "32", feature = "v210"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn v210_dispatcher_rejects_words_times_16_overflow() {
    let candidate = ((usize::MAX / 16) + 1) * 6;
    let p: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    v210_to_rgb_row(&p, &mut rgb, candidate, ColorMatrix::Bt601, true, false);
  }

  // ---- Y2xx dispatcher — `width x 2` overflow ----
  //
  // Y210 (and the upcoming Y212 / Y216) sources consume `2 * width`
  // u16 elements per row (one quadruple per chroma pair = 4 u16 per
  // 2 pixels). Without the [`y2xx_row_elems`] helper, a 32-bit caller
  // could overflow `width * 2` to a small value, pass the input-side
  // `assert!` with an undersized slice, and reach unsafe SIMD loads.

  #[cfg(all(target_pointer_width = "32", feature = "y2xx"))]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn y210_dispatcher_rejects_width_times_2_overflow() {
    // Reuse OVERFLOW_WIDTH_TIMES_2 — an even width whose `x 2`
    // overflows 32-bit `usize`. The previous `(usize::MAX / 2) + 2`
    // value was odd on i686 (since `usize::MAX / 2` is odd) and
    // tripped the even-width check before the overflow guard,
    // causing this test to panic with the wrong message under
    // miri-i686. The shared constant has the parity fixup.
    let p: [u16; 0] = [];
    let mut rgb: [u8; 0] = [];
    y210_to_rgb_row(
      &p,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_2,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }
}

#[cfg(all(test, feature = "std", feature = "bayer"))]
mod bayer_dispatcher_tests {
  //! Boundary-contract tests for the public Bayer row dispatchers.
  //! Walker / kernel correctness lives in `crate::raw::bayer*` and
  //! `crate::row::scalar`; these tests target the dispatcher's own
  //! preflight (notably the new `assert_color_transform_well_formed`
  //! check and the existing length / `BITS` / `rgb_out` checks)
  //! since that surface is what unsafe SIMD backends will rely on.
  use super::*;
  use crate::raw::{BayerDemosaic, BayerPattern};

  fn ident() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  }

  #[test]
  #[should_panic(expected = "non-finite")]
  fn bayer_dispatcher_rejects_nan_in_m() {
    let above = [0u8; 4];
    let mid = [0u8; 4];
    let below = [0u8; 4];
    let mut rgb = [0u8; 12];
    let mut m = ident();
    m[1][1] = f32::NAN;
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  #[test]
  #[should_panic(expected = "non-finite")]
  fn bayer_dispatcher_rejects_infinity_in_m() {
    let above = [0u8; 4];
    let mid = [0u8; 4];
    let below = [0u8; 4];
    let mut rgb = [0u8; 12];
    let mut m = ident();
    m[0][2] = f32::INFINITY;
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  #[test]
  #[should_panic(expected = "non-finite")]
  fn bayer16_u8_dispatcher_rejects_neg_infinity_in_m() {
    let above = [0u16; 4];
    let mid = [0u16; 4];
    let below = [0u16; 4];
    let mut rgb = [0u8; 12];
    let mut m = ident();
    m[2][1] = f32::NEG_INFINITY;
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  #[test]
  #[should_panic(expected = "non-finite")]
  fn bayer16_u16_dispatcher_rejects_nan_in_m() {
    let above = [0u16; 4];
    let mid = [0u16; 4];
    let below = [0u16; 4];
    let mut rgb = [0u16; 12];
    let mut m = ident();
    m[2][2] = f32::NAN;
    bayer16_to_rgb_u16_row::<10>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  #[test]
  fn bayer_dispatcher_accepts_finite_m() {
    // Sanity: the assertion doesn't fire for ordinary finite
    // matrices. Realistic inputs (CCM with negative crosstalk,
    // WB > 1) all qualify.
    let above = [10u8; 4];
    let mid = [20u8; 4];
    let below = [30u8; 4];
    let mut rgb = [0u8; 12];
    let m: [[f32; 3]; 3] = [[1.5, -0.3, -0.2], [-0.1, 1.2, -0.1], [-0.05, -0.15, 1.2]];
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  /// A direct row-API caller that bypasses
  /// [`crate::raw::WhiteBalance::try_new`] /
  /// [`crate::raw::ColorCorrectionMatrix::try_new`] cannot inject
  /// finite-but-extreme matrices that would overflow during the
  /// per-pixel matmul. The dispatcher's
  /// `assert_color_transform_well_formed` enforces the same
  /// magnitude bound (1e12) that validated WB x CCM can produce.
  #[test]
  #[should_panic(expected = "exceeds magnitude bound")]
  fn bayer_dispatcher_rejects_finite_extreme_m() {
    let above = [0u8; 4];
    let mid = [0u8; 4];
    let below = [0u8; 4];
    let mut rgb = [0u8; 12];
    let mut m = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    m[1][1] = f32::MAX; // finite but past the bound
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  /// Same regression for the Bayer16→u8 dispatcher.
  #[test]
  #[should_panic(expected = "exceeds magnitude bound")]
  fn bayer16_u8_dispatcher_rejects_finite_extreme_m() {
    let above = [0u16; 4];
    let mid = [0u16; 4];
    let below = [0u16; 4];
    let mut rgb = [0u8; 12];
    let mut m = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    m[2][0] = -f32::MAX;
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  /// Same regression for the Bayer16→u16 dispatcher.
  #[test]
  #[should_panic(expected = "exceeds magnitude bound")]
  fn bayer16_u16_dispatcher_rejects_finite_extreme_m() {
    let above = [0u16; 4];
    let mid = [0u16; 4];
    let below = [0u16; 4];
    let mut rgb = [0u16; 12];
    let mut m = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    m[0][0] = 1e20; // finite but past the 1e12 bound
    bayer16_to_rgb_u16_row::<10>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  /// At-bound element passes (boundary inclusive, matches the
  /// constructor bounds).
  #[test]
  fn bayer_dispatcher_accepts_at_bound_m() {
    let above = [0u8; 4];
    let mid = [0u8; 4];
    let below = [0u8; 4];
    let mut rgb = [0u8; 12];
    let m = [
      [super::MAX_FUSED_TRANSFORM_ABS, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ];
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  // ---- Direct Bayer16 row-API contract behavior --------------------------
  //
  // The walker path (`bayer16_to`) cannot reach the kernel with
  // out-of-range samples because `BayerFrame16::try_new` validates
  // every active sample at construction. The direct row API
  // (`bayer16_to_rgb_row`, `bayer16_to_rgb_u16_row`) takes raw
  // `&[u16]` slices and trusts the low-packed contract — out-of-
  // range samples are documented as "defined-but-saturated output,
  // no panic, no UB." These regressions pin that behavior so a
  // future change can't silently flip it (e.g., to a panic or to
  // masking) without updating the documented contract first.

  /// 12-bit dispatcher with MSB-aligned `0x8000` input
  /// (the classic packing-mismatch bug, where the caller forgot
  /// to right-shift before feeding the row API). Out-of-range
  /// per the low-packed contract; the kernel saturates the matmul
  /// output to `255` rather than panicking. Walker users get
  /// `Err(SampleOutOfRange)` from `try_new` instead.
  #[test]
  fn bayer16_u8_dispatcher_saturates_on_msb_aligned_input() {
    let above = [0x8000u16; 4];
    let mid = [0x8000u16; 4];
    let below = [0x8000u16; 4];
    let mut rgb = [0u8; 12];
    let m = ident();
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
    // 0x8000 = 32768 ≫ max_in (4095). All output channels clamp
    // to 255. No panic, no UB — defined behavior.
    assert!(
      rgb.iter().all(|&c| c == 255),
      "MSB-aligned 12-bit input expected to saturate to 255 across all channels; got {rgb:?}"
    );
  }

  /// Same regression for the u16 dispatcher: MSB-aligned 10-bit
  /// input saturates to the low-packed max (1023) rather than
  /// panicking.
  #[test]
  fn bayer16_u16_dispatcher_saturates_on_msb_aligned_input() {
    let above = [0xFFC0u16; 4]; // MSB-aligned 10-bit "white" (1023 << 6)
    let mid = [0xFFC0u16; 4];
    let below = [0xFFC0u16; 4];
    let mut rgb = [0u16; 12];
    let m = ident();
    bayer16_to_rgb_u16_row::<10>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
    // 0xFFC0 ≫ low-packed-10 max (1023). Output saturates to
    // 1023 (the u16 path's max_out). No panic.
    assert!(
      rgb.iter().all(|&c| c == 1023),
      "MSB-aligned 10-bit input expected to saturate to 1023 across all channels; got {rgb:?}"
    );
  }

  /// In-range Bayer16 input still works correctly through the
  /// direct row API (this protects the rest of the contract while
  /// the saturation tests pin the out-of-range behavior).
  #[test]
  fn bayer16_u8_dispatcher_in_range_input_correct() {
    let above = [4095u16; 4]; // 12-bit white, in range
    let mid = [4095u16; 4];
    let below = [4095u16; 4];
    let mut rgb = [0u8; 12];
    let m = ident();
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
    // Solid white (4095) at every site → output 255 on every
    // channel. Same final value as the saturated case, but the
    // path is correct (not a clamp).
    assert!(
      rgb.iter().all(|&c| c == 255),
      "in-range 12-bit white expected to map to 255; got {rgb:?}"
    );
  }

  /// Source-compatibility lock for the public little-endian Bayer16
  /// dispatchers: both `bayer16_to_rgb_row` and
  /// `bayer16_to_rgb_u16_row` must keep accepting the original
  /// **single-const** turbofish `::<BITS>` (the pre-endian public
  /// arity). Appending `BE` to those names in place would have broken
  /// every existing `::<BITS>` caller; the LE wrappers forward to the
  /// `_endian` variants with `BE = false` so this stays a compile-time
  /// guarantee. The body also pins that each LE wrapper produces output
  /// byte-identical to its `_endian::<BITS, false>` counterpart.
  #[test]
  fn bayer16_le_dispatchers_accept_single_const_turbofish() {
    let above = [1234u16; 4];
    let mid = [2345u16; 4];
    let below = [3210u16; 4];
    let m = ident();

    // u8 path: single-const `::<12>` (LE wrapper) vs `_endian::<12, false>`.
    let mut rgb_wrapper = [0u8; 12];
    let mut rgb_endian = [0u8; 12];
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb_wrapper,
      false,
    );
    bayer16_to_rgb_row_endian::<12, false>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb_endian,
      false,
    );
    assert_eq!(
      rgb_wrapper, rgb_endian,
      "LE wrapper must equal _endian::<_, false> (u8 RGB)"
    );

    // u16 path: single-const `::<10>` (LE wrapper) vs `_endian::<10, false>`.
    let above10 = [123u16; 4];
    let mid10 = [456u16; 4];
    let below10 = [321u16; 4];
    let mut rgb16_wrapper = [0u16; 12];
    let mut rgb16_endian = [0u16; 12];
    bayer16_to_rgb_u16_row::<10>(
      &above10,
      &mid10,
      &below10,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb16_wrapper,
      false,
    );
    bayer16_to_rgb_u16_row_endian::<10, false>(
      &above10,
      &mid10,
      &below10,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb16_endian,
      false,
    );
    assert_eq!(
      rgb16_wrapper, rgb16_endian,
      "LE wrapper must equal _endian::<_, false> (u16 RGB)"
    );
  }
}
