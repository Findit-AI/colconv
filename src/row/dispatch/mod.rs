//! Public row-dispatcher submodules. The dispatchers were extracted
//! from `row::mod` here so the parent module stays focused on
//! shared helpers, runtime CPU feature detection, and crate-private
//! `arch` / `scalar` glue.
//!
//! Submodules are `pub(super)` — they are re-exported via `pub use` in
//! `row::mod`, so the public API still appears at `crate::row::*`
//! (e.g. `crate::row::yuv_420_to_rgb_row`). Sinker code that needs
//! dispatcher functions reaches them through those same re-exports.
//! Callers see no API change from the split.

// Consumers: source families with a source-α channel (`gbr` Gbrap,
// `yuv-444-packed` AYUV64, `yuva` planar α).
#[cfg(any(feature = "gbr", feature = "yuv-444-packed", feature = "yuva"))]
pub(crate) mod alpha_extract;
// Consumer: the fused-downscale engine (`crate::resample`).
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
pub(super) mod area_reduce;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod ayuv64;
#[cfg(feature = "bayer")]
pub(super) mod bayer;
// Consumer: the centered-siting (#302) chroma-reconstruction sink paths.
// Union gate over the u8 / u16 / P-format consumer families; the per-fn
// gates inside select which dispatchers exist in a given feature subset.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuva"
  )
))]
pub(super) mod chroma_upsample;
// Consumer: the separable filter resampler (`crate::resample::filter`),
// the signed twin of `area_reduce`; same 14-feature engine cascade.
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
pub(super) mod filter_reduce;
#[cfg(feature = "gray")]
pub(super) mod gray;
#[cfg(feature = "gray")]
pub(super) mod grayf16;
#[cfg(feature = "gray")]
pub(super) mod grayf32;
#[cfg(feature = "rgb-legacy")]
pub(super) mod legacy_rgb;
#[cfg(feature = "mono")]
pub(super) mod mono1bit;
#[cfg(feature = "yuv-semi-planar")]
pub(super) mod nv;
#[cfg(feature = "yuv-semi-planar")]
pub(super) mod nv20;
#[cfg(feature = "rgb")]
pub(super) mod packed_rgb_16bit;
#[cfg(feature = "rgb")]
pub(super) mod packed_rgb_32bit;
#[cfg(feature = "yuv-packed")]
pub(super) mod packed_yuv411;
#[cfg(feature = "yuv-packed")]
pub(super) mod packed_yuv422;
#[cfg(feature = "mono")]
pub(super) mod pal8;
#[cfg(feature = "gbr")]
pub(super) mod planar_gbr;
#[cfg(feature = "gbr")]
pub(super) mod planar_gbr_32bit;
#[cfg(feature = "gbr")]
pub(super) mod planar_gbr_float;
#[cfg(feature = "gbr")]
pub(super) mod planar_gbr_high_bit;
#[cfg(feature = "gbr")]
pub(super) mod planar_gbr_msb;
#[cfg(feature = "yuv-semi-planar")]
pub(super) mod pn;
#[cfg(feature = "rgb-float")]
pub(super) mod rgb_f16_ops;
#[cfg(feature = "rgb-float")]
pub(super) mod rgb_float_ops;
// rgb_ops contains both cross-format HSV/luma helpers (rgb_to_hsv_row,
// rgb_to_luma_row, rgb_to_luma_u16_row) used by every sinker AND
// packed RGB / RGBA / X2RGB10 / X2BGR10 dispatchers. Kept always-on so
// the sinker HSV / luma derivations stay reachable when the `rgb`
// family is disabled. Unused packed-RGB dispatchers within will emit
// dead-code warnings under no-`rgb` builds, which is acceptable.
#[cfg(feature = "yuv-444-packed")]
pub(super) mod ayuv;
pub(super) mod rgb_ops;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod uyva;
#[cfg(feature = "v210")]
pub(super) mod v210;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod v30x;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod v410;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod vuya;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod vuyx;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod vyu444;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod xv36;
#[cfg(feature = "yuv-444-packed")]
pub(super) mod xv48;
#[cfg(all(feature = "xyz", any(feature = "std", feature = "alloc")))]
pub(super) mod xyz12;
#[cfg(feature = "y2xx")]
pub(super) mod y210;
#[cfg(feature = "y2xx")]
pub(super) mod y212;
#[cfg(feature = "y2xx")]
pub(super) mod y216;
// Consumers: source families that ship a u8 luma plane to MixedSinker's
// u16 luma fan-out (`gray`, `yuv-planar`, `yuv-semi-planar`, `yuva`).
#[cfg(any(
  feature = "gray",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
pub(crate) mod y_plane_to_luma_u16;
#[cfg(feature = "gray")]
pub(super) mod ya16;
#[cfg(feature = "gray")]
pub(super) mod ya8;
#[cfg(feature = "gray")]
pub(super) mod yaf16;
#[cfg(feature = "gray")]
pub(super) mod yaf32;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv411p;
// `yuv420` hosts both the planar 4:2:0 dispatchers (`yuv-planar`) and
// the semi-planar P010/P012/P016 dispatchers (`yuv-semi-planar`); the
// per-submodule gates inside `yuv420/mod` keep each family separable,
// so the parent compiles whenever either family is on.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
pub(super) mod yuv420;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv444;
#[cfg(feature = "yuva")]
pub(super) mod yuva;

// Dispatch-level BE/LE parity tests for the high-bit YUV planar and
// P-format families. The per-format dispatchers in `yuv420::*`,
// `yuv444::*`, and `pn::*` expose `_endian` entry points; this module
// asserts that the BE path is reachable and produces byte-identical
// output to the LE path on matching fixtures.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
mod be_yuv_hb_parity_tests;
