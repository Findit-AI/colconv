//! Semver-exempt bench-only re-exports of internal `row` kernels.
//!
//! The 0.2 surface cut made `row` `pub(crate)` (RFC
//! [#392](https://github.com/findit-studio/colconv/issues/392)). This module,
//! gated behind the `unstable-bench-internals` feature and hidden from the docs,
//! re-exposes exactly the per-row dispatchers the repository's Criterion benches
//! (`benches/*.rs`) call directly, and nothing else. It is **not** public API:
//! it has no stability guarantee, may change or vanish in any release, and must
//! never be relied on by external code — use [`Convert`](crate::Convert) or
//! [`MixedSinker`](crate::sinker::MixedSinker) with the `{fmt}_to` walkers.
//!
//! Every re-export carries the same family feature gate its underlying kernel
//! does (mirroring `row`'s own re-export table), so this module compiles under
//! any feature powerset that also enables `unstable-bench-internals`.

// These re-exports name the crate-private `row` module in their source path; the
// module is `#[doc(hidden)]` and not public API, so let the private-path lint
// pass rather than laundering `row` behind another public alias.
#![allow(rustdoc::private_intra_doc_links)]

// Cross-format HSV / luma helpers — always compiled (used by every sinker), so
// no family gate. `benches/rgb_to_hsv.rs`, `rgb24_to_luma.rs`.
pub use crate::row::{rgb_to_hsv_row, rgb_to_luma_row};

// Packed 8/16-bit RGB / RGBA dispatchers (`rgb`). `benches/bgr24_to_rgb.rs`,
// `rgba_to_luma.rs`, `x2rgb10_to_rgb.rs`, `rgb48_to_rgb.rs`, `rgba64_to_rgba.rs`.
#[cfg(feature = "rgb")]
pub use crate::row::{
  bgr_to_rgb_row, rgb48_to_rgb_row, rgba_to_rgb_row, rgba64_to_rgba_row, x2rgb10_to_rgb_row,
};

// Packed float RGB dispatchers (`rgb-float`). `benches/rgbf16_to_rgb.rs`,
// `rgbf32_to_rgb.rs`.
#[cfg(feature = "rgb-float")]
pub use crate::row::{rgbf16_to_rgb_row, rgbf32_to_rgb_row};

// Legacy packed RGB dispatchers (`rgb-legacy`). `benches/rgb565_to_rgb.rs`.
#[cfg(feature = "rgb-legacy")]
pub use crate::row::rgb565_to_rgb_row;

// Planar GBR(+A) dispatchers (`gbr`). `benches/gbrp_to_rgb.rs`,
// `gbrap_to_rgba.rs`, `gbrp12_to_rgb.rs`.
#[cfg(feature = "gbr")]
pub use crate::row::{gbr_to_rgb_high_bit_row, gbr_to_rgb_row, gbra_to_rgba_row};

// Indexed-palette dispatchers (`mono`). `benches/pal8_simd.rs`.
#[cfg(feature = "mono")]
pub use crate::row::{
  pal8_to_rgb_row, pal8_to_rgb_u16_row, pal8_to_rgba_row, pal8_to_rgba_u16_row,
};

// Bayer demosaic dispatchers (`bayer`). `benches/bayer_to_rgb.rs`,
// `bayer16_to_rgb.rs`.
#[cfg(feature = "bayer")]
pub use crate::row::{bayer_to_rgb_row, bayer16_to_rgb_row};

// CIE-XYZ (DCP) dispatcher (`xyz`; the XYZ decode allocates, so it rides the
// same `any(std, alloc)` gate as its `row` re-export). `benches/xyz12_to_rgb.rs`.
#[cfg(all(feature = "xyz", any(feature = "std", feature = "alloc")))]
pub use crate::row::xyz12_to_rgb_row;

// Planar YUV dispatchers (`yuv-planar`): 4:2:0 / 4:4:4 8-bit + low-packed
// high-bit, plus the 4:1:0 / 4:1:1 legacy planars. `benches/yuv_4*p*_to_rgb.rs`,
// `yuv_410p_to_rgb.rs`, `yuv_411p_to_rgb.rs`.
#[cfg(feature = "yuv-planar")]
pub use crate::row::{
  yuv_410_to_rgb_row, yuv_411_to_rgb_row, yuv_420_to_rgb_row, yuv_444_to_rgb_row,
  yuv420p10_to_rgb_row, yuv420p10_to_rgb_u16_row, yuv420p12_to_rgb_row, yuv420p12_to_rgb_u16_row,
  yuv420p14_to_rgb_row, yuv420p14_to_rgb_u16_row, yuv420p16_to_rgb_row, yuv420p16_to_rgb_u16_row,
  yuv444p10_to_rgb_row, yuv444p10_to_rgb_u16_row, yuv444p12_to_rgb_row, yuv444p12_to_rgb_u16_row,
  yuv444p14_to_rgb_row, yuv444p14_to_rgb_u16_row, yuv444p16_to_rgb_row, yuv444p16_to_rgb_u16_row,
};

// Semi-planar YUV dispatchers (`yuv-semi-planar`): NV12/21/24/42 8-bit and the
// high-packed P010/012/016 (4:2:0) / P410/412/416 (4:4:4). `benches/nv*_to_rgb.rs`,
// `p0*_to_rgb.rs`, `p2*_to_rgb.rs`, `p4*_to_rgb.rs`.
#[cfg(feature = "yuv-semi-planar")]
pub use crate::row::{
  nv12_to_rgb_row, nv21_to_rgb_row, nv24_to_rgb_row, nv42_to_rgb_row, p010_to_rgb_row,
  p010_to_rgb_u16_row, p012_to_rgb_row, p012_to_rgb_u16_row, p016_to_rgb_row, p016_to_rgb_u16_row,
  p410_to_rgb_row, p410_to_rgb_u16_row, p412_to_rgb_row, p412_to_rgb_u16_row, p416_to_rgb_row,
  p416_to_rgb_u16_row,
};

// Planar YUVA alpha-preserving dispatchers (`yuva`). `benches/yuva_4_*_a_plus_combo.rs`.
#[cfg(feature = "yuva")]
pub use crate::row::{
  yuva420p_to_rgba_row, yuva420p10_to_rgba_row, yuva420p10_to_rgba_u16_row, yuva420p16_to_rgba_row,
  yuva420p16_to_rgba_u16_row, yuva444p_to_rgba_row, yuva444p10_to_rgba_row,
  yuva444p10_to_rgba_u16_row, yuva444p16_to_rgba_row, yuva444p16_to_rgba_u16_row,
};

// Packed YUV 4:2:2 / 4:1:1 dispatchers (`yuv-packed`). `benches/yuyv422_to_rgb.rs`,
// `uyvy422_to_rgb.rs`, `yvyu422_to_rgb.rs`, `uyyvyy411_to_rgb.rs`.
#[cfg(feature = "yuv-packed")]
pub use crate::row::{
  uyvy422_to_rgb_row, uyyvyy411_to_rgb_row, yuyv422_to_rgb_row, yvyu422_to_rgb_row,
};

// Packed YUV 4:4:4 dispatchers (`yuv-444-packed`): AYUV64, VUYA/VUYX, V30X,
// V410, XV36. `benches/ayuv64_a_plus_combo.rs`, `vuya_a_plus_combo.rs`,
// `vuyx_to_rgb.rs`, `v30x_to_rgb.rs`, `v410_to_rgb.rs`, `xv36_to_rgb.rs`.
#[cfg(feature = "yuv-444-packed")]
pub use crate::row::{
  ayuv64_to_rgb_row, ayuv64_to_rgb_u16_row, ayuv64_to_rgba_row, ayuv64_to_rgba_u16_row,
  v30x_to_rgb_row, v410_to_rgb_row, vuya_to_rgb_row, vuya_to_rgba_row, vuyx_to_rgba_row,
  xv36_to_rgb_row,
};

// Packed Y2xx (MSB-aligned u16) dispatchers (`y2xx`). `benches/y21*_to_rgb.rs`.
#[cfg(feature = "y2xx")]
pub use crate::row::{y210_to_rgb_row, y212_to_rgb_row, y216_to_rgb_row};

// Packed V210 (3x10-bit/u32) dispatcher (`v210`). `benches/v210_to_rgb.rs`.
#[cfg(feature = "v210")]
pub use crate::row::v210_to_rgb_row;
