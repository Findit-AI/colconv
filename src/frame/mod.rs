//! Validated source-frame types.
//!
//! Each pixel family has its own frame struct carrying the backing
//! plane slice(s), pixel dimensions, and byte strides. Construction
//! validates strides vs. widths and that each plane covers its
//! declared area.

// Per-format-family submodules. Each file groups one logical pixel
// family so no single source ends up over ~1.5k lines, mirroring the
// split already done for `yuva/`, `packed_rgb_8bit`, etc.
mod bayer;
mod packed_rgb_10bit;
mod packed_rgb_8bit;
mod packed_yuv_4_4_4;
mod packed_yuv_8bit;
mod planar_8bit;
mod semi_planar_8bit;
mod subsampled_high_bit_planar;
mod subsampled_high_bit_pn;
mod v210;
mod y2xx;
mod yuva;

pub use bayer::*;
pub use packed_rgb_8bit::*;
pub use packed_rgb_10bit::*;
pub use packed_yuv_4_4_4::*;
pub use packed_yuv_8bit::*;
pub use planar_8bit::*;
pub use semi_planar_8bit::*;
pub use subsampled_high_bit_planar::*;
pub use subsampled_high_bit_pn::*;
pub use v210::*;
pub use y2xx::*;
pub use yuva::*;

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests;
