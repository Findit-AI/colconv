//! High-bit 4:2:0 (Yuv420pN / P010 / P012 / P016) MixedSinker tests,
//! split by sub-bit-depth so no single file exceeds ~1.5 KLoC.
//!
//! `pub(super) use super::*;` forwards `tests/mod.rs`'s prelude into
//! this module so each per-bit-depth submodule can pick it up via
//! its own `use super::*;`. The `pub(super) use yuv420p10_p010::*`
//! / `yuv420p16_p016::*` re-exports surface the `solid_yuv420p10_frame`
//! / `solid_yuv420p16_frame` builders to siblings of this module
//! (currently `tests/yuva/sub_4_2_0.rs` reaches in for high-bit
//! YUVA cross-fixtures).

#[allow(unused_imports)]
pub(super) use super::*;

mod yuv420p10_p010;
mod yuv420p12_14_p012;
mod yuv420p16_p016;

#[allow(unused_imports)]
pub(super) use yuv420p10_p010::solid_yuv420p10_frame;
#[allow(unused_imports)]
pub(super) use yuv420p16_p016::solid_yuv420p16_frame;
