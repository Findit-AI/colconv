//! Public row-dispatcher submodules. The dispatchers were extracted
//! from `row::mod` here so the parent module stays focused on
//! shared helpers, runtime CPU feature detection, and crate-private
//! `arch` / `scalar` glue.
//!
//! Submodules are gated `pub(super) mod` and re-exported via
//! `pub use` in `row::mod`, so the public API still appears at
//! `crate::row::*` (e.g. `crate::row::yuv_420_to_rgb_row`). Callers
//! see no API change from the split.

pub(super) mod bayer;
pub(super) mod nv;
pub(super) mod packed_yuv422;
pub(super) mod pn;
pub(super) mod rgb_ops;
pub(super) mod v210;
pub(super) mod y210;
pub(super) mod y212;
pub(super) mod y216;
pub(super) mod yuv420;
pub(super) mod yuv444;
pub(super) mod yuva;
