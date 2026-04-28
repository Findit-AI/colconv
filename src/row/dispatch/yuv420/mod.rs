//! YUV 4:2:0 dispatchers, split per source format for readability.
//!
//! - `yuv_420` — 8-bit YUV 4:2:0 → RGB / RGBA.
//! - `yuv420p9` / `yuv420p10` / `yuv420p12` / `yuv420p14` /
//!   `yuv420p16` — high-bit planar 4:2:0 (4 variants per format:
//!   RGB, RGB-u16, RGBA, RGBA-u16).
//! - `p010` / `p012` / `p016` — high-bit semi-planar 4:2:0
//!   (4 variants per format).
//!
//! Public functions re-exported up to `crate::row::*` via parent
//! `dispatch/mod.rs`.

pub(super) mod p010;
pub(super) mod p012;
pub(super) mod p016;
pub(super) mod yuv420p10;
pub(super) mod yuv420p12;
pub(super) mod yuv420p14;
pub(super) mod yuv420p16;
pub(super) mod yuv420p9;
pub(super) mod yuv_420;

pub use p010::*;
pub use p012::*;
pub use p016::*;
pub use yuv_420::*;
pub use yuv420p9::*;
pub use yuv420p10::*;
pub use yuv420p12::*;
pub use yuv420p14::*;
pub use yuv420p16::*;
