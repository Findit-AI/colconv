//! [`PixelSink`](crate::PixelSink) implementations shipped with the
//! crate.
//!
//! Currently ships [`MixedSinker`](mixed::MixedSinker), which writes
//! any subset of `{RGB, Luma, HSV}` into caller-provided buffers.
//! It has per-format `PixelSink` impls for all eight shipped YUV
//! source formats (see [`crate::yuv`] for the list). Narrow newtype
//! shortcuts (luma-only, RGB-only, HSV-only) are a follow-up.
//!
//! `MixedSinker` keeps a lazily‑grown `Vec<u8>` scratch buffer for
//! the HSV‑without‑RGB path, so it is only compiled under the `std`
//! or `alloc` feature.

#[cfg(any(feature = "std", feature = "alloc"))]
pub mod mixed;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use mixed::MixedSinker;
