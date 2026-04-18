//! [`PixelSink`](crate::PixelSink) implementations shipped with the
//! crate.
//!
//! v0.1 ships [`MixedSinker`](mixed::MixedSinker), which writes any
//! subset of `{BGR, Luma, HSV}` into caller-provided buffers. Narrow
//! newtype shortcuts (luma-only, BGR-only, HSV-only) will be added in
//! follow-up commits once the MixedSinker path is proven.
//!
//! `MixedSinker` keeps a lazily‑grown `Vec<u8>` scratch buffer for
//! the HSV‑without‑BGR path, so it is only compiled under the `std`
//! or `alloc` feature.

#[cfg(any(feature = "std", feature = "alloc"))]
pub mod mixed;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use mixed::MixedSinker;
