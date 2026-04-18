//! [`PixelSink`](crate::PixelSink) implementations shipped with the
//! crate.
//!
//! v0.1 ships [`MixedSinker`](mixed::MixedSinker), which writes any
//! subset of `{BGR, Luma, HSV}` into caller-provided buffers. Narrow
//! newtype shortcuts (luma-only, BGR-only, HSV-only) will be added in
//! follow-up commits once the MixedSinker path is proven.

pub mod mixed;

pub use mixed::{HsvBuffers, MixedSinker};
