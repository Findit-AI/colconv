//! [`PixelSink`](crate::PixelSink) implementations shipped with the
//! crate.
//!
//! Currently ships [`MixedSinker`](mixed::MixedSinker), which writes
//! any subset of `{RGB, RGBA, Luma, LumaU16, HSV, RGB-u16, RGBA-u16,
//! RGB-f32, RGB-f16, XYZ-f32}` into caller-provided buffers. It has
//! per-format `PixelSink` impls for every shipped source format —
//! YUV planar / semi-planar / packed (8-bit, 9-16-bit, P010/P210/P410
//! families), packed RGB / RGBA / GBR planar (8-bit through f32),
//! Gray / Ya / Mono, packed YUV-α (AYUV64 / VUYA / VUYX), and DCP
//! XYZ12. See [`crate::source`] for the canonical format list.
//! Narrow newtype shortcuts (luma-only, RGB-only, HSV-only) are a
//! follow-up.
//!
//! `MixedSinker` keeps a lazily‑grown `Vec<u8>` scratch buffer for
//! the HSV‑without‑RGB path, so it is only compiled under the `std`
//! or `alloc` feature.

#[cfg(any(feature = "std", feature = "alloc"))]
pub mod mixed;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use mixed::{
  AlphaMode, CustomLumaCoefficients, DefaultAlphaMode, DimensionMismatch, GeometryOverflow,
  InsufficientBuffer, InsufficientHsvPlane, LumaChannel, LumaCoefficients, LumaCoefficientsError,
  MixedSinker, MixedSinkerError, NativeRouteChanged, ResampleOutputsChanged, RowIndexOutOfRange,
  RowShapeMismatch, WidthAlignment, WidthAlignmentRequirement,
};

/// The SMPTE ST 428-1 CIE-XYZ interpretation toggle (#310) and its rejection
/// error, gated on the sink's `yuv-planar` `ChromaDerivedNcl` path.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
pub use mixed::{St428CieXyzUnsupported, St428Interpretation};
