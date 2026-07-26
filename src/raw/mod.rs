//! RAW (Bayer) source kernels.
//!
//! `pixon` ingests Bayer-mosaic frames produced by upstream
//! camera-RAW pipelines (RED REDline / R3D, Blackmagic RAW / BRAW,
//! Nikon NRAW SDK, FFmpeg's `bayer_*` decoders) and runs demosaic +
//! white balance + 3x3 color-correction in a single per-row kernel.
//!
//! # Scope
//!
//! `pixon` covers **demosaic onwards**. Decoding the camera's
//! compressed bitstream into a Bayer plane is the vendor SDK's job
//! (RED SDK / BRAW SDK / Nikon Decoder); `pixon` consumes the
//! resulting [`crate::frame::BayerFrame`] /
//! [`crate::frame::BayerFrame16`] and produces RGB rows.
//!
//! # Parameters
//!
//! Three caller-supplied values shape the output (none of them
//! belong on the source frame itself, since the same Bayer plane can
//! legitimately render with different choices):
//!
//! - [`BayerPattern`] — which sensor color sits at the top-left of
//!   the repeating 2x2 tile.
//! - [`WhiteBalance`] — per-channel R / G / B gains.
//! - [`ColorCorrectionMatrix`] — 3x3 RGB→RGB transform from sensor
//!   primaries into the working space.
//!
//! The walker fuses [`WhiteBalance`] and [`ColorCorrectionMatrix`]
//! into a single 3x3 transform (`M = CCM · diag(wb)`) once at
//! `*_to` entry, so the per-pixel arithmetic is one 3x3 matmul.
//!
//! # Demosaic algorithm
//!
//! Selected via [`BayerDemosaic`]. Currently only
//! [`BayerDemosaic::Bilinear`] (3x3 row window, 4-tap horizontal /
//! vertical average for the missing channels) is wired up. The enum
//! is `#[non_exhaustive]` so future variants (e.g. Malvar-He-Cutler)
//! can land without a breaking change.
//!
//! # Memory model
//!
//! The walker performs **zero per-row and zero per-frame
//! allocation**. For row `r` it slices `above`, `mid`, `below`
//! references into the source plane (clamping at row 0 and
//! row `height − 1`) and hands them to the sink as a
//! `BayerRow{,16}` borrow. The sink owns the RGB output buffer for
//! the lifetime of the run; the kernel writes into it in place.

#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
pub use mediaframe::frame::{
  BayerDemosaic, BayerPattern, ColorCorrectionMatrix, ColorCorrectionMatrixError, WbChannel,
  WhiteBalance, WhiteBalanceError,
};

#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
mod bayer;
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
pub use bayer::*;
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
mod bayer16;
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
pub use bayer16::*;
#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
mod pal8;
mod types;

#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
pub use pal8::*;
