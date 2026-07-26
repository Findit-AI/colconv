//! 8-bit Bayer (`AV_PIX_FMT_BAYER_BGGR8` / `RGGB8` / `GRBG8` /
//! `GBRG8`) — single-plane mosaic source.
//!
//! Walker hands each output row to a [`BayerSink`] together with
//! the three row-aligned slices the demosaic kernel needs (`above`,
//! `mid`, `below`) and the fused `M = CCM · diag(wb)` transform.
//! The kernel does the bilinear demosaic and the 3x3 matmul in one
//! pass; the sink owns the RGB output buffer.

// The Bayer marker now lives in mediaframe::source and implements
// mediaframe::SourceFormat via its own sealed impl.  Re-export it at
// this path so downstream code that uses `pixon::raw::Bayer` keeps
// working without changes.
pub use mediaframe::{
  frame::{BayerFrameError, BayerRow, BayerSink, bayer_to},
  source::Bayer,
};

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests;
