//! 8-bit indexed-color (`AV_PIX_FMT_PAL8`) — single-plane mosaic with a
//! 256-entry BGRA palette.
//!
//! The full quartet (marker, [`Pal8Row`], [`Pal8Sink`], [`pal8_to`]) now
//! lives in `mediaframe::source` (under the `mono` feature). This module
//! is a thin re-export shim so downstream code that uses the
//! `pixon::raw::Pal8*` paths keeps compiling unchanged.

// Re-export everything so downstream code that uses `pixon::raw::Pal8`,
// `pixon::raw::Pal8Row`, `pixon::raw::Pal8Sink`, and
// `pixon::raw::pal8_to` keeps compiling unchanged.
pub use mediaframe::source::{Pal8, Pal8Row, Pal8Sink, pal8_to};
