//! [`MixedSinker`] — the common "I want some subset of {RGB, Luma, HSV}
//! written into my own buffers" consumer.
//!
//! Generic over the source format via an `F: SourceFormat` type
//! parameter. One `PixelSink` impl per supported format. Currently
//! ships impls for:
//!
//! - **8‑bit planar**: [`Yuv420p`](crate::yuv::Yuv420p),
//!   [`Yuv422p`](crate::yuv::Yuv422p),
//!   [`Yuv440p`](crate::yuv::Yuv440p),
//!   [`Yuv444p`](crate::yuv::Yuv444p).
//! - **8‑bit semi‑planar**: [`Nv12`](crate::yuv::Nv12),
//!   [`Nv21`](crate::yuv::Nv21), [`Nv16`](crate::yuv::Nv16),
//!   [`Nv24`](crate::yuv::Nv24), [`Nv42`](crate::yuv::Nv42).
//! - **9/10/12/14/16‑bit planar 4:2:0**:
//!   [`Yuv420p9`](crate::yuv::Yuv420p9),
//!   [`Yuv420p10`](crate::yuv::Yuv420p10),
//!   [`Yuv420p12`](crate::yuv::Yuv420p12),
//!   [`Yuv420p14`](crate::yuv::Yuv420p14),
//!   [`Yuv420p16`](crate::yuv::Yuv420p16).
//! - **9/10/12/14/16‑bit planar 4:2:2**:
//!   [`Yuv422p9`](crate::yuv::Yuv422p9),
//!   [`Yuv422p10`](crate::yuv::Yuv422p10),
//!   [`Yuv422p12`](crate::yuv::Yuv422p12),
//!   [`Yuv422p14`](crate::yuv::Yuv422p14),
//!   [`Yuv422p16`](crate::yuv::Yuv422p16).
//! - **10/12‑bit planar 4:4:0**:
//!   [`Yuv440p10`](crate::yuv::Yuv440p10),
//!   [`Yuv440p12`](crate::yuv::Yuv440p12).
//! - **9/10/12/14/16‑bit planar 4:4:4**:
//!   [`Yuv444p9`](crate::yuv::Yuv444p9),
//!   [`Yuv444p10`](crate::yuv::Yuv444p10),
//!   [`Yuv444p12`](crate::yuv::Yuv444p12),
//!   [`Yuv444p14`](crate::yuv::Yuv444p14),
//!   [`Yuv444p16`](crate::yuv::Yuv444p16).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:2:0**:
//!   [`P010`](crate::yuv::P010), [`P012`](crate::yuv::P012),
//!   [`P016`](crate::yuv::P016).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:2:2**:
//!   [`P210`](crate::yuv::P210), [`P212`](crate::yuv::P212),
//!   [`P216`](crate::yuv::P216).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:4:4**:
//!   [`P410`](crate::yuv::P410), [`P412`](crate::yuv::P412),
//!   [`P416`](crate::yuv::P416).
//! - **YUVA (alpha-bearing planar)**: the entire FFmpeg-shipped
//!   YUVA family — `Yuva420p` / `Yuva420p9/10/16`, `Yuva422p` /
//!   `Yuva422p9/10/12/16`, `Yuva444p` / `Yuva444p9/10/12/14/16`.
//!   Source-side alpha pass-through to `with_rgba` /
//!   `with_rgba_u16`, with native SIMD on every backend.
//! - **8‑bit packed RGB sources** (Tier 6):
//!   [`Rgb24`](crate::yuv::Rgb24) (`R, G, B` bytes),
//!   [`Bgr24`](crate::yuv::Bgr24) (`B, G, R` bytes),
//!   [`Rgba`](crate::yuv::Rgba) (`R, G, B, A` bytes),
//!   [`Bgra`](crate::yuv::Bgra) (`B, G, R, A` bytes),
//!   [`Argb`](crate::yuv::Argb) (`A, R, G, B` bytes — leading alpha),
//!   [`Abgr`](crate::yuv::Abgr) (`A, B, G, R` bytes — leading alpha),
//!   [`Xrgb`](crate::yuv::Xrgb) / [`Rgbx`](crate::yuv::Rgbx) /
//!   [`Xbgr`](crate::yuv::Xbgr) / [`Bgrx`](crate::yuv::Bgrx)
//!   (4-byte packed RGB with one ignored padding byte at the leading
//!   or trailing position).
//!   The source row is already 8‑bit RGB at the byte level —
//!   `with_rgb` is an identity copy / channel swap /
//!   drop-alpha-or-padding, `with_rgba` is a memcpy / channel
//!   reorder (alpha passed through for the alpha-bearing 4-byte
//!   sources, forced to `0xFF` for the 3-byte sources and the
//!   padding-byte family), `with_luma` derives Y' from R/G/B,
//!   `with_hsv` reuses the existing kernel.
//! - **10‑bit packed RGB sources** (Tier 6 — Ship 9e):
//!   [`X2Rgb10`](crate::yuv::X2Rgb10) and
//!   [`X2Bgr10`](crate::yuv::X2Bgr10). Each pixel is a 32-bit LE word
//!   with `(MSB) 2X | 10c2 | 10c1 | 10c0 (LSB)` (R/G/B for X2RGB10,
//!   B/G/R for X2BGR10). Unlike the 8‑bit byte-shuffle family above,
//!   the source is **not** byte-aligned RGB — every output path
//!   starts with bit-level extraction of the three 10‑bit channels:
//!   `with_rgb` extracts and down-shifts each channel from 10→8 bits,
//!   `with_rgba` does the same and forces alpha to `0xFF` (the 2‑bit
//!   field is padding, not real alpha), `with_rgb_u16` preserves
//!   native 10‑bit precision (low-bit aligned in `u16`, value range
//!   `[0, 1023]`), and `with_luma` / `with_hsv` reuse the staged u8
//!   RGB scratch path.
//!
//! High‑bit‑depth source impls expose both `with_rgb` (u8 output) and
//! `with_rgb_u16` (native‑depth u16 output). Calling `with_rgb_u16` on
//! an 8‑bit source format is a compile error.
//!
//! All configuration and processing methods are fallible — no panics
//! under normal contract violations — so the sink is usable on
//! `panic = "abort"` targets.

use core::marker::PhantomData;

use std::vec::Vec;

use derive_more::{Display, IsVariant};
use thiserror::Error;

// Per-format imports moved to the child modules (`planar_8bit`,
// `semi_planar_8bit`, `subsampled_4_*_high_bit`, `bayer`). mod.rs only
// keeps the prelude types and the helpers — neither of which references
// any specific source-format type.
use crate::{HsvBuffers, SourceFormat};
// PixelSink is referenced only via intra-doc links (`[`PixelSink::*`]`)
// in this file; the rustc lint can't see those uses, so silence it.
#[allow(unused_imports)]
use crate::PixelSink;

/// Errors returned by [`MixedSinker`] configuration and per-frame
/// preflight.
///
/// All variants are recoverable: the sinker never mutates caller
/// buffers on an error return, so the caller can inspect the variant,
/// rebuild or resize buffers, and retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MixedSinkerError {
  /// The frame handed to the walker does not match the dimensions
  /// declared at [`MixedSinker::new`]. Returned from
  /// [`PixelSink::begin_frame`] before any row is processed.
  #[error(
    "MixedSinker frame dimensions mismatch: configured {configured_w}×{configured_h} but got {frame_w}×{frame_h}"
  )]
  DimensionMismatch {
    /// Width declared at sinker construction.
    configured_w: usize,
    /// Height declared at sinker construction.
    configured_h: usize,
    /// Width of the frame handed to the walker.
    frame_w: u32,
    /// Height of the frame handed to the walker.
    frame_h: u32,
  },

  /// RGB buffer attached via [`MixedSinker::with_rgb`] /
  /// [`MixedSinker::set_rgb`] is shorter than `width × height × 3`.
  #[error("MixedSinker rgb buffer too short: expected >= {expected} bytes, got {actual}")]
  RgbBufferTooShort {
    /// Minimum bytes required (`width × height × 3`).
    expected: usize,
    /// Bytes supplied.
    actual: usize,
  },

  /// `u16` RGB buffer attached via [`MixedSinker::with_rgb_u16`] /
  /// [`MixedSinker::set_rgb_u16`] is shorter than `width × height × 3`
  /// `u16` elements. Only the high‑bit‑depth source impls
  /// (currently [`Yuv420p10`](crate::yuv::Yuv420p10)) write into this
  /// buffer.
  #[error("MixedSinker rgb_u16 buffer too short: expected >= {expected} elements, got {actual}")]
  RgbU16BufferTooShort {
    /// Minimum `u16` elements required (`width × height × 3`).
    expected: usize,
    /// `u16` elements supplied.
    actual: usize,
  },

  /// RGBA buffer attached via [`MixedSinker::with_rgba`] /
  /// [`MixedSinker::set_rgba`] is shorter than `width × height × 4`.
  /// The fourth byte per pixel is alpha — opaque (`0xFF`) by default
  /// when the source has no alpha plane.
  #[error("MixedSinker rgba buffer too short: expected >= {expected} bytes, got {actual}")]
  RgbaBufferTooShort {
    /// Minimum bytes required (`width × height × 4`).
    expected: usize,
    /// Bytes supplied.
    actual: usize,
  },

  /// `u16` RGBA buffer attached via `with_rgba_u16` / `set_rgba_u16`
  /// (per-format impl, not yet shipped on any sink) is shorter than
  /// `width × height × 4` `u16` elements. Only high‑bit‑depth source
  /// impls write into this buffer; the fourth `u16` per pixel is
  /// alpha — opaque (`(1 << BITS) - 1`) by default when the source
  /// has no alpha plane.
  #[error("MixedSinker rgba_u16 buffer too short: expected >= {expected} elements, got {actual}")]
  RgbaU16BufferTooShort {
    /// Minimum `u16` elements required (`width × height × 4`).
    expected: usize,
    /// `u16` elements supplied.
    actual: usize,
  },

  /// Luma buffer is shorter than `width × height`.
  #[error("MixedSinker luma buffer too short: expected >= {expected} bytes, got {actual}")]
  LumaBufferTooShort {
    /// Minimum bytes required (`width × height`).
    expected: usize,
    /// Bytes supplied.
    actual: usize,
  },

  /// One of the three HSV planes is shorter than `width × height`.
  #[error("MixedSinker hsv {which:?} plane too short: expected >= {expected} bytes, got {actual}")]
  HsvPlaneTooShort {
    /// Which HSV plane was short (H, S, or V).
    which: HsvPlane,
    /// Minimum bytes required (`width × height`).
    expected: usize,
    /// Bytes supplied.
    actual: usize,
  },

  /// Declared frame geometry does not fit in `usize`. Only reachable
  /// on 32‑bit targets (wasm32, i686) with extreme dimensions.
  #[error("MixedSinker frame size overflows usize: {width} × {height} × channels={channels}")]
  GeometryOverflow {
    /// Configured width.
    width: usize,
    /// Configured height.
    height: usize,
    /// Channel count the overflowing product was computed with.
    channels: usize,
  },

  /// A row handed directly to [`PixelSink::process`] has a slice
  /// length that doesn't match the sink's configured width. Returned
  /// by `process` as a defense-in-depth check — [`PixelSink::begin_frame`]
  /// already validates frame-level dimensions, but this catches
  /// direct `process` callers that bypass the walker (hand-crafted
  /// rows, replayed rows, etc.) before a wrong-shaped slice reaches
  /// an unsafe SIMD kernel.
  ///
  /// Lengths are expressed in **slice elements** — `u8` bytes for
  /// the 8‑bit source rows (Y, U/V half, UV/VU half) and `u16`
  /// elements for the 10‑bit source rows (Y10, U/V half 10). The
  /// message deliberately says "elements" rather than "bytes" so the
  /// same variant can serve both the `u8` and `u16` row families.
  #[error(
    "MixedSinker row shape mismatch at row {row}: {which} slice has {actual} elements, expected {expected}"
  )]
  RowShapeMismatch {
    /// Which slice mismatched. See [`RowSlice`] for variants.
    which: RowSlice,
    /// Row index reported by the offending row.
    row: usize,
    /// Expected slice length in elements of the slice's element type
    /// (`u8` for 8‑bit source rows; `u16` for 10‑bit source rows).
    expected: usize,
    /// Actual slice length in the same unit as `expected`.
    actual: usize,
  },

  /// A row handed to [`PixelSink::process`] has `row.row() >=
  /// configured_height`. The walker bounds `idx < height` via its
  /// `for row in 0..h` loop combined with the `begin_frame`
  /// dimension check, but a direct caller could pass any value.
  /// Returning an error instead of slice-indexing past the end keeps
  /// the no-panic contract intact.
  #[error("MixedSinker row index {row} is out of range for configured height {configured_height}")]
  RowIndexOutOfRange {
    /// Row index reported by the offending row.
    row: usize,
    /// Sink's configured height.
    configured_height: usize,
  },

  /// The sinker's configured `width` is odd. 4:2:0 formats
  /// (YUV420p / NV12 / NV21, plus future 4:2:2 variants) subsample
  /// chroma 2:1 in width, and the row primitives (scalar + every
  /// SIMD backend) assume `width & 1 == 0` — calling them with an
  /// odd width panics. `MixedSinker::new` is infallible and accepts
  /// any width, so this error surfaces the misconfiguration at the
  /// first use site ([`PixelSink::begin_frame`] or
  /// [`PixelSink::process`]) before any row primitive is invoked,
  /// preserving the no-panic contract.
  #[error("MixedSinker configured width {width} is odd; 4:2:0 formats require even width")]
  OddWidth {
    /// Sink's configured width.
    width: usize,
  },
}

/// Identifies which of the three HSV planes a
/// [`MixedSinkerError::HsvPlaneTooShort`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HsvPlane {
  /// Hue plane.
  H,
  /// Saturation plane.
  S,
  /// Value plane.
  V,
}

/// Identifies which slice of a multi‑plane source row mismatched in
/// [`MixedSinkerError::RowShapeMismatch`].
///
/// `#[non_exhaustive]` because each new source format the crate grows
/// support for — YUV422p / YUV444p (full‑width chroma), P010 / P016
/// (10/16‑bit planes), etc. — will add its own variant. Pattern
/// matches from downstream code should include a `_ => …` arm.
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash, IsVariant)]
#[non_exhaustive]
pub enum RowSlice {
  /// Y (luma) plane — every 4:2:0 / 4:2:2 / 4:4:4 source.
  #[display("Y")]
  Y,
  /// Half‑width U (Cb) plane in a planar 4:2:0 source ([`Yuv420p`]).
  #[display("U Half")]
  UHalf,
  /// Half‑width V (Cr) plane in a planar 4:2:0 source ([`Yuv420p`]).
  #[display("V Half")]
  VHalf,
  /// Half‑width interleaved UV plane in a semi‑planar 4:2:0 source
  /// ([`Nv12`]). Each row is `U0, V0, U1, V1, …` for `width / 2` pairs.
  #[display("UV Half")]
  UvHalf,
  /// Half‑width interleaved VU plane in a semi‑planar 4:2:0 source
  /// ([`Nv21`]). Each row is `V0, U0, V1, U1, …` for `width / 2`
  /// pairs — byte order swapped relative to [`Self::UvHalf`].
  #[display("VU Half")]
  VuHalf,
  /// Full-width U (Cb) plane in a planar 4:4:4 source
  /// ([`Yuv444p`](crate::yuv::Yuv444p)). `width` bytes per row.
  #[display("U Full")]
  UFull,
  /// Full-width V (Cr) plane in a planar 4:4:4 source
  /// ([`Yuv444p`](crate::yuv::Yuv444p)). `width` bytes per row.
  #[display("V Full")]
  VFull,
  /// Full-width alpha plane in an 8‑bit YUVA source
  /// ([`Yuva420p`](crate::yuv::Yuva420p)). `width` bytes per row
  /// (1:1 with Y).
  #[display("A Full")]
  AFull,
  /// Full-width U row of a **10-bit** 4:4:4 planar source. `u16`
  /// samples, `width` elements.
  #[display("U Full 10")]
  UFull10,
  /// Full-width V row of a **10-bit** 4:4:4 planar source.
  #[display("V Full 10")]
  VFull10,
  /// Full-width alpha row of a **10-bit** 4:4:4 planar source with an
  /// alpha plane ([`Yuva444p10`](crate::yuv::Yuva444p10)). `u16`
  /// samples, `width` elements, low-bit-packed.
  #[display("A Full 10")]
  AFull10,
  /// Full-width U row of a **12-bit** 4:4:4 planar source.
  #[display("U Full 12")]
  UFull12,
  /// Full-width V row of a **12-bit** 4:4:4 planar source.
  #[display("V Full 12")]
  VFull12,
  /// Full-width alpha row of a **12-bit** YUVA planar source
  /// ([`Yuva422p12`](crate::yuv::Yuva422p12) /
  /// [`Yuva444p12`](crate::yuv::Yuva444p12)). `u16` samples, `width`
  /// elements, low-bit-packed.
  #[display("A Full 12")]
  AFull12,
  /// Full-width U row of a **14-bit** 4:4:4 planar source.
  #[display("U Full 14")]
  UFull14,
  /// Full-width V row of a **14-bit** 4:4:4 planar source.
  #[display("V Full 14")]
  VFull14,
  /// Full-width alpha row of a **14-bit** YUVA planar source
  /// ([`Yuva444p14`](crate::yuv::Yuva444p14)). `u16` samples, `width`
  /// elements, low-bit-packed.
  #[display("A Full 14")]
  AFull14,
  /// Full‑width interleaved UV plane in a semi‑planar **4:4:4** source
  /// ([`Nv24`](crate::yuv::Nv24)). Each row is `U0, V0, U1, V1, …` for
  /// `width` pairs (`2 * width` bytes). One UV pair per Y pixel — no
  /// chroma subsampling.
  #[display("UV Full")]
  UvFull,
  /// Full‑width interleaved VU plane in a semi‑planar **4:4:4** source
  /// ([`Nv42`](crate::yuv::Nv42)). Each row is `V0, U0, V1, U1, …` for
  /// `width` pairs — byte order swapped relative to [`Self::UvFull`].
  #[display("VU Full")]
  VuFull,
  /// Full‑width Y row of a **9‑bit** planar source
  /// ([`Yuv420p9`](crate::yuv::Yuv420p9) /
  /// [`Yuv422p9`](crate::yuv::Yuv422p9) /
  /// [`Yuv444p9`](crate::yuv::Yuv444p9)). `u16` samples, `width`
  /// elements (low 9 bits active).
  #[display("Y9")]
  Y9,
  /// Half‑width U row of a **9‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 9")]
  UHalf9,
  /// Half‑width V row of a **9‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 9")]
  VHalf9,
  /// Full‑width U row of a **9‑bit** 4:4:4 planar source.
  #[display("U Full 9")]
  UFull9,
  /// Full‑width V row of a **9‑bit** 4:4:4 planar source.
  #[display("V Full 9")]
  VFull9,
  /// Full-width alpha row of a **9-bit** YUVA planar source
  /// ([`Yuva420p9`](crate::yuv::Yuva420p9)). `u16` samples, `width`
  /// elements, low-bit-packed.
  #[display("A Full 9")]
  AFull9,
  /// Full‑width Y row of a **10‑bit** planar source ([`Yuv420p10`]).
  /// `u16` samples, `width` elements.
  #[display("Y10")]
  Y10,
  /// Half‑width U row of a **10‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 10")]
  UHalf10,
  /// Half‑width V row of a **10‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 10")]
  VHalf10,
  /// Half‑width interleaved UV row of a **10‑bit semi‑planar** source
  /// ([`P010`]). `u16` samples, `width` elements laid out as
  /// `U0, V0, U1, V1, …` (high‑bit‑packed: each element's 10 active
  /// bits sit in the high 10 of its `u16`).
  #[display("UV Half 10")]
  UvHalf10,
  /// Full‑width Y row of a **12‑bit** source — used for both the
  /// planar ([`Yuv420p12`], low‑bit‑packed) and semi‑planar
  /// ([`P012`], high‑bit‑packed) families. `u16` samples, `width`
  /// elements. The packing direction depends on the source format;
  /// the row‑shape check only verifies length, so a single variant
  /// covers both.
  #[display("Y12")]
  Y12,
  /// Half‑width U row of a **12‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 12")]
  UHalf12,
  /// Half‑width V row of a **12‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 12")]
  VHalf12,
  /// Half‑width interleaved UV row of a **12‑bit semi‑planar** source
  /// ([`P012`]). `u16` samples, `width` elements (high‑bit‑packed: 12
  /// active bits in the high 12 of each `u16`).
  #[display("UV Half 12")]
  UvHalf12,
  /// Full‑width Y row of a **14‑bit** planar source ([`Yuv420p14`]).
  /// `u16` samples, `width` elements, low‑bit‑packed.
  #[display("Y14")]
  Y14,
  /// Half‑width U row of a **14‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 14")]
  UHalf14,
  /// Half‑width V row of a **14‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 14")]
  VHalf14,
  /// Full‑width Y row of a **16‑bit** source — used for both the
  /// planar ([`Yuv420p16`](crate::yuv::Yuv420p16)) and semi‑planar
  /// ([`P016`](crate::yuv::P016)) families. At 16 bits there is no
  /// high‑vs‑low packing distinction.
  #[display("Y16")]
  Y16,
  /// Half‑width U row of a **16‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 16")]
  UHalf16,
  /// Half‑width V row of a **16‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 16")]
  VHalf16,
  /// Half‑width interleaved UV row of a **16‑bit semi‑planar** source
  /// ([`P016`](crate::yuv::P016)). `u16` samples, `width` elements.
  #[display("UV Half 16")]
  UvHalf16,
  /// Full-width alpha row of a **16-bit** YUVA planar source
  /// ([`Yuva420p16`](crate::yuv::Yuva420p16) /
  /// [`Yuva444p16`](crate::yuv::Yuva444p16)). `u16` samples,
  /// `width` elements (full u16 range).
  #[display("A Full 16")]
  AFull16,
  /// Full-width U row of a **16-bit** 4:4:4 planar source
  /// ([`Yuv444p16`](crate::yuv::Yuv444p16) /
  /// [`Yuva444p16`](crate::yuv::Yuva444p16)). `u16` samples,
  /// `width` elements (full u16 range).
  #[display("U Full 16")]
  UFull16,
  /// Full-width V row of a **16-bit** 4:4:4 planar source. `u16`
  /// samples, `width` elements (full u16 range).
  #[display("V Full 16")]
  VFull16,
  /// Full‑width interleaved UV row of a **10‑bit semi‑planar 4:4:4**
  /// source ([`P410`](crate::yuv::P410)). `u16` samples, `2 * width`
  /// elements, high‑bit‑packed.
  #[display("UV Full 10")]
  UvFull10,
  /// Full‑width interleaved UV row of a **12‑bit semi‑planar 4:4:4**
  /// source ([`P412`](crate::yuv::P412)). `u16` samples, `2 * width`
  /// elements, high‑bit‑packed.
  #[display("UV Full 12")]
  UvFull12,
  /// Full‑width interleaved UV row of a **16‑bit semi‑planar 4:4:4**
  /// source ([`P416`](crate::yuv::P416)). `u16` samples, `2 * width`
  /// elements (no high/low packing distinction at 16 bits).
  #[display("UV Full 16")]
  UvFull16,
  /// `above` row of an **8-bit Bayer** source
  /// ([`Bayer`](crate::raw::Bayer)). `u8` samples, `width` elements;
  /// supplied by the walker via the **mirror-by-2** boundary
  /// contract — see [`crate::raw::BayerRow::above`] — so at the
  /// top edge this is `mid_row(1)`, not `mid` itself. Replicate
  /// fallback (`above == mid`) only when `height < 2` (no mirror
  /// partner exists).
  #[display("Bayer Above")]
  BayerAbove,
  /// `mid` row of an **8-bit Bayer** source. `u8` samples, `width`
  /// elements — the row currently being produced.
  #[display("Bayer Mid")]
  BayerMid,
  /// `below` row of an **8-bit Bayer** source. `u8` samples, `width`
  /// elements; mirror-by-2 supplies `mid_row(h - 2)` at the bottom
  /// edge — see [`crate::raw::BayerRow::below`]. Replicate fallback
  /// (`below == mid`) only when `height < 2`.
  #[display("Bayer Below")]
  BayerBelow,
  /// `above` row of a **high-bit-depth Bayer** source
  /// ([`Bayer16<BITS>`](crate::raw::Bayer16)). `u16` samples,
  /// `width` elements; mirror-by-2 supplies `mid_row(1)` at the
  /// top edge. Replicate fallback (`above == mid`) only when
  /// `height < 2`.
  #[display("Bayer16 Above")]
  Bayer16Above,
  /// `mid` row of a **high-bit-depth Bayer** source. `u16` samples,
  /// `width` elements.
  #[display("Bayer16 Mid")]
  Bayer16Mid,
  /// `below` row of a **high-bit-depth Bayer** source. `u16`
  /// samples, `width` elements; mirror-by-2 supplies
  /// `mid_row(h - 2)` at the bottom edge. Replicate fallback
  /// (`below == mid`) only when `height < 2`.
  #[display("Bayer16 Below")]
  Bayer16Below,
  /// Packed `R, G, B` row of an [`Rgb24`](crate::yuv::Rgb24) source.
  /// `3 * width` `u8` bytes.
  #[display("RGB packed")]
  RgbPacked,
  /// Packed `B, G, R` row of a [`Bgr24`](crate::yuv::Bgr24) source.
  /// `3 * width` `u8` bytes (channel-order swapped relative to
  /// [`RgbPacked`](Self::RgbPacked)).
  #[display("BGR packed")]
  BgrPacked,
  /// Packed `R, G, B, A` row of an [`Rgba`](crate::yuv::Rgba) source.
  /// `4 * width` `u8` bytes — alpha is real (not padding).
  #[display("RGBA packed")]
  RgbaPacked,
  /// Packed `B, G, R, A` row of a [`Bgra`](crate::yuv::Bgra) source.
  /// `4 * width` `u8` bytes — alpha lane preserved, channel order
  /// swapped on the first three bytes relative to
  /// [`RgbaPacked`](Self::RgbaPacked).
  #[display("BGRA packed")]
  BgraPacked,
  /// Packed `A, R, G, B` row of an [`Argb`](crate::yuv::Argb) source.
  /// `4 * width` `u8` bytes — alpha at the **leading** position vs
  /// [`RgbaPacked`](Self::RgbaPacked).
  #[display("ARGB packed")]
  ArgbPacked,
  /// Packed `A, B, G, R` row of an [`Abgr`](crate::yuv::Abgr) source.
  /// `4 * width` `u8` bytes — leading alpha + reversed RGB order vs
  /// [`ArgbPacked`](Self::ArgbPacked).
  #[display("ABGR packed")]
  AbgrPacked,
  /// Packed `X, R, G, B` row of an [`Xrgb`](crate::yuv::Xrgb) source
  /// (FFmpeg `0rgb`). `4 * width` `u8` bytes — leading **padding**
  /// byte (not alpha).
  #[display("XRGB packed")]
  XrgbPacked,
  /// Packed `R, G, B, X` row of an [`Rgbx`](crate::yuv::Rgbx) source
  /// (FFmpeg `rgb0`). `4 * width` `u8` bytes — trailing padding byte.
  #[display("RGBX packed")]
  RgbxPacked,
  /// Packed `X, B, G, R` row of an [`Xbgr`](crate::yuv::Xbgr) source
  /// (FFmpeg `0bgr`). `4 * width` `u8` bytes — leading padding byte
  /// + reversed RGB order vs [`XrgbPacked`](Self::XrgbPacked).
  #[display("XBGR packed")]
  XbgrPacked,
  /// Packed `B, G, R, X` row of a [`Bgrx`](crate::yuv::Bgrx) source
  /// (FFmpeg `bgr0`). `4 * width` `u8` bytes — trailing padding byte
  /// + reversed RGB order vs [`RgbxPacked`](Self::RgbxPacked).
  #[display("BGRX packed")]
  BgrxPacked,
  /// Packed `X2RGB10` LE row of an
  /// [`X2Rgb10`](crate::yuv::X2Rgb10) source. `4 * width` `u8` bytes
  /// (one little-endian `u32` per pixel with `(MSB) 2X | 10R | 10G |
  /// 10B (LSB)` packing).
  #[display("X2RGB10 packed")]
  X2Rgb10Packed,
  /// Packed `X2BGR10` LE row of an
  /// [`X2Bgr10`](crate::yuv::X2Bgr10) source. `4 * width` `u8` bytes
  /// — channel positions reversed relative to
  /// [`X2Rgb10Packed`](Self::X2Rgb10Packed).
  #[display("X2BGR10 packed")]
  X2Bgr10Packed,
  /// Packed `Y0, U0, Y1, V0, …` row of a
  /// [`Yuyv422`](crate::yuv::Yuyv422) source (FFmpeg `yuyv422` /
  /// YUY2). `2 * width` `u8` bytes — Y in even byte positions, U/V
  /// in odd positions with U preceding V.
  #[display("YUYV422 packed")]
  Yuyv422Packed,
  /// Packed `U0, Y0, V0, Y1, …` row of a
  /// [`Uyvy422`](crate::yuv::Uyvy422) source (FFmpeg `uyvy422` /
  /// UYVY). `2 * width` `u8` bytes — Y in odd byte positions, U/V
  /// in even positions with U preceding V.
  #[display("UYVY422 packed")]
  Uyvy422Packed,
  /// Packed `Y0, V0, Y1, U0, …` row of a
  /// [`Yvyu422`](crate::yuv::Yvyu422) source (FFmpeg `yvyu422` /
  /// YVYU). `2 * width` `u8` bytes — Y in even byte positions, V/U
  /// in odd positions with V preceding U (chroma order swapped vs
  /// [`Yuyv422Packed`](Self::Yuyv422Packed)).
  #[display("YVYU422 packed")]
  Yvyu422Packed,
}

/// A sink that writes any subset of `{RGB, Luma, HSV}` into
/// caller-provided buffers.
///
/// Each output is optional — provide `Some(buffer)` to have that
/// channel written, leave it `None` to skip. Providing no outputs is
/// legal (the kernel still walks the source and calls `process`
/// for each row, but nothing is written).
///
/// When HSV is requested **without** RGB, `MixedSinker` keeps a single
/// row of intermediate RGB in an internal scratch buffer (allocated
/// lazily on first use). If RGB output is also requested, the user's
/// RGB buffer serves as the intermediate for HSV and no scratch is
/// allocated.
///
/// # Type parameter
///
/// `F` identifies the source format — `Yuv420p`, `Nv12`, `Nv21`,
/// `Yuv420p10`, `Yuv420p12`, `Yuv420p14`, `P010`, `P012`, etc. Each
/// format provides its own `impl PixelSink for MixedSinker<'_, F>`.
/// See the module‑level docs for the full list of shipped impls.
pub struct MixedSinker<'a, F: SourceFormat> {
  rgb: Option<&'a mut [u8]>,
  rgb_u16: Option<&'a mut [u16]>,
  rgba: Option<&'a mut [u8]>,
  rgba_u16: Option<&'a mut [u16]>,
  luma: Option<&'a mut [u8]>,
  hsv: Option<HsvBuffers<'a>>,
  width: usize,
  height: usize,
  /// Lazily grown to `3 * width` bytes when HSV is requested without a
  /// user RGB buffer. Empty otherwise.
  rgb_scratch: Vec<u8>,
  /// Whether row primitives dispatch to their SIMD backend. Defaults
  /// to `true`; benchmarks flip this with [`Self::with_simd`] /
  /// [`Self::set_simd`] to A/B test scalar vs SIMD on the same frame.
  simd: bool,
  /// Q8 fixed-point luma coefficients `(cr, cg, cb)` such that
  /// `luma = ((cr * R + cg * G + cb * B + 128) >> 8) as u8`. Only
  /// consulted by source impls that *derive* luma from RGB
  /// (currently the `Bayer` / `Bayer16<BITS>` family — YUV impls
  /// memcpy from the native Y plane and ignore this field).
  /// Default: BT.709 `(54, 183, 19)`.
  luma_coefficients_q8: (u32, u32, u32),
  _fmt: PhantomData<F>,
}

/// Luma coefficient set for sources that derive luma from RGB.
///
/// Only consulted by `MixedSinker` impls whose source is *not* YUV
/// (currently the Bayer / Bayer16 family — YUV impls memcpy from
/// the native Y plane). For Bayer the choice should match the
/// gamut your [`crate::raw::ColorCorrectionMatrix`] targets:
///
/// - CCM target = Rec.709 / sRGB → use [`Self::Bt709`] (the default)
/// - CCM target = Rec.2020 (UHDTV / HDR10) → use [`Self::Bt2020`]
/// - CCM target = DCI-P3 (cinema) → use [`Self::DciP3`]
/// - CCM target = ACEScg / ACES AP1 → use [`Self::AcesAp1`]
/// - CCM target = SDTV (rare for RAW) → use [`Self::Bt601`]
/// - CCM target = something else, or you've measured your own
///   weights → use [`Self::Custom`] (constructed via
///   [`Self::try_custom`] or [`Self::custom`])
///
/// Picking the wrong set still produces a **valid** luma plane,
/// but its numeric values won't match what a downstream
/// luma-driven analysis (scene-cut detection, brightness
/// thresholding, perceptual diff) expects for non-grayscale
/// content. Uniform-gray content is unaffected — every coefficient
/// set agrees on gray.
///
/// Each variant resolves to a Q8 `(cr, cg, cb)` triple summing to
/// `256` so `(cr * R + cg * G + cb * B + 128) >> 8` produces
/// `u8` luma without bias. The triples come from each standard's
/// published coefficients rounded to nearest u32.
#[derive(Debug, Clone, Copy, PartialEq, IsVariant)]
#[non_exhaustive]
pub enum LumaCoefficients {
  /// **BT.709 / sRGB** (`R=0.2126, G=0.7152, B=0.0722`) → Q8
  /// `(54, 183, 19)`. The default; most common output gamut and
  /// the implicit weights every YUV→RGB→luma video pipeline uses.
  Bt709,
  /// **BT.2020 / Rec.2020** (`R=0.2627, G=0.6780, B=0.0593`) → Q8
  /// `(67, 174, 15)`. UHDTV / HDR10 / Rec.2100 (HLG, PQ).
  Bt2020,
  /// **BT.601 / SMPTE 170M** (`R=0.2990, G=0.5870, B=0.1140`) →
  /// Q8 `(77, 150, 29)`. Legacy SDTV / NTSC / PAL. Rare for RAW
  /// pipelines but included for completeness.
  Bt601,
  /// **DCI-P3** (`R=0.228975, G=0.691739, B=0.079287`) → Q8
  /// `(59, 177, 20)`. Theatrical / cinema P3 displays. Note the
  /// **D65 white point** is the same as Rec.709, so for
  /// luma-only purposes this is close to `Bt709` (within ~1 LSB
  /// for most content).
  DciP3,
  /// **ACES AP1 / ACEScg** (`R=0.2722287, G=0.6740818,
  /// B=0.0536895`) → Q8 `(70, 172, 14)`. Cinema grading working
  /// space. Numerically very close to BT.2020. (Naïve nearest
  /// rounding gives `(70, 173, 14)` which sums to 257; the `cg`
  /// term is rounded down by 1 LSB so the triple sums to 256
  /// without biasing the `>> 8` divisor.)
  AcesAp1,
  /// Caller-supplied coefficients. Use [`Self::try_custom`] or
  /// [`Self::custom`] to construct — the inner
  /// [`CustomLumaCoefficients`] keeps fields private so every
  /// `Custom` value is guaranteed finite, non-negative, and
  /// magnitude-bounded.
  Custom(CustomLumaCoefficients),
}

/// Validated red / green / blue luma weights, accessible only through
/// [`LumaCoefficients::Custom`] (or [`Self::try_new`] /
/// [`Self::new`]).
///
/// Each weight is a finite, non-negative `f32` ≤
/// [`Self::MAX_COEFFICIENT`]. The bound is much tighter than
/// [`crate::raw::WhiteBalance::MAX_GAIN`] (`1e6`) because the luma
/// kernel multiplies these into a `u32` accumulator — see
/// [`Self::MAX_COEFFICIENT`] for the overflow analysis.
///
/// The struct intentionally has no public fields. Use
/// [`Self::r`] / [`Self::g`] / [`Self::b`] to read components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CustomLumaCoefficients {
  r: f32,
  g: f32,
  b: f32,
}

impl CustomLumaCoefficients {
  /// Maximum permitted per-channel weight. `10.0` is far above any
  /// realistic published luma coefficient (the standard sets all
  /// individual weights are ≤ `1.0`) and far below the value at
  /// which the per-pixel `u32` accumulator could overflow:
  /// `(coef * 256 + 0.5) as u32 ≤ 10 * 256 + 1 = 2_561`, so the
  /// largest per-row term is `2_561 * 255 = 653_055`, and the
  /// three-channel sum + bias `3 * 653_055 + 128 = 1_959_293` —
  /// six orders of magnitude below `u32::MAX`.
  ///
  /// `1e6` (the
  /// [`crate::raw::WhiteBalance::MAX_GAIN`] bound) **would not be
  /// safe here** — `1e6 * 256 = 256_000_000`, and `256_000_000 *
  /// 255 ≈ 6.5e10` overflows `u32`.
  pub const MAX_COEFFICIENT: f32 = 10.0;

  /// Constructs a [`CustomLumaCoefficients`] from explicit R / G / B
  /// weights, validating that each is **finite, non-negative, and
  /// ≤ [`Self::MAX_COEFFICIENT`]**.
  ///
  /// Returns [`LumaCoefficientsError`] for the first failing
  /// channel. A weight of `0` is permitted (the channel doesn't
  /// contribute to luma — degenerate but well-defined).
  ///
  /// The weights are *not* required to sum to `1.0`; sums far from
  /// `1.0` produce a brightness-scaled luma plane (the doc on
  /// [`LumaCoefficients`] flags this), which is sometimes
  /// intentional (matte / key extraction). Only NaN / ±∞ /
  /// negative / out-of-range weights are rejected because those
  /// would silently corrupt the luma plane via the `f32 → u32`
  /// saturating cast or overflow the accumulator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(r: f32, g: f32, b: f32) -> Result<Self, LumaCoefficientsError> {
    if !r.is_finite() {
      return Err(LumaCoefficientsError::NonFinite {
        channel: LumaChannel::R,
        value: r,
      });
    }
    if !g.is_finite() {
      return Err(LumaCoefficientsError::NonFinite {
        channel: LumaChannel::G,
        value: g,
      });
    }
    if !b.is_finite() {
      return Err(LumaCoefficientsError::NonFinite {
        channel: LumaChannel::B,
        value: b,
      });
    }
    if r < 0.0 {
      return Err(LumaCoefficientsError::Negative {
        channel: LumaChannel::R,
        value: r,
      });
    }
    if g < 0.0 {
      return Err(LumaCoefficientsError::Negative {
        channel: LumaChannel::G,
        value: g,
      });
    }
    if b < 0.0 {
      return Err(LumaCoefficientsError::Negative {
        channel: LumaChannel::B,
        value: b,
      });
    }
    if r > Self::MAX_COEFFICIENT {
      return Err(LumaCoefficientsError::OutOfBounds {
        channel: LumaChannel::R,
        value: r,
        max: Self::MAX_COEFFICIENT,
      });
    }
    if g > Self::MAX_COEFFICIENT {
      return Err(LumaCoefficientsError::OutOfBounds {
        channel: LumaChannel::G,
        value: g,
        max: Self::MAX_COEFFICIENT,
      });
    }
    if b > Self::MAX_COEFFICIENT {
      return Err(LumaCoefficientsError::OutOfBounds {
        channel: LumaChannel::B,
        value: b,
        max: Self::MAX_COEFFICIENT,
      });
    }
    Ok(Self { r, g, b })
  }

  /// Constructs a [`CustomLumaCoefficients`], panicking on invalid
  /// input. Prefer [`Self::try_new`] for caller-supplied values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(r: f32, g: f32, b: f32) -> Self {
    match Self::try_new(r, g, b) {
      Ok(c) => c,
      Err(_) => panic!("invalid CustomLumaCoefficients (non-finite, negative, or out of range)"),
    }
  }

  /// Red weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> f32 {
    self.r
  }

  /// Green weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> f32 {
    self.g
  }

  /// Blue weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> f32 {
    self.b
  }
}

/// Identifies which luma weight failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant)]
#[non_exhaustive]
pub enum LumaChannel {
  /// Red weight.
  R,
  /// Green weight.
  G,
  /// Blue weight.
  B,
}

/// Errors returned by [`CustomLumaCoefficients::try_new`] (and the
/// convenience [`LumaCoefficients::try_custom`]).
#[derive(Debug, Clone, Copy, PartialEq, IsVariant, Error)]
#[non_exhaustive]
pub enum LumaCoefficientsError {
  /// A weight is non-finite (NaN, +∞, or -∞).
  #[error("CustomLumaCoefficients.{channel:?} is non-finite (got {value})")]
  NonFinite {
    /// Which channel failed validation.
    channel: LumaChannel,
    /// The offending weight value.
    value: f32,
  },
  /// A weight is negative. Zero is allowed (zeroes the channel).
  #[error("CustomLumaCoefficients.{channel:?} is negative (got {value})")]
  Negative {
    /// Which channel failed validation.
    channel: LumaChannel,
    /// The offending weight value.
    value: f32,
  },
  /// A weight exceeds [`CustomLumaCoefficients::MAX_COEFFICIENT`]
  /// (`10.0`). The bound is far above any realistic luma weight
  /// but closes the door on values that would saturate the
  /// `f32 → u32` cast in [`LumaCoefficients::to_q8`] or overflow
  /// the per-row `u32` accumulator.
  #[error("CustomLumaCoefficients.{channel:?} = {value} exceeds the magnitude bound ({max})")]
  OutOfBounds {
    /// Which channel failed validation.
    channel: LumaChannel,
    /// The offending weight value.
    value: f32,
    /// The bound that was exceeded
    /// ([`CustomLumaCoefficients::MAX_COEFFICIENT`]).
    max: f32,
  },
}

impl LumaCoefficients {
  /// Resolves the coefficient set to its Q8 fixed-point triple
  /// `(cr, cg, cb)` such that
  /// `luma = ((cr * R + cg * G + cb * B + 128) >> 8) as u8`. The
  /// preset triples come from each standard's published weights
  /// rounded to nearest u32 and (for the published presets) sum
  /// to exactly `256`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn to_q8(self) -> (u32, u32, u32) {
    match self {
      Self::Bt709 => (54, 183, 19),
      Self::Bt2020 => (67, 174, 15),
      Self::Bt601 => (77, 150, 29),
      Self::DciP3 => (59, 177, 20),
      // Naïve nearest rounding gives `(70, 173, 14)` which sums
      // to 257; the `>> 8` divisor implicitly assumes 256, so we
      // shave 1 LSB off `cg` (the largest, smallest-relative-
      // -error coefficient). Resulting (R, G, B) error vs. the
      // published weights is `(+0.0012, -0.0022, +0.0010)`.
      Self::AcesAp1 => (70, 172, 14),
      // Custom values are guaranteed finite + non-negative +
      // ≤ `MAX_COEFFICIENT` (= 10.0) by `CustomLumaCoefficients::
      // try_new`, so the `as u32` cast cannot saturate to
      // `u32::MAX` and the downstream accumulator cannot overflow.
      Self::Custom(c) => (
        (c.r * 256.0 + 0.5) as u32,
        (c.g * 256.0 + 0.5) as u32,
        (c.b * 256.0 + 0.5) as u32,
      ),
    }
  }

  /// Constructs [`Self::Custom`] from explicit R / G / B weights,
  /// validating each via [`CustomLumaCoefficients::try_new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_custom(r: f32, g: f32, b: f32) -> Result<Self, LumaCoefficientsError> {
    match CustomLumaCoefficients::try_new(r, g, b) {
      Ok(c) => Ok(Self::Custom(c)),
      Err(e) => Err(e),
    }
  }

  /// Constructs [`Self::Custom`] from explicit R / G / B weights,
  /// panicking on invalid input. Prefer [`Self::try_custom`] for
  /// caller-supplied values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn custom(r: f32, g: f32, b: f32) -> Self {
    Self::Custom(CustomLumaCoefficients::new(r, g, b))
  }
}

impl Default for LumaCoefficients {
  /// Default is [`Self::Bt709`] — matches the implicit weights
  /// every YUV-source → RGB → luma video pipeline uses.
  fn default() -> Self {
    Self::Bt709
  }
}

impl<F: SourceFormat> MixedSinker<'_, F> {
  /// Creates an empty [`MixedSinker`] for the given output dimensions.
  /// Attach output buffers with `with_rgb` / `with_luma` / `with_hsv`;
  /// each attachment validates that the buffer is at least
  /// `width * height * bytes_per_pixel` so short-buffer bugs surface
  /// *before* any rows are written — not after half the frame has
  /// been mutated.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(width: usize, height: usize) -> Self {
    Self {
      rgb: None,
      rgb_u16: None,
      rgba: None,
      rgba_u16: None,
      luma: None,
      hsv: None,
      width,
      height,
      rgb_scratch: Vec::new(),
      simd: true,
      // BT.709 by default — matches the implicit weights every
      // YUV→RGB→luma pipeline uses, and is the most common Bayer
      // CCM target. Per-format impls (`MixedSinker<Bayer>` etc.)
      // expose `with_luma_coefficients` for callers whose CCM
      // targets a different gamut.
      luma_coefficients_q8: (54, 183, 19),
      _fmt: PhantomData,
    }
  }

  /// Returns `true` iff the sinker will write 8‑bit RGB.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb(&self) -> bool {
    self.rgb.is_some()
  }

  /// Returns `true` iff the sinker will write `u16` RGB at the
  /// source's native bit depth. Only high‑bit‑depth source impls
  /// (currently [`Yuv420p10`](crate::yuv::Yuv420p10)) honor this
  /// buffer — attaching it on an 8‑bit source format is legal but
  /// no writes occur.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb_u16(&self) -> bool {
    self.rgb_u16.is_some()
  }

  /// Returns `true` iff the sinker will write 8‑bit RGBA. The
  /// fourth byte per pixel is alpha — opaque (`0xFF`) by default
  /// when the source has no alpha plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgba(&self) -> bool {
    self.rgba.is_some()
  }

  /// Returns `true` iff the sinker will write `u16` RGBA at the
  /// source's native bit depth. The fourth `u16` per pixel is alpha
  /// — opaque (`(1 << BITS) - 1`) by default when the source has no
  /// alpha plane. Only high‑bit‑depth source impls honor this
  /// buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgba_u16(&self) -> bool {
    self.rgba_u16.is_some()
  }

  /// Returns `true` iff the sinker will write luma.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_luma(&self) -> bool {
    self.luma.is_some()
  }

  /// Returns `true` iff the sinker will write HSV.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_hsv(&self) -> bool {
    self.hsv.is_some()
  }

  /// Frame width in pixels. Declared at construction.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// Frame height in pixels. Declared at construction.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> usize {
    self.height
  }

  /// Returns `true` iff row primitives dispatch to their SIMD backend.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn simd(&self) -> bool {
    self.simd
  }

  /// Toggles the SIMD dispatch in place. See [`Self::with_simd`] for the
  /// consuming builder variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_simd(&mut self, simd: bool) -> &mut Self {
    self.simd = simd;
    self
  }

  /// Sets whether row primitives dispatch to their SIMD backend.
  /// Defaults to `true` — pass `false` to force the scalar reference
  /// path (intended for benchmarks, fuzzing, and differential
  /// testing). See [`Self::set_simd`] for the in‑place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_simd(mut self, simd: bool) -> Self {
    self.set_simd(simd);
    self
  }

  /// Full-frame size in bytes for a given channel count, with
  /// overflow checking. Returns `Err(GeometryOverflow)` if
  /// `width × height × channels` cannot fit in `usize` — only
  /// reachable on 32‑bit targets with extreme dimensions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn frame_bytes(&self, channels: usize) -> Result<usize, MixedSinkerError> {
    self
      .width
      .checked_mul(self.height)
      .and_then(|n| n.checked_mul(channels))
      .ok_or(MixedSinkerError::GeometryOverflow {
        width: self.width,
        height: self.height,
        channels,
      })
  }
}

impl<'a, F: SourceFormat> MixedSinker<'a, F> {
  /// Attaches a packed 24-bit RGB output buffer.
  /// Returns `Err(RgbBufferTooShort)` if `buf.len() < width × height × 3`,
  /// or `Err(GeometryOverflow)` on 32‑bit targets when the product
  /// overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgb(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb`](Self::with_rgb).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb = Some(buf);
    Ok(self)
  }

  // NOTE: `with_rgb_u16` / `set_rgb_u16` are **not** declared here.
  // They live on a format‑specific impl block further down (currently
  // [`MixedSinker<Yuv420p10>`]) so the buffer can only be attached to
  // sink types whose `PixelSink` impl actually writes it. Attaching a
  // `u16` RGB buffer to a [`Yuv420p`] / [`Nv12`] / [`Nv21`] sink is a
  // compile error, not a silent stale‑state bug. Future high‑bit‑depth
  // markers (12‑bit, 14‑bit, P010) will add their own impl blocks.

  // NOTE: `with_rgba` / `set_rgba` are **not** declared here either —
  // same rationale as `with_rgb_u16` above. The Ship 8 RGBA path is
  // currently wired only on [`MixedSinker<Yuv420p>`]; attaching an
  // RGBA buffer to a sink whose `PixelSink::process` doesn't write
  // it would silently leave the caller buffer untouched while
  // `produces_rgba()` returned `true`. Each format that writes RGBA
  // gets its own format‑specific impl block exposing the accessors.
  // Future formats (NV12 / NV21 / Yuv422p / Yuv444p / P010 / etc.)
  // add their own impl blocks as RGBA support lands.

  // NOTE: `with_rgba_u16` / `set_rgba_u16` are **not** declared here
  // for the same reason — they live on the format‑specific impl
  // blocks for high‑bit‑depth sources that actually write
  // native‑depth RGBA.

  /// Attaches a single-plane luma output buffer.
  /// Returns `Err(LumaBufferTooShort)` if `buf.len() < width × height`,
  /// or `Err(GeometryOverflow)` on 32‑bit overflow.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_luma(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_luma`](Self::with_luma).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::LumaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.luma = Some(buf);
    Ok(self)
  }

  /// Attaches three HSV output planes. Returns
  /// `Err(HsvPlaneTooShort { which, .. })` naming the first short
  /// plane, or `Err(GeometryOverflow)` on 32‑bit overflow.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_hsv(
    mut self,
    h: &'a mut [u8],
    s: &'a mut [u8],
    v: &'a mut [u8],
  ) -> Result<Self, MixedSinkerError> {
    self.set_hsv(h, s, v)?;
    Ok(self)
  }

  /// In-place variant of [`with_hsv`](Self::with_hsv).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_hsv(
    &mut self,
    h: &'a mut [u8],
    s: &'a mut [u8],
    v: &'a mut [u8],
  ) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(1)?;
    if h.len() < expected {
      return Err(MixedSinkerError::HsvPlaneTooShort {
        which: HsvPlane::H,
        expected,
        actual: h.len(),
      });
    }
    if s.len() < expected {
      return Err(MixedSinkerError::HsvPlaneTooShort {
        which: HsvPlane::S,
        expected,
        actual: s.len(),
      });
    }
    if v.len() < expected {
      return Err(MixedSinkerError::HsvPlaneTooShort {
        which: HsvPlane::V,
        expected,
        actual: v.len(),
      });
    }
    self.hsv = Some(HsvBuffers { h, s, v });
    Ok(self)
  }
}

/// Returns `Ok(())` iff the walker's frame dimensions exactly match
/// the sinker's configured dimensions. Called from
/// [`PixelSink::begin_frame`] in every `MixedSinker<F>` impl.
///
/// The sinker's RGB / luma / HSV buffers were sized for
/// `configured_w × configured_h`. A shorter frame would silently
/// leave the bottom rows of those buffers stale from the previous
/// frame; a taller frame would overrun them. Either is a real
/// failure mode, but neither is a panic-worthy bug — the caller can
/// recover by rebuilding the sinker. Returning `Err` before any row
/// is processed guarantees no partial output.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn check_dimensions_match(
  configured_w: usize,
  configured_h: usize,
  frame_w: u32,
  frame_h: u32,
) -> Result<(), MixedSinkerError> {
  let fw = frame_w as usize;
  let fh = frame_h as usize;
  if fw != configured_w || fh != configured_h {
    return Err(MixedSinkerError::DimensionMismatch {
      configured_w,
      configured_h,
      frame_w,
      frame_h,
    });
  }
  Ok(())
}

/// Slice the RGBA row out of an attached RGBA plane buffer. Returns
/// `Err(GeometryOverflow)` if `one_plane_end × 4` wraps `usize` (only
/// reachable on 32-bit targets at extreme dimensions).
///
/// Centralises the duplicated overflow/bounds-check pattern that every
/// `MixedSinker<F>::process` impl runs in both the standalone-RGBA
/// branch and the Strategy-A expand branch.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgba_plane_row_slice(
  buf: &mut [u8],
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&mut [u8], MixedSinkerError> {
  let end = one_plane_end
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow {
      width,
      height,
      channels: 4,
    })?;
  let start = one_plane_start * 4; // ≤ end, fits.
  Ok(&mut buf[start..end])
}

/// `u16` analogue of [`rgba_plane_row_slice`] — slices the RGBA row out
/// of an attached `u16` RGBA plane buffer. This helper indexes in `u16`
/// elements, not bytes: like the `u8` variant, RGBA rows use `× 4`
/// elements per pixel, so the overflow check is the same, but the byte
/// offsets differ because each element is 2 bytes. Used by the
/// high-bit-depth 4:2:0 sinkers that fan `u16` RGB out to `u16` RGBA.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgba_u16_plane_row_slice(
  buf: &mut [u16],
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&mut [u16], MixedSinkerError> {
  let end = one_plane_end
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow {
      width,
      height,
      channels: 4,
    })?;
  let start = one_plane_start * 4; // ≤ end, fits.
  Ok(&mut buf[start..end])
}

/// Pick an RGB row buffer for the kernel to write into: caller's RGB
/// plane slice when attached, or the growing scratch buffer otherwise
/// (HSV-only callers don't allocate an RGB plane). Returns
/// `Err(GeometryOverflow)` if `width × 3` or `one_plane_end × 3` wraps
/// `usize` — see [`rgba_plane_row_slice`] for the rationale.
///
/// `rgb_scratch` is grown via `Vec::resize` only when too small; the
/// caller keeps the existing capacity across rows so steady-state
/// processing allocates zero times.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgb_row_buf_or_scratch<'a>(
  rgb: Option<&'a mut [u8]>,
  rgb_scratch: &'a mut Vec<u8>,
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&'a mut [u8], MixedSinkerError> {
  match rgb {
    Some(buf) => {
      let end = one_plane_end
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width,
          height,
          channels: 3,
        })?;
      let start = one_plane_start * 3;
      Ok(&mut buf[start..end])
    }
    None => {
      let row_bytes = width
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width,
          height,
          channels: 3,
        })?;
      if rgb_scratch.len() < row_bytes {
        rgb_scratch.resize(row_bytes, 0);
      }
      Ok(&mut rgb_scratch[..row_bytes])
    }
  }
}

/// Configurable-coefficients luma derivation from packed
/// `R, G, B` u8 row.
///
/// Q8 fixed-point: `Y ≈ (cr·R + cg·G + cb·B + 128) >> 8`, where
/// `(cr, cg, cb)` is the caller's [`LumaCoefficients`] resolved
/// via [`LumaCoefficients::to_q8`]. The presets all sum to `256`
/// so the divisor is implicit in the `>> 8`. `rgb` carries
/// `3 * luma.len()` packed bytes; the loop writes one luma
/// sample per pixel.
///
/// Used by Bayer / Bayer16 [`MixedSinker`] paths whose source has
/// no native luma plane to memcpy from. YUV source impls take
/// their luma directly off the Y plane and don't go through this
/// helper, so they don't need a configurable coefficient set —
/// the source's `ColorMatrix` already fixed it at encode time.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgb_row_to_luma_row(rgb: &[u8], luma: &mut [u8], coeffs_q8: (u32, u32, u32)) {
  // Caller's contract: `rgb` packs `3 * luma.len()` bytes. The
  // current callers (`MixedSinker<Bayer>` and
  // `MixedSinker<Bayer16<BITS>>`) both slice their `luma` and
  // `rgb_row` from the same `width`, so the relationship holds
  // structurally — but the `debug_assert` makes that obvious to
  // any future caller and turns silent OOB indexing into a clear
  // failure under tests.
  //
  // `checked_mul` instead of `3 * luma.len()` because, while the
  // existing `frame_bytes` validation in caller paths makes the
  // product fit, a future caller passing a raw slice with no such
  // upstream check could trigger a `usize` overflow inside the
  // assert message itself (panic before the assertion runs).
  // Failing the assert on overflow yields a clean diagnostic.
  debug_assert!(
    luma
      .len()
      .checked_mul(3)
      .is_some_and(|need| rgb.len() >= need),
    "rgb_row_to_luma_row: rgb.len()={} but need {} (= 3 × luma.len()={})",
    rgb.len(),
    luma.len().saturating_mul(3),
    luma.len(),
  );
  let (cr, cg, cb) = coeffs_q8;
  for (i, d) in luma.iter_mut().enumerate() {
    let r = rgb[3 * i] as u32;
    let g = rgb[3 * i + 1] as u32;
    let b = rgb[3 * i + 2] as u32;
    *d = ((cr * r + cg * g + cb * b + 128) >> 8).min(255) as u8;
  }
}

// ---- Format-specific impl blocks (split out of mod.rs) ------------------
//
// Each child module hosts the `MixedSinker<'_, F>` impl blocks for a
// related family of source formats. mod.rs keeps only the shared
// prelude (errors, types, struct, generic impls, helpers) and the
// `LumaCoefficients` API. Per-format `with_rgba` / `set_rgba` builders
// and `PixelSink` impls live in the child modules below.

mod bayer;
mod packed_rgb_10bit;
mod packed_rgb_8bit;
mod packed_yuv_8bit;
mod planar_8bit;
mod semi_planar_8bit;
mod subsampled_4_2_0_high_bit;
mod subsampled_4_2_2_high_bit;
mod subsampled_4_4_4_high_bit;
mod yuva_4_2_0;
mod yuva_4_2_2;
mod yuva_4_4_4;

#[cfg(all(test, feature = "std"))]
mod tests;
