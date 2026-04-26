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

use crate::{
  HsvBuffers, PixelSink, SourceFormat,
  raw::{Bayer, Bayer16, BayerRow, BayerRow16, BayerSink, BayerSink16},
  row::{
    bayer_to_rgb_row, bayer16_to_rgb_row, bayer16_to_rgb_u16_row, expand_rgb_to_rgba_row,
    nv12_to_rgb_row, nv12_to_rgba_row, nv21_to_rgb_row, nv21_to_rgba_row, nv24_to_rgb_row,
    nv24_to_rgba_row, nv42_to_rgb_row, nv42_to_rgba_row, p010_to_rgb_row, p010_to_rgb_u16_row,
    p012_to_rgb_row, p012_to_rgb_u16_row, p016_to_rgb_row, p016_to_rgb_u16_row, p410_to_rgb_row,
    p410_to_rgb_u16_row, p412_to_rgb_row, p412_to_rgb_u16_row, p416_to_rgb_row,
    p416_to_rgb_u16_row, rgb_to_hsv_row, yuv_420_to_rgb_row, yuv_420_to_rgba_row,
    yuv_444_to_rgb_row, yuv_444_to_rgba_row, yuv420p9_to_rgb_row, yuv420p9_to_rgb_u16_row,
    yuv420p10_to_rgb_row, yuv420p10_to_rgb_u16_row, yuv420p12_to_rgb_row, yuv420p12_to_rgb_u16_row,
    yuv420p14_to_rgb_row, yuv420p14_to_rgb_u16_row, yuv420p16_to_rgb_row, yuv420p16_to_rgb_u16_row,
    yuv444p9_to_rgb_row, yuv444p9_to_rgb_u16_row, yuv444p10_to_rgb_row, yuv444p10_to_rgb_u16_row,
    yuv444p12_to_rgb_row, yuv444p12_to_rgb_u16_row, yuv444p14_to_rgb_row, yuv444p14_to_rgb_u16_row,
    yuv444p16_to_rgb_row, yuv444p16_to_rgb_u16_row,
  },
  yuv::{
    Nv12, Nv12Row, Nv12Sink, Nv16, Nv16Row, Nv16Sink, Nv21, Nv21Row, Nv21Sink, Nv24, Nv24Row,
    Nv24Sink, Nv42, Nv42Row, Nv42Sink, P010, P010Row, P010Sink, P012, P012Row, P012Sink, P016,
    P016Row, P016Sink, P210, P210Row, P210Sink, P212, P212Row, P212Sink, P216, P216Row, P216Sink,
    P410, P410Row, P410Sink, P412, P412Row, P412Sink, P416, P416Row, P416Sink, Yuv420p, Yuv420p9,
    Yuv420p9Row, Yuv420p9Sink, Yuv420p10, Yuv420p10Row, Yuv420p10Sink, Yuv420p12, Yuv420p12Row,
    Yuv420p12Sink, Yuv420p14, Yuv420p14Row, Yuv420p14Sink, Yuv420p16, Yuv420p16Row, Yuv420p16Sink,
    Yuv420pRow, Yuv420pSink, Yuv422p, Yuv422p9, Yuv422p9Row, Yuv422p9Sink, Yuv422p10, Yuv422p10Row,
    Yuv422p10Sink, Yuv422p12, Yuv422p12Row, Yuv422p12Sink, Yuv422p14, Yuv422p14Row, Yuv422p14Sink,
    Yuv422p16, Yuv422p16Row, Yuv422p16Sink, Yuv422pRow, Yuv422pSink, Yuv440p, Yuv440p10,
    Yuv440p10Row, Yuv440p10Sink, Yuv440p12, Yuv440p12Row, Yuv440p12Sink, Yuv440pRow, Yuv440pSink,
    Yuv444p, Yuv444p9, Yuv444p9Row, Yuv444p9Sink, Yuv444p10, Yuv444p10Row, Yuv444p10Sink,
    Yuv444p12, Yuv444p12Row, Yuv444p12Sink, Yuv444p14, Yuv444p14Row, Yuv444p14Sink, Yuv444p16,
    Yuv444p16Row, Yuv444p16Sink, Yuv444pRow, Yuv444pSink,
  },
};

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
  /// Full-width U row of a **10-bit** 4:4:4 planar source. `u16`
  /// samples, `width` elements.
  #[display("U Full 10")]
  UFull10,
  /// Full-width V row of a **10-bit** 4:4:4 planar source.
  #[display("V Full 10")]
  VFull10,
  /// Full-width U row of a **12-bit** 4:4:4 planar source.
  #[display("U Full 12")]
  UFull12,
  /// Full-width V row of a **12-bit** 4:4:4 planar source.
  #[display("V Full 12")]
  VFull12,
  /// Full-width U row of a **14-bit** 4:4:4 planar source.
  #[display("U Full 14")]
  UFull14,
  /// Full-width V row of a **14-bit** 4:4:4 planar source.
  #[display("V Full 14")]
  VFull14,
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

// ---- Yuv420p impl --------------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — calling `with_rgba` on a sink that doesn't (e.g.
  /// [`MixedSinker<Nv12>`] today) is a compile error rather than a
  /// silent no‑op that would leave the caller's buffer stale while
  /// [`Self::produces_rgba`] returned `true`. The compile-time
  /// scoping is load-bearing: if a future format adds RGBA, it must
  /// add its own impl block here, which both wires the new path and
  /// prevents accidental cross-format leakage.
  ///
  /// The fourth byte per pixel is alpha. [`Yuv420p`] has no alpha
  /// plane, so every alpha byte is filled with `0xFF` (opaque).
  /// Future YUVA source impls will copy alpha through from the
  /// source plane.
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ```compile_fail
  /// // Attaching RGBA to a sink that doesn't write it is rejected
  /// // at compile time. Yuv440p (4:4:0 planar) has not yet been
  /// // wired for RGBA; once that lands the negative example here
  /// // moves to the next not‑yet‑wired format.
  /// use colconv::{sinker::MixedSinker, yuv::Yuv440p};
  /// let mut buf = vec![0u8; 16 * 8 * 4];
  /// let _ = MixedSinker::<Yuv440p>::new(16, 8).with_rgba(&mut buf);
  /// ```
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl PixelSink for MixedSinker<'_, Yuv420p> {
  type Input<'r> = Yuv420pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Reject odd-width sinkers up front — the underlying row
    // primitives assume `width & 1 == 0` and would panic on the
    // first `process` call otherwise (`MixedSinker::new` is
    // infallible and accepts any width).
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth: `begin_frame` already validated frame‑level
    // dimensions, so these checks are unreachable from the walker.
    // They guard direct `process` callers (hand-crafted rows, row
    // replay) from handing a wrong-shaped row or out-of-range index
    // to unsafe SIMD kernels. Report the offending slice length and
    // row index directly — don't reuse `DimensionMismatch`, whose
    // `frame_w` / `frame_h` fields would be meaningless here.
    //
    // Odd-width check first: the row primitives assume
    // `width & 1 == 0` and would panic past this point. Keeping the
    // check here (and in `begin_frame`) preserves the no-panic
    // contract for direct `process` callers that skip `begin_frame`.
    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    // Split-borrow so the `rgb_scratch` path and the `hsv` write don't
    // collide with the `rgb` read-after-write chain below.
    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    // Single-plane row ranges are guaranteed not to overflow: `idx <
    // h` and `with_luma` / `with_hsv` validated `w × h × 1` fits
    // usize, so `(idx + 1) * w ≤ h * w` fits too. The `× 3` RGB
    // ranges are only needed when RGB output is requested — computed
    // lazily below with overflow checking.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma — YUV420p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Output mode resolution (Strategy A):
    // - RGBA-only: run dedicated `yuv_420_to_rgba_row` (4 bpp store).
    // - RGB / HSV (with or without RGBA): run RGB kernel once, then if
    //   RGBA is also requested, fan it out via `expand_rgb_to_rgba_row`
    //   (memory-bound copy + 0xFF alpha pad). Saves the second YUV→RGB
    //   per-pixel math when both buffers are attached.
    // - None of the above: nothing to do beyond luma above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      yuv_420_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut rgba_buf[rgba_plane_start..rgba_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    // Pick where the RGB row lands. If the caller wants RGB in their
    // own buffer, write directly there; otherwise use the scratch.
    // Either way, the slice we hold is `&mut [u8]` that we then
    // reborrow as `&[u8]` for the HSV step.
    //
    // RGB byte ranges use `checked_mul` because `w × 3` (and
    // `(idx + 1) × w × 3`) can wrap 32-bit `usize` for large widths
    // even when the single-plane ranges fit — a caller can attach
    // only `with_hsv` (which validates `w × h × 1`) and never go
    // through the `× 3` check at buffer attachment. Overflow here
    // returns `GeometryOverflow` instead of panicking inside the row
    // dispatcher's own checked multiplication.
    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3; // ≤ rgb_plane_end, fits.
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    // Fused YUV→RGB: upsample chroma in registers inside the row
    // primitive, no intermediate memory.
    yuv_420_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    // HSV from the RGB row we just wrote.
    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      expand_rgb_to_rgba_row(rgb_row, &mut buf[rgba_plane_start..rgba_plane_end], w);
    }

    Ok(())
  }
}

// ---- Yuv422p impl -------------------------------------------------------
//
// 4:2:2 is 4:2:0's vertical-axis twin: same per-row chroma shape
// (half-width U / V, one pair per Y pair), just one chroma row per Y
// row instead of one per two. This impl reuses `yuv_420_to_rgb_row`
// (and `yuv_420_to_rgba_row` for the RGBA path) — no new kernels
// needed.

impl<'a> MixedSinker<'a, Yuv422p> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Yuv422p has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Yuv422pSink for MixedSinker<'_, Yuv422p> {}

impl PixelSink for MixedSinker<'_, Yuv422p> {
  type Input<'r> = Yuv422pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv422pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    // Reuses Yuv420p dispatchers (RGB and RGBA) since 4:2:2's per-row
    // contract is identical (half-width chroma, one pair per Y pair).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      yuv_420_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut rgba_buf[rgba_plane_start..rgba_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    // Reuses the Yuv420p dispatcher — 4:2:2's per-row contract is
    // identical (half-width chroma, one pair per Y pair).
    yuv_420_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      expand_rgb_to_rgba_row(rgb_row, &mut buf[rgba_plane_start..rgba_plane_end], w);
    }

    Ok(())
  }
}

// ---- Yuv444p impl -------------------------------------------------------
//
// 4:4:4 planar: U and V are full-width, full-height. No width parity
// constraint. Uses the `yuv_444_to_rgb_row` / `yuv_444_to_rgba_row`
// kernel family.

impl<'a> MixedSinker<'a, Yuv444p> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Yuv444p has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Yuv444pSink for MixedSinker<'_, Yuv444p> {}

impl PixelSink for MixedSinker<'_, Yuv444p> {
  type Input<'r> = Yuv444pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv444pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      yuv_444_to_rgba_row(
        row.y(),
        row.u(),
        row.v(),
        &mut rgba_buf[rgba_plane_start..rgba_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv_444_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      expand_rgb_to_rgba_row(rgb_row, &mut buf[rgba_plane_start..rgba_plane_end], w);
    }

    Ok(())
  }
}

// ---- Nv12 impl ----------------------------------------------------------

impl<'a> MixedSinker<'a, Nv12> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — calling `with_rgba` on a sink that doesn't (e.g. a
  /// not‑yet‑wired `MixedSinker<Nv16>` today) is a compile error
  /// rather than a silent no‑op. Each format that adds RGBA support
  /// adds its own impl block here.
  ///
  /// The fourth byte per pixel is alpha. NV12 has no alpha plane,
  /// so every alpha byte is filled with `0xFF` (opaque). Future
  /// YUVA source impls will copy alpha through from the source
  /// plane.
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Nv12Sink for MixedSinker<'_, Nv12> {}

impl PixelSink for MixedSinker<'_, Nv12> {
  type Input<'r> = Nv12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Reject odd-width sinkers up front — the underlying row
    // primitives assume `width & 1 == 0` and would panic on the
    // first `process` call otherwise (`MixedSinker::new` is
    // infallible and accepts any width).
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Nv12Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth shape check (see Yuv420p impl above). An NV12
    // UV row is `width` bytes of interleaved U / V payload — same
    // length as Y — so both slices must equal `self.width`. Odd-width
    // check comes first since the row primitive would panic on it.
    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    // Single-plane row ranges are guaranteed to fit; RGB / RGBA
    // ranges use checked arithmetic (see the Yuv420p impl above for
    // the full rationale — hsv-only attachment never validated × 3).
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma — NV12 luma is the Y plane. Copy verbatim.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      nv12_to_rgba_row(
        row.y(),
        row.uv_half(),
        &mut rgba_buf[rgba_plane_start..rgba_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    // Fused NV12 → RGB: UV deinterleave + chroma upsample both happen
    // in registers inside the row primitive, no intermediate memory.
    nv12_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      expand_rgb_to_rgba_row(rgb_row, &mut buf[rgba_plane_start..rgba_plane_end], w);
    }

    Ok(())
  }
}

impl Yuv420pSink for MixedSinker<'_, Yuv420p> {}

// ---- Nv16 impl ----------------------------------------------------------
//
// 4:2:2 is 4:2:0's vertical‑axis twin: one UV row per Y row instead of
// one per two. Per‑row math is identical, so this impl calls the same
// `nv12_to_rgb_row` / `nv12_to_rgba_row` dispatchers — no new kernels
// needed.

impl<'a> MixedSinker<'a, Nv16> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. NV16 has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Nv16Sink for MixedSinker<'_, Nv16> {}

impl PixelSink for MixedSinker<'_, Nv16> {
  type Input<'r> = Nv16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Nv16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    // NV16 UV row is `width` bytes of interleaved U/V — identical shape
    // to NV12's `uv_half`.
    if row.uv().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf,
        row: idx,
        expected: w,
        actual: row.uv().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    // Reuses NV12 dispatchers (RGB and RGBA) since 4:2:2's row
    // contract is identical.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      nv12_to_rgba_row(
        row.y(),
        row.uv(),
        &mut rgba_buf[rgba_plane_start..rgba_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    // Reuses the NV12 dispatcher — 4:2:2's row contract is identical.
    nv12_to_rgb_row(
      row.y(),
      row.uv(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      expand_rgb_to_rgba_row(rgb_row, &mut buf[rgba_plane_start..rgba_plane_end], w);
    }

    Ok(())
  }
}

// ---- Nv21 impl ----------------------------------------------------------
//
// Structurally identical to the Nv12 impl — the row primitives hide
// the U/V byte-order difference. Only the trait `Input<'r>` and the
// primitive name change.

impl<'a> MixedSinker<'a, Nv21> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Nv12>::with_rgba`] for the same
  /// rationale and constraints. NV21 has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Nv21Sink for MixedSinker<'_, Nv21> {}

impl PixelSink for MixedSinker<'_, Nv21> {
  type Input<'r> = Nv21Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Nv21Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth: same shape check as the Nv12 impl. A VU row
    // has `width` bytes of interleaved V / U payload — same length
    // as Y — so both slices must equal `self.width`.
    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.vu_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VuHalf,
        row: idx,
        expected: w,
        actual: row.vu_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      nv21_to_rgba_row(
        row.y(),
        row.vu_half(),
        &mut rgba_buf[rgba_plane_start..rgba_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    // Fused NV21 → RGB: VU deinterleave + chroma upsample both happen
    // in registers inside the row primitive, no intermediate memory.
    nv21_to_rgb_row(
      row.y(),
      row.vu_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      expand_rgb_to_rgba_row(rgb_row, &mut buf[rgba_plane_start..rgba_plane_end], w);
    }

    Ok(())
  }
}

// ---- Nv24 impl ----------------------------------------------------------
//
// 4:4:4 semi-planar: UV plane is full-width (`2 * width` bytes per
// row), one UV pair per Y pixel. No width parity constraint. Kernel
// is its own family (`nv24_to_rgb_row`) since chroma is no longer
// duplicated across columns.

impl<'a> MixedSinker<'a, Nv24> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Nv24 has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Nv24Sink for MixedSinker<'_, Nv24> {}

impl PixelSink for MixedSinker<'_, Nv24> {
  type Input<'r> = Nv24Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Nv24Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    // NV24 UV row is `2 * width` bytes. `checked_mul` covers the
    // boundary where `2 * width` could overflow `usize` on 32-bit
    // targets with very large widths.
    let uv_expected = w.checked_mul(2).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 2,
    })?;
    if row.uv().len() != uv_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvFull,
        row: idx,
        expected: uv_expected,
        actual: row.uv().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Standalone RGBA path: the caller wants only RGBA (no RGB / HSV),
    // so run the dedicated RGBA kernel directly into the output buffer.
    // Avoids both the scratch allocation and the expand-pad pass.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      nv24_to_rgba_row(
        row.y(),
        row.uv(),
        &mut rgba_buf[rgba_plane_start..rgba_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    nv24_to_rgb_row(
      row.y(),
      row.uv(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A: when both RGB-side and RGBA outputs are requested,
    // derive RGBA from the just-computed RGB row (memory-bound copy +
    // 0xFF alpha pad) instead of running a second YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      expand_rgb_to_rgba_row(rgb_row, &mut buf[rgba_plane_start..rgba_plane_end], w);
    }

    Ok(())
  }
}

// ---- Nv42 impl ----------------------------------------------------------
//
// Structurally identical to the Nv24 impl — the row primitive hides
// the V/U byte-order difference.

impl<'a> MixedSinker<'a, Nv42> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Nv24>::with_rgba`] for the same rationale and
  /// constraints; Nv42 differs only in chroma byte order (V before U).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Nv42Sink for MixedSinker<'_, Nv42> {}

impl PixelSink for MixedSinker<'_, Nv42> {
  type Input<'r> = Nv42Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Nv42Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    let vu_expected = w.checked_mul(2).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 2,
    })?;
    if row.vu().len() != vu_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VuFull,
        row: idx,
        expected: vu_expected,
        actual: row.vu().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      nv42_to_rgba_row(
        row.y(),
        row.vu(),
        &mut rgba_buf[rgba_plane_start..rgba_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    nv42_to_rgb_row(
      row.y(),
      row.vu(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_plane_end =
        one_plane_end
          .checked_mul(4)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 4,
          })?;
      let rgba_plane_start = one_plane_start * 4;
      expand_rgb_to_rgba_row(rgb_row, &mut buf[rgba_plane_start..rgba_plane_end], w);
    }

    Ok(())
  }
}

// ---- Yuv420p10 impl -----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p10> {
  /// Attaches a packed **`u16`** RGB output buffer. Only available on
  /// sinkers whose source format populates native‑depth `u16` RGB —
  /// calling `with_rgb_u16` on an 8‑bit source sinker (e.g.
  /// [`MixedSinker<Yuv420p>`]) is a compile error rather than a
  /// silent no‑op that would leave the caller's buffer stale.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width × height × 3`. Each element carries a 10‑bit value in
  /// the **low** 10 bits (upper 6 bits zero), matching FFmpeg's
  /// `yuv420p10le` convention. This is **not** the `p010` layout
  /// (which stores samples in the high 10 bits); callers feeding a
  /// p010 consumer must shift the output left by 6.
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if
  /// `buf.len() < width × height × 3`, or `Err(GeometryOverflow)`
  /// on 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16). The
  /// required length is measured in `u16` **elements**, not bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    // Packed RGB requires `width × height × 3` channel values —
    // that's the same count whether the element type is `u8` or
    // `u16`, so the [`Self::frame_bytes`] helper (named for the u8
    // RGB path's byte count) gives the element count here too. No
    // size conversion needed.
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv420p10Sink for MixedSinker<'_, Yuv420p10> {}

impl PixelSink for MixedSinker<'_, Yuv420p10> {
  type Input<'r> = Yuv420p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p10Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (10) — declared as a const so
    // the downshift for u8 luma stays obvious at the call site.
    const BITS: u32 = 10;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth — see the [`Yuv420p`] impl for the rationale.
    // Row slice checks use the 10‑bit variants of [`RowSlice`] so
    // downstream log output disambiguates from the 8‑bit source impls.
    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf10,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf10,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma: downshift 10‑bit Y to 8‑bit for the existing u8 luma
    // buffer contract. Bit‑extension by `(BITS - 8)` preserves the
    // most significant bits — functionally equivalent to FFmpeg's
    // `>> (BITS - 8)` conversion used by many downstream analyses.
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    // `u16` RGB output — written directly via the native‑depth row
    // primitive. Computed independently of the u8 path: the two
    // outputs have different scale params inside `range_params_n`,
    // so they can't share an intermediate without losing precision.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p10_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return Ok(());
    }

    // 8‑bit RGB path — either writes to the caller's buffer (when
    // `with_rgb` is set) or to the lazily‑grown scratch (when HSV is
    // requested without RGB). Mirrors the 8‑bit source impls' layout.
    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p10_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P010 impl ---------------------------------------------------------

impl<'a> MixedSinker<'a, P010> {
  /// Attaches a packed **`u16`** RGB output buffer. Mirrors
  /// [`MixedSinker<Yuv420p10>::with_rgb_u16`] — compile‑time gated to
  /// sinkers whose source format populates native‑depth RGB.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width × height × 3`. Output is **low‑bit‑packed** (10‑bit
  /// values in the low 10 of each `u16`, upper 6 zero) — matches
  /// FFmpeg `yuv420p10le` convention. This is **not** P010 packing
  /// (which puts the 10 bits in the high 10); callers feeding a P010
  /// consumer must shift the output left by 6.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16). The
  /// required length is measured in `u16` **elements**, not bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P010Sink for MixedSinker<'_, P010> {}

impl PixelSink for MixedSinker<'_, P010> {
  type Input<'r> = P010Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P010Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    // Semi-planar UV: `width` u16 elements total (`width / 2` pairs).
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf10,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma: P010 samples are high-bit-packed (`value << 6`). Taking
    // the high byte via `>> 8` gives the top 8 bits of the 10-bit
    // value — functionally equivalent to
    // `(value >> 2)` for the yuv420p10 path.
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    // `u16` RGB output — low-bit-packed 10-bit values (yuv420p10le
    // convention), not P010's high-bit packing.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p010_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p010_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv420p12 impl ----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p12> {
  /// Attaches a packed **`u16`** RGB output buffer. Mirrors
  /// [`MixedSinker<Yuv420p10>::with_rgb_u16`] but produces 12‑bit
  /// output (values in `[0, 4095]` in the low 12 of each `u16`, upper
  /// 4 zero). Length is measured in `u16` **elements** (`width ×
  /// height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv420p12Sink for MixedSinker<'_, Yuv420p12> {}

impl PixelSink for MixedSinker<'_, Yuv420p12> {
  type Input<'r> = Yuv420p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p12Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (12) — declared as a const so
    // the downshift for u8 luma stays obvious at the call site.
    const BITS: u32 = 12;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y12,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf12,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf12,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p12_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p12_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv420p14 impl ----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p14> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 14‑bit
  /// output (values in `[0, 16383]` in the low 14 of each `u16`, upper
  /// 2 zero). Length is measured in `u16` **elements** (`width ×
  /// height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv420p14Sink for MixedSinker<'_, Yuv420p14> {}

impl PixelSink for MixedSinker<'_, Yuv420p14> {
  type Input<'r> = Yuv420p14Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p14Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 14;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y14,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf14,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf14,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p14_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p14_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv422p10 / 12 / 14 impl ------------------------------------------
//
// 4:2:2 is 4:2:0's vertical-axis twin at each bit depth: same per-row
// chroma shape (half-width U / V samples, one pair per Y pair), just
// one chroma row per Y row instead of one per two. These impls reuse
// `yuv420p10_to_rgb_*` / `yuv420p12_to_rgb_*` / `yuv420p14_to_rgb_*`
// verbatim — no new row kernels.

impl<'a> MixedSinker<'a, Yuv422p10> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit low-packed
  /// values (`(1 << 10) - 1 = 1023` max).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv422p10Sink for MixedSinker<'_, Yuv422p10> {}

impl PixelSink for MixedSinker<'_, Yuv422p10> {
  type Input<'r> = Yuv422p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv422p10Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf10,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf10,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p10_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p10_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

impl<'a> MixedSinker<'a, Yuv422p12> {
  /// Attaches a packed **`u16`** RGB output buffer. 12-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv422p12Sink for MixedSinker<'_, Yuv422p12> {}

impl PixelSink for MixedSinker<'_, Yuv422p12> {
  type Input<'r> = Yuv422p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv422p12Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y12,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf12,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf12,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p12_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p12_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

impl<'a> MixedSinker<'a, Yuv422p14> {
  /// Attaches a packed **`u16`** RGB output buffer. 14-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv422p14Sink for MixedSinker<'_, Yuv422p14> {}

impl PixelSink for MixedSinker<'_, Yuv422p14> {
  type Input<'r> = Yuv422p14Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv422p14Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 14;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y14,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf14,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf14,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p14_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p14_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv444p10 / 12 / 14 impl ------------------------------------------

impl<'a> MixedSinker<'a, Yuv444p10> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv444p10Sink for MixedSinker<'_, Yuv444p10> {}

impl PixelSink for MixedSinker<'_, Yuv444p10> {
  type Input<'r> = Yuv444p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv444p10Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull10,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull10,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv444p10_to_rgb_u16_row(
        row.y(),
        row.u(),
        row.v(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv444p10_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

impl<'a> MixedSinker<'a, Yuv444p12> {
  /// Attaches a packed **`u16`** RGB output buffer. 12-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv444p12Sink for MixedSinker<'_, Yuv444p12> {}

impl PixelSink for MixedSinker<'_, Yuv444p12> {
  type Input<'r> = Yuv444p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv444p12Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y12,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull12,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull12,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv444p12_to_rgb_u16_row(
        row.y(),
        row.u(),
        row.v(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv444p12_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

impl<'a> MixedSinker<'a, Yuv444p14> {
  /// Attaches a packed **`u16`** RGB output buffer. 14-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv444p14Sink for MixedSinker<'_, Yuv444p14> {}

impl PixelSink for MixedSinker<'_, Yuv444p14> {
  type Input<'r> = Yuv444p14Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv444p14Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 14;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y14,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull14,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull14,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv444p14_to_rgb_u16_row(
        row.y(),
        row.u(),
        row.v(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv444p14_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv422p16 / Yuv444p16 impl ----------------------------------------
//
// 16-bit family. Yuv422p16 reuses the 4:2:0 16-bit kernel family
// (identical per-row shape); Yuv444p16 has its own kernels.

impl<'a> MixedSinker<'a, Yuv422p16> {
  /// Attaches a packed **`u16`** RGB output buffer. Output covers
  /// full `u16` range `[0, 65535]` (16 active bits, no packing).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv422p16Sink for MixedSinker<'_, Yuv422p16> {}

impl PixelSink for MixedSinker<'_, Yuv422p16> {
  type Input<'r> = Yuv422p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv422p16Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y16,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf16,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf16,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      // Reuses Yuv420p16's u16-output kernel — 4:2:2 per-row shape
      // matches 4:2:0's (half-width UV, one pair per Y pair).
      yuv420p16_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p16_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

impl<'a> MixedSinker<'a, Yuv444p16> {
  /// Attaches a packed **`u16`** RGB output buffer. Output covers
  /// full `u16` range `[0, 65535]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv444p16Sink for MixedSinker<'_, Yuv444p16> {}

impl PixelSink for MixedSinker<'_, Yuv444p16> {
  type Input<'r> = Yuv444p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv444p16Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y16,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv444p16_to_rgb_u16_row(
        row.y(),
        row.u(),
        row.v(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv444p16_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P012 impl ---------------------------------------------------------

impl<'a> MixedSinker<'a, P012> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 12‑bit
  /// output in **low‑bit‑packed** `yuv420p12le` convention (values in
  /// `[0, 4095]` in the low 12 of each `u16`, upper 4 zero) —
  /// **not** P012's high‑bit packing. Callers feeding a P012 consumer
  /// must shift the output left by 4.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P012Sink for MixedSinker<'_, P012> {}

impl PixelSink for MixedSinker<'_, P012> {
  type Input<'r> = P012Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P012Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y12,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf12,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma: P012 samples are high‑bit‑packed (`value << 4`). Taking
    // the high byte via `>> 8` gives the top 8 bits of the 12‑bit
    // value — identical accessor to P010 (both put active bits in the
    // high `BITS` positions of the `u16`).
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p012_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p012_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv420p16 impl ----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p16> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 16‑bit
  /// output (values in `[0, 65535]` — full `u16` range). Length is
  /// measured in `u16` **elements** (`width × height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv420p16Sink for MixedSinker<'_, Yuv420p16> {}

impl PixelSink for MixedSinker<'_, Yuv420p16> {
  type Input<'r> = Yuv420p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p16Row<'_>) -> Result<(), Self::Error> {
    // Luma downshift is `>> 8` — top 8 bits of the 16-bit Y value.
    const BITS: u32 = 16;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y16,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf16,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf16,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p16_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p16_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P016 impl ---------------------------------------------------------

impl<'a> MixedSinker<'a, P016> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 16‑bit
  /// output in `[0, 65535]` — at 16 bits there is no high‑ vs
  /// low‑packing distinction, so the output matches
  /// [`MixedSinker<Yuv420p16>::with_rgb_u16`] numerically.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P016Sink for MixedSinker<'_, P016> {}

impl PixelSink for MixedSinker<'_, P016> {
  type Input<'r> = P016Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P016Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y16,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf16,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma: 16‑bit Y value >> 8 is the top byte.
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p016_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p016_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv420p9 impl -----------------------------------------------------
//
// 9-bit 4:2:0 planar. AV_PIX_FMT_YUV420P9LE — niche AVC High 9 only.
// Reuses the Q15 i32 kernel family at `BITS = 9` via the
// `yuv420p9_to_rgb_*` row primitives (which dispatch to
// `yuv_420p_n_to_rgb_*<9>` internally).

impl<'a> MixedSinker<'a, Yuv420p9> {
  /// Attaches a packed **`u16`** RGB output buffer. 9‑bit low‑packed
  /// (`(1 << 9) - 1 = 511` max).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv420p9Sink for MixedSinker<'_, Yuv420p9> {}

impl PixelSink for MixedSinker<'_, Yuv420p9> {
  type Input<'r> = Yuv420p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p9Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 9;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y9,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf9,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf9,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p9_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p9_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv422p9 impl -----------------------------------------------------
//
// 4:2:2 planar 9‑bit — same per-row chroma shape as 4:2:0 (half-width
// U / V), one chroma row per Y row instead of one per two. Reuses
// `yuv420p9_to_rgb_*` row primitives verbatim.

impl<'a> MixedSinker<'a, Yuv422p9> {
  /// Attaches a packed **`u16`** RGB output buffer. 9-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv422p9Sink for MixedSinker<'_, Yuv422p9> {}

impl PixelSink for MixedSinker<'_, Yuv422p9> {
  type Input<'r> = Yuv422p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv422p9Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 9;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y9,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf9,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf9,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv420p9_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv420p9_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv444p9 impl -----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv444p9> {
  /// Attaches a packed **`u16`** RGB output buffer. 9-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv444p9Sink for MixedSinker<'_, Yuv444p9> {}

impl PixelSink for MixedSinker<'_, Yuv444p9> {
  type Input<'r> = Yuv444p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv444p9Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 9;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y9,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull9,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull9,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv444p9_to_rgb_u16_row(
        row.y(),
        row.u(),
        row.v(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv444p9_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv440p impl -------------------------------------------------------
//
// 4:4:0 planar 8‑bit — full-width chroma, half-height. Per-row math
// matches 4:4:4 (full-width U / V); only the walker reads chroma row
// `r / 2`. Reuses `yuv_444_to_rgb_row` verbatim.

impl Yuv440pSink for MixedSinker<'_, Yuv440p> {}

impl PixelSink for MixedSinker<'_, Yuv440p> {
  type Input<'r> = Yuv440pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv440pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv_444_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv440p10 impl -----------------------------------------------------
//
// 4:4:0 planar 10‑bit. Same row math as 4:4:4 10-bit; reuses
// `yuv444p10_to_rgb_*`. Walker handles the half-height chroma.

impl<'a> MixedSinker<'a, Yuv440p10> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv440p10Sink for MixedSinker<'_, Yuv440p10> {}

impl PixelSink for MixedSinker<'_, Yuv440p10> {
  type Input<'r> = Yuv440p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv440p10Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull10,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull10,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv444p10_to_rgb_u16_row(
        row.y(),
        row.u(),
        row.v(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv444p10_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Yuv440p12 impl -----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv440p12> {
  /// Attaches a packed **`u16`** RGB output buffer. 12-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl Yuv440p12Sink for MixedSinker<'_, Yuv440p12> {}

impl PixelSink for MixedSinker<'_, Yuv440p12> {
  type Input<'r> = Yuv440p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv440p12Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y12,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull12,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull12,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      yuv444p12_to_rgb_u16_row(
        row.y(),
        row.u(),
        row.v(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    yuv444p12_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P210 impl ----------------------------------------------------------
//
// 4:2:2 high-bit-packed semi-planar (10-bit). Per-row UV layout is
// identical to P010 (`width` u16 elements, half-width interleaved);
// only the walker reads chroma row `r` instead of `r / 2`. Reuses the
// `p010_to_rgb_*` row primitives verbatim.

impl<'a> MixedSinker<'a, P210> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit
  /// **low-bit-packed** output (yuv420p10le convention, not P210
  /// packing). Length is in `u16` elements: `width × height × 3`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P210Sink for MixedSinker<'_, P210> {}

impl PixelSink for MixedSinker<'_, P210> {
  type Input<'r> = P210Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P210Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf10,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p010_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p010_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P212 impl ----------------------------------------------------------
//
// 4:2:2 high-bit-packed semi-planar (12-bit). Reuses `p012_to_rgb_*`
// row primitives — only the walker reads chroma row `r` not `r / 2`.

impl<'a> MixedSinker<'a, P212> {
  /// Attaches a packed **`u16`** RGB output buffer. 12-bit
  /// **low-bit-packed** output.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P212Sink for MixedSinker<'_, P212> {}

impl PixelSink for MixedSinker<'_, P212> {
  type Input<'r> = P212Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P212Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y12,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf12,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p012_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p012_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P216 impl ----------------------------------------------------------
//
// 4:2:2 16-bit semi-planar. Reuses `p016_to_rgb_*` row primitives.

impl<'a> MixedSinker<'a, P216> {
  /// Attaches a packed **`u16`** RGB output buffer. 16-bit output
  /// (full `[0, 65535]` range, every bit active).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P216Sink for MixedSinker<'_, P216> {}

impl PixelSink for MixedSinker<'_, P216> {
  type Input<'r> = P216Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P216Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y16,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf16,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // 16-bit Y >> 8 is the top byte (all bits active).
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p016_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p016_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P410 impl ----------------------------------------------------------
//
// 4:4:4 high-bit-packed semi-planar (10-bit). Full-width interleaved
// UV (`2 * width` u16 elements per row). Uses the new
// `p410_to_rgb_*` row primitives (which dispatch to the
// `p_n_444_to_rgb_*<10>` family).

impl<'a> MixedSinker<'a, P410> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P410Sink for MixedSinker<'_, P410> {}

impl PixelSink for MixedSinker<'_, P410> {
  type Input<'r> = P410Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P410Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    // 4:4:4 semi-planar: full-width × 2 elements per pair.
    if row.uv_full().len() != 2 * w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvFull10,
        row: idx,
        expected: 2 * w,
        actual: row.uv_full().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p410_to_rgb_u16_row(
        row.y(),
        row.uv_full(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p410_to_rgb_row(
      row.y(),
      row.uv_full(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P412 impl ----------------------------------------------------------

impl<'a> MixedSinker<'a, P412> {
  /// Attaches a packed **`u16`** RGB output buffer. 12-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P412Sink for MixedSinker<'_, P412> {}

impl PixelSink for MixedSinker<'_, P412> {
  type Input<'r> = P412Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P412Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y12,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_full().len() != 2 * w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvFull12,
        row: idx,
        expected: 2 * w,
        actual: row.uv_full().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p412_to_rgb_u16_row(
        row.y(),
        row.uv_full(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p412_to_rgb_row(
      row.y(),
      row.uv_full(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- P416 impl ----------------------------------------------------------
//
// 4:4:4 16-bit semi-planar. Uses `p416_to_rgb_*` (parallel i64-chroma
// family for u16 output, i32 for u8).

impl<'a> MixedSinker<'a, P416> {
  /// Attaches a packed **`u16`** RGB output buffer. 16-bit output
  /// (full `[0, 65535]` range).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl P416Sink for MixedSinker<'_, P416> {}

impl PixelSink for MixedSinker<'_, P416> {
  type Input<'r> = P416Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P416Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y16,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_full().len() != 2 * w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvFull16,
        row: idx,
        expected: 2 * w,
        actual: row.uv_full().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      p416_to_rgb_u16_row(
        row.y(),
        row.uv_full(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if rgb.is_none() && hsv.is_none() {
      return Ok(());
    }

    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    p416_to_rgb_row(
      row.y(),
      row.uv_full(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
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
fn check_dimensions_match(
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
fn rgb_row_to_luma_row(rgb: &[u8], luma: &mut [u8], coeffs_q8: (u32, u32, u32)) {
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

// ---- Bayer (8-bit) impl --------------------------------------------------

impl MixedSinker<'_, Bayer> {
  /// Sets the luma coefficient set used to derive the luma plane
  /// from demosaiced RGB. Only matters when `with_luma` is also
  /// attached. Default: [`LumaCoefficients::Bt709`].
  ///
  /// Pick the set that matches the gamut your
  /// [`crate::raw::ColorCorrectionMatrix`] targets — see
  /// [`LumaCoefficients`] for guidance. Choosing the wrong set
  /// still produces a valid `u8` luma plane, but its numeric
  /// values won't match what a downstream luma-driven analysis
  /// (scene-cut detection, brightness thresholding, perceptual
  /// diff) expects for non-grayscale content.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_coefficients(mut self, coeffs: LumaCoefficients) -> Self {
    self.set_luma_coefficients(coeffs);
    self
  }

  /// In-place variant of
  /// [`with_luma_coefficients`](Self::with_luma_coefficients).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_coefficients(&mut self, coeffs: LumaCoefficients) -> &mut Self {
    self.luma_coefficients_q8 = coeffs.to_q8();
    self
  }
}

impl BayerSink for MixedSinker<'_, Bayer> {}

impl PixelSink for MixedSinker<'_, Bayer> {
  type Input<'r> = BayerRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Bayer accepts odd dimensions — see `BayerFrame::try_new` for
    // the rationale (cropped Bayer is a real workflow).
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: BayerRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks. The walker always hands
    // matching slices, but a caller bypassing the walker (or one of
    // the future unsafe SIMD backends being wired up) needs the
    // no-panic contract: bad lengths surface as `RowShapeMismatch`,
    // not as a kernel-level `assert!` panic.
    if row.mid().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BayerMid,
        row: idx,
        expected: w,
        actual: row.mid().len(),
      });
    }
    if row.above().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BayerAbove,
        row: idx,
        expected: w,
        actual: row.above().len(),
      });
    }
    if row.below().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BayerBelow,
        row: idx,
        expected: w,
        actual: row.below().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    // `Copy`, captured before the `Self { .. }` destructure so the
    // luma path doesn't have to re-borrow `self`.
    let luma_coeffs_q8 = self.luma_coefficients_q8;

    let Self {
      rgb,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_luma && !want_hsv {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // 8-bit RGB scratch / output buffer. Bayer always derives every
    // output channel from the demosaiced RGB, so the RGB row exists
    // unconditionally when any of `rgb` / `luma` / `hsv` is set.
    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    bayer_to_rgb_row(
      row.above(),
      row.mid(),
      row.below(),
      row.row_parity(),
      row.pattern(),
      row.demosaic(),
      row.m(),
      rgb_row,
      use_simd,
    );

    if let Some(luma) = luma.as_deref_mut() {
      rgb_row_to_luma_row(
        rgb_row,
        &mut luma[one_plane_start..one_plane_end],
        luma_coeffs_q8,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

// ---- Bayer16<BITS> impl --------------------------------------------------

impl<'a, const BITS: u32> MixedSinker<'a, Bayer16<BITS>> {
  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width × height × 3`. Output is **low-packed** at `BITS`
  /// (10-bit white = 1023, 12-bit = 4095, 14-bit = 16383, 16-bit =
  /// 65535) — matches the rest of the high-bit-depth crate.
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if
  /// `buf.len() < width × height × 3`, or `Err(GeometryOverflow)`
  /// on 32-bit overflow.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16). The
  /// required length is measured in `u16` **elements**, not bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Sets the luma coefficient set used to derive the (8-bit)
  /// luma plane from demosaiced RGB. Only matters when `with_luma`
  /// is also attached. Default: [`LumaCoefficients::Bt709`].
  ///
  /// Pick the set that matches the gamut your
  /// [`crate::raw::ColorCorrectionMatrix`] targets — see
  /// [`LumaCoefficients`] for guidance. Choosing the wrong set
  /// still produces a valid `u8` luma plane, but its numeric
  /// values won't match what a downstream luma-driven analysis
  /// (scene-cut detection, brightness thresholding, perceptual
  /// diff) expects for non-grayscale content.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_coefficients(mut self, coeffs: LumaCoefficients) -> Self {
    self.set_luma_coefficients(coeffs);
    self
  }

  /// In-place variant of
  /// [`with_luma_coefficients`](Self::with_luma_coefficients).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_coefficients(&mut self, coeffs: LumaCoefficients) -> &mut Self {
    self.luma_coefficients_q8 = coeffs.to_q8();
    self
  }
}

impl<const BITS: u32> BayerSink16<BITS> for MixedSinker<'_, Bayer16<BITS>> {}

impl<const BITS: u32> PixelSink for MixedSinker<'_, Bayer16<BITS>> {
  type Input<'r> = BayerRow16<'r, BITS>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Bayer accepts odd dimensions — see `BayerFrame::try_new` for
    // the rationale (cropped Bayer is a real workflow).
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // See the 8-bit Bayer impl for the row-shape rationale.
    if row.mid().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bayer16Mid,
        row: idx,
        expected: w,
        actual: row.mid().len(),
      });
    }
    if row.above().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bayer16Above,
        row: idx,
        expected: w,
        actual: row.above().len(),
      });
    }
    if row.below().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bayer16Below,
        row: idx,
        expected: w,
        actual: row.below().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    // `Copy`, captured before the `Self { .. }` destructure so the
    // luma path doesn't have to re-borrow `self`.
    let luma_coeffs_q8 = self.luma_coefficients_q8;

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // u16 RGB output runs the native-depth kernel directly. Output
    // is low-packed at `BITS` per the `*_to_rgb_u16_row` convention.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      bayer16_to_rgb_u16_row::<BITS>(
        row.above(),
        row.mid(),
        row.below(),
        row.row_parity(),
        row.pattern(),
        row.demosaic(),
        row.m(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_luma && !want_hsv {
      return Ok(());
    }

    // 8-bit RGB scratch / output. Same lazy-grow pattern as the
    // 8-bit Bayer impl above.
    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    bayer16_to_rgb_row::<BITS>(
      row.above(),
      row.mid(),
      row.below(),
      row.row_parity(),
      row.pattern(),
      row.demosaic(),
      row.m(),
      rgb_row,
      use_simd,
    );

    if let Some(luma) = luma.as_deref_mut() {
      rgb_row_to_luma_row(
        rgb_row,
        &mut luma[one_plane_start..one_plane_end],
        luma_coeffs_q8,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    Ok(())
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
