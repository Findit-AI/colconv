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
    bayer_to_rgb_row, bayer16_to_rgb_row, bayer16_to_rgb_u16_row, nv12_to_rgb_row, nv21_to_rgb_row,
    nv24_to_rgb_row, nv42_to_rgb_row, p010_to_rgb_row, p010_to_rgb_u16_row, p012_to_rgb_row,
    p012_to_rgb_u16_row, p016_to_rgb_row, p016_to_rgb_u16_row, p410_to_rgb_row,
    p410_to_rgb_u16_row, p412_to_rgb_row, p412_to_rgb_u16_row, p416_to_rgb_row,
    p416_to_rgb_u16_row, rgb_to_hsv_row, yuv_420_to_rgb_row, yuv_444_to_rgb_row,
    yuv420p9_to_rgb_row, yuv420p9_to_rgb_u16_row, yuv420p10_to_rgb_row, yuv420p10_to_rgb_u16_row,
    yuv420p12_to_rgb_row, yuv420p12_to_rgb_u16_row, yuv420p14_to_rgb_row, yuv420p14_to_rgb_u16_row,
    yuv420p16_to_rgb_row, yuv420p16_to_rgb_u16_row, yuv444p9_to_rgb_row, yuv444p9_to_rgb_u16_row,
    yuv444p10_to_rgb_row, yuv444p10_to_rgb_u16_row, yuv444p12_to_rgb_row, yuv444p12_to_rgb_u16_row,
    yuv444p14_to_rgb_row, yuv444p14_to_rgb_u16_row, yuv444p16_to_rgb_row, yuv444p16_to_rgb_u16_row,
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
///   weights → use [`Self::Custom`]
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
  /// Caller-supplied coefficients. Each is a non-negative `f32`;
  /// they should sum to `~1.0` (the constructor doesn't enforce
  /// this — out-of-spec sums produce a brightness-scaled luma
  /// plane). Resolves to Q8 by `(coef * 256 + 0.5) as u32`.
  Custom {
    /// Red weight.
    r: f32,
    /// Green weight.
    g: f32,
    /// Blue weight.
    b: f32,
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
      Self::Custom { r, g, b } => (
        (r * 256.0 + 0.5) as u32,
        (g * 256.0 + 0.5) as u32,
        (b * 256.0 + 0.5) as u32,
      ),
    }
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

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
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
    Ok(())
  }
}

// ---- Yuv422p impl -------------------------------------------------------
//
// 4:2:2 is 4:2:0's vertical-axis twin: same per-row chroma shape
// (half-width U / V, one pair per Y pair), just one chroma row per Y
// row instead of one per two. This impl reuses `yuv_420_to_rgb_row`
// — no new kernels needed.

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
    Ok(())
  }
}

// ---- Yuv444p impl -------------------------------------------------------
//
// 4:4:4 planar: U and V are full-width, full-height. No width parity
// constraint. Uses the new `yuv_444_to_rgb_row` kernel family.

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

// ---- Nv12 impl ----------------------------------------------------------

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
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    // Single-plane row ranges are guaranteed to fit; RGB ranges use
    // checked arithmetic (see the Yuv420p impl above for the full
    // rationale — hsv-only attachment never validated `× 3`).
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma — NV12 luma is the Y plane. Copy verbatim.
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
    Ok(())
  }
}

impl Yuv420pSink for MixedSinker<'_, Yuv420p> {}

// ---- Nv16 impl ----------------------------------------------------------
//
// 4:2:2 is 4:2:0's vertical‑axis twin: one UV row per Y row instead of
// one per two. Per‑row math is identical, so this impl calls the same
// `nv12_to_rgb_row` dispatcher — no new kernels needed.

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
    Ok(())
  }
}

// ---- Nv21 impl ----------------------------------------------------------
//
// Structurally identical to the Nv12 impl — the row primitives hide
// the U/V byte-order difference. Only the trait `Input<'r>` and the
// primitive name change.

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
    Ok(())
  }
}

// ---- Nv24 impl ----------------------------------------------------------
//
// 4:4:4 semi-planar: UV plane is full-width (`2 * width` bytes per
// row), one UV pair per Y pixel. No width parity constraint. Kernel
// is its own family (`nv24_to_rgb_row`) since chroma is no longer
// duplicated across columns.

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
    Ok(())
  }
}

// ---- Nv42 impl ----------------------------------------------------------
//
// Structurally identical to the Nv24 impl — the row primitive hides
// the V/U byte-order difference.

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
mod tests {
  use super::*;
  use crate::{
    ColorMatrix,
    frame::{
      Nv12Frame, Nv16Frame, Nv21Frame, Nv24Frame, Nv42Frame, P010Frame, P012Frame, P016Frame,
      P210Frame, P212Frame, P216Frame, P410Frame, P412Frame, P416Frame, Yuv420p9Frame,
      Yuv420p10Frame, Yuv420p12Frame, Yuv420p14Frame, Yuv420p16Frame, Yuv420pFrame, Yuv422p9Frame,
      Yuv422p10Frame, Yuv422p12Frame, Yuv422p14Frame, Yuv422p16Frame, Yuv422pFrame, Yuv440p10Frame,
      Yuv440p12Frame, Yuv440pFrame, Yuv444p9Frame, Yuv444p10Frame, Yuv444p12Frame, Yuv444p14Frame,
      Yuv444p16Frame, Yuv444pFrame,
    },
    yuv::{
      nv12_to, nv16_to, nv21_to, nv24_to, nv42_to, p010_to, p012_to, p016_to, p210_to, p212_to,
      p216_to, p410_to, p412_to, p416_to, yuv420p_to, yuv420p9_to, yuv420p10_to, yuv420p12_to,
      yuv420p14_to, yuv420p16_to, yuv422p_to, yuv422p9_to, yuv422p10_to, yuv422p12_to,
      yuv422p14_to, yuv422p16_to, yuv440p_to, yuv440p10_to, yuv440p12_to, yuv444p_to, yuv444p9_to,
      yuv444p10_to, yuv444p12_to, yuv444p14_to, yuv444p16_to,
    },
  };

  fn solid_yuv420p_frame(
    width: u32,
    height: u32,
    y: u8,
    u: u8,
    v: u8,
  ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    (
      std::vec![y; w * h],
      std::vec![u; cw * ch],
      std::vec![v; cw * ch],
    )
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn luma_only_copies_y_plane() {
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 42, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 42), "luma should be solid 42");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn rgb_only_converts_gray_to_gray() {
    // Neutral chroma → gray RGB; solid Y=128 → ~128 in every RGB byte.
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn hsv_only_allocates_scratch_and_produces_gray_hsv() {
    // Neutral gray → H=0, S=0, V=~128. No RGB buffer provided.
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut h = std::vec![0xFFu8; 16 * 8];
    let mut s = std::vec![0xFFu8; 16 * 8];
    let mut v = std::vec![0xFFu8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn mixed_all_three_outputs_populated() {
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 200, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut luma = std::vec![0u8; 16 * 8];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    // Luma = Y plane verbatim.
    assert!(luma.iter().all(|&y| y == 200));
    // RGB gray.
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(200) <= 1);
    }
    // HSV of gray.
    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn rgb_with_hsv_uses_user_buffer_not_scratch() {
    // When caller provides RGB, the scratch should remain empty (Vec len 0).
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 100, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert_eq!(
      sink.rgb_scratch.len(),
      0,
      "scratch should stay unallocated when RGB buffer is provided"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn with_simd_false_matches_with_simd_true() {
    // A/B test: same frame, one sinker forces scalar, the other uses
    // SIMD. NEON is bit‑exact to scalar so outputs must match.
    let w = 32usize;
    let h = 16usize;
    let (yp, up, vp) = solid_yuv420p_frame(w as u32, h as u32, 180, 60, 200);
    let src = Yuv420pFrame::new(
      &yp,
      &up,
      &vp,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
    );

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];

    let mut sink_simd = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    assert!(sink_simd.simd());
    assert!(!sink_scalar.simd());

    yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn stride_padded_source_reads_correct_pixels() {
    // 16×8 frame, Y stride 32 (padding), chroma stride 16.
    let w = 16usize;
    let h = 8usize;
    let y_stride = 32usize;
    let c_stride = 16usize;
    let mut yp = std::vec![0xFFu8; y_stride * h]; // padding = 0xFF
    let mut up = std::vec![0xFFu8; c_stride * h / 2];
    let mut vp = std::vec![0xFFu8; c_stride * h / 2];
    // Write actual pixel data in non-padding bytes.
    for row in 0..h {
      for x in 0..w {
        yp[row * y_stride + x] = 50;
      }
    }
    for row in 0..h / 2 {
      for x in 0..w / 2 {
        up[row * c_stride + x] = 128;
        vp[row * c_stride + x] = 128;
      }
    }

    let src = Yuv420pFrame::new(
      &yp,
      &up,
      &vp,
      w as u32,
      h as u32,
      y_stride as u32,
      c_stride as u32,
      c_stride as u32,
    );

    let mut luma = std::vec![0u8; w * h];
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_luma(&mut luma)
      .unwrap();
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(
      luma.iter().all(|&y| y == 50),
      "padding bytes leaked into output"
    );
  }

  // ---- NV12 ---------------------------------------------------------------

  fn solid_nv12_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let ch = h / 2;
    // UV row payload = `width` bytes = `width/2` interleaved UV pairs.
    let mut uv = std::vec![0u8; w * ch];
    for row in 0..ch {
      for i in 0..w / 2 {
        uv[row * w + i * 2] = u;
        uv[row * w + i * 2 + 1] = v;
      }
    }
    (std::vec![y; w * h], uv)
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_luma_only_copies_y_plane() {
    let (yp, uvp) = solid_nv12_frame(16, 8, 42, 128, 128);
    let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Nv12>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 42));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_rgb_only_converts_gray_to_gray() {
    let (yp, uvp) = solid_nv12_frame(16, 8, 128, 128, 128);
    let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
    nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_mixed_all_three_outputs_populated() {
    let (yp, uvp) = solid_nv12_frame(16, 8, 200, 128, 128);
    let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut luma = std::vec![0u8; 16 * 8];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Nv12>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 200));
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(200) <= 1);
    }
    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_with_simd_false_matches_with_simd_true() {
    // 32×16 pseudo-random frame so the SIMD path exercises its main
    // loop and the scalar path processes the full width too.
    let w = 32usize;
    let h = 16usize;
    let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uvp: Vec<u8> = (0..w * h / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let src = Nv12Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Nv12>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Nv12>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    nv12_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    nv12_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar);
  }

  // ---- preflight buffer-size errors ------------------------------------
  //
  // Undersized RGB / luma / HSV buffers must be rejected at attachment
  // time, not part-way through processing. Catching the mistake before
  // any rows are written avoids partially-mutated caller buffers
  // flagged by the adversarial review. With the fallible API these
  // surface as `Err(MixedSinkerError::*BufferTooShort)` / `HsvPlaneTooShort`.

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn attach_short_rgb_returns_err() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3 - 1]; // 1 byte short
    let err = MixedSinker::<Yuv420p>::new(16, 8)
      .with_rgb(&mut rgb)
      .err()
      .unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RgbBufferTooShort {
        expected: 16 * 8 * 3,
        actual: 16 * 8 * 3 - 1,
      }
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn attach_short_luma_returns_err() {
    let mut luma = std::vec![0u8; 16 * 8 - 1];
    let err = MixedSinker::<Yuv420p>::new(16, 8)
      .with_luma(&mut luma)
      .err()
      .unwrap();
    assert_eq!(
      err,
      MixedSinkerError::LumaBufferTooShort {
        expected: 16 * 8,
        actual: 16 * 8 - 1,
      }
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn attach_short_hsv_returns_err() {
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8 - 1]; // V plane short
    let err = MixedSinker::<Yuv420p>::new(16, 8)
      .with_hsv(&mut h, &mut s, &mut v)
      .err()
      .unwrap();
    assert_eq!(
      err,
      MixedSinkerError::HsvPlaneTooShort {
        which: HsvPlane::V,
        expected: 16 * 8,
        actual: 16 * 8 - 1,
      }
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn taller_frame_returns_err_before_any_row_written() {
    // Sink sized for 16×8, feed a 16×10 frame. `begin_frame` returns
    // `Err(DimensionMismatch)` before row 0 — no partial writes.
    let (yp, up, vp) = solid_yuv420p_frame(16, 10, 42, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 10, 16, 8, 8);

    const SENTINEL: u8 = 0xEE;
    let mut luma = std::vec![SENTINEL; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
      .err()
      .unwrap();
    assert_eq!(
      err,
      MixedSinkerError::DimensionMismatch {
        configured_w: 16,
        configured_h: 8,
        frame_w: 16,
        frame_h: 10,
      }
    );
    assert!(
      luma.iter().all(|&b| b == SENTINEL),
      "no rows should have been written before the Err"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn shorter_frame_returns_err_before_any_row_written() {
    // Sink sized 16×8, frame is 16×4. Without the `begin_frame`
    // preflight, the walker would silently process 4 rows and leave
    // rows 4..7 stale from the previous frame. Preflight returns
    // `Err(DimensionMismatch)` with no side effects.
    let (yp, up, vp) = solid_yuv420p_frame(16, 4, 42, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 4, 16, 8, 8);

    const SENTINEL: u8 = 0xEE;
    let mut luma = std::vec![SENTINEL; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
      .err()
      .unwrap();
    assert_eq!(
      err,
      MixedSinkerError::DimensionMismatch {
        configured_w: 16,
        configured_h: 8,
        frame_w: 16,
        frame_h: 4,
      }
    );
    assert!(
      luma.iter().all(|&b| b == SENTINEL),
      "no rows should have been written before the Err"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_width_mismatch_returns_err() {
    let (yp, uvp) = solid_nv12_frame(16, 8, 42, 128, 128);
    let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 32 * 8 * 3];
    let mut sink = MixedSinker::<Nv12>::new(32, 8).with_rgb(&mut rgb).unwrap();
    let err = nv12_to(&src, true, ColorMatrix::Bt601, &mut sink)
      .err()
      .unwrap();
    assert!(
      matches!(
        err,
        MixedSinkerError::DimensionMismatch {
          configured_w: 32,
          frame_w: 16,
          ..
        }
      ),
      "unexpected error variant: {err:?}"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p_width_mismatch_returns_err() {
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 42, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 32 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(32, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
      .err()
      .unwrap();
    assert!(
      matches!(
        err,
        MixedSinkerError::DimensionMismatch {
          configured_w: 32,
          frame_w: 16,
          ..
        }
      ),
      "unexpected error variant: {err:?}"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_shorter_frame_returns_err_before_any_row_written() {
    let (yp, uvp) = solid_nv12_frame(16, 4, 42, 128, 128);
    let src = Nv12Frame::new(&yp, &uvp, 16, 4, 16, 16);

    const SENTINEL: u8 = 0xEE;
    let mut luma = std::vec![SENTINEL; 16 * 8];
    let mut sink = MixedSinker::<Nv12>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    let err = nv12_to(&src, true, ColorMatrix::Bt601, &mut sink)
      .err()
      .unwrap();
    assert!(matches!(err, MixedSinkerError::DimensionMismatch { .. }));
    assert!(
      luma.iter().all(|&b| b == SENTINEL),
      "no rows should have been written before the Err"
    );
  }

  /// Sanity check that an Infallible sink (compile-time proof of
  /// no-error) compiles and runs. Mirrors the trait-docs pattern.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn infallible_sink_compiles_and_runs() {
    use core::convert::Infallible;

    struct RowCounter(usize);
    impl PixelSink for RowCounter {
      type Input<'a> = Yuv420pRow<'a>;
      type Error = Infallible;
      fn process(&mut self, _row: Yuv420pRow<'_>) -> Result<(), Infallible> {
        self.0 += 1;
        Ok(())
      }
    }
    impl Yuv420pSink for RowCounter {}

    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);
    let mut counter = RowCounter(0);
    // `Result<(), Infallible>` — the compiler knows Err is
    // uninhabited, so `.unwrap()` here is free and infallible.
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut counter).unwrap();
    assert_eq!(counter.0, 8);
  }

  // ---- direct process() bypass paths ----------------------------------
  //
  // The walker normally guarantees (a) begin_frame runs first and
  // validates frame dimensions, (b) row.y()/u/v/uv slices have the
  // right length, (c) `idx < height`. A direct `process` call can
  // break any of these. The defense-in-depth checks in `process`
  // must return a specific error variant, not panic — verified here
  // by constructing rows manually and calling `process`.

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p_process_rejects_short_y_slice() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    // Build a row with a 15-byte Y slice (wrong — sink configured for 16).
    let y = [0u8; 15];
    let u = [128u8; 8];
    let v = [128u8; 8];
    let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: 0,
        expected: 16,
        actual: 15,
      }
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p_process_rejects_short_u_half() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    let y = [0u8; 16];
    let u = [128u8; 7]; // expected 8
    let v = [128u8; 8];
    let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf,
        row: 0,
        expected: 8,
        actual: 7,
      }
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p_process_rejects_out_of_range_row_idx() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    let y = [0u8; 16];
    let u = [128u8; 8];
    let v = [128u8; 8];
    // idx = 8 exceeds configured height 8 — would otherwise panic on
    // `rgb[idx * w * 3 ..]` indexing.
    let row = Yuv420pRow::new(&y, &u, &v, 8, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RowIndexOutOfRange {
        row: 8,
        configured_height: 8,
      }
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p_odd_width_sink_returns_err_at_begin_frame() {
    // A sink configured with an odd width would later panic inside
    // `yuv_420_to_rgb_row` (which asserts `width & 1 == 0`). The
    // fallible API surfaces this as `OddWidth` at frame start — no
    // rows are processed, no panic. Width=15, height=8 — matching
    // frame so `DimensionMismatch` can't fire first.
    let w = 15usize;
    let h = 8usize;
    let y = std::vec![0u8; w * h];
    let u = std::vec![128u8; w.div_ceil(2) * h / 2 + 8]; // any valid size
    let v = std::vec![128u8; w.div_ceil(2) * h / 2 + 8];
    // Build the Frame separately — Yuv420pFrame rejects odd width
    // too, so we can't construct a 15-wide frame. That's fine: we
    // only need to hit `begin_frame`, which takes (width, height)
    // parameters directly. Call it manually.
    let mut rgb = std::vec![0u8; 16 * 8 * 3]; // Dummy; not touched.
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap();
    let err = sink.begin_frame(w as u32, h as u32).err().unwrap();
    assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
    // Silence unused-vec warnings — these would have been the plane data.
    let _ = (y, u, v);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p_odd_width_sink_returns_err_at_direct_process() {
    // Direct `process` caller bypassing `begin_frame`. Process must
    // still reject odd width before calling the kernel.
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(15, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    let y = [0u8; 15];
    let u = [128u8; 7]; // ceil(15/2) = 8; 7 triggers the width check first
    let v = [128u8; 7];
    let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_odd_width_sink_returns_err_at_begin_frame() {
    let w = 15usize;
    let h = 8usize;
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv12>::new(w, h).with_rgb(&mut rgb).unwrap();
    let err = sink.begin_frame(w as u32, h as u32).err().unwrap();
    assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_odd_width_sink_returns_err_at_direct_process() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv12>::new(15, 8).with_rgb(&mut rgb).unwrap();
    let y = [0u8; 15];
    let uv = [128u8; 15];
    let row = Nv12Row::new(&y, &uv, 0, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_process_rejects_short_uv_slice() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
    let y = [0u8; 16];
    let uv = [128u8; 15]; // expected 16
    let row = Nv12Row::new(&y, &uv, 0, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf,
        row: 0,
        expected: 16,
        actual: 15,
      }
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_process_rejects_out_of_range_row_idx() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
    let y = [0u8; 16];
    let uv = [128u8; 16];
    let row = Nv12Row::new(&y, &uv, 8, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RowIndexOutOfRange {
        row: 8,
        configured_height: 8,
      }
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_matches_yuv420p_mixed_sinker() {
    // Cross-format guarantee: an NV12 frame built from the same U / V
    // bytes as a Yuv420p frame produces byte-identical RGB output via
    // MixedSinker on both families.
    let w = 32u32;
    let h = 16u32;
    let ws = w as usize;
    let hs = h as usize;
    let yp: Vec<u8> = (0..ws * hs).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let up: Vec<u8> = (0..(ws / 2) * (hs / 2))
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let vp: Vec<u8> = (0..(ws / 2) * (hs / 2))
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    // Build NV12 UV plane: chroma row r, column c → uv[r * w + 2*c] = U,
    // uv[r * w + 2*c + 1] = V, where U / V come from the same (r, c)
    // sample of the planar fixture above.
    let mut uvp: Vec<u8> = std::vec![0u8; ws * (hs / 2)];
    for r in 0..hs / 2 {
      for c in 0..ws / 2 {
        uvp[r * ws + 2 * c] = up[r * (ws / 2) + c];
        uvp[r * ws + 2 * c + 1] = vp[r * (ws / 2) + c];
      }
    }

    let yuv420p_src = Yuv420pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
    let nv12_src = Nv12Frame::new(&yp, &uvp, w, h, w, w);

    let mut rgb_yuv420p = std::vec![0u8; ws * hs * 3];
    let mut rgb_nv12 = std::vec![0u8; ws * hs * 3];
    let mut s_yuv = MixedSinker::<Yuv420p>::new(ws, hs)
      .with_rgb(&mut rgb_yuv420p)
      .unwrap();
    let mut s_nv = MixedSinker::<Nv12>::new(ws, hs)
      .with_rgb(&mut rgb_nv12)
      .unwrap();
    yuv420p_to(&yuv420p_src, false, ColorMatrix::Bt709, &mut s_yuv).unwrap();
    nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv).unwrap();

    assert_eq!(rgb_yuv420p, rgb_nv12);
  }

  // ---- NV16 MixedSinker ---------------------------------------------------
  //
  // 4:2:2: chroma is half-width, full-height. Per-row math is
  // identical to NV12 (the impl calls `nv12_to_rgb_row`), so the
  // tests mirror the NV12 set and add a cross-layout parity check
  // against an NV12-shaped frame whose chroma rows are each
  // duplicated (simulating 4:2:0 from 4:2:2 by vertical downsampling).

  fn solid_nv16_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    // NV16 UV is full-height (h rows, not h/2).
    let mut uv = std::vec![0u8; w * h];
    for row in 0..h {
      for i in 0..w / 2 {
        uv[row * w + i * 2] = u;
        uv[row * w + i * 2 + 1] = v;
      }
    }
    (std::vec![y; w * h], uv)
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv16_luma_only_copies_y_plane() {
    let (yp, uvp) = solid_nv16_frame(16, 8, 42, 128, 128);
    let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Nv16>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 42));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv16_rgb_only_converts_gray_to_gray() {
    let (yp, uvp) = solid_nv16_frame(16, 8, 128, 128, 128);
    let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv16>::new(16, 8).with_rgb(&mut rgb).unwrap();
    nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv16_mixed_all_three_outputs_populated() {
    let (yp, uvp) = solid_nv16_frame(16, 8, 200, 128, 128);
    let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut luma = std::vec![0u8; 16 * 8];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Nv16>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 200));
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(200) <= 1);
    }
    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv16_with_simd_false_matches_with_simd_true() {
    let w = 32usize;
    let h = 16usize;
    let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uvp: Vec<u8> = (0..w * h).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
    let src = Nv16Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Nv16>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Nv16>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    nv16_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    nv16_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv16_matches_nv12_mixed_sinker_with_duplicated_chroma() {
    // Cross-layout parity: if we build an NV12 frame whose `uv_half`
    // plane contains only the even NV16 chroma rows (row 0, 2, 4, …),
    // the two frames must produce identical RGB output at every Y
    // row. This validates that NV16's walker + NV12's row primitive
    // yield the right 4:2:2 semantics (one UV row per Y row) on a
    // 4:2:0 reference that shares chroma across row pairs.
    let w = 32usize;
    let h = 16usize;
    let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv_nv16: Vec<u8> = (0..w * h).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
    // Build NV12 chroma by sampling only even NV16 chroma rows.
    let mut uv_nv12 = std::vec![0u8; w * h / 2];
    for c_row in 0..h / 2 {
      let src_row = c_row * 2; // even NV16 chroma rows
      uv_nv12[c_row * w..(c_row + 1) * w].copy_from_slice(&uv_nv16[src_row * w..(src_row + 1) * w]);
    }
    // …and make the NV16 odd chroma rows match their even neighbors so
    // the 4:2:0 vertical upsample (same chroma for row pairs) matches
    // what NV16 carries through.
    let mut uv_nv16_aligned = uv_nv16.clone();
    for c_row in 0..h / 2 {
      let even_row = c_row * 2;
      let odd_row = even_row + 1;
      let (even, odd) = uv_nv16_aligned.split_at_mut(odd_row * w);
      odd[..w].copy_from_slice(&even[even_row * w..even_row * w + w]);
    }
    let nv16_src = Nv16Frame::new(
      &yp,
      &uv_nv16_aligned,
      w as u32,
      h as u32,
      w as u32,
      w as u32,
    );
    let nv12_src = Nv12Frame::new(&yp, &uv_nv12, w as u32, h as u32, w as u32, w as u32);

    let mut rgb_nv16 = std::vec![0u8; w * h * 3];
    let mut rgb_nv12 = std::vec![0u8; w * h * 3];
    let mut s_nv16 = MixedSinker::<Nv16>::new(w, h)
      .with_rgb(&mut rgb_nv16)
      .unwrap();
    let mut s_nv12 = MixedSinker::<Nv12>::new(w, h)
      .with_rgb(&mut rgb_nv12)
      .unwrap();
    nv16_to(&nv16_src, false, ColorMatrix::Bt709, &mut s_nv16).unwrap();
    nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv12).unwrap();

    assert_eq!(rgb_nv16, rgb_nv12);
  }

  #[test]
  fn nv16_odd_width_sink_returns_err_at_begin_frame() {
    let mut rgb = std::vec![0u8; 15 * 8 * 3];
    let mut sink = MixedSinker::<Nv16>::new(15, 8).with_rgb(&mut rgb).unwrap();
    let (yp, uvp) = solid_nv16_frame(16, 8, 0, 0, 0); // dummy 16-wide frame
    let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);
    let err = nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap_err();
    assert!(matches!(err, MixedSinkerError::OddWidth { width: 15 }));
  }

  // ---- NV21 MixedSinker ---------------------------------------------------

  fn solid_nv21_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let ch = h / 2;
    // VU row payload = `width` bytes = `width/2` interleaved V/U pairs
    // (V first).
    let mut vu = std::vec![0u8; w * ch];
    for row in 0..ch {
      for i in 0..w / 2 {
        vu[row * w + i * 2] = v;
        vu[row * w + i * 2 + 1] = u;
      }
    }
    (std::vec![y; w * h], vu)
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_luma_only_copies_y_plane() {
    let (yp, vup) = solid_nv21_frame(16, 8, 42, 128, 128);
    let src = Nv21Frame::new(&yp, &vup, 16, 8, 16, 16);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Nv21>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 42));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_rgb_only_converts_gray_to_gray() {
    let (yp, vup) = solid_nv21_frame(16, 8, 128, 128, 128);
    let src = Nv21Frame::new(&yp, &vup, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv21>::new(16, 8).with_rgb(&mut rgb).unwrap();
    nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_mixed_all_three_outputs_populated() {
    let (yp, vup) = solid_nv21_frame(16, 8, 200, 128, 128);
    let src = Nv21Frame::new(&yp, &vup, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut luma = std::vec![0u8; 16 * 8];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Nv21>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 200));
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(200) <= 1);
    }
    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_matches_nv12_mixed_sinker_with_swapped_chroma() {
    // Cross-format guarantee: an NV21 frame built from the same U / V
    // bytes as an NV12 frame (just byte-swapped in the chroma plane)
    // must produce identical RGB output via MixedSinker.
    let w = 32u32;
    let h = 16u32;
    let ws = w as usize;
    let hs = h as usize;

    let yp: Vec<u8> = (0..ws * hs).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let mut uvp: Vec<u8> = std::vec![0u8; ws * (hs / 2)];
    for r in 0..hs / 2 {
      for c in 0..ws / 2 {
        uvp[r * ws + 2 * c] = ((c + r * 53) & 0xFF) as u8; // U
        uvp[r * ws + 2 * c + 1] = ((c + r * 71) & 0xFF) as u8; // V
      }
    }
    // Byte-swap each chroma pair to get the VU-ordered stream.
    let mut vup: Vec<u8> = uvp.clone();
    for r in 0..hs / 2 {
      for c in 0..ws / 2 {
        vup[r * ws + 2 * c] = uvp[r * ws + 2 * c + 1];
        vup[r * ws + 2 * c + 1] = uvp[r * ws + 2 * c];
      }
    }

    let nv12_src = Nv12Frame::new(&yp, &uvp, w, h, w, w);
    let nv21_src = Nv21Frame::new(&yp, &vup, w, h, w, w);

    let mut rgb_nv12 = std::vec![0u8; ws * hs * 3];
    let mut rgb_nv21 = std::vec![0u8; ws * hs * 3];
    let mut s_nv12 = MixedSinker::<Nv12>::new(ws, hs)
      .with_rgb(&mut rgb_nv12)
      .unwrap();
    let mut s_nv21 = MixedSinker::<Nv21>::new(ws, hs)
      .with_rgb(&mut rgb_nv21)
      .unwrap();
    nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv12).unwrap();
    nv21_to(&nv21_src, false, ColorMatrix::Bt709, &mut s_nv21).unwrap();

    assert_eq!(rgb_nv12, rgb_nv21);
  }

  // ---- NV24 MixedSinker ---------------------------------------------------
  //
  // 4:4:4 semi-planar: UV row is `2 * width` bytes (one UV pair per
  // Y pixel). Tests mirror the NV12 set plus one cross-format parity
  // check against a synthetic NV42 frame (byte-swap the interleaved
  // chroma → identical RGB output).

  fn solid_nv24_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    // UV row payload = `2 * width` bytes = `width` interleaved U/V pairs.
    let mut uv = std::vec![0u8; 2 * w * h];
    for row in 0..h {
      for i in 0..w {
        uv[row * 2 * w + i * 2] = u;
        uv[row * 2 * w + i * 2 + 1] = v;
      }
    }
    (std::vec![y; w * h], uv)
  }

  #[test]
  fn nv24_luma_only_copies_y_plane() {
    let (yp, uvp) = solid_nv24_frame(16, 8, 42, 128, 128);
    let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Nv24>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 42));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv24_rgb_only_converts_gray_to_gray() {
    let (yp, uvp) = solid_nv24_frame(16, 8, 128, 128, 128);
    let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
    nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv24_mixed_all_three_outputs_populated() {
    let (yp, uvp) = solid_nv24_frame(16, 8, 200, 128, 128);
    let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut luma = std::vec![0u8; 16 * 8];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Nv24>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&y| y == 200));
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(200) <= 1);
    }
    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv24_accepts_odd_width() {
    // 4:4:4 removes the width parity constraint. A 17-wide frame
    // should round-trip cleanly.
    let (yp, uvp) = solid_nv24_frame(17, 8, 200, 128, 128);
    let src = Nv24Frame::new(&yp, &uvp, 17, 8, 17, 34);

    let mut rgb = std::vec![0u8; 17 * 8 * 3];
    let mut sink = MixedSinker::<Nv24>::new(17, 8).with_rgb(&mut rgb).unwrap();
    nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(200) <= 1);
    }
  }

  // ---- NV42 MixedSinker ---------------------------------------------------

  fn solid_nv42_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    // VU row payload = `2 * width` bytes = `width` interleaved V/U pairs
    // (byte-swapped relative to NV24).
    let mut vu = std::vec![0u8; 2 * w * h];
    for row in 0..h {
      for i in 0..w {
        vu[row * 2 * w + i * 2] = v;
        vu[row * 2 * w + i * 2 + 1] = u;
      }
    }
    (std::vec![y; w * h], vu)
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv42_rgb_only_converts_gray_to_gray() {
    let (yp, vup) = solid_nv42_frame(16, 8, 128, 128, 128);
    let src = Nv42Frame::new(&yp, &vup, 16, 8, 16, 32);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv42>::new(16, 8).with_rgb(&mut rgb).unwrap();
    nv42_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv42_matches_nv24_mixed_sinker_with_swapped_chroma() {
    // Cross-format parity: for the same Y plane and byte-swapped
    // interleaved chroma, NV24 and NV42 must produce identical RGB
    // output. Mirrors the NV21↔NV12 test.
    let w = 33usize; // deliberately odd to exercise the no-parity-constraint path
    let h = 8usize;
    let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv_nv24: Vec<u8> = (0..2 * w * h)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    // Build NV42 chroma by swapping each (U, V) pair.
    let mut vu_nv42 = std::vec![0u8; 2 * w * h];
    for i in 0..w * h {
      vu_nv42[i * 2] = uv_nv24[i * 2 + 1];
      vu_nv42[i * 2 + 1] = uv_nv24[i * 2];
    }
    let nv24_src = Nv24Frame::new(&yp, &uv_nv24, w as u32, h as u32, w as u32, (2 * w) as u32);
    let nv42_src = Nv42Frame::new(&yp, &vu_nv42, w as u32, h as u32, w as u32, (2 * w) as u32);

    let mut rgb_nv24 = std::vec![0u8; w * h * 3];
    let mut rgb_nv42 = std::vec![0u8; w * h * 3];
    let mut s_nv24 = MixedSinker::<Nv24>::new(w, h)
      .with_rgb(&mut rgb_nv24)
      .unwrap();
    let mut s_nv42 = MixedSinker::<Nv42>::new(w, h)
      .with_rgb(&mut rgb_nv42)
      .unwrap();
    nv24_to(&nv24_src, false, ColorMatrix::Bt709, &mut s_nv24).unwrap();
    nv42_to(&nv42_src, false, ColorMatrix::Bt709, &mut s_nv42).unwrap();

    assert_eq!(rgb_nv24, rgb_nv42);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv24_with_simd_false_matches_with_simd_true() {
    // Widths chosen to force each backend's main loop AND its
    // scalar-tail path:
    // - 16, 17 → NEON/SSE4.1/wasm main (16-Y block), AVX2 + AVX-512 no main.
    // - 32, 33 → AVX2 main (32-Y block), AVX-512 no main.
    // - 64, 65 → AVX-512 main (64-Y block) once + optional 1-px tail.
    // - 127, 128 → AVX-512 main twice, 127 also forces a 63-px tail.
    // - 1920 → wide real-world baseline.
    for &w in &[16usize, 17, 32, 33, 64, 65, 127, 128, 1920] {
      let h = 4usize;
      let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
      let uvp: Vec<u8> = (0..2 * w * h)
        .map(|i| ((i * 53 + 23) & 0xFF) as u8)
        .collect();
      let src = Nv24Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, (2 * w) as u32);

      let mut rgb_simd = std::vec![0u8; w * h * 3];
      let mut rgb_scalar = std::vec![0u8; w * h * 3];
      let mut sink_simd = MixedSinker::<Nv24>::new(w, h)
        .with_rgb(&mut rgb_simd)
        .unwrap();
      let mut sink_scalar = MixedSinker::<Nv24>::new(w, h)
        .with_rgb(&mut rgb_scalar)
        .unwrap()
        .with_simd(false);
      nv24_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
      nv24_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

      assert_eq!(rgb_simd, rgb_scalar, "NV24 SIMD≠scalar at width {w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv42_with_simd_false_matches_with_simd_true() {
    // Same width coverage as the NV24 variant — exercises every
    // backend's main loop + scalar tail for the `SWAP_UV = true`
    // monomorphization.
    for &w in &[16usize, 17, 32, 33, 64, 65, 127, 128, 1920] {
      let h = 4usize;
      let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
      let vup: Vec<u8> = (0..2 * w * h)
        .map(|i| ((i * 53 + 23) & 0xFF) as u8)
        .collect();
      let src = Nv42Frame::new(&yp, &vup, w as u32, h as u32, w as u32, (2 * w) as u32);

      let mut rgb_simd = std::vec![0u8; w * h * 3];
      let mut rgb_scalar = std::vec![0u8; w * h * 3];
      let mut sink_simd = MixedSinker::<Nv42>::new(w, h)
        .with_rgb(&mut rgb_simd)
        .unwrap();
      let mut sink_scalar = MixedSinker::<Nv42>::new(w, h)
        .with_rgb(&mut rgb_scalar)
        .unwrap()
        .with_simd(false);
      nv42_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
      nv42_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

      assert_eq!(rgb_simd, rgb_scalar, "NV42 SIMD≠scalar at width {w}");
    }
  }

  #[test]
  fn nv24_width_mismatch_returns_err() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
    // 8-tall src matches the sink; width 17 vs sink's 16 triggers the
    // mismatch in `begin_frame`.
    let (yp, uvp) = solid_nv24_frame(17, 8, 0, 0, 0);
    let src = Nv24Frame::new(&yp, &uvp, 17, 8, 17, 34);
    let err = nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap_err();
    assert!(matches!(err, MixedSinkerError::DimensionMismatch { .. }));
  }

  #[test]
  fn nv24_process_rejects_short_uv_slice() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
    let y = [0u8; 16];
    let uv = [128u8; 31]; // expected 2 * 16 = 32
    let row = Nv24Row::new(&y, &uv, 0, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvFull,
        row: 0,
        expected: 32,
        actual: 31,
      }
    );
  }

  #[test]
  fn nv24_process_rejects_out_of_range_row_idx() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
    let y = [0u8; 16];
    let uv = [128u8; 32];
    let row = Nv24Row::new(&y, &uv, 8, ColorMatrix::Bt601, true); // row 8 == height
    let err = sink.process(row).err().unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RowIndexOutOfRange {
        row: 8,
        configured_height: 8,
      }
    );
  }

  #[test]
  fn nv42_process_rejects_short_vu_slice() {
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv42>::new(16, 8).with_rgb(&mut rgb).unwrap();
    let y = [0u8; 16];
    let vu = [128u8; 31]; // expected 32
    let row = Nv42Row::new(&y, &vu, 0, ColorMatrix::Bt601, true);
    let err = sink.process(row).err().unwrap();
    assert_eq!(
      err,
      MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VuFull,
        row: 0,
        expected: 32,
        actual: 31,
      }
    );
  }

  // ---- Yuv420p10 --------------------------------------------------------

  fn solid_yuv420p10_frame(
    width: u32,
    height: u32,
    y: u16,
    u: u16,
    v: u16,
  ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    (
      std::vec![y; w * h],
      std::vec![u; cw * ch],
      std::vec![v; cw * ch],
    )
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p10_rgb_u8_only_gray_is_gray() {
    // 10-bit mid-gray: Y=512, UV=512 → 8-bit RGB ≈ 128 on every channel.
    let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
    let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p10_rgb_u16_only_native_depth_gray() {
    // Same mid-gray frame → u16 RGB output in native 10-bit depth, so
    // each channel should be ≈ 512 (the 10-bit mid).
    let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
    let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .unwrap();
    yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
      // Upper 6 bits of each u16 must be zero — 10-bit convention.
      assert!(px[0] <= 1023);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p10_rgb_u8_and_u16_both_populated() {
    // 10-bit full-range white: Y=1023, UV=512. Both buffers should
    // fill with their respective "white" values (255 for u8, 1023 for
    // u16) in the same call.
    let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 1023, 512, 512);
    let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(rgb_u8.iter().all(|&c| c == 255));
    assert!(rgb_u16.iter().all(|&c| c == 1023));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p10_luma_downshifts_to_8bit() {
    // Y=512 at 10 bits → 512 >> 2 = 128 at 8 bits.
    let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
    let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&l| l == 128));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p10_hsv_from_gray_is_zero_hue_zero_sat() {
    // HSV derived from the internal u8 RGB scratch: neutral gray →
    // H=0, S=0, V≈128. Exercises the "HSV without RGB" scratch path
    // on the 10-bit source.
    let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
    let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut h = std::vec![0xFFu8; 16 * 8];
    let mut s = std::vec![0xFFu8; 16 * 8];
    let mut v = std::vec![0xFFu8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p10_rgb_u16_too_short_returns_err() {
    let mut rgb = std::vec![0u16; 10]; // Way too small.
    let err = MixedSinker::<Yuv420p10>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .err()
      .unwrap();
    assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p10_with_simd_false_matches_with_simd_true() {
    // The SIMD toggle exercises scalar-vs-SIMD dispatch. Both paths
    // must produce byte-identical results on both outputs.
    let (yp, up, vp) = solid_yuv420p10_frame(64, 16, 600, 400, 700);
    let src = Yuv420p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

    let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
    let mut s_scalar = MixedSinker::<Yuv420p10>::new(64, 16)
      .with_simd(false)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    yuv420p10_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
    let mut s_simd = MixedSinker::<Yuv420p10>::new(64, 16)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    yuv420p10_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

    assert_eq!(rgb_scalar, rgb_simd);
    assert_eq!(rgb_u16_scalar, rgb_u16_simd);
  }

  // ---- P010 --------------------------------------------------------------
  //
  // Semi-planar 10-bit, high-bit-packed (samples in high 10 of each
  // u16). Mirrors the Yuv420p10 test shape but with UV interleaved.

  fn solid_p010_frame(
    width: u32,
    height: u32,
    y_10bit: u16,
    u_10bit: u16,
    v_10bit: u16,
  ) -> (Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    // Shift into the high 10 bits (P010 packing).
    let y = std::vec![y_10bit << 6; w * h];
    let uv: Vec<u16> = (0..cw * ch)
      .flat_map(|_| [u_10bit << 6, v_10bit << 6])
      .collect();
    (y, uv)
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p010_rgb_u8_only_gray_is_gray() {
    // 10-bit mid-gray Y=512, UV=512 → ~128 u8 RGB across the frame.
    let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
    let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<P010>::new(16, 8).with_rgb(&mut rgb).unwrap();
    p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p010_rgb_u16_only_native_depth_gray() {
    // Output u16 is yuv420p10le-packed (10-bit in low 10) even though
    // the input is P010-packed.
    let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
    let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<P010>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .unwrap();
    p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
      assert!(
        px[0] <= 1023,
        "output must stay within 10-bit low-packed range"
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p010_rgb_u8_and_u16_both_populated() {
    // 10-bit full-range white: Y=1023, UV=512. Both buffers fill in
    // one call.
    let (yp, uvp) = solid_p010_frame(16, 8, 1023, 512, 512);
    let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<P010>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(rgb_u8.iter().all(|&c| c == 255));
    assert!(rgb_u16.iter().all(|&c| c == 1023));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p010_luma_downshifts_to_8bit() {
    // Y=512 at 10 bits, P010-packed (0x8000). After >> 8, the 8-bit
    // luma is 0x80 = 128.
    let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
    let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<P010>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&l| l == 128));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p010_matches_yuv420p10_mixed_sinker_with_shifted_samples() {
    // Logical equivalence: same samples fed through the two formats
    // (low-packed as yuv420p10, high-packed as P010) must produce
    // byte-identical u8 RGB.
    let w = 16u32;
    let h = 8u32;
    let y = 600u16;
    let u = 400u16;
    let v = 700u16;

    let (yp_p10, up_p10, vp_p10) = solid_yuv420p10_frame(w, h, y, u, v);
    let src_p10 = Yuv420p10Frame::new(&yp_p10, &up_p10, &vp_p10, w, h, w, w / 2, w / 2);

    let (yp_p010, uvp_p010) = solid_p010_frame(w, h, y, u, v);
    let src_p010 = P010Frame::new(&yp_p010, &uvp_p010, w, h, w, w);

    let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_p010 = std::vec![0u8; (w * h * 3) as usize];
    let mut s_yuv = MixedSinker::<Yuv420p10>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_yuv)
      .unwrap();
    let mut s_p010 = MixedSinker::<P010>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_p010)
      .unwrap();
    yuv420p10_to(&src_p10, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
    p010_to(&src_p010, true, ColorMatrix::Bt709, &mut s_p010).unwrap();
    assert_eq!(rgb_yuv, rgb_p010);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p010_rgb_u16_too_short_returns_err() {
    let mut rgb = std::vec![0u16; 10];
    let err = MixedSinker::<P010>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .err()
      .unwrap();
    assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p010_with_simd_false_matches_with_simd_true() {
    // Stubs delegate to scalar so simd=true and simd=false produce
    // byte-identical output for now. Real SIMD backends will replace
    // the stubs — equivalence is preserved by design.
    let (yp, uvp) = solid_p010_frame(64, 16, 600, 400, 700);
    let src = P010Frame::new(&yp, &uvp, 64, 16, 64, 64);

    let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
    let mut s_scalar = MixedSinker::<P010>::new(64, 16)
      .with_simd(false)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    p010_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
    let mut s_simd = MixedSinker::<P010>::new(64, 16)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    p010_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

    assert_eq!(rgb_scalar, rgb_simd);
    assert_eq!(rgb_u16_scalar, rgb_u16_simd);
  }

  // ---- Yuv420p12 ---------------------------------------------------------
  //
  // Planar 12-bit, low-bit-packed. Mirrors the Yuv420p10 shape — same
  // planar layout, wider sample range. `mid-gray` for 12-bit is
  // Y=UV=2048; native-depth white (full-range) is 4095.

  fn solid_yuv420p12_frame(
    width: u32,
    height: u32,
    y: u16,
    u: u16,
    v: u16,
  ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    (
      std::vec![y; w * h],
      std::vec![u; cw * ch],
      std::vec![v; cw * ch],
    )
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p12_rgb_u8_only_gray_is_gray() {
    let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
    let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p12_rgb_u16_only_native_depth_gray() {
    let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
    let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .unwrap();
    yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
      // Upper 4 bits must be zero — 12-bit low-packed convention.
      assert!(px[0] <= 4095);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p12_rgb_u8_and_u16_both_populated() {
    // Full-range white: Y=4095, UV=2048.
    let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 4095, 2048, 2048);
    let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(rgb_u8.iter().all(|&c| c == 255));
    assert!(rgb_u16.iter().all(|&c| c == 4095));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p12_luma_downshifts_to_8bit() {
    // Y=2048 at 12 bits → 2048 >> (12 - 8) = 128 at 8 bits.
    let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
    let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&l| l == 128));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p12_hsv_from_gray_is_zero_hue_zero_sat() {
    let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
    let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut h = std::vec![0xFFu8; 16 * 8];
    let mut s = std::vec![0xFFu8; 16 * 8];
    let mut v = std::vec![0xFFu8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p12_rgb_u16_too_short_returns_err() {
    let mut rgb = std::vec![0u16; 10];
    let err = MixedSinker::<Yuv420p12>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .err()
      .unwrap();
    assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p12_with_simd_false_matches_with_simd_true() {
    let (yp, up, vp) = solid_yuv420p12_frame(64, 16, 2400, 1600, 2800);
    let src = Yuv420p12Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

    let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
    let mut s_scalar = MixedSinker::<Yuv420p12>::new(64, 16)
      .with_simd(false)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    yuv420p12_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
    let mut s_simd = MixedSinker::<Yuv420p12>::new(64, 16)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    yuv420p12_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

    assert_eq!(rgb_scalar, rgb_simd);
    assert_eq!(rgb_u16_scalar, rgb_u16_simd);
  }

  // ---- Yuv420p14 ---------------------------------------------------------

  fn solid_yuv420p14_frame(
    width: u32,
    height: u32,
    y: u16,
    u: u16,
    v: u16,
  ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    (
      std::vec![y; w * h],
      std::vec![u; cw * ch],
      std::vec![v; cw * ch],
    )
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p14_rgb_u8_only_gray_is_gray() {
    // 14-bit mid-gray: Y=UV=8192.
    let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
    let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p14_rgb_u16_only_native_depth_gray() {
    let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
    let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .unwrap();
    yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
      assert!(px[0] <= 16383);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p14_luma_downshifts_to_8bit() {
    // Y=8192 at 14 bits → 8192 >> (14 - 8) = 128.
    let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
    let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&l| l == 128));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p14_rgb_u8_and_u16_both_populated() {
    let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 16383, 8192, 8192);
    let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(rgb_u8.iter().all(|&c| c == 255));
    assert!(rgb_u16.iter().all(|&c| c == 16383));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p14_with_simd_false_matches_with_simd_true() {
    let (yp, up, vp) = solid_yuv420p14_frame(64, 16, 9600, 6400, 11200);
    let src = Yuv420p14Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

    let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
    let mut s_scalar = MixedSinker::<Yuv420p14>::new(64, 16)
      .with_simd(false)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    yuv420p14_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
    let mut s_simd = MixedSinker::<Yuv420p14>::new(64, 16)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    yuv420p14_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

    assert_eq!(rgb_scalar, rgb_simd);
    assert_eq!(rgb_u16_scalar, rgb_u16_simd);
  }

  // ---- P012 --------------------------------------------------------------
  //
  // Semi-planar 12-bit, high-bit-packed (samples in high 12 of each
  // u16). Mirrors the P010 test shape — UV interleaved, `value << 4`.

  fn solid_p012_frame(
    width: u32,
    height: u32,
    y_12bit: u16,
    u_12bit: u16,
    v_12bit: u16,
  ) -> (Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    // Shift into the high 12 bits (P012 packing).
    let y = std::vec![y_12bit << 4; w * h];
    let uv: Vec<u16> = (0..cw * ch)
      .flat_map(|_| [u_12bit << 4, v_12bit << 4])
      .collect();
    (y, uv)
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p012_rgb_u8_only_gray_is_gray() {
    let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
    let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<P012>::new(16, 8).with_rgb(&mut rgb).unwrap();
    p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p012_rgb_u16_only_native_depth_gray() {
    // Output is low-bit-packed 12-bit (yuv420p12le convention).
    let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
    let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<P012>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .unwrap();
    p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
      assert!(
        px[0] <= 4095,
        "output must stay within 12-bit low-packed range"
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p012_rgb_u8_and_u16_both_populated() {
    let (yp, uvp) = solid_p012_frame(16, 8, 4095, 2048, 2048);
    let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<P012>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(rgb_u8.iter().all(|&c| c == 255));
    assert!(rgb_u16.iter().all(|&c| c == 4095));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p012_luma_downshifts_to_8bit() {
    // Y=2048 at 12 bits, P012-packed (2048 << 4 = 0x8000). After >> 8,
    // the 8-bit luma is 0x80 = 128 — same accessor as P010 since both
    // store active bits in the high positions.
    let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
    let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<P012>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&l| l == 128));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p012_matches_yuv420p12_mixed_sinker_with_shifted_samples() {
    // Logical equivalence — same 12-bit samples fed through both
    // layouts must produce byte-identical u8 RGB.
    let w = 16u32;
    let h = 8u32;
    let y = 2400u16;
    let u = 1600u16;
    let v = 2800u16;

    let (yp_p12, up_p12, vp_p12) = solid_yuv420p12_frame(w, h, y, u, v);
    let src_p12 = Yuv420p12Frame::new(&yp_p12, &up_p12, &vp_p12, w, h, w, w / 2, w / 2);

    let (yp_p012, uvp_p012) = solid_p012_frame(w, h, y, u, v);
    let src_p012 = P012Frame::new(&yp_p012, &uvp_p012, w, h, w, w);

    let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_p012 = std::vec![0u8; (w * h * 3) as usize];
    let mut s_yuv = MixedSinker::<Yuv420p12>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_yuv)
      .unwrap();
    let mut s_p012 = MixedSinker::<P012>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_p012)
      .unwrap();
    yuv420p12_to(&src_p12, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
    p012_to(&src_p012, true, ColorMatrix::Bt709, &mut s_p012).unwrap();
    assert_eq!(rgb_yuv, rgb_p012);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p012_rgb_u16_too_short_returns_err() {
    let mut rgb = std::vec![0u16; 10];
    let err = MixedSinker::<P012>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .err()
      .unwrap();
    assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p012_with_simd_false_matches_with_simd_true() {
    let (yp, uvp) = solid_p012_frame(64, 16, 2400, 1600, 2800);
    let src = P012Frame::new(&yp, &uvp, 64, 16, 64, 64);

    let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
    let mut s_scalar = MixedSinker::<P012>::new(64, 16)
      .with_simd(false)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    p012_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
    let mut s_simd = MixedSinker::<P012>::new(64, 16)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    p012_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

    assert_eq!(rgb_scalar, rgb_simd);
    assert_eq!(rgb_u16_scalar, rgb_u16_simd);
  }

  // ---- Yuv420p16 ---------------------------------------------------------
  //
  // Planar 16-bit, full u16 range. Mid-gray is Y=UV=32768; full-range
  // white luma is 65535.

  fn solid_yuv420p16_frame(
    width: u32,
    height: u32,
    y: u16,
    u: u16,
    v: u16,
  ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    (
      std::vec![y; w * h],
      std::vec![u; cw * ch],
      std::vec![v; cw * ch],
    )
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p16_rgb_u8_only_gray_is_gray() {
    let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
    let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p16_rgb_u16_only_native_depth_gray() {
    let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
    let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .unwrap();
    yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(32768) <= 1, "got {px:?}");
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p16_rgb_u8_and_u16_both_populated() {
    // Full-range white: Y=65535, UV=32768.
    let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 65535, 32768, 32768);
    let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(rgb_u8.iter().all(|&c| c == 255));
    assert!(rgb_u16.iter().all(|&c| c == 65535));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p16_luma_downshifts_to_8bit() {
    // Y=32768 at 16 bits → 32768 >> (16 - 8) = 128.
    let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
    let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&l| l == 128));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p16_hsv_from_gray_is_zero_hue_zero_sat() {
    let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
    let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut h = std::vec![0xFFu8; 16 * 8];
    let mut s = std::vec![0xFFu8; 16 * 8];
    let mut v = std::vec![0xFFu8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p16_rgb_u16_too_short_returns_err() {
    let mut rgb = std::vec![0u16; 10];
    let err = MixedSinker::<Yuv420p16>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .err()
      .unwrap();
    assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p16_with_simd_false_matches_with_simd_true() {
    let (yp, up, vp) = solid_yuv420p16_frame(64, 16, 40000, 20000, 45000);
    let src = Yuv420p16Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

    let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
    let mut s_scalar = MixedSinker::<Yuv420p16>::new(64, 16)
      .with_simd(false)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    yuv420p16_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
    let mut s_simd = MixedSinker::<Yuv420p16>::new(64, 16)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    yuv420p16_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

    assert_eq!(rgb_scalar, rgb_simd);
    assert_eq!(rgb_u16_scalar, rgb_u16_simd);
  }

  // ---- P016 --------------------------------------------------------------

  fn solid_p016_frame(width: u32, height: u32, y: u16, u: u16, v: u16) -> (Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    // At 16 bits there's no shift — samples go in raw.
    let y_plane = std::vec![y; w * h];
    let uv: Vec<u16> = (0..cw * ch).flat_map(|_| [u, v]).collect();
    (y_plane, uv)
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p016_rgb_u8_only_gray_is_gray() {
    let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
    let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<P016>::new(16, 8).with_rgb(&mut rgb).unwrap();
    p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p016_rgb_u16_only_native_depth_gray() {
    let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
    let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<P016>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .unwrap();
    p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(32768) <= 1, "got {px:?}");
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p016_rgb_u8_and_u16_both_populated() {
    let (yp, uvp) = solid_p016_frame(16, 8, 65535, 32768, 32768);
    let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<P016>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(rgb_u8.iter().all(|&c| c == 255));
    assert!(rgb_u16.iter().all(|&c| c == 65535));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p016_luma_downshifts_to_8bit() {
    let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
    let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<P016>::new(16, 8)
      .with_luma(&mut luma)
      .unwrap();
    p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    assert!(luma.iter().all(|&l| l == 128));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p016_matches_yuv420p16_mixed_sinker() {
    // At 16 bits P016 and yuv420p16 are numerically identical —
    // the packing distinction degenerates when every bit is active.
    // Only the plane count / interleave layout differs.
    let w = 16u32;
    let h = 8u32;
    let y = 40000u16;
    let u = 20000u16;
    let v = 45000u16;

    let (yp_p16, up_p16, vp_p16) = solid_yuv420p16_frame(w, h, y, u, v);
    let src_p16 = Yuv420p16Frame::new(&yp_p16, &up_p16, &vp_p16, w, h, w, w / 2, w / 2);

    let (yp_p016, uvp_p016) = solid_p016_frame(w, h, y, u, v);
    let src_p016 = P016Frame::new(&yp_p016, &uvp_p016, w, h, w, w);

    let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_p016 = std::vec![0u8; (w * h * 3) as usize];
    let mut s_yuv = MixedSinker::<Yuv420p16>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_yuv)
      .unwrap();
    let mut s_p016 = MixedSinker::<P016>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_p016)
      .unwrap();
    yuv420p16_to(&src_p16, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
    p016_to(&src_p016, true, ColorMatrix::Bt709, &mut s_p016).unwrap();
    assert_eq!(rgb_yuv, rgb_p016);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p016_rgb_u16_too_short_returns_err() {
    let mut rgb = std::vec![0u16; 10];
    let err = MixedSinker::<P016>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .err()
      .unwrap();
    assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p016_with_simd_false_matches_with_simd_true() {
    let (yp, uvp) = solid_p016_frame(64, 16, 40000, 20000, 45000);
    let src = P016Frame::new(&yp, &uvp, 64, 16, 64, 64);

    let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
    let mut s_scalar = MixedSinker::<P016>::new(64, 16)
      .with_simd(false)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    p016_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
    let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
    let mut s_simd = MixedSinker::<P016>::new(64, 16)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    p016_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

    assert_eq!(rgb_scalar, rgb_simd);
    assert_eq!(rgb_u16_scalar, rgb_u16_simd);
  }

  // ---- Ship 6: sanity tests for new 4:2:2 / 4:4:4 formats ---------------

  fn solid_yuv422p_frame(
    width: u32,
    height: u32,
    y: u8,
    u: u8,
    v: u8,
  ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    // 4:2:2: chroma is half-width, FULL-height.
    (
      std::vec![y; w * h],
      std::vec![u; cw * h],
      std::vec![v; cw * h],
    )
  }

  fn solid_yuv444p_frame(
    width: u32,
    height: u32,
    y: u8,
    u: u8,
    v: u8,
  ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    (
      std::vec![y; w * h],
      std::vec![u; w * h],
      std::vec![v; w * h],
    )
  }

  fn solid_yuv422p_n_frame(
    width: u32,
    height: u32,
    y: u16,
    u: u16,
    v: u16,
  ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    (
      std::vec![y; w * h],
      std::vec![u; cw * h],
      std::vec![v; cw * h],
    )
  }

  fn solid_yuv444p_n_frame(
    width: u32,
    height: u32,
    y: u16,
    u: u16,
    v: u16,
  ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    (
      std::vec![y; w * h],
      std::vec![u; w * h],
      std::vec![v; w * h],
    )
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv422p_gray_to_gray() {
    let (yp, up, vp) = solid_yuv422p_frame(16, 8, 128, 128, 128);
    let src = Yuv422pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv422p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv444p_gray_to_gray() {
    let (yp, up, vp) = solid_yuv444p_frame(16, 8, 128, 128, 128);
    let src = Yuv444pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv444p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv422p10_gray_to_gray() {
    let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 512, 512, 512);
    let src = Yuv422p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv422p10>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv422p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv422p12_gray_to_gray() {
    let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 2048, 2048, 2048);
    let src = Yuv422p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv422p12>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv422p14_gray_to_gray() {
    let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 8192, 8192, 8192);
    let src = Yuv422p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv422p14>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv422p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv422p16_gray_to_gray_u16() {
    let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 32768, 32768, 32768);
    let src = Yuv422p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv422p16>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    yuv422p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb_u8.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(32768) <= 256);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv444p10_gray_to_gray() {
    let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 512, 512, 512);
    let src = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv444p10>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv444p12_gray_to_gray() {
    let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 2048, 2048, 2048);
    let src = Yuv444p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv444p12>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv444p14_gray_to_gray() {
    let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 8192, 8192, 8192);
    let src = Yuv444p14Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv444p14>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv444p16_gray_to_gray_u16() {
    let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 32768, 32768, 32768);
    let src = Yuv444p16Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv444p16>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    yuv444p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb_u8.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(32768) <= 256);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv422p_matches_yuv420p_luma_when_chroma_matches() {
    // 4:2:2 and 4:2:0 differ only in vertical chroma walk. With solid
    // chroma planes they must produce identical RGB output — this is
    // the whole reason Yuv422p reuses the yuv_420 row kernel.
    let w = 32u32;
    let h = 8u32;
    let (yp, up422, vp422) = solid_yuv422p_frame(w, h, 140, 100, 160);
    let src422 = Yuv422pFrame::new(&yp, &up422, &vp422, w, h, w, w / 2, w / 2);

    let (yp420, up420, vp420) = solid_yuv420p_frame(w, h, 140, 100, 160);
    let src420 = Yuv420pFrame::new(&yp420, &up420, &vp420, w, h, w, w / 2, w / 2);

    let mut rgb422 = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb420 = std::vec![0u8; (w * h * 3) as usize];
    let mut s422 = MixedSinker::<Yuv422p>::new(w as usize, h as usize)
      .with_rgb(&mut rgb422)
      .unwrap();
    let mut s420 = MixedSinker::<Yuv420p>::new(w as usize, h as usize)
      .with_rgb(&mut rgb420)
      .unwrap();
    yuv422p_to(&src422, true, ColorMatrix::Bt709, &mut s422).unwrap();
    yuv420p_to(&src420, true, ColorMatrix::Bt709, &mut s420).unwrap();
    assert_eq!(rgb422, rgb420);
  }

  // ---- 9-bit family + 4:4:0 family sanity tests ------------------------

  fn solid_yuv440p_frame(
    width: u32,
    height: u32,
    y: u8,
    u: u8,
    v: u8,
  ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let ch = (height as usize).div_ceil(2);
    (
      std::vec![y; w * h],
      std::vec![u; w * ch],
      std::vec![v; w * ch],
    )
  }

  fn solid_yuv440p_n_frame(
    width: u32,
    height: u32,
    y: u16,
    u: u16,
    v: u16,
  ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let ch = (height as usize).div_ceil(2);
    (
      std::vec![y; w * h],
      std::vec![u; w * ch],
      std::vec![v; w * ch],
    )
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p9_gray_to_gray() {
    let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
    // 4:2:0 chroma is w/2 × h/2; reuse the 4:2:2 helper's `cw * h` and
    // truncate to the 4:2:0 layout (cw = 8, ch = 4).
    let up = up[..8 * 4].to_vec();
    let vp = vp[..8 * 4].to_vec();
    let src = Yuv420p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p9>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv422p9_gray_to_gray() {
    let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
    let src = Yuv422p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv422p9>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv444p9_gray_to_gray() {
    let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 256, 256, 256);
    let src = Yuv444p9Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv444p9>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv440p_gray_to_gray() {
    let (yp, up, vp) = solid_yuv440p_frame(16, 8, 128, 128, 128);
    let src = Yuv440pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv440p>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv440p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv440p10_gray_to_gray() {
    let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 512, 512, 512);
    let src = Yuv440p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv440p10>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv440p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv440p12_gray_to_gray() {
    let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 2048, 2048, 2048);
    let src = Yuv440p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv440p12>::new(16, 8)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv440p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv440p_matches_yuv444p_when_chroma_constant_per_pair() {
    // 4:4:0 reuses the 4:4:4 row math; the only difference is the
    // walker reads chroma row r/2. With the same chroma value at every
    // (r, c), Yuv440p must produce identical RGB to Yuv444p with
    // duplicated chroma rows.
    let w = 32u32;
    let h = 8u32;
    let (yp, up440, vp440) = solid_yuv440p_frame(w, h, 140, 100, 160);
    let src440 = Yuv440pFrame::new(&yp, &up440, &vp440, w, h, w, w, w);

    // Yuv444p needs full-height chroma, so duplicate each of the 4 4:4:0
    // chroma rows into 2 rows.
    let mut up444 = std::vec::Vec::with_capacity((w * h) as usize);
    let mut vp444 = std::vec::Vec::with_capacity((w * h) as usize);
    for r in 0..h {
      let cr = (r / 2) as usize;
      let row_start = cr * w as usize;
      let row_end = row_start + w as usize;
      up444.extend_from_slice(&up440[row_start..row_end]);
      vp444.extend_from_slice(&vp440[row_start..row_end]);
    }
    let src444 = Yuv444pFrame::new(&yp, &up444, &vp444, w, h, w, w, w);

    let mut rgb440 = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb444 = std::vec![0u8; (w * h * 3) as usize];
    let mut s440 = MixedSinker::<Yuv440p>::new(w as usize, h as usize)
      .with_rgb(&mut rgb440)
      .unwrap();
    let mut s444 = MixedSinker::<Yuv444p>::new(w as usize, h as usize)
      .with_rgb(&mut rgb444)
      .unwrap();
    yuv440p_to(&src440, true, ColorMatrix::Bt709, &mut s440).unwrap();
    yuv444p_to(&src444, true, ColorMatrix::Bt709, &mut s444).unwrap();
    assert_eq!(rgb440, rgb444);
  }

  // ---- Walker-level SIMD-vs-scalar equivalence for 9-bit 4:2:x --------
  //
  // Per-arch row-kernel tests cover the BITS=9 path with non-neutral
  // chroma directly. These walker-level tests additionally pin the
  // public dispatcher behavior — Yuv420p9 / Yuv422p9 read through the
  // same `yuv_420p_n_to_rgb_*<9>` half-width kernels, so a backend
  // bug here would silently corrupt user output. Width 1922 forces
  // both the SIMD main loop and a scalar tail; chroma is non-neutral
  // and limited-range parameters are exercised below.

  fn pseudo_random_u16_low_n_bits(buf: &mut [u16], seed: u32, bits: u32) {
    let mask = ((1u32 << bits) - 1) as u16;
    let mut state = seed;
    for b in buf {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      *b = ((state >> 8) as u16) & mask;
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv420p9_walker_simd_matches_scalar_with_random_chroma() {
    let w = 1922u32; // forces tail handling on every backend
    let h = 4u32;
    let mut yp = std::vec![0u16; (w * h) as usize];
    let mut up = std::vec![0u16; ((w / 2) * (h / 2)) as usize];
    let mut vp = std::vec![0u16; ((w / 2) * (h / 2)) as usize];
    pseudo_random_u16_low_n_bits(&mut yp, 0x1111, 9);
    pseudo_random_u16_low_n_bits(&mut up, 0x2222, 9);
    pseudo_random_u16_low_n_bits(&mut vp, 0x3333, 9);
    let src = Yuv420p9Frame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);

    for &full_range in &[true, false] {
      let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
      let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

      let mut s_simd = MixedSinker::<Yuv420p9>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_simd)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16_simd)
        .unwrap();
      yuv420p9_to(&src, full_range, ColorMatrix::Bt709, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv420p9>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_scalar)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv420p9_to(&src, full_range, ColorMatrix::Bt709, &mut s_scalar).unwrap();

      assert_eq!(rgb_simd, rgb_scalar, "Yuv420p9 SIMD u8 ≠ scalar u8");
      assert_eq!(
        rgb_u16_simd, rgb_u16_scalar,
        "Yuv420p9 SIMD u16 ≠ scalar u16"
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn yuv422p9_walker_simd_matches_scalar_with_random_chroma() {
    let w = 1922u32;
    let h = 4u32;
    let mut yp = std::vec![0u16; (w * h) as usize];
    let mut up = std::vec![0u16; ((w / 2) * h) as usize];
    let mut vp = std::vec![0u16; ((w / 2) * h) as usize];
    pseudo_random_u16_low_n_bits(&mut yp, 0x4444, 9);
    pseudo_random_u16_low_n_bits(&mut up, 0x5555, 9);
    pseudo_random_u16_low_n_bits(&mut vp, 0x6666, 9);
    let src = Yuv422p9Frame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);

    for &full_range in &[true, false] {
      let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
      let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

      let mut s_simd = MixedSinker::<Yuv422p9>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_simd)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16_simd)
        .unwrap();
      yuv422p9_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv422p9>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_scalar)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv422p9_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_scalar).unwrap();

      assert_eq!(rgb_simd, rgb_scalar, "Yuv422p9 SIMD u8 ≠ scalar u8");
      assert_eq!(
        rgb_u16_simd, rgb_u16_scalar,
        "Yuv422p9 SIMD u16 ≠ scalar u16"
      );
    }
  }

  // ---- P210 / P212 / P216 / P410 / P412 / P416 sanity tests --------------

  /// 4:2:2 P-family solid frame helper. UV is `width` u16 elements per
  /// row, **full-height** chroma. All samples are high-bit-packed
  /// (shifted left by `16 - bits`).
  fn solid_p2x0_frame(
    width: u32,
    height: u32,
    bits: u32,
    y_value: u16,
    u_value: u16,
    v_value: u16,
  ) -> (Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let shift = 16 - bits;
    let y = std::vec![y_value << shift; w * h];
    // 4:2:2: full-height chroma, half-width × 2 elements per pair.
    let uv: Vec<u16> = (0..cw * h)
      .flat_map(|_| [u_value << shift, v_value << shift])
      .collect();
    (y, uv)
  }

  /// 4:4:4 P-family solid frame helper. UV is `2 * width` u16 elements
  /// per row, **full-height** chroma (one pair per pixel).
  fn solid_p4x0_frame(
    width: u32,
    height: u32,
    bits: u32,
    y_value: u16,
    u_value: u16,
    v_value: u16,
  ) -> (Vec<u16>, Vec<u16>) {
    let w = width as usize;
    let h = height as usize;
    let shift = 16 - bits;
    let y = std::vec![y_value << shift; w * h];
    // 4:4:4: full-height × full-width × 2 elements per pair.
    let uv: Vec<u16> = (0..w * h)
      .flat_map(|_| [u_value << shift, v_value << shift])
      .collect();
    (y, uv)
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p210_gray_to_gray() {
    let (yp, uvp) = solid_p2x0_frame(16, 8, 10, 512, 512, 512);
    let src = P210Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<P210>::new(16, 8).with_rgb(&mut rgb).unwrap();
    p210_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p212_gray_to_gray() {
    let (yp, uvp) = solid_p2x0_frame(16, 8, 12, 2048, 2048, 2048);
    let src = P212Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<P212>::new(16, 8).with_rgb(&mut rgb).unwrap();
    p212_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p216_gray_to_gray_u16() {
    let (yp, uvp) = solid_p2x0_frame(16, 8, 16, 32768, 32768, 32768);
    let src = P216Frame::new(&yp, &uvp, 16, 8, 16, 16);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<P216>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    p216_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

    for px in rgb_u8.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(32768) <= 256);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p410_gray_to_gray() {
    // 4:4:4: uv_stride = 2 * width = 32 (16 pairs × 2 elements).
    let (yp, uvp) = solid_p4x0_frame(16, 8, 10, 512, 512, 512);
    let src = P410Frame::new(&yp, &uvp, 16, 8, 16, 32);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<P410>::new(16, 8).with_rgb(&mut rgb).unwrap();
    p410_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p412_gray_to_gray() {
    let (yp, uvp) = solid_p4x0_frame(16, 8, 12, 2048, 2048, 2048);
    let src = P412Frame::new(&yp, &uvp, 16, 8, 16, 32);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<P412>::new(16, 8).with_rgb(&mut rgb).unwrap();
    p412_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p416_gray_to_gray_u16() {
    let (yp, uvp) = solid_p4x0_frame(16, 8, 16, 32768, 32768, 32768);
    let src = P416Frame::new(&yp, &uvp, 16, 8, 16, 32);

    let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
    let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
    let mut sink = MixedSinker::<P416>::new(16, 8)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    p416_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

    for px in rgb_u8.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(32768) <= 256);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  // ---- Walker-level SIMD-vs-scalar equivalence for P410 (4:4:4 Pn) ------
  //
  // P410 is the only new format in Ship 7 that ships a genuinely new
  // SIMD kernel family (`p_n_444_to_rgb_*<BITS>`). Validate the
  // walker against scalar with non-neutral chroma and tail widths.
  // P210/P212/P216 reuse 4:2:0 P-family kernels (already covered by
  // earlier ships' tests).

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p410_walker_simd_matches_scalar_with_random_chroma() {
    let w = 1922u32; // forces tail handling on every backend
    let h = 4u32;
    let mut yp = std::vec![0u16; (w * h) as usize];
    let mut uvp = std::vec![0u16; (2 * w * h) as usize];

    // Seed pseudo-random samples in the high 10 bits.
    let mut state: u32 = 0x1111_2222;
    for s in yp.iter_mut().chain(uvp.iter_mut()) {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      *s = ((state & 0x3FF) as u16) << 6;
    }

    let src = P410Frame::new(&yp, &uvp, w, h, w, 2 * w);

    for &full_range in &[true, false] {
      let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
      let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

      let mut s_simd = MixedSinker::<P410>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_simd)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16_simd)
        .unwrap();
      p410_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<P410>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_scalar)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      p410_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_scalar).unwrap();

      assert_eq!(rgb_simd, rgb_scalar, "P410 SIMD u8 ≠ scalar u8");
      assert_eq!(rgb_u16_simd, rgb_u16_scalar, "P410 SIMD u16 ≠ scalar u16");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p412_walker_simd_matches_scalar_with_random_chroma() {
    let w = 1922u32;
    let h = 4u32;
    let mut yp = std::vec![0u16; (w * h) as usize];
    let mut uvp = std::vec![0u16; (2 * w * h) as usize];

    let mut state: u32 = 0x3333_4444;
    for s in yp.iter_mut().chain(uvp.iter_mut()) {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      *s = ((state & 0xFFF) as u16) << 4;
    }

    let src = P412Frame::new(&yp, &uvp, w, h, w, 2 * w);

    for &full_range in &[true, false] {
      let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];

      let mut s_simd = MixedSinker::<P412>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_simd)
        .unwrap();
      p412_to(&src, full_range, ColorMatrix::Bt709, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<P412>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      p412_to(&src, full_range, ColorMatrix::Bt709, &mut s_scalar).unwrap();

      assert_eq!(rgb_simd, rgb_scalar, "P412 SIMD u8 ≠ scalar u8");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn p416_walker_simd_matches_scalar_with_random_chroma() {
    let w = 1922u32;
    let h = 4u32;
    let mut yp = std::vec![0u16; (w * h) as usize];
    let mut uvp = std::vec![0u16; (2 * w * h) as usize];

    let mut state: u32 = 0x5555_6666;
    for s in yp.iter_mut().chain(uvp.iter_mut()) {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      *s = state as u16;
    }

    let src = P416Frame::new(&yp, &uvp, w, h, w, 2 * w);

    for &full_range in &[true, false] {
      let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
      let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
      let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

      let mut s_simd = MixedSinker::<P416>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_simd)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16_simd)
        .unwrap();
      p416_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<P416>::new(w as usize, h as usize)
        .with_rgb(&mut rgb_scalar)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      p416_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_scalar).unwrap();

      assert_eq!(rgb_simd, rgb_scalar, "P416 SIMD u8 ≠ scalar u8");
      assert_eq!(rgb_u16_simd, rgb_u16_scalar, "P416 SIMD u16 ≠ scalar u16");
    }
  }

  // ---- Bayer + Bayer16 MixedSinker integration tests ----------------------

  /// Build a solid-channel RGGB Bayer plane (8-bit) so every R site
  /// holds `r`, every B site holds `b`, and both G sites hold `g`.
  fn solid_rggb8(width: u32, height: u32, r: u8, g: u8, b: u8) -> std::vec::Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut data = std::vec![0u8; w * h];
    for y in 0..h {
      for x in 0..w {
        data[y * w + x] = match (y & 1, x & 1) {
          (0, 0) => r,
          (0, 1) => g,
          (1, 0) => g,
          (1, 1) => b,
          _ => unreachable!(),
        };
      }
    }
    data
  }

  /// Build a 12-bit low-packed RGGB Bayer plane.
  fn solid_rggb12(width: u32, height: u32, r: u16, g: u16, b: u16) -> std::vec::Vec<u16> {
    let w = width as usize;
    let h = height as usize;
    let mut data = std::vec![0u16; w * h];
    for y in 0..h {
      for x in 0..w {
        let v = match (y & 1, x & 1) {
          (0, 0) => r,
          (0, 1) => g,
          (1, 0) => g,
          (1, 1) => b,
          _ => unreachable!(),
        };
        data[y * w + x] = v;
      }
    }
    data
  }

  #[test]
  fn bayer_mixed_sinker_with_rgb_red_interior() {
    use crate::{
      frame::BayerFrame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
    };
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb8(w, h, 255, 0, 0);
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
      .with_rgb(&mut rgb)
      .unwrap();
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    // Interior should be exactly red.
    let wu = w as usize;
    for y in 0..(h as usize) {
      for x in 0..wu {
        let i = (y * wu + x) * 3;
        assert_eq!(rgb[i], 255, "px ({x},{y}) R");
        assert_eq!(rgb[i + 1], 0, "px ({x},{y}) G");
        assert_eq!(rgb[i + 2], 0, "px ({x},{y}) B");
      }
    }
  }

  #[test]
  fn bayer_mixed_sinker_with_luma_uniform_byte() {
    use crate::{
      frame::BayerFrame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
    };
    // Uniform byte → uniform RGB → uniform luma at the same value.
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![200u8; (w * h) as usize];
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut luma = std::vec![0u8; (w * h) as usize];
    let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
      .with_luma(&mut luma)
      .unwrap();
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    // BT.709 luma of (200, 200, 200) = 200 (within 1 LSB rounding).
    for &y in &luma {
      assert!((y as i32 - 200).abs() <= 1, "luma got {y}");
    }
  }

  #[test]
  fn bayer_mixed_sinker_with_hsv_solid_red_interior() {
    use crate::{
      frame::BayerFrame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
    };
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb8(w, h, 255, 0, 0);
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut hh = std::vec![0u8; (w * h) as usize];
    let mut ss = std::vec![0u8; (w * h) as usize];
    let mut vv = std::vec![0u8; (w * h) as usize];
    let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    // Pure red at interior → H = 0 (red), S = 255 (max), V = 255.
    let wu = w as usize;
    for y in 0..(h as usize) {
      for x in 0..wu {
        let i = y * wu + x;
        assert_eq!(hh[i], 0, "px ({x},{y}) H");
        assert_eq!(ss[i], 255, "px ({x},{y}) S");
        assert_eq!(vv[i], 255, "px ({x},{y}) V");
      }
    }
  }

  #[test]
  fn bayer16_mixed_sinker_with_rgb_red_interior() {
    use crate::{
      frame::Bayer12Frame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
    };
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb12(w, h, 4095, 0, 0);
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
      .with_rgb(&mut rgb)
      .unwrap();
    bayer16_to::<12, _>(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    let wu = w as usize;
    for y in 0..(h as usize) {
      for x in 0..wu {
        let i = (y * wu + x) * 3;
        assert_eq!(rgb[i], 255, "px ({x},{y}) R");
        assert_eq!(rgb[i + 1], 0, "px ({x},{y}) G");
        assert_eq!(rgb[i + 2], 0, "px ({x},{y}) B");
      }
    }
  }

  #[test]
  fn bayer16_mixed_sinker_with_rgb_u16_red_interior() {
    use crate::{
      frame::Bayer12Frame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
    };
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb12(w, h, 4095, 0, 0);
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u16; (w * h * 3) as usize];
    let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
      .with_rgb_u16(&mut rgb)
      .unwrap();
    bayer16_to::<12, _>(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    // Low-packed 12-bit white = 4095 at interior.
    let wu = w as usize;
    for y in 0..(h as usize) {
      for x in 0..wu {
        let i = (y * wu + x) * 3;
        assert_eq!(rgb[i], 4095, "px ({x},{y}) R");
        assert_eq!(rgb[i + 1], 0, "px ({x},{y}) G");
        assert_eq!(rgb[i + 2], 0, "px ({x},{y}) B");
      }
    }
  }

  #[test]
  fn bayer16_mixed_sinker_dual_rgb_and_rgb_u16() {
    use crate::{
      frame::Bayer12Frame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
    };
    // Both u8 RGB and u16 RGB attached — both kernels run.
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb12(w, h, 4095, 0, 0);
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb_u8 = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16 = std::vec![0u16; (w * h * 3) as usize];
    let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    bayer16_to::<12, _>(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    let wu = w as usize;
    for y in 0..(h as usize) {
      for x in 0..wu {
        let i = (y * wu + x) * 3;
        assert_eq!(rgb_u8[i], 255);
        assert_eq!(rgb_u16[i], 4095);
      }
    }
  }

  #[test]
  fn bayer_mixed_sinker_returns_row_shape_mismatch_on_bad_above() {
    use crate::raw::{BayerDemosaic, BayerPattern, BayerRow};
    let mut rgb = std::vec![0u8; 8 * 6 * 3];
    let mut sinker = MixedSinker::<Bayer>::new(8, 6).with_rgb(&mut rgb).unwrap();
    sinker.begin_frame(8, 6).unwrap();
    let mid = std::vec![0u8; 8];
    let below = std::vec![0u8; 8];
    let bad_above = std::vec![0u8; 7]; // wrong length
    let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let row = BayerRow::new(
      &bad_above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      m,
    );
    let err = sinker.process(row).unwrap_err();
    assert!(matches!(
      err,
      MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BayerAbove,
        expected: 8,
        actual: 7,
        ..
      }
    ));
  }

  #[test]
  fn bayer16_mixed_sinker_returns_row_shape_mismatch_on_bad_mid() {
    use crate::raw::{BayerDemosaic, BayerPattern, BayerRow16};
    let mut rgb = std::vec![0u8; 8 * 6 * 3];
    let mut sinker = MixedSinker::<Bayer16<12>>::new(8, 6)
      .with_rgb(&mut rgb)
      .unwrap();
    sinker.begin_frame(8, 6).unwrap();
    let above = std::vec![0u16; 8];
    let bad_mid = std::vec![0u16; 7]; // wrong length
    let below = std::vec![0u16; 8];
    let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let row = BayerRow16::<12>::new(
      &above,
      &bad_mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      m,
    );
    let err = sinker.process(row).unwrap_err();
    assert!(matches!(
      err,
      MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bayer16Mid,
        expected: 8,
        actual: 7,
        ..
      }
    ));
  }

  // ---- Bayer luma-coefficients tests --------------------------------------
  //
  // Cover the gap that earlier `bayer_mixed_sinker_with_luma_uniform_byte`
  // missed: every coefficient set agrees on gray, so a hard-coded BT.709
  // path could go undetected. The non-gray cases below force the rows
  // apart — solid red goes through `cr` only, so each variant produces a
  // distinct luma value.

  /// Resolve a [`LumaCoefficients`] preset and run a solid-red 8-bit
  /// Bayer frame through it; return the `cr` actually applied.
  fn bayer8_solid_red_luma(coeffs: LumaCoefficients) -> u8 {
    use crate::{
      frame::BayerFrame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
    };
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb8(w, h, 255, 0, 0);
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut luma = std::vec![0u8; (w * h) as usize];
    let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_coefficients(coeffs);
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    let center = luma[(h as usize / 2) * (w as usize) + (w as usize / 2)];
    for (i, &y) in luma.iter().enumerate() {
      assert_eq!(
        y, center,
        "luma not uniform at idx {i}: {y} vs center {center}"
      );
    }
    center
  }

  #[test]
  fn bayer_with_luma_coefficients_solid_red_differs_by_preset() {
    // Solid red after demosaic is `(255, 0, 0)` everywhere
    // (`bayer_mixed_sinker_with_rgb_red_interior` proves this).
    // Luma reduces to `(cr * 255 + 128) >> 8` for each preset, so
    // each coefficient set must produce a different value. The
    // hard-coded BT.709 bug Codex flagged would make these all 54.
    let bt709 = bayer8_solid_red_luma(LumaCoefficients::Bt709);
    let bt2020 = bayer8_solid_red_luma(LumaCoefficients::Bt2020);
    let bt601 = bayer8_solid_red_luma(LumaCoefficients::Bt601);
    let dcip3 = bayer8_solid_red_luma(LumaCoefficients::DciP3);
    let aces = bayer8_solid_red_luma(LumaCoefficients::AcesAp1);

    assert_eq!(bt709, 54, "BT.709 red luma");
    assert_eq!(bt2020, 67, "BT.2020 red luma");
    assert_eq!(bt601, 77, "BT.601 red luma");
    assert_eq!(dcip3, 59, "DCI-P3 red luma");
    assert_eq!(aces, 70, "ACES AP1 red luma");

    // Distinct values guard against silent collapse to the default.
    let mut all = std::vec![bt709, bt2020, bt601, dcip3, aces];
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 5, "presets collapsed to fewer values: {all:?}");
  }

  #[test]
  fn bayer_with_luma_coefficients_custom_round_trips_to_q8() {
    // Custom weights `(1.0, 0.0, 0.0)` → Q8 `(256, 0, 0)`. Solid red
    // 255 then reduces to `(256 * 255 + 128) >> 8 = 255` (clamped).
    let red = bayer8_solid_red_luma(LumaCoefficients::Custom {
      r: 1.0,
      g: 0.0,
      b: 0.0,
    });
    assert_eq!(red, 255, "Custom (1.0, 0.0, 0.0) on red 255 → 255");
  }

  #[test]
  fn bayer_with_luma_coefficients_default_is_bt709() {
    // No `with_luma_coefficients` call → default (BT.709). Same red
    // input must produce the BT.709 value (54). This pins the
    // public default so a future refactor can't silently change it.
    use crate::{
      frame::BayerFrame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
    };
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb8(w, h, 255, 0, 0);
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut luma = std::vec![0u8; (w * h) as usize];
    let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
      .with_luma(&mut luma)
      .unwrap();
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    for (i, &y) in luma.iter().enumerate() {
      assert_eq!(y, 54, "default red luma at idx {i}");
    }
    assert_eq!(LumaCoefficients::default(), LumaCoefficients::Bt709);
  }

  #[test]
  fn bayer_with_luma_coefficients_uniform_gray_invariant() {
    // The reverse of the above: gray content *must* be invariant
    // under any preset (this is the property the original
    // `*_with_luma_uniform_byte` test relied on, and the reason
    // the hard-coded BT.709 bug was invisible there).
    use crate::{
      frame::BayerFrame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
    };
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![200u8; (w * h) as usize];
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let presets = [
      LumaCoefficients::Bt709,
      LumaCoefficients::Bt2020,
      LumaCoefficients::Bt601,
      LumaCoefficients::DciP3,
      LumaCoefficients::AcesAp1,
    ];
    for preset in presets {
      let mut luma = std::vec![0u8; (w * h) as usize];
      let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_coefficients(preset);
      bayer_to(
        &frame,
        BayerPattern::Rggb,
        BayerDemosaic::Bilinear,
        WhiteBalance::neutral(),
        ColorCorrectionMatrix::identity(),
        &mut sinker,
      )
      .unwrap();
      for &y in &luma {
        assert!(
          (y as i32 - 200).abs() <= 1,
          "{preset:?} on gray 200 → {y} (expected ~200)"
        );
      }
    }
  }

  #[test]
  fn bayer16_with_luma_coefficients_solid_red_differs_by_preset() {
    // Mirror of the 8-bit test for the high-bit-depth path
    // (`MixedSinker<Bayer16<BITS>>`). 12-bit white = 4095 →
    // demosaic produces `(255, 0, 0)` u8 RGB after CCM identity
    // and right-shift to u8 (the bayer16→u8 path reduces samples
    // before the luma kernel).
    use crate::{
      frame::Bayer12Frame,
      raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
    };
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb12(w, h, 4095, 0, 0);
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();

    let run = |coeffs: LumaCoefficients| -> u8 {
      let mut luma = std::vec![0u8; (w * h) as usize];
      let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_coefficients(coeffs);
      bayer16_to(
        &frame,
        BayerPattern::Rggb,
        BayerDemosaic::Bilinear,
        WhiteBalance::neutral(),
        ColorCorrectionMatrix::identity(),
        &mut sinker,
      )
      .unwrap();
      let center = luma[(h as usize / 2) * (w as usize) + (w as usize / 2)];
      for (i, &y) in luma.iter().enumerate() {
        assert_eq!(y, center, "luma not uniform at idx {i}");
      }
      center
    };

    let bt709 = run(LumaCoefficients::Bt709);
    let bt2020 = run(LumaCoefficients::Bt2020);
    let bt601 = run(LumaCoefficients::Bt601);
    let dcip3 = run(LumaCoefficients::DciP3);
    let aces = run(LumaCoefficients::AcesAp1);

    assert_eq!(bt709, 54, "BT.709 red luma (Bayer16<12>)");
    assert_eq!(bt2020, 67, "BT.2020 red luma (Bayer16<12>)");
    assert_eq!(bt601, 77, "BT.601 red luma (Bayer16<12>)");
    assert_eq!(dcip3, 59, "DCI-P3 red luma (Bayer16<12>)");
    assert_eq!(aces, 70, "ACES AP1 red luma (Bayer16<12>)");

    let mut all = std::vec![bt709, bt2020, bt601, dcip3, aces];
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 5, "Bayer16 presets collapsed: {all:?}");
  }

  #[test]
  fn luma_coefficients_to_q8_presets_sum_to_256() {
    // Round-to-nearest of the published weights for each preset
    // must still sum to exactly 256 — the rgb_row_to_luma_row
    // kernel divides by 256 implicitly via `>> 8`, so any preset
    // that drifts from 256 produces a brightness-scaled luma plane.
    for preset in [
      LumaCoefficients::Bt709,
      LumaCoefficients::Bt2020,
      LumaCoefficients::Bt601,
      LumaCoefficients::DciP3,
      LumaCoefficients::AcesAp1,
    ] {
      let (cr, cg, cb) = preset.to_q8();
      assert_eq!(cr + cg + cb, 256, "{preset:?} Q8 weights don't sum to 256");
    }
  }
}
