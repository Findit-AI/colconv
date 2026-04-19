//! SIMD-dispatched per-row color-conversion kernels for the FFmpeg
//! `AVPixelFormat` space.
//!
//! # Design
//!
//! Every source pixel format has its own kernel (`yuv420p_to`,
//! `nv12_to`, `bgr24_to`, …) that walks the source row by row and hands
//! each row to a caller-supplied [`PixelSink`]. The Sink decides what
//! to derive — luma only, RGB only, HSV only, all three, or something
//! custom — and writes into whatever buffers it owns.
//!
//! The row the Sink receives (`Self::Input<'_>`) has a shape that
//! reflects the source format: [`yuv::Yuv420pRow`] carries Y / U / V
//! slices plus matrix / range metadata; future packed‑RGB row types
//! (`Rgb24Row`, `Bgr24Row`) will carry a single packed slice; etc.
//! Each source family declares a subtrait
//! (`Yuv420pSink: PixelSink<Input<'_> = Yuv420pRow<'_>>`) so kernel
//! signatures stay sharp.
//!
//! For the common case — "give me RGB / Luma / HSV or any subset" —
//! the crate ships [`sinker::MixedSinker`], configured via
//! [`with_rgb`](sinker::MixedSinker::with_rgb) /
//! [`with_luma`](sinker::MixedSinker::with_luma) /
//! [`with_hsv`](sinker::MixedSinker::with_hsv) to select which channels
//! to derive.
//!
//! The crate design also follows a per-format expansion plan with
//! defined implementation priority tiers for the conversion kernels.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

use derive_more::IsVariant;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

pub mod frame;

pub mod row;
pub mod sinker;
pub mod yuv;

/// A per-row sink for color-converted pixel data.
///
/// Consumers ([`sinker::MixedSinker`], the application's own reducers,
/// etc.) implement this once per source format they want to accept. The
/// source kernel calls [`Self::process`] for every output row of the
/// frame and may propagate the sink's error back to the caller.
///
/// # Input type
///
/// Each source family pins the associated `Input` to a concrete row
/// struct via a subtrait. For example, [`yuv::Yuv420pSink`] requires
/// `for<'a> PixelSink<Input<'a> = yuv::Yuv420pRow<'a>>`. A single
/// concrete sink type can therefore only consume one source format —
/// which is intentional. To handle multiple sources, use the
/// `SourceFormat` type-parameter pattern demonstrated by
/// [`sinker::MixedSinker`].
///
/// # Why fallible (`Result<(), Self::Error>`)
///
/// Both [`begin_frame`](Self::begin_frame) and [`process`](Self::process)
/// return `Result<(), Self::Error>` so the crate can run on
/// panic-sensitive targets — `#![no_std]` with `panic = "abort"`,
/// embedded RTOS codebases that lint against `unwrap`/`panic!`, and
/// similar environments where a single bad frame must not crash the
/// process. This mirrors the `embedded-hal` / `embedded-graphics`
/// convention for per-pixel and per-row sinks.
///
/// Sinks that genuinely cannot fail (pure compute — histogram, hash,
/// …) declare `type Error = core::convert::Infallible;` and return
/// `Ok(())` unconditionally. LLVM strips the result wrapping away at
/// the `Infallible` call sites, so there's no hot-path overhead
/// versus a `()` return.
///
/// # Error philosophy
///
/// - **Input geometry errors** (malformed source plane, odd width)
///   surface at [`frame::Yuv420pFrame::try_new`] /
///   [`frame::Nv12Frame::try_new`], not in the sink.
/// - **Sink configuration errors** (undersized buffer) surface at
///   sink construction — `MixedSinker::with_rgb` etc. return
///   `Result<Self, MixedSinkerError>` so a short buffer never reaches
///   the walker.
/// - **Per-frame setup errors** (frame dims don't match the sink's
///   configuration) surface at [`begin_frame`](Self::begin_frame),
///   before the first row is processed — so the caller's buffers are
///   never partially mutated before the error is returned.
/// - **Runtime sink errors** (I/O failure, GPU upload, …) surface
///   naturally as `Err` returns from `process`. The walker short-
///   circuits on the first error, so no wasted work on subsequent
///   rows.
///
/// # Example: an Infallible counting sink
///
/// ```ignore
/// use core::convert::Infallible;
/// use colconv::{PixelSink, yuv::Yuv420pRow};
///
/// struct RowCounter(usize);
/// impl PixelSink for RowCounter {
///     type Input<'a> = Yuv420pRow<'a>;
///     type Error = Infallible;
///     fn process(&mut self, _row: Yuv420pRow<'_>) -> Result<(), Infallible> {
///         self.0 += 1;
///         Ok(())
///     }
/// }
/// ```
///
/// # Example: a fallible file-writing sink
///
/// ```ignore
/// use std::io::{self, BufWriter, Write};
/// use colconv::{PixelSink, yuv::Yuv420pRow};
///
/// struct FileSink { w: BufWriter<std::fs::File> }
///
/// impl PixelSink for FileSink {
///     type Input<'a> = Yuv420pRow<'a>;
///     type Error = io::Error;
///     fn process(&mut self, row: Yuv420pRow<'_>) -> io::Result<()> {
///         self.w.write_all(row.y())
///     }
/// }
/// ```
///
/// The walker returns `Result<(), io::Error>`; `?` propagates
/// cleanly through the caller's code.
pub trait PixelSink {
  /// The shape of one input unit chosen by the per-format subtrait —
  /// e.g. [`yuv::Yuv420pRow`] for YUV 4:2:0, one row at a time.
  type Input<'a>;

  /// The error type surfaced by this sink. Use
  /// [`core::convert::Infallible`] for sinks that can't fail — the
  /// compiler eliminates the `Result` branching at the call sites.
  type Error;

  /// Called by the walker exactly once per frame, **before** any
  /// [`process`](Self::process) call, with the source frame's
  /// dimensions.
  ///
  /// Sinks that care about geometry — buffer-backed sinks like
  /// [`sinker::MixedSinker`] — override this to validate the frame
  /// against their configured dimensions *before* any row is written.
  /// This catches the two stale-state failure modes that a per-row
  /// `idx < height` guard can't: shorter frames that would silently
  /// leave bottom rows unwritten, and taller frames that would
  /// partially mutate the output before failing halfway through.
  ///
  /// Default is `Ok(())`, so pure-computation sinks (histogram, hash,
  /// etc.) that don't care about source geometry don't need to
  /// override.
  ///
  /// Any `Err` returned here is propagated by the walker before any
  /// row is processed.
  #[allow(unused_variables)]
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    Ok(())
  }

  /// Consume one input unit. Called by the kernel once per unit (one
  /// row, for the row-granular kernels v0.1 ships). Input borrows may
  /// be invalidated after the call returns — implementations must not
  /// retain them.
  ///
  /// Returns `Err` to short-circuit the walker: on the first `Err`,
  /// the walker returns immediately without processing further rows.
  fn process(&mut self, input: Self::Input<'_>) -> Result<(), Self::Error>;
}

/// YUV → RGB conversion matrix.
///
/// Read from `AVFrame.colorspace` when decoding via FFmpeg. Each
/// variant maps to one or more `AVCOL_SPC_*` values:
///
/// | `AVCOL_SPC_*`                    | Variant      | Note                                     |
/// |---                               |---           |---                                       |
/// | `BT709`                          | `Bt709`      | HDTV default                             |
/// | `BT2020_NCL`                     | `Bt2020Ncl`  | UHDTV / HDR10                            |
/// | `SMPTE170M` (NTSC SD)            | `Bt601`      | alias — identical coefficients to BT.601 |
/// | `BT470BG` (PAL/SECAM SD)         | `Bt601`      | alias — identical coefficients to BT.601 |
/// | `SMPTE240M`                      | `Smpte240m`  | legacy HD                                |
/// | `FCC`                            | `Fcc`        | legacy NTSC variant                      |
/// | `YCGCO`                          | `YCgCo`      | screen-codec intra / alpha paths (H.273) |
///
/// For `AVCOL_SPC_UNSPECIFIED` (value `2`), FFmpeg's convention is
/// `Bt709` for sources with `height >= 720` and `Bt601` otherwise —
/// the caller should apply that rule and pick accordingly.
///
/// **Not covered** (rarely encountered in video-indexing workloads):
/// `BT2020_CL` (constant luminance, needs a non-linear math path),
/// `ICTCP` (Dolby Vision P5 — separate decode path anyway),
/// `SMPTE2085`, `IPT_C2`, `CHROMA_DERIVED_NCL/CL`, and
/// `YCGCO_RE`/`YCGCO_RO`. The enum is `#[non_exhaustive]` so variants
/// can be added without a breaking change when a real use case arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant)]
#[non_exhaustive]
pub enum ColorMatrix {
  /// ITU-R BT.601 (SDTV). `R' = Y + 1.402·(V - 128)` etc. in 8-bit space.
  /// Also the correct choice for `AVCOL_SPC_SMPTE170M` (NTSC) and
  /// `AVCOL_SPC_BT470BG` (PAL/SECAM) — all three share identical
  /// coefficients.
  Bt601,
  /// ITU-R BT.709 (HDTV).
  Bt709,
  /// ITU-R BT.2020 non-constant-luminance (UHDTV / HDR10).
  Bt2020Ncl,
  /// SMPTE 240M (legacy 1990s HDTV).
  Smpte240m,
  /// FCC CFR 47 §73.682 (legacy NTSC, very close to BT.601 numerically).
  Fcc,
  /// YCgCo per ITU-T H.273 MatrixCoefficients = 8.
  ///
  /// U plane carries Cg (chroma-green), V plane carries Co
  /// (chroma-orange). Encountered in screen-codec workflows,
  /// VP9/AV1 intra-frame paths, and some WebRTC streams.
  ///
  /// Inverse transform (Co, Cg de-biased against 128):
  /// `R = Y - Cg + Co`, `G = Y + Cg`, `B = Y - Cg - Co`.
  YCgCo,
}

/// Sealed marker trait identifying a source pixel format.
///
/// Used as a type parameter on sinks that specialize per source —
/// [`sinker::MixedSinker<'_, F>`] for example. Implementors are the
/// zero-sized markers in [`yuv`], [`rgb`](sinker) etc.
pub trait SourceFormat: sealed::Sealed {}

/// Internal module implementing the sealed‑trait pattern for
/// [`SourceFormat`]. External crates cannot name `Sealed`, so they
/// cannot implement [`SourceFormat`] themselves — the variant list
/// stays closed.
pub(crate) mod sealed {
  /// Crate‑private marker trait used to prevent downstream
  /// implementations of [`super::SourceFormat`].
  pub trait Sealed {}
}

/// The three output planes for HSV, bundled so `MixedSinker` stores a
/// single `Option<HsvBuffers>` rather than three independent options.
#[cfg(any(feature = "std", feature = "alloc"))]
struct HsvBuffers<'a> {
  h: &'a mut [u8],
  s: &'a mut [u8],
  v: &'a mut [u8],
}
