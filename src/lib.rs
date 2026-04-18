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

#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod row;
pub mod sinker;

#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod yuv;

/// A per-row sink for color-converted pixel data.
///
/// Consumers ([`sinker::MixedSinker`], the application's own reducers,
/// etc.) implement this once per source format they want to accept. The
/// source kernel calls [`Self::process`] for every output row of
/// the frame.
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
pub trait PixelSink {
  /// The shape of one input unit chosen by the per-format subtrait —
  /// e.g. [`yuv::Yuv420pRow`] for YUV 4:2:0, one row at a time.
  type Input<'a>;

  /// Consume one input unit. Called by the kernel once per unit (one
  /// row, for the row-granular kernels v0.1 ships). Input borrows may
  /// be invalidated after the call returns — implementations must not
  /// retain them.
  fn process(&mut self, input: Self::Input<'_>);
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

pub(crate) mod sealed {
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
