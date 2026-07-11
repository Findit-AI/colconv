//! SIMD-dispatched color-conversion kernels for the FFmpeg `AVPixelFormat`
//! space, exposed through honest, independently-usable tiers.
//!
//! # Tier 0: [`Convert`] — the golden one-call decode
//!
//! `Convert::from(&frame)` decodes a validated source frame to one or more
//! output buffers with zero redundant parameters: the dimensions come from the
//! frame, the colorimetry from a single [`ColorSpec`], and each per-format walk
//! knob is *derived* from that spec (overridable). Attach outputs —
//! [`rgb`](Convert::rgb) / [`rgba`](Convert::rgba) / [`luma`](Convert::luma) /
//! [`hsv`](Convert::hsv) — plus an optional area or filtered
//! [`resize`](Convert::resize) with infallible consuming setters;
//! [`Convert::run`] is the only fallible call. Failures surface as [`Error`], an
//! alias of [`MixedSinkerError`](sinker::MixedSinkerError).
//!
//! ```
//! # #[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
//! # fn golden() -> Result<(), colconv::Error> {
//! use colconv::{Convert, ColorMatrix, ColorSpec, DynamicRange, PixelFormat};
//! use colconv::frame::Yuv420pFrame;
//!
//! let (w, h) = (8u32, 8u32);
//! let y = [16u8; 64];
//! let (u, v) = ([128u8; 16], [128u8; 16]);
//! let frame = Yuv420pFrame::new(&y, &u, &v, w, h, w, w / 2, w / 2);
//! let spec = ColorSpec::resolve(PixelFormat::Yuv420p, DynamicRange::Limited, ColorMatrix::Bt709);
//!
//! let mut rgb = [0u8; 4 * 4 * 3];
//! let mut luma = [0u8; 4 * 4];
//! Convert::from(&frame)
//!     .spec(spec)
//!     .resize(4, 4)
//!     .rgb(&mut rgb)
//!     .luma(&mut luma)
//!     .run()?;
//! # Ok(())
//! # }
//! # #[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
//! # golden().unwrap();
//! ```
//!
//! # Tier 1: [`Walker`] / [`MixedSinker`](sinker::MixedSinker) / `{fmt}_to` — streaming & custom sinks
//!
//! Every source pixel format also has a per-format walker (`yuv420p_to`,
//! `nv12_to`, `bgr24_to`, …) that walks the source row by row and hands each row
//! to a caller-supplied [`PixelSink`]. The Sink decides what to derive — luma
//! only, RGB only, HSV only, all three, or something custom — and writes into
//! whatever buffers it owns. Reach for this tier when Tier 0's fixed shape is
//! too rigid: custom output geometry / strides, partial-frame feeding, a bespoke
//! sink, or a Bayer CFA source (whose mosaic pattern is frame-intrinsic rather
//! than spec-derivable, so it is Tier-1-only — see [`raw::bayer_to`] /
//! [`raw::bayer16_to`]).
//!
//! The row the Sink receives (`Self::Input<'_>`) has a shape that reflects the
//! source format: [`source::Yuv420pRow`] carries Y / U / V slices plus matrix /
//! range metadata; packed‑RGB row types (e.g. [`source::Rgb24Row`],
//! [`source::Bgr24Row`]) carry a single packed slice; etc. Each source family
//! declares a subtrait (`Yuv420pSink: PixelSink<Input<'_> = Yuv420pRow<'_>>`) so
//! kernel signatures stay sharp. For the common "give me RGB / Luma / HSV or any
//! subset" case the crate ships [`sinker::MixedSinker`], configured via
//! [`with_rgb`](sinker::MixedSinker::with_rgb) /
//! [`with_luma`](sinker::MixedSinker::with_luma) /
//! [`with_hsv`](sinker::MixedSinker::with_hsv) to select which channels to
//! derive; the generic [`Walker`] plus per-format `Options` ([`YuvOptions`], …)
//! give the same walk a uniform, spec-configurable shape.
//!
//! As of 0.2 the per-row SIMD kernels (`row::*`) are crate-internal plumbing
//! consumed by the sinker and walkers — they are no longer public API.
//!
//! # Supported source formats
//!
//! Shipped (4:1:1, 4:2:0, 4:2:2, 4:4:0, and 4:4:4 subsampling):
//!
//! | Family           | Bit depth | Subsampling | Packing                  | FFmpeg name           |
//! | ---------------- | --------- | ----------- | ------------------------ | --------------------- |
//! | [`Yuv411p`]      |  8        | 4:1:1       | planar (DV-NTSC legacy)  | `yuv411p`             |
//! | [`Yuv420p`]      |  8        | 4:2:0       | planar                   | `yuv420p`             |
//! | [`Yuv422p`]      |  8        | 4:2:2       | planar                   | `yuv422p`             |
//! | [`Yuv440p`]      |  8        | 4:4:0       | planar                   | `yuv440p`             |
//! | [`Yuv444p`]      |  8        | 4:4:4       | planar                   | `yuv444p`             |
//! | [`Nv12`]         |  8        | 4:2:0       | semi-planar UV           | `nv12`                |
//! | [`Nv21`]         |  8        | 4:2:0       | semi-planar VU           | `nv21`                |
//! | [`Nv16`]         |  8        | 4:2:2       | semi-planar UV           | `nv16`                |
//! | [`Nv24`]         |  8        | 4:4:4       | semi-planar UV           | `nv24`                |
//! | [`Nv42`]         |  8        | 4:4:4       | semi-planar VU           | `nv42`                |
//! | [`Yuv420p9`]     |  9        | 4:2:0       | planar, low-packed       | `yuv420p9le`          |
//! | [`Yuv420p10`]    | 10        | 4:2:0       | planar, low-packed       | `yuv420p10le`         |
//! | [`Yuv420p12`]    | 12        | 4:2:0       | planar, low-packed       | `yuv420p12le`         |
//! | [`Yuv420p14`]    | 14        | 4:2:0       | planar, low-packed       | `yuv420p14le`         |
//! | [`Yuv420p16`]    | 16        | 4:2:0       | planar                   | `yuv420p16le`         |
//! | [`Yuv422p9`]     |  9        | 4:2:2       | planar, low-packed       | `yuv422p9le`          |
//! | [`Yuv422p10`]    | 10        | 4:2:2       | planar, low-packed       | `yuv422p10le`         |
//! | [`Yuv422p12`]    | 12        | 4:2:2       | planar, low-packed       | `yuv422p12le`         |
//! | [`Yuv422p14`]    | 14        | 4:2:2       | planar, low-packed       | `yuv422p14le`         |
//! | [`Yuv422p16`]    | 16        | 4:2:2       | planar                   | `yuv422p16le`         |
//! | [`Yuv440p10`]    | 10        | 4:4:0       | planar, low-packed       | `yuv440p10le`         |
//! | [`Yuv440p12`]    | 12        | 4:4:0       | planar, low-packed       | `yuv440p12le`         |
//! | [`Yuv444p9`]     |  9        | 4:4:4       | planar, low-packed       | `yuv444p9le`          |
//! | [`Yuv444p10`]    | 10        | 4:4:4       | planar, low-packed       | `yuv444p10le`         |
//! | [`Yuv444p12`]    | 12        | 4:4:4       | planar, low-packed       | `yuv444p12le`         |
//! | [`Yuv444p14`]    | 14        | 4:4:4       | planar, low-packed       | `yuv444p14le`         |
//! | [`Yuv444p16`]    | 16        | 4:4:4       | planar                   | `yuv444p16le`         |
//! | [`P010`]         | 10        | 4:2:0       | semi-planar, high-packed | `p010le`              |
//! | [`P012`]         | 12        | 4:2:0       | semi-planar, high-packed | `p012le`              |
//! | [`P016`]         | 16        | 4:2:0       | semi-planar              | `p016le`              |
//! | [`P210`]         | 10        | 4:2:2       | semi-planar, high-packed | `p210le`              |
//! | [`P212`]         | 12        | 4:2:2       | semi-planar, high-packed | `p212le`              |
//! | [`P216`]         | 16        | 4:2:2       | semi-planar              | `p216le`              |
//! | [`P410`]         | 10        | 4:4:4       | semi-planar, high-packed | `p410le`              |
//! | [`P412`]         | 12        | 4:4:4       | semi-planar, high-packed | `p412le`              |
//! | [`P416`]         | 16        | 4:4:4       | semi-planar              | `p416le`              |
//! | [`V210`]         | 10        | 4:2:2       | packed (3 x 10-bit/u32)  | `v210`                |
//! | [`Y210`]         | 10        | 4:2:2       | packed, MSB-aligned u16  | `y210le`              |
//! | [`Y212`]         | 12        | 4:2:2       | packed, MSB-aligned u16  | `y212le`              |
//! | [`Y216`]         | 16        | 4:2:2       | packed, full-range u16   | `y216le`              |
//! | [`V410`]         | 10        | 4:4:4       | packed (one 32-bit word) | `v410`                |
//! | [`V30X`]         | 10        | 4:4:4       | packed (one 32-bit word) | `v30xle`              |
//! | [`Xv36`]         | 12        | 4:4:4       | packed u16 quadruple     | `xv36le`              |
//! | [`Vuya`]         |  8        | 4:4:4       | packed byte quadruple, source α | `vuya`         |
//! | [`Vuyx`]         |  8        | 4:4:4       | packed byte quadruple, α-as-padding | `vuyx`     |
//! | [`Ayuv64`]       | 16        | 4:4:4       | packed u16 quadruple, source α  | `ayuv64le`     |
//! | [`Gbrp`]         |  8        | 4:4:4       | planar GBR (3 planes)            | `gbrp`        |
//! | [`Gbrap`]        |  8        | 4:4:4       | planar GBR + A (4 planes, source α) | `gbrap`   |
//! | [`Xyz12`](crate::source::Xyz12) | 12 | 4:4:4 | packed CIE XYZ (3 x u16, high-bit-packed: bits `[15:4]`) | `xyz12le` / `xyz12be` |
//!
//! [`Xyz12`](crate::source::Xyz12) is the **DCP / digital-cinema** source format. Decoding
//! it requires a SMPTE ST 428-1 §8 inverse OETF, a 3x3 matrix to one
//! of three target gamuts ([`DcpTargetGamut::DciP3`] /
//! [`DcpTargetGamut::Rec709`] / [`DcpTargetGamut::Rec2020`]), then a
//! sRGB-shape forward OETF and integer narrow. Every backend is
//! native SIMD; the OETFs run scalar per lane to preserve the 0-ULP
//! scalar↔SIMD parity contract.
//!
//! ## RAW (Bayer) sources
//!
//! [`raw::Bayer`] (8-bit) and [`raw::Bayer16<BITS>`] (10/12/14/16-bit
//! low-packed `u16`, range `[0, (1 << BITS) - 1]`) feed bilinear
//! demosaic + white balance + 3x3
//! color-correction in a single per-row kernel. Caller supplies
//! [`raw::BayerPattern`] (BGGR / RGGB / GRBG / GBRG),
//! [`raw::WhiteBalance`] gains, and a [`raw::ColorCorrectionMatrix`].
//! See [`raw`] for the full design and parameter docs.
//!
//! Scope: `colconv` covers demosaic onwards. Producing the Bayer
//! plane itself is the upstream pipeline's job — vendor-SDK
//! camera-RAW decoders (R3D / BRAW / NRAW) for compressed
//! camera bitstreams, or FFmpeg's `AV_PIX_FMT_BAYER_*` pixel
//! formats / `bayer_*` decoders for already-uncompressed Bayer
//! sources. Once you have a `BayerFrame` / `BayerFrame16`, hand it
//! to [`raw::bayer_to`] / [`raw::bayer16_to`] with your sink of
//! choice.
//!
//! ## YUVA sources (alpha-drop)
//!
//! Every shipped 4:2:0 / 4:2:2 / 4:4:4 planar family also covers its
//! `yuva*` alpha variant by **alpha-drop**: the caller hands the
//! Y / U / V slices from a 4-plane YUVA buffer to the matching
//! `Yuv*p*Frame` constructor and ignores the alpha plane. This works
//! today for `yuva420p`, `yuva420p9le`, `yuva420p10le`,
//! `yuva420p16le`, `yuva422p`, `yuva422p9le`, `yuva422p10le`,
//! `yuva422p16le`, `yuva444p`, `yuva444p9le`, `yuva444p10le`, and
//! `yuva444p16le` (the full set of YUVA pixel formats FFmpeg
//! produces). RGBA pass-through (preserving the alpha channel into
//! the output) is the dedicated **Ship 8** work item — it adds
//! `with_rgba` / `with_rgba_u16` accessors on `MixedSinker` plus
//! native YUVA frame types.
//!
//! # Kernel families
//!
//! - **Q15 i32 family** — 8-bit kernels (`yuv_420_to_rgb_row`,
//!   `yuv_444_to_rgb_row`, `nv12_to_rgb_row`, `nv24_to_rgb_row` etc.)
//!   and 10/12/14-bit kernels (`yuv_420p_n_to_rgb_*<BITS>`,
//!   `yuv_444p_n_to_rgb_*<BITS>`, `p_n_to_rgb_*<BITS>`). Native SIMD
//!   on every backend (NEON / SSE4.1 / AVX2 / AVX-512 / wasm
//!   simd128). [`Yuv422p`] (and the [`Yuv422p10`] / [`Yuv422p12`] /
//!   [`Yuv422p14`] family) reuses [`Yuv420p`]'s per-row kernels
//!   (4:2:2 differs only in the vertical walker); same for
//!   [`Nv16`] ↔ [`Nv12`]. [`Yuv444p`] and [`Yuv444p10`] /
//!   [`Yuv444p12`] / [`Yuv444p14`] use a dedicated 4:4:4 kernel
//!   family (no horizontal chroma duplication step); [`Nv24`] and
//!   [`Nv42`] share a 4:4:4 kernel family via a `SWAP_UV` const
//!   generic.
//! - **16-bit family** — dedicated `yuv_420p16_to_rgb_*`,
//!   `yuv444p16_to_rgb_*`, `p16_to_rgb_*`. [`Yuv422p16`] reuses the
//!   4:2:0 16-bit kernels by shape equivalence. The **u8-output**
//!   kernels stay on i32 (output-range scaling keeps `coeff x u_d`
//!   within i32). The **u16-output** kernels widen the chroma matrix
//!   multiply-add to i64 to avoid the ~2.31·10⁹ chroma-channel sum
//!   overflowing i32 at `BITS == 16`; the Y path also widens to i64
//!   to handle limited-range unclamped samples.
//!
//! # SIMD coverage
//!
//! Every format above has a native SIMD backend for each supported
//! target (NEON on aarch64; SSE4.1 / AVX2 / AVX-512 on x86_64; wasm
//! simd128). Every u8-output and u16-output path has a native
//! implementation on every backend — including the 16-bit u16-output
//! paths for `Yuv420p16`, `P016`, and `Yuv444p16`, which use the
//! backend-native i64 arithmetic (native `_mm512_srai_epi64` on
//! AVX-512 and `i64x2_shr` on wasm; `srai64_15` bias trick on SSE4.1
//! and AVX2 because those ISAs lack native i64 arithmetic right
//! shift).
//!
//! # Not yet shipped (follow-up)
//!
//! - **Bayer SIMD backends** — Tier 14 currently dispatches to the
//!   scalar reference path on every target; NEON / SSE4.1 / AVX2 /
//!   AVX-512 / wasm simd128 follow-ups will land per the established
//!   backend-symmetry pattern.
//! - **Cinema-camera RAW source formats** — vendor-decoded sensor RGB
//!   in camera-native log + gamut (LogC4 / S-Log3 / REDLog3G10 /
//!   Canon Log 2/3 / BMD Film Gen 5 / V-Log / F-Log) → working-space
//!   conversion via inverse-OETF + 3x3 matrix + sRGB OETF. Roadmap
//!   tracked in `docs/superpowers/plans/2026-05-07-be-rollout-tracking.md`
//!   under "Cinema Camera RAW Support Roadmap". Mirrors the Tier 12
//!   ([`source::Xyz12`]) shape: per-vendor source format, full
//!   `MixedSinker` output coverage, polynomial OETF, 5 SIMD backends.
//! - **Higher-quality Bayer demosaic** — current scalar Bayer kernel
//!   does bilinear demosaic; AHD / Malvar / DCB are quality levers
//!   for cinema-grade proxies (CinemaDNG / DJI Inspire workflows).
//! - **3D LUT (`.cube`) row kernel** — for OCIO-style color management
//!   in cinema pipelines.
//!
//! See [`source`] for the per-format module-level breakdown and
//! [`frame`] for the validated frame types plus the `BITS` const
//! generic on the high-bit-depth families (`Yuv420pFrame16<BITS>`
//! and `PnFrame<BITS>`).
//!
//! [`Yuv411p`]: crate::source::Yuv411p
//! [`Yuv420p`]: crate::source::Yuv420p
//! [`Yuv422p`]: crate::source::Yuv422p
//! [`Yuv440p`]: crate::source::Yuv440p
//! [`Yuv444p`]: crate::source::Yuv444p
//! [`Nv12`]: crate::source::Nv12
//! [`Nv16`]: crate::source::Nv16
//! [`Nv21`]: crate::source::Nv21
//! [`Nv24`]: crate::source::Nv24
//! [`Nv42`]: crate::source::Nv42
//! [`Yuv420p9`]: crate::source::Yuv420p9
//! [`Yuv420p10`]: crate::source::Yuv420p10
//! [`Yuv420p12`]: crate::source::Yuv420p12
//! [`Yuv420p14`]: crate::source::Yuv420p14
//! [`Yuv420p16`]: crate::source::Yuv420p16
//! [`Yuv422p9`]: crate::source::Yuv422p9
//! [`Yuv422p10`]: crate::source::Yuv422p10
//! [`Yuv422p12`]: crate::source::Yuv422p12
//! [`Yuv422p14`]: crate::source::Yuv422p14
//! [`Yuv422p16`]: crate::source::Yuv422p16
//! [`Yuv440p10`]: crate::source::Yuv440p10
//! [`Yuv440p12`]: crate::source::Yuv440p12
//! [`Yuv444p9`]: crate::source::Yuv444p9
//! [`Yuv444p10`]: crate::source::Yuv444p10
//! [`Yuv444p12`]: crate::source::Yuv444p12
//! [`Yuv444p14`]: crate::source::Yuv444p14
//! [`Yuv444p16`]: crate::source::Yuv444p16
//! [`P010`]: crate::source::P010
//! [`P012`]: crate::source::P012
//! [`P016`]: crate::source::P016
//! [`P210`]: crate::source::P210
//! [`P212`]: crate::source::P212
//! [`P216`]: crate::source::P216
//! [`P410`]: crate::source::P410
//! [`P412`]: crate::source::P412
//! [`P416`]: crate::source::P416
//! [`V210`]: crate::source::V210
//! [`Y210`]: crate::source::Y210
//! [`Y212`]: crate::source::Y212
//! [`Y216`]: crate::source::Y216
//! [`V410`]: crate::source::V410
//! [`V30X`]: crate::source::V30X
//! [`Xv36`]: crate::source::Xv36
//! [`Vuya`]: crate::source::Vuya
//! [`Vuyx`]: crate::source::Vuyx
//! [`Ayuv64`]: crate::source::Ayuv64
//! [`Gbrp`]: crate::source::Gbrp
//! [`Gbrap`]: crate::source::Gbrap

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

pub use mediaframe::{
  PixelSink,
  SourceFormat,
  // `mediaframe::color::Matrix` is re-exported as `ColorMatrix` so colconv's
  // public surface and every internal `crate::ColorMatrix` reference keep
  // the disambiguated name (`videoframe::color::ColorMatrix` was renamed to
  // `Matrix` upstream during the videoframe → mediaframe rename). `Info` is
  // likewise re-exported as `ColorInfo` so the generic `Info` name stays out
  // of colconv's root while `ColorSpec::from_info` can name it.
  color::{
    ChromaLocation, DcpTargetGamut, DynamicRange, Info as ColorInfo, Matrix as ColorMatrix,
    Primaries, Transfer,
  },
  frame,
  pixel_format::PixelFormat,
  source,
};

// The `Convert` tier assembles a `MixedSinker` internally, so it rides the
// sinker's `any(std, alloc)` gate.
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod convert;
pub mod raw;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod resample;
// The per-row SIMD kernels are crate-internal plumbing as of 0.2 (they are
// consumed only by `MixedSinker` / the `{fmt}_to` walkers). The crate's own
// benches reach a curated subset through the semver-exempt
// `unstable-bench-internals` shim (`bench_internals`), never `row` directly.
pub(crate) mod row;
pub mod sinker;
pub mod walker;

/// Semver-EXEMPT re-export shim exposing the exact crate-internal
/// [`row`](crate::row) kernels the repository's own Criterion benches measure.
///
/// Gated behind the `unstable-bench-internals` feature, which carries **no**
/// stability guarantee and is intended solely for `cargo bench` inside this
/// repository — external code must use [`Convert`] or
/// [`MixedSinker`](sinker::MixedSinker) plus the `{fmt}_to` walkers instead.
#[cfg(feature = "unstable-bench-internals")]
#[doc(hidden)]
pub mod bench_internals;

#[cfg(feature = "bayer")]
pub use walker::BayerOptions;
pub use walker::{ColorSpec, Walker, Xyz12Options, YuvOptions};

/// The Tier-0 golden entry point ([`Convert`]) and its sealed [`Source`] /
/// [`FromSpec`] contract.
#[cfg(any(feature = "std", feature = "alloc"))]
pub use convert::{Convert, FromSpec, Source};
/// The canonical crate-level error name for the [`Convert`] tier — an alias of
/// [`MixedSinkerError`](sinker::MixedSinkerError), which remains available under
/// its original path.
#[cfg(any(feature = "std", feature = "alloc"))]
pub use sinker::MixedSinkerError as Error;

#[cfg(feature = "yuv-444-packed")]
pub use frame::{Ayuv64Frame, Ayuv64FrameError};
#[cfg(feature = "yuv-444-packed")]
pub use source::{Ayuv64, Ayuv64Row, Ayuv64Sink, ayuv64_to};
