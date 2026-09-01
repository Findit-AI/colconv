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
//! # fn golden() -> Result<(), pixon::Error> {
//! use pixon::{Convert, ColorMatrix, ColorSpec, DynamicRange, PixelFormat};
//! use pixon::frame::Yuv420pFrame;
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
//! Shipped (4:1:0, 4:1:1, 4:2:0, 4:2:2, 4:4:0, and 4:4:4 subsampling, plus
//! non-subsampled RGB / gray / indexed-color sources). This table is
//! mechanically derived from the crate's own `Source`-trait exhaustiveness
//! proof (`src/convert/source_coverage.rs`, a `#[cfg(test)]`-only internal
//! module) — every row below has a corresponding `assert_source::<F>()`
//! call there, gated by the same Cargo feature:
//!
//! | Family                          | Bit depth | Subsampling | Packing                                                  | FFmpeg name                     |
//! | ------------------------------- | --------- | ----------- | --------------------------------------------------------- | -------------------------------- |
//! | [`Yuv411p`]                     | 8         | 4:1:1       | planar (DV-NTSC legacy)                                  | `yuv411p`                       |
//! | [`Yuv410p`]                     | 8         | 4:1:0       | planar (DV-NTSC / Cinepak-Sorenson legacy)               | `yuv410p`                       |
//! | [`Yuv420p`]                     | 8         | 4:2:0       | planar                                                   | `yuv420p`                       |
//! | [`Yuv422p`]                     | 8         | 4:2:2       | planar                                                   | `yuv422p`                       |
//! | [`Yuv440p`]                     | 8         | 4:4:0       | planar                                                   | `yuv440p`                       |
//! | [`Yuv444p`]                     | 8         | 4:4:4       | planar                                                   | `yuv444p`                       |
//! | [`Nv12`]                        | 8         | 4:2:0       | semi-planar UV                                           | `nv12`                          |
//! | [`Nv21`]                        | 8         | 4:2:0       | semi-planar VU                                           | `nv21`                          |
//! | [`Nv16`]                        | 8         | 4:2:2       | semi-planar UV                                           | `nv16`                          |
//! | [`Nv20`]                        | 10        | 4:2:2       | semi-planar UV, low-packed                               | `nv20le`                        |
//! | [`Nv24`]                        | 8         | 4:4:4       | semi-planar UV                                           | `nv24`                          |
//! | [`Nv42`]                        | 8         | 4:4:4       | semi-planar VU                                           | `nv42`                          |
//! | [`Yuv420p9`]                    | 9         | 4:2:0       | planar, low-packed                                       | `yuv420p9le`                    |
//! | [`Yuv420p10`]                   | 10        | 4:2:0       | planar, low-packed                                       | `yuv420p10le`                   |
//! | [`Yuv420p12`]                   | 12        | 4:2:0       | planar, low-packed                                       | `yuv420p12le`                   |
//! | [`Yuv420p14`]                   | 14        | 4:2:0       | planar, low-packed                                       | `yuv420p14le`                   |
//! | [`Yuv420p16`]                   | 16        | 4:2:0       | planar                                                   | `yuv420p16le`                   |
//! | [`Yuv422p9`]                    | 9         | 4:2:2       | planar, low-packed                                       | `yuv422p9le`                    |
//! | [`Yuv422p10`]                   | 10        | 4:2:2       | planar, low-packed                                       | `yuv422p10le`                   |
//! | [`Yuv422p12`]                   | 12        | 4:2:2       | planar, low-packed                                       | `yuv422p12le`                   |
//! | [`Yuv422p14`]                   | 14        | 4:2:2       | planar, low-packed                                       | `yuv422p14le`                   |
//! | [`Yuv422p16`]                   | 16        | 4:2:2       | planar                                                   | `yuv422p16le`                   |
//! | [`Yuv440p10`]                   | 10        | 4:4:0       | planar, low-packed                                       | `yuv440p10le`                   |
//! | [`Yuv440p12`]                   | 12        | 4:4:0       | planar, low-packed                                       | `yuv440p12le`                   |
//! | [`Yuv444p9`]                    | 9         | 4:4:4       | planar, low-packed                                       | `yuv444p9le`                    |
//! | [`Yuv444p10`]                   | 10        | 4:4:4       | planar, low-packed                                       | `yuv444p10le`                   |
//! | [`Yuv444p12`]                   | 12        | 4:4:4       | planar, low-packed                                       | `yuv444p12le`                   |
//! | [`Yuv444p14`]                   | 14        | 4:4:4       | planar, low-packed                                       | `yuv444p14le`                   |
//! | [`Yuv444p16`]                   | 16        | 4:4:4       | planar                                                   | `yuv444p16le`                   |
//! | [`Yuv444p10Msb`]                | 10        | 4:4:4       | planar, MSB-packed                                       | `yuv444p10msble`                |
//! | [`Yuv444p12Msb`]                | 12        | 4:4:4       | planar, MSB-packed                                       | `yuv444p12msble`                |
//! | [`P010`]                        | 10        | 4:2:0       | semi-planar, high-packed                                 | `p010le`                        |
//! | [`P012`]                        | 12        | 4:2:0       | semi-planar, high-packed                                 | `p012le`                        |
//! | [`P016`]                        | 16        | 4:2:0       | semi-planar                                              | `p016le`                        |
//! | [`P210`]                        | 10        | 4:2:2       | semi-planar, high-packed                                 | `p210le`                        |
//! | [`P212`]                        | 12        | 4:2:2       | semi-planar, high-packed                                 | `p212le`                        |
//! | [`P216`]                        | 16        | 4:2:2       | semi-planar                                              | `p216le`                        |
//! | [`P410`]                        | 10        | 4:4:4       | semi-planar, high-packed                                 | `p410le`                        |
//! | [`P412`]                        | 12        | 4:4:4       | semi-planar, high-packed                                 | `p412le`                        |
//! | [`P416`]                        | 16        | 4:4:4       | semi-planar                                              | `p416le`                        |
//! | [`Yuva420p`]                    | 8         | 4:2:0       | planar Y/U/V/A (4 planes), source α                      | `yuva420p`                      |
//! | [`Yuva420p9`]                   | 9         | 4:2:0       | planar Y/U/V/A, low-packed, source α                     | `yuva420p9le`                   |
//! | [`Yuva420p10`]                  | 10        | 4:2:0       | planar Y/U/V/A, low-packed, source α                     | `yuva420p10le`                  |
//! | [`Yuva420p12`]                  | 12        | 4:2:0       | planar Y/U/V/A, low-packed, source α (no FFmpeg enum)    | `yuva420p12le`                  |
//! | [`Yuva420p16`]                  | 16        | 4:2:0       | planar Y/U/V/A, source α                                 | `yuva420p16le`                  |
//! | [`Yuva422p`]                    | 8         | 4:2:2       | planar Y/U/V/A (4 planes), source α                      | `yuva422p`                      |
//! | [`Yuva422p9`]                   | 9         | 4:2:2       | planar Y/U/V/A, low-packed, source α                     | `yuva422p9le`                   |
//! | [`Yuva422p10`]                  | 10        | 4:2:2       | planar Y/U/V/A, low-packed, source α                     | `yuva422p10le`                  |
//! | [`Yuva422p12`]                  | 12        | 4:2:2       | planar Y/U/V/A, low-packed, source α                     | `yuva422p12le`                  |
//! | [`Yuva422p16`]                  | 16        | 4:2:2       | planar Y/U/V/A, source α                                 | `yuva422p16le`                  |
//! | [`Yuva444p`]                    | 8         | 4:4:4       | planar Y/U/V/A (4 planes), source α                      | `yuva444p`                      |
//! | [`Yuva444p9`]                   | 9         | 4:4:4       | planar Y/U/V/A, low-packed, source α                     | `yuva444p9le`                   |
//! | [`Yuva444p10`]                  | 10        | 4:4:4       | planar Y/U/V/A, low-packed, source α                     | `yuva444p10le`                  |
//! | [`Yuva444p12`]                  | 12        | 4:4:4       | planar Y/U/V/A, low-packed, source α                     | `yuva444p12le`                  |
//! | [`Yuva444p14`]                  | 14        | 4:4:4       | planar Y/U/V/A, low-packed, source α (no FFmpeg enum)    | `yuva444p14le`                  |
//! | [`Yuva444p16`]                  | 16        | 4:4:4       | planar Y/U/V/A, source α                                 | `yuva444p16le`                  |
//! | [`Yuyv422`]                     | 8         | 4:2:2       | packed byte quad (Y0,U0,Y1,V0)                           | `yuyv422`                       |
//! | [`Uyvy422`]                     | 8         | 4:2:2       | packed byte quad (U0,Y0,V0,Y1)                           | `uyvy422`                       |
//! | [`Yvyu422`]                     | 8         | 4:2:2       | packed byte quad (Y0,V0,Y1,U0)                           | `yvyu422`                       |
//! | [`Uyyvyy411`]                   | 8         | 4:1:1       | packed, 6B/4px block (U,Y0,Y1,V,Y2,Y3)                   | `uyyvyy411`                     |
//! | [`V210`]                        | 10        | 4:2:2       | packed (3 x 10-bit/u32)                                  | `v210`                          |
//! | [`Y210`]                        | 10        | 4:2:2       | packed, MSB-aligned u16                                  | `y210le`                        |
//! | [`Y212`]                        | 12        | 4:2:2       | packed, MSB-aligned u16                                  | `y212le`                        |
//! | [`Y216`]                        | 16        | 4:2:2       | packed, full-range u16                                   | `y216le`                        |
//! | [`V410`]                        | 10        | 4:4:4       | packed (one 32-bit word)                                 | `v410`                          |
//! | [`V30X`]                        | 10        | 4:4:4       | packed (one 32-bit word)                                 | `v30xle`                        |
//! | [`Xv36`]                        | 12        | 4:4:4       | packed u16 quadruple, α-as-padding                       | `xv36le`                        |
//! | [`Xv48`]                        | 16        | 4:4:4       | packed u16 quadruple, α-as-padding                       | `xv48le`                        |
//! | [`Vuya`]                        | 8         | 4:4:4       | packed byte quadruple, source α                          | `vuya`                          |
//! | [`Vuyx`]                        | 8         | 4:4:4       | packed byte quadruple, α-as-padding                      | `vuyx`                          |
//! | [`Ayuv`]                        | 8         | 4:4:4       | packed byte quadruple (A,Y,U,V), source α                | `ayuv`                          |
//! | [`Uyva`]                        | 8         | 4:4:4       | packed byte quadruple (U,Y,V,A), source α                | `uyva`                          |
//! | [`Vyu444`]                      | 8         | 4:4:4       | packed byte triple (V,Y,U), no alpha                     | `vyu444`                        |
//! | [`Ayuv64`]                      | 16        | 4:4:4       | packed u16 quadruple, source α                           | `ayuv64le`                      |
//! | [`Gbrp`]                        | 8         | 4:4:4       | planar GBR (3 planes)                                    | `gbrp`                          |
//! | [`Gbrp9`]                       | 9         | 4:4:4       | planar GBR (3 planes), low-packed                        | `gbrp9le`                       |
//! | [`Gbrp10`]                      | 10        | 4:4:4       | planar GBR (3 planes), low-packed                        | `gbrp10le`                      |
//! | [`Gbrp12`]                      | 12        | 4:4:4       | planar GBR (3 planes), low-packed                        | `gbrp12le`                      |
//! | [`Gbrp14`]                      | 14        | 4:4:4       | planar GBR (3 planes), low-packed                        | `gbrp14le`                      |
//! | [`Gbrp16`]                      | 16        | 4:4:4       | planar GBR (3 planes)                                    | `gbrp16le`                      |
//! | [`Gbrp10Msb`]                   | 10        | 4:4:4       | planar GBR (3 planes), MSB-packed                        | `gbrp10msble`                   |
//! | [`Gbrp12Msb`]                   | 12        | 4:4:4       | planar GBR (3 planes), MSB-packed                        | `gbrp12msble`                   |
//! | [`Gbrap`]                       | 8         | 4:4:4       | planar GBR + A (4 planes, source α)                      | `gbrap`                         |
//! | [`Gbrap10`]                     | 10        | 4:4:4       | planar GBR+A (4 planes), low-packed, source α            | `gbrap10le`                     |
//! | [`Gbrap12`]                     | 12        | 4:4:4       | planar GBR+A (4 planes), low-packed, source α            | `gbrap12le`                     |
//! | [`Gbrap14`]                     | 14        | 4:4:4       | planar GBR+A (4 planes), low-packed, source α            | `gbrap14le`                     |
//! | [`Gbrap16`]                     | 16        | 4:4:4       | planar GBR+A (4 planes), source α                        | `gbrap16le`                     |
//! | [`Gbrap32`]                     | 32        | 4:4:4       | planar GBR+A (4 planes), full u32, source α              | `gbrap32le`                     |
//! | [`Gbrpf16`]                     | 16f       | 4:4:4       | planar GBR (3 planes), half-float                        | `gbrpf16le`                     |
//! | [`Gbrpf32`]                     | 32f       | 4:4:4       | planar GBR (3 planes), f32                               | `gbrpf32le`                     |
//! | [`Gbrapf16`]                    | 16f       | 4:4:4       | planar GBR+A (4 planes), half-float, source α            | `gbrapf16le`                    |
//! | [`Gbrapf32`]                    | 32f       | 4:4:4       | planar GBR+A (4 planes), f32, source α                   | `gbrapf32le`                    |
//! | [`Rgb24`]                       | 8         | 4:4:4       | packed (R,G,B)                                           | `rgb24`                         |
//! | [`Bgr24`]                       | 8         | 4:4:4       | packed (B,G,R)                                           | `bgr24`                         |
//! | [`Rgba`]                        | 8         | 4:4:4       | packed (R,G,B,A), source α                               | `rgba`                          |
//! | [`Bgra`]                        | 8         | 4:4:4       | packed (B,G,R,A), source α                               | `bgra`                          |
//! | [`Argb`]                        | 8         | 4:4:4       | packed (A,R,G,B), source α                               | `argb`                          |
//! | [`Abgr`]                        | 8         | 4:4:4       | packed (A,B,G,R), source α                               | `abgr`                          |
//! | [`Xrgb`]                        | 8         | 4:4:4       | packed (X,R,G,B), α-as-padding                           | `0rgb`                          |
//! | [`Rgbx`]                        | 8         | 4:4:4       | packed (R,G,B,X), α-as-padding                           | `rgb0`                          |
//! | [`Xbgr`]                        | 8         | 4:4:4       | packed (X,B,G,R), α-as-padding                           | `0bgr`                          |
//! | [`Bgrx`]                        | 8         | 4:4:4       | packed (B,G,R,X), α-as-padding                           | `bgr0`                          |
//! | [`Rgb48`]                       | 16        | 4:4:4       | packed u16 triple (R,G,B)                                | `rgb48le`                       |
//! | [`Bgr48`]                       | 16        | 4:4:4       | packed u16 triple (B,G,R)                                | `bgr48le`                       |
//! | [`Rgba64`]                      | 16        | 4:4:4       | packed u16 quadruple, source α                           | `rgba64le`                      |
//! | [`Bgra64`]                      | 16        | 4:4:4       | packed u16 quadruple, source α                           | `bgra64le`                      |
//! | [`Rgb96`]                       | 32        | 4:4:4       | packed u32 triple (R,G,B)                                | `rgb96le`                       |
//! | [`Rgba128`]                     | 32        | 4:4:4       | packed u32 quadruple, source α                           | `rgba128le`                     |
//! | [`X2Rgb10`]                     | 10        | 4:4:4       | packed u32, 2-bit pad (R,G,B)                            | `x2rgb10le`                     |
//! | [`X2Bgr10`]                     | 10        | 4:4:4       | packed u32, 2-bit pad (B,G,R)                            | `x2bgr10le`                     |
//! | [`Rgb444`]                      | 4         | 4:4:4       | packed u16, 4-bit pad (R,G,B)                            | `rgb444le`                      |
//! | [`Bgr444`]                      | 4         | 4:4:4       | packed u16, 4-bit pad (B,G,R)                            | `bgr444le`                      |
//! | [`Rgb555`]                      | 5         | 4:4:4       | packed u16, 1-bit pad (R,G,B)                            | `rgb555le`                      |
//! | [`Bgr555`]                      | 5         | 4:4:4       | packed u16, 1-bit pad (B,G,R)                            | `bgr555le`                      |
//! | [`Rgb565`]                      | 5/6/5     | 4:4:4       | packed u16 (R,G,B)                                       | `rgb565le`                      |
//! | [`Bgr565`]                      | 5/6/5     | 4:4:4       | packed u16 (B,G,R)                                       | `bgr565le`                      |
//! | [`Rgb8`]                        | 3/3/2     | 4:4:4       | packed byte (R,G,B)                                      | `rgb8`                          |
//! | [`Bgr8`]                        | 2/3/3     | 4:4:4       | packed byte (B,G,R)                                      | `bgr8`                          |
//! | [`Rgb4Byte`]                    | 1/2/1     | 4:4:4       | packed byte, low nibble (R,G,B)                          | `rgb4_byte`                     |
//! | [`Bgr4Byte`]                    | 1/2/1     | 4:4:4       | packed byte, low nibble (B,G,R)                          | `bgr4_byte`                     |
//! | [`Rgb4`]                        | 1/2/1     | 4:4:4       | bitstream, 2px/byte (R,G,B)                              | `rgb4`                          |
//! | [`Bgr4`]                        | 1/2/1     | 4:4:4       | bitstream, 2px/byte (B,G,R)                              | `bgr4`                          |
//! | [`Rgbf16`]                      | 16f       | 4:4:4       | packed half-float triple (R,G,B)                         | `rgbf16`                        |
//! | [`Rgbf32`]                      | 32f       | 4:4:4       | packed f32 triple (R,G,B)                                | `rgbf32`                        |
//! | [`Rgbaf16`]                     | 16f       | 4:4:4       | packed half-float quadruple, source α                    | `rgbaf16le`                     |
//! | [`Rgbaf32`]                     | 32f       | 4:4:4       | packed f32 quadruple, source α                           | `rgbaf32le`                     |
//! | [`Gray8`]                       | 8         | N/A         | single luma plane                                        | `gray`                          |
//! | [`Gray9`]                       | 9         | N/A         | single luma plane, low-packed u16                        | `gray9le`                       |
//! | [`Gray10`]                      | 10        | N/A         | single luma plane, low-packed u16                        | `gray10le`                      |
//! | [`Gray12`]                      | 12        | N/A         | single luma plane, low-packed u16                        | `gray12le`                      |
//! | [`Gray14`]                      | 14        | N/A         | single luma plane, low-packed u16                        | `gray14le`                      |
//! | [`Gray16`]                      | 16        | N/A         | single luma plane                                        | `gray16le`                      |
//! | [`Gray32`]                      | 32        | N/A         | single luma plane, u32                                   | `gray32le`                      |
//! | [`Grayf16`]                     | 16f       | N/A         | single luma plane, half-float                            | `grayf16le`                     |
//! | [`Grayf32`]                     | 32f       | N/A         | single luma plane, f32                                   | `grayf32le`                     |
//! | [`Ya8`]                         | 8         | N/A         | packed luma+alpha byte pairs, source α                   | `ya8`                           |
//! | [`Ya16`]                        | 16        | N/A         | packed luma+alpha u16 pairs, source α                    | `ya16le`                        |
//! | [`Yaf16`]                       | 16f       | N/A         | packed luma+alpha half-float pairs, source α             | `yaf16le`                       |
//! | [`Yaf32`]                       | 32f       | N/A         | packed luma+alpha f32 pairs, source α                    | `yaf32le`                       |
//! | [`Monoblack`]                   | 1         | N/A         | 1bpp bitmap, MSB-first, 0=black                          | `monoblack`                     |
//! | [`Monowhite`]                   | 1         | N/A         | 1bpp bitmap, MSB-first, 0=white                          | `monowhite`                     |
//! | [`Pal8`]                        | 8         | N/A         | single index plane + 256-entry palette                   | `pal8`                          |
//! | [`Xyz12`](crate::source::Xyz12) | 12        | 4:4:4       | packed CIE XYZ (3 x u16, high-bit-packed: bits `[15:4]`) | `xyz12le` / `xyz12be`           |
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
//! Scope: `pixon` covers demosaic onwards. Producing the Bayer
//! plane itself is the upstream pipeline's job — vendor-SDK
//! camera-RAW decoders (R3D / BRAW / NRAW) for compressed
//! camera bitstreams, or FFmpeg's `AV_PIX_FMT_BAYER_*` pixel
//! formats / `bayer_*` decoders for already-uncompressed Bayer
//! sources. Once you have a `BayerFrame` / `BayerFrame16`, hand it
//! to [`raw::bayer_to`] / [`raw::bayer16_to`] with your sink of
//! choice.
//!
//! ## YUVA sources
//!
//! Every FFmpeg `yuva*` pixel format (4:2:0 / 4:2:2 / 4:4:4, 8- through
//! 16-bit) plus the two non-FFmpeg extension depths (`yuva420p12le`,
//! `yuva444p14le`, shipped for symmetry with their non-alpha siblings)
//! ships as a **native** [`Source`] format — [`Yuva420p`] and its
//! high-bit / 4:2:2 / 4:4:4 siblings, gated behind the `yuva` feature
//! (which auto-enables `yuv-planar`). See the roster table above for
//! the full 16-row list; it is mechanically derived from the same
//! `source_coverage.rs` proof these markers satisfy, rather than
//! hand-copied, so it cannot silently drift out of sync the way the list
//! that used to live in this section did.
//!
//! RGBA pass-through — preserving the source's real alpha plane into the
//! output, rather than a constant fill — is shipped:
//! [`with_rgba`](sinker::MixedSinker::with_rgba) /
//! [`with_rgba_u16`](sinker::MixedSinker::with_rgba_u16) on
//! [`sinker::MixedSinker`] attach an 8-bit or native-depth RGBA buffer
//! that carries the YUVA source's actual alpha channel;
//! [`rgba`](Convert::rgba) rides the same path.
//!
//! A caller who only wants RGB / Luma output and would rather not enable
//! the `yuva` feature can still reach for **alpha-drop**: hand the
//! Y / U / V slices from a 4-plane YUVA buffer to the matching non-alpha
//! `Yuv*p*Frame` constructor (available under `yuv-planar` alone) and
//! ignore the alpha plane. This remains a valid lighter-weight
//! alternative, not the only mechanism.
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
//! [`Yuv410p`]: crate::source::Yuv410p
//! [`Nv20`]: crate::source::Nv20
//! [`Yuv444p10Msb`]: crate::source::Yuv444p10Msb
//! [`Yuv444p12Msb`]: crate::source::Yuv444p12Msb
//! [`Yuva420p`]: crate::source::Yuva420p
//! [`Yuva420p9`]: crate::source::Yuva420p9
//! [`Yuva420p10`]: crate::source::Yuva420p10
//! [`Yuva420p12`]: crate::source::Yuva420p12
//! [`Yuva420p16`]: crate::source::Yuva420p16
//! [`Yuva422p`]: crate::source::Yuva422p
//! [`Yuva422p9`]: crate::source::Yuva422p9
//! [`Yuva422p10`]: crate::source::Yuva422p10
//! [`Yuva422p12`]: crate::source::Yuva422p12
//! [`Yuva422p16`]: crate::source::Yuva422p16
//! [`Yuva444p`]: crate::source::Yuva444p
//! [`Yuva444p9`]: crate::source::Yuva444p9
//! [`Yuva444p10`]: crate::source::Yuva444p10
//! [`Yuva444p12`]: crate::source::Yuva444p12
//! [`Yuva444p14`]: crate::source::Yuva444p14
//! [`Yuva444p16`]: crate::source::Yuva444p16
//! [`Yuyv422`]: crate::source::Yuyv422
//! [`Uyvy422`]: crate::source::Uyvy422
//! [`Yvyu422`]: crate::source::Yvyu422
//! [`Uyyvyy411`]: crate::source::Uyyvyy411
//! [`Xv48`]: crate::source::Xv48
//! [`Ayuv`]: crate::source::Ayuv
//! [`Uyva`]: crate::source::Uyva
//! [`Vyu444`]: crate::source::Vyu444
//! [`Gbrp9`]: crate::source::Gbrp9
//! [`Gbrp10`]: crate::source::Gbrp10
//! [`Gbrp12`]: crate::source::Gbrp12
//! [`Gbrp14`]: crate::source::Gbrp14
//! [`Gbrp16`]: crate::source::Gbrp16
//! [`Gbrp10Msb`]: crate::source::Gbrp10Msb
//! [`Gbrp12Msb`]: crate::source::Gbrp12Msb
//! [`Gbrap10`]: crate::source::Gbrap10
//! [`Gbrap12`]: crate::source::Gbrap12
//! [`Gbrap14`]: crate::source::Gbrap14
//! [`Gbrap16`]: crate::source::Gbrap16
//! [`Gbrap32`]: crate::source::Gbrap32
//! [`Gbrpf16`]: crate::source::Gbrpf16
//! [`Gbrpf32`]: crate::source::Gbrpf32
//! [`Gbrapf16`]: crate::source::Gbrapf16
//! [`Gbrapf32`]: crate::source::Gbrapf32
//! [`Rgb24`]: crate::source::Rgb24
//! [`Bgr24`]: crate::source::Bgr24
//! [`Rgba`]: crate::source::Rgba
//! [`Bgra`]: crate::source::Bgra
//! [`Argb`]: crate::source::Argb
//! [`Abgr`]: crate::source::Abgr
//! [`Xrgb`]: crate::source::Xrgb
//! [`Rgbx`]: crate::source::Rgbx
//! [`Xbgr`]: crate::source::Xbgr
//! [`Bgrx`]: crate::source::Bgrx
//! [`Rgb48`]: crate::source::Rgb48
//! [`Bgr48`]: crate::source::Bgr48
//! [`Rgba64`]: crate::source::Rgba64
//! [`Bgra64`]: crate::source::Bgra64
//! [`Rgb96`]: crate::source::Rgb96
//! [`Rgba128`]: crate::source::Rgba128
//! [`X2Rgb10`]: crate::source::X2Rgb10
//! [`X2Bgr10`]: crate::source::X2Bgr10
//! [`Rgb444`]: crate::source::Rgb444
//! [`Bgr444`]: crate::source::Bgr444
//! [`Rgb555`]: crate::source::Rgb555
//! [`Bgr555`]: crate::source::Bgr555
//! [`Rgb565`]: crate::source::Rgb565
//! [`Bgr565`]: crate::source::Bgr565
//! [`Rgb8`]: crate::source::Rgb8
//! [`Bgr8`]: crate::source::Bgr8
//! [`Rgb4Byte`]: crate::source::Rgb4Byte
//! [`Bgr4Byte`]: crate::source::Bgr4Byte
//! [`Rgb4`]: crate::source::Rgb4
//! [`Bgr4`]: crate::source::Bgr4
//! [`Rgbf16`]: crate::source::Rgbf16
//! [`Rgbf32`]: crate::source::Rgbf32
//! [`Rgbaf16`]: crate::source::Rgbaf16
//! [`Rgbaf32`]: crate::source::Rgbaf32
//! [`Gray8`]: crate::source::Gray8
//! [`Gray9`]: crate::source::Gray9
//! [`Gray10`]: crate::source::Gray10
//! [`Gray12`]: crate::source::Gray12
//! [`Gray14`]: crate::source::Gray14
//! [`Gray16`]: crate::source::Gray16
//! [`Gray32`]: crate::source::Gray32
//! [`Grayf16`]: crate::source::Grayf16
//! [`Grayf32`]: crate::source::Grayf32
//! [`Ya8`]: crate::source::Ya8
//! [`Ya16`]: crate::source::Ya16
//! [`Yaf16`]: crate::source::Yaf16
//! [`Yaf32`]: crate::source::Yaf32
//! [`Monoblack`]: crate::source::Monoblack
//! [`Monowhite`]: crate::source::Monowhite
//! [`Pal8`]: crate::source::Pal8

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
  // Two colour-matrix vocabularies, and the split is load-bearing.
  //
  // `mediaframe::color::Matrix` is re-exported as `ColorMatrix`: the open,
  // `#[non_exhaustive]` **descriptor** a stream is tagged with. It can name a
  // matrix no kernel tabulates coefficients for, so it stays out of the
  // kernels and lives on [`ColorSpec`] — the stream's colour description —
  // where the non-affine decodes (#303) read it. (The name is
  // disambiguating: `videoframe::color::ColorMatrix` was renamed to `Matrix`
  // upstream during the videoframe → mediaframe rename.)
  //
  // `KernelMatrix` is the closed, `Copy` **coefficient selector** the row
  // walkers carry and every kernel matches on exhaustively. It is re-exported
  // under mediaframe's own name because it is mediaframe's own concept, and
  // the exchange between the two is `TryFrom<&Matrix>` — the one place an
  // unconvertible matrix is refused ([`UnsupportedKernelMatrixError`]).
  // `KernelGamut` / `UnsupportedKernelGamutError` are the same pair for the
  // XYZ12 target gamut.
  //
  // `Info` is likewise re-exported as `ColorInfo` so the generic `Info` name
  // stays out of pixon's root while `ColorSpec::from_info` can name it.
  color::{
    ChromaLocation, DcpTargetGamut, DynamicRange, Info as ColorInfo, KernelGamut, KernelMatrix,
    Matrix as ColorMatrix, Primaries, Transfer, UnsupportedKernelGamutError,
    UnsupportedKernelMatrixError,
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
pub use walker::{ColorSpec, Walker, YuvOptions};

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
