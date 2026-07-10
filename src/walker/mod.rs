//! A uniform `Walker` layer over the per-format frame walkers.
//!
//! Every source pixel format already ships a free walker fn
//! (`xyz12_to`, `yuv420p_to`, `bayer_to`, …) that iterates a
//! `crate::frame::*Frame` row-by-row and dispatches each row to a
//! [`PixelSink`]. Those fns each take their own bespoke value
//! parameters — `xyz12_to` takes a [`DcpTargetGamut`], the YUV walkers
//! take a `full_range` flag plus a [`ColorMatrix`], the Bayer walkers
//! take a pattern / demosaic / white-balance / colour-correction
//! bundle. [`Walker`] unifies them behind one associated-fn surface:
//! the per-format conversion knobs move into a format-specific
//! `Options` value type ([`Xyz12Options`], [`YuvOptions`],
//! [`BayerOptions`]), and [`Walker::walk`] forwards them to the
//! underlying free fn.
//!
//! This module is **purely additive** — it sits on top of the existing
//! walkers and sinks and changes none of their behaviour. The marker
//! types it implements [`Walker`] for are mediaframe's foreign
//! `crate::source::*` ZSTs; [`Walker`] is colconv's own local trait, so
//! the impls satisfy the orphan rule (a local trait for a foreign
//! type).

#[cfg(all(test, feature = "std"))]
mod tests;

use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, DcpTargetGamut, DynamicRange, PixelFormat, PixelSink,
  Primaries, Transfer,
};
#[cfg(feature = "xyz")]
use crate::{
  frame::Xyz12Frame,
  source::{Xyz12, Xyz12Sink, xyz12_to},
};
#[cfg(feature = "bayer")]
use crate::{
  frame::{BayerFrame, BayerFrame16, BayerSink, BayerSink16, bayer_to, bayer16_to_endian},
  source::{Bayer, Bayer16},
};
#[cfg(feature = "mono")]
use crate::{
  frame::{MonoblackFrame, MonowhiteFrame, Pal8Frame},
  source::{
    Monoblack, MonoblackSink, Monowhite, MonowhiteSink, Pal8, Pal8Sink, monoblack_to, monowhite_to,
    pal8_to,
  },
};
// Planar YUV — 8-bit (`Yuv*pFrame`) on the plain arm, plus the high-bit
// families on the `@const_bits` arm. The 8-bit walkers are the uniform
// `(src, full_range, matrix, sink)` fns. The high-bit families are
// endian-generic: their marker is `Yuv*pN<const BE>`, the underlying
// frame struct `Yuv*pFrame16<'a, BITS, BE>` carries the depth as a
// leading const, and the const-generic `{fmt}_to_endian::<S, BE>` walker
// covers both LE and BE (the LE `{fmt}_to` is its `BE = false` wrapper).
#[cfg(feature = "yuv-planar")]
use crate::{
  frame::{
    Yuv410pFrame, Yuv411pFrame, Yuv420pFrame, Yuv420pFrame16, Yuv422pFrame, Yuv422pFrame16,
    Yuv440pFrame, Yuv440pFrame16, Yuv444pFrame, Yuv444pFrame16, Yuv444pMsbFrame,
  },
  source::{
    Yuv410p, Yuv410pSink, Yuv411p, Yuv411pSink, Yuv420p, Yuv420p9, Yuv420p9Sink, Yuv420p10,
    Yuv420p10Sink, Yuv420p12, Yuv420p12Sink, Yuv420p14, Yuv420p14Sink, Yuv420p16, Yuv420p16Sink,
    Yuv420pSink, Yuv422p, Yuv422p9, Yuv422p9Sink, Yuv422p10, Yuv422p10Sink, Yuv422p12,
    Yuv422p12Sink, Yuv422p14, Yuv422p14Sink, Yuv422p16, Yuv422p16Sink, Yuv422pSink, Yuv440p,
    Yuv440p10, Yuv440p10Sink, Yuv440p12, Yuv440p12Sink, Yuv440pSink, Yuv444p, Yuv444p9,
    Yuv444p9Sink, Yuv444p10, Yuv444p10Msb, Yuv444p10MsbSink, Yuv444p10Sink, Yuv444p12,
    Yuv444p12Msb, Yuv444p12MsbSink, Yuv444p12Sink, Yuv444p14, Yuv444p14Sink, Yuv444p16,
    Yuv444p16Sink, Yuv444pSink, yuv410p_to, yuv411p_to, yuv420p_to, yuv420p9_to_endian,
    yuv420p10_to_endian, yuv420p12_to_endian, yuv420p14_to_endian, yuv420p16_to_endian, yuv422p_to,
    yuv422p9_to_endian, yuv422p10_to_endian, yuv422p12_to_endian, yuv422p14_to_endian,
    yuv422p16_to_endian, yuv440p_to, yuv440p10_to_endian, yuv440p12_to_endian, yuv444p_to,
    yuv444p9_to_endian, yuv444p10_msb_to_endian, yuv444p10_to_endian, yuv444p12_msb_to_endian,
    yuv444p12_to_endian, yuv444p14_to_endian, yuv444p16_to_endian,
  },
};
// Semi-planar YUV — Nv* (8-bit) on the plain arm + P0xx/P2xx/P4xx
// (high-bit) on the `@const_bits` arm. The high-bit markers are
// endian-generic (`P010<const BE>` …) over the shared `PnFrame` /
// `PnFrame422` / `PnFrame444` structs (`<'a, BITS, BE>`), and their
// const-generic `{fmt}_to_endian::<S, BE>` covers LE + BE.
#[cfg(feature = "yuv-semi-planar")]
use crate::{
  frame::{
    Nv12Frame, Nv16Frame, Nv20Frame, Nv21Frame, Nv24Frame, Nv42Frame, PnFrame, PnFrame422,
    PnFrame444,
  },
  source::{
    Nv12, Nv12Sink, Nv16, Nv16Sink, Nv20, Nv20Sink, Nv21, Nv21Sink, Nv24, Nv24Sink, Nv42, Nv42Sink,
    P010, P010Sink, P012, P012Sink, P016, P016Sink, P210, P210Sink, P212, P212Sink, P216, P216Sink,
    P410, P410Sink, P412, P412Sink, P416, P416Sink, nv12_to, nv16_to, nv20_to_endian, nv21_to,
    nv24_to, nv42_to, p010_to_endian, p012_to_endian, p016_to_endian, p210_to_endian,
    p212_to_endian, p216_to_endian, p410_to_endian, p412_to_endian, p416_to_endian,
  },
};
// Packed YUV 4:2:2 / 4:1:1 — single-buffer `(src, full_range, matrix,
// sink)` walkers.
#[cfg(feature = "yuv-packed")]
use crate::{
  frame::{Uyvy422Frame, Uyyvyy411Frame, Yuyv422Frame, Yvyu422Frame},
  source::{
    Uyvy422, Uyvy422Sink, Uyyvyy411, Uyyvyy411Sink, Yuyv422, Yuyv422Sink, Yvyu422, Yvyu422Sink,
    uyvy422_to, uyyvyy411_to, yuyv422_to, yvyu422_to,
  },
};
// Packed YUV 4:2:2 high-bit (Y2xx) — endian-generic markers
// (`Y210<const BE>` …) over the shared `Y2xxFrame<'a, BITS, BE>` struct;
// the const-generic `{fmt}_to_endian::<S, BE>` covers LE + BE.
#[cfg(feature = "y2xx")]
use crate::{
  frame::Y2xxFrame,
  source::{
    Y210, Y210Sink, Y212, Y212Sink, Y216, Y216Sink, y210_to_endian, y212_to_endian, y216_to_endian,
  },
};
// Packed YUV 4:4:4. Two topologies: the byte-order-fixed 8-bit `Vuya` /
// `Vuyx` and the LE-only 10-bit `V30X` ride the plain arm (frames
// `VuyaFrame<'_>` / `VuyxFrame<'_>` / `V30XFrame<'_>`, walker
// `{fmt}_to(src, full_range, matrix, sink)`); the endian-generic `V410`
// (10-bit; FFmpeg `Y410`/`XV30` are the same wire format), `Xv36`
// (12-bit), and `Ayuv64` (16-bit + source alpha) ride the `@const BE`
// arm — marker `Fmt<const BE>` over the trailing-`BE` frame
// `FmtFrame<'a, BE>` (no leading bit-depth const, same shape as XYZ12 /
// Rgb48), delegating to the const-generic `{fmt}_to_endian::<_, BE>` (the
// LE `{fmt}_to` is its `BE = false` wrapper). The packed YUV→RGB outputs
// are matrix-weighted + full_range-scaled, so every family reuses
// [`YuvOptions`]; the alpha plane of `Ayuv64` is read inside the walker
// (RGBA outputs only), never an `Options` knob.
#[cfg(feature = "yuv-444-packed")]
use crate::{
  frame::{
    Ayuv64Frame, AyuvFrame, UyvaFrame, V30XFrame, V410Frame, VuyaFrame, VuyxFrame, Vyu444Frame,
    Xv36Frame, Xv48Frame,
  },
  source::{
    Ayuv, Ayuv64, Ayuv64Sink, AyuvSink, Uyva, UyvaSink, V30X, V30XSink, V410, V410Sink, Vuya,
    VuyaSink, Vuyx, VuyxSink, Vyu444, Vyu444Sink, Xv36, Xv36Sink, Xv48, Xv48Sink, ayuv_to,
    ayuv64_to_endian, uyva_to, v30x_to, v410_to_endian, vuya_to, vuyx_to, vyu444_to,
    xv36_to_endian, xv48_to_endian,
  },
};
// Packed YUV 4:2:2 10-bit `V210` (6 pixels per 16-byte block) —
// endian-generic marker (`V210<const BE>`) over the trailing-`BE` frame
// `V210Frame<'a, BE>` (no leading bit-depth const), so it rides the
// `@const BE` arm and delegates to the const-generic
// `v210_to_endian::<_, BE>` (the LE `v210_to` is its `BE = false`
// wrapper). Matrix-weighted + full_range-scaled, so it reuses
// [`YuvOptions`].
#[cfg(feature = "v210")]
use crate::{
  frame::V210Frame,
  source::{V210, V210Sink, v210_to_endian},
};
// Planar YUVA — uniform `(full_range, matrix)` sources; the alpha plane
// is read inside the walker from the frame (never an `Options` knob), so
// they reuse `YuvOptions`. 8-bit `Yuva*pFrame` on the plain arm; the
// high-bit families on the `@const_bits` arm — endian-generic markers
// (`Yuva420p10<const BE>` …) over the shared `Yuva*pFrame16<'a, BITS, BE>`
// structs, with const-generic `{fmt}_to_endian::<S, BE>` covering LE + BE.
#[cfg(feature = "yuva")]
use crate::{
  frame::{
    Yuva420pFrame, Yuva420pFrame16, Yuva422pFrame, Yuva422pFrame16, Yuva444pFrame, Yuva444pFrame16,
  },
  source::{
    Yuva420p, Yuva420p9, Yuva420p9Sink, Yuva420p10, Yuva420p10Sink, Yuva420p12, Yuva420p12Sink,
    Yuva420p16, Yuva420p16Sink, Yuva420pSink, Yuva422p, Yuva422p9, Yuva422p9Sink, Yuva422p10,
    Yuva422p10Sink, Yuva422p12, Yuva422p12Sink, Yuva422p16, Yuva422p16Sink, Yuva422pSink, Yuva444p,
    Yuva444p9, Yuva444p9Sink, Yuva444p10, Yuva444p10Sink, Yuva444p12, Yuva444p12Sink, Yuva444p14,
    Yuva444p14Sink, Yuva444p16, Yuva444p16Sink, Yuva444pSink, yuva420p_to, yuva420p9_to_endian,
    yuva420p10_to_endian, yuva420p12_to_endian, yuva420p16_to_endian, yuva422p_to,
    yuva422p9_to_endian, yuva422p10_to_endian, yuva422p12_to_endian, yuva422p16_to_endian,
    yuva444p_to, yuva444p9_to_endian, yuva444p10_to_endian, yuva444p12_to_endian,
    yuva444p14_to_endian, yuva444p16_to_endian,
  },
};
// Packed RGB — already-RGB sources (no chroma matrix). The 8-bit packed
// families (`Rgb24`/`Bgr24`/`Rgba`/…/`Bgrx`) ride the plain arm; the
// 16-bit families (`Rgb48`/`Bgr48`/`Rgba64`/`Bgra64`) are endian-generic
// — marker `Rgb48<const BE>` over the trailing-`BE` frame
// `Rgb48Frame<'a, BE>` (no leading bit-depth const), so they ride the
// `@const BE` arm and delegate to the const-generic
// `{fmt}_to_endian::<_, BE>` (the LE `{fmt}_to` is its `BE = false`
// wrapper). The free `{fmt}_to` / `{fmt}_to_endian` walkers still take
// `(full_range, matrix)` — the RGB-input row carries them for the
// `with_luma` / `with_hsv` outputs — so every RGB family reuses
// [`YuvOptions`]; the RGB-only outputs (`with_rgb`/`with_rgba`/`…`)
// ignore them. The 10-bit 2-10-10-10 packed families (`X2Rgb10` /
// `X2Bgr10`) are a distinct word-packed topology: endian-generic marker
// `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>` (no
// leading bit-depth const), so they ride the `@const BE` arm and delegate
// to `{fmt}_to_endian::<_, BE>`.
#[cfg(feature = "rgb")]
use crate::{
  frame::{
    AbgrFrame, ArgbFrame, Bgr24Frame, Bgr48Frame, Bgra64Frame, BgraFrame, BgrxFrame, Rgb24Frame,
    Rgb48Frame, Rgb96Frame, Rgba64Frame, Rgba128Frame, RgbaFrame, RgbxFrame, X2Bgr10Frame,
    X2Rgb10Frame, XbgrFrame, XrgbFrame,
  },
  source::{
    Abgr, AbgrSink, Argb, ArgbSink, Bgr24, Bgr24Sink, Bgr48, Bgr48Sink, Bgra, Bgra64, Bgra64Sink,
    BgraSink, Bgrx, BgrxSink, Rgb24, Rgb24Sink, Rgb48, Rgb48Sink, Rgb96, Rgb96Sink, Rgba, Rgba64,
    Rgba64Sink, Rgba128, Rgba128Sink, RgbaSink, Rgbx, RgbxSink, X2Bgr10, X2Bgr10Sink, X2Rgb10,
    X2Rgb10Sink, Xbgr, XbgrSink, Xrgb, XrgbSink, abgr_to, argb_to, bgr24_to, bgr48_to_endian,
    bgra_to, bgra64_to_endian, bgrx_to, rgb24_to, rgb48_to_endian, rgb96_to_endian, rgba_to,
    rgba64_to_endian, rgba128_to_endian, rgbx_to, x2bgr10_to_endian, x2rgb10_to_endian, xbgr_to,
    xrgb_to,
  },
};
// Legacy packed RGB (5/5/6/5/5/5/4/4/4-bit, `AV_PIX_FMT_*565/555/444LE`).
// Byte-order-fixed LE (no `_to_endian` walker), so they ride the plain
// arm exactly like the 8-bit packed families and reuse [`YuvOptions`].
#[cfg(feature = "rgb-legacy")]
use crate::{
  frame::{
    Bgr4ByteFrame, Bgr4Frame, Bgr8Frame, Bgr444Frame, Bgr555Frame, Bgr565Frame, Rgb4ByteFrame,
    Rgb4Frame, Rgb8Frame, Rgb444Frame, Rgb555Frame, Rgb565Frame,
  },
  source::{
    Bgr4, Bgr4Byte, Bgr4ByteSink, Bgr4Sink, Bgr8, Bgr8Sink, Bgr444, Bgr444Sink, Bgr555, Bgr555Sink,
    Bgr565, Bgr565Sink, Rgb4, Rgb4Byte, Rgb4ByteSink, Rgb4Sink, Rgb8, Rgb8Sink, Rgb444, Rgb444Sink,
    Rgb555, Rgb555Sink, Rgb565, Rgb565Sink, bgr4_byte_to, bgr4_to, bgr8_to, bgr444_to, bgr555_to,
    bgr565_to, rgb4_byte_to, rgb4_to, rgb8_to, rgb444_to, rgb555_to, rgb565_to,
  },
};
// Packed float RGB — already-RGB half/single-precision sources (no chroma
// matrix). Both are endian-generic over the f16/f32 byte order: marker
// `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>` (no
// leading bit-depth const, same shape as XYZ12 / Rgb48), so they ride the
// `@const BE` arm and delegate to the const-generic `{fmt}_to_endian::<_,
// BE>` (the LE `{fmt}_to` is its `BE = false` wrapper). The free walkers
// still take `(full_range, matrix)` — the RGB-input row threads them to the
// `with_luma` / `with_hsv` outputs — so each reuses [`YuvOptions`]; the
// float-RGB outputs (`with_rgb` / `with_rgb_f16` / `…`) ignore them. The
// `f16` element type rides on `half`, already a `rgb-float` dependency. No
// tone-mapping / transfer parameter exists: the float-to-integer conversion
// is a fixed clamp + scale.
#[cfg(feature = "rgb-float")]
use crate::{
  frame::{Rgbaf16Frame, Rgbaf32Frame, Rgbf16Frame, Rgbf32Frame},
  source::{
    Rgbaf16, Rgbaf16Sink, Rgbaf32, Rgbaf32Sink, Rgbf16, Rgbf16Sink, Rgbf32, Rgbf32Sink,
    rgbaf16_to_endian, rgbaf32_to_endian, rgbf16_to_endian, rgbf32_to_endian,
  },
};
// Gray — single-luma (`Gray8`/`GrayN`/`Gray16`) and luma+alpha
// (`Ya8`/`Ya16`) sources. Every gray walker takes `(full_range, matrix)`
// (the RGB / HSV outputs rescale limited-range luma; `matrix` is carried
// but unused by the chroma-free gray kernels), so they all reuse
// [`YuvOptions`]. `Gray8` / `Ya8` ride the plain arm. `Gray16` / `Ya16`
// are endian-generic over the trailing-`BE` frames `Gray16Frame<'a, BE>`
// / `Ya16Frame<'a, BE>` (no leading bit-depth const), so they ride the
// `@const BE` arm (same shape as XYZ12 / Rgb48). The high-bit
// `GrayN` (9/10/12/14) carry the depth as a leading const on the shared
// `GrayNFrame<'a, BITS, BE>`, so they ride the `@const_bits` arm; each
// delegates to the const-generic `{fmt}_to_endian::<_, BE>` (the LE
// `{fmt}_to` is its `BE = false` wrapper), covering LE + BE in one impl.
//
// The float-luma `Grayf16` / `Grayf32` are endian-generic over the f16/f32
// byte order: marker `Fmt<const BE>` over the trailing-`BE` frame
// `FmtFrame<'a, BE>` (no leading bit-depth const), so each rides the
// `@const BE` arm and delegates to `{fmt}_to_endian::<_, BE>`. Their free
// walkers also take `(full_range, matrix)` — `full_range` selects whether the
// RGB output rescales the luma — so they reuse [`YuvOptions`] like the integer
// gray families. (`Grayf16` is the half-float twin of `Grayf32`.)
#[cfg(feature = "gray")]
use crate::{
  frame::{
    Gray8Frame, Gray16Frame, Gray32Frame, GrayNFrame, Grayf16Frame, Grayf32Frame, Ya8Frame,
    Ya16Frame, Yaf16Frame, Yaf32Frame,
  },
  source::{
    Gray8, Gray8Sink, Gray9, Gray9Sink, Gray10, Gray10Sink, Gray12, Gray12Sink, Gray14, Gray14Sink,
    Gray16, Gray16Sink, Gray32, Gray32Sink, Grayf16, Grayf16Sink, Grayf32, Grayf32Sink, Ya8,
    Ya8Sink, Ya16, Ya16Sink, Yaf16, Yaf16Sink, Yaf32, Yaf32Sink, gray8_to, gray9_to_endian,
    gray10_to_endian, gray12_to_endian, gray14_to_endian, gray16_to_endian, gray32_to_endian,
    grayf16_to_endian, grayf32_to_endian, ya8_to, ya16_to_endian, yaf16_to_endian, yaf32_to_endian,
  },
};
// Planar GBR — already-RGB sources (G/B/R planes, no chroma matrix). The
// free walkers still take `(full_range, matrix)` (the RGB-input row
// threads them to the `with_luma` / `with_hsv` outputs; the `with_rgb`
// output ignores them), so every GBR family reuses [`YuvOptions`].
// 8-bit `Gbrp` / `Gbrap` ride the plain arm. The high-bit
// `Gbrp{9,10,12,14,16}` / `Gbrap{10,12,14,16}` carry the depth as a
// leading const on the shared `GbrpHighBitFrame<'a, BITS, BE>` /
// `GbrapHighBitFrame<'a, BITS, BE>`, so they ride the `@const_bits` arm
// and delegate to the const-generic `{fmt}_to_endian::<_, BE>`, covering
// LE + BE in one impl. (FFmpeg has no `GBRAP9`, so the planar-GBRA
// high-bit set starts at 10.)
//
// The float GBR families (`Gbrpf16` / `Gbrpf32` single G/B/R, `Gbrapf16` /
// `Gbrapf32` + alpha) are endian-generic over the f16/f32 byte order:
// marker `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>`
// (no leading bit-depth const), so they ride the `@const BE` arm and
// delegate to `{fmt}_to_endian::<_, BE>`. **Unlike** the integer GBR
// families, their free walkers take only `(src, sink)` — they carry **no**
// `full_range` / `matrix` knobs (the float row is already RGB and ships no
// conversion metadata), so each uses the unit [`Options`](Walker::Options)
// `()`. The alpha plane of the `Gbrapf*` frames is read inside the walker
// (RGBA outputs only), never an `Options` knob.
#[cfg(feature = "gbr")]
use crate::{
  frame::{
    Gbrap32Frame, GbrapFrame, GbrapHighBitFrame, Gbrapf16Frame, Gbrapf32Frame, GbrpFrame,
    GbrpHighBitFrame, GbrpMsbFrame, Gbrpf16Frame, Gbrpf32Frame,
  },
  source::{
    Gbrap, Gbrap10, Gbrap10Sink, Gbrap12, Gbrap12Sink, Gbrap14, Gbrap14Sink, Gbrap16, Gbrap16Sink,
    Gbrap32, Gbrap32Sink, GbrapSink, Gbrapf16, Gbrapf16Sink, Gbrapf32, Gbrapf32Sink, Gbrp, Gbrp9,
    Gbrp9Sink, Gbrp10, Gbrp10Msb, Gbrp10MsbSink, Gbrp10Sink, Gbrp12, Gbrp12Msb, Gbrp12MsbSink,
    Gbrp12Sink, Gbrp14, Gbrp14Sink, Gbrp16, Gbrp16Sink, GbrpSink, Gbrpf16, Gbrpf16Sink, Gbrpf32,
    Gbrpf32Sink, gbrap_to, gbrap10_to_endian, gbrap12_to_endian, gbrap14_to_endian,
    gbrap16_to_endian, gbrap32_to_endian, gbrapf16_to_endian, gbrapf32_to_endian, gbrp_to,
    gbrp9_to_endian, gbrp10_msb_to_endian, gbrp10_to_endian, gbrp12_msb_to_endian,
    gbrp12_to_endian, gbrp14_to_endian, gbrp16_to_endian, gbrpf16_to_endian, gbrpf32_to_endian,
  },
};

/// A uniform entry point over a source format's frame walker.
///
/// `S` is the [`PixelSink`] implementation the rows are dispatched to.
/// Implementors are the per-format marker ZSTs from
/// [`crate::source`]; each names the matching frame borrow as
/// [`Frame`](Self::Frame) and its conversion knobs as
/// [`Options`](Self::Options).
///
/// [`walk`](Self::walk) is an associated fn (no `&self`) — the marker
/// is a ZST and carries no state, so the walk is fully described by the
/// frame, the options, and the sink.
pub trait Walker<S> {
  /// The validated source frame borrow this walker iterates — e.g.
  /// [`Xyz12Frame`] for the XYZ12 source.
  type Frame<'a>;

  /// The per-format conversion options forwarded to the underlying
  /// walker fn — e.g. [`Xyz12Options`] for the XYZ12 source.
  type Options;

  /// Walks `src` row by row, applying `opts`, dispatching each row to
  /// `sink`.
  fn walk(src: &Self::Frame<'_>, opts: &Self::Options, sink: &mut S) -> Result<(), S::Error>
  where
    S: PixelSink;
}

/// Generates a [`Walker`] impl for one source marker, forwarding
/// [`walk`](Walker::walk) to that format's free `{fmt}_to` walker fn.
///
/// `$marker` is the foreign `crate::source::*` ZST, `$sink` the marker's
/// [`PixelSink`] subtrait (the single per-impl bound the `{fmt}_to` fn
/// requires — the trait's method-scoped `where S: PixelSink` is implied
/// by it), `$frame` the per-format frame borrow's base type (the macro
/// appends the GAT lifetime), `$opts` the [`Options`](Walker::Options)
/// value type, and the closure-shaped tail names the `src` / `opts` /
/// `sink` bindings the `$body` expression delegates with.
///
/// The `@const $c: $cty;` arm handles the source families whose marker
/// carries a *single* const parameter that is also the last item in the
/// frame's generic list (the XYZ12 `BE` byte-order bool over
/// `Xyz12Frame<'a, BE>`): it threads the const through the impl header,
/// the marker, the sink bound, and the frame's generic list.
///
/// The `@const_bits $bits, BE;` arm handles the high-bit YUV / YUVA /
/// Y2xx families. Their marker is endian-generic (`Yuv420p10<const BE>`,
/// the `marker!` macro's endian-aware arm) but the *bit depth is baked
/// into the marker name* and lives as a separate leading const on the
/// underlying frame struct (`Yuv420pFrame16<'a, BITS, BE>`,
/// `PnFrame<'a, BITS, BE>`, `Y2xxFrame<'a, BITS, BE>`, …). So only `BE`
/// is generic in the impl, while `$bits` is a literal spliced between the
/// frame lifetime and `BE`. The walk delegates to the const-generic
/// `{fmt}_to_endian::<S, BE>` (the public `{fmt}_to` is just its
/// `BE = false` wrapper), so a single impl covers **both** the LE
/// (`BE = false`) and BE (`BE = true`) high-bit sources.
///
/// The `@const2 $bits: $bty, $be: $bety;` arm handles Bayer16, whose
/// marker carries *two* generic consts — the depth `BITS` (kept generic
/// over {10, 12, 14, 16}, so it cannot use the literal-bit-depth
/// `@const_bits` arm) and the wire byte order `BE`. Both consts thread
/// through the impl header, the marker (`Bayer16<BITS, BE>`), the sink
/// bound (`BayerSink16<BITS, BE>`), and the frame's generic list
/// (`BayerFrame16<'a, BITS, BE>`). The walk delegates to the
/// fully-generic `bayer16_to_endian::<BITS, BE, _>`, so one impl serves
/// every depth in both byte orders.
///
/// (The `@` sentinel avoids the `<const …>` matcher mis-parse —
/// rust-lang/rust#143874.)
// Gated to the union of the source families it generates impls for —
// otherwise a build with none of them active sees the macro as dead and
// `-D unused-macros` rejects it.
#[cfg(any(
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "yuva",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "gray",
  feature = "gbr",
))]
macro_rules! walker {
  ($marker:ty, $sink:path, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    impl<S> Walker<S> for $marker
    where
      S: $sink,
    {
      type Frame<'a> = $frame<'a>;
      type Options = $opts;

      #[inline(always)]
      fn walk($s: &Self::Frame<'_>, $o: &Self::Options, $k: &mut S) -> Result<(), S::Error>
      where
        S: PixelSink,
      {
        $body
      }
    }
  };
  // Convert-tier rows: emit the plain-arm [`Walker`] impl *plus* the sealed
  // [`crate::convert::Source`] impl (frame -> marker -> options -> walker) and
  // its RGBA attach hook, so the frame decodes through the golden `Convert`
  // builder. The `Source` half names `MixedSinker`, so it rides the sinker's
  // `any(std, alloc)` gate; the `Walker` half stays as-is (generic over the
  // sink type). `walk_mixed` is generic over the resampler `R`: a Convert-tier
  // format's per-format sink implements its sink trait for every `R` (the
  // blanket `impl<R> …Sink for MixedSinker<'_, $marker, R>`), so the direct
  // `{fmt}_to` body type-checks for all `R` with no extra bound.
  (@convert $marker:ty, $sink:path, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    walker!($marker, $sink, $frame, $opts, |$s, $o, $k| $body);

    #[cfg(any(feature = "std", feature = "alloc"))]
    impl $crate::convert::sealed::Sealed for $frame<'_> {}

    #[cfg(any(feature = "std", feature = "alloc"))]
    impl $crate::convert::Source for $frame<'_> {
      type Marker = $marker;
      type Options = $opts;

      #[inline]
      fn dimensions(&self) -> (usize, usize) {
        (self.width() as usize, self.height() as usize)
      }

      #[inline]
      fn walk_mixed<R>(
        &self,
        $o: &$opts,
        $k: &mut $crate::sinker::MixedSinker<'_, $marker, R>,
      ) -> Result<(), $crate::sinker::MixedSinkerError>
      where
        R: $crate::resample::Resampler,
      {
        let $s = self;
        $body
      }

      #[inline]
      fn attach_rgba<'sink, R>(
        sink: &mut $crate::sinker::MixedSinker<'sink, $marker, R>,
        buf: &'sink mut [u8],
      ) -> Result<(), $crate::sinker::MixedSinkerError>
      where
        R: $crate::resample::Resampler,
      {
        sink.set_rgba(buf)?;
        Ok(())
      }
    }
  };
  // Convert-tier `@const` rows: the endian-const variant of the `@convert` arm.
  // Delegates to the plain `@const` arm for the byte-identical [`Walker`] impl,
  // then adds the sealed [`crate::convert::Source`] impl with the byte-order
  // const `$c` threaded through the `Source` header and the frame's generic
  // list (`$frame<'_, $c>`) exactly as the `Walker` impl threads it. `$marker`
  // is the full `Fmt<$c>` marker type, so it is reused verbatim for both
  // `Source::Marker` and the `MixedSinker` the walk / attach names.
  (@convert @const $c:ident: $cty:ty; $marker:ty, $sink:ident, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    walker!(@const $c: $cty; $marker, $sink, $frame, $opts, |$s, $o, $k| $body);

    #[cfg(any(feature = "std", feature = "alloc"))]
    impl<const $c: $cty> $crate::convert::sealed::Sealed for $frame<'_, $c> {}

    #[cfg(any(feature = "std", feature = "alloc"))]
    impl<const $c: $cty> $crate::convert::Source for $frame<'_, $c> {
      type Marker = $marker;
      type Options = $opts;

      #[inline]
      fn dimensions(&self) -> (usize, usize) {
        (self.width() as usize, self.height() as usize)
      }

      #[inline]
      fn walk_mixed<R>(
        &self,
        $o: &$opts,
        $k: &mut $crate::sinker::MixedSinker<'_, $marker, R>,
      ) -> Result<(), $crate::sinker::MixedSinkerError>
      where
        R: $crate::resample::Resampler,
      {
        let $s = self;
        $body
      }

      #[inline]
      fn attach_rgba<'sink, R>(
        sink: &mut $crate::sinker::MixedSinker<'sink, $marker, R>,
        buf: &'sink mut [u8],
      ) -> Result<(), $crate::sinker::MixedSinkerError>
      where
        R: $crate::resample::Resampler,
      {
        sink.set_rgba(buf)?;
        Ok(())
      }
    }
  };
  // Convert-tier `@const` rows whose per-format sink impls are generic over the
  // resampler `R` only under an *extra* cfg (`$src_cfg`) narrower than the row's
  // own feature gate. The `Source` half — `walk_mixed<R>` / `attach_rgba<R>`,
  // generic over every `R: Resampler` — is only well-formed where the blanket
  // `impl<R> …Sink for MixedSinker<'_, $marker, R>` exists, so it carries the
  // same `$src_cfg` (ANDed onto the `any(std, alloc)` gate). The `Walker` half
  // stays ungated: outside `$src_cfg` the sink is `NoopResampler`-only and the
  // format is Walker/Tier-1-only (Tier-0-with-resize could never type-check
  // there anyway). Used by the packed-float-RGB rows, whose blanket sink impls
  // live under `any(feature = "yuv-planar", feature = "rgb")`.
  (@convert @const(cfg: $src_cfg:meta) $c:ident: $cty:ty; $marker:ty, $sink:ident, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    walker!(@const $c: $cty; $marker, $sink, $frame, $opts, |$s, $o, $k| $body);

    #[cfg(all(any(feature = "std", feature = "alloc"), $src_cfg))]
    impl<const $c: $cty> $crate::convert::sealed::Sealed for $frame<'_, $c> {}

    #[cfg(all(any(feature = "std", feature = "alloc"), $src_cfg))]
    impl<const $c: $cty> $crate::convert::Source for $frame<'_, $c> {
      type Marker = $marker;
      type Options = $opts;

      #[inline]
      fn dimensions(&self) -> (usize, usize) {
        (self.width() as usize, self.height() as usize)
      }

      #[inline]
      fn walk_mixed<R>(
        &self,
        $o: &$opts,
        $k: &mut $crate::sinker::MixedSinker<'_, $marker, R>,
      ) -> Result<(), $crate::sinker::MixedSinkerError>
      where
        R: $crate::resample::Resampler,
      {
        let $s = self;
        $body
      }

      #[inline]
      fn attach_rgba<'sink, R>(
        sink: &mut $crate::sinker::MixedSinker<'sink, $marker, R>,
        buf: &'sink mut [u8],
      ) -> Result<(), $crate::sinker::MixedSinkerError>
      where
        R: $crate::resample::Resampler,
      {
        sink.set_rgba(buf)?;
        Ok(())
      }
    }
  };
  // Convert-tier `@const_bits` rows: the high-bit variant of the `@convert`
  // arm. Delegates to the plain `@const_bits` arm for the byte-identical
  // [`Walker`] impl, then adds the sealed [`crate::convert::Source`] impl. Only
  // `$be` is generic; `$bits` is the literal depth spliced as the *leading*
  // const on the frame struct (`$frame<'_, $bits, $be>`), and the marker is
  // `$marker<$be>` — matching the `Walker` impl exactly, so one impl covers LE
  // (`$be = false`) and BE (`$be = true`).
  (@convert @const_bits $bits:literal, $be:ident; $marker:ident, $sink:ident, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    walker!(@const_bits $bits, $be; $marker, $sink, $frame, $opts, |$s, $o, $k| $body);

    #[cfg(any(feature = "std", feature = "alloc"))]
    impl<const $be: bool> $crate::convert::sealed::Sealed for $frame<'_, $bits, $be> {}

    #[cfg(any(feature = "std", feature = "alloc"))]
    impl<const $be: bool> $crate::convert::Source for $frame<'_, $bits, $be> {
      type Marker = $marker<$be>;
      type Options = $opts;

      #[inline]
      fn dimensions(&self) -> (usize, usize) {
        (self.width() as usize, self.height() as usize)
      }

      #[inline]
      fn walk_mixed<R>(
        &self,
        $o: &$opts,
        $k: &mut $crate::sinker::MixedSinker<'_, $marker<$be>, R>,
      ) -> Result<(), $crate::sinker::MixedSinkerError>
      where
        R: $crate::resample::Resampler,
      {
        let $s = self;
        $body
      }

      #[inline]
      fn attach_rgba<'sink, R>(
        sink: &mut $crate::sinker::MixedSinker<'sink, $marker<$be>, R>,
        buf: &'sink mut [u8],
      ) -> Result<(), $crate::sinker::MixedSinkerError>
      where
        R: $crate::resample::Resampler,
      {
        sink.set_rgba(buf)?;
        Ok(())
      }
    }
  };
  (@const $c:ident: $cty:ty; $marker:ty, $sink:ident, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    impl<const $c: $cty, S> Walker<S> for $marker
    where
      S: $sink<$c>,
    {
      type Frame<'a> = $frame<'a, $c>;
      type Options = $opts;

      #[inline(always)]
      fn walk($s: &Self::Frame<'_>, $o: &Self::Options, $k: &mut S) -> Result<(), S::Error>
      where
        S: PixelSink,
      {
        $body
      }
    }
  };
  // High-bit YUV / YUVA / Y2xx: BE-generic marker, BITS literal baked
  // into the marker name and spliced as the *leading* const on the
  // underlying frame struct (`$frame<'a, $bits, BE>`). `$marker` /
  // `$sink` here are the bare identifiers (the macro appends `<BE>`); the
  // `$body` delegates to the const-generic `{fmt}_to_endian`, so the one
  // impl serves LE (`BE = false`) and BE (`BE = true`).
  (@const_bits $bits:literal, $be:ident; $marker:ident, $sink:ident, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    impl<const $be: bool, S> Walker<S> for $marker<$be>
    where
      S: $sink<$be>,
    {
      type Frame<'a> = $frame<'a, $bits, $be>;
      type Options = $opts;

      #[inline(always)]
      fn walk($s: &Self::Frame<'_>, $o: &Self::Options, $k: &mut S) -> Result<(), S::Error>
      where
        S: PixelSink,
      {
        $body
      }
    }
  };
  // Bayer16: *two* generic consts — the depth `BITS` (kept generic over
  // {10, 12, 14, 16}, unlike the `@const_bits` arm which bakes a single
  // bit depth into the marker name) and the wire byte order `BE`. The
  // marker (`Bayer16<BITS, BE>`), the sink bound (`BayerSink16<BITS, BE>`),
  // and the frame (`BayerFrame16<'a, BITS, BE>`) all carry both consts;
  // the `$body` delegates to the const-generic `bayer16_to_endian`, so the
  // one impl serves all four depths in both LE (`BE = false`) and BE
  // (`BE = true`).
  (@const2 $bits:ident: $bty:ty, $be:ident: $bety:ty; $marker:ident, $sink:ident, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    impl<const $bits: $bty, const $be: $bety, S> Walker<S> for $marker<$bits, $be>
    where
      S: $sink<$bits, $be>,
    {
      type Frame<'a> = $frame<'a, $bits, $be>;
      type Options = $opts;

      #[inline(always)]
      fn walk($s: &Self::Frame<'_>, $o: &Self::Options, $k: &mut S) -> Result<(), S::Error>
      where
        S: PixelSink,
      {
        $body
      }
    }
  };
}

/// Conversion options for the XYZ12 ([`Xyz12`]) source — the target RGB
/// gamut its inverse-OETF + 3×3 matrix converts into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Xyz12Options {
  target_gamut: DcpTargetGamut,
}

impl Xyz12Options {
  /// Creates options with the default target gamut
  /// ([`DcpTargetGamut`]'s own default — `DciP3`, the SMPTE ST 428-1
  /// D-Cinema decode target).
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      target_gamut: DcpTargetGamut::DciP3,
    }
  }

  /// The target RGB gamut the XYZ → RGB matrix converts into.
  #[inline(always)]
  pub const fn target_gamut(&self) -> DcpTargetGamut {
    self.target_gamut
  }

  /// Sets the target RGB gamut (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_target_gamut(mut self, target_gamut: DcpTargetGamut) -> Self {
    self.target_gamut = target_gamut;
    self
  }
}

impl Default for Xyz12Options {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

/// Conversion options shared by the YUV-family sources — the
/// quantisation range (`full_range`) and the YCbCr [`ColorMatrix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct YuvOptions {
  full_range: bool,
  matrix: ColorMatrix,
}

impl YuvOptions {
  /// Creates options for limited-range [`ColorMatrix::Bt709`] — the
  /// implicit default of the common HD YUV pipeline.
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      full_range: false,
      matrix: ColorMatrix::Bt709,
    }
  }

  /// Builds options from a resolved [`ColorSpec`] — the **range-safe**
  /// entry point.
  ///
  /// The spec's resolved [`full_range`](ColorSpec::full_range) and
  /// [`matrix`](ColorSpec::matrix) become this `YuvOptions`. Because a
  /// `ColorSpec` can only be produced by [`ColorSpec::resolve`] — which
  /// honours a format's *pinned* range — this is the constructor a caller
  /// cannot use to decode a range-pinned source (a `yuvj*` alias) with the
  /// wrong range. It feeds the **same** kernels as the raw builders, so it
  /// carries zero per-pixel cost.
  ///
  /// Only the **walker-consumed** half of the spec lands here: the
  /// [`chroma_location`](ColorSpec::chroma_location) is *intentionally not*
  /// carried, because mediaframe's YUV walker row threads only range +
  /// matrix to the kernels, never the siting. It is **sink-consumed** instead
  /// — pass the same `spec` to
  /// [`MixedSinker::with_color_spec`](crate::sinker::MixedSinker::with_color_spec)
  /// to drive siting-aware 4:2:0 upsampling (#302).
  #[inline(always)]
  pub const fn from_color_spec(spec: ColorSpec) -> Self {
    Self {
      full_range: spec.full_range,
      matrix: spec.matrix,
    }
  }

  /// Whether the source samples are full-range (`true`) or
  /// limited/studio-range (`false`).
  #[inline(always)]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }

  /// The YCbCr matrix the source was encoded with.
  #[inline(always)]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }

  /// Marks the source as full-range (`true`) in place.
  #[deprecated(note = "use ColorSpec::resolve to pin format-implied ranges")]
  #[inline(always)]
  pub const fn set_full_range(&mut self) -> &mut Self {
    self.full_range = true;
    self
  }

  /// Marks the source as full-range (`true`), consuming builder.
  #[deprecated(note = "use ColorSpec::resolve to pin format-implied ranges")]
  #[must_use]
  #[inline(always)]
  pub const fn with_full_range(mut self) -> Self {
    self.full_range = true;
    self
  }

  /// Assigns the raw `full_range` flag in place.
  #[deprecated(note = "use ColorSpec::resolve to pin format-implied ranges")]
  #[inline(always)]
  pub const fn update_full_range(&mut self, full_range: bool) -> &mut Self {
    self.full_range = full_range;
    self
  }

  /// Assigns the raw `full_range` flag, consuming builder.
  #[deprecated(note = "use ColorSpec::resolve to pin format-implied ranges")]
  #[must_use]
  #[inline(always)]
  pub const fn maybe_full_range(mut self, full_range: bool) -> Self {
    self.full_range = full_range;
    self
  }

  /// Marks the source as limited/studio-range (`false`) in place.
  #[deprecated(note = "use ColorSpec::resolve to pin format-implied ranges")]
  #[inline(always)]
  pub const fn clear_full_range(&mut self) -> &mut Self {
    self.full_range = false;
    self
  }

  /// Sets the YCbCr matrix (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_matrix(mut self, matrix: ColorMatrix) -> Self {
    self.matrix = matrix;
    self
  }
}

impl Default for YuvOptions {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

/// A colour-decode spec resolved against a source format's *pinned*
/// metadata — so a format's pinned range **can't be passed wrong**.
///
/// Some pixel formats pin their quantisation range by identity: FFmpeg's
/// deprecated `yuvj*` ("JPEG") aliases always decode **full-range**, no
/// matter what range a container's stream metadata claims. The raw
/// builders ([`YuvOptions::maybe_full_range`] and friends) let a caller
/// set `full_range` freely, so a `yuvj420p` source can be decoded
/// limited-range — silently wrong colours.
///
/// [`ColorSpec::resolve`] closes that seam **by construction**: it
/// consults [`PixelFormat::canonical`], and when the format pins a range
/// that pin wins while the caller-supplied `stream_range` is ignored. So
/// a "`Yuvj420p` as limited-range" `ColorSpec` is *unconstructable*. Feed
/// the result to [`YuvOptions::from_color_spec`] to drive the walk.
///
/// `resolve` also follows the format **alias** to its canonical decode
/// form ([`format`](Self::format)) — e.g. `Gray8a` → `Ya8`, `Xv30Le` →
/// `V410Le` — so an aliased source is decoded as its one representative
/// layout.
///
/// Beyond the range and [`matrix`](Self::matrix) the YCbCr→RGB kernels
/// consume, a `ColorSpec` also **carries** the rest of mediaframe's
/// [`ColorInfo`]: the [`primaries`](Self::primaries),
/// [`transfer`](Self::transfer) and [`chroma_location`](Self::chroma_location).
/// These three are pass-through metadata — the conversion math does not
/// consume them yet (siting-aware chroma upsampling is what will read
/// `chroma_location`) — but threading them through keeps a source's full
/// colour description in one value. [`from_info`](Self::from_info) populates
/// all of them from a [`ColorInfo`]; the range-only
/// [`resolve`](Self::resolve) leaves the three `Unspecified`.
///
/// ```
/// use colconv::{ColorMatrix, ColorSpec, DynamicRange, PixelFormat, YuvOptions};
///
/// // A `yuvj420p` source pins full-range: the stream's claimed range loses.
/// let spec = ColorSpec::resolve(PixelFormat::Yuvj420p, DynamicRange::Limited, ColorMatrix::Bt601);
/// assert!(spec.full_range());
/// assert_eq!(spec.format(), PixelFormat::Yuv420p);
///
/// // A plain `yuv420p` source is stream-driven.
/// let spec = ColorSpec::resolve(PixelFormat::Yuv420p, DynamicRange::Limited, ColorMatrix::Bt601);
/// assert!(!spec.full_range());
///
/// let opts = YuvOptions::from_color_spec(spec);
/// assert!(!opts.full_range());
/// assert_eq!(opts.matrix(), ColorMatrix::Bt601);
/// ```
///
/// [`from_info`](Self::from_info) pins the range exactly like `resolve`
/// while carrying the full colour description:
///
/// ```
/// use colconv::{ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, Primaries, PixelFormat, Transfer};
///
/// // The stream claims limited range, but a `yuvj420p` source still pins full.
/// let info = ColorInfo::new(
///     Primaries::Bt709,
///     Transfer::Iec6196621,
///     ColorMatrix::Bt709,
///     DynamicRange::Limited,
///     ChromaLocation::Left,
/// );
/// let spec = ColorSpec::from_info(PixelFormat::Yuvj420p, info);
/// assert!(spec.full_range());
/// assert_eq!(spec.format(), PixelFormat::Yuv420p);
/// // primaries / transfer / chroma_location carry through unchanged.
/// assert_eq!(spec.primaries(), Primaries::Bt709);
/// assert_eq!(spec.transfer(), Transfer::Iec6196621);
/// assert_eq!(spec.chroma_location(), ChromaLocation::Left);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorSpec {
  format: PixelFormat,
  full_range: bool,
  matrix: ColorMatrix,
  primaries: Primaries,
  transfer: Transfer,
  chroma_location: ChromaLocation,
}

impl ColorSpec {
  /// Resolves the colour-decode spec for a source `fmt`.
  ///
  /// [`PixelFormat::canonical`] yields (a) the canonical decode format the
  /// alias maps to and (b) the range the format *pins*, if any. When a
  /// range is pinned — the `yuvj*` full-range aliases — it **wins** and
  /// `stream_range` is ignored; otherwise the range is stream-driven from
  /// `stream_range`. `matrix` always passes through unchanged (a format's
  /// identity pins no matrix).
  ///
  /// The resolved [`DynamicRange`] maps to colconv's internal `full_range`
  /// flag as: [`Full`](DynamicRange::Full) → `true`;
  /// [`Limited`](DynamicRange::Limited) /
  /// [`Unspecified`](DynamicRange::Unspecified) → `false`. An
  /// [`Unknown`](DynamicRange::Unknown) code is treated **conservatively
  /// as limited-range** (`false`) — an unrecognised range falls back to
  /// studio swing, the video-pipeline norm (and what FFmpeg's
  /// `AVCOL_RANGE_UNSPECIFIED` itself implies).
  ///
  /// This range-only entry point leaves [`primaries`](Self::primaries),
  /// [`transfer`](Self::transfer) and
  /// [`chroma_location`](Self::chroma_location) `Unspecified`; use
  /// [`from_info`](Self::from_info) to carry those through as well.
  #[inline]
  pub const fn resolve(fmt: PixelFormat, stream_range: DynamicRange, matrix: ColorMatrix) -> Self {
    // Delegate to `from_info` so the format-pinned-range rule lives in one
    // place: an all-`Unspecified` `ColorInfo` carrying just the stream range
    // and matrix reproduces the original range-only resolution.
    Self::from_info(
      fmt,
      ColorInfo::UNSPECIFIED
        .with_range(stream_range)
        .with_matrix(matrix),
    )
  }

  /// Resolves the colour-decode spec from a source `fmt` and its full
  /// stream [`ColorInfo`].
  ///
  /// Range pinning is identical to [`resolve`](Self::resolve): when
  /// [`PixelFormat::canonical`] reports a pinned range (the `yuvj*`
  /// full-range aliases) it **wins** and `stream.range()` is ignored;
  /// otherwise the range follows `stream.range()`. The
  /// [`matrix`](Self::matrix) likewise carries through from
  /// `stream.matrix()`.
  ///
  /// The remaining colour metadata — [`primaries`](Self::primaries),
  /// [`transfer`](Self::transfer) and
  /// [`chroma_location`](Self::chroma_location) — is carried verbatim from
  /// `stream`. These three are **not** consumed by the YCbCr→RGB kernels;
  /// they are threaded through so a source's full colour description lives
  /// in one value (siting-aware chroma upsampling is what will read
  /// `chroma_location`).
  #[inline]
  pub const fn from_info(fmt: PixelFormat, stream: ColorInfo) -> Self {
    let (format, pinned) = fmt.canonical();
    let range = match pinned {
      Some(pinned_range) => pinned_range,
      None => stream.range(),
    };
    Self {
      format,
      full_range: matches!(range, DynamicRange::Full),
      matrix: stream.matrix(),
      primaries: stream.primaries(),
      transfer: stream.transfer(),
      chroma_location: stream.chroma_location(),
    }
  }

  /// The canonical decode format the source alias resolves to (e.g.
  /// `Gray8a` → `Ya8`). A non-alias format resolves to itself.
  #[inline(always)]
  pub const fn format(&self) -> PixelFormat {
    self.format
  }

  /// Whether the resolved range is full (`true`) or limited/studio
  /// (`false`) — pinned by the format when applicable, else stream-driven.
  #[inline(always)]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }

  /// The YCbCr matrix carried through from [`resolve`](Self::resolve) /
  /// [`from_info`](Self::from_info). Consumed by the YCbCr→RGB kernels.
  #[inline(always)]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }

  /// The colour [`Primaries`] carried from the source's
  /// [`ColorInfo`] (`Unspecified` after
  /// [`resolve`](Self::resolve)).
  ///
  /// Carried-for-completeness metadata: the YCbCr→RGB conversion does not
  /// consume it yet.
  #[inline(always)]
  pub const fn primaries(&self) -> Primaries {
    self.primaries
  }

  /// The [`Transfer`] characteristics carried from the
  /// source's [`ColorInfo`] (`Unspecified` after
  /// [`resolve`](Self::resolve)).
  ///
  /// Carried-for-completeness metadata: the YCbCr→RGB conversion does not
  /// consume it yet.
  #[inline(always)]
  pub const fn transfer(&self) -> Transfer {
    self.transfer
  }

  /// The [`ChromaLocation`] carried from the
  /// source's [`ColorInfo`] (`Unspecified` after
  /// [`resolve`](Self::resolve)).
  ///
  /// Carried-for-completeness metadata today; siting-aware chroma
  /// upsampling is what will consume it.
  #[inline(always)]
  pub const fn chroma_location(&self) -> ChromaLocation {
    self.chroma_location
  }
}

/// Conversion options for the Bayer ([`raw::Bayer`](crate::raw::Bayer))
/// sources — the mosaic `pattern`, the `demosaic` algorithm, the
/// white-balance `wb` gains, and the colour-correction matrix `ccm`.
///
/// There is **no `Default`**: the [`BayerPattern`](crate::raw::BayerPattern)
/// is frame-intrinsic (it describes the sensor's mosaic and cannot be
/// guessed), so callers must name it via [`new`](Self::new). This is
/// why [`Walker`] does not bound `Options: Default`.
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayerOptions {
  pattern: crate::raw::BayerPattern,
  demosaic: crate::raw::BayerDemosaic,
  wb: crate::raw::WhiteBalance,
  ccm: crate::raw::ColorCorrectionMatrix,
}

#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
impl BayerOptions {
  /// Creates options for the given mosaic `pattern`, defaulting the
  /// demosaic to [`BayerDemosaic::Bilinear`](crate::raw::BayerDemosaic),
  /// the white balance to
  /// [`WhiteBalance::neutral`](crate::raw::WhiteBalance::neutral), and
  /// the colour-correction matrix to
  /// [`ColorCorrectionMatrix::identity`](crate::raw::ColorCorrectionMatrix::identity).
  #[inline(always)]
  pub const fn new(pattern: crate::raw::BayerPattern) -> Self {
    Self {
      pattern,
      demosaic: crate::raw::BayerDemosaic::Bilinear,
      wb: crate::raw::WhiteBalance::neutral(),
      ccm: crate::raw::ColorCorrectionMatrix::identity(),
    }
  }

  /// The sensor's Bayer mosaic pattern.
  #[inline(always)]
  pub const fn pattern(&self) -> crate::raw::BayerPattern {
    self.pattern
  }

  /// The demosaic reconstruction algorithm.
  #[inline(always)]
  pub const fn demosaic(&self) -> crate::raw::BayerDemosaic {
    self.demosaic
  }

  /// The per-channel white-balance gains.
  #[inline(always)]
  pub const fn wb(&self) -> crate::raw::WhiteBalance {
    self.wb
  }

  /// The 3×3 colour-correction matrix applied after white balance.
  #[inline(always)]
  pub const fn ccm(&self) -> crate::raw::ColorCorrectionMatrix {
    self.ccm
  }

  /// Sets the demosaic algorithm (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_demosaic(mut self, demosaic: crate::raw::BayerDemosaic) -> Self {
    self.demosaic = demosaic;
    self
  }

  /// Sets the white-balance gains (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_wb(mut self, wb: crate::raw::WhiteBalance) -> Self {
    self.wb = wb;
    self
  }

  /// Sets the colour-correction matrix (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_ccm(mut self, ccm: crate::raw::ColorCorrectionMatrix) -> Self {
    self.ccm = ccm;
    self
  }
}

// Every impl below pairs colconv's local [`Walker`] trait with a
// foreign `crate::source::*` marker, so each satisfies the orphan rule
// (local trait, foreign type). The single per-impl `where S: …Sink`
// bound is the one its `{fmt}_to` fn requires; the trait's
// method-scoped `where S: PixelSink` is implied by it (every `…Sink`
// supertraits `PixelSink`).

// XYZ12 — the target RGB gamut its inverse-OETF + 3×3 matrix decodes
// into rides on the [`Xyz12Options`]; `BE` is the wire byte order.
#[cfg(feature = "xyz")]
#[cfg_attr(docsrs, doc(cfg(feature = "xyz")))]
walker!(@convert @const BE: bool; Xyz12<BE>, Xyz12Sink, Xyz12Frame, Xyz12Options,
  |src, opts, sink| xyz12_to::<BE, _>(src, opts.target_gamut(), sink));

// Bayer (8-bit) — the mosaic pattern, demosaic, white balance, and
// colour-correction matrix all ride on the [`BayerOptions`]. Stays on the
// plain (non-`@convert`) arm: [`BayerOptions`] has no [`Default`] (the CFA
// mosaic pattern is frame-intrinsic, not spec-derivable), so it cannot satisfy
// `Source::Options: FromSpec`. Bayer is Tier-1-only by design — decode it via
// [`Walker`] / [`MixedSinker`], not the [`Convert`](crate::Convert) builder.
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
walker!(
  Bayer,
  BayerSink,
  BayerFrame,
  BayerOptions,
  |src, opts, sink| bayer_to(
    src,
    opts.pattern(),
    opts.demosaic(),
    opts.wb(),
    opts.ccm(),
    sink
  )
);

// Bayer16 (10/12/14/16-bit, LE or BE) — same parameter bundle as 8-bit
// Bayer, so it reuses [`BayerOptions`]; `BITS` is the active sample depth
// and `BE` the plane's wire byte order (`false` = LE, `true` = BE). The
// `BE = false` default keeps the single-generic `Bayer16<BITS>` spelling
// (and the `Bayer{10,12,14}` / `Bayer16Bit` LE aliases) walkable, while
// the `Bayer{10,12,14}Be` / `Bayer16BitBe` aliases route through the same
// impl with `BE = true`. Delegates to the byte-order-generic
// [`bayer16_to_endian`](crate::frame::bayer16_to_endian) (the public
// `bayer16_to` is its `BE = false` wrapper), mirroring how the Y2xx family
// threads `BE` from `Y2xxFrame<'_, BITS, BE>` into `Y216Sink<BE>`. Like 8-bit
// Bayer it stays off the `@convert` arm — [`BayerOptions`] has no [`Default`],
// so it cannot satisfy `Source::Options: FromSpec`; Bayer16 is Tier-1-only by
// design.
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
walker!(@const2 BITS: u32, BE: bool; Bayer16, BayerSink16, BayerFrame16, BayerOptions,
  |src, opts, sink| bayer16_to_endian::<BITS, BE, _>(src, opts.pattern(), opts.demosaic(), opts.wb(), opts.ccm(), sink));

// Pal8 — the BGRA palette is frame-intrinsic (carried by the
// [`Pal8Frame`], not the caller), so there are no conversion knobs and
// [`Options`](Walker::Options) is the unit type.
#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
walker!(@convert Pal8, Pal8Sink, Pal8Frame, (), |src, _opts, sink| pal8_to(
  src, sink
));

// Monoblack — 1-bit-per-pixel, bit 0 → black. Its `full_range` /
// `matrix` knobs match the YUV shape, so it reuses [`YuvOptions`]
// (`YuvOptions::matrix()` is the same `mediaframe::color::Matrix` the
// `monoblack_to` walker takes).
#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
walker!(@convert
  Monoblack,
  MonoblackSink,
  MonoblackFrame,
  YuvOptions,
  |src, opts, sink| monoblack_to(src, opts.full_range(), opts.matrix(), sink)
);

// Monowhite — inverted-polarity sibling of Monoblack (bit 0 → white);
// same `full_range` / `matrix` knobs, so it reuses [`YuvOptions`] too.
#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
walker!(@convert
  Monowhite,
  MonowhiteSink,
  MonowhiteFrame,
  YuvOptions,
  |src, opts, sink| monowhite_to(src, opts.full_range(), opts.matrix(), sink)
);

// ===== Uniform YUV families =============================================
//
// Every source below is a *uniform* YUV format: its only conversion
// knobs are the quantisation range (`full_range`) and the YCbCr
// [`ColorMatrix`], so they all reuse [`YuvOptions`].
//
// The 8-bit families (planar `Yuv*p`, semi-planar `Nv*`, packed
// `Yuyv422`/`Uyvy422`/…, 8-bit `Yuva*p`) have no byte-order axis: they
// ride the **plain** arm and forward to their uniform
// `{fmt}_to(src, full_range, matrix, sink)` walker.
//
// The high-bit families (9/10/12/14/16-bit planar, P0xx/P2xx/P4xx
// semi-planar, Y2xx, and the high-bit YUVA families) are const-generic
// in the wire byte order (`<const BE: bool>`), and mediaframe exposes a
// matching const-generic walker for each — `{fmt}_to_endian::<S, BE>`
// (macro-generated by the `walker!`/`marker!` `*_be` arms, alongside the
// LE-only `{fmt}_to` wrapper which is just its `BE = false` shim). So
// every high-bit family rides the **`@const_bits`** arm: the marker is
// `Fmt<const BE>`, the [`Frame`](Walker::Frame) GAT is the underlying
// `*Frame16` / `PnFrame*` / `Y2xxFrame` struct with the depth literal
// spliced before `BE` (`Yuv420pFrame16<'a, 10, BE>`, `PnFrame<'a, 10,
// BE>`, `Y2xxFrame<'a, 10, BE>`, …), and [`Walker::walk`] delegates to
// `{fmt}_to_endian::<_, BE>`. A single impl per family therefore covers
// **both** the LE source (`BE = false`, the impl at the marker's default)
// and the BE source (`BE = true`) — byte-identical to calling
// `{fmt}_to_endian` directly. The module stays additive: the existing
// `{fmt}_to` / `{fmt}_to_endian` walkers, sinks, and kernels are
// untouched.
//
// The YUVA families are uniform too: the alpha plane is read *inside*
// the walker straight from the frame, never threaded as an `Options`
// knob, so they share [`YuvOptions`] with the alpha-less YUV families.

// ---- Planar YUV, 8-bit -------------------------------------------------
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert
  Yuv420p,
  Yuv420pSink,
  Yuv420pFrame,
  YuvOptions,
  |src, opts, sink| yuv420p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert
  Yuv422p,
  Yuv422pSink,
  Yuv422pFrame,
  YuvOptions,
  |src, opts, sink| yuv422p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert
  Yuv444p,
  Yuv444pSink,
  Yuv444pFrame,
  YuvOptions,
  |src, opts, sink| yuv444p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert
  Yuv440p,
  Yuv440pSink,
  Yuv440pFrame,
  YuvOptions,
  |src, opts, sink| yuv440p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert
  Yuv410p,
  Yuv410pSink,
  Yuv410pFrame,
  YuvOptions,
  |src, opts, sink| yuv410p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert
  Yuv411p,
  Yuv411pSink,
  Yuv411pFrame,
  YuvOptions,
  |src, opts, sink| yuv411p_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Planar YUV, high-bit (BE-generic marker; LE + BE via `_to_endian`) -
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 9, BE; Yuv420p9, Yuv420p9Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 10, BE; Yuv420p10, Yuv420p10Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 12, BE; Yuv420p12, Yuv420p12Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 14, BE; Yuv420p14, Yuv420p14Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 16, BE; Yuv420p16, Yuv420p16Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 9, BE; Yuv422p9, Yuv422p9Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 10, BE; Yuv422p10, Yuv422p10Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 12, BE; Yuv422p12, Yuv422p12Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 14, BE; Yuv422p14, Yuv422p14Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 16, BE; Yuv422p16, Yuv422p16Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 9, BE; Yuv444p9, Yuv444p9Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 10, BE; Yuv444p10, Yuv444p10Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 12, BE; Yuv444p12, Yuv444p12Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 10, BE; Yuv444p10Msb, Yuv444p10MsbSink, Yuv444pMsbFrame, YuvOptions,
  |src, opts, sink| yuv444p10_msb_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 12, BE; Yuv444p12Msb, Yuv444p12MsbSink, Yuv444pMsbFrame, YuvOptions,
  |src, opts, sink| yuv444p12_msb_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 14, BE; Yuv444p14, Yuv444p14Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 16, BE; Yuv444p16, Yuv444p16Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 10, BE; Yuv440p10, Yuv440p10Sink, Yuv440pFrame16, YuvOptions,
  |src, opts, sink| yuv440p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@convert @const_bits 12, BE; Yuv440p12, Yuv440p12Sink, Yuv440pFrame16, YuvOptions,
  |src, opts, sink| yuv440p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Semi-planar YUV, 8-bit (Nv*) --------------------------------------
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert Nv12, Nv12Sink, Nv12Frame, YuvOptions, |src, opts, sink| {
  nv12_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert Nv16, Nv16Sink, Nv16Frame, YuvOptions, |src, opts, sink| {
  nv16_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert Nv21, Nv21Sink, Nv21Frame, YuvOptions, |src, opts, sink| {
  nv21_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert Nv24, Nv24Sink, Nv24Frame, YuvOptions, |src, opts, sink| {
  nv24_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert Nv42, Nv42Sink, Nv42Frame, YuvOptions, |src, opts, sink| {
  nv42_to(src, opts.full_range(), opts.matrix(), sink)
});

// ---- Semi-planar YUV, 10-bit low-bit-packed (NV20; LE + BE via `_to_endian`)
//
// NV20 is the low-bit-packed 4:2:2 twin of P210. Its marker is
// endian-generic (`Nv20<const BE>`) over the **trailing**-`BE` frame
// `Nv20Frame<'a, BE>` (no leading bit-depth const — the 10-bit depth is
// baked into the format, not carried as a frame generic), so it rides the
// `@const BE` arm (same shape as XYZ12 / V410 / Gray16), NOT the
// `@const_bits` arm the high-bit P-formats use. It reuses [`YuvOptions`]
// like every YUV family and delegates to the const-generic
// `nv20_to_endian::<_, BE>` (the LE `nv20_to` is its `BE = false` wrapper),
// so one impl covers LE + BE.
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const BE: bool; Nv20<BE>, Nv20Sink, Nv20Frame, YuvOptions,
  |src, opts, sink| nv20_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Semi-planar YUV, high-bit (P0xx/P2xx/P4xx; LE + BE via `_to_endian`)
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 10, BE; P010, P010Sink, PnFrame, YuvOptions,
  |src, opts, sink| p010_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 12, BE; P012, P012Sink, PnFrame, YuvOptions,
  |src, opts, sink| p012_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 16, BE; P016, P016Sink, PnFrame, YuvOptions,
  |src, opts, sink| p016_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 10, BE; P210, P210Sink, PnFrame422, YuvOptions,
  |src, opts, sink| p210_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 12, BE; P212, P212Sink, PnFrame422, YuvOptions,
  |src, opts, sink| p212_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 16, BE; P216, P216Sink, PnFrame422, YuvOptions,
  |src, opts, sink| p216_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 10, BE; P410, P410Sink, PnFrame444, YuvOptions,
  |src, opts, sink| p410_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 12, BE; P412, P412Sink, PnFrame444, YuvOptions,
  |src, opts, sink| p412_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@convert @const_bits 16, BE; P416, P416Sink, PnFrame444, YuvOptions,
  |src, opts, sink| p416_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Packed YUV 4:2:2 / 4:1:1 ------------------------------------------
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
walker!(@convert
  Yuyv422,
  Yuyv422Sink,
  Yuyv422Frame,
  YuvOptions,
  |src, opts, sink| yuyv422_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
walker!(@convert
  Uyvy422,
  Uyvy422Sink,
  Uyvy422Frame,
  YuvOptions,
  |src, opts, sink| uyvy422_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
walker!(@convert
  Yvyu422,
  Yvyu422Sink,
  Yvyu422Frame,
  YuvOptions,
  |src, opts, sink| yvyu422_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
walker!(@convert
  Uyyvyy411,
  Uyyvyy411Sink,
  Uyyvyy411Frame,
  YuvOptions,
  |src, opts, sink| uyyvyy411_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Packed YUV 4:2:2 high-bit (Y2xx; LE + BE via `_to_endian`) --------
#[cfg(feature = "y2xx")]
#[cfg_attr(docsrs, doc(cfg(feature = "y2xx")))]
walker!(@convert @const_bits 10, BE; Y210, Y210Sink, Y2xxFrame, YuvOptions,
  |src, opts, sink| y210_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "y2xx")]
#[cfg_attr(docsrs, doc(cfg(feature = "y2xx")))]
walker!(@convert @const_bits 12, BE; Y212, Y212Sink, Y2xxFrame, YuvOptions,
  |src, opts, sink| y212_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "y2xx")]
#[cfg_attr(docsrs, doc(cfg(feature = "y2xx")))]
walker!(@convert @const_bits 16, BE; Y216, Y216Sink, Y2xxFrame, YuvOptions,
  |src, opts, sink| y216_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ===== Packed YUV 4:4:4 families =======================================
//
// Single-buffer packed 4:4:4 sources. Their packed YUV → RGB outputs are
// matrix-weighted + full_range-scaled, so every family reuses
// [`YuvOptions`]; the `Vuya` / `Ayuv64` source alpha is read inside the
// walker for the RGBA outputs only, never an `Options` knob. The module
// stays additive: the existing walkers, sinks, and kernels are untouched.

// ---- Packed YUV 4:4:4, byte-order-fixed (plain arm) --------------------
// 8-bit `Vuya` (real source α) / `Vuyx` (α padding) and LE-only 10-bit
// `V30X` carry no byte-order axis, so they ride the plain arm and forward
// to their uniform `{fmt}_to(src, full_range, matrix, sink)` walker.
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert Vuya, VuyaSink, VuyaFrame, YuvOptions, |src, opts, sink| {
  vuya_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert Vuyx, VuyxSink, VuyxFrame, YuvOptions, |src, opts, sink| {
  vuyx_to(src, opts.full_range(), opts.matrix(), sink)
});
// `Ayuv` / `Uyva` (real source α) / `Vyu444` (no alpha, 24bpp) are the
// 8-bit packed 4:4:4 channel re-orderings of `Vuya` / `Vuyx` — byte-order-
// fixed, so they ride the plain arm with the same uniform walker forward.
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert Ayuv, AyuvSink, AyuvFrame, YuvOptions, |src, opts, sink| {
  ayuv_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert Uyva, UyvaSink, UyvaFrame, YuvOptions, |src, opts, sink| {
  uyva_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
#[rustfmt::skip]
walker!(@convert Vyu444, Vyu444Sink, Vyu444Frame, YuvOptions, |src, opts, sink| {
  vyu444_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert V30X, V30XSink, V30XFrame, YuvOptions, |src, opts, sink| {
  v30x_to(src, opts.full_range(), opts.matrix(), sink)
});

// ---- Packed YUV 4:4:4, endian-generic (`@const BE` arm; LE + BE) -------
// Marker `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>`
// (no leading bit-depth const, same shape as XYZ12 / Rgb48), delegating
// to `{fmt}_to_endian::<_, BE>`; one impl covers LE (`BE = false`) and BE
// (`BE = true`). `V410` is the 10-bit format (FFmpeg `Y410` / `XV30` name
// the same wire layout); `Xv36` is 12-bit; `Xv48` is its full-16-bit
// sibling (X slot padding); `Ayuv64` is 16-bit + source α.
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert @const BE: bool; V410<BE>, V410Sink, V410Frame, YuvOptions,
  |src, opts, sink| v410_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert @const BE: bool; Xv36<BE>, Xv36Sink, Xv36Frame, YuvOptions,
  |src, opts, sink| xv36_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert @const BE: bool; Xv48<BE>, Xv48Sink, Xv48Frame, YuvOptions,
  |src, opts, sink| xv48_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-444-packed")))]
walker!(@convert @const BE: bool; Ayuv64<BE>, Ayuv64Sink, Ayuv64Frame, YuvOptions,
  |src, opts, sink| ayuv64_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ===== Packed YUV 4:2:2 10-bit `V210` ==================================
//
// 6 pixels per 16-byte block. Endian-generic marker `V210<const BE>` over
// the trailing-`BE` frame `V210Frame<'a, BE>` (no leading bit-depth
// const), so it rides the `@const BE` arm and delegates to the
// const-generic `v210_to_endian::<_, BE>` (the LE `v210_to` is its
// `BE = false` wrapper). Matrix-weighted + full_range-scaled, so it
// reuses [`YuvOptions`].
#[cfg(feature = "v210")]
#[cfg_attr(docsrs, doc(cfg(feature = "v210")))]
walker!(@convert @const BE: bool; V210<BE>, V210Sink, V210Frame, YuvOptions,
  |src, opts, sink| v210_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Planar YUVA, 8-bit (alpha read inside `{fmt}_to`, not an Option) --
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert
  Yuva420p,
  Yuva420pSink,
  Yuva420pFrame,
  YuvOptions,
  |src, opts, sink| yuva420p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert
  Yuva422p,
  Yuva422pSink,
  Yuva422pFrame,
  YuvOptions,
  |src, opts, sink| yuva422p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert
  Yuva444p,
  Yuva444pSink,
  Yuva444pFrame,
  YuvOptions,
  |src, opts, sink| yuva444p_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Planar YUVA, high-bit (BE-generic marker; LE + BE via `_to_endian`)
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 9, BE; Yuva420p9, Yuva420p9Sink, Yuva420pFrame16, YuvOptions,
  |src, opts, sink| yuva420p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 10, BE; Yuva420p10, Yuva420p10Sink, Yuva420pFrame16, YuvOptions,
  |src, opts, sink| yuva420p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 12, BE; Yuva420p12, Yuva420p12Sink, Yuva420pFrame16, YuvOptions,
  |src, opts, sink| yuva420p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 16, BE; Yuva420p16, Yuva420p16Sink, Yuva420pFrame16, YuvOptions,
  |src, opts, sink| yuva420p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 9, BE; Yuva422p9, Yuva422p9Sink, Yuva422pFrame16, YuvOptions,
  |src, opts, sink| yuva422p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 10, BE; Yuva422p10, Yuva422p10Sink, Yuva422pFrame16, YuvOptions,
  |src, opts, sink| yuva422p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 12, BE; Yuva422p12, Yuva422p12Sink, Yuva422pFrame16, YuvOptions,
  |src, opts, sink| yuva422p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 16, BE; Yuva422p16, Yuva422p16Sink, Yuva422pFrame16, YuvOptions,
  |src, opts, sink| yuva422p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 9, BE; Yuva444p9, Yuva444p9Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 10, BE; Yuva444p10, Yuva444p10Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 12, BE; Yuva444p12, Yuva444p12Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 14, BE; Yuva444p14, Yuva444p14Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@convert @const_bits 16, BE; Yuva444p16, Yuva444p16Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ===== Packed RGB families =============================================
//
// These sources are *already RGB* — there is no chroma matrix. The
// underlying `{fmt}_to` / `{fmt}_to_endian` walkers nonetheless take
// `(full_range, matrix)` because the RGB-input row carries them through
// to the `with_luma` / `with_hsv` outputs (the `with_rgb` / `with_rgba`
// / `with_rgb_u16` outputs ignore them). So every RGB family reuses
// [`YuvOptions`] and forwards `opts.full_range()` / `opts.matrix()`,
// byte-identical to a direct walker call. The module stays additive: the
// existing walkers, sinks, and kernels are untouched.

// ---- Packed RGB, 8-bit (plain arm; no byte-order axis) -----------------
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert
  Rgb24,
  Rgb24Sink,
  Rgb24Frame,
  YuvOptions,
  |src, opts, sink| rgb24_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert
  Bgr24,
  Bgr24Sink,
  Bgr24Frame,
  YuvOptions,
  |src, opts, sink| bgr24_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert Rgba, RgbaSink, RgbaFrame, YuvOptions, |src, opts, sink| {
  rgba_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert Bgra, BgraSink, BgraFrame, YuvOptions, |src, opts, sink| {
  bgra_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert Argb, ArgbSink, ArgbFrame, YuvOptions, |src, opts, sink| {
  argb_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert Abgr, AbgrSink, AbgrFrame, YuvOptions, |src, opts, sink| {
  abgr_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert Xrgb, XrgbSink, XrgbFrame, YuvOptions, |src, opts, sink| {
  xrgb_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert Rgbx, RgbxSink, RgbxFrame, YuvOptions, |src, opts, sink| {
  rgbx_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert Xbgr, XbgrSink, XbgrFrame, YuvOptions, |src, opts, sink| {
  xbgr_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert Bgrx, BgrxSink, BgrxFrame, YuvOptions, |src, opts, sink| {
  bgrx_to(src, opts.full_range(), opts.matrix(), sink)
});

// ---- Packed RGB, 16-bit (BE-generic marker; LE + BE via `_to_endian`) --
// Marker `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>`
// (no leading bit-depth const), so these ride the `@const BE` arm (same
// shape as XYZ12) and delegate to `{fmt}_to_endian::<_, BE>`; one impl
// covers both LE (`BE = false`) and BE (`BE = true`).
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert @const BE: bool; Rgb48<BE>, Rgb48Sink, Rgb48Frame, YuvOptions,
  |src, opts, sink| rgb48_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert @const BE: bool; Bgr48<BE>, Bgr48Sink, Bgr48Frame, YuvOptions,
  |src, opts, sink| bgr48_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert @const BE: bool; Rgba64<BE>, Rgba64Sink, Rgba64Frame, YuvOptions,
  |src, opts, sink| rgba64_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert @const BE: bool; Bgra64<BE>, Bgra64Sink, Bgra64Frame, YuvOptions,
  |src, opts, sink| bgra64_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Packed RGB, 32-bit per channel (BE-generic marker; LE + BE) -------
// `Rgb96` (`R, G, B`) is the full-bit `u32` twin of `Rgb48`. Marker
// `Rgb96<const BE>` over the trailing-`BE` frame `Rgb96Frame<'a, BE>` (no
// leading bit-depth const), so it rides the `@const BE` arm and delegates to
// `rgb96_to_endian::<_, BE>`; one impl covers both LE and BE.
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert @const BE: bool; Rgb96<BE>, Rgb96Sink, Rgb96Frame, YuvOptions,
  |src, opts, sink| rgb96_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
// `Rgba128` (`R, G, B, A`, real alpha) is the full-bit `u32` twin of `Rgba64`.
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert @const BE: bool; Rgba128<BE>, Rgba128Sink, Rgba128Frame, YuvOptions,
  |src, opts, sink| rgba128_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Packed RGB, 10-bit 2-10-10-10 (BE-generic marker; LE + BE) --------
// `X2Rgb10` / `X2Bgr10` pack one pixel per 32-bit word
// (`(MSB) 2X | 10 | 10 | 10 (LSB)`, the 2 leading bits padding). Marker
// `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>` (no
// leading bit-depth const), so they ride the `@const BE` arm and delegate
// to `{fmt}_to_endian::<_, BE>`; one impl covers both LE and BE.
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert @const BE: bool; X2Rgb10<BE>, X2Rgb10Sink, X2Rgb10Frame, YuvOptions,
  |src, opts, sink| x2rgb10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@convert @const BE: bool; X2Bgr10<BE>, X2Bgr10Sink, X2Bgr10Frame, YuvOptions,
  |src, opts, sink| x2bgr10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Packed float RGB (Rgbf16 / Rgbf32; LE + BE via `_to_endian`) ------
// Half/single-precision packed RGB. Marker `Fmt<const BE>` over the
// trailing-`BE` frame `FmtFrame<'a, BE>` (no leading bit-depth const), so
// they ride the `@const BE` arm and delegate to `{fmt}_to_endian::<_, BE>`;
// one impl covers both LE (`BE = false`) and BE (`BE = true`). The free
// walkers still take `(full_range, matrix)` (the RGB-input row threads them
// to `with_luma` / `with_hsv`; the float-RGB outputs ignore them), so each
// reuses [`YuvOptions`] — byte-identical to a direct walker call.
//
// Their blanket `impl<R> …Sink for MixedSinker<'_, _, R>` lives under
// `any(feature = "yuv-planar", feature = "rgb")` (Noop-only otherwise), so the
// `@const(cfg: …)` arm gates the `Source` half to match — under `rgb-float`
// alone these stay Walker/Tier-1-only.
#[cfg(feature = "rgb-float")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-float")))]
walker!(@convert @const(cfg: any(feature = "rgb", feature = "yuv-planar")) BE: bool; Rgbf16<BE>, Rgbf16Sink, Rgbf16Frame, YuvOptions,
  |src, opts, sink| rgbf16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb-float")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-float")))]
walker!(@convert @const(cfg: any(feature = "rgb", feature = "yuv-planar")) BE: bool; Rgbf32<BE>, Rgbf32Sink, Rgbf32Frame, YuvOptions,
  |src, opts, sink| rgbf32_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
// Alpha-bearing twins (`Rgbaf16` / `Rgbaf32`): same `@const BE` arm over the
// trailing-`BE` frames, delegating to `{fmt}_to_endian::<_, BE>`. The source
// alpha rides the `with_rgba` / `with_rgba_*` outputs; the RGB / luma / HSV
// outputs drop it. Both reuse [`YuvOptions`] (the float outputs ignore the
// matrix / range; luma / HSV thread them through). Same `@const(cfg: …)`
// `Source`-half gate as their alpha-less twins above — Walker/Tier-1-only
// under `rgb-float` alone.
#[cfg(feature = "rgb-float")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-float")))]
walker!(@convert @const(cfg: any(feature = "rgb", feature = "yuv-planar")) BE: bool; Rgbaf16<BE>, Rgbaf16Sink, Rgbaf16Frame, YuvOptions,
  |src, opts, sink| rgbaf16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb-float")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-float")))]
walker!(@convert @const(cfg: any(feature = "rgb", feature = "yuv-planar")) BE: bool; Rgbaf32<BE>, Rgbaf32Sink, Rgbaf32Frame, YuvOptions,
  |src, opts, sink| rgbaf32_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Legacy packed RGB (byte-order-fixed LE; plain arm) ----------------
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert
  Rgb565,
  Rgb565Sink,
  Rgb565Frame,
  YuvOptions,
  |src, opts, sink| rgb565_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert
  Bgr565,
  Bgr565Sink,
  Bgr565Frame,
  YuvOptions,
  |src, opts, sink| bgr565_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert
  Rgb555,
  Rgb555Sink,
  Rgb555Frame,
  YuvOptions,
  |src, opts, sink| rgb555_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert
  Bgr555,
  Bgr555Sink,
  Bgr555Frame,
  YuvOptions,
  |src, opts, sink| bgr555_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert
  Rgb444,
  Rgb444Sink,
  Rgb444Frame,
  YuvOptions,
  |src, opts, sink| rgb444_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert
  Bgr444,
  Bgr444Sink,
  Bgr444Frame,
  YuvOptions,
  |src, opts, sink| bgr444_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Legacy bit-packed RGB (8bpp 3:3:2 + 1:2:1; 4bpp 1:2:1) ------------
// One byte per pixel (Rgb8 / Bgr8 / Rgb4Byte / Bgr4Byte) or two pixels per
// byte (Rgb4 / Bgr4). No byte-order axis (sub-byte / single-byte), so each
// rides the plain arm and reuses [`YuvOptions`].
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert Rgb8, Rgb8Sink, Rgb8Frame, YuvOptions, |src, opts, sink| {
  rgb8_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert Bgr8, Bgr8Sink, Bgr8Frame, YuvOptions, |src, opts, sink| {
  bgr8_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert
  Rgb4Byte,
  Rgb4ByteSink,
  Rgb4ByteFrame,
  YuvOptions,
  |src, opts, sink| rgb4_byte_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert
  Bgr4Byte,
  Bgr4ByteSink,
  Bgr4ByteFrame,
  YuvOptions,
  |src, opts, sink| bgr4_byte_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert Rgb4, Rgb4Sink, Rgb4Frame, YuvOptions, |src, opts, sink| {
  rgb4_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(@convert Bgr4, Bgr4Sink, Bgr4Frame, YuvOptions, |src, opts, sink| {
  bgr4_to(src, opts.full_range(), opts.matrix(), sink)
});

// ===== Gray families ===================================================
//
// Single-luma (`Gray8` / `GrayN` / `Gray16`) and luma+alpha
// (`Ya8` / `Ya16`) sources. The free walkers take `(full_range, matrix)`:
// `full_range` selects whether the RGB / HSV outputs rescale limited-range
// luma; `matrix` is carried through but unused by the chroma-free gray
// kernels. So every gray family reuses [`YuvOptions`], forwarding both —
// byte-identical to a direct walker call. The module stays additive: the
// existing walkers, sinks, and kernels are untouched.

// ---- Gray, 8-bit (plain arm; no byte-order axis) -----------------------
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert
  Gray8,
  Gray8Sink,
  Gray8Frame,
  YuvOptions,
  |src, opts, sink| gray8_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert
  Ya8,
  Ya8Sink,
  Ya8Frame,
  YuvOptions,
  |src, opts, sink| ya8_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Gray, high-bit GrayN (BE-generic marker; LE + BE via `_to_endian`) -
// Marker `GrayN<const BE>` over the shared `GrayNFrame<'a, BITS, BE>` (the
// depth is a leading const), so these ride the `@const_bits` arm.
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const_bits 9, BE; Gray9, Gray9Sink, GrayNFrame, YuvOptions,
  |src, opts, sink| gray9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const_bits 10, BE; Gray10, Gray10Sink, GrayNFrame, YuvOptions,
  |src, opts, sink| gray10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const_bits 12, BE; Gray12, Gray12Sink, GrayNFrame, YuvOptions,
  |src, opts, sink| gray12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const_bits 14, BE; Gray14, Gray14Sink, GrayNFrame, YuvOptions,
  |src, opts, sink| gray14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Gray, 16-bit + Ya16 (BE-generic marker; LE + BE via `_to_endian`) -
// Marker `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>`
// (no leading bit-depth const), so these ride the `@const BE` arm (same
// shape as XYZ12 / Rgb48) and delegate to `{fmt}_to_endian::<_, BE>`.
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const BE: bool; Gray16<BE>, Gray16Sink, Gray16Frame, YuvOptions,
  |src, opts, sink| gray16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
// `Gray32` is the full-bit integer twin of `Gray16` (one `u32` luma plane);
// same `@const BE` arm, delegating to `gray32_to_endian::<_, BE>`.
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const BE: bool; Gray32<BE>, Gray32Sink, Gray32Frame, YuvOptions,
  |src, opts, sink| gray32_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const BE: bool; Ya16<BE>, Ya16Sink, Ya16Frame, YuvOptions,
  |src, opts, sink| ya16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Gray, float-luma Grayf16 / Grayf32 (BE-generic marker; LE + BE) --------
// Marker `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>` (no
// leading bit-depth const), so each rides the `@const BE` arm (same shape as
// Gray16) and delegates to `{fmt}_to_endian::<_, BE>`. The free walker takes
// `(full_range, matrix)` — `full_range` selects whether the RGB output rescales
// the luma — so it reuses [`YuvOptions`]. `Grayf16` is the half-float twin of
// `Grayf32`; its outputs widen each `f16` to `f32` before the same conversion.
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const BE: bool; Grayf16<BE>, Grayf16Sink, Grayf16Frame, YuvOptions,
  |src, opts, sink| grayf16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const BE: bool; Grayf32<BE>, Grayf32Sink, Grayf32Frame, YuvOptions,
  |src, opts, sink| grayf32_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Gray, float-luma + alpha Yaf16 / Yaf32 (BE-generic marker; LE + BE) -----
// The half-float / single-precision gray+alpha twins of `Grayf16` / `Grayf32`:
// marker `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>`, so each
// rides the `@const BE` arm and delegates to `{fmt}_to_endian::<_, BE>`. Real
// source alpha (slot 1 of each pixel pair) is read inside the sinker, never an
// `Options` knob, so they reuse [`YuvOptions`] like the other gray families.
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const BE: bool; Yaf16<BE>, Yaf16Sink, Yaf16Frame, YuvOptions,
  |src, opts, sink| yaf16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gray")]
#[cfg_attr(docsrs, doc(cfg(feature = "gray")))]
walker!(@convert @const BE: bool; Yaf32<BE>, Yaf32Sink, Yaf32Frame, YuvOptions,
  |src, opts, sink| yaf32_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ===== Planar GBR families =============================================
//
// Already-RGB sources (G / B / R planes, no chroma matrix). The free
// walkers still take `(full_range, matrix)` because the RGB-input row
// threads them to the `with_luma` / `with_hsv` outputs (the `with_rgb`
// output ignores them), so every GBR family reuses [`YuvOptions`] and
// forwards both, byte-identical to a direct walker call. The module stays
// additive: the existing walkers, sinks, and kernels are untouched.

// ---- Planar GBR, 8-bit (plain arm; no byte-order axis) -----------------
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert Gbrp, GbrpSink, GbrpFrame, YuvOptions, |src, opts, sink| {
  gbrp_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert
  Gbrap,
  GbrapSink,
  GbrapFrame,
  YuvOptions,
  |src, opts, sink| gbrap_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Planar GBR, high-bit (BE-generic marker; LE + BE via `_to_endian`) -
// Marker `Fmt<const BE>` over the shared `GbrpHighBitFrame<'a, BITS, BE>`
// / `GbrapHighBitFrame<'a, BITS, BE>` (the depth is a leading const), so
// these ride the `@const_bits` arm.
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 9, BE; Gbrp9, Gbrp9Sink, GbrpHighBitFrame, YuvOptions,
  |src, opts, sink| gbrp9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 10, BE; Gbrp10, Gbrp10Sink, GbrpHighBitFrame, YuvOptions,
  |src, opts, sink| gbrp10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 12, BE; Gbrp12, Gbrp12Sink, GbrpHighBitFrame, YuvOptions,
  |src, opts, sink| gbrp12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 14, BE; Gbrp14, Gbrp14Sink, GbrpHighBitFrame, YuvOptions,
  |src, opts, sink| gbrp14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 16, BE; Gbrp16, Gbrp16Sink, GbrpHighBitFrame, YuvOptions,
  |src, opts, sink| gbrp16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 10, BE; Gbrap10, Gbrap10Sink, GbrapHighBitFrame, YuvOptions,
  |src, opts, sink| gbrap10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 12, BE; Gbrap12, Gbrap12Sink, GbrapHighBitFrame, YuvOptions,
  |src, opts, sink| gbrap12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 14, BE; Gbrap14, Gbrap14Sink, GbrapHighBitFrame, YuvOptions,
  |src, opts, sink| gbrap14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 16, BE; Gbrap16, Gbrap16Sink, GbrapHighBitFrame, YuvOptions,
  |src, opts, sink| gbrap16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// `Gbrap32` (four `u32` G/B/R/A planes, real alpha) is the full-bit `u32`
// twin of `Gbrap16`. Unlike the bit-depth-parameterized `GbrapN` markers, its
// frame `Gbrap32Frame<'a, BE>` carries only the trailing `BE` const (no
// leading bit-depth), so it rides the `@const BE` arm and delegates to
// `gbrap32_to_endian::<_, BE>` (same shape as `Rgba128` / `Gray32`).
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const BE: bool; Gbrap32<BE>, Gbrap32Sink, Gbrap32Frame, YuvOptions,
  |src, opts, sink| gbrap32_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Planar GBR, MSB-aligned high-bit (BE-generic marker; LE + BE) -----
// MSB-aligned twins of `Gbrp10` / `Gbrp12` — the sample is in the high
// `BITS` bits of each `u16`. Marker `Fmt<const BE>` over the shared
// `GbrpMsbFrame<'a, BITS, BE>` (the depth is a leading const), so these ride
// the `@const_bits` arm and delegate to the const-generic
// `{fmt}_to_endian::<_, BE>`. Three planes, no alpha; reuses [`YuvOptions`].
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 10, BE; Gbrp10Msb, Gbrp10MsbSink, GbrpMsbFrame, YuvOptions,
  |src, opts, sink| gbrp10_msb_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const_bits 12, BE; Gbrp12Msb, Gbrp12MsbSink, GbrpMsbFrame, YuvOptions,
  |src, opts, sink| gbrp12_msb_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Planar GBR, float (Gbrpf16/32, Gbrapf16/32; LE + BE via `_to_endian`)
// Half/single-precision planar GBR (+ alpha for the `Gbrapf*` pair). Marker
// `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>` (no
// leading bit-depth const), so they ride the `@const BE` arm and delegate
// to `{fmt}_to_endian::<_, BE>`; one impl covers both LE and BE. **Unlike**
// the integer GBR families, the float walkers take only `(src, sink)` — no
// `full_range` / `matrix` knobs — so the [`Options`](Walker::Options) is the
// unit type `()`. The `Gbrapf*` alpha plane is read inside the walker for
// the RGBA outputs only, never an `Options` knob.
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const BE: bool; Gbrpf16<BE>, Gbrpf16Sink, Gbrpf16Frame, (),
  |src, _opts, sink| gbrpf16_to_endian::<_, BE>(src, sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const BE: bool; Gbrpf32<BE>, Gbrpf32Sink, Gbrpf32Frame, (),
  |src, _opts, sink| gbrpf32_to_endian::<_, BE>(src, sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const BE: bool; Gbrapf16<BE>, Gbrapf16Sink, Gbrapf16Frame, (),
  |src, _opts, sink| gbrapf16_to_endian::<_, BE>(src, sink));
#[cfg(feature = "gbr")]
#[cfg_attr(docsrs, doc(cfg(feature = "gbr")))]
walker!(@convert @const BE: bool; Gbrapf32<BE>, Gbrapf32Sink, Gbrapf32Frame, (),
  |src, _opts, sink| gbrapf32_to_endian::<_, BE>(src, sink));
