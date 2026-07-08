//! [`MixedSinker`] — the common "I want some subset of {RGB, Luma, HSV}
//! written into my own buffers" consumer.
//!
//! Generic over the source format via an `F: SourceFormat` type
//! parameter. One `PixelSink` impl per supported format. Currently
//! ships impls for:
//!
//! - **8‑bit planar**: [`Yuv411p`](crate::source::Yuv411p),
//!   [`Yuv420p`](crate::source::Yuv420p),
//!   [`Yuv422p`](crate::source::Yuv422p),
//!   [`Yuv440p`](crate::source::Yuv440p),
//!   [`Yuv444p`](crate::source::Yuv444p).
//! - **8‑bit semi‑planar**: [`Nv12`](crate::source::Nv12),
//!   [`Nv21`](crate::source::Nv21), [`Nv16`](crate::source::Nv16),
//!   [`Nv24`](crate::source::Nv24), [`Nv42`](crate::source::Nv42).
//! - **9/10/12/14/16‑bit planar 4:2:0**:
//!   [`Yuv420p9`](crate::source::Yuv420p9),
//!   [`Yuv420p10`](crate::source::Yuv420p10),
//!   [`Yuv420p12`](crate::source::Yuv420p12),
//!   [`Yuv420p14`](crate::source::Yuv420p14),
//!   [`Yuv420p16`](crate::source::Yuv420p16).
//! - **9/10/12/14/16‑bit planar 4:2:2**:
//!   [`Yuv422p9`](crate::source::Yuv422p9),
//!   [`Yuv422p10`](crate::source::Yuv422p10),
//!   [`Yuv422p12`](crate::source::Yuv422p12),
//!   [`Yuv422p14`](crate::source::Yuv422p14),
//!   [`Yuv422p16`](crate::source::Yuv422p16).
//! - **10/12‑bit planar 4:4:0**:
//!   [`Yuv440p10`](crate::source::Yuv440p10),
//!   [`Yuv440p12`](crate::source::Yuv440p12).
//! - **9/10/12/14/16‑bit planar 4:4:4**:
//!   [`Yuv444p9`](crate::source::Yuv444p9),
//!   [`Yuv444p10`](crate::source::Yuv444p10),
//!   [`Yuv444p12`](crate::source::Yuv444p12),
//!   [`Yuv444p14`](crate::source::Yuv444p14),
//!   [`Yuv444p16`](crate::source::Yuv444p16).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:2:0**:
//!   [`P010`](crate::source::P010), [`P012`](crate::source::P012),
//!   [`P016`](crate::source::P016).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:2:2**:
//!   [`P210`](crate::source::P210), [`P212`](crate::source::P212),
//!   [`P216`](crate::source::P216).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:4:4**:
//!   [`P410`](crate::source::P410), [`P412`](crate::source::P412),
//!   [`P416`](crate::source::P416).
//! - **YUVA (alpha-bearing planar)**: the entire FFmpeg-shipped
//!   YUVA family — `Yuva420p` / `Yuva420p9/10/16`, `Yuva422p` /
//!   `Yuva422p9/10/12/16`, `Yuva444p` / `Yuva444p9/10/12/14/16`.
//!   Source-side alpha pass-through to `with_rgba` /
//!   `with_rgba_u16`, with native SIMD on every backend.
//! - **8‑bit packed RGB sources** (Tier 6):
//!   [`Rgb24`](crate::source::Rgb24) (`R, G, B` bytes),
//!   [`Bgr24`](crate::source::Bgr24) (`B, G, R` bytes),
//!   [`Rgba`](crate::source::Rgba) (`R, G, B, A` bytes),
//!   [`Bgra`](crate::source::Bgra) (`B, G, R, A` bytes),
//!   [`Argb`](crate::source::Argb) (`A, R, G, B` bytes — leading alpha),
//!   [`Abgr`](crate::source::Abgr) (`A, B, G, R` bytes — leading alpha),
//!   [`Xrgb`](crate::source::Xrgb) / [`Rgbx`](crate::source::Rgbx) /
//!   [`Xbgr`](crate::source::Xbgr) / [`Bgrx`](crate::source::Bgrx)
//!   (4-byte packed RGB with one ignored padding byte at the leading
//!   or trailing position).
//!   The source row is already 8‑bit RGB at the byte level —
//!   `with_rgb` is an identity copy / channel swap /
//!   drop-alpha-or-padding, `with_rgba` is a memcpy / channel
//!   reorder (alpha passed through for the alpha-bearing 4-byte
//!   sources, forced to `0xFF` for the 3-byte sources and the
//!   padding-byte family), `with_luma` derives Y' from R/G/B,
//!   `with_hsv` reuses the existing kernel.
//! - **8‑bit planar GBR sources** (Tier 10):
//!   [`Gbrp`](crate::source::Gbrp) (three planes: G, B, R) and
//!   [`Gbrap`](crate::source::Gbrap) (four planes: G, B, R, A — real
//!   per-pixel α). Both reuse the standard `with_rgb` / `with_rgba` /
//!   `with_luma` / `with_luma_u16` / `with_hsv` channels via dedicated
//!   `gbr_to_rgb_row` / `gbra_to_rgba_row` / `gbr_to_rgba_opaque_row`
//!   SIMD kernels (no chroma matrix — the source is already component
//!   RGB). `Gbrap`'s `with_rgb + with_rgba` combo uses Strategy A+
//!   (expand RGB → RGBA, then α-overwrite from the source plane).
//! - **10‑bit packed RGB sources** (Tier 6 — Ship 9e):
//!   [`X2Rgb10`](crate::source::X2Rgb10) and
//!   [`X2Bgr10`](crate::source::X2Bgr10). Each pixel is a 32-bit LE word
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

// `Vec<u8>` is only used by the `rgb_scratch` lazy scratch buffer
// (gated on the same 15-feature any as the per-format `process`
// impls). The import is left unconditional because gating it would
// also leave `extern crate alloc as std` unused under
// `--features "alloc"` alone, which is harder to express.
#[allow(unused_imports)]
use std::vec::Vec;

use derive_more::{Display, IsVariant, TryUnwrap, Unwrap};
use thiserror::Error;

use crate::{
  SourceFormat,
  resample::{NoopResampler, ResampleError, ResamplePlan, Resampler},
};
// The RFC #238 averaging-domain selector + caller-configurable transfer
// curve drive the `yuv-planar`-only linear-light path; the fields and
// builders are gated to that family.
#[cfg(feature = "yuv-planar")]
use crate::resample::{AveragingDomain, LinearMode, TransferFunction};
// PixelSink is referenced only via intra-doc links (`[`PixelSink::*`]`)
// in this file; the rustc lint can't see those uses, so silence it.
#[allow(unused_imports)]
use crate::PixelSink;

pub use mediaframe::{
  frame::{WidthAlignment, WidthAlignmentRequirement},
  source::{HsvFrame, HsvFrameMut, HsvPlane},
};

/// Whether `loc` places 4:2:0 chroma at the **horizontal center** between
/// its two luma columns — the MPEG-1 / JPEG phase that needs phase-0.5
/// reconstruction ([`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h))
/// rather than colconv's default nearest-neighbor decode (#302).
///
/// The full [`ChromaLocation`](crate::ChromaLocation) → 4:2:0 sample-phase
/// map (FFmpeg `AVChromaLocation` semantics — horizontal phase in
/// luma-column units, `0` = co-sited with the left column, `0.5` = centered;
/// vertical phase `0` = top luma row, `0.5` = between rows, `1` = bottom):
///
/// | Variant            | Horizontal | Vertical |
/// |--------------------|------------|----------|
/// | `Unspecified`      | default    | default  |
/// | `Unknown`          | default    | default  |
/// | `Left`             | `0`        | `0.5`    |
/// | `Center`           | `0.5`      | `0.5`    |
/// | `TopLeft`          | `0`        | `0`      |
/// | `Top`              | `0.5`      | `0`      |
/// | `BottomLeft`       | `0`        | `1`      |
/// | `Bottom`           | `0.5`      | `1`      |
///
/// This predicate consumes only the **horizontal** axis: it returns `true`
/// for the centered group (`Center` / `Top` / `Bottom`) and `false` for the
/// co-sited group (`Left` / `TopLeft` / `BottomLeft`) and the
/// unspecified/unknown codes. The co-sited and unspecified sitings all keep
/// the byte-identical default path — colconv's existing nearest-neighbor
/// decode already places chroma at the left-column (phase-0) position, so
/// they need no new interpolation.
///
/// This predicate consumes only the **horizontal** axis; the **vertical** phase
/// is a separate concern (see [`chroma_420_bottom_sited_v`]) because mediaframe's
/// 4:2:0 walker streams the sink one chroma row per luma row
/// (`chroma_row = row / 2`), so a vertical blend needs an adjacent chroma row the
/// single-row `process` call does not directly hold.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) const fn chroma_420_center_sited_h(loc: crate::ChromaLocation) -> bool {
  matches!(
    loc,
    crate::ChromaLocation::Center | crate::ChromaLocation::Top | crate::ChromaLocation::Bottom
  )
}

/// Whether `loc` places 4:2:0 chroma at the **bottom** vertical phase (`v = 1`)
/// — the FFmpeg `AVCHROMA_LOC_BOTTOM` (`Bottom`, `h = 0.5`) and
/// `AVCHROMA_LOC_BOTTOMLEFT` (`BottomLeft`, `h = 0`) sitings, whose chroma is
/// co-sited with the **bottom** luma row of each vertical pair; the full
/// [`ChromaLocation`](crate::ChromaLocation) → sample-phase map is in
/// [`chroma_420_center_sited_h`]'s table. At `v = 1` the chroma sample sits on
/// the lower luma row, so:
///
/// - the **even** luma row `2i` lies halfway between chroma rows `i-1` and `i`,
///   and its reconstructed chroma is the vertical box average of those two rows,
///   fused with the siting's horizontal phase —
///   [`chroma_upsample_420_bottom_even_h`](crate::row::scalar::chroma_upsample_420_bottom_even_h)
///   for `Bottom`'s `h = 0.5` centered fold,
///   [`chroma_upsample_420_bottomleft_even_h`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h)
///   for `BottomLeft`'s `h = 0` co-sited (nearest-neighbor) fold;
/// - the **odd** luma row `2i+1` is co-sited with chroma row `i`, so it needs no
///   vertical blend and keeps the siting's plain horizontal upsample.
///
/// The `v = 1` phase is the one vertical siting a top-to-bottom stream can
/// reconstruct with a **backward** one-row chroma lookback: the even row's
/// `c[i-1]` is the *previous* chroma row (already streamed), so the blend is
/// causal. The horizontal axis is independent — `Bottom` rides the centered
/// full-width staging the `h = 0.5` identity path already uses, `BottomLeft`
/// rides the co-sited (nearest-neighbor) full-width staging — so both fold the
/// same vertical box average and this predicate matches both (the 4:2:0 twin of
/// the wider [`chroma_440_bottom_sited_v`] match). The `v = 0` (`Top` /
/// `TopLeft`) and `v = 0.5` (`Center` / `Left`) phases instead need the *next*
/// chroma row for the odd output row — which the row-at-a-time `process` cannot
/// see without a forward delay line — so they keep their vertical-co-sited decode
/// (`Center` / `Top` still apply their horizontal phase via
/// [`chroma_420_center_sited_h`]). `Left` (the `yuv420p` default, `v = 0.5`) and
/// every unspecified siting keep the byte-identical vertical-replicate decode.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) const fn chroma_420_bottom_sited_v(loc: crate::ChromaLocation) -> bool {
  matches!(
    loc,
    crate::ChromaLocation::Bottom | crate::ChromaLocation::BottomLeft
  )
}

/// Whether `loc` places 4:2:0 chroma at the **top** vertical phase (`v = 0`) —
/// the FFmpeg `AVCHROMA_LOC_TOP` (`Top`, `h = 0.5`) and `AVCHROMA_LOC_TOPLEFT`
/// (`TopLeft`, `h = 0`) sitings, whose chroma is co-sited with the **top** luma
/// row of each vertical pair; the full
/// [`ChromaLocation`](crate::ChromaLocation) → sample-phase map is in
/// [`chroma_420_center_sited_h`]'s table. At `v = 0` the chroma sample sits on
/// the upper luma row, so:
///
/// - the **even** luma row `2i` is co-sited with chroma row `i`, so it needs no
///   vertical blend and keeps the siting's plain horizontal upsample;
/// - the **odd** luma row `2i+1` lies halfway between chroma rows `i` and `i+1`,
///   and its reconstructed chroma is the vertical box average of those two rows,
///   fused with the siting's horizontal phase.
///
/// `Top` is the vertical MIRROR of [`chroma_420_bottom_sited_v`]'s `Bottom`
/// (`v = 1`): where Bottom's even row reaches BACKWARD to the already-streamed
/// `c[i-1]`, Top's odd row reaches FORWARD to `c[i+1]` — the *next* chroma row,
/// which a top-to-bottom row-at-a-time stream has not been fed when the odd row
/// arrives. The 4:2:0 sink therefore reconstructs `Top` with a **forward
/// one-row output delay** (the mirror of Bottom's backward `chroma_prev`
/// lookback): the odd output row is held until the next chroma row arrives, then
/// box-blended and emitted — the same `(c[i] + c[i+1] + 1) >> 1` fold Bottom's
/// even row applies to `(c[i-1], c[i])`, reusing its fused-`h` kernels verbatim.
/// The horizontal axis is independent — `Top` rides the centered full-width
/// staging, `TopLeft` the co-sited staging — so both fold the same vertical box
/// average and this predicate matches both. The native / HSV-only binning tiers
/// instead fold the `v = 0` FORWARD triangle into `area_chroma_420`'s vertical
/// weights ([`AxisSpans::area_chroma_phased_v_top`](crate::resample)), needing
/// no delay because the [`AreaStream`](crate::resample) window absorbs the
/// forward reach. `Center` (`h = 0.5, v = 0.5`) / `Left` (the `yuv420p` default,
/// `v = 0.5`) and every unspecified siting keep the byte-identical
/// vertical-replicate decode.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) const fn chroma_420_top_sited_v(loc: crate::ChromaLocation) -> bool {
  matches!(
    loc,
    crate::ChromaLocation::Top | crate::ChromaLocation::TopLeft
  )
}

/// Whether `loc` places 4:4:0 chroma at the **bottom** vertical phase (`v = 1`)
/// — the one vertical siting a top-to-bottom stream can reconstruct with a
/// backward one-row chroma lookback (the even luma row's predecessor chroma row
/// is already streamed).
///
/// 4:4:0 subsamples chroma 2:1 **vertically only** (full-width chroma), so the
/// siting reduces to its vertical axis alone and there is NO horizontal phase to
/// stage. This is the 4:4:0 companion of the 4:2:0 [`chroma_420_bottom_sited_v`]
/// but with a WIDER match: both `Bottom` (`h = 0.5`, `v = 1`) AND `BottomLeft`
/// (`h = 0`, `v = 1`) are `v = 1`, and because 4:4:0 never reconstructs
/// horizontally their horizontal difference is irrelevant — both fold the
/// identical vertical box average. (4:2:0 excludes `BottomLeft` because there the
/// `h = 0` phase would need a co-sited full-width horizontal staging its
/// `h = 0.5` path lacks; 4:4:0 has no such constraint.) The `v = 0` (`Top` /
/// `TopLeft`) phase needs the *next* chroma row and is handled by its FORWARD
/// one-row-delay companion [`chroma_440_top_sited_v`]; the `v = 0.5` (`Center` /
/// `Left`) phase keeps the byte-identical vertical-replicate decode.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) const fn chroma_440_bottom_sited_v(loc: crate::ChromaLocation) -> bool {
  matches!(
    loc,
    crate::ChromaLocation::Bottom | crate::ChromaLocation::BottomLeft
  )
}

/// Whether `loc` places 4:4:0 chroma at the **top** vertical phase (`v = 0`) —
/// the FFmpeg `AVCHROMA_LOC_TOP` (`Top`) and `AVCHROMA_LOC_TOPLEFT` (`TopLeft`)
/// sitings, whose chroma is co-sited with the **top** luma row of each vertical
/// pair. The 4:4:0 companion of the 4:2:0 [`chroma_420_top_sited_v`] and the
/// vertical MIRROR of [`chroma_440_bottom_sited_v`]'s `Bottom` (`v = 1`): where
/// Bottom's even row reaches BACKWARD to the already-streamed `c[i-1]`, Top's
/// odd row reaches FORWARD to `c[i+1]` — the *next* chroma row, which a
/// top-to-bottom row-at-a-time stream has not been fed when the odd row arrives.
///
/// 4:4:0 subsamples chroma 2:1 **vertically only** (full-width chroma), so the
/// siting reduces to its vertical axis alone and there is NO horizontal phase to
/// stage. Both `Top` (`h = 0.5`, `v = 0`) AND `TopLeft` (`h = 0`, `v = 0`) are
/// `v = 0`, and because 4:4:0 never reconstructs horizontally their horizontal
/// difference is irrelevant — both fold the identical vertical box average, so
/// this predicate matches both (the wider match mirrors [`chroma_440_bottom_sited_v`]).
///
/// The 4:4:0 sink reconstructs `Top` with a **forward one-row output delay** (the
/// mirror of Bottom's backward `chroma_prev` lookback): an EVEN luma row `2i` is
/// co-sited with chroma row `i` (a plain co-sited decode), while an ODD luma row
/// `2i+1` needs the vertical box average of chroma rows `i` and `i+1` — so its
/// colour output is held until the following even row supplies `c[i+1]`, then
/// box-blended through the SAME `(c[i] + c[i+1]) / 2` fold a `Bottom` EVEN row
/// applies to `(c[i-1], c[i])` (reused verbatim at the even index). The native /
/// HSV-only binning tiers instead fold the `v = 0` FORWARD triangle into
/// `area_chroma_440`'s vertical weights
/// ([`AxisSpans::area_chroma_phased_v_top`](crate::resample)), needing no delay
/// because the [`AreaStream`](crate::resample) window absorbs the forward reach.
/// `Center` / `Left` (`v = 0.5`) and every unspecified siting keep the
/// byte-identical vertical-replicate decode.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) const fn chroma_440_top_sited_v(loc: crate::ChromaLocation) -> bool {
  matches!(
    loc,
    crate::ChromaLocation::Top | crate::ChromaLocation::TopLeft
  )
}

/// Whether `loc` places 4:2:2 chroma at the **horizontal center** between its
/// two luma columns — the MPEG-1 / JPEG horizontal phase that needs phase-0.5
/// reconstruction
/// ([`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h),
/// the SAME 2:1 horizontal upsample 4:2:0 uses) rather than colconv's default
/// nearest-neighbor decode (#302).
///
/// 4:2:2 subsamples chroma 2:1 **horizontally only** — one chroma row per luma
/// row, no vertical subsampling — so the siting reduces to its horizontal axis
/// alone. This consumes exactly that axis: it returns `true` for the centered
/// group (`Center` / `Top` / `Bottom`, all `h = 0.5`) and `false` for the
/// co-sited group (`Left` / `TopLeft` / `BottomLeft`, all `h = 0`) and the
/// unspecified / unknown codes, which keep the byte-identical nearest-neighbor
/// decode. Unlike 4:2:0 there is no vertical companion predicate: for 4:2:2 the
/// vertical phase drives no chroma reconstruction (the per-axis sample-phase map
/// is in [`chroma_420_center_sited_h`]).
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) const fn chroma_422_center_sited_h(loc: crate::ChromaLocation) -> bool {
  matches!(
    loc,
    crate::ChromaLocation::Center | crate::ChromaLocation::Top | crate::ChromaLocation::Bottom
  )
}

/// Whether `loc` places 4:1:1 chroma at the **horizontal center** of the four
/// luma columns it covers — the MPEG-1 / JPEG horizontal phase that needs the
/// `1→4` phase reconstruction
/// ([`chroma_upsample_4to1_center_h`](crate::row::scalar::chroma_upsample_4to1_center_h))
/// rather than colconv's default nearest-neighbor decode (#302).
///
/// 4:1:1 subsamples chroma 4:1 **horizontally only** — one chroma row per luma
/// row, no vertical subsampling — so the siting reduces to its horizontal axis
/// alone (the 4:1:1 analog of [`chroma_422_center_sited_h`]). This consumes
/// exactly that axis: it returns `true` for the centered group (`Center` / `Top`
/// / `Bottom`, all `h = 0.5` luma = `+1.5` luma from the left-co-sited column)
/// and `false` for the co-sited group (`Left` / `TopLeft` / `BottomLeft`) and
/// the unspecified / unknown codes, which keep the byte-identical nearest decode.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) const fn chroma_411_center_sited_h(loc: crate::ChromaLocation) -> bool {
  matches!(
    loc,
    crate::ChromaLocation::Center | crate::ChromaLocation::Top | crate::ChromaLocation::Bottom
  )
}

/// Whether `loc` places 4:1:0 chroma at the **horizontal center** of the four
/// luma columns it covers — the same `1→4` horizontal phase as 4:1:1
/// ([`chroma_411_center_sited_h`]); 4:1:0 additionally subsamples chroma 4:1
/// **vertically**, but that vertical axis drives no reconstruction here (this
/// consumes the horizontal axis alone, matching the horizontal-only siting this
/// PR wires). Returns `true` for the centered group (`Center` / `Top` /
/// `Bottom`) and `false` for the co-sited / unspecified sitings, which keep the
/// byte-identical nearest decode.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) const fn chroma_410_center_sited_h(loc: crate::ChromaLocation) -> bool {
  matches!(
    loc,
    crate::ChromaLocation::Center | crate::ChromaLocation::Top | crate::ChromaLocation::Bottom
  )
}

// L2 shared chroma-reconstruction stage (RFC #238): the element-generic
// centered-siting reconstruction the `Yuv422p` identity decode funnels through.
// Re-exported here alongside the siting predicates above so the per-format sink
// modules reach it on the same `super` path as `chroma_422_center_sited_h`.
#[cfg(feature = "yuv-planar")]
pub(super) use chroma_reconstruct::{Chroma411U8, ChromaU8, ChromaU16, reconstruct_chroma};

/// Frame dimensions handed to `begin_frame` don't match the sinker's
/// configured size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionMismatch {
  /// Width declared at sinker construction.
  configured_w: usize,
  /// Height declared at sinker construction.
  configured_h: usize,
  /// Width of the frame handed to the walker.
  frame_w: u32,
  /// Height of the frame handed to the walker.
  frame_h: u32,
}

impl DimensionMismatch {
  /// Constructs a new `DimensionMismatch` payload.
  #[inline]
  pub const fn new(configured_w: usize, configured_h: usize, frame_w: u32, frame_h: u32) -> Self {
    Self {
      configured_w,
      configured_h,
      frame_w,
      frame_h,
    }
  }

  /// Width declared at sinker construction.
  #[inline]
  pub const fn configured_w(&self) -> usize {
    self.configured_w
  }

  /// Height declared at sinker construction.
  #[inline]
  pub const fn configured_h(&self) -> usize {
    self.configured_h
  }

  /// Width of the frame handed to the walker.
  #[inline]
  pub const fn frame_w(&self) -> u32 {
    self.frame_w
  }

  /// Height of the frame handed to the walker.
  #[inline]
  pub const fn frame_h(&self) -> u32 {
    self.frame_h
  }
}

/// Generic "insufficient buffer" payload, shared across every
/// `MixedSinkerError::Insufficient*Buffer` variant. `expected` / `actual`
/// are expressed in the unit reported by each variant's Display
/// impl (`bytes` for the byte buffers, `elements` for the typed
/// `u16` / `f32` / `f16` buffers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsufficientBuffer {
  /// Minimum elements (bytes or typed elements) required.
  expected: usize,
  /// Elements supplied.
  actual: usize,
}

impl InsufficientBuffer {
  /// Constructs a new `InsufficientBuffer` payload.
  #[inline]
  pub const fn new(expected: usize, actual: usize) -> Self {
    Self { expected, actual }
  }

  /// Minimum elements (bytes or typed elements) required.
  #[inline]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Elements supplied.
  #[inline]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

/// HSV plane identification and size mismatch payload for
/// [`MixedSinkerError::InsufficientHsvPlane`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsufficientHsvPlane {
  /// Which HSV plane was short (H, S, or V).
  which: HsvPlane,
  /// Minimum bytes required (`width x height`).
  expected: usize,
  /// Bytes supplied.
  actual: usize,
}

impl InsufficientHsvPlane {
  /// Constructs a new `InsufficientHsvPlane` payload.
  #[inline]
  pub const fn new(which: HsvPlane, expected: usize, actual: usize) -> Self {
    Self {
      which,
      expected,
      actual,
    }
  }

  /// Which HSV plane was short (H, S, or V).
  #[inline]
  pub const fn which(&self) -> HsvPlane {
    self.which
  }

  /// Minimum bytes required (`width x height`).
  #[inline]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Bytes supplied.
  #[inline]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

/// Frame-size overflow payload for [`MixedSinkerError::GeometryOverflow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryOverflow {
  /// Configured width.
  width: usize,
  /// Configured height.
  height: usize,
  /// Channel count the overflowing product was computed with.
  channels: usize,
}

impl GeometryOverflow {
  /// Constructs a new `GeometryOverflow` payload.
  #[inline]
  pub const fn new(width: usize, height: usize, channels: usize) -> Self {
    Self {
      width,
      height,
      channels,
    }
  }

  /// Configured width.
  #[inline]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// Configured height.
  #[inline]
  pub const fn height(&self) -> usize {
    self.height
  }

  /// Channel count the overflowing product was computed with.
  #[inline]
  pub const fn channels(&self) -> usize {
    self.channels
  }
}

/// Row shape mismatch payload for [`MixedSinkerError::RowShapeMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowShapeMismatch {
  /// Which slice mismatched. See [`RowSlice`] for variants.
  which: RowSlice,
  /// Row index reported by the offending row.
  row: usize,
  /// Expected slice length in elements of the slice's element type.
  expected: usize,
  /// Actual slice length in the same unit as `expected`.
  actual: usize,
}

impl RowShapeMismatch {
  /// Constructs a new `RowShapeMismatch` payload.
  #[inline]
  pub const fn new(which: RowSlice, row: usize, expected: usize, actual: usize) -> Self {
    Self {
      which,
      row,
      expected,
      actual,
    }
  }

  /// Which slice mismatched. See [`RowSlice`] for variants.
  #[inline]
  pub const fn which(&self) -> RowSlice {
    self.which
  }

  /// Row index reported by the offending row.
  #[inline]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Expected slice length in elements of the slice's element type.
  #[inline]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Actual slice length in the same unit as `expected`.
  #[inline]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

/// Row-index-out-of-range payload for [`MixedSinkerError::RowIndexOutOfRange`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowIndexOutOfRange {
  /// Row index reported by the offending row.
  row: usize,
  /// Sink's configured height.
  configured_height: usize,
}

impl RowIndexOutOfRange {
  /// Constructs a new `RowIndexOutOfRange` payload.
  #[inline]
  pub const fn new(row: usize, configured_height: usize) -> Self {
    Self {
      row,
      configured_height,
    }
  }

  /// Row index reported by the offending row.
  #[inline]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Sink's configured height.
  #[inline]
  pub const fn configured_height(&self) -> usize {
    self.configured_height
  }
}

/// Snapshot of a resampled frame's complete output configuration:
/// presence plus attachment identity (data pointer and length) for
/// luma, luma_u16, rgb, rgba, rgb_u16, rgba_u16, the `rgb_f32` /
/// `rgba_f32` float-RGB(A) outputs, the `xyz_f32` linear-XYZ output,
/// the `rgb_f16` / `rgba_f16` half-float outputs, the `luma_f32`
/// float-luma output, and the three HSV planes. Equality is the
/// per-frame immutability check — in safe code a mid-frame `set_*`
/// necessarily supplies a different borrow, so an identity change is
/// exactly a reattachment. The `*_u16` / `rgb_f32` / `rgba_f32` /
/// `xyz_f32` / `*_f16` / `luma_f32` slots are `(0, 0)` for every format
/// that attaches no such output, so adding them leaves those formats'
/// snapshots unchanged.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FrozenOutputs {
  idents: [(usize, usize); 15],
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
impl FrozenOutputs {
  /// Identity of one attached buffer: `(data pointer, length)`, or
  /// `(0, 0)` for an absent output (a slice pointer is never null).
  fn ident<T>(buf: Option<&[T]>) -> (usize, usize) {
    buf.map_or((0, 0), |b| (b.as_ptr() as usize, b.len()))
  }

  /// Builds the snapshot from the currently attached outputs.
  #[allow(clippy::too_many_arguments)]
  pub(super) fn snapshot(
    luma: Option<&[u8]>,
    luma_u16: Option<&[u16]>,
    rgb: Option<&[u8]>,
    rgba: Option<&[u8]>,
    rgb_u16: Option<&[u16]>,
    rgba_u16: Option<&[u16]>,
    rgb_f32: Option<&[f32]>,
    rgba_f32: Option<&[f32]>,
    xyz_f32: Option<&[f32]>,
    rgb_f16: Option<&[half::f16]>,
    rgba_f16: Option<&[half::f16]>,
    hsv: Option<(&[u8], &[u8], &[u8])>,
    luma_f32: Option<&[f32]>,
  ) -> Self {
    let (h, s, v) = match hsv {
      Some((h, s, v)) => (
        Self::ident(Some(h)),
        Self::ident(Some(s)),
        Self::ident(Some(v)),
      ),
      None => ((0, 0), (0, 0), (0, 0)),
    };
    Self {
      idents: [
        Self::ident(luma),
        Self::ident(luma_u16),
        Self::ident(rgb),
        Self::ident(rgba),
        Self::ident(rgb_u16),
        Self::ident(rgba_u16),
        Self::ident(rgb_f32),
        Self::ident(rgba_f32),
        Self::ident(xyz_f32),
        Self::ident(rgb_f16),
        Self::ident(rgba_f16),
        h,
        s,
        v,
        Self::ident(luma_f32),
      ],
    }
  }
}

/// Compare-only resample preflight (RFC #238 L1): the no-output short-circuit,
/// out-of-sequence rejection, and mid-frame output-set-change rejection every
/// routed row-stage / filter resample arm runs before it allocates a stream —
/// but with **NO commit** (the output freeze is NOT stored). Each arm builds its
/// streams (and any centered-chroma scratch) into locals, then commits the
/// output-set freeze via [`frozen_outputs_check`] and inserts the streams only
/// AFTER every pre-feed allocation has succeeded — so a first-row allocation
/// failure leaves `resample_outputs` and the streams untouched, state-atomic and
/// retryable even with a changed output attachment (the same compare-before /
/// commit-after discipline the linear-light tail uses). It also gates the RFC
/// #238 centered filter / row-stage chroma reserve, so a recoverable reserve
/// failure likewise leaves `resample_outputs` `None`.
///
/// `expected` is supplied by the caller because the per-row sequence counter
/// lives on whichever concrete stream (`AreaStream` / `FilterStream` / the HSV
/// join's Y stream) is fed every row, keeping this helper free of any
/// stream-element type. `ControlFlow::Break` (no output attached) is turned into
/// `Ok(())` by the caller. Rejects mirror the committing path exactly, minus the
/// freeze:
/// [`ResampleOutputsChanged`](MixedSinkerError::ResampleOutputsChanged)
/// on a mid-frame output-set change,
/// [`OutOfSequenceRow`](crate::resample::OutOfSequenceRow) on a row-order
/// violation. Shared with the packed / planar routed paths (this is the
/// "shared transactional resample preflight" the RFC #238 tail-collapse migrates
/// every family onto), so it lives here beside [`frozen_outputs_check`] rather
/// than in the `yuv-planar`-gated `planar_resample` module — the gray / mono
/// 1-channel families run it without pulling in the planar machinery.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(super) fn resample_preflight_check_only(
  resample_outputs: &Option<FrozenOutputs>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  rgb_f32: &Option<&mut [f32]>,
  rgba_f32: &Option<&mut [f32]>,
  xyz_f32: &Option<&mut [f32]>,
  rgb_f16: &Option<&mut [half::f16]>,
  rgba_f16: &Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_f32: &Option<&mut [f32]>,
  expected: Option<usize>,
  idx: usize,
) -> Result<core::ops::ControlFlow<()>, MixedSinkerError> {
  let Some(expected) = expected else {
    return Ok(core::ops::ControlFlow::Break(()));
  };
  // First row only: reject an out-of-sequence row BEFORE the output-set compare,
  // exactly mirroring the committing path's ordering (minus the commit). On a
  // LATER row the compare runs first, so a genuine mid-frame output attachment
  // change surfaces as ResampleOutputsChanged, never OutOfSequenceRow.
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  // Mid-frame output-set change → reject WITHOUT committing (a later row only;
  // the first row's `None` stores no snapshot).
  if let Some(frozen) = resample_outputs {
    let snapshot = FrozenOutputs::snapshot(
      luma.as_deref(),
      luma_u16.as_deref(),
      rgb.as_deref(),
      rgba.as_deref(),
      rgb_u16.as_deref(),
      rgba_u16.as_deref(),
      rgb_f32.as_deref(),
      rgba_f32.as_deref(),
      xyz_f32.as_deref(),
      rgb_f16.as_deref(),
      rgba_f16.as_deref(),
      hsv.as_mut().map(|f| {
        let (h, s, v) = f.hsv();
        (&h[..], &s[..], &v[..])
      }),
      luma_f32.as_deref(),
    );
    if *frozen != snapshot {
      return Err(MixedSinkerError::ResampleOutputsChanged(
        ResampleOutputsChanged::new(idx),
      ));
    }
  }
  // Post-compare: reject an out-of-sequence later row (mirrors
  // the committing path's post-freeze sequence check).
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  Ok(core::ops::ControlFlow::Continue(()))
}

/// Enforces the per-frame frozen output configuration for resampling
/// sinkers — presence AND buffer identity of every output the emit
/// closures consult. Shared by every routed format's resampled paths.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(super) fn frozen_outputs_check(
  resample_outputs: &mut Option<FrozenOutputs>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  rgb_f32: &Option<&mut [f32]>,
  rgba_f32: &Option<&mut [f32]>,
  xyz_f32: &Option<&mut [f32]>,
  rgb_f16: &Option<&mut [half::f16]>,
  rgba_f16: &Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_f32: &Option<&mut [f32]>,
  idx: usize,
) -> Result<(), MixedSinkerError> {
  let snapshot = FrozenOutputs::snapshot(
    luma.as_deref(),
    luma_u16.as_deref(),
    rgb.as_deref(),
    rgba.as_deref(),
    rgb_u16.as_deref(),
    rgba_u16.as_deref(),
    rgb_f32.as_deref(),
    rgba_f32.as_deref(),
    xyz_f32.as_deref(),
    rgb_f16.as_deref(),
    rgba_f16.as_deref(),
    hsv.as_mut().map(|f| {
      let (h, s, v) = f.hsv();
      (&h[..], &s[..], &v[..])
    }),
    luma_f32.as_deref(),
  );
  match resample_outputs {
    None => *resample_outputs = Some(snapshot),
    Some(frozen) if *frozen != snapshot => {
      return Err(MixedSinkerError::ResampleOutputsChanged(
        ResampleOutputsChanged::new(idx),
      ));
    }
    Some(_) => {}
  }
  Ok(())
}

/// Mid-frame output-set change payload for
/// [`MixedSinkerError::ResampleOutputsChanged`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleOutputsChanged {
  /// Source row whose `process` call observed the changed output set.
  row: usize,
}

impl ResampleOutputsChanged {
  /// Constructs a new `ResampleOutputsChanged` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(row: usize) -> Self {
    Self { row }
  }

  /// Source row whose `process` call observed the changed output set.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Mid-frame native-vs-row-stage route change payload for
/// [`MixedSinkerError::NativeRouteChanged`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeRouteChanged {
  /// Source row whose `process` call observed the changed route.
  row: usize,
}

impl NativeRouteChanged {
  /// Constructs a new `NativeRouteChanged` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(row: usize) -> Self {
    Self { row }
  }

  /// Source row whose `process` call observed the changed route.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Payload for [`MixedSinkerError::UnsupportedMatrixResample`]: a non-affine
/// colour matrix was paired with a resampling (non-identity) plan, which the
/// resample tier does not (yet) decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedMatrixResample {
  /// Source row whose `process` call hit the unsupported combination.
  row: usize,
  /// Human-readable name of the offending non-affine matrix
  /// (`"ChromaDerivedCl"` / `"Ictcp"` / `"IptC2"` / `"Smpte2085"`).
  matrix: &'static str,
}

impl UnsupportedMatrixResample {
  /// Constructs a new `UnsupportedMatrixResample` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(row: usize, matrix: &'static str) -> Self {
    Self { row, matrix }
  }

  /// Source row whose `process` call hit the unsupported combination.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Name of the offending non-affine matrix.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> &'static str {
    self.matrix
  }
}

/// Mid-frame [`AveragingDomain::Linear`] transfer-function change payload
/// for [`MixedSinkerError::TransferFunctionChanged`].
///
/// [`AveragingDomain::Linear`]: crate::resample::AveragingDomain::Linear
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferFunctionChanged {
  /// Source row whose `process` call observed the changed transfer function.
  row: usize,
}

impl TransferFunctionChanged {
  /// Constructs a new `TransferFunctionChanged` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(row: usize) -> Self {
    Self { row }
  }

  /// Source row whose `process` call observed the changed transfer function.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Mid-frame [`AveragingDomain::Linear`] [`LinearMode`] change payload for
/// [`MixedSinkerError::LinearModeChanged`].
///
/// [`AveragingDomain::Linear`]: crate::resample::AveragingDomain::Linear
/// [`LinearMode`]: crate::resample::LinearMode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearModeChanged {
  /// Source row whose `process` call observed the changed linear mode.
  row: usize,
}

impl LinearModeChanged {
  /// Constructs a new `LinearModeChanged` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(row: usize) -> Self {
    Self { row }
  }

  /// Source row whose `process` call observed the changed linear mode.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Mid-frame [`AveragingDomain`] change payload for
/// [`MixedSinkerError::AveragingDomainChanged`].
///
/// [`AveragingDomain`]: crate::resample::AveragingDomain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AveragingDomainChanged {
  /// Source row whose `process` call observed the changed averaging domain.
  row: usize,
}

impl AveragingDomainChanged {
  /// Constructs a new `AveragingDomainChanged` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(row: usize) -> Self {
    Self { row }
  }

  /// Source row whose `process` call observed the changed averaging domain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Mid-frame 4:2:2 chroma-siting (sampling-phase) change payload for
/// [`MixedSinkerError::ChromaSitingChanged`].
///
/// RFC #238 routes chroma siting through the resample as a sampling
/// phase frozen on the first output-bearing row; a later row that
/// observes a different siting would bin a mixture of phases, so it is
/// rejected with this recoverable error (the caller restarts the frame
/// via `begin_frame` to change siting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChromaSitingChanged {
  /// Source row whose `process` call observed the changed chroma siting.
  row: usize,
}

impl ChromaSitingChanged {
  /// Constructs a new `ChromaSitingChanged` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(row: usize) -> Self {
    Self { row }
  }

  /// Source row whose `process` call observed the changed chroma siting.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Errors returned by [`MixedSinker`] configuration and per-frame
/// preflight.
///
/// All variants are recoverable: the sinker never mutates caller
/// buffers on an error return, so the caller can inspect the variant,
/// rebuild or resize buffers, and retry.
///
/// **Note (API change):** the former `*BufferTooShort` / `HsvPlaneTooShort`
/// variants were renamed to `Insufficient*Buffer` / `InsufficientHsvPlane`
/// and their payload structs renamed from `BufferTooShort` /
/// `HsvPlaneTooShort` to `InsufficientBuffer` / `InsufficientHsvPlane`.
// No `Eq`: the wrapped `ResampleError` carries an `f64` (the filter
// kernel support in `InvalidFilterSupport`), so it is `PartialEq` only.
// Nothing requires `Eq` on this type.
#[derive(Debug, Clone, Copy, PartialEq, IsVariant, TryUnwrap, Unwrap, Error)]
#[non_exhaustive]
pub enum MixedSinkerError {
  /// The frame handed to the walker does not match the dimensions
  /// declared at [`MixedSinker::new`]. Returned from
  /// [`PixelSink::begin_frame`] before any row is processed.
  #[error(
    "MixedSinker frame dimensions mismatch: configured {}x{} but got {}x{}",
    .0.configured_w(), .0.configured_h(), .0.frame_w(), .0.frame_h()
  )]
  DimensionMismatch(DimensionMismatch),

  /// RGB buffer attached via [`MixedSinker::with_rgb`] /
  /// [`MixedSinker::set_rgb`] is shorter than `width x height x 3`.
  #[error("MixedSinker insufficient rgb buffer: expected >= {} bytes, got {}", .0.expected(), .0.actual())]
  InsufficientRgbBuffer(InsufficientBuffer),

  /// `u16` RGB buffer attached via [`MixedSinker::with_rgb_u16`] /
  /// [`MixedSinker::set_rgb_u16`] is shorter than `width x height x 3`
  /// `u16` elements. Only the high‑bit‑depth source impls
  /// (currently [`Yuv420p10`](crate::source::Yuv420p10)) write into this
  /// buffer.
  #[error("MixedSinker insufficient rgb_u16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbU16Buffer(InsufficientBuffer),

  /// Native-depth `u16` luma buffer attached via per-format
  /// `with_luma_u16` is shorter than `width x height` `u16`
  /// elements. Tier 4 sources (V210 / Y210 / Y212 / Y216) are the
  /// first consumers of this API.
  #[error("MixedSinker insufficient luma_u16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientLumaU16Buffer(InsufficientBuffer),

  /// RGBA buffer attached via [`MixedSinker::with_rgba`] /
  /// [`MixedSinker::set_rgba`] is shorter than `width x height x 4`.
  /// The fourth byte per pixel is alpha — opaque (`0xFF`) by default
  /// when the source has no alpha plane.
  #[error("MixedSinker insufficient rgba buffer: expected >= {} bytes, got {}", .0.expected(), .0.actual())]
  InsufficientRgbaBuffer(InsufficientBuffer),

  /// `u16` RGBA buffer attached via `with_rgba_u16` / `set_rgba_u16`
  /// (per-format impl, not yet shipped on any sink) is shorter than
  /// `width x height x 4` `u16` elements. Only high‑bit‑depth source
  /// impls write into this buffer; the fourth `u16` per pixel is
  /// alpha — opaque (`(1 << BITS) - 1`) by default when the source
  /// has no alpha plane.
  #[error("MixedSinker insufficient rgba_u16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbaU16Buffer(InsufficientBuffer),

  /// `f32` RGB buffer attached via per-format `with_rgb_f32` /
  /// `set_rgb_f32` is shorter than `width x height x 3` `f32` elements.
  /// Only float-source impls (currently
  /// [`Rgbf32`](crate::source::Rgbf32)) write into this buffer.
  #[error("MixedSinker insufficient rgb_f32 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbF32Buffer(InsufficientBuffer),

  /// `half::f16` RGB buffer attached via per-format `with_rgb_f16` /
  /// `set_rgb_f16` is shorter than `width x height x 3` `f16` elements.
  /// Only half-float-source impls (currently
  /// [`Rgbf16`](crate::source::Rgbf16)) write into this buffer.
  #[error("MixedSinker insufficient rgb_f16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbF16Buffer(InsufficientBuffer),

  /// `f32` RGBA buffer attached via per-format `with_rgba_f32` /
  /// `set_rgba_f32` is shorter than `width x height x 4` `f32` elements.
  /// Only float-planar-GBR source impls write into this buffer.
  #[error("MixedSinker insufficient rgba_f32 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbaF32Buffer(InsufficientBuffer),

  /// `half::f16` RGBA buffer attached via per-format `with_rgba_f16` /
  /// `set_rgba_f16` is shorter than `width x height x 4` `f16` elements.
  /// Only float-planar-GBR source impls write into this buffer.
  #[error("MixedSinker insufficient rgba_f16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbaF16Buffer(InsufficientBuffer),

  /// `f32` XYZ buffer attached via `with_xyz_f32` / `set_xyz_f32` is
  /// shorter than `width x height x 3` `f32` elements. Only the
  /// [`Xyz12`](crate::source::Xyz12) source impl writes into this buffer.
  #[error("MixedSinker insufficient xyz_f32 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientXyzF32Buffer(InsufficientBuffer),

  /// `f32` luma buffer attached via `with_luma_f32` / `set_luma_f32` is
  /// shorter than `width x height` `f32` elements.
  #[error("MixedSinker insufficient luma_f32 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientLumaF32Buffer(InsufficientBuffer),

  /// Luma buffer is shorter than `width x height`.
  #[error("MixedSinker insufficient luma buffer: expected >= {} bytes, got {}", .0.expected(), .0.actual())]
  InsufficientLumaBuffer(InsufficientBuffer),

  /// One of the three HSV planes is shorter than `width x height`.
  #[error("MixedSinker insufficient hsv {:?} plane: expected >= {} bytes, got {}", .0.which(), .0.expected(), .0.actual())]
  InsufficientHsvPlane(InsufficientHsvPlane),

  /// Declared frame geometry does not fit in `usize`. Only reachable
  /// on 32‑bit targets (wasm32, i686) with extreme dimensions.
  #[error("MixedSinker frame size overflows usize: {} x {} x channels={}", .0.width(), .0.height(), .0.channels())]
  GeometryOverflow(GeometryOverflow),

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
    "MixedSinker row shape mismatch at row {}: {} slice has {} elements, expected {}",
    .0.row(), .0.which(), .0.actual(), .0.expected()
  )]
  RowShapeMismatch(RowShapeMismatch),

  /// A row handed to [`PixelSink::process`] has `row.row() >=
  /// configured_height`. The walker bounds `idx < height` via its
  /// `for row in 0..h` loop combined with the `begin_frame`
  /// dimension check, but a direct caller could pass any value.
  /// Returning an error instead of slice-indexing past the end keeps
  /// the no-panic contract intact.
  #[error(
    "MixedSinker row index {} is out of range for configured height {}",
    .0.row(), .0.configured_height()
  )]
  RowIndexOutOfRange(RowIndexOutOfRange),

  /// The sinker's configured `width` violates the format's chroma-group
  /// stride requirement. For 4:2:0 / 4:2:2 formats the width must be
  /// even (`WidthAlignmentRequirement::Even`); for planar 4:1:0
  /// ([`Yuv410p`](crate::source::Yuv410p)) and packed 4:1:1
  /// ([`Uyyvyy411`](crate::source::Uyyvyy411)) the width must be a
  /// multiple of 4 (`WidthAlignmentRequirement::MultipleOfFour`).
  /// Planar 4:1:1 ([`Yuv411p`](crate::source::Yuv411p)) accepts
  /// non-4-aligned widths via `width.div_ceil(4)` and does not produce
  /// this error. Supersedes the former `OddWidth` (even-only) and
  /// `WidthNotMultipleOf4` variants.
  ///
  /// `MixedSinker::new` is infallible and accepts any width, so this error
  /// surfaces the misconfiguration at the first use site
  /// ([`PixelSink::begin_frame`] or [`PixelSink::process`]) before any row
  /// primitive is invoked, preserving the no-panic contract.
  #[error("MixedSinker configured width {} {}", .0.width(), .0.required())]
  WidthAlignment(WidthAlignment),

  /// Building the resampling plan failed in
  /// [`MixedSinker::with_resampler`]: the strategy rejected the
  /// requested output geometry. Surfaces before the sinker exists, so
  /// no buffer state is touched.
  #[error(transparent)]
  Resample(#[from] ResampleError),

  /// On a resampling sinker the attached-output configuration is
  /// frozen per frame: streams carry frame progress, so an output
  /// attached, detached, or **replaced with a different buffer**
  /// after the first processed row would silently miss (or split)
  /// the rows already finalized. The offending `process` call fails
  /// before any stream mutates caller output; re-attach and call
  /// [`PixelSink::begin_frame`] to restart the frame with the new
  /// configuration. The direct (identity) path is unaffected.
  #[error(
    "MixedSinker resampled output set changed mid-frame at source row {}; \
     restart the frame via begin_frame",
    .0.row()
  )]
  ResampleOutputsChanged(ResampleOutputsChanged),

  /// On a resampling sinker whose source format carries the native
  /// decimation tier (the high-bit semi-planar `P010` / `P012` /
  /// `P016` family), the native-vs-row-stage route is frozen per frame
  /// alongside the output set: the route is chosen on the first
  /// resampled row, and the two tiers carry independent, in-order
  /// stream state. A caller manually driving `process()` that flips
  /// [`MixedSinker::set_native`] mid-frame would feed later rows through
  /// the *other* tier's fresh streams, splitting one frame across two
  /// incompatible state machines. The offending `process` call fails
  /// before either tier consumes the row; re-select the tier and call
  /// [`PixelSink::begin_frame`] to restart the frame. Enforced by every
  /// native 4:2:0 family: the 8-bit / high-bit planar (Yuv420p /
  /// Yuv420p10/12/14/16) and 8-bit / high-bit semi-planar (Nv12 / Nv21 /
  /// P010/P012/P016) tiers.
  #[error(
    "MixedSinker resample route (native vs row-stage) changed mid-frame at \
     source row {}; restart the frame via begin_frame",
    .0.row()
  )]
  NativeRouteChanged(NativeRouteChanged),

  /// The [`AveragingDomain::Linear`] transfer function changed mid-frame.
  ///
  /// The linear-light tail buffers each source row linearised under the
  /// transfer function resolved on the first output-bearing row; a later
  /// row resolving a different transfer (a caller flipping
  /// [`MixedSinker::with_transfer_function`] or the source `ColorMatrix`
  /// mid-frame) would bin rows linearised under inconsistent curves. The
  /// offending `process` call fails before the row is consumed; restore the
  /// transfer and call [`PixelSink::begin_frame`] to restart the frame.
  ///
  /// [`AveragingDomain::Linear`]: crate::resample::AveragingDomain::Linear
  #[error(
    "MixedSinker Linear transfer function changed mid-frame at source row \
     {}; restart the frame via begin_frame",
    .0.row()
  )]
  TransferFunctionChanged(TransferFunctionChanged),

  /// The [`AveragingDomain::Linear`] [`LinearMode`] changed mid-frame.
  ///
  /// The linear-light tail picks the `YUV→RGB` decode — display-referred
  /// (Q15-clamped) or scene-referred (unclamped `f32`) — from the
  /// [`LinearMode`] resolved on the first output-bearing row and buffers every
  /// source row decoded under it; a later row resolving a different mode (a
  /// caller flipping [`MixedSinker::with_linear_mode`] / [`MixedSinker::set_linear_mode`]
  /// mid-frame) would bin rows decoded under inconsistent referents, mixing
  /// display- and scene-decoded rows in one frame. The offending `process` call
  /// fails before the row is consumed; restore the mode and call
  /// [`PixelSink::begin_frame`] to restart the frame.
  ///
  /// [`AveragingDomain::Linear`]: crate::resample::AveragingDomain::Linear
  /// [`LinearMode`]: crate::resample::LinearMode
  #[error(
    "MixedSinker Linear mode changed mid-frame at source row \
     {}; restart the frame via begin_frame",
    .0.row()
  )]
  LinearModeChanged(LinearModeChanged),

  /// The [`AveragingDomain`] chosen on the first output-bearing resampled row
  /// is frozen for the frame, alongside the output set, the native-vs-row-stage
  /// route, and (for [`AveragingDomain::Linear`]) the transfer function and the
  /// [`LinearMode`]: the domain selects the pipeline stage the area average is
  /// spliced at, so a
  /// caller flipping [`MixedSinker::set_averaging_domain`] mid-frame (between
  /// `process` rows, without [`PixelSink::begin_frame`]) would split one frame
  /// across two incompatible averaging stages. The offending `process` call
  /// fails before any stream consumes the row or writes caller output; restore
  /// the domain and call [`PixelSink::begin_frame`] to restart the frame. The
  /// freeze is enforced by the planar 8-bit YUV family (Yuv420p / Yuv422p /
  /// Yuv444p / Yuv440p), the formats whose dispatch reads the configured domain.
  ///
  /// [`AveragingDomain`]: crate::resample::AveragingDomain
  /// [`AveragingDomain::Linear`]: crate::resample::AveragingDomain::Linear
  #[error(
    "MixedSinker averaging domain changed mid-frame at source row \
     {}; restart the frame via begin_frame",
    .0.row()
  )]
  AveragingDomainChanged(AveragingDomainChanged),

  /// The 4:2:2 chroma siting (sampling phase) changed mid-frame. RFC #238
  /// freezes the effective siting on the first output-bearing row of a
  /// frame so every row bins the same phase; restart the frame via
  /// begin_frame to change siting.
  #[error(
    "MixedSinker chroma siting changed mid-frame at source row \
     {}; restart the frame via begin_frame",
    .0.row()
  )]
  ChromaSitingChanged(ChromaSitingChanged),

  /// A non-affine colour matrix was paired with a resampling (non-identity)
  /// plan, which the resample tier cannot decode.
  ///
  /// The constant-luminance [`ColorMatrix::ChromaDerivedCl`] (BT.2020
  /// `YcCbcCrc`), [`ColorMatrix::Ictcp`] (BT.2100), [`ColorMatrix::IptC2`]
  /// (IPT-C2 / Dolby Vision Profile 5), and [`ColorMatrix::Smpte2085`]
  /// (SMPTE ST 2085 "Y'D'zD'x" X'Y'Z') decodes are non-affine: they
  /// reconstruct RGB through a per-channel non-linear transfer, not a single
  /// Q15 matrix, and are wired only on the **identity** (no-resample) path. The
  /// resample tier averages in the encoded-code / native-code domain and
  /// decodes through the *affine* `yuv444p*_to_rgb*` kernels, which would
  /// silently produce wrong, route-dependent colour for these matrices (the
  /// averaged result is not the constant-luminance / ICtCp / IPT-C2 / SMPTE
  /// 2085 decode).
  /// Rather than emit corrupt output, the offending `process` call fails before
  /// any stream consumes the row. Decode these matrices on an identity sink (no
  /// resample plan), or downscale after the colour convert. Full non-affine
  /// resample support is tracked as follow-up.
  ///
  /// Only raised when the matrix actually **resolves** to its non-affine
  /// decode (BT.2020 primaries for `ChromaDerivedCl`, a PQ/HLG transfer for
  /// `Ictcp`, a PQ transfer for `IptC2` / `Smpte2085`); an unresolved tag
  /// already falls back to the affine path on the identity route, so it
  /// resamples affinely without error, unchanged.
  ///
  /// [`ColorMatrix::ChromaDerivedCl`]: crate::ColorMatrix::ChromaDerivedCl
  /// [`ColorMatrix::Ictcp`]: crate::ColorMatrix::Ictcp
  /// [`ColorMatrix::IptC2`]: crate::ColorMatrix::IptC2
  /// [`ColorMatrix::Smpte2085`]: crate::ColorMatrix::Smpte2085
  #[error(
    "MixedSinker cannot resample the non-affine {} matrix at source row {}; \
     decode on an identity sink or downscale after the colour convert",
    .0.matrix(),
    .0.row()
  )]
  UnsupportedMatrixResample(UnsupportedMatrixResample),
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
  /// Quarter‑width U (Cb) plane in a planar 4:1:1 / 4:1:0 source
  /// ([`Yuv411p`](crate::source::Yuv411p) — DV-NTSC legacy;
  /// [`Yuv410p`](crate::source::Yuv410p) — Cinepak / extreme-old codecs).
  /// `width.div_ceil(4)` bytes per row — each chroma sample covers
  /// four Y columns horizontally. Yuv410p enforces `width % 4 == 0`
  /// at the frame layer (so `width.div_ceil(4) == width / 4`); Yuv411p
  /// accepts arbitrary widths via FFmpeg ceiling chroma. In 4:1:0 the
  /// same chroma row also covers four consecutive Y rows vertically;
  /// in 4:1:1 chroma is full-height.
  #[display("U Quarter")]
  UQuarter,
  /// Quarter‑width V (Cr) plane in a planar 4:1:1 / 4:1:0 source
  /// ([`Yuv411p`](crate::source::Yuv411p) /
  /// [`Yuv410p`](crate::source::Yuv410p)). `width.div_ceil(4)` bytes per
  /// row (see [`Self::UQuarter`] for the Yuv410p-vs-Yuv411p
  /// width-rounding distinction).
  #[display("V Quarter")]
  VQuarter,
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
  /// ([`Yuv444p`](crate::source::Yuv444p)). `width` bytes per row.
  #[display("U Full")]
  UFull,
  /// Full-width V (Cr) plane in a planar 4:4:4 source
  /// ([`Yuv444p`](crate::source::Yuv444p)). `width` bytes per row.
  #[display("V Full")]
  VFull,
  /// Full-width alpha plane in an 8‑bit YUVA source
  /// ([`Yuva420p`](crate::source::Yuva420p)). `width` bytes per row
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
  /// alpha plane ([`Yuva444p10`](crate::source::Yuva444p10)). `u16`
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
  /// ([`Yuva422p12`](crate::source::Yuva422p12) /
  /// [`Yuva444p12`](crate::source::Yuva444p12)). `u16` samples, `width`
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
  /// ([`Yuva444p14`](crate::source::Yuva444p14)). `u16` samples, `width`
  /// elements, low-bit-packed.
  #[display("A Full 14")]
  AFull14,
  /// Full‑width interleaved UV plane in a semi‑planar **4:4:4** source
  /// ([`Nv24`](crate::source::Nv24)). Each row is `U0, V0, U1, V1, …` for
  /// `width` pairs (`2 * width` bytes). One UV pair per Y pixel — no
  /// chroma subsampling.
  #[display("UV Full")]
  UvFull,
  /// Full‑width interleaved VU plane in a semi‑planar **4:4:4** source
  /// ([`Nv42`](crate::source::Nv42)). Each row is `V0, U0, V1, U1, …` for
  /// `width` pairs — byte order swapped relative to [`Self::UvFull`].
  #[display("VU Full")]
  VuFull,
  /// Full‑width Y row of a **9‑bit** planar source
  /// ([`Yuv420p9`](crate::source::Yuv420p9) /
  /// [`Yuv422p9`](crate::source::Yuv422p9) /
  /// [`Yuv444p9`](crate::source::Yuv444p9)). `u16` samples, `width`
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
  /// ([`Yuva420p9`](crate::source::Yuva420p9)). `u16` samples, `width`
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
  /// planar ([`Yuv420p16`](crate::source::Yuv420p16)) and semi‑planar
  /// ([`P016`](crate::source::P016)) families. At 16 bits there is no
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
  /// ([`P016`](crate::source::P016)). `u16` samples, `width` elements.
  #[display("UV Half 16")]
  UvHalf16,
  /// Full-width alpha row of a **16-bit** YUVA planar source
  /// ([`Yuva420p16`](crate::source::Yuva420p16) /
  /// [`Yuva444p16`](crate::source::Yuva444p16)). `u16` samples,
  /// `width` elements (full u16 range).
  #[display("A Full 16")]
  AFull16,
  /// Full-width U row of a **16-bit** 4:4:4 planar source
  /// ([`Yuv444p16`](crate::source::Yuv444p16) /
  /// [`Yuva444p16`](crate::source::Yuva444p16)). `u16` samples,
  /// `width` elements (full u16 range).
  #[display("U Full 16")]
  UFull16,
  /// Full-width V row of a **16-bit** 4:4:4 planar source. `u16`
  /// samples, `width` elements (full u16 range).
  #[display("V Full 16")]
  VFull16,
  /// Full‑width interleaved UV row of a **10‑bit semi‑planar 4:4:4**
  /// source ([`P410`](crate::source::P410)). `u16` samples, `2 * width`
  /// elements, high‑bit‑packed.
  #[display("UV Full 10")]
  UvFull10,
  /// Full‑width interleaved UV row of a **12‑bit semi‑planar 4:4:4**
  /// source ([`P412`](crate::source::P412)). `u16` samples, `2 * width`
  /// elements, high‑bit‑packed.
  #[display("UV Full 12")]
  UvFull12,
  /// Full‑width interleaved UV row of a **16‑bit semi‑planar 4:4:4**
  /// source ([`P416`](crate::source::P416)). `u16` samples, `2 * width`
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
  /// Pixel-index row of a [`Pal8`](crate::raw::Pal8) source.
  /// `u8` samples, `width` elements — each value is an index into
  /// the 256-entry BGRA palette carried alongside in `Pal8Row`.
  #[display("Pal8 index row")]
  Pal8IndexRow,
  /// Packed **RGB565** LE row of an [`Rgb565`](crate::source::Rgb565) source.
  /// `2 * width` `u8` bytes — one `u16` LE word per pixel.
  #[display("RGB565 packed")]
  Rgb565Packed,
  /// Packed **BGR565** LE row of a [`Bgr565`](crate::source::Bgr565) source.
  /// Same `2 * width` byte shape as [`Rgb565Packed`](Self::Rgb565Packed)
  /// with R↔B channel positions swapped.
  #[display("BGR565 packed")]
  Bgr565Packed,
  /// Packed **RGB555** LE row of an [`Rgb555`](crate::source::Rgb555) source.
  /// `2 * width` `u8` bytes — one `u16` LE word per pixel (bit 15 unused).
  #[display("RGB555 packed")]
  Rgb555Packed,
  /// Packed **BGR555** LE row of a [`Bgr555`](crate::source::Bgr555) source.
  /// Same shape as [`Rgb555Packed`](Self::Rgb555Packed) with R↔B swapped.
  #[display("BGR555 packed")]
  Bgr555Packed,
  /// Packed **RGB444** LE row of an [`Rgb444`](crate::source::Rgb444) source.
  /// `2 * width` `u8` bytes — one `u16` LE word per pixel (bits [15:12]
  /// unused).
  #[display("RGB444 packed")]
  Rgb444Packed,
  /// Packed **BGR444** LE row of a [`Bgr444`](crate::source::Bgr444) source.
  /// Same shape as [`Rgb444Packed`](Self::Rgb444Packed) with R↔B swapped.
  #[display("BGR444 packed")]
  Bgr444Packed,
  /// Packed **RGB8** row of an [`Rgb8`](crate::source::Rgb8) source.
  /// `width` `u8` bytes — one 3:3:2 pixel per byte.
  #[display("RGB8 packed")]
  Rgb8Packed,
  /// Packed **BGR8** row of a [`Bgr8`](crate::source::Bgr8) source.
  /// Same `width` byte shape as [`Rgb8Packed`](Self::Rgb8Packed) with the
  /// 3:3:2 channel positions in B, G, R order.
  #[display("BGR8 packed")]
  Bgr8Packed,
  /// Packed **RGB4_BYTE** row of an [`Rgb4Byte`](crate::source::Rgb4Byte)
  /// source. `width` `u8` bytes — one 1:2:1 pixel in each low nibble
  /// (the high nibble is unused padding).
  #[display("RGB4_BYTE packed")]
  Rgb4BytePacked,
  /// Packed **BGR4_BYTE** row of a [`Bgr4Byte`](crate::source::Bgr4Byte)
  /// source. Same shape as [`Rgb4BytePacked`](Self::Rgb4BytePacked) with
  /// R↔B swapped.
  #[display("BGR4_BYTE packed")]
  Bgr4BytePacked,
  /// Packed **RGB4** row of an [`Rgb4`](crate::source::Rgb4) source.
  /// `width.div_ceil(2)` `u8` bytes — two 1:2:1 pixels per byte (the even
  /// pixel in the high nibble).
  #[display("RGB4 packed")]
  Rgb4Packed,
  /// Packed **BGR4** row of a [`Bgr4`](crate::source::Bgr4) source.
  /// Same shape as [`Rgb4Packed`](Self::Rgb4Packed) with R↔B swapped.
  #[display("BGR4 packed")]
  Bgr4Packed,
  /// Packed `R, G, B` row of an [`Rgb24`](crate::source::Rgb24) source.
  /// `3 * width` `u8` bytes.
  #[display("RGB packed")]
  RgbPacked,
  /// Packed `B, G, R` row of a [`Bgr24`](crate::source::Bgr24) source.
  /// `3 * width` `u8` bytes (channel-order swapped relative to
  /// [`RgbPacked`](Self::RgbPacked)).
  #[display("BGR packed")]
  BgrPacked,
  /// Packed `R, G, B, A` row of an [`Rgba`](crate::source::Rgba) source.
  /// `4 * width` `u8` bytes — alpha is real (not padding).
  #[display("RGBA packed")]
  RgbaPacked,
  /// Packed `B, G, R, A` row of a [`Bgra`](crate::source::Bgra) source.
  /// `4 * width` `u8` bytes — alpha lane preserved, channel order
  /// swapped on the first three bytes relative to
  /// [`RgbaPacked`](Self::RgbaPacked).
  #[display("BGRA packed")]
  BgraPacked,
  /// Packed `A, R, G, B` row of an [`Argb`](crate::source::Argb) source.
  /// `4 * width` `u8` bytes — alpha at the **leading** position vs
  /// [`RgbaPacked`](Self::RgbaPacked).
  #[display("ARGB packed")]
  ArgbPacked,
  /// Packed `A, B, G, R` row of an [`Abgr`](crate::source::Abgr) source.
  /// `4 * width` `u8` bytes — leading alpha + reversed RGB order vs
  /// [`ArgbPacked`](Self::ArgbPacked).
  #[display("ABGR packed")]
  AbgrPacked,
  /// Packed `X, R, G, B` row of an [`Xrgb`](crate::source::Xrgb) source
  /// (FFmpeg `0rgb`). `4 * width` `u8` bytes — leading **padding**
  /// byte (not alpha).
  #[display("XRGB packed")]
  XrgbPacked,
  /// Packed `R, G, B, X` row of an [`Rgbx`](crate::source::Rgbx) source
  /// (FFmpeg `rgb0`). `4 * width` `u8` bytes — trailing padding byte.
  #[display("RGBX packed")]
  RgbxPacked,
  /// Packed `X, B, G, R` row of an [`Xbgr`](crate::source::Xbgr) source
  /// (FFmpeg `0bgr`). `4 * width` `u8` bytes — leading padding byte
  /// + reversed RGB order vs [`XrgbPacked`](Self::XrgbPacked).
  #[display("XBGR packed")]
  XbgrPacked,
  /// Packed `B, G, R, X` row of a [`Bgrx`](crate::source::Bgrx) source
  /// (FFmpeg `bgr0`). `4 * width` `u8` bytes — trailing padding byte
  /// + reversed RGB order vs [`RgbxPacked`](Self::RgbxPacked).
  #[display("BGRX packed")]
  BgrxPacked,
  /// Packed `X2RGB10` LE row of an
  /// [`X2Rgb10`](crate::source::X2Rgb10) source. `4 * width` `u8` bytes
  /// (one little-endian `u32` per pixel with `(MSB) 2X | 10R | 10G |
  /// 10B (LSB)` packing).
  #[display("X2RGB10 packed")]
  X2Rgb10Packed,
  /// Packed `X2BGR10` LE row of an
  /// [`X2Bgr10`](crate::source::X2Bgr10) source. `4 * width` `u8` bytes
  /// — channel positions reversed relative to
  /// [`X2Rgb10Packed`](Self::X2Rgb10Packed).
  #[display("X2BGR10 packed")]
  X2Bgr10Packed,
  /// Packed `Y0, U0, Y1, V0, …` row of a
  /// [`Yuyv422`](crate::source::Yuyv422) source (FFmpeg `yuyv422` /
  /// YUY2). `2 * width` `u8` bytes — Y in even byte positions, U/V
  /// in odd positions with U preceding V.
  #[display("YUYV422 packed")]
  Yuyv422Packed,
  /// Packed `U0, Y0, V0, Y1, …` row of a
  /// [`Uyvy422`](crate::source::Uyvy422) source (FFmpeg `uyvy422` /
  /// UYVY). `2 * width` `u8` bytes — Y in odd byte positions, U/V
  /// in even positions with U preceding V.
  #[display("UYVY422 packed")]
  Uyvy422Packed,
  /// Packed `Y0, V0, Y1, U0, …` row of a
  /// [`Yvyu422`](crate::source::Yvyu422) source (FFmpeg `yvyu422` /
  /// YVYU). `2 * width` `u8` bytes — Y in even byte positions, V/U
  /// in odd positions with V preceding U (chroma order swapped vs
  /// [`Yuyv422Packed`](Self::Yuyv422Packed)).
  #[display("YVYU422 packed")]
  Yvyu422Packed,
  /// Packed `U0, Y0, Y1, V0, Y2, Y3, …` row of a
  /// [`Uyyvyy411`](crate::source::Uyyvyy411) source (FFmpeg
  /// `uyyvyy411`). `width * 3 / 2` `u8` bytes — one (U, V) chroma
  /// pair shared across 4 luma samples (4:1:1 horizontal
  /// subsampling, 12 bpp, DV legacy).
  #[display("UYYVYY411 packed")]
  Uyyvyy411Packed,
  /// Packed `v210` row of a [`V210`](crate::source::V210) source —
  /// Tier 4 10-bit pro-broadcast SDI capture format. Each 16-byte
  /// word holds 12 x 10-bit samples = 6 pixels (4:2:2: 6 Y +
  /// 3 Cb + 3 Cr). Row length: `(width / 6) * 16` `u8` bytes.
  #[display("V210 packed")]
  V210Packed,
  /// Packed `y210` row of a [`Y210`](crate::source::Y210) source —
  /// Tier 4 10-bit MSB-aligned in u16 with YUYV422 byte order.
  /// Row length: `2 * width` `u16` elements (= `4 * width` bytes).
  #[display("Y210 packed")]
  Y210Packed,
  /// Packed `y212` row — same shape as Y210 with BITS=12.
  #[display("Y212 packed")]
  Y212Packed,
  /// Packed `y216` row — same shape as Y210 with BITS=16.
  #[display("Y216 packed")]
  Y216Packed,
  /// Packed `v410` row of a `V410` source — Tier 5 10-bit 4:4:4
  /// packed format. One `u32` word per pixel; row length: `width`
  /// `u32` elements (= `4 * width` bytes).
  #[display("V410 packed")]
  V410Packed,
  /// Packed `v30x` row of a `V30X` source — Tier 5 10-bit 4:4:4
  /// packed format, sibling of V410 with 2-bit padding at the
  /// **low** end. One `u32` word per pixel; row length: `width`
  /// `u32` elements (= `4 * width` bytes).
  #[display("V30X packed")]
  V30XPacked,
  /// Packed `xv36` row of an `Xv36` source — Tier 5 16-bit 4:4:4
  /// packed format. Four `u16` elements per pixel (one per channel);
  /// row length: `4 * width` `u16` elements (= `8 * width` bytes).
  #[display("XV36 packed")]
  Xv36Packed,
  /// Packed `xv48` row of an `Xv48` source — Tier 5 16-bit 4:4:4
  /// packed format (full-depth sibling of XV36). Four `u16` elements per
  /// pixel (one per channel); row length: `4 * width` `u16` elements
  /// (= `8 * width` bytes).
  #[display("XV48 packed")]
  Xv48Packed,
  /// Packed `vuya` row of a `Vuya` source — Tier 5 8-bit 4:4:4
  /// packed format. Four bytes per pixel in V/U/Y/A order; row
  /// length: `4 * width` bytes.
  #[display("VUYA packed")]
  VuyaPacked,
  /// Packed `vuyx` row of a `Vuyx` source — Tier 5 8-bit 4:4:4
  /// packed format. Four bytes per pixel in V/U/Y/X order (X is
  /// padding); row length: `4 * width` bytes.
  #[display("VUYX packed")]
  VuyxPacked,
  /// Packed `ayuv64` row of an `Ayuv64` source — Tier 5 16-bit
  /// 4:4:4 packed format. Four `u16` elements per pixel in A/Y/U/V
  /// order; row length: `4 * width` `u16` elements (= `8 * width`
  /// bytes).
  #[display("AYUV64 packed")]
  Ayuv64Packed,
  /// Packed `ayuv` row of an `Ayuv` source — 8-bit 4:4:4 packed format.
  /// Four bytes per pixel in A/Y/U/V order (A is real source alpha);
  /// row length: `4 * width` bytes.
  #[display("AYUV packed")]
  AyuvPacked,
  /// Packed `uyva` row of a `Uyva` source — 8-bit 4:4:4 packed format.
  /// Four bytes per pixel in U/Y/V/A order (A is real source alpha);
  /// row length: `4 * width` bytes.
  #[display("UYVA packed")]
  UyvaPacked,
  /// Packed `vyu444` row of a `Vyu444` source — 8-bit 4:4:4 packed format
  /// with **no alpha** (24bpp). Three bytes per pixel in V/Y/U order;
  /// row length: `3 * width` bytes.
  #[display("VYU444 packed")]
  Vyu444Packed,
  /// Packed `R, G, B` row of an [`Rgbf32`](crate::source::Rgbf32) source —
  /// Tier 9 32-bit float per channel. Row length: `3 * width` `f32`
  /// elements (= `12 * width` bytes).
  #[display("RGBF32 packed")]
  RgbF32Packed,
  /// Packed `R, G, B` row of an [`Rgbf16`](crate::source::Rgbf16) source —
  /// Tier 9 16-bit half-precision float per channel. Row length:
  /// `3 * width` `half::f16` elements (= `6 * width` bytes).
  #[display("RGBF16 packed")]
  RgbF16Packed,
  /// Packed `R, G, B, A` row of an [`Rgbaf32`](crate::source::Rgbaf32)
  /// source — Tier 9 32-bit float per channel, real alpha. Row length:
  /// `4 * width` `f32` elements (= `16 * width` bytes).
  #[display("RGBAF32 packed")]
  RgbaF32Packed,
  /// Packed `R, G, B, A` row of an [`Rgbaf16`](crate::source::Rgbaf16)
  /// source — Tier 9 16-bit half-precision float per channel, real alpha.
  /// Row length: `4 * width` `half::f16` elements (= `8 * width` bytes).
  #[display("RGBAF16 packed")]
  RgbaF16Packed,
  /// Packed `X, Y, Z` row of an [`Xyz12`](crate::source::Xyz12) source —
  /// Tier 12 12-bit CIE XYZ packed in u16 triples — active 12 bits
  /// in `[15:4]`, low 4 bits zero (per FFmpeg
  /// `AV_PIX_FMT_XYZ12LE/BE`). Row length: `3 * width` `u16` elements
  /// (= `6 * width` bytes).
  #[display("XYZ12 packed")]
  Xyz12Packed,
  /// Green plane row of an 8-bit planar GBR source
  /// ([`Gbrp`](crate::source::Gbrp) /
  /// [`Gbrap`](crate::source::Gbrap)). `u8` samples, `width` elements.
  #[display("G plane")]
  GPlane,
  /// Blue plane row of an 8-bit planar GBR source. `u8` samples,
  /// `width` elements.
  #[display("B plane")]
  BPlane,
  /// Red plane row of an 8-bit planar GBR source. `u8` samples,
  /// `width` elements.
  #[display("R plane")]
  RPlane,
  /// Plane row of a float-32 planar GBR source (`Gbrpf32` /
  /// `Gbrapf32`). `f32` samples, `width` elements per plane.
  #[display("GBR f32 plane")]
  GbrF32Plane,
  /// Plane row of a float-16 planar GBR source (`Gbrpf16` /
  /// `Gbrapf16`). `half::f16` samples, `width` elements per plane.
  #[display("GBR f16 plane")]
  GbrF16Plane,
  /// Packed `R, G, B` row of an [`Rgb48`](crate::source::Rgb48) source —
  /// `width * 3` u16 elements (each channel 16 bits, R, G, B order).
  #[display("RGB48 packed")]
  Rgb48Packed,
  /// Packed `B, G, R` row of a [`Bgr48`](crate::source::Bgr48) source —
  /// `width * 3` u16 elements (channel order reversed vs
  /// [`Rgb48Packed`](Self::Rgb48Packed)).
  #[display("BGR48 packed")]
  Bgr48Packed,
  /// Packed `R, G, B, A` row of an [`Rgba64`](crate::source::Rgba64) source —
  /// `width * 4` u16 elements (each channel 16 bits; alpha is real).
  #[display("RGBA64 packed")]
  Rgba64Packed,
  /// Packed `B, G, R, A` row of a [`Bgra64`](crate::source::Bgra64) source —
  /// `width * 4` u16 elements (channel order reversed on RGB vs
  /// [`Rgba64Packed`](Self::Rgba64Packed); alpha at slot 3 is real).
  #[display("BGRA64 packed")]
  Bgra64Packed,
  /// Packed `R, G, B` row of an [`Rgb96`](crate::source::Rgb96) source —
  /// `width * 3` u32 elements (each channel 32 bits, R, G, B order).
  #[display("RGB96 packed")]
  Rgb96Packed,
  /// Packed `R, G, B, A` row of an [`Rgba128`](crate::source::Rgba128) source —
  /// `width * 4` u32 elements (each channel 32 bits; alpha at slot 3 is real).
  #[display("RGBA128 packed")]
  Rgba128Packed,
}

/// How a packed-alpha source's color channels relate to its alpha when
/// the frame is **area-resampled** — the only place the distinction is
/// observable, because area-binning averages color and alpha together.
///
/// In [`Self::Straight`] (a.k.a. *unassociated* / *non-premultiplied*)
/// alpha, the RGB triple stores the surface's own color and α is an
/// independent coverage term; averaging the channels independently is
/// correct. In [`Self::Premultiplied`] (a.k.a. *associated*) alpha, RGB
/// has already been multiplied by α, so a correct area-average must bin
/// the premultiplied channels and un-premultiply afterwards — averaging
/// straight RGB of a premultiplied source would let fully-transparent
/// pixels (whose stored RGB is arbitrary) bleed into the result.
///
/// Every packed-RGBA source format colconv ships today is straight (see
/// [`DefaultAlphaMode`]); the mode only matters on the resample path and
/// is a no-op for the direct (identity-plan) conversions, which copy
/// alpha through untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IsVariant)]
pub enum AlphaMode {
  /// Unassociated alpha: RGB is the surface color, α an independent
  /// coverage term. Channels area-average independently.
  #[default]
  Straight,
  /// Associated alpha: RGB is already premultiplied by α. The resample
  /// path bins the premultiplied channels and un-premultiplies per
  /// finalized output row.
  Premultiplied,
}

impl AlphaMode {
  /// Returns the lowercase string name of the mode (`"straight"` /
  /// `"premultiplied"`), matching the variant's conventional spelling.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Straight => "straight",
      Self::Premultiplied => "premultiplied",
    }
  }
}

/// Per-format default [`AlphaMode`], consulted by [`MixedSinker::new`]
/// to seed the sink's alpha mode before any
/// [`MixedSinker::with_alpha_mode`] override.
///
/// The blanket impl below makes every [`SourceFormat`] default to
/// [`AlphaMode::Straight`] — true of every packed-RGBA source colconv
/// ships today (`Rgba` / `Bgra` / `Argb` / `Abgr` / `Rgba64` /
/// `Bgra64`). A future source format whose wire alpha is associated
/// would carry its premultiplied default here (replacing the blanket
/// with per-format impls), so callers get correct area-resampling
/// without having to pass [`MixedSinker::with_alpha_mode`] by hand.
pub trait DefaultAlphaMode: SourceFormat {
  /// The alpha interpretation a freshly built [`MixedSinker`] over this
  /// format starts in.
  const DEFAULT_ALPHA_MODE: AlphaMode = AlphaMode::Straight;
}

impl<F: SourceFormat> DefaultAlphaMode for F {}

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
/// # Type parameters
///
/// `F` identifies the source format — `Yuv420p`, `Nv12`, `Nv21`,
/// `Yuv420p10`, `Yuv420p12`, `Yuv420p14`, `P010`, `P012`, etc. Each
/// format provides its own `impl PixelSink for MixedSinker<'_, F>`.
/// See the module‑level docs for the full list of shipped impls.
///
/// `R` is the resampling strategy deciding the sinker's **output**
/// geometry, injected via [`Self::with_resampler`]. The default
/// [`NoopResampler`] is the identity:
/// output geometry == source geometry, i.e. exactly the historical
/// behavior of [`Self::new`]. Output buffers always validate against
/// the output geometry; [`PixelSink::begin_frame`] always validates
/// the walker against the source geometry.
///
/// Formats route non-identity plans as they wire into the streaming
/// engine (currently [`Yuv420p`](crate::source::Yuv420p)). Every other
/// per-format [`PixelSink`] impl stays pinned to the default strategy:
/// a sinker built with a non-identity strategy can attach
/// (output-validated) buffers but does not implement [`PixelSink`], so
/// routing it through a walker is a compile error rather than a
/// geometry-mismatch panic.
pub struct MixedSinker<'a, F: SourceFormat, R = NoopResampler> {
  rgb: Option<&'a mut [u8]>,
  rgb_u16: Option<&'a mut [u16]>,
  rgb_f32: Option<&'a mut [f32]>,
  rgb_f16: Option<&'a mut [half::f16]>,
  rgba: Option<&'a mut [u8]>,
  rgba_u16: Option<&'a mut [u16]>,
  rgba_f32: Option<&'a mut [f32]>,
  rgba_f16: Option<&'a mut [half::f16]>,
  luma: Option<&'a mut [u8]>,
  luma_u16: Option<&'a mut [u16]>,
  luma_f32: Option<&'a mut [f32]>,
  // `HsvFrameMut` is cfg-gated to the same 15-feature any as the
  // per-format `process` impls that read it.
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  hsv: Option<HsvFrameMut<'a>>,
  /// Lossless linear-XYZ pass-through buffer used by the
  /// [`Xyz12`](crate::source::Xyz12) source's `with_xyz_f32` accessor.
  /// `None` for every other source format.
  #[cfg(feature = "xyz")]
  xyz_f32: Option<&'a mut [f32]>,
  width: usize,
  height: usize,
  /// Output geometry from the resampler's plan; equals
  /// `(width, height)` under the identity plan. Every output-buffer
  /// length validation sizes against these, never against the source
  /// geometry.
  out_width: usize,
  out_height: usize,
  /// The non-identity plan fixed by [`MixedSinker::with_resampler`];
  /// `None` for [`MixedSinker::new`] and identity plans (the sinker
  /// then takes the direct conversion path).
  plan: Option<ResamplePlan>,
  /// Row-stage area streams (color group / luma group) for formats
  /// that route non-identity plans. Lazily created in `process`,
  /// reset in `begin_frame`. Gated like the engine itself, widening
  /// as families wire in.
  #[cfg(any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  ))]
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  rgb_stream: Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  /// Row-stage area stream for high-bit packed-RGB sources (`u16`
  /// elements binned at native depth). Lazily created in `process`,
  /// reset in `begin_frame`. Gated to `rgb` (high-bit packed RGB),
  /// `gbr` (high-bit planar GBR scatters into the same u16 tail),
  /// `yuv-444-packed` / `y2xx` / `v210` (the high-bit packed YUV color
  /// groups bin their converted native-u16 RGB row here),
  /// `yuv-planar` (the high-bit planar YUV 4:4:4 / 4:2:2 color group
  /// bins its converted native-u16 RGB row here), and `yuv-semi-planar`
  /// (the high-bit semi-planar P-format color group bins its converted
  /// native-u16 RGB row here); widens as high-bit families wire in.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  ))]
  rgb_stream_u16: Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  /// Row-stage **4-channel** `u8` area stream for the alpha-aware u8 color
  /// of packed straight/premult RGBA sources (`Rgba` / `Bgra` / `Argb` /
  /// `Abgr`), the planar GBR+alpha family (`Gbrap`, decoded to the same
  /// canonical RGBA row), gray+alpha (`Ya8`), and the packed YUVA family
  /// (`Vuya`, and `Ayuv64`'s u8 color group — the converted u8 RGBA row).
  /// Bins the staged canonical `R, G, B, A` row so resampled alpha is a
  /// real area mean (not forced opaque) and — under
  /// [`AlphaMode::Premultiplied`] — color is binned premultiplied.
  /// Lazily created in `process`, reset in `begin_frame`. Gated to `rgb` /
  /// `gbr` / `gray` / `mono` / `yuv-444-packed`; the 3-channel
  /// [`Self::rgb_stream`] still serves the rgb-only straight path with no
  /// regression. (`mono` joins for `Pal8`, whose palette carries real
  /// per-entry alpha — the expand-to-RGBA-then-bin route.)
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "yuva",
    feature = "yuv-444-packed"
  ))]
  rgba_stream: Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  /// Row-stage **4-channel** `u16` area stream for the native-depth u16
  /// color of the high-bit packed RGBA sources (`Rgba64` / `Bgra64`), the
  /// high-bit planar GBR+alpha family (`Gbrap10` … `Gbrap16`), gray+alpha
  /// (`Ya16`), and the packed YUVA `Ayuv64` (its independent u16 color
  /// group — the converted native u16 RGBA row). Bins the staged canonical
  /// `R, G, B, A` row at native depth; the native-depth `rgba_u16` output
  /// copies it (the RGB-source narrowed outputs derive via
  /// `>> (SRC_BITS - 8)`, but `Ayuv64` instead bins its u8 color
  /// independently in [`Self::rgba_stream`]). Lazily created in `process`,
  /// reset in `begin_frame`. Gated to `rgb` / `gbr` / `gray` /
  /// `yuv-444-packed`.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "yuva",
    feature = "yuv-444-packed"
  ))]
  rgba_stream_u16: Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  /// Row-stage 3-channel `u32` **area** stream for the native-`u32` packed-RGB
  /// resample tier (`Rgb96`, and the rgb-only / alpha-dropped paths of
  /// `Rgba128` / `Gbrap32`): bins the staged host-native `R, G, B` `u32` row at
  /// full `u32` precision so every output narrows only after binning — exact
  /// 0-ULP versus the narrow-first u16 stream (closes issue #289). Lazily
  /// created in `process`, reset in `begin_frame`. Gated to `rgb` / `gbr`.
  #[cfg(any(feature = "rgb", feature = "gbr"))]
  rgb_stream_u32: Option<std::boxed::Box<crate::resample::AreaStream<u32>>>,
  /// Row-stage 3-channel `u32` **filter** stream — the
  /// [`SpanKind::Filter`](crate::resample::SpanKind) twin of
  /// [`Self::rgb_stream_u32`]. Filters the staged host-native `R, G, B` `u32`
  /// row at the `f64` domain (carries a `u32` sample exactly), narrowing only
  /// per output (0-ULP). Gated to `rgb` / `gbr`.
  #[cfg(any(feature = "rgb", feature = "gbr"))]
  rgb_filter_stream_u32: Option<std::boxed::Box<crate::resample::FilterStream<u32>>>,
  /// Row-stage 4-channel `u32` **area** stream for the native-`u32` alpha-aware
  /// packed-RGBA resample tier (`Rgba128` / `Gbrap32`): bins the staged
  /// canonical host-native `R, G, B, A` `u32` row at full `u32` precision (and,
  /// under [`AlphaMode::Premultiplied`], premultiplied at `u32`) so resampled
  /// alpha and colour are exact 0-ULP after the post-bin narrow (issue #289).
  /// Gated to `rgb` / `gbr`.
  #[cfg(any(feature = "rgb", feature = "gbr"))]
  rgba_stream_u32: Option<std::boxed::Box<crate::resample::AreaStream<u32>>>,
  /// Row-stage 4-channel `u32` **filter** stream — the
  /// [`SpanKind::Filter`](crate::resample::SpanKind) twin of
  /// [`Self::rgba_stream_u32`]; filters R, G, B, A independently at `u32`
  /// precision (straight alpha only — premultiplied has no filter analogue).
  /// Gated to `rgb` / `gbr`.
  #[cfg(any(feature = "rgb", feature = "gbr"))]
  rgba_filter_stream_u32: Option<std::boxed::Box<crate::resample::FilterStream<u32>>>,
  /// Alpha mode frozen at a resampled frame's first row. A mid-frame
  /// [`Self::set_alpha_mode`] change is then rejected before any stream
  /// is fed, since a stream mixing straight and premultiplied rows would
  /// match neither all-straight nor all-premultiplied output. `None`
  /// between frames; re-armed on each frame's first resampled row (so a
  /// stale value never leaks across frames). Gated to `rgb` / `gbr` /
  /// `gray` / `mono` / `yuv-444-packed`.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "yuva",
    feature = "yuv-444-packed"
  ))]
  frozen_alpha_mode: Option<AlphaMode>,
  /// Row-stage area stream for packed-float-RGB sources (`f32`
  /// elements binned in float). Lazily created in `process`, reset in
  /// `begin_frame`. The `rgb-float` family needs the engine fenced in
  /// (`AreaStream` is gated to `yuv-planar` / `rgb`, which `rgb-float`
  /// does not imply); the `gbr` family already pulls the engine via the
  /// #146 cascade, so its float-GBR scatter reaches this same tail with
  /// no separate fence.
  #[cfg(any(
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
    feature = "gbr"
  ))]
  rgb_stream_f32: Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  /// Row-stage **filter** stream for packed-float-RGB sources
  /// ([`Rgbf32`](crate::source::Rgbf32)) — the
  /// [`SpanKind::Filter`](crate::resample::SpanKind) twin of
  /// [`Self::rgb_stream_f32`]. Lazily created in `process`, reset in
  /// `begin_frame`. Fed when the plan kind is `Filter`; bins at f32
  /// precision and emits unclamped (full-range float, PIL `F`-mode). Gated
  /// like [`Self::rgb_stream_f32`]: the packed-float-RGB consumers
  /// (`rgb-float`, with the engine fenced in: `FilterStream` is gated to
  /// `yuv-planar` / `rgb`, which `rgb-float` does not imply) AND `gbr` — the
  /// float planar GBR sources `Gbrpf32` / `Gbrpf16` scatter their G/B/R
  /// planes into the same packed `f32` RGB row and filter through this
  /// stream (the `gbr` build already carries the engine via the #146
  /// cascade).
  #[cfg(any(
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
    feature = "gbr"
  ))]
  rgb_filter_stream_f32: Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  /// Row-stage **4-channel** `f32` area stream for the float planar
  /// GBR+alpha family ([`Gbrapf32`](crate::source::Gbrapf32) /
  /// [`Gbrapf16`](crate::source::Gbrapf16), the latter widened f16 ->
  /// host-native f32). Bins the staged canonical `R, G, B, A` f32 row so
  /// resampled alpha is a real area mean (not forced opaque) and — under
  /// [`AlphaMode::Premultiplied`] — color is binned premultiplied. Lazily
  /// created in `process`, reset in `begin_frame`. GBR-only: there is no
  /// packed-float RGBA source, so this is gated to `gbr` (which already
  /// carries the engine via the #146 cascade); the 3-channel
  /// [`Self::rgb_stream_f32`] still serves the rgb-only straight float path.
  #[cfg(any(
    feature = "gbr",
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
  ))]
  rgba_stream_f32: Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  /// Row-stage **4-channel** `f32` **filter** stream for the float planar
  /// GBR+alpha family ([`Gbrapf32`](crate::source::Gbrapf32) /
  /// [`Gbrapf16`](crate::source::Gbrapf16), the latter widened f16 ->
  /// host-native f32) — the [`SpanKind::Filter`](crate::resample::SpanKind)
  /// twin of [`Self::rgba_stream_f32`] and the 4-channel analogue of
  /// [`Self::rgb_filter_stream_f32`]. PIL filters R, G, B, A independently
  /// with no premultiplication, so the staged canonical `R, G, B, A` f32 row
  /// bins through one 4-channel [`FilterStream<f32>`](crate::resample::FilterStream)
  /// and resampled alpha is a real filtered channel (never forced opaque).
  /// Lazily created in `process`, reset in `begin_frame`. There is no
  /// premultiplied filter analogue (the filter engine cannot un-premultiply),
  /// so a premultiplied frame stays on the area [`Self::rgba_stream_f32`];
  /// the filter stream is reached only for straight alpha. GBR-only (there is
  /// no packed-float RGBA source), gated to `gbr` to match
  /// [`Self::rgba_stream_f32`]; the engine is already pulled in for `gbr` via
  /// the #146 cascade. The float `rgb_f32` / `rgba_f32` outputs are
  /// full-range by design (HDR > 1 and negatives preserved), so there is no
  /// native-depth clamp on the filter output — the integer / f16 outputs
  /// clamp `[0, 1]` in their own narrows, so a signed-coefficient overshoot
  /// cannot wrap them.
  #[cfg(any(
    feature = "gbr",
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
  ))]
  rgba_filter_stream_f32: Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  /// Row-stage area stream for the packed-CIE-XYZ-12-bit source
  /// ([`Xyz12`](crate::source::Xyz12)). The wire row converts to
  /// **linear XYZ** `f32` (post-OETF, pre-matrix) and bins in float so
  /// the area mean is taken in linear light — the matrix and gamma are
  /// applied per finalized output row, after the bin. Gated to `xyz`;
  /// the engine is already pulled in for `xyz` by the shared
  /// [`AreaStream`] gate (the `#145`/`#146` cascade widened it to
  /// `xyz`), so no separate engine feature is required.
  #[cfg(feature = "xyz")]
  xyz_stream_f32: Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  /// Row-stage **filter** stream for the packed-CIE-XYZ-12-bit source
  /// ([`Xyz12`](crate::source::Xyz12)) — the
  /// [`SpanKind::Filter`](crate::resample::SpanKind) twin of
  /// [`Self::xyz_stream_f32`]. Bins the staged **linear XYZ** `f32` row
  /// (post-OETF, pre-matrix) with signed filter coefficients, then the
  /// per-finalized-row matrix + gamma + narrow run exactly as on the area
  /// path. Lazily created in `process`, reset in `begin_frame`. Gated to
  /// `xyz` to match [`Self::xyz_stream_f32`]: the shared engine gate
  /// already compiles [`FilterStream`] in for `xyz`. There is no
  /// native-depth clamp on the filter output — the integer / f16 outputs
  /// clamp `[0, 1]` in their own narrows and the `rgb_f32` / `xyz_f32`
  /// outputs are full-range by design (HDR > 1 and out-of-gamut negatives
  /// preserved), so a signed-coefficient overshoot cannot wrap them.
  #[cfg(feature = "xyz")]
  xyz_filter_stream_f32: Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  /// Row-stage area stream for single-plane luma binning. Used by the
  /// planar YUV family (Y-plane luma), the [`Gray8`](crate::source::Gray8)
  /// source (Gray *is* a luma plane), and `mono` (bin the expanded
  /// 0/255 luma plane). Lazily created in `process`, reset in
  /// `begin_frame`. Gated like the engine; widens as families wire in.
  #[cfg(any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  ))]
  #[cfg_attr(
    not(any(feature = "yuv-planar", feature = "gray", feature = "mono")),
    allow(dead_code)
  )]
  luma_stream: Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  /// Row-stage area stream for single-plane **u16** luma binning. Used
  /// by the [`Gray16`](crate::source::Gray16) source, whose luma plane
  /// is a native `u16` and so bins at u16 precision (the `u8`
  /// [`Self::luma_stream`] would lose the low byte), by the high-bit
  /// packed YUV families (`yuv-444-packed` / `y2xx` / `v210`), and by the
  /// high-bit planar YUV families (`yuv-planar`: Yuv444p / Yuv422p
  /// 10/12/14/16), which bin their native Y here so resampled luma stays
  /// the area-downscaled Y at native depth. Lazily created in `process`,
  /// reset in `begin_frame`. Gated to `gray` / `yuv-444-packed` / `y2xx`
  /// / `v210` / `yuv-planar` / `yuv-semi-planar` (the high-bit semi-planar
  /// P-format family bins its de-packed native Y here, then narrows since
  /// P-formats expose no `luma_u16`); widens as u16 luma families wire in.
  #[cfg(any(
    feature = "gray",
    feature = "yuva",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  ))]
  luma_stream_u16: Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  /// Row-stage area stream for single-plane **u32** luma binning. Used by
  /// the [`Gray32`](crate::source::Gray32) source, whose luma plane is a
  /// native `u32`: binning at native `u32` precision (then narrowing each
  /// output) is **0-ULP** (closes issue #289), whereas the `u16`
  /// [`Self::luma_stream_u16`] would drop the low 16 bits before averaging
  /// (≤1-LSB off the exact mean). Lazily created in `process`, reset in
  /// `begin_frame`. Gated to `gray`; widens if more `u32` luma families
  /// wire in.
  #[cfg(feature = "gray")]
  luma_stream_u32: Option<std::boxed::Box<crate::resample::AreaStream<u32>>>,
  /// Row-stage area stream for single-plane **f32** luma binning. Used
  /// by the [`Grayf32`](crate::source::Grayf32) source, whose luma plane
  /// is a native `f32` and so bins at f32 precision (the `u8` / `u16`
  /// luma streams would quantize every sample before averaging). Lazily
  /// created in `process`, reset in `begin_frame`. Gated to `gray`;
  /// widens as f32 luma families wire in.
  #[cfg(feature = "gray")]
  luma_stream_f32: Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  /// Row-stage **filter** stream for the packed-RGB `u8` color group
  /// ([`Rgb24`](crate::source::Rgb24)) — the signed-coefficient
  /// (PIL-parity) twin of [`Self::rgb_stream`], fed when the plan's
  /// [`SpanKind`](crate::resample::SpanKind) is `Filter`. Lazily created
  /// in `process`, reset in `begin_frame`. The first format routed through
  /// the filter engine in this stage; the gate widens with the area
  /// engine as more packed-RGB sources wire in (`yuv-444-packed` joins for
  /// the high-bit packed YUV 4:4:4 colour group — `V410` / `V30X` convert
  /// their YUV to a u8 RGB row and filter it here, the filter twin of the
  /// area [`Self::rgb_stream`]; `y2xx` joins for the high-bit packed YUV
  /// 4:2:2 colour group — `Y210` / `Y212` / `Y216`, whose 4:2:2 chroma the
  /// convert closures upsample before this RGB-space filter; `yuv-planar`
  /// joins for the 8-bit planar YUV family — `Yuv410p` / `Yuv420p` /
  /// `Yuv422p` / `Yuv444p` / `Yuv440p` convert their separate Y/U/V planes
  /// to a source-width u8 RGB row and filter it here, matching the area
  /// [`Self::rgb_stream`] gate; `yuv-semi-planar` joins for the 8-bit
  /// semi-planar YUV family — `Nv12` / `Nv16` / `Nv21` / `Nv24` / `Nv42`
  /// convert their de-interleaved chroma into the same source-width u8 RGB
  /// row and filter it here, the filter twin of their area
  /// [`Self::rgb_stream`] use; `yuv-packed` joins for the 8-bit packed YUV
  /// 4:2:2 colour group — `Yuyv422` / `Uyvy422` / `Yvyu422` — and the 8-bit
  /// packed YUV 4:1:1 `Uyyvyy411`, whose `*_to_rgb_row` kernels
  /// de-interleave + horizontally upsample the chroma to a u8 RGB row
  /// before this RGB-space filter).
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuv-packed"
  ))]
  rgb_filter_stream: Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  /// Row-stage **filter** stream for the 8-bit packed-RGBA `u8` color
  /// group ([`Rgba`](crate::source::Rgba) and the leading-/trailing-alpha
  /// reorderings) — the 4-channel, signed-coefficient twin of
  /// [`Self::rgb_filter_stream`], fed when a real-alpha packed-RGBA source
  /// takes a `Filter` plan. PIL resizes RGBA by filtering R, G, B, A
  /// independently with no premultiplication, so the four interleaved
  /// channels bin through one 4-channel filter and a resampled RGBA frame
  /// is byte-exact versus PIL's RGBA resize. Lazily created in `process`,
  /// reset in `begin_frame`. Padding-alpha sources keep the 3-channel
  /// [`Self::rgb_filter_stream`] (the X byte is never filtered). Gated to
  /// `any(rgb, gbr, yuv-444-packed, mono)`: besides the 8-bit planar
  /// GBR+alpha source [`Gbrap`], the 8-bit packed YUV 4:4:4 sources `Vuya`
  /// (real source alpha) and `Vuyx` (padding byte forced opaque) convert
  /// their YUVA to a canonical u8 `R, G, B, A` row and filter through this
  /// same 4-channel stream, and the 8-bit palette-indexed [`Pal8`] expands
  /// each index to its palette color (real per-entry straight alpha) and
  /// filters that canonical RGBA row through this stream too — its non-color
  /// outputs (luma / u16) keep the direct Q8 / `(v << 8) | v` derivations,
  /// not the matrix-based ones the chromatic packed-RGBA path uses. `yuva`
  /// joins for the 8-bit planar YUVA family (`Yuva420p` / `Yuva422p` /
  /// `Yuva444p`), which converts its Y/U/V/A planes to a canonical u8
  /// `R, G, B, A` row and filters through this same 4-channel stream (the
  /// filter twin of its area [`Self::rgba_stream`] use); luma stays native Y.
  /// `gray` joins for the 8-bit packed gray+alpha source `Ya8`, a degenerate
  /// YUVA (`R = G = B = Y`, neutral chroma, real straight alpha): it decodes
  /// each `[Y, A]` row into the canonical u8 RGBA via `ya8_to_rgba_row` and
  /// filters through this same 4-channel stream, with native-Y luma on the
  /// `u8` [`Self::luma_filter_stream`] (parity with `Gray8`).
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "mono",
    feature = "yuva",
    feature = "gray"
  ))]
  rgba_filter_stream: Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  /// Row-stage **filter** stream for the high-bit packed-RGB `u16` color
  /// group ([`Rgb48`](crate::source::Rgb48), and the high-bit planar GBR
  /// sources `Gbrp9`…`Gbrp16` which scatter into the same packed `u16` RGB
  /// row, and the high-bit packed YUV 4:4:4 colour group `V410` / `V30X`,
  /// and the high-bit packed YUV 4:2:2 colour group `Y210` / `Y212` /
  /// `Y216`, which convert their YUV to a native-depth u16 RGB row) — the
  /// filter twin of [`Self::rgb_stream_u16`]. Lazily created in `process`,
  /// reset in `begin_frame`. Fed when the plan kind is `Filter`. The
  /// [`packed_rgb_u16_resample_emit`] tail clamps a signed-kernel overshoot
  /// to the source's native max `(1 << SRC_BITS) - 1` for a sub-16-bit
  /// source (`V410` / `V30X` / `Y210` are 10-bit, `Y212` 12-bit), matching
  /// the in-range area path. `yuv-planar` joins for the high-bit planar YUV
  /// family (`Yuv4{2,4}{0,2,4}p{10,12,14,16}`), which converts its separate
  /// Y/U/V planes to a native-depth u16 RGB row and filters it here — the
  /// filter twin of its area [`Self::rgb_stream_u16`] use, with the same
  /// `SRC_BITS` native-max clamp keyed by its source depth. `yuv-semi-planar`
  /// joins for the high-bit semi-planar P-format family, the filter twin of
  /// its area [`Self::rgb_stream_u16`] use.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  ))]
  rgb_filter_stream_u16: Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  /// Row-stage **filter** stream for the high-bit packed-RGBA `u16` color
  /// group ([`Rgba64`](crate::source::Rgba64) /
  /// [`Bgra64`](crate::source::Bgra64)) — the 4-channel, signed-coefficient
  /// twin of [`Self::rgb_filter_stream_u16`] and the `u16` analogue of
  /// [`Self::rgba_filter_stream`], fed when a real-alpha `Rgba64` / `Bgra64`
  /// source takes a `Filter` plan. PIL resizes RGBA by filtering R, G, B, A
  /// independently with no premultiplication, so the four interleaved native
  /// u16 channels bin through one 4-channel filter and a resampled RGBA frame
  /// is byte-exact (per channel) versus the merged engine. Lazily created in
  /// `process`, reset in `begin_frame`. Gated to
  /// `any(rgb, gbr, yuv-444-packed)`: besides `Rgba64` / `Bgra64`, the
  /// high-bit planar GBR+alpha family (`Gbrap10`…`Gbrap16`) de-interleaves
  /// its native-depth G/B/R/A planes into the same canonical packed RGBA
  /// u16 row and filters through this 4-channel stream (the native-max clamp
  /// keyed by the source `BITS`); the packed YUVA 4:4:4 filter route shares
  /// this `u16` colour stream for any sub-16-bit source that exposes native
  /// u16 colour outputs (the 8-bit `Vuya` / `Vuyx` do not, so they leave it
  /// unused). `yuva` joins for the 8-bit planar YUVA family (`Yuva420p` /
  /// `Yuva422p` / `Yuva444p`), which routes through the same 4:4:4 filter
  /// tail; those sources expose no u16 colour outputs either, so they leave
  /// this stream unused too. `gray` joins for the high-bit packed gray+alpha
  /// source `Ya16` (a degenerate YUVA, `R = G = B = Y`, real straight alpha):
  /// it decodes each `[Y, A]` u16 row into the canonical host-native u16 RGBA
  /// via `ya16_to_rgba_u16_row` and filters its native-depth colour through
  /// this 4-channel stream (full 16-bit, so the `FilterStream`'s `0..=65535`
  /// clamp is the native clamp), with native-Y luma on the `u16`
  /// [`Self::luma_filter_stream_u16`] (parity with `Gray16`).
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "yuva",
    feature = "gray"
  ))]
  rgba_filter_stream_u16: Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  /// Row-stage **filter** stream for single-plane `u8` native-Y luma
  /// resampling — the [`SpanKind::Filter`](crate::resample::SpanKind) twin
  /// of [`Self::luma_stream`] for the 8-bit planar YUV family (`Yuv410p` /
  /// `Yuv420p` / `Yuv422p` / `Yuv444p` / `Yuv440p`). On a `Filter` plan the
  /// Y plane is filter-resampled here as a 1-channel `u8` stream, so luma
  /// stays **native Y** (never colour-derived) — byte-exact to the direct
  /// Y-plane copy's filter resample, the filter analogue of the area path's
  /// native-Y bin ([`planar_dual_resample`](planar_resample::planar_dual_resample)).
  /// These sources are 8-bit, so no native-depth clamp is needed (the
  /// stream finalizes to the full `u8` range, which *is* the native range);
  /// `luma_u16` zero-extends each resampled Y byte. Lazily created in
  /// `process`, reset in `begin_frame`. Gated to `yuv-planar` /
  /// `yuv-semi-planar` (the 8-bit semi-planar family `Nv12` / `Nv16` /
  /// `Nv21` / `Nv24` / `Nv42` bins its native Y here too) / `yuv-packed`
  /// (packed 4:2:2/4:1:1) / `mono` (the 1-bit bilevel `Monoblack` /
  /// `Monowhite` filter-resample the expanded 0/255 luma through this
  /// single-channel stream — full-range `u8`, so no native-depth clamp) /
  /// `gray` (the 8-bit single-channel [`Gray8`](crate::source::Gray8) *is*
  /// a luma plane — its Y is filter-resampled here directly, full-range
  /// `u8`, no native-depth clamp); widens as more 8-bit native-Y filter
  /// families wire in.
  #[cfg(any(
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "mono",
    feature = "gray"
  ))]
  luma_filter_stream: Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  /// Row-stage **filter** stream for single-plane `f32` luma binning
  /// ([`Grayf32`](crate::source::Grayf32)) — the filter twin of
  /// [`Self::luma_stream_f32`]. Lazily created in `process`, reset in
  /// `begin_frame`. Fed when the plan kind is `Filter`; bins at f32
  /// precision and emits unclamped (PIL `F`-mode).
  #[cfg(feature = "gray")]
  luma_filter_stream_f32: Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  /// Row-stage **filter** stream for single-plane **`u16`** native-Y luma
  /// resampling — the [`SpanKind::Filter`](crate::resample::SpanKind) twin
  /// of [`Self::luma_stream_u16`] for the high-bit packed YUV 4:4:4 colour
  /// group (`V410` / `V30X`) and the high-bit packed YUV 4:2:2 colour group
  /// (`Y210` / `Y212` / `Y216`). On a `Filter` plan the native Y is
  /// de-interleaved into a source-width `u16` scratch and resampled here at
  /// native depth, so luma stays **native Y** (never colour-derived) —
  /// byte-exact to the direct `*_to_luma*` kernels' filter resample, the
  /// filter analogue of the area path's native-Y bin. Lazily created in
  /// `process`, reset in `begin_frame`. Gated to `any(yuv-444-packed,
  /// y2xx, yuv-planar, yuva)` — the high-bit planar YUV family
  /// (`Yuv4{2,4}{0,2,4}p{10,12,14,16}`) de-interleaves its native Y plane
  /// into a source-width `u16` scratch and resamples it here, the filter
  /// twin of its area native-Y bin (with the same sub-16-bit native-max
  /// clamp). `gray` joins for the high-bit single-channel
  /// [`Gray16`](crate::source::Gray16) (full 16-bit, native max == u16 max,
  /// so the `FilterStream`'s `0..=65535` clamp *is* the native clamp — no
  /// extra clamp needed). The 8-bit planar YUVA family (`Yuva420p` /
  /// `Yuva422p` / `Yuva444p`, `yuva` — which also pulls in `yuv-planar`)
  /// de-interleaves its zero-extended native Y here too (the 8-bit
  /// native-max clamp is a value no-op for in-range Y); the high-bit
  /// semi-planar P-format family (`yuv-semi-planar`) filter-resamples its
  /// de-packed native Y here (then narrows, exposing no `luma_u16`);
  /// widens as more native-Y filter families wire in.
  #[cfg(any(
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "yuv-planar",
    feature = "gray",
    feature = "yuva",
    feature = "yuv-semi-planar"
  ))]
  luma_filter_stream_u16: Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  /// Row-stage **filter** stream for single-plane **u32** luma — the
  /// signed-coefficient twin of [`Self::luma_stream_u32`] for
  /// [`Gray32`](crate::source::Gray32). Filters the native `u32` luma plane
  /// at full precision (the `f64` filter domain carries a `u32` sample
  /// exactly), then narrows each output, so the filter resample is 0-ULP
  /// versus narrowing first (closes issue #289). Lazily created in
  /// `process`, reset in `begin_frame`. Gated to `gray`.
  #[cfg(feature = "gray")]
  luma_filter_stream_u32: Option<std::boxed::Box<crate::resample::FilterStream<u32>>>,
  /// Output configuration frozen at a resampled frame's first
  /// processed row; `None` between frames. Captures presence AND
  /// attachment identity (pointer/length) of every output the emit
  /// closures consult, so both membership changes and same-channel
  /// buffer replacement trip
  /// [`MixedSinkerError::ResampleOutputsChanged`].
  #[cfg(any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  ))]
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  resample_outputs: Option<FrozenOutputs>,
  /// Whether resampled processing may take the native decimation tier
  /// (bin native planes, convert once at output resolution). Defaults
  /// to `true`; benchmarks and differential tests flip it to force the
  /// row-stage tier — the [`Self::with_simd`] pattern. The native tier
  /// exists for the 8-bit planar 4:2:0
  /// ([`Yuv420p`](crate::source::Yuv420p)), the 8-bit semi-planar 4:2:0
  /// ([`Nv12`](crate::source::Nv12) / [`Nv21`](crate::source::Nv21)),
  /// the high-bit planar 4:2:0 family
  /// ([`Yuv420p10`](crate::source::Yuv420p10) /12/14/16), and the high-bit
  /// semi-planar 4:2:0 family ([`P010`](crate::source::P010) /
  /// [`P012`](crate::source::P012) / [`P016`](crate::source::P016)); every
  /// other routed family always takes the row-stage tier and ignores this
  /// flag. The native tier itself is compiled only under `yuv-planar` (the
  /// high-bit P-format tier reuses the planar join), so under a
  /// yuv-semi-planar-solo build the flag is accepted by the builder but
  /// never read — the P-format sinks always take the row-stage tail there.
  #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
  #[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
  native: bool,
  /// Native-tier join state for the 4:2:0 planar family; lazily
  /// created in `process`, reset in `begin_frame`.
  ///
  /// Boxed so the `MixedSinker` — constructed as a stack local across the
  /// crate's tests and callers — holds only a pointer for this >1 KiB join;
  /// the heap allocation still happens lazily on the first native row, so the
  /// recoverable-allocation contract is unchanged.
  #[cfg(feature = "yuv-planar")]
  native_420: Option<std::boxed::Box<planar_8bit::NativeYuv420>>,
  /// Native-tier join state for the **straight-alpha** 4:2:0 planar source
  /// `Yuva420p` (RFC #238 Phase 5 / the #235 alpha resolution) — the
  /// alpha-bearing sibling of [`Self::native_420`]: it embeds the no-alpha
  /// 4:2:0 join for Y / U / V and adds a luma-grid α [`AreaStream`] whose
  /// binned α is substituted into the RGBA output. Taken ONLY under
  /// [`AlphaMode::Straight`] (premultiplied stays on the packed-YUVA area
  /// tail); lazily created in `process`, reset in `begin_frame`.
  ///
  /// Boxed: the join embeds the full no-alpha 4:2:0 join plus an α stream
  /// (>1 KiB), so storing it inline would enlarge every `MixedSinker` — held
  /// on the stack across the crate — by that much; the box keeps the field a
  /// single pointer and the heap allocation only happens on the first native
  /// straight-alpha row.
  #[cfg(feature = "yuva")]
  native_yuva_420: Option<std::boxed::Box<planar_8bit::NativeYuva420>>,
  /// BICUBLIN per-plane filter join for the 4:2:0 planar family — the
  /// signed-coefficient twin of [`Self::native_420`]. Holds the three
  /// [`FilterStream`](crate::resample::FilterStream)s (Y with the cubic
  /// luma kernel, U / V with the linear chroma kernel) and the per-plane
  /// landing buffers; lazily created in `process` on a BICUBLIN plan,
  /// reset in `begin_frame`. Only a [`Bicublin`](crate::resample::Bicublin)
  /// plan populates it. **Boxed** — the three filter streams make this the
  /// largest single join, so keeping it behind a pointer holds the already
  /// large `MixedSinker` stack footprint down (the join is heap-allocated
  /// lazily anyway).
  #[cfg(feature = "yuv-planar")]
  bicublin_420: Option<std::boxed::Box<planar_8bit::BicublinYuv420>>,
  /// Native-tier join state for the non-4:2:0 8-bit planar families
  /// (`Yuv422p` / `Yuv444p` / `Yuv440p`) — the sibling of
  /// [`Self::native_420`] for chroma layouts that are not half-resolution
  /// in both axes. A given sink instantiates exactly one planar family, so
  /// one field is shared across the three impls; lazily created in
  /// `process`, reset in `begin_frame`.
  #[cfg(feature = "yuv-planar")]
  native_planar: Option<std::boxed::Box<planar_8bit::NativePlanarYuv>>,
  /// RGB-free YUV-domain HSV-only **area** join for the planar ROW-STAGE
  /// resample (#263 follow-up): when ONLY `with_hsv()` is attached (no RGB /
  /// RGBA), the row-stage path bins Y / U / V on their own grids straight to
  /// output resolution and converts each output row through
  /// `yuv_444_to_hsv_row` — RGB-free, and bit-identical to the native fast
  /// tier's HSV-only output. Used by the planar 8-bit families (`Yuv420p` /
  /// `Yuv410p` / `Yuv411p` / `Yuv422p` / `Yuv444p` / `Yuv440p`). The 8-bit
  /// semi-planar family (`Nv12` etc.), whose chroma arrives interleaved and
  /// whose 4:2:0 / 4:4:0 chroma-plan builders are `yuv-planar`-gated, keeps
  /// the RGB-staged HSV-only row-stage for now (a documented follow-up), so
  /// this field is `yuv-planar`-gated. Lazily created in `process` on an
  /// HSV-only row-stage row, reset in `begin_frame`. Boxed to keep the
  /// (large) `MixedSinker` stack footprint down; the heap allocation happens
  /// lazily on the first such row.
  #[cfg(feature = "yuv-planar")]
  hsv_planar: Option<std::boxed::Box<planar_resample::HsvDirectPlanarYuv>>,
  /// Native-tier join state for the HIGH-BIT planar 4:2:0 family
  /// (`Yuv420p10/12/14/16`) — the `u16` twin of [`Self::native_420`].
  /// Lazily created in `process`, reset in `begin_frame`.
  #[cfg(feature = "yuv-planar")]
  native_420_u16: Option<std::boxed::Box<subsampled_4_2_0_high_bit::NativeYuv420U16>>,
  /// Native-tier join state for the HIGH-BIT non-4:2:0 planar families
  /// (`Yuv422p10/12/14/16` / `Yuv444p10/12/14/16` / `Yuv440p10/12`) — the
  /// `u16` twin of [`Self::native_planar`] and the non-4:2:0 sibling of
  /// [`Self::native_420_u16`]. A given sink instantiates exactly one planar
  /// family, so one field is shared across the three impls; lazily created in
  /// `process`, reset in `begin_frame`.
  #[cfg(feature = "yuv-planar")]
  native_planar_u16: Option<std::boxed::Box<planar_high_bit_native::NativePlanarYuvU16>>,
  /// De-pack staging for the native fast tier of the **MSB-aligned high-bit
  /// planar** 4:4:4 family ([`Yuv444p10Msb`](crate::source::Yuv444p10Msb) /
  /// [`Yuv444p12Msb`](crate::source::Yuv444p12Msb)). Each plane is
  /// high-bit-packed (`logical << (16 - BITS)`); the native wrapper
  /// wire-decodes (`from_le` / `from_be`) and DE-PACKS (`>> (16 - BITS)`) each
  /// of Y / U / V into these host-native LOGICAL `u16` scratches before the
  /// reused HIGH-BIT non-4:2:0 planar join
  /// ([`planar_high_bit_native::yuv_planar16_process_native`]) bins them at
  /// `Yuv444p` geometry (the MSB twin of the low-bit family, which feeds the
  /// join directly — no de-pack). `yuv444_msb_y_depack` grows to `width` on
  /// every native row; `yuv444_msb_u_depack` / `yuv444_msb_v_depack` grow to
  /// `width` each only on a colour native row (4:4:4 chroma is full-width);
  /// empty otherwise.
  #[cfg(feature = "yuv-planar")]
  yuv444_msb_y_depack: Vec<u16>,
  /// U-plane de-pack scratch for the native MSB-aligned planar 4:4:4 tier;
  /// twin of [`Self::yuv444_msb_y_depack`].
  #[cfg(feature = "yuv-planar")]
  yuv444_msb_u_depack: Vec<u16>,
  /// V-plane de-pack scratch for the native MSB-aligned planar 4:4:4 tier;
  /// twin of [`Self::yuv444_msb_u_depack`].
  #[cfg(feature = "yuv-planar")]
  yuv444_msb_v_depack: Vec<u16>,
  /// Half-width U / V de-interleave staging for the native 4:2:0
  /// decimation tier of the **semi-planar** family
  /// ([`Nv12`](crate::source::Nv12) / [`Nv21`](crate::source::Nv21)):
  /// the interleaved chroma row splits into these two `width / 2`
  /// scratch planes so [`planar_8bit::yuv420p_process_native`] bins
  /// Y + U + V through the same per-plane join the planar twin uses.
  /// Lazily grown to `width / 2` `u8` each on the first chroma-bearing
  /// native row; empty otherwise. Gated to the intersection — the
  /// native tier reuses the planar join, so it only exists when
  /// `yuv-planar` is also compiled.
  #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
  semi_planar_u_half: Vec<u8>,
  /// V-plane twin of [`Self::semi_planar_u_half`].
  #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
  semi_planar_v_half: Vec<u8>,
  /// De-pack staging for the native 4:2:0 decimation tier of the
  /// **high-bit semi-planar** P-format family
  /// ([`P010`](crate::source::P010) / [`P012`](crate::source::P012) /
  /// [`P016`](crate::source::P016)). The P-format Y plane is
  /// high-bit-packed (`logical << (16 - BITS)`) and the chroma plane is
  /// interleaved + high-packed; the native wrapper de-interleaves + DE-PACKS
  /// each wire plane into these host-native LOGICAL `u16` scratches before
  /// the reused planar high-bit join ([`subsampled_4_2_0_high_bit::yuv420p16_process_native`])
  /// bins Y + U + V. `p0xx_y_half` grows to `width` on every native row;
  /// `p0xx_u_half` / `p0xx_v_half` grow to `width / 2` each only on a
  /// chroma-bearing native row; empty otherwise. Gated to the intersection
  /// — the native tier reuses the planar join, so it only exists when
  /// `yuv-planar` is also compiled.
  #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
  p0xx_y_half: Vec<u16>,
  /// U-plane de-pack scratch for the native high-bit semi-planar tier;
  /// twin of [`Self::p0xx_y_half`] at chroma width.
  #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
  p0xx_u_half: Vec<u16>,
  /// V-plane de-pack scratch for the native high-bit semi-planar tier;
  /// twin of [`Self::p0xx_u_half`].
  #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
  p0xx_v_half: Vec<u16>,
  /// De-pack staging for the native fast tier of the **packed** 8-bit
  /// 4:2:2 YUV family ([`Yuyv422`](crate::source::Yuyv422) /
  /// [`Uyvy422`](crate::source::Uyvy422) / [`Yvyu422`](crate::source::Yvyu422)).
  /// Each format carries one fully-interleaved plane (`Y U Y V …` etc., two
  /// bytes per pixel); the native wrapper de-packs each packed row into
  /// these separate Y (`width`) / U (`width / 2`) / V (`width / 2`)
  /// scratch planes at the per-format byte offsets, then the reused planar
  /// 8-bit join ([`planar_8bit::yuv_planar_process_native`]) bins Y + U + V
  /// at `Yuv422p` geometry. `packed_yuv_y_full` grows to `width` on every
  /// native row; `packed_yuv_u_half` / `packed_yuv_v_half` grow to
  /// `width / 2` each only on a colour native row; empty otherwise. Gated
  /// to the intersection — the native tier reuses a yuv-planar fn, so it
  /// only exists when `yuv-planar` is also compiled (a `yuv-packed`-solo
  /// build takes the row-stage tail).
  #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
  packed_yuv_y_full: Vec<u8>,
  /// U-plane de-pack scratch for the native packed 4:2:2 tier; twin of
  /// [`Self::packed_yuv_y_full`] at chroma width (`width / 2`).
  #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
  packed_yuv_u_half: Vec<u8>,
  /// V-plane de-pack scratch for the native packed 4:2:2 tier; twin of
  /// [`Self::packed_yuv_u_half`].
  #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
  packed_yuv_v_half: Vec<u8>,
  /// De-pack staging for the native fast tier of the **high-bit packed**
  /// 4:2:2 YUV family ([`Y210`](crate::source::Y210) /
  /// [`Y212`](crate::source::Y212) / [`Y216`](crate::source::Y216)). Each
  /// format carries one fully-interleaved plane of MSB-aligned u16 words in
  /// YUYV order (`Y₀ U Y₁ V …`, four words per 2-pixel group); the native
  /// wrapper wire-decodes (`from_le` / `from_be`), de-packs (`>> (16 - BITS)`),
  /// and de-interleaves each packed row into these separate host-native
  /// LOGICAL Y (`width`) / U (`width / 2`) / V (`width / 2`) scratch planes,
  /// then the reused HIGH-BIT non-4:2:0 planar join
  /// ([`planar_high_bit_native::yuv_planar16_process_native`]) bins Y + U + V
  /// at `Yuv422p` geometry. `y2xx_y_full` grows to `width` on every native
  /// row; `y2xx_u_half` / `y2xx_v_half` grow to `width / 2` each only on a
  /// colour native row; empty otherwise. The `u16` analog of
  /// [`Self::packed_yuv_y_full`] (the 8-bit packed tier) combined with the
  /// MSB-bridge the high-bit semi-planar tier ([`Self::p0xx_y_half`]) carries.
  /// Gated to the intersection — the native tier reuses a yuv-planar fn, so it
  /// only exists when `yuv-planar` is also compiled (a `y2xx`-solo build takes
  /// the row-stage tail).
  #[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
  y2xx_y_full: Vec<u16>,
  /// U-plane de-pack scratch for the native high-bit packed 4:2:2 tier; twin
  /// of [`Self::y2xx_y_full`] at chroma width (`width / 2`).
  #[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
  y2xx_u_half: Vec<u16>,
  /// V-plane de-pack scratch for the native high-bit packed 4:2:2 tier; twin
  /// of [`Self::y2xx_u_half`].
  #[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
  y2xx_v_half: Vec<u16>,
  /// De-pack staging for the native fast tier of the **exotic 10-bit packed
  /// 4:2:2** YUV format ([`V210`](crate::source::V210)). V210 packs 12 × 10-bit
  /// samples per 16-byte word (four 32-bit LE/BE words) = 6 pixels (6 Y + 3 Cb +
  /// 3 Cr); the native wrapper wire-decodes (`from_le` / `from_be`) and
  /// bit-extracts each word into these separate host-native LOGICAL Y (`width`)
  /// / U (`width / 2`) / V (`width / 2`) scratch planes, then the reused HIGH-BIT
  /// non-4:2:0 planar join
  /// ([`planar_high_bit_native::yuv_planar16_process_native`]) bins Y + U + V at
  /// `Yuv422p10` geometry. `v210_y_full` grows to `width` on every native row;
  /// `v210_u_half` / `v210_v_half` grow to `width / 2` each only on a colour
  /// native row; empty otherwise. The fixed-`BITS = 10` sibling of
  /// [`Self::y2xx_y_full`] (the high-bit packed 4:2:2 tier), with the V210 word
  /// packing in place of the YUYV MSB-aligned u16 words. Gated to the
  /// intersection — the native tier reuses a yuv-planar fn, so it only exists
  /// when `yuv-planar` is also compiled (a `v210`-solo build takes the row-stage
  /// tail).
  #[cfg(all(feature = "v210", feature = "yuv-planar"))]
  v210_y_full: Vec<u16>,
  /// U-plane de-pack scratch for the native V210 tier; twin of
  /// [`Self::v210_y_full`] at chroma width (`width / 2`).
  #[cfg(all(feature = "v210", feature = "yuv-planar"))]
  v210_u_half: Vec<u16>,
  /// V-plane de-pack scratch for the native V210 tier; twin of
  /// [`Self::v210_u_half`].
  #[cfg(all(feature = "v210", feature = "yuv-planar"))]
  v210_v_half: Vec<u16>,
  /// De-pack staging for the native fast tier of the **8-bit packed 4:4:4**
  /// non-alpha YUV format ([`Vuyx`](crate::source::Vuyx) — bytes `V U Y X`
  /// per pixel, `X` padding). The native wrapper de-packs each packed row
  /// into these separate Y / U / V scratch planes — all FULL width (`width`),
  /// since 4:4:4 has no chroma subsampling — then the reused planar 8-bit join
  /// ([`planar_8bit::yuv_planar_process_native`]) bins Y + U + V at
  /// [`Yuv444p`](crate::source::Yuv444p) geometry (`chroma_w = width`,
  /// `chroma_vsub = 1`). `packed_444_y_full` grows to `width` on every native
  /// row; `packed_444_u_full` / `packed_444_v_full` grow to `width` each only
  /// on a colour native row; empty otherwise. The 4:4:4 (full-width chroma)
  /// analog of [`Self::packed_yuv_y_full`] (the 8-bit packed 4:2:2 tier).
  /// Gated to the intersection — the native tier reuses a yuv-planar fn, so it
  /// only exists when `yuv-planar` is also compiled (a `yuv-444-packed`-solo
  /// build takes the row-stage tail).
  #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
  packed_444_y_full: Vec<u8>,
  /// U-plane de-pack scratch for the native 8-bit packed 4:4:4 tier; twin of
  /// [`Self::packed_444_y_full`] at FULL chroma width (`width`).
  #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
  packed_444_u_full: Vec<u8>,
  /// V-plane de-pack scratch for the native 8-bit packed 4:4:4 tier; twin of
  /// [`Self::packed_444_u_full`].
  #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
  packed_444_v_full: Vec<u8>,
  /// De-pack staging for the native fast tier of the **high-bit packed 4:4:4**
  /// non-alpha YUV formats ([`V410`](crate::source::V410) — one 10-bit
  /// bit-field-packed u32 word per pixel; [`Xv36`](crate::source::Xv36) — four
  /// MSB-aligned 12-bit u16 slots `U Y V A` per pixel, `A` padding). Each
  /// wrapper bit-extracts / de-packs its OWN wire layout into these separate
  /// host-native LOGICAL u16 Y / U / V scratch planes — all FULL width
  /// (`width`), since 4:4:4 has no chroma subsampling — then the reused HIGH-BIT
  /// non-4:2:0 planar join
  /// ([`planar_high_bit_native::yuv_planar16_process_native`]) bins Y + U + V at
  /// [`Yuv444p10`](crate::source::Yuv444p10) /
  /// [`Yuv444p12`](crate::source::Yuv444p12) geometry (`chroma_w = width`,
  /// `chroma_vsub = 1`). `packed_444_y_full_u16` grows to `width` on every
  /// native row; `packed_444_u_full_u16` / `packed_444_v_full_u16` grow to
  /// `width` each only on a colour native row; empty otherwise. A given sink
  /// instantiates exactly one format, so V410 and Xv36 (both high-bit 4:4:4)
  /// share these fields. The 4:4:4 (full-width chroma) analog of
  /// [`Self::y2xx_y_full`] (the high-bit packed 4:2:2 tier). Gated to the
  /// intersection — the native tier reuses a yuv-planar fn, so it only exists
  /// when `yuv-planar` is also compiled.
  #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
  packed_444_y_full_u16: Vec<u16>,
  /// U-plane de-pack scratch for the native high-bit packed 4:4:4 tier; twin of
  /// [`Self::packed_444_y_full_u16`] at FULL chroma width (`width`).
  #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
  packed_444_u_full_u16: Vec<u16>,
  /// V-plane de-pack scratch for the native high-bit packed 4:4:4 tier; twin of
  /// [`Self::packed_444_u_full_u16`].
  #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
  packed_444_v_full_u16: Vec<u16>,
  /// The native / row-stage route chosen on the first resampled row of a
  /// frame; a mid-frame change is rejected. The two tiers carry
  /// independent, in-order stream state, so flipping
  /// [`Self::set_native`] mid-frame would split one frame across two
  /// incompatible state machines. Shared by every native 4:2:0 family
  /// the guard covers — the 8-bit planar (Yuv420p) and high-bit planar
  /// (Yuv420p10/12/14/16) tiers (both `yuv-planar`), the 8-bit
  /// semi-planar (Nv12/Nv21) tier, and the high-bit semi-planar P-format
  /// (P010/P012/P016) tier (both additionally `yuv-semi-planar`, a subset
  /// of `yuv-planar`). Reset to `None` per frame: via
  /// `reset_high_bit_yuv_streams` for the high-bit families and inline in
  /// the 8-bit families' `begin_frame`. Gated to `yuv-planar`, the union
  /// where the native tier exists (a yuv-semi-planar-solo build can't
  /// enable the native tier, so no guard is needed there).
  #[cfg(feature = "yuv-planar")]
  frozen_native_route: Option<bool>,
  /// The [`AveragingDomain`] captured on the first output-bearing resampled row
  /// of a frame; a mid-frame change is rejected. The domain selects the pipeline
  /// stage the area average is spliced at (encoded codes vs linear light vs
  /// premultiplied), so flipping [`Self::set_averaging_domain`] mid-frame would
  /// split one frame across two incompatible averaging stages. Completes the
  /// per-frame freeze family (output set, native route, transfer, domain). Reset
  /// to `None` per frame in the planar 8-bit families' `begin_frame` (the only
  /// formats whose dispatch reads the configured domain). Gated to `yuv-planar`,
  /// where that dispatch — and so the only reader of this field — lives; the
  /// high-bit / semi-planar / packed dispatches splice `AveragingDomain::Encoded`
  /// as a literal and never consult it.
  ///
  /// [`AveragingDomain`]: crate::resample::AveragingDomain
  #[cfg(feature = "yuv-planar")]
  frozen_domain: Option<crate::resample::AveragingDomain>,
  /// RFC #238: the effective 4:2:2 chroma sampling phase (`true` =
  /// centered, `false` = co-sited) frozen on the first output-bearing
  /// row of the current frame. A later row observing a different siting
  /// is rejected with [`MixedSinkerError::ChromaSitingChanged`] so the
  /// frame never bins a mixture of phases; cleared each `begin_frame`.
  #[cfg(feature = "yuv-planar")]
  frozen_chroma_centered: Option<bool>,
  /// RFC #238 S4-B companion to [`Self::frozen_chroma_centered`]: the effective
  /// 4:2:0 **vertical** chroma phase (`true` = `Bottom` `v = 1`, `false` =
  /// co-sited vertical) frozen on the first output-bearing row of the current
  /// frame. `Center` and `Bottom` are BOTH horizontally centered, so the
  /// centered flag alone cannot tell them apart; freezing the vertical phase
  /// too rejects a mid-frame `Center` ⇆ `Bottom` flip with
  /// [`MixedSinkerError::ChromaSitingChanged`]. Only the `Yuv420p` dispatch
  /// (the sole vertical-siting reader) sets it; every other format leaves it
  /// `None` and passes `v_phase = 0`. Cleared each `begin_frame`.
  #[cfg(feature = "yuv-planar")]
  frozen_chroma_bottom_v: Option<bool>,
  /// RFC #238 Top companion to [`Self::frozen_chroma_bottom_v`]: the effective
  /// 4:2:0 **Top** vertical phase (`true` = `Top` / `TopLeft` `v = 0` forward
  /// fold, `false` otherwise) frozen on the first output-bearing row of the
  /// current frame. `Center` and `Top` are BOTH horizontally centered and BOTH
  /// carry `frozen_chroma_bottom_v = Some(false)`, so neither prior flag can
  /// tell them apart; freezing the Top phase too rejects a mid-frame
  /// `Center`/`Bottom` ⇆ `Top` flip with
  /// [`MixedSinkerError::ChromaSitingChanged`]. Only the `Yuv420p` dispatch sets
  /// it; every other format leaves it `None`. Cleared each `begin_frame`.
  #[cfg(feature = "yuv-planar")]
  frozen_chroma_top_v: Option<bool>,
  /// Forward one-row output-delay state for the 4:2:0 **Top** (`v = 0`) identity
  /// decode (RFC #238) — the mirror of the backward [`Self::chroma_prev`]
  /// lookback `Bottom` uses. A `Top` chroma sample sits on the TOP luma row of
  /// each pair, so an ODD output luma row `2i+1` needs the *next* chroma row
  /// `c[i+1]`, which the row-at-a-time stream has not been fed when `2i+1`
  /// arrives. Its colour output is therefore HELD here until the following even
  /// row `2i+2` (carrying `c[i+1]`) arrives, then box-blended and emitted.
  /// `Some((row, matrix, full_range))` records the deferred odd row's index and
  /// its decode parameters; `None` when nothing is pending. Reset to `None` each
  /// `begin_frame` so a held row never leaks across frames. Gated to
  /// `yuv-planar`, like [`Self::chroma_prev`].
  #[cfg(feature = "yuv-planar")]
  chroma_top_pending: Option<(usize, crate::ColorMatrix, bool)>,
  /// Buffered Y row for the deferred [`Self::chroma_top_pending`] odd output row
  /// — a copy of that row's `width` `u8` luma, retained because the walker's row
  /// borrow is invalidated after `process` returns. Lazily grown to `width` `u8`
  /// on the first deferred `Top` row; empty otherwise. Gated to `yuv-planar`.
  #[cfg(feature = "yuv-planar")]
  chroma_top_y: Vec<u8>,
  /// Buffered **alpha** row for the deferred [`Self::chroma_top_pending`] odd
  /// output row on the 8-bit YUVA 4:2:0 **Top** (`v = 0`) forward one-row delay
  /// (RFC #238) — the alpha companion of [`Self::chroma_top_y`]. The 8-bit
  /// `Yuva420p` carries a full-resolution `u8` alpha plane; the Top forward delay
  /// defers the odd row's whole decode, so its alpha is held here alongside its
  /// luma until the following even row flushes it. Lazily grown to `width` `u8`
  /// on the first deferred `Top` row; empty otherwise. Gated to `yuv-planar`.
  #[cfg(feature = "yuv-planar")]
  chroma_top_a: Vec<u8>,
  /// Lazily grown to `3 * width` bytes when HSV is requested without a
  /// user RGB buffer. Empty otherwise.
  ///
  /// Consumed by per-format `process` impls that derive HSV from RGB
  /// via the lazy scratch path. Under `--features "alloc"` alone (no
  /// per-format family), no `process` impl reads this field, so the
  /// cfg enumerates every source family.
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  rgb_scratch: Vec<u8>,
  /// Full-width `u8` chroma staging for the siting-aware 4:2:0 upsample
  /// (#302). Holds the horizontally upsampled U then V planes back-to-back
  /// (`2 * width` bytes: `[0..width]` = U, `[width..2*width]` = V) so the
  /// centered-siting identity path can reuse the 4:4:4 decode kernels. Shared
  /// by the planar `Yuv420p` and the semi-planar `Nv12` / `Nv21` decodes (the
  /// latter de-interleave the interleaved UV into the half-width
  /// [`Self::semi_planar_u_half`] / `_v_half` scratch first, then upsample into
  /// this buffer). Lazily grown to `2 * width` `u8` on the first centered-siting
  /// chroma row; empty otherwise (the default left/unspecified siting never
  /// touches it). Gated to `yuv-planar` (the 4:4:4 kernels + scalar upsample it
  /// reuses live there; the semi-planar path additionally requires
  /// `yuv-semi-planar`).
  #[cfg(feature = "yuv-planar")]
  chroma_full: Vec<u8>,
  /// One-row chroma **lookback** for the bottom-sited (`AVCHROMA_LOC_BOTTOM`)
  /// vertical-phase 4:2:0 upsample (#302). Holds a copy of a chroma row's
  /// half-width U then V back-to-back (`width` `u8`: `[0..width/2]` = `U`,
  /// `[width/2..width]` = `V`), so an *even* output luma row `2i` can vertically
  /// box-blend chroma rows `i-1` (here) and `i` (the current row) before the
  /// horizontal centered upsample. WHICH chroma row this buffer currently holds
  /// is tracked by [`Self::chroma_prev_row`]; the blend only trusts it when it
  /// holds exactly the wanted predecessor `i-1`, otherwise it falls back to the
  /// top-edge clamp (current row as its own predecessor) — so a direct
  /// `process` caller that replays / skips / reorders rows or attaches colour
  /// late can never blend STALE chroma from an older pair or a previous frame.
  /// Lazily grown to `width` `u8` on the first bottom-sited chroma row; empty
  /// otherwise (only `ChromaLocation::Bottom` touches it — every other siting,
  /// the default included, keeps the vertical-replicate decode). Gated to
  /// `yuv-planar`, like [`Self::chroma_full`] which it feeds.
  #[cfg(feature = "yuv-planar")]
  chroma_prev: Vec<u8>,
  /// Which chroma row [`Self::chroma_prev`] currently holds, or `None` when it
  /// holds nothing trustworthy (`MixedSinker::new`, after every
  /// [`begin_frame`](crate::PixelSink::begin_frame), and before the first
  /// bottom-sited row) — the validity tag that makes the bottom-sited
  /// vertical-phase lookback row-order-safe (#302). The walker feeds rows in
  /// order so it tracks `idx / 2`, but a direct `process` caller can break that
  /// sequence; the bottom even-row blend at pair `k` checks this equals
  /// `Some(k - 1)` and clamps (current row as its own predecessor) when it does
  /// not, never blending stale data. Reset to `None` in `begin_frame` so no
  /// stale row leaks across frames. Gated to `yuv-planar`, like
  /// [`Self::chroma_prev`].
  #[cfg(feature = "yuv-planar")]
  chroma_prev_row: Option<usize>,
  /// Full-width `u16` chroma staging for the **high-bit** siting-aware upsample
  /// (#302) — the `u16` twin of [`Self::chroma_full`]. Shared by the planar
  /// `Yuv420p9` … `Yuv420p16` / `Yuv422p9` … `Yuv422p16` decodes (which lay it
  /// out as the horizontally upsampled U then V planes back-to-back:
  /// `[0..width]` = U, `[width..2*width]` = V) AND the semi-planar P-format
  /// `P010` / `P012` / `P016` (4:2:0) / `P210` / `P212` / `P216` (4:2:2) decodes
  /// (which lay it out as a full-width INTERLEAVED `U V U V…` plane). Either way
  /// it is `2 * width` `u16` in the source's wire byte order, so the
  /// centered-siting identity path can reuse the high-bit 4:4:4 decode kernels.
  /// Lazily grown to `2 * width` `u16` on the first centered-siting chroma row;
  /// empty otherwise (the default left/unspecified siting never touches it).
  /// Gated to `yuv-planar` (the high-bit 4:4:4 kernels + `u16` scalar upsample
  /// it reuses live there).
  #[cfg(feature = "yuv-planar")]
  chroma_full_u16: Vec<u16>,
  /// One-row chroma **lookback** for the bottom-sited (`AVCHROMA_LOC_BOTTOM`)
  /// vertical-phase HIGH-BIT 4:2:0 upsample (RFC #238 S6d/S6e) — the `u16` twin
  /// of [`Self::chroma_prev`]. Holds a copy of one chroma row (`width` `u16`, in
  /// the source's wire byte order) so an *even* output luma row `2i` can
  /// vertically box-blend chroma rows `i-1` (here) and `i` (the current row)
  /// before the horizontal centered upsample. The planar `Yuv420p9` …
  /// `Yuv420p16` sinks lay it out as the half-width U then V back-to-back
  /// (`[0..width/2]` = `U`, `[width/2..width]` = `V`, RFC #238 S6d); the
  /// semi-planar `P010` / `P012` / `P016` sinks lay it out as the packed
  /// INTERLEAVED `U V U V…` half-row (RFC #238 S6e) — the SAME `width` `u16`,
  /// whichever layout its owning monomorphized sink writes. WHICH chroma row it
  /// holds is tracked by the SHARED [`Self::chroma_prev_row`] tag
  /// (element-agnostic, so the `u8` and `u16` bottom paths — never both live on
  /// one monomorphized sink — reuse it); the blend trusts it only for the wanted
  /// predecessor `i-1`, else falls back to the top-edge clamp. Lazily grown to
  /// `width` `u16` on the first bottom-sited chroma row; empty otherwise (only
  /// `ChromaLocation::Bottom` on a `Yuv420p9` … `Yuv420p16` / `P010` / `P012` /
  /// `P016` sink touches it). Gated to `yuv-planar`, like
  /// [`Self::chroma_full_u16`] which it feeds.
  #[cfg(feature = "yuv-planar")]
  chroma_prev_u16: Vec<u16>,
  /// Buffered wire-format Y row for the deferred [`Self::chroma_top_pending`] odd
  /// output row on the HIGH-BIT 4:2:0 **Top** (`v = 0`) forward one-row delay
  /// (RFC #238) — the `u16` twin of [`Self::chroma_top_y`]. Holds a copy of the
  /// odd row's `width` `u16` luma in the source's wire byte order (the walker's
  /// row borrow is invalidated after `process` returns), retained until the
  /// following even row flushes it. Shared by the planar `Yuv420p9` …
  /// `Yuv420p16` identity decode and the row-stage / filter reconstruction
  /// tiers. Lazily grown to `width` `u16` on the first deferred `Top` row; empty
  /// otherwise. Gated to `yuv-planar`, like [`Self::chroma_prev_u16`].
  #[cfg(feature = "yuv-planar")]
  chroma_top_y_u16: Vec<u16>,
  /// Buffered wire-format **alpha** row for the deferred
  /// [`Self::chroma_top_pending`] odd output row on the HIGH-BIT 4:2:0 YUVA
  /// **Top** (`v = 0`) forward one-row delay (RFC #238) — the alpha companion of
  /// [`Self::chroma_top_y_u16`]. YUVA carries a FULL-RESOLUTION alpha plane
  /// (never chroma-subsampled) alongside Y; the Top forward delay defers the odd
  /// row's whole decode, so its alpha — like its luma — must be held until the
  /// following even row flushes it (the walker's row borrow expires after
  /// `process` returns). Holds a copy of the odd row's `width` `u16` alpha in the
  /// source's wire byte order. Only a `Yuva420p9` … `Yuva420p16` sink under
  /// `Top` / `TopLeft` touches it; lazily grown to `width` `u16` on the first
  /// deferred `Top` row, empty otherwise. Gated to `yuv-planar`, like
  /// [`Self::chroma_top_y_u16`].
  #[cfg(feature = "yuv-planar")]
  chroma_top_a_u16: Vec<u16>,
  /// Source-width `u8` luma staging for the **packed YUV 4:2:2** resample
  /// path (the interleaved Y bytes are de-interleaved here via the format's
  /// own `*_to_luma_row` kernel — the exact Y→luma derivation the direct
  /// path uses) and the `Ya8` gray+alpha resample (the native Y bytes of
  /// the packed `[Y, A]` row are de-interleaved here via `ya8_to_luma_row`)
  /// before feeding the single-channel [`Self::luma_stream`]. The colour
  /// stream simultaneously stages its RGB / RGBA row in
  /// [`Self::rgb_scratch`] / [`Self::rgba_scratch`], so the Y row needs its
  /// own buffer rather than sharing that scratch. Lazily grown to `width`
  /// `u8`; empty otherwise. Gated to `yuv-packed` / `gray`.
  #[cfg(any(feature = "yuv-packed", feature = "gray"))]
  luma_scratch: Vec<u8>,
  /// Source-width **native-channel** `u8` staging for the legacy 16-bit
  /// packed-RGB ([`Rgb565`](crate::source::Rgb565) and family) resample
  /// path: each packed source row is unpacked to its 3 **native** R/G/B
  /// channels (5/6/5, 5/5/5 or 4/4/4 values, each `<= 63`, NOT expanded
  /// to 8-bit) here before feeding the shared u8 [`Self::rgb_stream`], so
  /// the area mean is taken at native depth. Lazily grown to `3 * width`
  /// `u8`; empty otherwise. Gated to `rgb-legacy`.
  #[cfg(feature = "rgb-legacy")]
  legacy_rgb_native_scratch: Vec<u8>,
  /// Out-width **re-packed source-format** `u8` staging for the legacy
  /// 16-bit packed-RGB resample tail. Per finalized output row the binned
  /// native R/G/B channels are re-packed back into the source's packed
  /// `u16` word (LE bytes, `2 * out_width`) here, then the **exact**
  /// direct `*_to_*` kernels run over it — so every output is
  /// byte-identical to a direct conversion of the area-downscaled
  /// source-format frame. The integer twin of the `gbr` family's
  /// [`Self::rgb_plane_scratch_f32`] de-interleave staging. Lazily grown
  /// to `2 * out_width` `u8`; empty for a no-output sink. Gated to
  /// `rgb-legacy`.
  #[cfg(feature = "rgb-legacy")]
  legacy_rgb_packed_scratch: Vec<u8>,
  /// Source-width `u16` RGB staging for high-bit packed-RGB resampling:
  /// the wire row converts here before feeding [`Self::rgb_stream_u16`].
  /// Lazily grown to `3 * width` `u16`; empty otherwise. Gated to `rgb`
  /// (high-bit packed RGB), `gbr` (high-bit planar GBR scatters its
  /// G/B/R planes here before the same u16 tail), the high-bit packed
  /// YUV color groups (`yuv-444-packed` / `y2xx` / `v210`), the
  /// high-bit planar YUV color groups (`yuv-planar`: Yuv444p / Yuv422p
  /// 10/12/14/16), and the high-bit semi-planar P-format color groups
  /// (`yuv-semi-planar`) which stage their converted native-u16 RGB row
  /// here.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  ))]
  rgb_scratch_u16: Vec<u16>,
  /// Source-width canonical `R, G, B, A` `u8` staging for the alpha-aware
  /// u8-color resample tails: each source row is converted to canonical
  /// RGBA (`Rgba` identity, `Bgra` swap, `Argb` / `Abgr` rotate α to slot
  /// 3; `Gbrap` de-interleaves its G/B/R/A planes; `Ya8` replicates Y;
  /// `Vuya` / `Ayuv64` run the u8 `YUV→RGB` kernel with source α) here —
  /// and, under [`AlphaMode::Premultiplied`], premultiplied in place —
  /// before feeding the 4-channel [`Self::rgba_stream`]. Lazily grown to
  /// `4 * width` `u8`; empty otherwise. Gated to `rgb` / `gbr` / `gray` /
  /// `mono` / `yuv-444-packed`. (`mono` joins for `Pal8`, which stages its
  /// per-pixel palette lookup `[R, G, B, A]` here before binning.)
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "yuva",
    feature = "yuv-444-packed"
  ))]
  rgba_scratch: Vec<u8>,
  /// Source-width canonical `R, G, B, A` host-native `u16` staging for the
  /// alpha-aware native-u16-color resample tails (`Rgba64` / `Bgra64`, the
  /// high-bit planar GBR+alpha family `Gbrap10` … `Gbrap16`, `Ya16`, and
  /// the packed YUVA `Ayuv64`'s independent u16 color group): the wire row
  /// converts to host-native u16 RGBA here (and is premultiplied in place
  /// under [`AlphaMode::Premultiplied`]) before feeding the 4-channel
  /// [`Self::rgba_stream_u16`]. Lazily grown to `4 * width` `u16`; empty
  /// otherwise. Gated to `rgb` / `gbr` / `gray` / `yuv-444-packed`.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "yuva",
    feature = "yuv-444-packed"
  ))]
  rgba_scratch_u16: Vec<u16>,
  /// Out-width host-native straight `R, G, B, A` `u16` staging for the
  /// native-u16-color resample tails: per finalized output row the binned
  /// native RGBA is resolved to its straight form here (a copy in
  /// [`AlphaMode::Straight`], an un-premultiply in
  /// [`AlphaMode::Premultiplied`]), then every output derives from this
  /// single straight row (the high-bit packed RGBA tail's narrowed u8
  /// outputs and `Ayuv64`'s u16 color group's rgb_u16 / rgba_u16). Lazily
  /// grown to `4 * out_width` `u16`; empty for an output-less sink. Gated
  /// to `rgb` / `gbr` / `gray` / `yuv-444-packed`.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "yuva",
    feature = "yuv-444-packed"
  ))]
  rgba_color_scratch_u16: Vec<u16>,
  /// Source-width host-native `R, G, B` `u32` staging (`3 * width`) for the
  /// native-`u32` packed-RGB resample tier: the wire row converts here (the
  /// `BE` swap only — NO depth narrow) before feeding [`Self::rgb_stream_u32`]
  /// / [`Self::rgb_filter_stream_u32`], so binning runs at full `u32` precision
  /// (0-ULP, issue #289). Lazily grown; empty otherwise. Gated to `rgb` / `gbr`.
  #[cfg(any(feature = "rgb", feature = "gbr"))]
  rgb_scratch_u32: Vec<u32>,
  /// Source-width canonical host-native `R, G, B, A` `u32` staging (`4 * width`)
  /// for the native-`u32` alpha-aware packed-RGBA resample tier (and
  /// premultiplied in place at `u32` under
  /// [`AlphaMode::Premultiplied`]) before feeding [`Self::rgba_stream_u32`] /
  /// [`Self::rgba_filter_stream_u32`]. Lazily grown; empty otherwise. Gated to
  /// `rgb` / `gbr`.
  #[cfg(any(feature = "rgb", feature = "gbr"))]
  rgba_scratch_u32: Vec<u32>,
  /// Out-width straight `R, G, B, A` `u32` staging (`4 * out_width`) for the
  /// native-`u32` packed-RGBA area tier: per finalized output row the binned
  /// premultiplied `u32` RGBA is un-premultiplied here (the straight path reads
  /// the binned row directly), then narrowed `>> 16` and derived. Lazily grown;
  /// empty for a straight-alpha or output-less sink. Gated to `rgb` / `gbr`.
  #[cfg(any(feature = "rgb", feature = "gbr"))]
  rgba_color_scratch_u32: Vec<u32>,
  /// Source-width host-native `u16` luma staging for the
  /// [`Gray16`](crate::source::Gray16) resample path: the wire `Gray16`
  /// row converts here (source wire `BE` → host-native u16, the same
  /// kernel the direct `luma_u16` path uses) before feeding
  /// [`Self::luma_stream_u16`]. The high-bit packed YUV families
  /// (`yuv-444-packed` / `y2xx` / `v210`) reuse it to stage their
  /// de-interleaved native Y row before the same u16 luma stream, as do
  /// the high-bit planar YUV families (`yuv-planar`: Yuv444p / Yuv422p
  /// 10/12/14/16) staging their host-native Y plane, as do the high-bit
  /// semi-planar P-format families (`yuv-semi-planar`) staging their
  /// de-packed native Y. Lazily grown to `width` `u16`; empty otherwise.
  /// Gated to `gray` / `yuv-444-packed` / `y2xx` / `v210` / `yuv-planar`
  /// / `yuv-semi-planar`.
  #[cfg(any(
    feature = "gray",
    feature = "yuva",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  ))]
  luma_scratch_u16: Vec<u16>,
  /// Source-width host-native `u32` luma staging for the
  /// [`Gray32`](crate::source::Gray32) resample path: the wire `Gray32` row
  /// converts here (source wire `BE` → host-native `u32` value) before
  /// feeding [`Self::luma_stream_u32`] / [`Self::luma_filter_stream_u32`],
  /// so binning runs at native `u32` precision and each output narrows only
  /// afterwards (0-ULP, closes issue #289). Lazily grown to `width` `u32`;
  /// empty otherwise. Gated to `gray`.
  #[cfg(feature = "gray")]
  luma_scratch_u32: Vec<u32>,
  /// Source-width host-native `f32` luma staging for the
  /// [`Grayf32`](crate::source::Grayf32) resample path: the wire
  /// `Grayf32` row converts here (source wire `BE` → host-native f32 via
  /// the same kernel the direct `luma_f32` path uses) before feeding
  /// [`Self::luma_stream_f32`]. Lazily grown to `width` `f32`; empty
  /// otherwise. Gated to `gray`.
  #[cfg(feature = "gray")]
  luma_scratch_f32: Vec<f32>,
  /// Row-stage **3-channel** `f32` **area** stream for the
  /// [`Yaf32`](crate::source::Yaf32) /
  /// [`Yaf16`](crate::source::Yaf16) (widened) resample path. Channel
  /// layout per pixel is `[Yc, A, Y]`: `Yc` is the colour luma fed straight
  /// (`= Y`) or premultiplied (`= Y * A`), `A` the real source alpha, and
  /// `Y` the independent native luma (mode-independent `mean(Y)`, so
  /// `luma*` outputs stay byte-identical to [`Grayf32`](crate::source::Grayf32)
  /// for the same Y). The engine bins each channel independently, so channel
  /// 2 equals a 1-channel native-Y bin. Lazily created in `process`, reset in
  /// `begin_frame`. Gated to `gray`.
  #[cfg(feature = "gray")]
  yaf_stream_f32: Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  /// Row-stage **3-channel** `f32` **filter** stream — the filter twin of
  /// [`Self::yaf_stream_f32`] (straight alpha only; a premultiplied filter
  /// plan routes to the area tail, which surfaces `UnsupportedFilter`). f32
  /// is full-range, so there is no native-depth clamp on the filter output.
  /// Lazily created in `process`, reset in `begin_frame`. Gated to `gray`.
  #[cfg(feature = "gray")]
  yaf_filter_stream_f32: Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  /// Source-width 3-channel `f32` staging for the Yaf resample: each wire
  /// `[Y, A]` pixel decodes here as `[Y, A, Y]` (host-native) before the
  /// optional premultiply (`row[3x] *= row[3x+1]`) and the single feed of
  /// [`Self::yaf_stream_f32`] / [`Self::yaf_filter_stream_f32`]. Lazily grown
  /// to `3 * width` `f32`; empty otherwise. Gated to `gray`.
  #[cfg(feature = "gray")]
  yaf_stage_scratch_f32: Vec<f32>,
  /// Out-width 3-channel `f32` de-interleave staging for the Yaf resample
  /// emit: each finalized binned `[cy, ca, ny]` row de-interleaves into
  /// contiguous `colorY | alpha | nativeY` planes (`colorY` resolved straight
  /// in place) so the direct `grayf32_*` kernels can derive every output.
  /// Lazily grown to `3 * out_width` `f32`; empty otherwise. Gated to `gray`.
  #[cfg(feature = "gray")]
  yaf_plane_scratch_f32: Vec<f32>,
  /// Source-width `f32` RGB staging for packed-float-RGB resampling:
  /// the wire row converts here (host-native, lossless) before feeding
  /// [`Self::rgb_stream_f32`]. Lazily grown to `3 * width` `f32`; empty
  /// otherwise. Gated like [`Self::rgb_stream_f32`]: the `rgb-float`
  /// family fences in the engine, `gbr` already carries it.
  #[cfg(any(
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
    feature = "gbr"
  ))]
  rgb_scratch_f32: Vec<f32>,
  /// Out-width G/B/R `f32` plane staging for the float planar-GBR
  /// ([`Gbrpf32`](crate::source::Gbrpf32)) resample tail. That tail
  /// de-interleaves each binned packed `R, G, B` row back into G/B/R
  /// planes (`[0..w]` = G, `[w..2w]` = B, `[2w..3w]` = R) so it can run
  /// the exact direct `gbrpf32_*` kernels — the `rgb-float`
  /// ([`Rgbf32`](crate::source::Rgbf32)) tail's packed `rgbf32_*` kernels
  /// are not compiled in a `gbr` build. Lazily grown to `3 * out_width`
  /// `f32`; empty for an `rgb_f32`-only sink (which copies the binned row
  /// directly). Gated to `gbr`.
  #[cfg(feature = "gbr")]
  rgb_plane_scratch_f32: Vec<f32>,
  /// Out-width G/B/R `half::f16` plane staging for the half-float
  /// planar-GBR ([`Gbrpf16`](crate::source::Gbrpf16)) resample tail.
  /// There is no `AreaStream<f16>`, so that tail bins in `f32` (the
  /// shared [`Self::rgb_stream_f32`]) and, per finalized output row,
  /// de-interleaves the binned packed `R, G, B` `f32` row into the
  /// `f32` planes ([`Self::rgb_plane_scratch_f32`]), **rounds each
  /// element to `half::f16`** into these planes (`[0..w]` = G,
  /// `[w..2w]` = B, `[2w..3w]` = R), then runs the exact direct
  /// `gbrpf16_*` kernels — so every output is byte-identical to a
  /// direct `Gbrpf16` conversion of the `f32` block-mean rounded to
  /// f16. Lazily grown to `3 * out_width` `half::f16`; empty for a
  /// sink with no f16-plane-derived output. Gated to `gbr`.
  #[cfg(feature = "gbr")]
  rgb_plane_scratch_f16: Vec<half::f16>,
  /// Source-width canonical `R, G, B, A` `f32` staging for the float
  /// planar GBR+alpha resample tail ([`Gbrapf32`](crate::source::Gbrapf32) /
  /// [`Gbrapf16`](crate::source::Gbrapf16)): the G/B/R/A planes interleave
  /// here (host-native f32, for `Gbrapf16` after the f16 -> f32 widen) —
  /// and, under [`AlphaMode::Premultiplied`], are premultiplied in place —
  /// before feeding the 4-channel [`Self::rgba_stream_f32`]. Lazily grown
  /// to `4 * width` `f32`; empty otherwise. Gated to `gbr` (the planar
  /// GBR+alpha float sources) plus the packed-float RGBA sources
  /// ([`Rgbaf32`](crate::source::Rgbaf32) /
  /// [`Rgbaf16`](crate::source::Rgbaf16), the latter widened f16 -> f32),
  /// which stage their wire row here before binning the 4-channel stream.
  #[cfg(any(
    feature = "gbr",
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
  ))]
  rgba_scratch_f32: Vec<f32>,
  /// Out-width host-native straight `R, G, B, A` `f32` staging for the
  /// float planar GBR+alpha resample tail: per finalized output row the
  /// binned packed RGBA is resolved to its straight form here (a copy in
  /// [`AlphaMode::Straight`], an un-premultiply in
  /// [`AlphaMode::Premultiplied`]) before it is de-interleaved into the
  /// G/B/R/A planes every output reads. Lazily grown to `4 * out_width`
  /// `f32`; empty for an output-less sink. Gated to `gbr`.
  #[cfg(feature = "gbr")]
  rgba_color_scratch_f32: Vec<f32>,
  /// Out-width G/B/R/A `f32` plane staging for the float planar GBR+alpha
  /// ([`Gbrapf32`](crate::source::Gbrapf32)) resample tail. That tail
  /// de-interleaves each resolved straight packed `R, G, B, A` row into
  /// G/B/R/A planes (`[0..ow]` = G, `[ow..2ow]` = B, `[2ow..3ow]` = R,
  /// `[3ow..4ow]` = A) so it can run the exact direct `gbrapf32_*` (RGBA) /
  /// `gbrpf32_*` (RGB / luma / hsv) kernels. Lazily grown to `4 * out_width`
  /// `f32`; empty for an `rgba_f32`-only sink (which copies the binned row
  /// directly). Gated to `gbr`.
  #[cfg(feature = "gbr")]
  rgba_plane_scratch_f32: Vec<f32>,
  /// Out-width G/B/R/A `half::f16` plane staging for the half-float planar
  /// GBR+alpha ([`Gbrapf16`](crate::source::Gbrapf16)) resample tail. There
  /// is no `AreaStream<f16>`, so that tail bins in `f32` (the shared
  /// [`Self::rgba_stream_f32`]) and, per finalized output row, resolves the
  /// straight binned RGBA, de-interleaves it into the `f32` planes, **rounds
  /// each element to `half::f16`** into these planes (`[0..ow]` = G,
  /// `[ow..2ow]` = B, `[2ow..3ow]` = R, `[3ow..4ow]` = A), then runs the
  /// exact direct `gbrapf16_*` / `gbrpf16_*` kernels — so every output is
  /// byte-identical to a direct `Gbrapf16` conversion of the `f32`
  /// block-mean rounded to f16. Lazily grown to `4 * out_width`
  /// `half::f16`; empty for a sink with no f16-plane-derived output. Gated
  /// to `gbr`.
  #[cfg(feature = "gbr")]
  rgba_plane_scratch_f16: Vec<half::f16>,
  /// Out-width **packed** `R, G, B` `half::f16` staging for the
  /// half-float packed-RGB ([`Rgbf16`](crate::source::Rgbf16)) resample
  /// tail. There is no `AreaStream<f16>`, so that tail bins in `f32` (the
  /// shared [`Self::rgb_stream_f32`]) and, per finalized output row,
  /// **rounds each binned packed `f32` element to `half::f16`** into this
  /// packed row, then runs the exact direct `rgbf16_*` kernels — so every
  /// output is byte-identical to a direct `Rgbf16` conversion of the `f32`
  /// block-mean rounded to f16. Unlike the planar
  /// [`Gbrpf16`](crate::source::Gbrpf16) tail this row stays **packed** (no
  /// de-interleave into planes), because the `rgbf16_*` kernels consume
  /// packed input. Lazily grown to `3 * out_width` `half::f16`; empty for a
  /// sink with no f16-derived output. Gated like [`Self::rgb_stream_f32`]:
  /// the `rgb-float` family fences in the engine.
  #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
  rgb_packed_scratch_f16: Vec<half::f16>,
  /// Out-width **packed** `R, G, B, A` `half::f16` staging for the
  /// half-float packed-RGBA ([`Rgbaf16`](crate::source::Rgbaf16)) resample
  /// tail — the 4-channel twin of [`Self::rgb_packed_scratch_f16`]. There
  /// is no `AreaStream<f16>`, so that tail bins in `f32` (the shared
  /// 4-channel [`Self::rgba_stream_f32`]) and, per finalized output row,
  /// **rounds each binned packed `f32` element to `half::f16`** into this
  /// packed `R, G, B, A` row, then runs the exact direct `rgbaf16_*`
  /// kernels — so every output is byte-identical to a direct `Rgbaf16`
  /// conversion of the `f32` block-mean rounded to f16. Lazily grown to
  /// `4 * out_width` `half::f16`; empty for a sink with no f16-derived
  /// output. Gated like [`Self::rgb_packed_scratch_f16`].
  #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
  rgba_packed_scratch_f16: Vec<half::f16>,
  /// Source-width **linear-XYZ** `f32` staging for the
  /// [`Xyz12`](crate::source::Xyz12) resample path: the wire row
  /// converts here (inverse-OETF only, no matrix) before feeding
  /// [`Self::xyz_stream_f32`]. Lazily grown to `3 * width` `f32`; empty
  /// otherwise. Gated to `xyz`.
  #[cfg(feature = "xyz")]
  xyz_scratch_f32: Vec<f32>,
  /// Whether row primitives dispatch to their SIMD backend. Defaults
  /// to `true`; benchmarks flip this with [`Self::with_simd`] /
  /// [`Self::set_simd`] to A/B test scalar vs SIMD on the same frame.
  simd: bool,
  /// How the source's packed alpha relates to its color when the frame
  /// is area-resampled. Seeded from `F::DEFAULT_ALPHA_MODE`
  /// ([`DefaultAlphaMode`]) at construction; overridden per call by
  /// [`Self::with_alpha_mode`] / [`Self::set_alpha_mode`], the
  /// [`Self::with_simd`] flag pattern. Only the packed-RGBA resample
  /// tail consults it; every direct (identity-plan) path and every
  /// non-RGBA source ignores it.
  alpha_mode: AlphaMode,
  /// The RFC #238 averaging domain an area downscale resamples in.
  /// Defaults to [`AveragingDomain::Encoded`] (the gamma-encoded /
  /// libswscale-class average — bit-identical to the pre-RFC behaviour).
  /// [`AveragingDomain::Linear`], set via [`Self::with_averaging_domain`],
  /// opts into the physically-correct linear-light average for the planar
  /// 8-bit YUV family (the only family wired in Phase 2); every other
  /// format ignores it and stays on the encoded path.
  #[cfg(feature = "yuv-planar")]
  averaging_domain: AveragingDomain,
  /// Optional caller override for the [`AveragingDomain::Linear`] transfer
  /// curve. `None` (the default) derives the curve from the sink's
  /// [`ColorMatrix`](crate::ColorMatrix) via [`TransferFunction::for_matrix`]; `Some(tf)` pins
  /// `tf` regardless of the matrix. Consulted only on the linear-light
  /// path; the encoded path never reads it. See
  /// [`Self::with_transfer_function`].
  #[cfg(feature = "yuv-planar")]
  transfer_function: Option<TransferFunction>,
  /// Chroma sample location driving siting-aware 4:2:0 chroma upsampling
  /// (#302). Defaults to [`ChromaLocation::Unspecified`](crate::ChromaLocation::Unspecified)
  /// — colconv's nearest-neighbor decode, byte-identical to the pre-#302
  /// behaviour. Set via [`Self::with_chroma_location`] /
  /// [`Self::set_chroma_location`]; the centered horizontal sitings
  /// (`Center` / `Top` / `Bottom`, per `chroma_420_center_sited_h`) route
  /// the identity-plan 4:2:0 decode through the phase-0.5 chroma upsample +
  /// 4:4:4 kernels. Consulted by the planar 8-bit `Yuv420p`, the semi-planar
  /// `Nv12` / `Nv21`, and the high-bit planar `Yuv420p9` … `Yuv420p16` identity
  /// paths; the remaining 4:2:0 families and the resampling tiers ignore it
  /// until the #302 rollout wires them in.
  #[cfg(feature = "yuv-planar")]
  chroma_location: crate::ChromaLocation,
  /// Source colour [`Primaries`](crate::Primaries) — **sink-consumed**, like
  /// [`chroma_location`](Self::chroma_location): mediaframe's YUV row carries
  /// only range + matrix, not the primaries, so the
  /// [`ColorMatrix::ChromaDerivedNcl`](crate::ColorMatrix::ChromaDerivedNcl)
  /// decode (whose `Kr` / `Kb` are *derived* from the primaries) reads them
  /// here. Defaults to [`Primaries::Unspecified`](crate::Primaries::Unspecified)
  /// (no chromaticities → the prior BT.709 fallback, byte-identical to the
  /// pre-#303 behaviour). Set via [`Self::with_color_spec`] /
  /// [`Self::set_color_spec`]. Currently consulted by the planar 8-bit
  /// `Yuv420p` identity RGB / RGBA path; other families and the resampling
  /// tiers keep the matrix-tag coefficients until wired in.
  #[cfg(feature = "yuv-planar")]
  primaries: crate::Primaries,
  /// Source [`Transfer`](crate::Transfer) characteristics — **sink-consumed**,
  /// like [`primaries`](Self::primaries): mediaframe's YUV row carries only
  /// range + matrix, not the transfer, so the
  /// [`ColorMatrix::Ictcp`](crate::ColorMatrix::Ictcp) non-affine decode
  /// (H.273 `MatrixCoefficients = 14`, BT.2100 ICtCp), which selects its
  /// inverse matrix + per-channel EOTF from the PQ/HLG transfer, reads it
  /// here. Defaults to [`Transfer::Unspecified`](crate::Transfer::Unspecified)
  /// (no PQ/HLG transfer → the affine matrix-tag fallback, byte-identical to
  /// the pre-#303 behaviour). Set via [`Self::with_color_spec`] /
  /// [`Self::set_color_spec`]. Currently consulted by the planar 12-bit
  /// `Yuv444p12` identity RGB / RGBA path (the representative #303 wiring);
  /// other families and the resampling tiers keep the affine path until
  /// wired in.
  #[cfg(feature = "yuv-planar")]
  transfer: crate::Transfer,
  /// Per-frame accumulator for the RFC #238 [`AveragingDomain::Linear`]
  /// linear-light tail (planar 8-bit YUV family). Lazily created on the
  /// first output-bearing row of a linear-domain frame; reset to `None` per
  /// `begin_frame`. Present only when the linear path is wired
  /// (`yuv-planar` + `rgb`).
  #[cfg(all(feature = "yuv-planar", feature = "rgb"))]
  linear_light_frame: Option<linear_light::LinearLightFrame>,
  /// How the [`AveragingDomain::Linear`] tail decodes `YUV→RGB` before the
  /// EOTF (RFC #238 #244). [`LinearMode::DisplayReferred`] (the default)
  /// decodes through the clamped Q15 kernel; [`LinearMode::SceneReferred`]
  /// decodes the same affine matrix in unclamped `f32`, preserving
  /// out-of-gamut excursions, and clamps only at the re-encoded output.
  /// Consulted only on the linear-light path; the encoded and direct paths
  /// ignore it. Set via [`Self::with_linear_mode`].
  #[cfg(feature = "yuv-planar")]
  linear_mode: LinearMode,
  /// Source-width unclamped normalized `f32` RGB staging for the
  /// scene-referred ([`LinearMode::SceneReferred`]) linear-light decode: the
  /// unclamped affine `YUV→RGB` decode writes here (`[0, 1]`-scale, may leave
  /// `[0, 1]`) before the tail lifts it through the EOTF. Lazily grown to
  /// `3 * width` `f32`; empty otherwise (the display-referred path uses the
  /// `u8` [`Self::rgb_scratch`] instead). Present only when the linear path
  /// is wired (`yuv-planar` + `rgb`).
  #[cfg(all(feature = "yuv-planar", feature = "rgb"))]
  linear_scene_scratch: Vec<f32>,
  /// Q8 fixed-point luma coefficients `(cr, cg, cb)` such that
  /// `luma = ((cr * R + cg * G + cb * B + 128) >> 8) as u8`. Only
  /// consulted by source impls that *derive* luma from RGB
  /// (currently the `Bayer` / `Bayer16<BITS>` family and the `Pal8`
  /// mono palette path — YUV impls memcpy from the native Y plane
  /// and ignore this field). Default: BT.709 `(54, 183, 19)`.
  #[cfg(any(feature = "bayer", feature = "mono"))]
  luma_coefficients_q8: (u32, u32, u32),
  _fmt: PhantomData<F>,
  _resampler: PhantomData<R>,
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

impl<F: SourceFormat> MixedSinker<'_, F, NoopResampler> {
  /// Creates an empty [`MixedSinker`] for the given dimensions, with
  /// the identity resampler (output geometry == source geometry).
  /// Attach output buffers with `with_rgb` / `with_luma` / `with_hsv`;
  /// each attachment validates that the buffer is at least
  /// `width * height * bytes_per_pixel` so short-buffer bugs surface
  /// *before* any rows are written — not after half the frame has
  /// been mutated. For a sinker whose outputs land at a smaller
  /// geometry, construct via [`MixedSinker::with_resampler`] instead.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(width: usize, height: usize) -> Self {
    Self::with_geometry(width, height, width, height)
  }
}

impl<F: SourceFormat, R> MixedSinker<'_, F, R> {
  /// Field initializer shared by [`MixedSinker::new`] and
  /// [`MixedSinker::with_resampler`]: source geometry plus the output
  /// geometry that buffer validation sizes against.
  fn with_geometry(width: usize, height: usize, out_width: usize, out_height: usize) -> Self {
    Self {
      rgb: None,
      rgb_u16: None,
      rgb_f32: None,
      rgb_f16: None,
      rgba: None,
      rgba_u16: None,
      rgba_f32: None,
      rgba_f16: None,
      luma: None,
      luma_u16: None,
      luma_f32: None,
      #[cfg(any(
        feature = "bayer",
        feature = "gbr",
        feature = "gray",
        feature = "mono",
        feature = "rgb",
        feature = "rgb-float",
        feature = "rgb-legacy",
        feature = "v210",
        feature = "xyz",
        feature = "y2xx",
        feature = "yuv-444-packed",
        feature = "yuv-packed",
        feature = "yuv-planar",
        feature = "yuv-semi-planar",
        feature = "yuva",
      ))]
      hsv: None,
      #[cfg(feature = "xyz")]
      xyz_f32: None,
      width,
      height,
      out_width,
      out_height,
      plan: None,
      #[cfg(any(
        feature = "yuv-planar",
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "xyz",
        feature = "bayer",
        feature = "mono",
        feature = "yuv-semi-planar",
        feature = "yuv-packed",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210",
        feature = "rgb-legacy"
      ))]
      rgb_stream: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210",
        feature = "yuv-planar",
        feature = "yuv-semi-planar"
      ))]
      rgb_stream_u16: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "mono",
        feature = "yuva",
        feature = "yuv-444-packed"
      ))]
      rgba_stream: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "yuva",
        feature = "yuv-444-packed"
      ))]
      rgba_stream_u16: None,
      #[cfg(any(feature = "rgb", feature = "gbr"))]
      rgb_stream_u32: None,
      #[cfg(any(feature = "rgb", feature = "gbr"))]
      rgb_filter_stream_u32: None,
      #[cfg(any(feature = "rgb", feature = "gbr"))]
      rgba_stream_u32: None,
      #[cfg(any(feature = "rgb", feature = "gbr"))]
      rgba_filter_stream_u32: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "mono",
        feature = "yuva",
        feature = "yuv-444-packed"
      ))]
      frozen_alpha_mode: None,
      #[cfg(any(
        all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
        feature = "gbr"
      ))]
      rgb_stream_f32: None,
      #[cfg(any(
        all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
        feature = "gbr"
      ))]
      rgb_filter_stream_f32: None,
      #[cfg(any(
        feature = "gbr",
        all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
      ))]
      rgba_stream_f32: None,
      #[cfg(any(
        feature = "gbr",
        all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
      ))]
      rgba_filter_stream_f32: None,
      #[cfg(feature = "xyz")]
      xyz_stream_f32: None,
      #[cfg(feature = "xyz")]
      xyz_filter_stream_f32: None,
      #[cfg(any(
        feature = "yuv-planar",
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "xyz",
        feature = "bayer",
        feature = "mono",
        feature = "yuv-semi-planar",
        feature = "yuv-packed",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210",
        feature = "rgb-legacy"
      ))]
      luma_stream: None,
      #[cfg(any(
        feature = "gray",
        feature = "yuva",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210",
        feature = "yuv-planar",
        feature = "yuv-semi-planar"
      ))]
      luma_stream_u16: None,
      #[cfg(feature = "gray")]
      luma_stream_u32: None,
      #[cfg(feature = "gray")]
      luma_stream_f32: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "yuv-planar",
        feature = "yuv-semi-planar",
        feature = "yuv-packed"
      ))]
      rgb_filter_stream: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "yuv-444-packed",
        feature = "mono",
        feature = "yuva",
        feature = "gray"
      ))]
      rgba_filter_stream: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "yuv-planar",
        feature = "yuv-semi-planar"
      ))]
      rgb_filter_stream_u16: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "yuv-444-packed",
        feature = "yuva",
        feature = "gray"
      ))]
      rgba_filter_stream_u16: None,
      #[cfg(any(
        feature = "yuv-planar",
        feature = "yuv-semi-planar",
        feature = "yuv-packed",
        feature = "mono",
        feature = "gray"
      ))]
      luma_filter_stream: None,
      #[cfg(feature = "gray")]
      luma_filter_stream_f32: None,
      #[cfg(any(
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "yuv-planar",
        feature = "gray",
        feature = "yuva",
        feature = "yuv-semi-planar"
      ))]
      luma_filter_stream_u16: None,
      #[cfg(feature = "gray")]
      luma_filter_stream_u32: None,
      #[cfg(any(
        feature = "yuv-planar",
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "xyz",
        feature = "bayer",
        feature = "mono",
        feature = "yuv-semi-planar",
        feature = "yuv-packed",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210",
        feature = "rgb-legacy"
      ))]
      resample_outputs: None,
      #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
      native: true,
      #[cfg(feature = "yuv-planar")]
      native_420: None,
      #[cfg(feature = "yuva")]
      native_yuva_420: None,
      #[cfg(feature = "yuv-planar")]
      bicublin_420: None,
      #[cfg(feature = "yuv-planar")]
      native_planar: None,
      #[cfg(feature = "yuv-planar")]
      hsv_planar: None,
      #[cfg(feature = "yuv-planar")]
      native_420_u16: None,
      #[cfg(feature = "yuv-planar")]
      native_planar_u16: None,
      #[cfg(feature = "yuv-planar")]
      yuv444_msb_y_depack: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      yuv444_msb_u_depack: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      yuv444_msb_v_depack: Vec::new(),
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      semi_planar_u_half: Vec::new(),
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      semi_planar_v_half: Vec::new(),
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_y_half: Vec::new(),
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_u_half: Vec::new(),
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_v_half: Vec::new(),
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_y_full: Vec::new(),
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_u_half: Vec::new(),
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_v_half: Vec::new(),
      #[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
      y2xx_y_full: Vec::new(),
      #[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
      y2xx_u_half: Vec::new(),
      #[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
      y2xx_v_half: Vec::new(),
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_y_full: Vec::new(),
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_u_half: Vec::new(),
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_v_half: Vec::new(),
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_y_full: Vec::new(),
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_u_full: Vec::new(),
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_v_full: Vec::new(),
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_y_full_u16: Vec::new(),
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_u_full_u16: Vec::new(),
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_v_full_u16: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      frozen_native_route: None,
      #[cfg(feature = "yuv-planar")]
      frozen_domain: None,
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_centered: None,
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_bottom_v: None,
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_top_v: None,
      #[cfg(feature = "yuv-planar")]
      chroma_top_pending: None,
      #[cfg(feature = "yuv-planar")]
      chroma_top_y: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      chroma_top_a: Vec::new(),
      #[cfg(any(
        feature = "bayer",
        feature = "gbr",
        feature = "gray",
        feature = "mono",
        feature = "rgb",
        feature = "rgb-float",
        feature = "rgb-legacy",
        feature = "v210",
        feature = "xyz",
        feature = "y2xx",
        feature = "yuv-444-packed",
        feature = "yuv-packed",
        feature = "yuv-planar",
        feature = "yuv-semi-planar",
        feature = "yuva",
      ))]
      rgb_scratch: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      chroma_full: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      chroma_prev: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      chroma_prev_row: None,
      #[cfg(feature = "yuv-planar")]
      chroma_full_u16: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      chroma_prev_u16: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      chroma_top_y_u16: Vec::new(),
      #[cfg(feature = "yuv-planar")]
      chroma_top_a_u16: Vec::new(),
      #[cfg(any(feature = "yuv-packed", feature = "gray"))]
      luma_scratch: Vec::new(),
      #[cfg(feature = "rgb-legacy")]
      legacy_rgb_native_scratch: Vec::new(),
      #[cfg(feature = "rgb-legacy")]
      legacy_rgb_packed_scratch: Vec::new(),
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210",
        feature = "yuv-planar",
        feature = "yuv-semi-planar"
      ))]
      rgb_scratch_u16: Vec::new(),
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "mono",
        feature = "yuva",
        feature = "yuv-444-packed"
      ))]
      rgba_scratch: Vec::new(),
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "yuva",
        feature = "yuv-444-packed"
      ))]
      rgba_scratch_u16: Vec::new(),
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "yuva",
        feature = "yuv-444-packed"
      ))]
      rgba_color_scratch_u16: Vec::new(),
      #[cfg(any(feature = "rgb", feature = "gbr"))]
      rgb_scratch_u32: Vec::new(),
      #[cfg(any(feature = "rgb", feature = "gbr"))]
      rgba_scratch_u32: Vec::new(),
      #[cfg(any(feature = "rgb", feature = "gbr"))]
      rgba_color_scratch_u32: Vec::new(),
      #[cfg(any(
        feature = "gray",
        feature = "yuva",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210",
        feature = "yuv-planar",
        feature = "yuv-semi-planar"
      ))]
      luma_scratch_u16: Vec::new(),
      #[cfg(feature = "gray")]
      luma_scratch_u32: Vec::new(),
      #[cfg(feature = "gray")]
      luma_scratch_f32: Vec::new(),
      #[cfg(feature = "gray")]
      yaf_stream_f32: None,
      #[cfg(feature = "gray")]
      yaf_filter_stream_f32: None,
      #[cfg(feature = "gray")]
      yaf_stage_scratch_f32: Vec::new(),
      #[cfg(feature = "gray")]
      yaf_plane_scratch_f32: Vec::new(),
      #[cfg(any(
        all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
        feature = "gbr"
      ))]
      rgb_scratch_f32: Vec::new(),
      #[cfg(feature = "gbr")]
      rgb_plane_scratch_f32: Vec::new(),
      #[cfg(feature = "gbr")]
      rgb_plane_scratch_f16: Vec::new(),
      #[cfg(any(
        feature = "gbr",
        all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
      ))]
      rgba_scratch_f32: Vec::new(),
      #[cfg(feature = "gbr")]
      rgba_color_scratch_f32: Vec::new(),
      #[cfg(feature = "gbr")]
      rgba_plane_scratch_f32: Vec::new(),
      #[cfg(feature = "gbr")]
      rgba_plane_scratch_f16: Vec::new(),
      #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
      rgb_packed_scratch_f16: Vec::new(),
      #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
      rgba_packed_scratch_f16: Vec::new(),
      #[cfg(feature = "xyz")]
      xyz_scratch_f32: Vec::new(),
      simd: true,
      alpha_mode: F::DEFAULT_ALPHA_MODE,
      #[cfg(feature = "yuv-planar")]
      averaging_domain: AveragingDomain::Encoded,
      #[cfg(feature = "yuv-planar")]
      transfer_function: None,
      #[cfg(feature = "yuv-planar")]
      chroma_location: crate::ChromaLocation::Unspecified,
      #[cfg(feature = "yuv-planar")]
      primaries: crate::Primaries::Unspecified,
      #[cfg(feature = "yuv-planar")]
      transfer: crate::Transfer::Unspecified,
      #[cfg(feature = "yuv-planar")]
      linear_mode: LinearMode::DisplayReferred,
      #[cfg(all(feature = "yuv-planar", feature = "rgb"))]
      linear_light_frame: None,
      #[cfg(all(feature = "yuv-planar", feature = "rgb"))]
      linear_scene_scratch: Vec::new(),
      // BT.709 by default — matches the implicit weights every
      // YUV→RGB→luma pipeline uses, and is the most common Bayer
      // CCM target. Per-format impls (`MixedSinker<Bayer>` etc.)
      // expose `with_luma_coefficients` for callers whose CCM
      // targets a different gamut.
      #[cfg(any(feature = "bayer", feature = "mono"))]
      luma_coefficients_q8: (54, 183, 19),
      _fmt: PhantomData,
      _resampler: PhantomData,
    }
  }

  /// Returns `true` iff the sinker will write 8‑bit RGB.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb(&self) -> bool {
    self.rgb.is_some()
  }

  /// Returns `true` iff the sinker will write `u16` RGB at the
  /// source's native bit depth. Only high‑bit‑depth source impls
  /// (currently [`Yuv420p10`](crate::source::Yuv420p10)) honor this
  /// buffer — attaching it on an 8‑bit source format is legal but
  /// no writes occur.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb_u16(&self) -> bool {
    self.rgb_u16.is_some()
  }

  /// Returns `true` iff the sinker will write `f32` RGB. Only
  /// float-source impls (currently [`Rgbf32`](crate::source::Rgbf32))
  /// honor this buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb_f32(&self) -> bool {
    self.rgb_f32.is_some()
  }

  /// Returns `true` iff the sinker will write `half::f16` RGB. Only
  /// half-float-source impls (currently [`Rgbf16`](crate::source::Rgbf16))
  /// honor this buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb_f16(&self) -> bool {
    self.rgb_f16.is_some()
  }

  /// Returns `true` iff the sinker will write `f32` RGBA. Only
  /// float-planar-GBR source impls (`Gbrpf32` / `Gbrapf32` / `Gbrpf16` /
  /// `Gbrapf16`) honor this buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgba_f32(&self) -> bool {
    self.rgba_f32.is_some()
  }

  /// Returns `true` iff the sinker will write `half::f16` RGBA. Only
  /// float-planar-GBR source impls (`Gbrpf32` / `Gbrapf32` / `Gbrpf16` /
  /// `Gbrapf16`) honor this buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgba_f16(&self) -> bool {
    self.rgba_f16.is_some()
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

  /// Returns `true` iff the sinker will write native-depth `u16`
  /// luma. Only honored by per-format impls that wire the
  /// `with_luma_u16` accessor (Tier 4 source families).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_luma_u16(&self) -> bool {
    self.luma_u16.is_some()
  }

  /// Returns `true` iff the sinker will write `f32` luma.
  /// Only honored by `Grayf32` source impls.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_luma_f32(&self) -> bool {
    self.luma_f32.is_some()
  }

  /// Returns `true` iff the sinker will write HSV.
  ///
  /// Gated on the same 15-feature any as the `hsv` field — under
  /// `--features "alloc"` alone, no per-format `process` impl
  /// compiles, the field doesn't exist, and this getter is also gone.
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
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

  /// Output width in pixels — what output buffers validate against.
  /// Equals [`Self::width`] unless constructed via
  /// [`MixedSinker::with_resampler`] with a non-identity plan.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_width(&self) -> usize {
    self.out_width
  }

  /// Output height in pixels — what output buffers validate against.
  /// Equals [`Self::height`] unless constructed via
  /// [`MixedSinker::with_resampler`] with a non-identity plan.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_height(&self) -> usize {
    self.out_height
  }

  /// The resampling plan fixed at construction — `Some` only for
  /// sinkers built via [`MixedSinker::with_resampler`] with a
  /// non-identity strategy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn resample_plan(&self) -> Option<&ResamplePlan> {
    self.plan.as_ref()
  }

  /// Capacity of the source-row staging scratch — a white-box probe
  /// for the resample ordering tests (a rejected row must not have
  /// grown the scratch). Gated on `std` like the tests that consume it,
  /// so it is not dead code in the alloc-only test build.
  // The `gbr` family routes its resample ordering through the dedicated
  // 4-channel RGBA tail and the high-bit u16 tail (probed via the
  // dedicated `*_scratch` capacities), and cannot construct an
  // out-of-sequence row to grow this u8 scratch (`GbrapRow` / `GbrpRow`
  // are `pub(crate)` in mediaframe), so no `gbr` test consumes this probe
  // — keep it out of the gate to avoid a `gbr`-solo dead-code warning.
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "rgb",
      feature = "xyz",
      feature = "bayer",
      feature = "gray",
      feature = "yuv-packed",
      feature = "yuv-444-packed",
      feature = "yuva",
      feature = "rgb-legacy",
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
    )
  ))]
  pub(crate) fn rgb_scratch_capacity(&self) -> usize {
    self.rgb_scratch.capacity()
  }

  /// Capacity of the legacy 16-bit packed-RGB source-row native-channel
  /// staging scratch — a white-box probe for the resample ordering tests
  /// (a rejected row must not have grown the scratch). Gated on
  /// `rgb-legacy` + `std` like the tests that consume it.
  #[cfg(all(test, feature = "rgb-legacy", feature = "std"))]
  pub(crate) fn legacy_rgb_native_scratch_capacity(&self) -> usize {
    self.legacy_rgb_native_scratch.capacity()
  }

  /// Capacity of the legacy 16-bit packed-RGB re-packed-source-row
  /// staging scratch — a white-box probe for the resample tests (a
  /// native-`u16`-only sink, which copies the binned row at native depth,
  /// must still size it because the re-pack feeds the `rgb_u16` kernel;
  /// a no-output sink must not). Gated on `rgb-legacy` + `std`.
  #[cfg(all(test, feature = "rgb-legacy", feature = "std"))]
  pub(crate) fn legacy_rgb_packed_scratch_capacity(&self) -> usize {
    self.legacy_rgb_packed_scratch.capacity()
  }

  /// Whether the high-bit packed-RGB `u16` area stream has been
  /// created — a white-box probe for the resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `std` + the families that bin into the u16
  /// tail (`rgb` high-bit packed RGB, `yuv-444-packed` high-bit packed
  /// 4:4:4 YUV color group, `y2xx` / `v210` high-bit packed 4:2:2 YUV).
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "rgb",
      feature = "yuv-444-packed",
      feature = "y2xx",
      feature = "v210",
      feature = "yuv-planar",
      feature = "yuv-semi-planar"
    )
  ))]
  pub(crate) fn rgb_stream_u16_allocated(&self) -> bool {
    self.rgb_stream_u16.is_some()
  }

  /// Whether the **4-channel** high-bit packed-RGBA `u16` area stream has been
  /// created — the alpha-aware twin of [`Self::rgb_stream_u16_allocated`], a
  /// white-box probe for the RFC #238 tail-collapse packed-RGBA `u16` / `Gbrap`
  /// resample atomicity tests (a first-row allocation failure must leave the
  /// field `None`, never a partial commit). Gated on `std` + the families that
  /// bin through the shared 4-channel `rgba_stream_u16`.
  #[cfg(all(
    test,
    feature = "std",
    any(feature = "rgb", feature = "gbr", feature = "gray")
  ))]
  pub(crate) fn rgba_stream_u16_allocated(&self) -> bool {
    self.rgba_stream_u16.is_some()
  }

  /// Whether the **3-channel** native-`u32` packed-RGB area stream has been
  /// created — the `u32` twin of [`Self::rgb_stream_u16_allocated`], a white-box
  /// probe for the RFC #238 tail-collapse native-`u32` packed-RGB / GBR resample
  /// atomicity tests (a first-row allocation failure must leave the field `None`,
  /// never a partial commit). Gated on `std` + the `rgb` / `gbr` families that
  /// bin through the shared 3-channel `rgb_stream_u32`.
  #[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
  pub(crate) fn rgb_stream_u32_allocated(&self) -> bool {
    self.rgb_stream_u32.is_some()
  }

  /// Whether the **4-channel** native-`u32` packed-RGBA area stream has been
  /// created — the alpha-aware `u32` twin of [`Self::rgb_stream_u32_allocated`], a
  /// white-box probe for the RFC #238 tail-collapse native-`u32` packed-RGBA /
  /// `Gbrap32` resample atomicity tests (a first-row allocation failure must leave
  /// the field `None`, never a partial commit). Gated on `std` + the `rgb` / `gbr`
  /// families that bin through the shared 4-channel `rgba_stream_u32`.
  #[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
  pub(crate) fn rgba_stream_u32_allocated(&self) -> bool {
    self.rgba_stream_u32.is_some()
  }

  /// Whether the packed-float-RGB `f32` area stream has been created —
  /// a white-box probe for the resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `std` and the families that drive the float
  /// tail: the `rgb-float` family (fenced to the engine) or `gbr` (which
  /// already carries the engine).
  #[cfg(all(
    test,
    feature = "std",
    any(
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
      feature = "gbr"
    )
  ))]
  pub(crate) fn rgb_stream_f32_allocated(&self) -> bool {
    self.rgb_stream_f32.is_some()
  }

  /// Whether the **4-channel** packed-RGBA / planar-GBR+alpha `f32` area stream
  /// has been created — a white-box probe for the resample tests (a no-output sink
  /// must not allocate it; a first-row allocation failure must leave it `None`, the
  /// RFC #238 tail-collapse atomicity proof). Gated on `std` + the packed-float-RGBA
  /// (`rgb-float`, engine fenced in) and planar-GBR+alpha (`gbr`) families that bin
  /// through the shared 4-channel `rgba_stream_f32`.
  #[cfg(all(
    test,
    feature = "std",
    any(
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
      feature = "gbr"
    )
  ))]
  pub(crate) fn rgba_stream_f32_allocated(&self) -> bool {
    self.rgba_stream_f32.is_some()
  }

  /// Whether the **3-channel** packed-`f32` **filter** stream has been
  /// created — a white-box probe for the float planar GBR filter resample
  /// tests (a no-output sink must not allocate it). Gated on `gbr` + `std`
  /// like the tests that consume it (the `rgb-float` consumers have their own
  /// coverage; the `gbr` arm widens the gate here so the probe is available
  /// in a `gbr`-only build).
  #[cfg(all(test, feature = "gbr", feature = "std"))]
  pub(crate) fn rgb_filter_stream_f32_allocated(&self) -> bool {
    self.rgb_filter_stream_f32.is_some()
  }

  /// Whether the **4-channel** float planar GBR+alpha `f32` **filter** stream
  /// has been created — a white-box probe for the filter resample tests (a
  /// no-output sink must not allocate it). Gated on `gbr` + `std` like the
  /// tests that consume it.
  #[cfg(all(test, feature = "gbr", feature = "std"))]
  pub(crate) fn rgba_filter_stream_f32_allocated(&self) -> bool {
    self.rgba_filter_stream_f32.is_some()
  }

  /// Capacity of the float planar-GBR G/B/R plane scratch — a white-box
  /// probe for the resample tests (an `rgb_f32`-only sink must not grow
  /// it). Gated on `gbr` + `std` like the test that consumes it.
  #[cfg(all(test, feature = "gbr", feature = "std"))]
  pub(crate) fn rgb_plane_scratch_capacity(&self) -> usize {
    self.rgb_plane_scratch_f32.capacity()
  }

  /// Capacity of the half-float planar-GBR ([`Gbrpf16`](crate::source::Gbrpf16))
  /// G/B/R f16 plane scratch — a white-box probe for the resample tests (a
  /// no-output sink must not grow it). Gated on `gbr` + `std` like the test
  /// that consumes it.
  #[cfg(all(test, feature = "gbr", feature = "std"))]
  pub(crate) fn rgb_plane_scratch_f16_capacity(&self) -> usize {
    self.rgb_plane_scratch_f16.capacity()
  }

  /// Capacity of the half-float packed-RGB ([`Rgbf16`](crate::source::Rgbf16))
  /// packed f16 scratch row — a white-box probe for the resample tests (a
  /// no-output sink must not grow it). Gated on the `rgb-float` engine fence
  /// + `std` like the test that consumes it.
  #[cfg(all(
    test,
    feature = "std",
    feature = "rgb-float",
    any(feature = "yuv-planar", feature = "rgb")
  ))]
  pub(crate) fn rgb_packed_scratch_f16_capacity(&self) -> usize {
    self.rgb_packed_scratch_f16.capacity()
  }

  /// Whether the [`Xyz12`](crate::source::Xyz12) linear-XYZ `f32` area
  /// stream has been created — a white-box probe for the resample
  /// ordering tests (an out-of-sequence first row must be rejected
  /// before the stream is allocated). Gated on `xyz` and `std` like the
  /// tests that consume it.
  #[cfg(all(test, feature = "xyz", feature = "std"))]
  pub(crate) fn xyz_stream_f32_allocated(&self) -> bool {
    self.xyz_stream_f32.is_some()
  }

  /// Capacity of the [`Xyz12`](crate::source::Xyz12) source-row
  /// linear-XYZ staging scratch — a white-box probe for the resample
  /// ordering tests (a rejected row must not have grown the scratch).
  /// Gated on `xyz` and `std` like the tests that consume it.
  #[cfg(all(test, feature = "xyz", feature = "std"))]
  pub(crate) fn xyz_scratch_f32_capacity(&self) -> usize {
    self.xyz_scratch_f32.capacity()
  }

  /// Whether the per-frame output-set snapshot has been committed (frozen)
  /// — a white-box probe for the RFC #238 tail-collapse atomicity tests: a
  /// first-row scratch OOM must leave `resample_outputs` UNCOMMITTED so the
  /// freeze half of the commit-together shape is proven directly, without
  /// relying on a walker retry (which would `begin_frame`-reset the field
  /// before it could be observed). Gated to cover the `xyz` + `mono` (Pal8) +
  /// `rgb-legacy` + `yuv-planar` (the 4:2:0 high-bit native rebuild-atomicity
  /// suite) test-module gates that consume it.
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "xyz",
      feature = "mono",
      feature = "rgb-legacy",
      feature = "yuv-planar"
    )
  ))]
  pub(crate) fn resample_outputs_frozen(&self) -> bool {
    self.resample_outputs.is_some()
  }

  /// Whether the single-channel luma `u8` area stream has been created
  /// — a white-box probe for the [`Gray8`](crate::source::Gray8),
  /// `mono`, and packed-YUV-4:2:2 resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `gray`/`mono`/`yuv-packed` and `std` like the
  /// tests that consume it.
  #[cfg(all(
    test,
    feature = "std",
    any(feature = "gray", feature = "mono", feature = "yuv-packed")
  ))]
  pub(crate) fn luma_stream_allocated(&self) -> bool {
    self.luma_stream.is_some()
  }

  /// Whether the 3-channel packed-RGB `u8` area stream has been created
  /// — a white-box probe for the packed-YUV-4:2:2, high-bit packed 4:4:4
  /// / 4:2:2 YUV, legacy packed-RGB, and RFC #238 tail-collapse
  /// (packed-RGB / GBR / Bayer) resample ordering tests (an out-of-sequence
  /// first row must be rejected — and a first-row allocation failure must
  /// leave the field `None` — before the stream is committed). Gated on
  /// `std` + the families that bin their converted u8 RGB through the shared
  /// `rgb_stream`.
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "yuv-packed",
      feature = "yuv-444-packed",
      feature = "y2xx",
      feature = "v210",
      feature = "rgb-legacy",
      feature = "yuv-planar",
      feature = "yuv-semi-planar",
      feature = "rgb",
      feature = "gbr",
      feature = "bayer"
    )
  ))]
  pub(crate) fn rgb_stream_allocated(&self) -> bool {
    self.rgb_stream.is_some()
  }

  /// Whether the 4-channel packed-RGBA `u8` area stream has been created —
  /// the alpha-aware twin of [`Self::rgb_stream_allocated`], a white-box
  /// probe for the RFC #238 tail-collapse packed-RGBA / `Gbrap` / `Pal8`
  /// resample atomicity tests (a first-row allocation failure must leave the
  /// field `None`, never a partial commit). Gated on `std` + the families that
  /// bin through the shared 4-channel `rgba_stream`.
  #[cfg(all(
    test,
    feature = "std",
    any(feature = "rgb", feature = "gbr", feature = "gray", feature = "mono")
  ))]
  pub(crate) fn rgba_stream_allocated(&self) -> bool {
    self.rgba_stream.is_some()
  }

  /// Whether the 3-channel packed-RGB `u8` **filter** stream has been
  /// created — a white-box probe for the 8-bit packed-YUV filter resample
  /// ordering tests (a no-output sink and an out-of-sequence first row must
  /// not allocate it). Gated on `std` + `yuv-packed` like the tests that
  /// consume it.
  #[cfg(all(test, feature = "std", feature = "yuv-packed"))]
  pub(crate) fn rgb_filter_stream_allocated(&self) -> bool {
    self.rgb_filter_stream.is_some()
  }

  /// Whether the 4-channel packed-RGBA `u8` **filter** stream has been
  /// created — a white-box probe for the `Pal8` filter resample ordering
  /// tests (a no-output sink must not allocate it). Gated on `std` + `mono`
  /// like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "mono"))]
  pub(crate) fn rgba_filter_stream_allocated(&self) -> bool {
    self.rgba_filter_stream.is_some()
  }

  /// Whether the single-channel native-Y `u8` **filter** stream has been
  /// created — the filter twin of [`Self::luma_stream_allocated`], a
  /// white-box probe for the 8-bit packed-YUV and `mono` filter resample
  /// ordering tests. Gated on `std` + `mono`/`yuv-packed` like the tests
  /// that consume it.
  #[cfg(all(test, feature = "std", any(feature = "mono", feature = "yuv-packed")))]
  pub(crate) fn luma_filter_stream_allocated(&self) -> bool {
    self.luma_filter_stream.is_some()
  }

  /// Capacity of the packed-YUV-4:2:2 source-row Y de-interleave staging
  /// scratch — a white-box probe for the resample ordering tests (a
  /// rejected row must not have grown the scratch). Gated on
  /// `yuv-packed` and `std` like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "yuv-packed"))]
  pub(crate) fn luma_scratch_capacity(&self) -> usize {
    self.luma_scratch.capacity()
  }

  /// Capacity of the source-row `u16` RGB staging scratch — a white-box
  /// probe for the high-bit packed 4:4:4 YUV resample ordering tests (a
  /// rejected row must not have grown the scratch). Gated on
  /// `yuv-444-packed` + `std` like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "yuv-444-packed"))]
  pub(crate) fn rgb_scratch_u16_capacity(&self) -> usize {
    self.rgb_scratch_u16.capacity()
  }

  /// Capacity of the source-row `u16` luma (native Y) staging scratch —
  /// a white-box probe for the high-bit packed 4:4:4 YUV resample
  /// ordering tests (a rejected row must not have grown the scratch).
  /// Gated on `yuv-444-packed` + `std` like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "yuv-444-packed"))]
  pub(crate) fn luma_scratch_u16_capacity(&self) -> usize {
    self.luma_scratch_u16.capacity()
  }

  /// Whether the single-channel **u16** luma area stream has been
  /// created — a white-box probe for the
  /// [`Gray16`](crate::source::Gray16) and high-bit packed 4:4:4 YUV
  /// (`yuv-444-packed`) / high-bit packed 4:2:2 YUV (`y2xx` / `v210`) /
  /// high-bit planar YUV (`yuv-planar`) resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `gray` / `yuv-444-packed` / `y2xx` / `v210` /
  /// `yuv-planar` / `yuv-semi-planar` and `std` like the tests that
  /// consume it.
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "gray",
      feature = "yuv-444-packed",
      feature = "y2xx",
      feature = "v210",
      feature = "yuv-planar",
      feature = "yuv-semi-planar"
    )
  ))]
  pub(crate) fn luma_stream_u16_allocated(&self) -> bool {
    self.luma_stream_u16.is_some()
  }

  /// Whether the single-channel **f32** luma area stream has been
  /// created — a white-box probe for the
  /// [`Grayf32`](crate::source::Grayf32) resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `gray` and `std` like the tests that consume
  /// it.
  #[cfg(all(test, feature = "std", feature = "gray"))]
  pub(crate) fn luma_stream_f32_allocated(&self) -> bool {
    self.luma_stream_f32.is_some()
  }

  /// Capacity of the [`Grayf32`](crate::source::Grayf32) source-row
  /// host-native `f32` luma staging scratch — a white-box probe for the
  /// resample tests (a rejected row must not have grown the scratch).
  /// Gated on `gray` and `std` like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "gray"))]
  pub(crate) fn luma_scratch_f32_capacity(&self) -> usize {
    self.luma_scratch_f32.capacity()
  }

  /// Returns `true` iff row primitives dispatch to their SIMD backend.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn simd(&self) -> bool {
    self.simd
  }

  /// Returns `true` iff resampled processing may take the native
  /// decimation tier. See [`Self::with_native`].
  #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
  #[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn native(&self) -> bool {
    self.native
  }

  /// Toggles the native decimation tier in place. See
  /// [`Self::with_native`] for the consuming builder variant.
  #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_native(&mut self, native: bool) -> &mut Self {
    self.native = native;
    self
  }

  /// Sets whether resampled processing may take the native decimation
  /// tier (bin native planes, convert once at output resolution).
  /// Defaults to `true`, mirroring [`Self::with_simd`].
  ///
  /// The tiers differ in color SEMANTICS, not just speed: native
  /// averages in the source (YUV) domain and converts once — the
  /// fused semantics video pipelines (libswscale-class) produce —
  /// while the row-stage tier converts every source pixel first and
  /// averages in RGB, matching `cv2.INTER_AREA` applied to decoded
  /// RGB. Luma is bit-identical either way (both tiers bin the same Y
  /// plane). In-gamut color differs only by per-pixel rounding;
  /// OUT-OF-GAMUT content (super-blacks/whites, illegal chroma
  /// excursions) diverges as far as the content sits outside the
  /// gamut — unbounded in principle, with measured examples of
  /// 34/255 on a mild extreme checkerboard and 117/255 on a crafted
  /// Bt2020 limited-range case (both pinned by regression). Pass
  /// `false` for strict RGB-domain `INTER_AREA` semantics at
  /// source-resolution conversion cost.
  ///
  /// Under a `yuv-semi-planar`-solo build (no `yuv-planar`) the native
  /// tier is not compiled, so the high-bit P-format sinks always use the
  /// row-stage tail; this flag is then accepted but inert.
  #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_native(mut self, native: bool) -> Self {
    self.set_native(native);
    self
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

  /// Returns how the source's packed alpha is interpreted when the
  /// frame is area-resampled. See [`Self::with_alpha_mode`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn alpha_mode(&self) -> AlphaMode {
    self.alpha_mode
  }

  /// Sets the alpha interpretation in place. See
  /// [`Self::with_alpha_mode`] for the consuming builder variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_alpha_mode(&mut self, mode: AlphaMode) -> &mut Self {
    self.alpha_mode = mode;
    self
  }

  /// Sets how the source's packed alpha relates to its color channels
  /// when the frame is **area-resampled**, overriding the per-format
  /// default ([`DefaultAlphaMode`], [`AlphaMode::Straight`] for every
  /// packed-RGBA source colconv ships). Mirrors the [`Self::with_simd`]
  /// builder pattern.
  ///
  /// [`AlphaMode::Premultiplied`] makes the packed-RGBA resample tail
  /// bin premultiplied color and un-premultiply per finalized output
  /// row, so fully-transparent pixels never bleed their stored color
  /// into a downscaled result. The mode is a no-op for the direct
  /// (identity-plan) conversions — which copy alpha through untouched —
  /// and for every non-RGBA source. See [`Self::set_alpha_mode`] for
  /// the in-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_alpha_mode(mut self, mode: AlphaMode) -> Self {
    self.set_alpha_mode(mode);
    self
  }

  /// Returns the RFC #238 [averaging domain](AveragingDomain) an area
  /// downscale resamples in. See [`Self::with_averaging_domain`].
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn averaging_domain(&self) -> AveragingDomain {
    self.averaging_domain
  }

  /// Sets the averaging domain in place. See
  /// [`Self::with_averaging_domain`] for the consuming builder variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_averaging_domain(&mut self, domain: AveragingDomain) -> &mut Self {
    self.averaging_domain = domain;
    self
  }

  /// Selects the colour domain an area downscale averages in, overriding
  /// the default [`AveragingDomain::Encoded`]. Mirrors the
  /// [`Self::with_simd`] builder pattern.
  ///
  /// [`AveragingDomain::Linear`] opts the **planar 8-bit YUV** family
  /// (`Yuv420p` / `Yuv422p` / `Yuv444p` / `Yuv440p`) into a
  /// physically-correct linear-light downscale: each source pixel is
  /// decoded to RGB, lifted to linear light by the
  /// [`TransferFunction`] EOTF, box-averaged, and re-encoded by the OETF.
  /// Because the YUV→RGB convert is affine (no transfer), this lands at a
  /// materially different RGB than the default encoded average — the
  /// quality-vs-default trade the domain offers. The transfer curve is
  /// resolved from the sink's [`ColorMatrix`](crate::ColorMatrix) unless overridden with
  /// [`Self::with_transfer_function`].
  ///
  /// The domain is a no-op for every other format and for the direct
  /// (identity-plan) conversions; [`AveragingDomain::Encoded`] keeps the
  /// bit-identical pre-RFC behaviour. See [`Self::set_averaging_domain`]
  /// for the in-place variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_averaging_domain(mut self, domain: AveragingDomain) -> Self {
    self.set_averaging_domain(domain);
    self
  }

  /// Returns the caller's [`TransferFunction`] override, or `None` when the
  /// linear-light transfer is derived from the [`ColorMatrix`](crate::ColorMatrix). See
  /// [`Self::with_transfer_function`].
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn transfer_function(&self) -> Option<TransferFunction> {
    self.transfer_function
  }

  /// Sets the transfer-function override in place. See
  /// [`Self::with_transfer_function`] for the consuming builder variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_transfer_function(&mut self, tf: TransferFunction) -> &mut Self {
    self.transfer_function = Some(tf);
    self
  }

  /// Pins the [`TransferFunction`] the [`AveragingDomain::Linear`] path
  /// linearises and re-encodes through, overriding the per-`ColorMatrix`
  /// default ([`TransferFunction::for_matrix`]: the [`ColorMatrix::Rgb`](crate::ColorMatrix::Rgb)
  /// identity → [`TransferFunction::Srgb`], every YCbCr video matrix →
  /// [`TransferFunction::Bt1886`]).
  ///
  /// Use this when the source's transfer characteristics are known out of
  /// band (colconv's YUV row stage carries the matrix but not the
  /// transfer). The override is consulted only on the linear-light path;
  /// the encoded and direct paths ignore it. See
  /// [`Self::set_transfer_function`] for the in-place variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_transfer_function(mut self, tf: TransferFunction) -> Self {
    self.set_transfer_function(tf);
    self
  }

  /// The [`ChromaLocation`](crate::ChromaLocation) driving siting-aware
  /// 4:2:0 chroma upsampling (#302). Defaults to
  /// [`ChromaLocation::Unspecified`](crate::ChromaLocation::Unspecified).
  /// See [`Self::with_chroma_location`].
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn chroma_location(&self) -> crate::ChromaLocation {
    self.chroma_location
  }

  /// The source colour [`Primaries`](crate::Primaries) carried via
  /// [`Self::with_color_spec`] — the coefficient source for the
  /// [`ColorMatrix::ChromaDerivedNcl`](crate::ColorMatrix::ChromaDerivedNcl)
  /// decode (#303). Defaults to
  /// [`Primaries::Unspecified`](crate::Primaries::Unspecified).
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn primaries(&self) -> crate::Primaries {
    self.primaries
  }

  /// The source [`Transfer`](crate::Transfer) carried via
  /// [`Self::with_color_spec`] — the PQ/HLG selector for the
  /// [`ColorMatrix::Ictcp`](crate::ColorMatrix::Ictcp) non-affine decode
  /// (#303). Defaults to
  /// [`Transfer::Unspecified`](crate::Transfer::Unspecified).
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn transfer(&self) -> crate::Transfer {
    self.transfer
  }

  /// Sets the chroma sample location in place. See
  /// [`Self::with_chroma_location`] for the consuming builder variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_chroma_location(&mut self, loc: crate::ChromaLocation) -> &mut Self {
    self.chroma_location = loc;
    self
  }

  /// Pins the source's [`ChromaLocation`](crate::ChromaLocation) so the
  /// 4:2:0 decode reconstructs chroma at the correct horizontal phase
  /// (#302). Source it from a [`ColorSpec`](crate::ColorSpec):
  /// `sink.with_chroma_location(spec.chroma_location())`.
  ///
  /// The default [`ChromaLocation::Unspecified`](crate::ChromaLocation::Unspecified)
  /// — and every horizontally co-sited value (`Left` / `TopLeft` /
  /// `BottomLeft`) — keeps colconv's nearest-neighbor decode, byte-identical
  /// to the pre-#302 output. The horizontally **centered** sitings (`Center`
  /// / `Top` / `Bottom`, the MPEG-1 / JPEG phase) route the identity-plan
  /// 4:2:0 decode through a phase-0.5 chroma upsample + the 4:4:4 kernels,
  /// correcting the half-pixel chroma shift. Wired for the planar `Yuv420p`
  /// and the semi-planar `Nv12` / `Nv21` identity decodes. The vertical phase
  /// is not yet consumed (see `chroma_420_center_sited_h`); the remaining 4:2:0
  /// families and the resampling tiers ignore this knob until the #302 rollout
  /// extends them. See [`Self::set_chroma_location`] for the in-place variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_chroma_location(mut self, loc: crate::ChromaLocation) -> Self {
    self.set_chroma_location(loc);
    self
  }

  /// Applies the **sink-consumed** colour metadata of a resolved
  /// [`ColorSpec`](crate::ColorSpec) in place — currently its
  /// [`ChromaLocation`](crate::ChromaLocation), driving siting-aware 4:2:0
  /// upsampling (#302). See [`Self::with_color_spec`].
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_color_spec(&mut self, spec: crate::ColorSpec) -> &mut Self {
    self.chroma_location = spec.chroma_location();
    self.primaries = spec.primaries();
    self.transfer = spec.transfer();
    self
  }

  /// Configures the sink from a resolved [`ColorSpec`](crate::ColorSpec),
  /// completing the **end-to-end ColorSpec decode path** (#301 / #302 / #303).
  ///
  /// A `ColorSpec` splits across two consumers: its
  /// [`matrix`](crate::ColorSpec::matrix) and
  /// [`full_range`](crate::ColorSpec::full_range) are **walker-consumed** —
  /// route them via [`YuvOptions::from_color_spec`](crate::YuvOptions::from_color_spec)
  /// to the `*_to` walk — while its
  /// [`chroma_location`](crate::ColorSpec::chroma_location) and
  /// [`primaries`](crate::ColorSpec::primaries) are **sink-consumed**
  /// (mediaframe's YUV row carries only range + matrix, neither the siting nor
  /// the primaries). This builder threads the latter pair so the **same
  /// `spec`** drives both halves — the `primaries` feed the
  /// [`ColorMatrix::ChromaDerivedNcl`](crate::ColorMatrix::ChromaDerivedNcl)
  /// decode (#303), whose `Kr` / `Kb` are derived from them:
  ///
  /// ```
  /// # #[cfg(all(feature = "yuv-planar", feature = "rgb"))] {
  /// use colconv::{
  ///   ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat,
  ///   Primaries, Transfer, YuvOptions, sinker::MixedSinker, source::Yuv420p,
  /// };
  ///
  /// let info = ColorInfo::new(
  ///     Primaries::Bt709, Transfer::Bt709, ColorMatrix::Bt601,
  ///     DynamicRange::Limited, ChromaLocation::Center,
  /// );
  /// let spec = ColorSpec::from_info(PixelFormat::Yuv420p, info);
  /// let opts = YuvOptions::from_color_spec(spec);
  /// let mut rgb = [0u8; 4 * 2 * 3];
  /// let sink = MixedSinker::<Yuv420p>::new(4, 2)
  ///     .with_rgb(&mut rgb)
  ///     .unwrap()
  ///     .with_color_spec(spec); // carries ChromaLocation::Center to the decode
  /// assert_eq!(sink.chroma_location(), ChromaLocation::Center);
  /// # let _ = opts;
  /// # }
  /// ```
  ///
  /// **`ChromaDerivedNcl` scope.** The `primaries`-derived `ChromaDerivedNcl`
  /// decode is currently wired for `Yuv420p` only (#316). Every other format —
  /// the planar siblings (`Yuv422p` / `Yuv444p` / `Yuv440p` / …) and the
  /// semi-planar `Nv12` / `Nv21` (#302) — resolves `ChromaDerivedNcl` via the
  /// BT.709 matrix-tag fallback (`Coefficients::for_matrix`), ignoring the
  /// `primaries`; the `chroma_location` is still honored. Wiring the
  /// primaries-derived path into those formats is a follow-up. (Setting a
  /// non-`ChromaDerivedNcl` matrix is unaffected — `primaries` are unused
  /// there.)
  ///
  /// See [`Self::set_color_spec`] for the in-place variant and
  /// [`Self::with_chroma_location`] to set the siting directly.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_color_spec(mut self, spec: crate::ColorSpec) -> Self {
    self.set_color_spec(spec);
    self
  }

  /// Returns how the [`AveragingDomain::Linear`] tail decodes `YUV→RGB`
  /// before the EOTF. See [`Self::with_linear_mode`].
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn linear_mode(&self) -> LinearMode {
    self.linear_mode
  }

  /// Sets the [`LinearMode`] in place. See [`Self::with_linear_mode`] for the
  /// consuming builder variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_linear_mode(&mut self, mode: LinearMode) -> &mut Self {
    self.linear_mode = mode;
    self
  }

  /// Selects whether the [`AveragingDomain::Linear`] average is
  /// **display-referred** (the default) or **scene-referred** (RFC #238
  /// #244), overriding the default [`LinearMode::DisplayReferred`].
  ///
  /// [`LinearMode::DisplayReferred`] decodes `YUV→RGB` through the production
  /// Q15 kernel, which clamps and quantizes to 8-bit `[0, 255]` before the
  /// EOTF, then averages the in-gamut RGB in linear light — a gamma-correct
  /// resize that discards out-of-gamut excursions.
  /// [`LinearMode::SceneReferred`] decodes the **same affine matrix** in
  /// unclamped `f32`, preserving super-black / super-white and saturated-chroma
  /// excursions, lifts that to linear light, averages, and clamps **only** at
  /// the re-encoded output — the physically faithful average for out-of-gamut
  /// content. The two coincide (modulo `f32` rounding) on content that stays in
  /// gamut through the decode.
  ///
  /// The mode is consulted only on the [`AveragingDomain::Linear`] path of the
  /// planar 8-bit YUV family (`Yuv420p` / `Yuv422p` / `Yuv444p` / `Yuv440p`);
  /// the encoded and direct paths ignore it, so it is a no-op unless
  /// [`Self::with_averaging_domain`]`(`[`AveragingDomain::Linear`]`)` is also
  /// set. See [`Self::set_linear_mode`] for the in-place variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_linear_mode(mut self, mode: LinearMode) -> Self {
    self.set_linear_mode(mode);
    self
  }

  /// Full-frame slot count (`out_width x out_height x channels`) with
  /// overflow checking — **output** geometry, since this sizes the
  /// caller's output buffers (`out == source` under the identity
  /// plan). The result is the minimum required `buf.len()` for any
  /// `&[T]` buffer holding `channels` slots per pixel — bytes for
  /// `&[u8]`, `u16` elements for `&[u16]`, `f32` elements for `&[f32]`,
  /// `f16` elements for `&[half::f16]`. The function does NOT scale by
  /// element size; callers compare against `buf.len()` (which Rust
  /// reports in elements of the slice's element type).
  ///
  /// Returns `Err(GeometryOverflow)` if the product cannot fit in
  /// `usize` — only reachable on 32‑bit targets with extreme dimensions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn frame_elems(&self, channels: usize) -> Result<usize, MixedSinkerError> {
    self
      .out_width
      .checked_mul(self.out_height)
      .and_then(|n| n.checked_mul(channels))
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        self.out_width,
        self.out_height,
        channels,
      )))
  }

  /// Full-frame element count (`out_width x out_height`) for a
  /// single-channel `&[T]` buffer, with overflow checking. Equivalent
  /// to [`frame_elems(1)`](Self::frame_elems) numerically, but the
  /// dedicated name documents "one slot per pixel" at the call site
  /// (e.g. luma planes) without the channels=1 magic number.
  ///
  /// Returns `Err(GeometryOverflow { channels: 1 })` on overflow.
  ///
  /// Consumed by every non-Bayer sinker family; Bayer is RGB-only and
  /// has no single-channel pixel-count sizing.
  #[cfg(any(
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn frame_pixels(&self) -> Result<usize, MixedSinkerError> {
    self
      .out_width
      .checked_mul(self.out_height)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        self.out_width,
        self.out_height,
        1,
      )))
  }
}

impl<'a, F: SourceFormat, R> MixedSinker<'a, F, R> {
  /// Creates an empty [`MixedSinker`] whose **output geometry** is
  /// decided by `resampler`: [`Resampler::plan`] runs once, here, and
  /// every buffer attached afterwards validates against the resulting
  /// output geometry. [`PixelSink::begin_frame`] keeps validating the
  /// walker against the `width x height` **source** geometry, so the
  /// existing frame-mismatch protection is unchanged.
  ///
  /// With [`NoopResampler`] this is equivalent to [`MixedSinker::new`]
  /// (identity plan, infallible in practice).
  ///
  /// Formats route non-identity plans as they wire into the streaming
  /// engine (currently [`Yuv420p`](crate::source::Yuv420p)); the
  /// remaining per-format [`PixelSink`] impls stay pinned to the
  /// default strategy, so a sinker built here for those formats
  /// validates buffers against its output geometry but cannot yet
  /// process frames (see the type-level docs).
  ///
  /// # Errors
  ///
  /// [`MixedSinkerError::Resample`] when the strategy rejects the
  /// requested output geometry — see
  /// [`ResampleError`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_resampler(width: usize, height: usize, resampler: R) -> Result<Self, MixedSinkerError>
  where
    R: Resampler,
  {
    let plan = resampler.plan(width, height)?;
    let (out_width, out_height) = match plan.as_ref() {
      Some(plan) => plan.out_dims(),
      None => (width, height),
    };
    let mut sink = Self::with_geometry(width, height, out_width, out_height);
    sink.plan = plan;
    Ok(sink)
  }

  /// Attaches a packed 24-bit RGB output buffer.
  /// Returns `Err(InsufficientRgbBuffer)` if
  /// `buf.len() < out_width x out_height x 3` (output geometry; equals
  /// `width x height x 3` under the default identity resampler), or
  /// `Err(GeometryOverflow)` on 32‑bit targets when the product
  /// overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgb(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb`](Self::with_rgb).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbBuffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb = Some(buf);
    Ok(self)
  }

  // NOTE: `with_rgb_f32` / `set_rgb_f32` and `with_luma_f32` /
  // `set_luma_f32` are **not** declared here. Same rationale as
  // `with_rgb_u16` below: only the float-output formats actually
  // write these buffers, so the setters live on format-specific
  // impl blocks (`Grayf32` writes both; `Rgbf32` and `Rgbf16` only
  // write `rgb_f32`). Attaching an f32 buffer to a sink whose
  // `process` doesn't write it would leave the caller buffer
  // silently stale — the format-specific scoping turns that into a
  // compile error.

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
  /// Returns `Err(InsufficientLumaBuffer)` if
  /// `buf.len() < out_width x out_height` (output geometry), or
  /// `Err(GeometryOverflow)` on 32‑bit overflow.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_luma(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_luma`](Self::with_luma).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaBuffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma = Some(buf);
    Ok(self)
  }

  /// Attaches three HSV output planes. Returns
  /// `Err(MixedSinkerError::InsufficientHsvPlane(e))` (inspect via
  /// `e.which()` / `e.expected()` / `e.actual()`) naming the first
  /// short plane, or `Err(MixedSinkerError::GeometryOverflow(_))` on
  /// 32-bit overflow.
  ///
  /// HSV is only meaningful when at least one source family is
  /// compiled, so this method is gated on the same 15-feature any as
  /// the per-format `process` impls that consume the `hsv` field.
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
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
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_hsv(
    &mut self,
    h: &'a mut [u8],
    s: &'a mut [u8],
    v: &'a mut [u8],
  ) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if h.len() < expected {
      return Err(MixedSinkerError::InsufficientHsvPlane(
        InsufficientHsvPlane::new(HsvPlane::H, expected, h.len()),
      ));
    }
    if s.len() < expected {
      return Err(MixedSinkerError::InsufficientHsvPlane(
        InsufficientHsvPlane::new(HsvPlane::S, expected, s.len()),
      ));
    }
    if v.len() < expected {
      return Err(MixedSinkerError::InsufficientHsvPlane(
        InsufficientHsvPlane::new(HsvPlane::V, expected, v.len()),
      ));
    }
    self.hsv = Some(HsvFrameMut::new(h, s, v));
    Ok(self)
  }
}

/// Returns `Ok(())` iff the walker's frame dimensions exactly match
/// the sinker's configured dimensions. Called from
/// [`PixelSink::begin_frame`] in every `MixedSinker<F>` impl.
///
/// The sinker's RGB / luma / HSV buffers were sized for
/// `configured_w x configured_h`. A shorter frame would silently
/// leave the bottom rows of those buffers stale from the previous
/// frame; a taller frame would overrun them. Either is a real
/// failure mode, but neither is a panic-worthy bug — the caller can
/// recover by rebuilding the sinker. Returning `Err` before any row
/// is processed guarantees no partial output.
///
/// Consumed by every per-format `MixedSinker<F>::process` impl.
/// Under `--features "alloc"` alone (no per-format family), no
/// `process` impl compiles and this helper would be flagged unused.
#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
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
    return Err(MixedSinkerError::DimensionMismatch(DimensionMismatch::new(
      configured_w,
      configured_h,
      frame_w,
      frame_h,
    )));
  }
  Ok(())
}

/// Slice the RGBA row out of an attached RGBA plane buffer. Returns
/// `Err(GeometryOverflow)` if `one_plane_end x 4` wraps `usize` (only
/// reachable on 32-bit targets at extreme dimensions).
///
/// Centralises the duplicated overflow/bounds-check pattern that every
/// `MixedSinker<F>::process` impl runs in both the standalone-RGBA
/// branch and the Strategy-A expand branch.
///
/// Consumed by every non-Bayer sinker family (Bayer is RGB-only, no
/// RGBA path).
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
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
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 4,
    )))?;
  let start = one_plane_start * 4; // ≤ end, fits.
  Ok(&mut buf[start..end])
}

/// `u16` analogue of [`rgba_plane_row_slice`] — slices the RGBA row out
/// of an attached `u16` RGBA plane buffer. This helper indexes in `u16`
/// elements, not bytes: like the `u8` variant, RGBA rows use `x 4`
/// elements per pixel, so the overflow check is the same, but the byte
/// offsets differ because each element is 2 bytes. Used by the
/// high-bit-depth 4:2:0 sinkers that fan `u16` RGB out to `u16` RGBA.
///
/// Bayer is RGB-only and packed YUV 4:2:2 / 4:1:1 (`yuv-packed`) emits
/// u8 only; semi-planar 8-bit NV is also u8-only and never reaches a
/// u16 RGBA fan-out path, so this helper is unused under those families.
/// `yuv-semi-planar` enables it for the high-bit P-format sinks
/// (P010/…), which DO fan `u16` RGB out to `u16` RGBA.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
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
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 4,
    )))?;
  let start = one_plane_start * 4; // ≤ end, fits.
  Ok(&mut buf[start..end])
}

// Test-only allocation failpoint for the RGB scratch grow in
// `rgb_row_buf_or_scratch`. When armed, the next call returns the crate's
// recoverable `AllocationFailed` error WITHOUT growing — letting the
// failure-path regression tests verify that no caller output buffer is
// partially written before the scratch preflight. `Cell<bool>` is plenty
// (single-threaded, take-on-read). Gated on `std` + `yuva` to match the
// only consumers (`thread_local!` needs `std`; the unit-test tree is
// `cfg(all(test, std))` and the failure-path tests are `yuva`-gated), so
// it is not dead code in a `std`-but-no-`yuva` test build.
#[cfg(all(
  test,
  feature = "std",
  any(feature = "yuva", all(feature = "gray", feature = "rgb"))
))]
std::thread_local! {
  static FORCE_RGB_SCRATCH_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the [`rgb_row_buf_or_scratch`] allocation failpoint for the **next**
/// call on the current thread, simulating a recoverable allocator refusal of
/// the RGB scratch grow. The flag is consumed (take-on-read) by that call, so
/// it fires exactly once and cannot leak into a later test. Test-only.
#[cfg(all(
  test,
  feature = "std",
  any(feature = "yuva", all(feature = "gray", feature = "rgb"))
))]
pub(crate) fn arm_rgb_scratch_alloc_failure() {
  FORCE_RGB_SCRATCH_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Clears the [`arm_rgb_scratch_alloc_failure`] failpoint without firing it.
/// Needed by RGB-free paths (e.g. the #263 direct YUVA→HSV route) that
/// deliberately never call [`rgb_row_buf_or_scratch`]: arming proves the path
/// does not consume the flag, and this disarm prevents an un-consumed flag
/// from leaking into a later same-thread test. Test-only. Only the `yuva` #263
/// direct-HSV route uses it, so it stays `yuva`-gated (unlike the arm/hook, which
/// the `gray` RGB-scratch atomicity test also exercises).
#[cfg(all(test, feature = "std", feature = "yuva"))]
pub(crate) fn disarm_rgb_scratch_alloc_failure() {
  FORCE_RGB_SCRATCH_ALLOC_FAILURE.with(|f| f.set(false));
}

// Test-only failpoint for the centered-siting 4:2:0 chroma scratch grow
// (#302), mirroring `FORCE_RGB_SCRATCH_ALLOC_FAILURE`. Gated on `std` +
// `yuv-planar` to match its consumers — the `chroma_siting_420` (planar) and
// `chroma_siting_nv` (semi-planar `Nv12` / `Nv21`) atomicity tests
// (`yuv-planar`-gated; `thread_local!` needs `std`) — so it is not dead code in
// a `std`-but-no-`yuva` test build. `reserve_420_chroma_full` (in
// `planar_8bit`) reads it via `super::` for both the planar and semi-planar
// centered paths.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_CHROMA_FULL_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the `reserve_420_chroma_full` (in `planar_8bit`) allocation
/// failpoint for the **next** call on the current thread,
/// simulating a recoverable allocator refusal of the centered chroma-scratch
/// grow. Consumed (take-on-read) by that call so it fires exactly once and
/// never leaks across tests. Test-only.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) fn arm_chroma_full_alloc_failure() {
  FORCE_CHROMA_FULL_ALLOC_FAILURE.with(|f| f.set(true));
}

// Test-only failpoint for the bottom-sited vertical-phase 4:2:0 chroma lookback
// grow (#302), mirroring `FORCE_CHROMA_FULL_ALLOC_FAILURE`. Gated on `std` +
// `yuv-planar` to match its consumer — the `chroma_siting_420` bottom-sited
// atomicity test. `reserve_420_chroma_prev` (in `planar_8bit`) reads it via
// `super::`.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_CHROMA_PREV_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the `reserve_420_chroma_prev` (in `planar_8bit`) allocation failpoint
/// for the **next** call on the current thread, simulating a recoverable
/// allocator refusal of the bottom-sited vertical-phase chroma-lookback grow.
/// Consumed (take-on-read) by that call so it fires exactly once and never leaks
/// across tests. Test-only.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) fn arm_chroma_prev_alloc_failure() {
  FORCE_CHROMA_PREV_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Pick an RGB row buffer for the kernel to write into: caller's RGB
/// plane slice when attached, or the growing scratch buffer otherwise
/// (HSV-only callers don't allocate an RGB plane). Returns
/// `Err(GeometryOverflow)` if `width x 3` or `one_plane_end x 3` wraps
/// `usize` — see [`rgba_plane_row_slice`] for the rationale.
///
/// `rgb_scratch` is grown via `Vec::resize` only when too small; the
/// caller keeps the existing capacity across rows so steady-state
/// processing allocates zero times.
///
/// Consumed by per-format `process` impls that need a stable RGB row
/// buffer (either user-attached or scratch-backed). Under
/// `--features "alloc"` alone (no per-format family), no impl
/// compiles and this helper would be flagged unused.
#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgb_row_buf_or_scratch<'a>(
  rgb: Option<&'a mut [u8]>,
  rgb_scratch: &'a mut Vec<u8>,
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&'a mut [u8], MixedSinkerError> {
  // Test-only allocation failpoint: simulate a recoverable allocator
  // refusal for the scratch grow WITHOUT actually exhausting memory, so
  // the failure-path regression tests can prove no caller output is
  // partially written before this preflight (see `arm_rgb_scratch_alloc_failure`).
  // `take()` clears the flag so an armed failure fires exactly once and
  // never leaks across tests. Strictly test-only — the non-test build is
  // byte-identical (this hook compiles away entirely).
  #[cfg(all(
    test,
    feature = "std",
    any(feature = "yuva", all(feature = "gray", feature = "rgb"))
  ))]
  if FORCE_RGB_SCRATCH_ALLOC_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(width, height, width, height),
    )));
  }
  match rgb {
    Some(buf) => {
      let end = one_plane_end
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          width, height, 3,
        )))?;
      let start = one_plane_start * 3;
      Ok(&mut buf[start..end])
    }
    None => {
      let row_bytes = width
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          width, height, 3,
        )))?;
      if rgb_scratch.len() < row_bytes {
        rgb_scratch
          .try_reserve_exact(row_bytes - rgb_scratch.len())
          .map_err(|_| {
            MixedSinkerError::Resample(ResampleError::AllocationFailed(
              crate::resample::PlanGeometry::new(width, height, width, height),
            ))
          })?;
        rgb_scratch.resize(row_bytes, 0);
      }
      Ok(&mut rgb_scratch[..row_bytes])
    }
  }
}

/// Grows `scratch` to a single source-width `u16` RGB row
/// (`width * 3` elements) for the **direct** (non-resample) path and
/// returns the slice. Follows the recoverable-allocation contract —
/// `try_reserve_exact` before the resize, mapping allocator refusal to a
/// recoverable [`MixedSinkerError`] instead of aborting in `process` — for
/// the 10-bit packed-RGB `rgba_u16` fan-out, which has no native α kernel
/// and so stages the native RGB row before
/// [`expand_rgb_u16_to_rgba_u16_row`].
#[cfg(feature = "rgb")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn direct_rgb_u16_scratch(
  scratch: &mut Vec<u16>,
  width: usize,
  height: usize,
) -> Result<&mut [u16], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(width, height, width, height),
        ))
      })?;
    scratch.resize(row, 0);
  }
  Ok(&mut scratch[..row])
}

/// Grows `rgb_scratch` to a **source-width** RGB row (`width * 3`
/// bytes) and returns the slice, following the planner's recoverable-
/// allocation contract (the exact reserve makes the resize incapable
/// of reallocating; refusal surfaces as `AllocationFailed` in the
/// preflight phase, not an abort in infallible growth).
///
/// The shared staging point for packed-RGB-canonical resampled
/// sources whose row must be channel-swapped or converted to RGB
/// before feeding the area stream. [`MixedSinker<Rgb24>`] skips it —
/// its source is already RGB and feeds the stream with zero copy.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn source_rgb_scratch<'s>(
  rgb_scratch: &'s mut Vec<u8>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u8], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a scratch OOM AFTER a successful stream build leaves
  // the stream field `None` AND `resample_outputs` uncommitted (retryable).
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "rgb",
      feature = "gbr",
      feature = "bayer",
      feature = "mono",
      feature = "xyz",
      feature = "rgb-legacy",
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
    )
  ))]
  if FORCE_SOURCE_RGB_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  let row_bytes = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      3,
    )))?;
  if rgb_scratch.len() < row_bytes {
    rgb_scratch
      .try_reserve_exact(row_bytes - rgb_scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    rgb_scratch.resize(row_bytes, 0);
  }
  Ok(&mut rgb_scratch[..row_bytes])
}

// Single-shot source-RGB-scratch grow failpoint for the RFC #238 tail-collapse
// transactional-preflight tests. Gated on the families that stage their converted
// or drop-alpha u8 RGB through this helper (packed-RGB / GBR / Bayer / Pal8 /
// legacy-RGB emit scratch and the `Xyz12` emit's narrow scratch);
// `source_rgb_scratch` consults it under the same gate. Separate from the `try_box`
// box-alloc failpoint so a test can fail the SCRATCH grow specifically (after a
// successful stream construction).
#[cfg(all(
  test,
  feature = "std",
  any(
    feature = "rgb",
    feature = "gbr",
    feature = "bayer",
    feature = "mono",
    feature = "xyz",
    feature = "rgb-legacy",
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
  )
))]
std::thread_local! {
  static FORCE_SOURCE_RGB_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_rgb_scratch`] grow
/// refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape (a scratch OOM after a successful stream build
/// leaves the stream field `None` AND `resample_outputs` uncommitted, so the row
/// is retryable). Test-only.
#[cfg(all(
  test,
  feature = "std",
  any(
    feature = "rgb",
    feature = "gbr",
    feature = "bayer",
    feature = "mono",
    feature = "xyz",
    feature = "rgb-legacy",
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
  )
))]
pub(crate) fn arm_source_rgb_scratch_failure() {
  FORCE_SOURCE_RGB_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Grows `scratch` to a **source-width** `u8` luma row (`width`
/// bytes) and returns the slice, following the planner's recoverable-
/// allocation contract (the exact reserve makes the resize incapable
/// of reallocating; refusal surfaces as `AllocationFailed` in the
/// preflight phase, not an abort in infallible growth).
///
/// The staging point for `mono` resampling (each 1-bit source row is
/// expanded to source-width 0/255 luma here), for **packed YUV 4:2:2**
/// resampling (the interleaved Y bytes are de-interleaved here), and for
/// the `Ya8` gray+alpha resample (the native Y bytes of the packed
/// `[Y, A]` row are de-interleaved here) before feeding the single-channel
/// area stream. Compiled wherever the packed-RGBA u8 tail is (it threads
/// this for the `Ya8` native-Y stream), so the `rgb` / `gbr` callers — for
/// which the native-Y path is inert — still link.
#[cfg(any(
  feature = "mono",
  feature = "yuv-packed",
  feature = "gray",
  feature = "rgb",
  feature = "gbr",
  feature = "yuv-444-packed",
  feature = "yuva"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn source_luma_scratch<'s>(
  scratch: &'s mut Vec<u8>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u8], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a scratch OOM AFTER a successful stream build leaves
  // the `mono` luma stream field `None` AND `resample_outputs` uncommitted
  // (retryable). Separate from the `try_box` box-alloc failpoint so a `mono` test
  // can fail the SCRATCH grow specifically.
  #[cfg(all(test, feature = "std", feature = "mono"))]
  if FORCE_SOURCE_LUMA_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  if scratch.len() < width {
    scratch
      .try_reserve_exact(width - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(width, 0);
  }
  Ok(&mut scratch[..width])
}

// Single-shot source-luma-`u8`-scratch grow failpoint for the RFC #238 tail-collapse
// transactional-preflight tests. Gated on `mono` to match the `mono` resample
// atomicity test module; `source_luma_scratch` consults it under the same gate.
#[cfg(all(test, feature = "std", feature = "mono"))]
std::thread_local! {
  static FORCE_SOURCE_LUMA_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_luma_scratch`] grow
/// refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape (a `mono` scratch OOM after a successful stream
/// build leaves the luma stream field `None` AND `resample_outputs` uncommitted, so
/// the row is retryable). Test-only.
#[cfg(all(test, feature = "std", feature = "mono"))]
pub(crate) fn arm_source_luma_scratch_failure() {
  FORCE_SOURCE_LUMA_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Grows `scratch` to a **source-width** host-native `u16` luma row
/// (`width` elements) and returns the slice, following the planner's
/// recoverable-allocation contract (the exact reserve makes the resize
/// incapable of reallocating; refusal surfaces as `AllocationFailed` in
/// the preflight phase, not an abort in infallible growth).
///
/// The staging point for the [`Gray16`](crate::source::Gray16) resample
/// path (the wire row converts to host-native u16 luma here) and for the
/// high-bit packed YUV families (`yuv-444-packed` / `y2xx` / `v210`),
/// whose native Y is de-interleaved here (each format's own
/// `*_to_luma_u16_row` kernel) before feeding the single-channel u16 luma
/// stream. The u16 twin of [`source_luma_scratch`]. Compiled wherever the
/// high-bit packed-RGBA u16 tail is (it threads this for the `Ya16`
/// native-Y stream), so the `rgb` / `gbr` callers — for which the native-Y
/// path is inert — still link.
#[cfg(any(
  feature = "gray",
  feature = "yuva",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb",
  feature = "gbr",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn source_luma_u16_scratch<'s>(
  scratch: &'s mut Vec<u16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u16], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a scratch OOM AFTER a successful stream build leaves
  // the u16 gray luma stream field `None` AND `resample_outputs` uncommitted
  // (retryable). Gated on `gray` to match the `Gray16` resample atomicity test.
  #[cfg(all(test, feature = "std", feature = "gray"))]
  if FORCE_SOURCE_LUMA_U16_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  if scratch.len() < width {
    scratch
      .try_reserve_exact(width - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(width, 0);
  }
  Ok(&mut scratch[..width])
}

// Single-shot source-luma-`u16`-scratch grow failpoint for the RFC #238 tail-collapse
// transactional-preflight tests. Gated on `gray` to match the `Gray16` resample
// atomicity test module; `source_luma_u16_scratch` consults it under the same gate.
#[cfg(all(test, feature = "std", feature = "gray"))]
std::thread_local! {
  static FORCE_SOURCE_LUMA_U16_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_luma_u16_scratch`]
/// grow refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape (a `Gray16` scratch OOM after a successful stream
/// build leaves the luma stream field `None` AND `resample_outputs` uncommitted, so
/// the row is retryable). Test-only.
#[cfg(all(test, feature = "std", feature = "gray"))]
pub(crate) fn arm_source_luma_u16_scratch_failure() {
  FORCE_SOURCE_LUMA_U16_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Source-width host-native `u32` luma staging for the
/// [`Gray32`](crate::source::Gray32) resample path — the wire row converts
/// here (source wire `BE` → host-native `u32` value) before feeding the
/// native-`u32` luma streams, so binning runs at full `u32` precision (the
/// `u32` twin of [`source_luma_u16_scratch`]). Grows the scratch under the
/// planner's recoverable-allocation contract (output-proportional, so a
/// refusal surfaces as `AllocationFailed` in the preflight, never an abort
/// in infallible growth).
#[cfg(feature = "gray")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn source_luma_u32_scratch<'s>(
  scratch: &'s mut Vec<u32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u32], MixedSinkerError> {
  if scratch.len() < width {
    scratch
      .try_reserve_exact(width - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(width, 0);
  }
  Ok(&mut scratch[..width])
}

/// The single-channel gray luma stream handed to a [`gray_luma_resample`] emit —
/// the span kind picks the concrete engine (`Area` bins, `Filter` runs the
/// signed-coefficient twin), so the one per-format emit closure serves both. A
/// `None` stream (the pregrow pass) never reaches the emit as a value; `Some`
/// carries whichever kind is fed this frame.
#[cfg(feature = "gray")]
pub(super) enum GrayLumaStreamMut<
  'a,
  E: crate::resample::AreaSample + crate::resample::FilterSample,
> {
  /// Area (box-averaging) luma stream.
  Area(&'a mut crate::resample::AreaStream<E>),
  /// Signed-coefficient filter luma stream.
  Filter(&'a mut crate::resample::FilterStream<E>),
}

/// RFC #238 tail-collapse — the transactional single-channel **gray luma** resample
/// skeleton shared by every pure-gray element family (`Gray8` u8, `GrayN` / `Gray16`
/// u16, `Gray32` u32, `Grayf32` / `Grayf16` f32; the element-generic 1-channel twin
/// of the packed [`packed_rgb_f32_resample`]). Folds the split preflight + luma
/// stream build + per-format emit into one call behind the shared compare-before /
/// commit-after preflight ([`resample_preflight_check_only`]): the missing 1-channel
/// stream is built into a LOCAL, the emit's own source / clamp / RGB scratch is
/// pre-grown via an `emit(None)` grow-only pass, and only THEN do the stream field
/// insert and the output-set freeze COMMIT TOGETHER — so a first-row allocation
/// failure leaves the stream field `None` AND `resample_outputs` uncommitted
/// (retryable even with a changed output attachment, the #180 discipline). Byte-
/// identical to the pre-collapse split path: only the freeze moves past the fallible
/// allocations.
///
/// The span kind selects the engine internally (area bin / signed-coefficient
/// filter) and hands the `emit` the fed stream as a [`GrayLumaStreamMut`], so the one
/// per-format emit closure serves both kinds. `emit(None, ..)` pregrows every
/// emit-owned scratch and returns; `emit(Some(stream), ..)` stages the source luma
/// and derives every attached output. The gray emits carry their own (format-varying)
/// output subset, so the driver threads the full gray superset — `rgb` / `rgba` /
/// `rgb_u16` / `rgba_u16` / `luma` / `luma_u16` / `rgb_f32` / `luma_f32` / `hsv` —
/// and a format passes `&mut None` for the outputs it never emits.
#[cfg(feature = "gray")]
#[allow(clippy::too_many_arguments)]
pub(super) fn gray_luma_resample<E: crate::resample::AreaSample + crate::resample::FilterSample>(
  luma_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<E>>>,
  luma_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<E>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  luma_f32: &mut Option<&mut [f32]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  plan: &ResamplePlan,
  idx: usize,
  mut emit: impl FnMut(
    Option<GrayLumaStreamMut<'_, E>>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [f32]>,
    &mut Option<HsvFrameMut<'_>>,
  ) -> Result<(), MixedSinkerError>,
) -> Result<(), MixedSinkerError> {
  // Single-kernel filter tail — reject a BICUBLIN plan before any state change (its
  // chroma windows are read only by the `Yuv420p` per-plane route). Ordered first,
  // exactly like the pre-collapse gray path, so even a no-output bicublin call errors.
  plan.ensure_single_kernel_filter()?;
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || luma_f32.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the fed stream is the sole
  // per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(match plan.kind() {
      crate::resample::SpanKind::Area => luma_stream.as_ref().map_or(0, |s| s.next_y()),
      crate::resample::SpanKind::Filter => luma_filter_stream.as_ref().map_or(0, |s| s.next_y()),
    })
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER the
  // fallible stream alloc + scratch pregrow below succeed, so a first-row allocation
  // failure leaves `resample_outputs` uncommitted (retryable with a changed output
  // attachment). A no-output sink short-circuits to a no-op.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    &None,
    &None,
    &None,
    &None,
    hsv,
    luma_f32,
    expected,
    idx,
  )? {
    return Ok(());
  }
  match plan.kind() {
    crate::resample::SpanKind::Area => {
      // Build the missing area stream into a LOCAL (NO field mutation), so the field
      // insert and the freeze commit TOGETHER only after every pre-feed allocation.
      let new_stream = if luma_stream.is_none() {
        let stream =
          crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
        Some(crate::resample::try_box(stream).map_err(|_| {
          MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
            crate::resample::PlanGeometry::new(
              plan.src_w(),
              plan.src_h(),
              plan.out_w(),
              plan.out_h(),
            ),
          ))
        })?)
      } else {
        None
      };
      // Pregrow every emit-owned scratch BEFORE the freeze (a `None` stream grows it
      // and returns), so NOTHING fallible runs after the commit.
      emit(
        None, rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, rgb_f32, luma_f32, hsv,
      )?;
      // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
      // freeze the output set together.
      if let Some(s) = new_stream {
        *luma_stream = Some(s);
      }
      frozen_outputs_check(
        resample_outputs,
        luma,
        luma_u16,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        rgb_f32,
        &None,
        &None,
        &None,
        &None,
        hsv,
        luma_f32,
        idx,
      )?;
      let stream = luma_stream.as_deref_mut().expect("created above");
      emit(
        Some(GrayLumaStreamMut::Area(stream)),
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        rgb_f32,
        luma_f32,
        hsv,
      )
    }
    crate::resample::SpanKind::Filter => {
      let new_stream = if luma_filter_stream.is_none() {
        let (fh, fv) = (
          plan
            .filter_h()
            .expect("filter plan carries horizontal windows"),
          plan
            .filter_v()
            .expect("filter plan carries vertical windows"),
        );
        let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
        Some(crate::resample::try_box(stream).map_err(|_| {
          MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
            crate::resample::PlanGeometry::new(
              plan.src_w(),
              plan.src_h(),
              plan.out_w(),
              plan.out_h(),
            ),
          ))
        })?)
      } else {
        None
      };
      emit(
        None, rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, rgb_f32, luma_f32, hsv,
      )?;
      if let Some(s) = new_stream {
        *luma_filter_stream = Some(s);
      }
      frozen_outputs_check(
        resample_outputs,
        luma,
        luma_u16,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        rgb_f32,
        &None,
        &None,
        &None,
        &None,
        hsv,
        luma_f32,
        idx,
      )?;
      let stream = luma_filter_stream.as_deref_mut().expect("created above");
      emit(
        Some(GrayLumaStreamMut::Filter(stream)),
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        rgb_f32,
        luma_f32,
        hsv,
      )
    }
  }
}

/// Feeds the prepared source-width canonical RGB row into the (already
/// sequence-checked) stream and derives every attached output (rgb,
/// rgba, luma, luma_u16, hsv) from each finalized output row.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[cfg_attr(
  not(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  )),
  allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_resample_emit(
  stream: &mut impl crate::resample::RowResampler<u8>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_rgb: &[u8],
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  stream.feed_row(idx, src_rgb, use_simd, |oy, out_row| {
    if let Some(buf) = rgb.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_row);
    }
    if let Some(buf) = luma.as_deref_mut() {
      crate::row::rgb_to_luma_row(
        out_row,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        use_simd,
      );
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::rgb_to_luma_u16_row(
        out_row,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        use_simd,
      );
    }
    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      crate::row::rgb_to_hsv_row(
        out_row,
        &mut h[oy * ow..(oy + 1) * ow],
        &mut s[oy * ow..(oy + 1) * ow],
        &mut v[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = rgba.as_deref_mut() {
      crate::row::expand_rgb_to_rgba_row(out_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
    }
  })?;

  Ok(())
}

/// RFC #238 tail-collapse — the transactional packed-RGB **area** resample.
/// Folds the split preflight + area-stream build + [`packed_rgb_resample_emit`]
/// sequence into one call behind the shared compare-before / commit-after
/// preflight
/// ([`resample_preflight_check_only`](resample_preflight_check_only)),
/// mirroring [`planar_dual_resample`](planar_resample::planar_dual_resample): the
/// missing area stream is built into a LOCAL and the source-width scratch is grown
/// with NO field mutation, so the stream field insert and the output-set freeze
/// COMMIT TOGETHER only after every fallible pre-feed allocation succeeds. A
/// first-row allocation failure therefore leaves `rgb_stream` `None` AND
/// `resample_outputs` uncommitted (never a partial commit) — state-atomic and
/// retryable even with a changed output attachment (the #180 discipline). This is
/// a pure reordering of the pre-collapse split path: it emits the identical bytes,
/// only deferring the field mutations past the fallible allocations.
///
/// `direct` is `Some(row)` for the identity source
/// ([`Rgb24`](crate::source::Rgb24) — the packed source row already IS the
/// canonical RGB row, fed zero-copy); the channel-swapped / padding-byte / planar
/// `Gbrp` sources pass `None` and a `convert` closure that stages a source-width
/// canonical RGB row into `rgb_scratch` (the fallible growth is hoisted ahead of
/// the freeze). The per-output emit is [`packed_rgb_resample_emit`].
///
/// The element-type PATTERN-SETTER for the tail-collapse: the u16 / u32 / f32
/// packed-RGB families mirror this shape against their own element-typed stream
/// builder + emit.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_resample(
  rgb_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  direct: Option<&[u8]>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  let need_any =
    rgb.is_some() || rgba.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the area stream is the sole
  // per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(rgb_stream.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream alloc + scratch growth below succeed, so a first-row
  // allocation failure leaves `resample_outputs` uncommitted (retryable with a
  // changed output attachment). A no-output sink short-circuits to a no-op.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Area-only sink: reject a filter plan before any allocation (the area
  // stream would feed empty spans under a filter plan). Runs AFTER the
  // compare-only preflight, so a no-output sink stays a no-op even under a
  // filter plan.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  // Build the missing area stream into a LOCAL and grow the staging scratch —
  // every fallible pre-feed step — with NO field mutation, so the stream field
  // insert and the freeze commit TOGETHER only after both succeed (the
  // `planar_dual_resample` atomic shape: a scratch OOM leaves `rgb_stream` `None`
  // AND `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgb_stream.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  if direct.is_none() {
    source_rgb_scratch(rgb_scratch, w, plan)?;
  }
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare above already caught any later
  // mid-frame output change; the freeze stores the snapshot on the first row).
  if let Some(s) = new_stream {
    *rgb_stream = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Stage the source-width canonical RGB row (identity zero-copy, or the caller's
  // conversion into the grown scratch) and run the shared per-output emit.
  let stream = rgb_stream.as_mut().expect("created above");
  let src: &[u8] = match direct {
    Some(row) => row,
    None => {
      let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
      convert(scratch);
      scratch
    }
  };
  packed_rgb_resample_emit(
    stream, plan, rgb, rgba, luma, luma_u16, hsv, src, matrix, full_range, idx, use_simd,
  )
}

/// RFC #238 tail-collapse — the transactional packed-RGB **filter** resample, the
/// [`SpanKind::Filter`](crate::resample::SpanKind) twin of [`packed_rgb_resample`]
/// over a 3-channel u8 [`FilterStream`](crate::resample::FilterStream). Identical
/// compare-before / commit-after discipline (the filter stream is built into a
/// local and its field insert + the freeze commit together only after the scratch
/// grows) and identical [`packed_rgb_resample_emit`] tail (the emit is generic
/// over the resampler kind), so the resampled colour equals the equivalent area
/// resample up to the signed kernel weights while remaining byte-identical to the
/// pre-collapse split filter path.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_resample_filter(
  rgb_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  direct: Option<&[u8]>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  let need_any =
    rgb.is_some() || rgba.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  let expected = if need_any {
    Some(rgb_filter_stream.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_rgb_resample`.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Single-kernel filter: reject a BICUBLIN plan before any allocation (its chroma
  // windows are read only by the `Yuv420p` per-plane route, so filtering every
  // channel with the luma kernel would be wrong output).
  plan.ensure_single_kernel_filter()?;
  // Build the missing filter stream into a LOCAL and grow the staging scratch —
  // every fallible pre-feed step — with NO field mutation, so the stream field
  // insert and the freeze commit TOGETHER (the `planar_dual_resample` atomic shape:
  // a scratch OOM leaves `rgb_filter_stream` `None` AND `resample_outputs`
  // uncommitted, never a partial commit).
  let new_stream = if rgb_filter_stream.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  if direct.is_none() {
    source_rgb_scratch(rgb_scratch, w, plan)?;
  }
  if let Some(s) = new_stream {
    *rgb_filter_stream = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgb_filter_stream.as_mut().expect("created above");
  let src: &[u8] = match direct {
    Some(row) => row,
    None => {
      let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
      convert(scratch);
      scratch
    }
  };
  packed_rgb_resample_emit(
    stream, plan, rgb, rgba, luma, luma_u16, hsv, src, matrix, full_range, idx, use_simd,
  )
}

/// Separable-filter fused resize for the **real-alpha** 8-bit packed-RGBA
/// sources ([`Rgba`](crate::source::Rgba) and the channel reorderings):
/// the `Filter`-plan twin of the area [`packed_rgba_resample`]. PIL resizes
/// RGBA by filtering R, G, B, A **independently with no premultiplication**,
/// so the source row is staged as one canonical source-width `R, G, B, A`
/// u8 row (`convert_rgba`) and fed to a single 4-channel
/// [`FilterStream`](crate::resample::FilterStream); each finalized output row
/// is the resampled RGBA. Because the u8 filter is byte-exact per channel,
/// the resampled RGBA frame is byte-exact versus PIL's RGBA resize.
///
/// Attached outputs derive from each finalized RGBA row: `rgba` copies it,
/// and `rgb` / `luma` / `hsv` come from the alpha-dropped RGB. These sources
/// are genuinely chromatic (no native luma plane), so luma is color-derived
/// from the resampled RGB.
///
/// Sequence-check precedes every allocation (the 4-channel stream creation
/// runs after the no-output and out-of-sequence rejections), keeping the
/// call atomic: a rejected first row stores no frozen-output snapshot and
/// `AllocationFailed` never masks `OutOfSequenceRow`. There is no
/// premultiplied route — a packed-RGBA source under premultiplied alpha
/// stays on the area path (which un-premultiplies); the filter path is
/// reached only for straight alpha.
///
/// Gated to `any(rgb, gbr)`: the 8-bit planar GBR+alpha `Gbrap` source
/// (straight alpha) scatters its G/B/R/A planes into the canonical packed
/// RGBA row via the same `convert_rgba` closure and shares this filter tail.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_filter_resample(
  rgba_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<u8>,
  rgb_drop_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let ow = plan.out_w();
  let need_any =
    rgb.is_some() || rgba.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the 4-channel filter stream
  // is the sole per-row stream), or `None` when no output is attached so the
  // preflight short-circuits to a no-op.
  let expected = if need_any {
    Some(rgba_filter_stream.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible 4-channel stream alloc + scratch growth below succeed (RFC #238
  // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
  // uncommitted — retryable with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // The rgb / hsv / luma / luma_u16 outputs need an alpha-dropped RGB row,
  // sized to the out-width RGB row only when one of those is attached, so an
  // rgba-only sink neither grows it nor risks its allocation failure.
  let need_rgb_drop = rgb.is_some() || hsv.is_some() || luma.is_some() || luma_u16.is_some();
  // Build the missing filter stream into a LOCAL and grow the scratches — every
  // fallible pre-feed step — with NO field mutation, so the stream field insert
  // and the freeze commit TOGETHER only after they all succeed (the
  // `planar_dual_resample` atomic shape: a scratch OOM leaves `rgba_filter_stream`
  // `None` AND `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgba_filter_stream.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?;
  }
  source_rgba_scratch(rgba_scratch, w, plan)?;
  // Every pre-feed allocation (stream, drop-alpha RGB scratch, RGBA scratch)
  // succeeded — COMMIT atomically: insert the stream and freeze the output set
  // together (the compare-only preflight above already caught any later mid-frame
  // output change).
  if let Some(s) = new_stream {
    *rgba_filter_stream = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Stage the drop-alpha RGB + canonical RGBA scratches (these grows re-run as
  // no-ops after the pre-commit grows above) and run the per-output emit.
  let rgb_drop: &mut [u8] = if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_scratch(rgba_scratch, w, plan)?;
  convert_rgba(src_rgba);
  let stream = rgba_filter_stream.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, finalized| {
    // Straight-alpha RGBA output — the finalized 4-channel filter row.
    if let Some(buf) = rgba.as_deref_mut() {
      buf[oy * 4 * ow..(oy + 1) * 4 * ow].copy_from_slice(finalized);
    }
    if need_rgb_drop {
      let nrow = &mut rgb_drop[..3 * ow];
      drop_alpha_rgba_to_rgb_row(finalized, nrow, ow);
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      // luma_u16 derives from the same alpha-dropped binned RGB (the u8-RGB →
      // u16-luma kernel) — byte-identical to the area 4-channel path. Only
      // `Gbrap` exposes a u16 luma through this u8 4-channel filter; the
      // packed-RGBA u8 sources leave it `None`.
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Separable-filter fused resize for the **real-alpha** high-bit packed-RGBA
/// `u16` sources ([`Rgba64`](crate::source::Rgba64) /
/// [`Bgra64`](crate::source::Bgra64)): the `Filter`-plan twin of the area
/// [`packed_rgba_u16_resample`], and the 4-channel analogue of the 3-channel
/// [`packed_rgb_u16_resample_emit`]. PIL resizes RGBA by filtering R, G, B, A
/// **independently with no premultiplication**, so the source row is staged as
/// one canonical source-width `R, G, B, A` host-native u16 row
/// (`convert_rgba_u16`, the format's `*_to_rgba_u16` kernel) and fed to a
/// single 4-channel [`FilterStream<u16>`](crate::resample::FilterStream); each
/// finalized output row is the resampled native-depth RGBA. Because the u16
/// filter is byte-exact per channel, every output channel — alpha included —
/// equals the single-plane filter of that channel.
///
/// Attached outputs derive from each finalized RGBA row: `rgba_u16` copies it
/// and `rgb_u16` drops alpha, both at native depth; `rgba` / `rgb` / `luma` /
/// `hsv` come from a single `>> (SRC_BITS - 8)` narrowing of the alpha-dropped
/// RGB — the source-of-truth ordering the 3-channel u16 path uses. These
/// sources are chromatic (no native luma plane), so luma is color-derived from
/// the resampled RGB.
///
/// `NATIVE_LUMA16` selects the `luma_u16` flavor, mirroring the 3-channel u16
/// emit and the area 4-channel path: when `true` (`GbrapN`) luma_u16 is the
/// full native-depth Y' from the clamped alpha-dropped native RGB
/// (`rgb_to_luma_u16_native_row` keyed on `SRC_BITS`), byte-identical to a
/// direct `GbrapN` conversion and to the area path; when `false` (`Rgba64` /
/// `Bgra64`) it is the 8-bit-precision Y' from the narrowed RGB, zero-extended,
/// matching those formats' direct and area paths. Merely attaching an alpha
/// output therefore no longer downgrades `GbrapN` luma_u16 to 0..255.
///
/// `SRC_BITS` is the source's active bit depth — `16` for the full-16-bit
/// `Rgba64` / `Bgra64`. A signed kernel (CatmullRom / Lanczos3) overshoots a
/// legal edge, so a finalized sample can exceed the source's native max
/// `(1 << SRC_BITS) - 1` even though the `FilterStream` clamps to the full
/// `u16` range; for `SRC_BITS < 16` every binned sample (alpha too) is clamped
/// to the native max before any u16 copy, RGBA expansion, or u8 narrowing —
/// the same native-depth clamp the merged high-bit GBR routing applies in the
/// shared u16 emit. For `SRC_BITS == 16` the native max is the u16 max, so the
/// clamp is a no-op (the `FilterStream`'s `0..=65535` clamp *is* the native
/// clamp for `Rgba64` / `Bgra64`).
///
/// Sequence-check precedes every allocation (the 4-channel stream creation
/// runs after the no-output and out-of-sequence rejections), keeping the call
/// atomic: a rejected first row stores no frozen-output snapshot and
/// `AllocationFailed` never masks `OutOfSequenceRow`. There is no
/// premultiplied route — a packed-RGBA source under premultiplied alpha stays
/// on the area path (which un-premultiplies); the filter path is reached only
/// for straight alpha.
///
/// Gated to `any(rgb, gbr)` (the filter twin of
/// [`Self::rgba_filter_stream`]): besides the full-16-bit `Rgba64` / `Bgra64`,
/// the high-bit planar GBR+alpha family (`Gbrap10`…`Gbrap16`, straight alpha)
/// de-interleaves its native-depth G/B/R/A planes into the same canonical
/// packed RGBA u16 row via `convert_rgba_u16` and shares this 4-channel tail,
/// with the `SRC_BITS` native-max clamp keyed by its source depth.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_u16_filter_resample<const SRC_BITS: u32, const NATIVE_LUMA16: bool>(
  rgba_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch_u16: &mut Vec<u16>,
  color_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba_u16: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS > 8 && SRC_BITS <= 16,
      "SRC_BITS must be in (8, 16] for the high-bit packed RGBA filter tail"
    )
  };
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the 4-channel filter stream
  // is the sole per-row stream), or `None` when no output is attached so the
  // preflight short-circuits to a no-op.
  let expected = if need_any {
    Some(rgba_filter_stream_u16.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible 4-channel stream alloc + scratch growth below succeed (RFC #238
  // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
  // uncommitted — retryable with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // The u8 / narrowed-luma_u16 outputs derive from a `>> (SRC_BITS - 8)`
  // narrowing of the alpha-dropped binned RGB; a native-u16-only sink (only
  // rgb_u16 / rgba_u16) never touches it, so the out-width u8 narrow scratch
  // is sized — and its allocation failure risked — only when a narrowed
  // output is attached. Under `NATIVE_LUMA16` (`GbrapN`) luma_u16 is computed
  // at full native precision from the alpha-dropped native RGB instead, so it
  // takes the native-luma path below rather than this narrowing.
  let narrowed_luma_u16 = !NATIVE_LUMA16 && luma_u16.is_some();
  let need_narrow =
    rgb.is_some() || rgba.is_some() || luma.is_some() || narrowed_luma_u16 || hsv.is_some();
  // Native-precision luma_u16 (`GbrapN`): drop alpha from the clamped native
  // binned RGB and run the same `rgb_to_luma_u16_native_row` the area / direct
  // paths use, so the resampled luma_u16 is byte-identical to a direct GbrapN
  // conversion of the binned frame. The narrowed `Rgba64` / `Bgra64` callers
  // leave `NATIVE_LUMA16` false and take the narrowed luma_u16 above.
  let native_luma = NATIVE_LUMA16 && luma_u16.is_some();
  // The clamped drop-alpha native u16 RGB row is the single source for both
  // the u8 narrowing and the native luma, so it is sized whenever either is
  // requested.
  let need_native_rgb = need_narrow || native_luma;
  // Build the missing filter stream into a LOCAL and grow the scratches — every
  // fallible pre-feed step — with NO field mutation, so the stream field insert and
  // the freeze commit TOGETHER only after they all succeed (the
  // `planar_dual_resample` atomic shape: a scratch OOM leaves
  // `rgba_filter_stream_u16` `None` AND `resample_outputs` uncommitted, never a
  // partial commit).
  let new_stream = if rgba_filter_stream_u16.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Grow every pre-feed scratch (the conditional native-RGB / narrow rows and the
  // always-sized source-width RGBA staging) BEFORE the commit, so a scratch OOM
  // likewise leaves the stream field `None` and `resample_outputs` uncommitted.
  if need_native_rgb {
    source_rgb_u16_scratch(color_scratch_u16, ow, plan)?;
  }
  if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?;
  }
  source_rgba_u16_scratch(rgba_scratch_u16, w, plan)?;
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare-only preflight above already caught
  // any later mid-frame output change).
  if let Some(s) = new_stream {
    *rgba_filter_stream_u16 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Re-fetch the scratches (these grows re-run as no-ops after the pre-commit grows
  // above). Out-width clamped drop-alpha native RGB row, resolved per finalized
  // output row; sized when any narrowed output OR native luma_u16 is attached
  // (every narrowed output and the native luma read this one canonical native RGB
  // row).
  let color_rgb: &mut [u16] = if need_native_rgb {
    source_rgb_u16_scratch(color_scratch_u16, ow, plan)?
  } else {
    &mut []
  };
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_u16_scratch(rgba_scratch_u16, w, plan)?;
  convert_rgba_u16(src_rgba);
  // A finalized 4-channel `binned` sample can overshoot the native max for a
  // sub-16-bit source; clamp before any output is derived (no-op at 16-bit —
  // the `FilterStream`'s `0..=65535` clamp is already the native clamp).
  let native_max: u16 = ((1u32 << SRC_BITS) - 1) as u16;
  let stream = rgba_filter_stream_u16.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // Native-depth u16 RGBA — copy the binned 4-channel row (clamped to the
    // native max for a sub-16-bit source).
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let out = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      if SRC_BITS < 16 {
        for (d, &s) in out.iter_mut().zip(binned.iter()) {
          *d = s.min(native_max);
        }
      } else {
        out.copy_from_slice(binned);
      }
    }
    // Native-depth u16 RGB — drop alpha from the binned row (clamped for a
    // sub-16-bit source).
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let out = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      for (rgb_px, rgba_px) in out.chunks_exact_mut(3).zip(binned.chunks_exact(4)) {
        if SRC_BITS < 16 {
          rgb_px[0] = rgba_px[0].min(native_max);
          rgb_px[1] = rgba_px[1].min(native_max);
          rgb_px[2] = rgba_px[2].min(native_max);
        } else {
          rgb_px.copy_from_slice(&rgba_px[..3]);
        }
      }
    }
    // Clamped drop-alpha native u16 RGB row — the single canonical source for
    // both the native luma_u16 and the u8 narrowing. Resolved once whenever
    // either is requested; a clipped-high edge is clamped to the native max
    // here so a sub-16-bit signed-kernel overshoot never wraps downstream.
    if need_native_rgb {
      let rgb_row = &mut color_rgb[..3 * ow];
      for (out_px, in_px) in rgb_row.chunks_exact_mut(3).zip(binned.chunks_exact(4)) {
        if SRC_BITS < 16 {
          out_px[0] = in_px[0].min(native_max);
          out_px[1] = in_px[1].min(native_max);
          out_px[2] = in_px[2].min(native_max);
        } else {
          out_px.copy_from_slice(&in_px[..3]);
        }
      }
    }
    // Native-precision luma_u16 (`GbrapN`): run the same
    // `rgb_to_luma_u16_native_row` the area / direct paths use on the clamped
    // native RGB, so the resampled luma_u16 is byte-identical to a direct
    // GbrapN conversion of the binned frame (full native depth, not the
    // narrowed 0..255 flavor). It clamps each channel to the native max
    // internally, a no-op since `color_rgb` is already clamped above.
    if native_luma && let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::rgb_to_luma_u16_native_row(
        &color_rgb[..3 * ow],
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        SRC_BITS,
      );
    }
    if need_narrow {
      // Narrow the clamped drop-alpha native RGB `>> (SRC_BITS - 8)`; a
      // clipped-high edge narrows to `255`, never a wrap.
      let rgb_row = &color_rgb[..3 * ow];
      let nrow = &mut narrow[..3 * ow];
      for (d, &s) in nrow.iter_mut().zip(rgb_row.iter()) {
        *d = (s >> (SRC_BITS - 8)) as u8;
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = rgba.as_deref_mut() {
        // Narrow all four filtered channels (α `>> (SRC_BITS - 8)` too) —
        // alpha is a real filtered channel here, never forced opaque (the
        // 4-channel filter's whole point). Mirrors the area 4-channel path.
        // Clamp to the native max before the shift for a sub-16-bit source so
        // a signed-kernel overshoot clips to `255` (no-op at 16-bit).
        let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        for (d, px) in dst.chunks_exact_mut(4).zip(binned.chunks_exact(4)) {
          if SRC_BITS < 16 {
            d[0] = (px[0].min(native_max) >> (SRC_BITS - 8)) as u8;
            d[1] = (px[1].min(native_max) >> (SRC_BITS - 8)) as u8;
            d[2] = (px[2].min(native_max) >> (SRC_BITS - 8)) as u8;
            d[3] = (px[3].min(native_max) >> (SRC_BITS - 8)) as u8;
          } else {
            d[0] = (px[0] >> (SRC_BITS - 8)) as u8;
            d[1] = (px[1] >> (SRC_BITS - 8)) as u8;
            d[2] = (px[2] >> (SRC_BITS - 8)) as u8;
            d[3] = (px[3] >> (SRC_BITS - 8)) as u8;
          }
        }
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      // Narrowed luma_u16 (`Rgba64` / `Bgra64`): 8-bit-precision Y' from the
      // narrowed RGB, zero-extended — byte-identical to those formats' direct
      // path. Skipped under `NATIVE_LUMA16` (`GbrapN`), where luma_u16 came
      // from the native path above.
      if narrowed_luma_u16 && let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Source-width `u16` RGB staging for high-bit packed-RGB resampling:
/// the wire row converts here before feeding [`AreaStream<u16>`]. Grows
/// `scratch` to `3 * width` `u16` under the planner's
/// recoverable-allocation contract. Mirrors [`source_rgb_scratch`] for
/// the 16-bit element path.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
pub(super) fn source_rgb_u16_scratch<'s>(
  scratch: &'s mut Vec<u16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u16], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a `u16` source-staging OOM AFTER a successful stream
  // build leaves the stream field `None` AND `resample_outputs` uncommitted
  // (retryable). The `u16` twin of the `source_rgb_scratch` failpoint, so a 3-ch
  // `u16` atomicity test can fail this specific pre-commit grow.
  #[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
  if FORCE_SOURCE_RGB_U16_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0);
  }
  Ok(&mut scratch[..row])
}

// Single-shot source-RGB-`u16`-scratch grow failpoint for the RFC #238
// tail-collapse transactional-preflight tests. Gated on `any(rgb, gbr)` (the
// high-bit packed-RGB / GBR families that stage a source-width `u16` RGB row
// pre-commit); `source_rgb_u16_scratch` consults it under the same gate. Separate
// from the `try_box` box-alloc failpoint so a 3-channel `u16` test can fail the
// SCRATCH grow specifically (after a successful stream construction).
#[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
std::thread_local! {
  static FORCE_SOURCE_RGB_U16_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_rgb_u16_scratch`]
/// grow refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape of [`packed_rgb_u16_resample`] (a source-staging
/// OOM after a successful stream build leaves the stream field `None` AND
/// `resample_outputs` uncommitted, so the row is retryable). Test-only.
#[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
pub(crate) fn arm_source_rgb_u16_scratch_failure() {
  FORCE_SOURCE_RGB_U16_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Feeds the prepared source-width `u16` RGB row into the (already
/// sequence-checked) stream and derives every attached output from each
/// finalized output row. Binning runs at the source's native depth
/// (`SRC_BITS` active bits per `u16` element); the `rgb_u16` /
/// `rgba_u16` outputs copy it directly, while the u8 and `luma_u16`
/// outputs derive from a single `>> (SRC_BITS - 8)` narrowing — the same
/// source-of-truth ordering the direct path uses (luma / luma_u16 / hsv
/// all read the narrowed u8 RGB). `SRC_BITS` is `16` for the packed
/// `Rgb48` / `Bgr48` sources (whose elements are full-range u16) and the
/// source bit depth for the high-bit planar GBR sources (`Gbrp9` … 14
/// carry fewer than 16 active bits, so their narrowing shift and opaque
/// `rgba_u16` alpha both track `SRC_BITS`, not a hard-coded 16).
/// `narrow_scratch` is sized to the out-width u8 RGB row only when one of
/// those narrowed outputs is attached, so a native-u16-only sink neither
/// grows it nor risks its allocation failure.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u16_resample_emit<const SRC_BITS: u32, const NATIVE_LUMA16: bool>(
  stream: &mut impl crate::resample::RowResampler<u16>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_u16: &[u16],
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16]"
    )
  };
  let ow = plan.out_w();
  // The u8 / luma_u16 outputs derive from a `>> (SRC_BITS - 8)`
  // narrowing of the binned row; a native-u16-only sink (only rgb_u16 /
  // rgba_u16) never touches it, so the out-width u8 scratch is sized —
  // and its allocation failure risked — only when one of those outputs
  // is attached. The predicate gates both the sizing here and the use in
  // the closure, so they cannot drift.
  let need_narrow = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || hsv.is_some()
    || (!NATIVE_LUMA16 && luma_u16.is_some());
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  // A signed filter kernel (CatmullRom / Lanczos3) overshoots a legal
  // edge, so a finalized `binned` sample can exceed the source's native
  // max `(1 << SRC_BITS) - 1` even though the `FilterStream` clamps it to
  // the full `u16` range. For a sub-16-bit source that overshoot is
  // out-of-contract: the native-depth u16 outputs would publish a value
  // above the documented range, and the u8 narrowing (`>> (SRC_BITS - 8)`)
  // would wrap a clipped-high edge to a small value instead of `255`. So
  // for `SRC_BITS < 16` every binned sample is clamped to the native max
  // before any u16 copy, RGBA expansion, native luma, or u8 narrowing.
  // For `SRC_BITS == 16` the native max is the u16 max, so this is a
  // no-op (`Rgb48` / `Bgr48` are unaffected); the area path never
  // overshoots, so it is a value no-op there too.
  let native_max: u16 = ((1u32 << SRC_BITS) - 1) as u16;
  stream.feed_row(idx, src_u16, use_simd, |oy, binned| {
    // Native-depth u16 outputs copy the binned row (clamped to the native
    // max for a sub-16-bit source).
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let out = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      if SRC_BITS < 16 {
        for (d, &s) in out.iter_mut().zip(binned.iter()) {
          *d = s.min(native_max);
        }
      } else {
        out.copy_from_slice(binned);
      }
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let out = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      if SRC_BITS < 16 {
        // Clamping twin of `expand_rgb_u16_to_rgba_u16_row` (opaque alpha
        // is the native max, which equals its `(1 << BITS) - 1`).
        for (rgba_px, rgb_px) in out.chunks_exact_mut(4).zip(binned.chunks_exact(3)) {
          rgba_px[0] = rgb_px[0].min(native_max);
          rgba_px[1] = rgb_px[1].min(native_max);
          rgba_px[2] = rgb_px[2].min(native_max);
          rgba_px[3] = native_max;
        }
      } else {
        crate::row::expand_rgb_u16_to_rgba_u16_row::<SRC_BITS>(binned, out, ow);
      }
    }
    // Native-precision `luma_u16`: derive directly from the native-depth
    // binned RGB, byte-identical to the direct
    // `gbr_to_luma_u16_high_bit_row` path. Only the high-bit-GBR callers
    // set `NATIVE_LUMA16`; the `Rgb48` / `Bgr48` callers leave it false
    // and take the narrowed `luma_u16` in the `need_narrow` block below.
    // `rgb_to_luma_u16_native_row` clamps each input channel to the
    // native max internally, so a filter overshoot is clipped before the
    // luma sum (a no-op for the in-range area / direct callers).
    if NATIVE_LUMA16 && let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::rgb_to_luma_u16_native_row(
        binned,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        SRC_BITS,
      );
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      // Clamp to the native max before the narrowing shift so a sub-16-bit
      // filter overshoot clips to `255` instead of wrapping (no-op for
      // 16-bit and for the in-range area path).
      if SRC_BITS < 16 {
        for (d, &s) in nrow.iter_mut().zip(binned.iter()) {
          *d = (s.min(native_max) >> (SRC_BITS - 8)) as u8;
        }
      } else {
        for (d, &s) in nrow.iter_mut().zip(binned.iter()) {
          *d = (s >> (SRC_BITS - 8)) as u8;
        }
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::expand_rgb_to_rgba_row(nrow, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if !NATIVE_LUMA16 && let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// RFC #238 tail-collapse — the transactional high-bit packed-RGB **area**
/// resample, the u16 element-family twin of [`packed_rgb_resample`]. Folds the
/// split high-bit preflight + area-stream build + [`packed_rgb_u16_resample_emit`]
/// sequence into one call behind the shared compare-before / commit-after
/// preflight
/// ([`resample_preflight_check_only`](resample_preflight_check_only)),
/// mirroring [`packed_rgb_resample`]: the missing area stream is built into a
/// LOCAL and every fallible pre-feed scratch (the source-width `u16` staging row
/// and — when a narrowed output is attached — the out-width `u8` narrow row the
/// emit reads) is grown with NO field mutation, so the stream field insert and the
/// output-set freeze COMMIT TOGETHER only after they all succeed. A first-row
/// allocation failure therefore leaves `rgb_stream_u16` `None` AND
/// `resample_outputs` uncommitted (never a partial commit) — state-atomic and
/// retryable even with a changed output attachment (the #180 discipline). A pure
/// reordering of the pre-collapse split path: it emits the identical bytes, only
/// deferring the field mutations past the fallible allocations.
///
/// `convert` stages the source-width canonical host-native `u16` RGB row into the
/// grown `rgb_scratch_u16` (each format's `*_to_rgb_u16` kernel); that fallible
/// growth is hoisted ahead of the freeze too. `SRC_BITS` / `NATIVE_LUMA16` thread
/// straight to [`packed_rgb_u16_resample_emit`] (the source's active depth drives
/// the u8 `>> (SRC_BITS - 8)` narrowing; `NATIVE_LUMA16` selects the native-depth
/// versus narrowed `luma_u16` flavour), so every output byte matches the
/// pre-collapse emit.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u16_resample<const SRC_BITS: u32, const NATIVE_LUMA16: bool>(
  rgb_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16]"
    )
  };
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the area stream is the sole
  // per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(rgb_stream_u16.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream alloc + scratch growth below succeed, so a first-row
  // allocation failure leaves `resample_outputs` uncommitted (retryable with a
  // changed output attachment). A no-output sink short-circuits to a no-op.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Area-only sink: reject a filter plan before any allocation (mirrors the
  // pre-collapse `packed_rgb_u16_resample_stream` guard). Runs AFTER the
  // compare-only preflight, so a no-output sink stays a no-op even under a filter
  // plan.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  // The emit narrows the binned row `>> (SRC_BITS - 8)` for its u8 / narrowed-
  // `luma_u16` outputs through the out-width `narrow_scratch`; pre-grow it here
  // with the emit's exact predicate so that fallible growth precedes the freeze
  // too (the emit's own grow then re-runs as a no-op). A native-`u16`-only sink
  // neither grows it nor risks its allocation failure.
  let need_narrow = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || hsv.is_some()
    || (!NATIVE_LUMA16 && luma_u16.is_some());
  // Build the missing area stream into a LOCAL and grow the staging scratches —
  // every fallible pre-feed step — with NO field mutation, so the stream field
  // insert and the freeze commit TOGETHER only after they all succeed (the
  // `planar_dual_resample` atomic shape: a scratch OOM leaves `rgb_stream_u16`
  // `None` AND `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgb_stream_u16.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
  if need_narrow {
    source_rgb_scratch(narrow_scratch, plan.out_w(), plan)?;
  }
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare above already caught any later
  // mid-frame output change; the freeze stores the snapshot on the first row).
  if let Some(s) = new_stream {
    *rgb_stream_u16 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Stage the source-width canonical native-`u16` RGB row (the caller's conversion
  // into the grown scratch) and run the shared per-output emit (its narrow grow
  // re-runs as a no-op after the pre-commit grow above).
  let stream = rgb_stream_u16.as_mut().expect("created above");
  let src_u16 = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
  convert(src_u16);
  packed_rgb_u16_resample_emit::<SRC_BITS, NATIVE_LUMA16>(
    stream,
    plan,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    hsv,
    src_u16,
    narrow_scratch,
    matrix,
    full_range,
    idx,
    use_simd,
  )
}

/// RFC #238 tail-collapse — the transactional high-bit packed-RGB **filter**
/// resample, the [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_rgb_u16_resample`] over a 3-channel `u16`
/// [`FilterStream`](crate::resample::FilterStream). Identical compare-before /
/// commit-after discipline (the filter stream is built into a local and its field
/// insert + the freeze + the pre-grown narrow scratch commit together only after
/// every allocation) and the identical [`packed_rgb_u16_resample_emit`] tail (the
/// emit is generic over the resampler kind), so the resampled colour equals the
/// equivalent area resample up to the signed kernel weights while remaining
/// byte-identical to the pre-collapse split filter path.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u16_resample_filter<const SRC_BITS: u32, const NATIVE_LUMA16: bool>(
  rgb_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16]"
    )
  };
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_rgb_u16_resample`.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Single-kernel filter: reject a BICUBLIN plan before any allocation (its chroma
  // windows are read only by the `Yuv420p` per-plane route, so filtering every
  // channel with the luma kernel would be wrong output).
  plan.ensure_single_kernel_filter()?;
  // Pre-grow the emit's out-width narrow scratch with its exact predicate so that
  // fallible growth precedes the freeze too (see `packed_rgb_u16_resample`).
  let need_narrow = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || hsv.is_some()
    || (!NATIVE_LUMA16 && luma_u16.is_some());
  // Build the missing filter stream into a LOCAL and grow the staging scratches —
  // every fallible pre-feed step — with NO field mutation, so the stream field
  // insert and the freeze commit TOGETHER (the `planar_dual_resample` atomic
  // shape: a scratch OOM leaves `rgb_filter_stream_u16` `None` AND
  // `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgb_filter_stream_u16.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
  if need_narrow {
    source_rgb_scratch(narrow_scratch, plan.out_w(), plan)?;
  }
  if let Some(s) = new_stream {
    *rgb_filter_stream_u16 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgb_filter_stream_u16.as_mut().expect("created above");
  let src_u16 = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
  convert(src_u16);
  packed_rgb_u16_resample_emit::<SRC_BITS, NATIVE_LUMA16>(
    stream,
    plan,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    hsv,
    src_u16,
    narrow_scratch,
    matrix,
    full_range,
    idx,
    use_simd,
  )
}

// ===== Native-`u32` packed-RGB(A) resample tier (0-ULP, issue #289) ==========
//
// The narrow-first u16 tiers above stage each wire `u32` as its `>> 16` narrow
// **before** binning, so a downscaled / filtered output is the mean of the
// narrowed samples — within 1 LSB of an exact `u32`-domain mean. These tiers
// instead stage the source as **host-native `u32`** (no narrow), bin at full
// `u32` precision (`AreaStream<u32>`'s `u128` accumulators / `FilterStream<u32>`'s
// `f64` domain), and narrow only in the per-output derive — exact 0-ULP for
// both `full_range` true and false. Because narrowing a binned `u32` `>> 16` to
// `u16` and deriving from there is byte-identical to the direct convert of the
// binned `u32` (the direct path itself narrows `>> 16` first), the derive below
// is the `SRC_BITS = 16` specialization of the u16 emit — every native-max
// clamp branch is a no-op at 16 bits and is dropped.

/// Source-width host-native `u32` RGB staging (`3 * width`) for the
/// native-`u32` packed-RGB resample tier. The `u32` twin of
/// [`source_rgb_u16_scratch`].
#[cfg(any(feature = "rgb", feature = "gbr"))]
pub(super) fn source_rgb_u32_scratch<'s>(
  scratch: &'s mut Vec<u32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u32], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a `u32` source-staging OOM AFTER a successful stream
  // build leaves the stream field `None` AND `resample_outputs` uncommitted
  // (retryable). The `u32` twin of the `source_rgb_u16_scratch` failpoint, so a
  // 3-ch `u32` atomicity test can fail this specific pre-commit grow.
  #[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
  if FORCE_SOURCE_RGB_U32_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0);
  }
  Ok(&mut scratch[..row])
}

// Single-shot source-RGB-`u32`-scratch grow failpoint for the RFC #238
// tail-collapse transactional-preflight tests. Gated on `any(rgb, gbr)` (the
// native-`u32` packed-RGB / GBR families that stage a source-width `u32` RGB row
// pre-commit); `source_rgb_u32_scratch` consults it under the same gate. Separate
// from the `try_box` box-alloc failpoint so a 3-channel `u32` test can fail the
// SCRATCH grow specifically (after a successful stream construction).
#[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
std::thread_local! {
  static FORCE_SOURCE_RGB_U32_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_rgb_u32_scratch`]
/// grow refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape of [`packed_rgb_u32_resample`] (a source-staging
/// OOM after a successful stream build leaves the stream field `None` AND
/// `resample_outputs` uncommitted, so the row is retryable). Test-only.
#[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
pub(crate) fn arm_source_rgb_u32_scratch_failure() {
  FORCE_SOURCE_RGB_U32_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Source-width host-native canonical `R, G, B, A` `u32` staging (`4 * width`)
/// for the native-`u32` packed-RGBA resample tier. The `u32` twin of
/// [`source_rgba_u16_scratch`].
#[cfg(any(feature = "rgb", feature = "gbr"))]
pub(super) fn source_rgba_u32_scratch<'s>(
  scratch: &'s mut Vec<u32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u32], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a 4-channel `u32` scratch OOM AFTER a successful
  // stream build leaves the stream field `None` AND `resample_outputs` uncommitted
  // (retryable). The 4-channel twin of the `source_rgb_u32_scratch` failpoint, so a
  // packed-RGBA `u32` atomicity test can fail this specific pre-commit grow.
  #[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
  if FORCE_SOURCE_RGBA_U32_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      4,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0);
  }
  Ok(&mut scratch[..row])
}

// Single-shot source-RGBA-`u32`-scratch grow failpoint for the RFC #238
// tail-collapse transactional-preflight tests. Gated on `any(rgb, gbr)` (the
// native-`u32` packed-RGBA / `Gbrap32` families that stage a source-width `u32`
// RGBA row pre-commit); `source_rgba_u32_scratch` consults it under the same gate.
// Separate from the `try_box` box-alloc failpoint so a 4-channel `u32` test can
// fail the SCRATCH grow specifically (after a successful stream construction,
// before the stream-field insert).
#[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
std::thread_local! {
  static FORCE_SOURCE_RGBA_U32_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_rgba_u32_scratch`]
/// grow refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape of [`packed_rgba_u32_resample`] (a source-staging
/// OOM after a successful stream build leaves the stream field `None` AND
/// `resample_outputs` uncommitted, so the row is retryable). Test-only.
#[cfg(all(test, feature = "std", any(feature = "rgb", feature = "gbr")))]
pub(crate) fn arm_source_rgba_u32_scratch_failure() {
  FORCE_SOURCE_RGBA_U32_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// RFC #238 tail-collapse — the transactional native-`u32` packed-RGB **area**
/// resample, the `u32` element-family twin of [`packed_rgb_u16_resample`]. Folds
/// the split preflight + area-stream build + [`packed_rgb_u32_resample_emit`]
/// sequence into one call behind the shared compare-before / commit-after
/// preflight
/// ([`resample_preflight_check_only`](resample_preflight_check_only)):
/// the missing area stream is built into a LOCAL and every fallible pre-feed
/// scratch (the source-width `u32` staging row, the always-sized out-width binned
/// `u16` row the emit narrows into, and — when a narrowed output is attached — the
/// out-width `u8` narrow row) is grown with NO field mutation, so the stream field
/// insert and the output-set freeze COMMIT TOGETHER only after they all succeed. A
/// first-row allocation failure therefore leaves `rgb_stream_u32` `None` AND
/// `resample_outputs` uncommitted (never a partial commit) — state-atomic and
/// retryable even with a changed output attachment (the #180 discipline). A pure
/// reordering of the pre-collapse split path: identical bytes, only deferring the
/// field mutations past the fallible allocations.
///
/// `convert` stages the source-width canonical host-native `u32` RGB row into the
/// grown `rgb_scratch_u32` (each format's `*_to_rgb_u32` kernel). `NATIVE_LUMA16`
/// threads straight to [`packed_rgb_u32_resample_emit`] (native-depth vs narrowed
/// `luma_u16`), so every output byte matches the pre-collapse emit — full `u32`
/// binning, narrowed only per output (0-ULP, issue #289).
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u32_resample<const NATIVE_LUMA16: bool>(
  rgb_stream_u32: &mut Option<std::boxed::Box<crate::resample::AreaStream<u32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch_u32: &mut Vec<u32>,
  binned_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert: impl FnOnce(&mut [u32]),
) -> Result<(), MixedSinkerError> {
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the area stream is the sole
  // per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(rgb_stream_u32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream alloc + scratch growth below succeed, so a first-row
  // allocation failure leaves `resample_outputs` uncommitted (retryable with a
  // changed output attachment). A no-output sink short-circuits to a no-op.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Area-only sink: reject a filter plan before any allocation (mirrors the
  // pre-collapse `packed_rgb_u32_resample_stream` guard). Runs AFTER the
  // compare-only preflight, so a no-output sink stays a no-op even under a filter
  // plan.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  // The emit narrows the binned `u32` row for its u8 / narrowed-`luma_u16` outputs
  // through the out-width `narrow_scratch`; pre-grow it here with the emit's exact
  // predicate so that fallible growth precedes the freeze too (the emit's own grow
  // then re-runs as a no-op). A native-`u16`-only sink neither grows it nor risks
  // its allocation failure.
  let need_narrow = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || hsv.is_some()
    || (!NATIVE_LUMA16 && luma_u16.is_some());
  // Build the missing area stream into a LOCAL and grow the staging scratches —
  // every fallible pre-feed step — with NO field mutation, so the stream field
  // insert and the freeze commit TOGETHER only after they all succeed (the
  // `planar_dual_resample` atomic shape: a scratch OOM leaves `rgb_stream_u32`
  // `None` AND `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgb_stream_u32.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  source_rgb_u32_scratch(rgb_scratch_u32, w, plan)?;
  source_rgb_u16_scratch(binned_scratch_u16, plan.out_w(), plan)?;
  if need_narrow {
    source_rgb_scratch(narrow_scratch, plan.out_w(), plan)?;
  }
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare above already caught any later
  // mid-frame output change; the freeze stores the snapshot on the first row).
  if let Some(s) = new_stream {
    *rgb_stream_u32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Stage the source-width canonical host-native `u32` RGB row (the caller's
  // conversion into the grown scratch) and run the shared per-output emit (its
  // binned / narrow grows re-run as no-ops after the pre-commit grows above).
  let stream = rgb_stream_u32.as_mut().expect("created above");
  let src_u32 = source_rgb_u32_scratch(rgb_scratch_u32, w, plan)?;
  convert(src_u32);
  packed_rgb_u32_resample_emit::<NATIVE_LUMA16>(
    stream,
    plan,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    hsv,
    src_u32,
    binned_scratch_u16,
    narrow_scratch,
    matrix,
    full_range,
    idx,
    use_simd,
  )
}

/// RFC #238 tail-collapse — the transactional native-`u32` packed-RGB **filter**
/// resample, the [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_rgb_u32_resample`] over a 3-channel `u32`
/// [`FilterStream`](crate::resample::FilterStream). Identical compare-before /
/// commit-after discipline (the filter stream is built into a local and its field
/// insert + the freeze + the pre-grown binned / narrow scratches commit together
/// only after every allocation) and the identical [`packed_rgb_u32_resample_emit`]
/// tail (the emit is generic over the resampler kind), so the resampled colour
/// equals the equivalent area resample up to the signed kernel weights while
/// remaining byte-identical to the pre-collapse split filter path.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u32_resample_filter<const NATIVE_LUMA16: bool>(
  rgb_filter_stream_u32: &mut Option<std::boxed::Box<crate::resample::FilterStream<u32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch_u32: &mut Vec<u32>,
  binned_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert: impl FnOnce(&mut [u32]),
) -> Result<(), MixedSinkerError> {
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgb_filter_stream_u32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_rgb_u32_resample`.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Single-kernel filter: reject a BICUBLIN plan before any allocation (its chroma
  // windows are read only by the `Yuv420p` per-plane route, so filtering every
  // channel with the luma kernel would be wrong output).
  plan.ensure_single_kernel_filter()?;
  // Pre-grow the emit's out-width narrow scratch with its exact predicate so that
  // fallible growth precedes the freeze too (see `packed_rgb_u32_resample`).
  let need_narrow = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || hsv.is_some()
    || (!NATIVE_LUMA16 && luma_u16.is_some());
  // Build the missing filter stream into a LOCAL and grow the staging scratches —
  // every fallible pre-feed step — with NO field mutation, so the stream field
  // insert and the freeze commit TOGETHER (the `planar_dual_resample` atomic
  // shape: a scratch OOM leaves `rgb_filter_stream_u32` `None` AND
  // `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgb_filter_stream_u32.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  source_rgb_u32_scratch(rgb_scratch_u32, w, plan)?;
  source_rgb_u16_scratch(binned_scratch_u16, plan.out_w(), plan)?;
  if need_narrow {
    source_rgb_scratch(narrow_scratch, plan.out_w(), plan)?;
  }
  if let Some(s) = new_stream {
    *rgb_filter_stream_u32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgb_filter_stream_u32.as_mut().expect("created above");
  let src_u32 = source_rgb_u32_scratch(rgb_scratch_u32, w, plan)?;
  convert(src_u32);
  packed_rgb_u32_resample_emit::<NATIVE_LUMA16>(
    stream,
    plan,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    hsv,
    src_u32,
    binned_scratch_u16,
    narrow_scratch,
    matrix,
    full_range,
    idx,
    use_simd,
  )
}

/// Feeds the prepared source-width host-native `u32` RGB row into the (already
/// sequence-checked) `u32` stream and derives every attached output from each
/// finalized output row. Binning runs at full `u32` precision; each output
/// narrows only afterwards — `rgb_u16` / `rgba_u16` take `binned >> 16`, the u8
/// outputs `binned >> 24` (i.e. the `>> 16` u16 then `>> 8`), and luma / hsv
/// derive from the narrowed u8 RGB — so the resampled output is byte-identical
/// to a direct convert of the `u32`-binned frame (0-ULP, both ranges).
/// `NATIVE_LUMA16` selects the `luma_u16` source-of-truth (native-depth Q15
/// from the binned u16 RGB for `Gbrap32`, else the narrowed 8-bit Y'), mirroring
/// the u16 emit's `NATIVE_LUMA16`. The single source-of-truth narrowed-binned
/// u16 RGB row reuses `binned_scratch_u16` (the u32 path's u16 streams are never
/// created for these formats, so it is free); the out-width u8 narrow reuses
/// `narrow_scratch` and is sized only when a narrowed output is attached.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u32_resample_emit<const NATIVE_LUMA16: bool>(
  stream: &mut impl crate::resample::RowResampler<u32>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_u32: &[u32],
  binned_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let narrowed_luma_u16 = !NATIVE_LUMA16 && luma_u16.is_some();
  let need_narrow =
    rgb.is_some() || rgba.is_some() || luma.is_some() || narrowed_luma_u16 || hsv.is_some();
  // The narrowed-binned u16 RGB row is the single source of truth for every
  // narrowed output; it is always sized (any output reads it). The out-width u8
  // RGB row is sized only when a narrowed output is attached.
  let binned16 = source_rgb_u16_scratch(binned_scratch_u16, ow, plan)?;
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_u32, use_simd, |oy, binned| {
    // Narrow the binned `u32` RGB `>> 16` to native u16 — the 0-ULP narrow,
    // after binning. This row is the source of truth for every output.
    let b16 = &mut binned16[..3 * ow];
    for (d, &s) in b16.iter_mut().zip(binned.iter()) {
      *d = (s >> 16) as u16;
    }
    let b16: &[u16] = b16;
    if let Some(buf) = rgb_u16.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(b16);
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      crate::row::expand_rgb_u16_to_rgba_u16_row::<16>(
        b16,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
      );
    }
    if NATIVE_LUMA16 && let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::rgb_to_luma_u16_native_row(
        b16,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        16,
      );
    }
    if need_narrow {
      // The u8 RGB is the binned `u32 >> 24` = `(u32 >> 16) >> 8`.
      let nrow = &mut narrow[..3 * ow];
      for (d, &s) in nrow.iter_mut().zip(b16.iter()) {
        *d = (s >> 8) as u8;
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::expand_rgb_to_rgba_row(nrow, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if narrowed_luma_u16 && let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Premultiplies one canonical `R, G, B, A` `u32` row in place at full `u32`
/// precision: each color channel becomes `round(c * α / u32::MAX)` (`u64`
/// intermediates); α is unchanged. The `u32` twin of
/// [`premultiply_rgba_row_in_place`] — the generic `u16` helper's `v * a`
/// overflows `u32`, so the `u32` tier needs the widened multiply.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn premultiply_rgba_u32_row_in_place(row: &mut [u32], width: usize) {
  const MAX: u64 = u32::MAX as u64;
  const HALF: u64 = MAX / 2;
  for px in row[..width * 4].chunks_exact_mut(4) {
    let a = u64::from(px[3]);
    for c in &mut px[..3] {
      let v = u64::from(*c);
      // `pm <= MAX` (since `v, a <= MAX`), so the cast back never truncates.
      *c = ((v * a + HALF) / MAX) as u32;
    }
  }
}

/// Un-premultiplies one binned canonical `R, G, B, A` `u32` row into a straight
/// `u32` RGBA destination (α copied through) at full `u32` precision. The `u32`
/// twin of [`unpremultiply_binned_rgba_into`].
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpremultiply_binned_rgba_u32_into(binned: &[u32], dst: &mut [u32], width: usize) {
  const MAX: u64 = u32::MAX as u64;
  for (out_px, in_px) in dst[..width * 4]
    .chunks_exact_mut(4)
    .zip(binned[..width * 4].chunks_exact(4))
  {
    let a = u64::from(in_px[3]);
    for c in 0..3 {
      let pm = u64::from(in_px[c]);
      // `checked_div` is `None` exactly when `α == 0` → a fully-transparent
      // binned pixel exposes no color (cannot bleed); else round-half-up + clamp.
      out_px[c] = (pm * MAX + a / 2).checked_div(a).map_or(0, |q| q.min(MAX)) as u32;
    }
    out_px[3] = in_px[3];
  }
}

/// Per-finalized-output-row derive shared by the native-`u32` packed-RGBA area
/// and filter tails: given the straight binned `u32` RGBA row `straight` (an
/// un-premultiply of the binned premult row on the area path, or the binned row
/// itself on the straight / filter paths), narrows `>> 16` to native u16 and
/// emits every attached output — the `SRC_BITS = 16` specialization of the u16
/// 4-channel emit. `color16` / `rgb16` / `narrow` are out-width scratch resolved
/// by the caller (the u32 path's u16 streams are never created for these
/// formats, so the reused u16 scratch is free).
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
fn derive_packed_rgba_u32_outputs<const NATIVE_LUMA16: bool>(
  oy: usize,
  ow: usize,
  straight: &[u32],
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  color16: &mut [u16],
  rgb16: &mut [u16],
  narrow: &mut [u8],
  need_narrow: bool,
  native_luma: bool,
  narrowed_luma_u16: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Narrow the straight binned `u32` RGBA `>> 16` to native u16 — the single
  // source of truth for every output.
  let c16 = &mut color16[..4 * ow];
  for (d, &s) in c16.iter_mut().zip(straight.iter()) {
    *d = (s >> 16) as u16;
  }
  let c16: &[u16] = c16;
  if let Some(buf) = rgba_u16.as_deref_mut() {
    buf[oy * 4 * ow..(oy + 1) * 4 * ow].copy_from_slice(c16);
  }
  if let Some(buf) = rgb_u16.as_deref_mut() {
    drop_alpha_rgba_to_rgb_row(c16, &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow], ow);
  }
  if native_luma && let Some(buf) = luma_u16.as_deref_mut() {
    let rgb_row = &mut rgb16[..3 * ow];
    drop_alpha_rgba_to_rgb_row(c16, rgb_row, ow);
    crate::row::rgb_to_luma_u16_native_row(
      rgb_row,
      &mut buf[oy * ow..(oy + 1) * ow],
      ow,
      matrix,
      full_range,
      16,
    );
  }
  if need_narrow {
    // u8 RGB = straight `u32 >> 24` = `(u32 >> 16) >> 8`, drop α.
    let nrow = &mut narrow[..3 * ow];
    for (d, px) in nrow.chunks_exact_mut(3).zip(c16.chunks_exact(4)) {
      d[0] = (px[0] >> 8) as u8;
      d[1] = (px[1] >> 8) as u8;
      d[2] = (px[2] >> 8) as u8;
    }
    if let Some(buf) = rgb.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
    }
    if let Some(buf) = rgba.as_deref_mut() {
      // Narrow all four straight channels (α `>> 24` too) — a real binned α.
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      for (d, px) in dst.chunks_exact_mut(4).zip(c16.chunks_exact(4)) {
        d[0] = (px[0] >> 8) as u8;
        d[1] = (px[1] >> 8) as u8;
        d[2] = (px[2] >> 8) as u8;
        d[3] = (px[3] >> 8) as u8;
      }
    }
    if let Some(buf) = luma.as_deref_mut() {
      crate::row::rgb_to_luma_row(
        nrow,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        use_simd,
      );
    }
    if narrowed_luma_u16 && let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::rgb_to_luma_u16_row(
        nrow,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        use_simd,
      );
    }
    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      crate::row::rgb_to_hsv_row(
        nrow,
        &mut h[oy * ow..(oy + 1) * ow],
        &mut s[oy * ow..(oy + 1) * ow],
        &mut v[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
  }
}

/// Native-`u32` fused **area** downscale for the alpha-aware packed-RGBA
/// sources (`Rgba128` / `Gbrap32`) — the 0-ULP twin of
/// [`packed_rgba_u16_resample`] (area-only, no native-Y-luma group: these
/// sources' `luma_u16` is colour-derived). `convert_rgba_u32` stages the source
/// as a canonical host-native `R, G, B, A` `u32` row; under
/// [`AlphaMode::Premultiplied`] it is premultiplied at full `u32` precision
/// before binning and un-premultiplied per output row, so transparent pixels
/// never bleed. RFC #238 tail-collapse: the compare-only preflight commits the
/// output-set freeze only AFTER the fallible stream + scratch allocs, and the
/// stream is built into a LOCAL inserted alongside that freeze, so a first-row
/// allocation failure leaves `rgba_stream_u32` `None` AND `resample_outputs`
/// uncommitted (state-atomic + retryable, the #180 discipline) — byte-identical to
/// the pre-collapse split path.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_u32_resample<const NATIVE_LUMA16: bool>(
  rgba_stream_u32: &mut Option<std::boxed::Box<crate::resample::AreaStream<u32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch_u32: &mut Vec<u32>,
  color_scratch_u32: &mut Vec<u32>,
  color_scratch_u16: &mut Vec<u16>,
  rgb_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba_u32: impl FnOnce(&mut [u32]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (native-`u32` packed RGBA is not routed to the filter path):
  // reject a filter plan before any work.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the colour stream is the sole
  // per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(rgba_stream_u32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream alloc + scratch growth below succeed (RFC #238
  // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
  // uncommitted — retryable with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  let narrowed_luma_u16 = !NATIVE_LUMA16 && luma_u16.is_some();
  let need_narrow =
    rgb.is_some() || rgba.is_some() || luma.is_some() || narrowed_luma_u16 || hsv.is_some();
  let native_luma = NATIVE_LUMA16 && luma_u16.is_some();
  let premult = alpha_mode.is_premultiplied();
  // Build the missing stream into a LOCAL and grow every pre-feed scratch — every
  // fallible step — with NO field mutation, so the stream field insert and the
  // freeze commit TOGETHER only after they all succeed (the `planar_dual_resample`
  // atomic shape: a scratch OOM leaves `rgba_stream_u32` `None` AND
  // `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgba_stream_u32.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Grow every pre-feed scratch (the conditional premult / native-luma / narrow
  // rows and the always-sized out-width colour + source-width RGBA staging) BEFORE
  // the commit, so a scratch OOM likewise leaves the stream field `None` and
  // `resample_outputs` uncommitted.
  if premult {
    source_rgba_u32_scratch(color_scratch_u32, ow, plan)?;
  }
  source_rgba_u16_scratch(color_scratch_u16, ow, plan)?;
  if native_luma {
    source_rgb_u16_scratch(rgb_scratch_u16, ow, plan)?;
  }
  if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?;
  }
  source_rgba_u32_scratch(rgba_scratch_u32, w, plan)?;
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare-only preflight above already caught
  // any later mid-frame output change).
  if let Some(s) = new_stream {
    *rgba_stream_u32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Re-fetch the scratches (these grows re-run as no-ops after the pre-commit grows
  // above). Out-width straight `u32` RGBA, resolved per output row — needed only to
  // hold the un-premultiply on the premult path (straight mode reads `binned`).
  let color_u32: &mut [u32] = if premult {
    source_rgba_u32_scratch(color_scratch_u32, ow, plan)?
  } else {
    &mut []
  };
  // Out-width straight native-u16 RGBA (always) and drop-alpha RGB (native luma).
  let color16 = source_rgba_u16_scratch(color_scratch_u16, ow, plan)?;
  let rgb16: &mut [u16] = if native_luma {
    source_rgb_u16_scratch(rgb_scratch_u16, ow, plan)?
  } else {
    &mut []
  };
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_u32_scratch(rgba_scratch_u32, w, plan)?;
  convert_rgba_u32(src_rgba);
  if premult {
    premultiply_rgba_u32_row_in_place(src_rgba, w);
  }
  let stream = rgba_stream_u32.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    let straight: &[u32] = if premult {
      let c = &mut color_u32[..4 * ow];
      unpremultiply_binned_rgba_u32_into(binned, c, ow);
      c
    } else {
      binned
    };
    derive_packed_rgba_u32_outputs::<NATIVE_LUMA16>(
      oy,
      ow,
      straight,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      color16,
      rgb16,
      narrow,
      need_narrow,
      native_luma,
      narrowed_luma_u16,
      matrix,
      full_range,
      use_simd,
    );
  })?;
  Ok(())
}

/// Native-`u32` fused **filter** downscale for the alpha-aware packed-RGBA
/// sources (`Rgba128` / `Gbrap32`) — the 0-ULP twin of
/// [`packed_rgba_u16_filter_resample`]. PIL filters R, G, B, A independently
/// with no premultiplication, so the canonical host-native `R, G, B, A` `u32`
/// row bins through one 4-channel [`FilterStream<u32>`](crate::resample::FilterStream)
/// and every output narrows only afterwards (0-ULP). Premultiplied alpha has no
/// filter analogue — those frames stay on the area path; this is reached only
/// for straight alpha. RFC #238 tail-collapse: same compare-before / commit-after
/// discipline as [`packed_rgba_u32_resample`] — the stream builds into a LOCAL and
/// its field insert + the output-set freeze commit together only after every
/// fallible alloc, so a first-row failure leaves `rgba_filter_stream_u32` `None`
/// AND `resample_outputs` uncommitted (byte-identical to the pre-collapse path).
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_u32_filter_resample<const NATIVE_LUMA16: bool>(
  rgba_filter_stream_u32: &mut Option<std::boxed::Box<crate::resample::FilterStream<u32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch_u32: &mut Vec<u32>,
  color_scratch_u16: &mut Vec<u16>,
  rgb_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba_u32: impl FnOnce(&mut [u32]),
) -> Result<(), MixedSinkerError> {
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are read
  // only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the 4-channel filter stream is
  // the sole per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(rgba_filter_stream_u32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream alloc + scratch growth below succeed (RFC #238
  // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
  // uncommitted — retryable with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  let narrowed_luma_u16 = !NATIVE_LUMA16 && luma_u16.is_some();
  let need_narrow =
    rgb.is_some() || rgba.is_some() || luma.is_some() || narrowed_luma_u16 || hsv.is_some();
  let native_luma = NATIVE_LUMA16 && luma_u16.is_some();
  // Build the missing filter stream into a LOCAL and grow every pre-feed scratch —
  // every fallible step — with NO field mutation, so the stream field insert and the
  // freeze commit TOGETHER only after they all succeed (the `planar_dual_resample`
  // atomic shape: a scratch OOM leaves `rgba_filter_stream_u32` `None` AND
  // `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgba_filter_stream_u32.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Grow every pre-feed scratch (the always-sized out-width colour + source-width
  // RGBA staging, plus the conditional native-luma / narrow rows) BEFORE the commit,
  // so a scratch OOM likewise leaves the stream field `None` and `resample_outputs`
  // uncommitted.
  source_rgba_u16_scratch(color_scratch_u16, ow, plan)?;
  if native_luma {
    source_rgb_u16_scratch(rgb_scratch_u16, ow, plan)?;
  }
  if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?;
  }
  source_rgba_u32_scratch(rgba_scratch_u32, w, plan)?;
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare-only preflight above already caught
  // any later mid-frame output change).
  if let Some(s) = new_stream {
    *rgba_filter_stream_u32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Re-fetch the scratches (these grows re-run as no-ops after the pre-commit grows
  // above).
  let color16 = source_rgba_u16_scratch(color_scratch_u16, ow, plan)?;
  let rgb16: &mut [u16] = if native_luma {
    source_rgb_u16_scratch(rgb_scratch_u16, ow, plan)?
  } else {
    &mut []
  };
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_u32_scratch(rgba_scratch_u32, w, plan)?;
  convert_rgba_u32(src_rgba);
  let stream = rgba_filter_stream_u32.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // Filter is straight-alpha only; the binned row is the straight color.
    derive_packed_rgba_u32_outputs::<NATIVE_LUMA16>(
      oy,
      ow,
      binned,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      color16,
      rgb16,
      narrow,
      need_narrow,
      native_luma,
      narrowed_luma_u16,
      matrix,
      full_range,
      use_simd,
    );
  })?;
  Ok(())
}

/// Source-width canonical `R, G, B, A` `u8` staging for the packed
/// straight/premult RGBA resample tail. Grows `scratch` to `4 * width`
/// `u8` under the planner's recoverable-allocation contract. Mirrors
/// [`source_rgb_scratch`] for the 4-channel RGBA row.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "yuva",
  feature = "yuv-444-packed"
))]
pub(super) fn source_rgba_scratch<'s>(
  scratch: &'s mut Vec<u8>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u8], MixedSinkerError> {
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      4,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0);
  }
  Ok(&mut scratch[..row])
}

/// `u16` analogue of [`source_rgba_scratch`] — source-width canonical
/// `R, G, B, A` host-native `u16` staging for the high-bit packed RGBA
/// resample tail. Grows `scratch` to `4 * width` `u16`.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "yuva",
  feature = "yuv-444-packed"
))]
pub(super) fn source_rgba_u16_scratch<'s>(
  scratch: &'s mut Vec<u16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u16], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a 4-channel `u16` scratch OOM AFTER a successful
  // stream build leaves the stream field `None` AND `resample_outputs` uncommitted
  // (retryable). The 4-channel twin of the `source_rgb_u16_scratch` failpoint, so a
  // packed-RGBA `u16` atomicity test can fail this specific pre-commit grow.
  #[cfg(all(
    test,
    feature = "std",
    any(feature = "rgb", feature = "gbr", feature = "gray")
  ))]
  if FORCE_SOURCE_RGBA_U16_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      4,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0);
  }
  Ok(&mut scratch[..row])
}

// Single-shot source-RGBA-`u16`-scratch grow failpoint for the RFC #238
// tail-collapse transactional-preflight tests. Gated on `any(rgb, gbr, gray)` (the
// families that bin through the shared 4-channel `rgba_stream_u16` and stage a
// source-width `u16` RGBA row pre-commit); `source_rgba_u16_scratch` consults it
// under the same gate. Separate from the `try_box` box-alloc failpoint so a
// 4-channel `u16` test can fail the SCRATCH grow specifically (after a successful
// stream construction, before the stream-field insert).
#[cfg(all(
  test,
  feature = "std",
  any(feature = "rgb", feature = "gbr", feature = "gray")
))]
std::thread_local! {
  static FORCE_SOURCE_RGBA_U16_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_rgba_u16_scratch`]
/// grow refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape of [`packed_rgba_u16_resample`] (a source-staging
/// OOM after a successful stream build leaves the stream fields `None` AND
/// `resample_outputs` uncommitted, so the row is retryable). Test-only.
#[cfg(all(
  test,
  feature = "std",
  any(feature = "rgb", feature = "gbr", feature = "gray")
))]
pub(crate) fn arm_source_rgba_u16_scratch_failure() {
  FORCE_SOURCE_RGBA_U16_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Premultiplies one canonical `R, G, B, A` row in place: each color
/// channel becomes `round(c * α / MAX)`; α is left unchanged. The exact
/// integer op the [`AlphaMode::Premultiplied`] oracle mirrors, so the
/// binned-then-un-premultiplied output is byte-exact. `MAX` is `255` for
/// the u8 path and the source's native max for the u16 path.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "yuva",
  feature = "yuv-444-packed"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn premultiply_rgba_row_in_place<T>(row: &mut [T], width: usize, max: u32)
where
  T: Copy + TryFrom<u32> + Into<u32>,
{
  let half = max / 2;
  for px in row[..width * 4].chunks_exact_mut(4) {
    let a: u32 = px[3].into();
    for c in &mut px[..3] {
      let v: u32 = (*c).into();
      let pm = (v * a + half) / max;
      // `pm <= max` (since `v, a <= max`), so the cast back never fails.
      *c = T::try_from(pm).unwrap_or_else(|_| unreachable!("premultiplied value <= max"));
    }
  }
}

/// Un-premultiplied straight color channel for one premultiplied binned
/// value: `round(pm * MAX / α)` clamped to `MAX`, or `0` when `α == 0`
/// (a fully-transparent binned pixel exposes no color, so it cannot
/// bleed). The exact integer inverse of [`premultiply_rgba_row_in_place`].
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "yuva",
  feature = "yuv-444-packed"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpremultiply_channel(pm: u32, a: u32, max: u32) -> u32 {
  // `checked_div` yields `None` exactly when `α == 0`, which maps to a
  // zero straight channel (a fully-transparent binned pixel exposes no
  // color); otherwise round-half-up and clamp to `max`.
  (pm * max + a / 2).checked_div(a).map_or(0, |q| q.min(max))
}

/// Un-premultiplies one binned canonical `R, G, B, A` row into the
/// caller's straight-RGBA destination (α copied through). Applied per
/// finalized output row when binning premultiplied.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "yuva",
  feature = "yuv-444-packed"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpremultiply_binned_rgba_into<T>(binned: &[T], dst: &mut [T], width: usize, max: u32)
where
  T: Copy + TryFrom<u32> + Into<u32>,
{
  for (out_px, in_px) in dst[..width * 4]
    .chunks_exact_mut(4)
    .zip(binned[..width * 4].chunks_exact(4))
  {
    let a: u32 = in_px[3].into();
    for c in 0..3 {
      let straight = unpremultiply_channel(in_px[c].into(), a, max);
      out_px[c] =
        T::try_from(straight).unwrap_or_else(|_| unreachable!("un-premultiplied value <= max"));
    }
    out_px[3] = in_px[3];
  }
}

/// Un-premultiplies one binned canonical `R, G, B, A` row into a
/// straight **RGB** destination (α dropped) — the packed RGB the
/// luma / hsv kernels consume in premultiplied mode.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "yuva",
  feature = "yuv-444-packed"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpremultiply_binned_rgb_into<T>(binned: &[T], dst: &mut [T], width: usize, max: u32)
where
  T: Copy + TryFrom<u32> + Into<u32>,
{
  for (out_px, in_px) in dst[..width * 3]
    .chunks_exact_mut(3)
    .zip(binned[..width * 4].chunks_exact(4))
  {
    let a: u32 = in_px[3].into();
    for c in 0..3 {
      let straight = unpremultiply_channel(in_px[c].into(), a, max);
      out_px[c] =
        T::try_from(straight).unwrap_or_else(|_| unreachable!("un-premultiplied value <= max"));
    }
  }
}

/// Drops α from one canonical `R, G, B, A` row into a packed `R, G, B`
/// destination — the straight-mode RGB the luma / hsv kernels consume.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "yuva",
  feature = "yuv-444-packed"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn drop_alpha_rgba_to_rgb_row<T: Copy>(rgba: &[T], dst: &mut [T], width: usize) {
  for (out_px, in_px) in dst[..width * 3]
    .chunks_exact_mut(3)
    .zip(rgba[..width * 4].chunks_exact(4))
  {
    out_px.copy_from_slice(&in_px[..3]);
  }
}

/// Rejects a mid-frame [`AlphaMode`] change for a resampled packed-RGBA
/// frame. The mode is snapshotted in `begin_frame` (`frozen` is the mode
/// at frame start); the resample route and binning use it, so each sink
/// calls this **before** route selection in `process` and a later
/// differing live mode trips `ResampleOutputsChanged` — no row is fed
/// under a changed mode, and the snapshot is immune to out-of-sequence
/// rows since it is taken at the frame boundary, not the first row.
/// (`frozen` is `None` only before the first `begin_frame`, a contract
/// violation, which is likewise rejected.)
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "yuva",
  feature = "yuv-444-packed"
))]
pub(super) fn check_frozen_alpha_mode(
  frozen: Option<AlphaMode>,
  current: AlphaMode,
  idx: usize,
) -> Result<(), MixedSinkerError> {
  if frozen != Some(current) {
    return Err(MixedSinkerError::ResampleOutputsChanged(
      ResampleOutputsChanged::new(idx),
    ));
  }
  Ok(())
}

/// Row-stage fused downscale for the packed straight/premultiplied RGBA
/// 8-bit family (`Rgba` / `Bgra` / `Argb` / `Abgr`) — the alpha-aware
/// 4-channel analogue of the 3-channel [`packed_rgb_resample_emit`]
/// path. `convert_rgba` stages the source row as a canonical
/// source-width `R, G, B, A` u8 row (identity / swap / α-rotate per
/// format); this tail bins all four channels so resampled alpha is a
/// real area mean (the forced-opaque-`0xFF` bug the 3-channel path hit),
/// then per finalized output row emits rgba (the binned row),
/// rgb (drop α), luma / luma_u16 / hsv (from the binned RGB).
///
/// Under [`AlphaMode::Premultiplied`] the staged row is premultiplied
/// in place before binning and un-premultiplied per output row, so the
/// color outputs are alpha-weighted and transparent pixels never bleed.
///
/// `NATIVE_Y_LUMA` selects the `luma` / `luma_u16` source-of-truth:
/// - `false` (`Rgba` / `Bgra` / `Argb` / `Abgr` / `Gbrap8`): both are
///   derived from the binned straight RGB via `rgb_to_luma*_row`,
///   honoring the matrix and range (the genuinely chromatic sources'
///   direct-path behavior). The native-Y luma stream / scratch are unused
///   and `deinterleave_y` is never called.
/// - `true` (`Ya8`): luma is a genuine **independent native-Y area bin**,
///   never derived from the alpha- or range-affected color. `deinterleave_y`
///   stages the native Y plane (the Y bytes of the packed `[Y, A]` row) into
///   a source-width u8 scratch; a 1-channel `AreaStream<u8>` (`y_luma_stream`)
///   bins it as a straight area mean, finalized in lockstep with the color
///   stream. `luma` is the binned Y byte and `luma_u16` its zero-extension —
///   byte-exact to the direct `ya8_to_luma_row` / `ya8_to_luma_u16_row`
///   kernels for every matrix, **both ranges**, AND every alpha mode (under
///   `AlphaMode::Premultiplied` the color collapses to `mean(Y*A)/mean(A)`,
///   so a color-derived luma would be wrong; the native-Y bin is `mean(Y)`
///   regardless of alpha). The color feed below emits no luma in this mode.
///
/// Atomic preflight with conditional ordering: a no-output call returns
/// before any freeze; an out-of-sequence FIRST row is rejected before
/// the freeze (so a rejected row stores no snapshot to poison a retry);
/// on a later row the freeze runs first (so a mid-frame output change
/// trips `ResampleOutputsChanged` rather than being masked by a
/// freshly-attached stream's row-0 mismatch). Both streams and every
/// scratch (color staging, drop-alpha RGB, and — under `NATIVE_Y_LUMA` —
/// the native-Y stream and its staging) are created only after the
/// sequence check, all before the single feed, so a failure mutates no
/// caller output. The color and native-Y streams advance in lockstep
/// (same `idx`, same plan), so the single sequence check on the color
/// stream governs both.
#[cfg(any(feature = "rgb", feature = "gbr", feature = "gray"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_resample<const NATIVE_Y_LUMA: bool>(
  rgba_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  y_luma_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<u8>,
  rgb_drop_scratch: &mut Vec<u8>,
  y_luma_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [u8]),
  deinterleave_y: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink: reject a filter plan before any work (these packed
  // RGBA / YA families are not routed to the filter path).
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the colour stream is the
  // per-row stream; the native-Y stream advances in lockstep with it), or `None`
  // when no output is attached so the preflight short-circuits to a no-op.
  let expected = if need_any {
    Some(rgba_stream.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream allocs + scratch growth below succeed (RFC #238
  // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
  // uncommitted — retryable with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  let premult = alpha_mode.is_premultiplied();
  // Under `NATIVE_Y_LUMA` (`Ya8`), luma / luma_u16 come from the
  // independent native-Y area stream, so the color path's drop-alpha RGB
  // row is needed only for the genuinely color-derived outputs.
  let need_y_luma = NATIVE_Y_LUMA && (luma.is_some() || luma_u16.is_some());
  // The rgb / rgb_u16 / hsv outputs (and, in the non-native-Y mode, luma /
  // luma_u16) need a packed RGB row. It is the per-mode binned color with
  // α dropped — sized to the out-width RGB row only when one of those is
  // attached, so an rgba-only sink neither grows it nor risks its
  // allocation failure. (`rgb_u16` zero-extends this same straight RGB;
  // `rgba_u16` zero-extends the straight RGBA resolved per pixel below —
  // the `Ya8` source exposes u16 RGB outputs the packed-RGBA / Gbrap8
  // sources do not, so they are threaded through the same u8 binning.)
  let need_rgb_drop = rgb.is_some()
    || rgb_u16.is_some()
    || hsv.is_some()
    || (!NATIVE_Y_LUMA && (luma.is_some() || luma_u16.is_some()));
  // Build the missing streams into LOCALS and grow the scratches — every fallible
  // pre-feed step — with NO field mutation, so both stream field inserts and the
  // freeze commit TOGETHER only after they all succeed (the `planar_dual_resample`
  // atomic shape: a scratch OOM leaves `rgba_stream` / `y_luma_stream` `None` AND
  // `resample_outputs` uncommitted, never a partial commit).
  let new_rgba = if rgba_stream.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Native-Y luma stream (`Ya8`): a 1-channel area bin of the native Y plane,
  // created in lockstep with the colour stream so both advance together (the
  // colour stream's sequence check governs both).
  let new_y_luma = if need_y_luma && y_luma_stream.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?;
  }
  if need_y_luma {
    source_luma_scratch(y_luma_scratch, w, plan)?;
  }
  source_rgba_scratch(rgba_scratch, w, plan)?;
  // Every pre-feed allocation (colour + native-Y streams, drop-alpha RGB scratch,
  // native-Y scratch, RGBA scratch) succeeded — COMMIT atomically: insert the
  // streams and freeze the output set together (the compare-only preflight above
  // already caught any later mid-frame output change).
  if let Some(s) = new_rgba {
    *rgba_stream = Some(s);
  }
  if let Some(s) = new_y_luma {
    *y_luma_stream = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Stage the scratches (these grows re-run as no-ops after the pre-commit grows
  // above); the native Y plane de-interleaves into its scratch before the feed.
  let rgb_drop: &mut [u8] = if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?
  } else {
    &mut []
  };
  let y_src: &mut [u8] = if need_y_luma {
    let scratch = source_luma_scratch(y_luma_scratch, w, plan)?;
    deinterleave_y(scratch);
    scratch
  } else {
    &mut []
  };
  let src_rgba = source_rgba_scratch(rgba_scratch, w, plan)?;
  convert_rgba(src_rgba);
  if premult {
    premultiply_rgba_row_in_place::<u8>(src_rgba, w, 255);
  }
  let stream = rgba_stream.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // RGBA output is the per-mode straight color: straight mode copies
    // the binned row; premult mode un-premultiplies it into the dst.
    if let Some(buf) = rgba.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      if premult {
        unpremultiply_binned_rgba_into::<u8>(binned, dst, ow, 255);
      } else {
        dst.copy_from_slice(binned);
      }
    }
    // rgba_u16 zero-extends the straight RGBA: straight mode zero-extends
    // the binned row; premult mode un-premultiplies each channel first.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      if premult {
        for (out_px, in_px) in dst.chunks_exact_mut(4).zip(binned.chunks_exact(4)) {
          let a = in_px[3] as u32;
          for c in 0..3 {
            out_px[c] = unpremultiply_channel(in_px[c] as u32, a, 255) as u16;
          }
          out_px[3] = in_px[3] as u16;
        }
      } else {
        for (d, &s) in dst.iter_mut().zip(binned.iter()) {
          *d = s as u16;
        }
      }
    }
    if need_rgb_drop {
      let nrow = &mut rgb_drop[..3 * ow];
      if premult {
        unpremultiply_binned_rgb_into::<u8>(binned, nrow, ow, 255);
      } else {
        drop_alpha_rgba_to_rgb_row(binned, nrow, ow);
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      // rgb_u16 zero-extends the straight RGB.
      if let Some(buf) = rgb_u16.as_deref_mut() {
        let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        for (d, &s) in dst.iter_mut().zip(nrow.iter()) {
          *d = s as u16;
        }
      }
      // luma / luma_u16 are color-derived only when NOT taking the
      // native-Y stream; under `NATIVE_Y_LUMA` they are emitted from the
      // independent native-Y bin below.
      if !NATIVE_Y_LUMA {
        if let Some(buf) = luma.as_deref_mut() {
          let dst = &mut buf[oy * ow..(oy + 1) * ow];
          crate::row::rgb_to_luma_row(nrow, dst, ow, matrix, full_range, use_simd);
        }
        if let Some(buf) = luma_u16.as_deref_mut() {
          let dst = &mut buf[oy * ow..(oy + 1) * ow];
          crate::row::rgb_to_luma_u16_row(nrow, dst, ow, matrix, full_range, use_simd);
        }
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  // Native-Y luma feed (`Ya8`): a straight single-channel area bin of the
  // native Y plane — alpha- and range-independent by construction. The
  // binned Y byte is the direct `ya8_to_luma_row` output; its zero
  // extension is the direct `ya8_to_luma_u16_row` output. Fed at the same
  // `idx` as the color stream, so the two stay in lockstep.
  if need_y_luma {
    let stream = y_luma_stream.as_mut().expect("created in the preflight");
    stream.feed_row(idx, y_src, use_simd, |oy, binned_y| {
      if let Some(buf) = luma.as_deref_mut() {
        buf[oy * ow..(oy + 1) * ow].copy_from_slice(binned_y);
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        let dst = &mut buf[oy * ow..(oy + 1) * ow];
        for (d, &y) in dst.iter_mut().zip(binned_y.iter()) {
          *d = y as u16;
        }
      }
    })?;
  }
  Ok(())
}

/// Row-stage fused downscale for `Pal8` (8-bit palette-indexed) — the
/// alpha-aware 4-channel analogue of the 3-channel Bayer feed. Averaging
/// palette *indices* is meaningless, so the only sensible area-resample is
/// to expand each pixel to its palette color and bin THAT: `convert_rgba`
/// stages the source row as a canonical source-width `R, G, B, A` u8 row
/// via the per-pixel palette lookup (`pal8_to_rgba_row`, FFmpeg `[B, G, R,
/// A]` → `[R, G, B, A]`), and this tail bins all four channels — so a
/// resampled frame is byte-identical to a direct full-res `Pal8` →
/// RGBA conversion followed by an area-bin of that color (the parity goal).
///
/// Like the genuinely-chromatic packed-RGBA sources, `Pal8` has **no
/// native luma plane**: its direct `luma` / `luma_u16` are derived from
/// the looked-up RGB. But unlike them it carries **no `ColorMatrix` /
/// range** on the row — its luma uses the sink's configured Q8 coefficient
/// set (`LumaCoefficients`, default BT.709), exactly as the Bayer / Pal8
/// identity path does. So this tail emits luma via the **Q8**
/// [`rgb_row_to_luma_row`] / [`rgb_row_to_luma_u16_row`] over the binned
/// straight RGB — NOT the matrix-based `rgb_to_luma_row` the
/// [`packed_rgba_resample`] tail uses — and `luma_u16` is the Q8 path's
/// `(y << 8) | y` full-range widening, the direct kernel's convention.
///
/// `Pal8`'s `rgb_u16` / `rgba_u16` outputs likewise widen each binned
/// 8-bit channel via `(v << 8) | v` (`pal8_to_*_u16_row`'s `expand_u8_to_u16`)
/// — the full-range expansion, **not** the zero-extension the `Ya8` /
/// `Rgba64` u16 paths use. (`Ya8`'s native-Y u16 keeps the low byte;
/// `Pal8`'s palette color is an 8-bit value mapped to the full u16 range.)
///
/// Under [`AlphaMode::Premultiplied`] the staged row is premultiplied in
/// place before binning and un-premultiplied per output row, so the color
/// outputs are alpha-weighted and a fully-transparent binned pixel
/// (`α == 0`) exposes zero color (never bleeds).
///
/// Same atomic conditional-ordering preflight as [`packed_rgba_resample`]:
/// a no-output call returns before any freeze; an out-of-sequence FIRST
/// row is rejected before the freeze (so a rejected row stores no snapshot
/// to poison a retry); a later-row output change trips
/// `ResampleOutputsChanged`; the color stream and both scratches are
/// created after the sequence check and before the single feed, so a
/// failure mutates no caller output.
#[cfg(feature = "mono")]
#[allow(clippy::too_many_arguments)]
pub(super) fn pal8_rgba_resample(
  rgba_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<u8>,
  rgb_drop_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  luma_coeffs_q8: (u32, u32, u32),
  convert_rgba: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (Pal8 is not routed to the filter path): reject a
  // filter plan before any work.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the RGBA colour stream is the
  // sole per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(rgba_stream.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER the
  // fallible stream alloc + scratch growth below succeed (RFC #238 tail-collapse), so
  // a first-row allocation failure leaves `resample_outputs` uncommitted — retryable
  // with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  let premult = alpha_mode.is_premultiplied();
  // The rgb / rgb_u16 / luma / luma_u16 / hsv outputs need a packed RGB
  // row (the per-mode binned color with α dropped); sized to the out-width
  // RGB row only when one of those is attached, so an rgba-only sink
  // neither grows it nor risks its allocation failure.
  let need_rgb_drop =
    rgb.is_some() || rgb_u16.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  // Build the missing stream into a LOCAL and grow the scratches with NO field
  // mutation, so the stream field insert and the freeze commit TOGETHER only after
  // they all succeed (a scratch OOM leaves `rgba_stream` `None` AND `resample_outputs`
  // uncommitted, never a partial commit).
  let new_stream = if rgba_stream.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?;
  }
  source_rgba_scratch(rgba_scratch, w, plan)?;
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare-only preflight above already caught
  // any later mid-frame output change).
  if let Some(s) = new_stream {
    *rgba_stream = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Stage the scratches (these grows re-run as no-ops after the pre-commit grows
  // above) and bin.
  let rgb_drop: &mut [u8] = if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_scratch(rgba_scratch, w, plan)?;
  convert_rgba(src_rgba);
  if premult {
    premultiply_rgba_row_in_place::<u8>(src_rgba, w, 255);
  }
  let stream = rgba_stream.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // RGBA output is the per-mode straight color: straight mode copies the
    // binned row; premult mode un-premultiplies it into the dst.
    if let Some(buf) = rgba.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      if premult {
        unpremultiply_binned_rgba_into::<u8>(binned, dst, ow, 255);
      } else {
        dst.copy_from_slice(binned);
      }
    }
    // rgba_u16 widens the straight RGBA via `(v << 8) | v` (the direct
    // `pal8_to_rgba_u16_row` convention), per channel including alpha.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      if premult {
        for (out_px, in_px) in dst.chunks_exact_mut(4).zip(binned.chunks_exact(4)) {
          let a = in_px[3] as u32;
          for c in 0..3 {
            let s = unpremultiply_channel(in_px[c] as u32, a, 255) as u16;
            out_px[c] = (s << 8) | s;
          }
          let a16 = in_px[3] as u16;
          out_px[3] = (a16 << 8) | a16;
        }
      } else {
        for (d, &s) in dst.iter_mut().zip(binned.iter()) {
          let s = s as u16;
          *d = (s << 8) | s;
        }
      }
    }
    if need_rgb_drop {
      let nrow = &mut rgb_drop[..3 * ow];
      if premult {
        unpremultiply_binned_rgb_into::<u8>(binned, nrow, ow, 255);
      } else {
        drop_alpha_rgba_to_rgb_row(binned, nrow, ow);
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      // rgb_u16 widens the straight RGB via `(v << 8) | v`.
      if let Some(buf) = rgb_u16.as_deref_mut() {
        let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        for (d, &s) in dst.iter_mut().zip(nrow.iter()) {
          let s = s as u16;
          *d = (s << 8) | s;
        }
      }
      // luma / luma_u16: Q8 coefficients over the binned straight RGB (the
      // direct Pal8 path's derivation — no matrix, no range). `luma_u16`
      // is the Q8 path's `(y << 8) | y` widening.
      if let Some(buf) = luma.as_deref_mut() {
        rgb_row_to_luma_row(nrow, &mut buf[oy * ow..(oy + 1) * ow], luma_coeffs_q8);
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        rgb_row_to_luma_u16_row(nrow, &mut buf[oy * ow..(oy + 1) * ow], luma_coeffs_q8);
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Separable-filter fused resize for `Pal8` (8-bit palette-indexed) — the
/// `Filter`-plan twin of the area [`pal8_rgba_resample`]. Filtering palette
/// *indices* is meaningless, exactly as averaging them is, so the source row
/// is first expanded to its palette color (`convert_rgba`, the same
/// `pal8_to_rgba_row` lookup the area path uses, real per-entry straight
/// alpha) and THAT canonical source-width `R, G, B, A` u8 row is fed to a
/// single 4-channel [`FilterStream`](crate::resample::FilterStream). Filtering
/// the looked-up colors (not the indices) is the convert-then-filter mirror of
/// the area convert-then-bin, so a resampled frame is byte-identical to a
/// direct full-res `Pal8` → RGBA conversion followed by the 4-channel
/// `FilterStream` of that color (the parity goal). Because the u8 filter is
/// byte-exact per channel, every output channel — alpha included — equals the
/// single-plane filter of that channel.
///
/// `Pal8` carries **real straight alpha** in the palette (never premultiplied),
/// so all four channels are filtered independently with no premultiplication —
/// the PIL RGBA convention, identical to the real-alpha packed-RGBA
/// [`packed_rgba_filter_resample`]. A premultiplied-mode frame has no filter
/// analogue (the filter engine cannot un-premultiply) and never reaches this
/// tail: the `Pal8` sink routes it to the area [`pal8_rgba_resample`], whose
/// `Filter`-plan reject surfaces the typed `UnsupportedFilter`.
///
/// The non-color outputs keep the **direct `Pal8` derivations**, matching the
/// area path exactly (NOT the matrix-based derivations the chromatic
/// [`packed_rgba_filter_resample`] uses): `rgb` drops alpha; `luma` /
/// `luma_u16` are the sink's **Q8** [`LumaCoefficients`] over the filtered
/// straight RGB ([`rgb_row_to_luma_row`] / [`rgb_row_to_luma_u16_row`], no
/// matrix / range), with `luma_u16` the Q8 path's `(y << 8) | y` full-range
/// widening; `rgb_u16` / `rgba_u16` widen each filtered 8-bit channel via
/// `(v << 8) | v` (the direct `pal8_to_*_u16_row` convention), per channel
/// including alpha. `Pal8` is an 8-bit source so there is no native-depth
/// clamp — the `FilterStream`'s `0..=255` clamp is exactly the source's
/// native range, even for a signed-kernel (CatmullRom / Lanczos3) overshoot.
///
/// Same atomic conditional-ordering preflight as [`pal8_rgba_resample`]: a
/// no-output call returns before any freeze; an out-of-sequence first row is
/// rejected before the freeze (so a rejected row stores no snapshot to poison
/// a retry); a later-row output change trips `ResampleOutputsChanged`; the
/// 4-channel stream and both scratches are created after the sequence check and
/// before the single feed, so a failure mutates no caller output and
/// `AllocationFailed` never masks `OutOfSequenceRow`.
#[cfg(feature = "mono")]
#[allow(clippy::too_many_arguments)]
pub(super) fn pal8_rgba_filter_resample(
  rgba_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<u8>,
  rgb_drop_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  luma_coeffs_q8: (u32, u32, u32),
  convert_rgba: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the RGBA filter stream is the
  // sole per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(rgba_filter_stream.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `pal8_rgba_resample`: the output-set
  // freeze commits only AFTER the fallible stream alloc + scratch growth below
  // succeed, so a first-row allocation failure leaves `resample_outputs` uncommitted.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // The rgb / rgb_u16 / luma / luma_u16 / hsv outputs need a packed RGB row
  // (the filtered color with α dropped); sized to the out-width RGB row only
  // when one of those is attached, so an rgba-only sink neither grows it nor
  // risks its allocation failure.
  let need_rgb_drop =
    rgb.is_some() || rgb_u16.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  // Build the missing filter stream into a LOCAL and grow the scratches with NO field
  // mutation, so the stream field insert and the freeze commit TOGETHER only after
  // they all succeed (a scratch OOM leaves `rgba_filter_stream` `None` AND
  // `resample_outputs` uncommitted, never a partial commit).
  let new_stream = if rgba_filter_stream.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?;
  }
  source_rgba_scratch(rgba_scratch, w, plan)?;
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare-only preflight above already caught
  // any later mid-frame output change).
  if let Some(s) = new_stream {
    *rgba_filter_stream = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Stage the scratches (these grows re-run as no-ops after the pre-commit grows
  // above) and filter.
  let rgb_drop: &mut [u8] = if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_scratch(rgba_scratch, w, plan)?;
  convert_rgba(src_rgba);
  let stream = rgba_filter_stream.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, finalized| {
    // Straight-alpha RGBA output — the finalized 4-channel filter row.
    if let Some(buf) = rgba.as_deref_mut() {
      buf[oy * 4 * ow..(oy + 1) * 4 * ow].copy_from_slice(finalized);
    }
    // rgba_u16 widens the straight RGBA via `(v << 8) | v` (the direct
    // `pal8_to_rgba_u16_row` convention), per channel including alpha.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      for (d, &s) in dst.iter_mut().zip(finalized.iter()) {
        let s = s as u16;
        *d = (s << 8) | s;
      }
    }
    if need_rgb_drop {
      let nrow = &mut rgb_drop[..3 * ow];
      drop_alpha_rgba_to_rgb_row(finalized, nrow, ow);
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      // rgb_u16 widens the straight RGB via `(v << 8) | v`.
      if let Some(buf) = rgb_u16.as_deref_mut() {
        let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        for (d, &s) in dst.iter_mut().zip(nrow.iter()) {
          let s = s as u16;
          *d = (s << 8) | s;
        }
      }
      // luma / luma_u16: Q8 coefficients over the filtered straight RGB (the
      // direct Pal8 path's derivation — no matrix, no range). `luma_u16` is
      // the Q8 path's `(y << 8) | y` widening.
      if let Some(buf) = luma.as_deref_mut() {
        rgb_row_to_luma_row(nrow, &mut buf[oy * ow..(oy + 1) * ow], luma_coeffs_q8);
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        rgb_row_to_luma_u16_row(nrow, &mut buf[oy * ow..(oy + 1) * ow], luma_coeffs_q8);
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Row-stage fused downscale for the high-bit packed straight/premult
/// RGBA family (`Rgba64` / `Bgra64`) and the high-bit planar GBR+alpha
/// family (`Gbrap10` … `Gbrap16`, decoded to the same canonical RGBA
/// row) — the alpha-aware 4-channel analogue of the 3-channel
/// [`packed_rgb_u16_resample_emit`] path. `convert_rgba_u16` stages the
/// wire row as a canonical host-native source-width `R, G, B, A` u16 row
/// (the format's `*_to_rgba_u16` kernel, source wire `BE`); this tail
/// bins all four channels at the source's native depth so resampled
/// alpha is a real area mean (not the forced-opaque-`(1 << SRC_BITS) - 1`
/// the 3-channel u16 path emitted), then per finalized output row
/// resolves the binned native RGBA to its straight form (a copy in
/// straight mode, an un-premultiply in premult mode) and emits: rgba_u16
/// / rgb_u16 at native depth, and rgba / rgb / luma / luma_u16 / hsv from
/// a single `>> (SRC_BITS - 8)` narrowing — the source-of-truth ordering
/// the 3-channel u16 path uses.
///
/// `SRC_BITS` is the source's active bit depth — `16` for the full-16-bit
/// `Rgba64` / `Bgra64`, and `10` / `12` / `14` / `16` for the high-bit
/// `Gbrap*` sources. It governs both the narrowing shift (`>> (SRC_BITS -
/// 8)`) and the native maximum `(1 << SRC_BITS) - 1` used for the
/// premultiply rounding and un-premultiply clamp (so a `Gbrap10` premult
/// bin un-premultiplies against `1023`, not `65535`). Mirrors the
/// `SRC_BITS` parameterization of [`packed_rgb_u16_resample_emit`].
///
/// `NATIVE_LUMA16` and `NATIVE_Y_LUMA` select the `luma` / `luma_u16`
/// source-of-truth (at most one is `true`):
/// - both `false` (`Rgba64` / `Bgra64`): luma_u16 at 8-bit precision from
///   the narrowed straight RGB (their direct path's behavior);
/// - `NATIVE_LUMA16` (`GbrapN`): full native precision from the binned
///   straight RGB via `rgb_to_luma_u16_native_row`, so a resampled
///   `GbrapN` luma_u16 is byte-identical to a direct `GbrapN` conversion
///   of the binned frame (grows `luma_rgb_scratch_u16` to the out-width
///   packed u16 RGB row when luma_u16 is attached);
/// - `NATIVE_Y_LUMA` (`Ya16`): luma is a genuine **independent native-Y
///   area bin**, never derived from the alpha- or range-affected color.
///   `deinterleave_y` stages the native Y plane (the Y elements of the
///   packed `[Y, A]` u16 row, host-native) into a source-width u16 scratch;
///   a 1-channel `AreaStream<u16>` (`y_luma_stream_u16`) bins it at native
///   depth, finalized in lockstep with the color stream. `luma_u16` is the
///   binned Y (host-native pass-through) and `luma` is `binned_y >> 8` —
///   byte-exact to the direct `ya16_to_luma_u16_row` / `ya16_to_luma_row`
///   kernels for every matrix (whereas `rgb_to_luma_u16_native_row` would
///   deviate for matrices whose Q15 weights do not sum to exactly `32768`,
///   e.g. SMPTE-240M), every range, AND every alpha mode (under
///   `AlphaMode::Premultiplied` the color collapses to `mean(Y*A)/mean(A)`,
///   so a color-derived luma would be wrong; the native-Y bin is `mean(Y)`
///   regardless of alpha). The narrowed color path emits no luma / luma_u16
///   in this mode.
///
/// Same atomic conditional-ordering preflight as [`packed_rgba_resample`]:
/// a no-output call returns before any freeze; an out-of-sequence first
/// row is rejected before the freeze; a later-row output change trips
/// `ResampleOutputsChanged`; both streams and every scratch are created
/// after the sequence check and before the single feed. The color and
/// native-Y streams advance in lockstep, so the single sequence check on
/// the color stream governs both.
#[cfg(any(feature = "rgb", feature = "gbr", feature = "gray"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_u16_resample<
  const SRC_BITS: u32,
  const NATIVE_LUMA16: bool,
  const NATIVE_Y_LUMA: bool,
>(
  rgba_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  y_luma_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch_u16: &mut Vec<u16>,
  color_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  luma_rgb_scratch_u16: &mut Vec<u16>,
  y_luma_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba_u16: impl FnOnce(&mut [u16]),
  deinterleave_y: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (high-bit packed RGBA is not routed to the filter
  // path): reject a filter plan before any work.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  const {
    assert!(
      SRC_BITS > 8 && SRC_BITS <= 16,
      "SRC_BITS must be in (8, 16] for the high-bit packed RGBA tail"
    );
    assert!(
      !(NATIVE_LUMA16 && NATIVE_Y_LUMA),
      "luma_u16 has a single source-of-truth: NATIVE_LUMA16 and NATIVE_Y_LUMA are mutually exclusive"
    );
  };
  // `1 << 16` does not overflow u32; the native max governs premultiply
  // rounding and the un-premultiply clamp.
  let max: u32 = (1u32 << SRC_BITS) - 1;
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the colour stream is the
  // per-row stream; the native-Y stream advances in lockstep with it), or `None`
  // when no output is attached so the preflight short-circuits to a no-op.
  let expected = if need_any {
    Some(rgba_stream_u16.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream allocs + scratch growth below succeed (RFC #238
  // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
  // uncommitted — retryable with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Under `NATIVE_Y_LUMA` (`Ya16`), luma / luma_u16 come from the
  // independent native-Y area stream, so the narrowed color path produces
  // them only in the non-native-Y modes.
  let need_y_luma = NATIVE_Y_LUMA && (luma.is_some() || luma_u16.is_some());
  // The u8 / narrowed outputs come from a `>> (SRC_BITS - 8)` narrowing of
  // the straight RGB. `luma_u16` narrows too in the plain branch
  // (`Rgba64` / `Bgra64`: their direct path takes luma at 8-bit precision),
  // but under `NATIVE_LUMA16` (`GbrapN`) luma_u16 is computed at full
  // native precision from the binned straight RGB instead — matching the
  // 3-channel high-bit path's native-luma parity — and under
  // `NATIVE_Y_LUMA` (`Ya16`) luma AND luma_u16 come from the native-Y
  // stream (not the color). None pull in the narrow scratch for luma. A
  // native-u16-only sink never touches the narrow scratch, so the
  // out-width u8 RGB scratch is sized — and its allocation failure risked —
  // only when one of the narrowed outputs is attached.
  let narrowed_luma_u16 = !NATIVE_LUMA16 && !NATIVE_Y_LUMA && luma_u16.is_some();
  let narrowed_luma = !NATIVE_Y_LUMA && luma.is_some();
  let need_narrow =
    rgb.is_some() || rgba.is_some() || narrowed_luma || narrowed_luma_u16 || hsv.is_some();
  // Native-precision luma_u16 (GbrapN) drops alpha from the straight color
  // into this out-width packed u16 RGB scratch, then runs the same
  // `rgb_to_luma_u16_native_row` the direct path uses — so the resampled
  // luma_u16 is byte-identical to a direct GbrapN conversion of the binned
  // frame. Sized only when that output is actually requested.
  let native_luma = NATIVE_LUMA16 && luma_u16.is_some();
  // Build the missing streams into LOCALS and grow the scratches — every fallible
  // pre-feed step — with NO field mutation, so both stream field inserts and the
  // freeze commit TOGETHER only after they all succeed (the `planar_dual_resample`
  // atomic shape: a scratch OOM leaves `rgba_stream_u16` / `y_luma_stream_u16`
  // `None` AND `resample_outputs` uncommitted, never a partial commit).
  let new_rgba = if rgba_stream_u16.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Native-Y luma stream (`Ya16`): a 1-channel native-depth area bin of the native
  // Y plane, created in lockstep with the colour stream so both advance together
  // (the colour stream's sequence check governs both).
  let new_y_luma = if need_y_luma && y_luma_stream_u16.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Grow every pre-feed scratch (the always-sized out-width colour row and
  // source-width RGBA staging, plus the conditional narrow / native-luma /
  // native-Y scratches) BEFORE the commit, so a scratch OOM likewise leaves the
  // stream fields `None` and `resample_outputs` uncommitted.
  source_rgba_u16_scratch(color_scratch_u16, ow, plan)?;
  if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?;
  }
  if native_luma {
    source_rgb_u16_scratch(luma_rgb_scratch_u16, ow, plan)?;
  }
  if need_y_luma {
    source_luma_u16_scratch(y_luma_scratch_u16, w, plan)?;
  }
  source_rgba_u16_scratch(rgba_scratch_u16, w, plan)?;
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the streams and
  // freeze the output set together (the compare-only preflight above already caught
  // any later mid-frame output change).
  if let Some(s) = new_rgba {
    *rgba_stream_u16 = Some(s);
  }
  if let Some(s) = new_y_luma {
    *y_luma_stream_u16 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Re-fetch the scratches (these grows re-run as no-ops after the pre-commit grows
  // above). Out-width straight RGBA colour row (resolved per output row): always
  // sized when any output is attached, so every native and narrowed output reads
  // one canonical straight row.
  let color = source_rgba_u16_scratch(color_scratch_u16, ow, plan)?;
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  let luma_rgb: &mut [u16] = if native_luma {
    source_rgb_u16_scratch(luma_rgb_scratch_u16, ow, plan)?
  } else {
    &mut []
  };
  // Stage the native Y plane (host-native u16) into a source-width scratch before
  // the feed (all fallible growth precedes the first feed).
  let y_src: &mut [u16] = if need_y_luma {
    let scratch = source_luma_u16_scratch(y_luma_scratch_u16, w, plan)?;
    deinterleave_y(scratch);
    scratch
  } else {
    &mut []
  };
  let premult = alpha_mode.is_premultiplied();
  let src_rgba = source_rgba_u16_scratch(rgba_scratch_u16, w, plan)?;
  convert_rgba_u16(src_rgba);
  if premult {
    premultiply_rgba_row_in_place::<u16>(src_rgba, w, max);
  }
  let stream = rgba_stream_u16.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // Resolve the per-mode straight native RGBA once.
    let color = &mut color[..4 * ow];
    if premult {
      unpremultiply_binned_rgba_into::<u16>(binned, color, ow, max);
    } else {
      color.copy_from_slice(binned);
    }
    // Native-depth u16 outputs copy from the straight color row.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      buf[oy * 4 * ow..(oy + 1) * 4 * ow].copy_from_slice(color);
    }
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      drop_alpha_rgba_to_rgb_row(color, dst, ow);
    }
    // Native-precision luma_u16 (GbrapN): drop alpha from the straight
    // native color into the packed u16 RGB scratch, then run the exact
    // `rgb_to_luma_u16_native_row` the direct path uses — full parity at
    // native depth (the `Rgba64` / `Bgra64` path narrows instead, below).
    if native_luma && let Some(buf) = luma_u16.as_deref_mut() {
      let rgb_row = &mut luma_rgb[..3 * ow];
      drop_alpha_rgba_to_rgb_row(color, rgb_row, ow);
      crate::row::rgb_to_luma_u16_native_row(
        rgb_row,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        SRC_BITS,
      );
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      for (d, px) in nrow.chunks_exact_mut(3).zip(color.chunks_exact(4)) {
        d[0] = (px[0] >> (SRC_BITS - 8)) as u8;
        d[1] = (px[1] >> (SRC_BITS - 8)) as u8;
        d[2] = (px[2] >> (SRC_BITS - 8)) as u8;
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = rgba.as_deref_mut() {
        // Narrow all four straight channels (α `>> (SRC_BITS - 8)` too).
        let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        for (d, px) in dst.chunks_exact_mut(4).zip(color.chunks_exact(4)) {
          d[0] = (px[0] >> (SRC_BITS - 8)) as u8;
          d[1] = (px[1] >> (SRC_BITS - 8)) as u8;
          d[2] = (px[2] >> (SRC_BITS - 8)) as u8;
          d[3] = (px[3] >> (SRC_BITS - 8)) as u8;
        }
      }
      // luma: 8-bit Y' from the narrowed straight RGB — the genuinely
      // chromatic sources' direct-path behavior. Skipped under
      // `NATIVE_Y_LUMA` (`Ya16`), where luma comes from the native-Y bin.
      if narrowed_luma && let Some(buf) = luma.as_deref_mut() {
        let dst = &mut buf[oy * ow..(oy + 1) * ow];
        crate::row::rgb_to_luma_row(nrow, dst, ow, matrix, full_range, use_simd);
      }
      // luma_u16: 8-bit-precision Y' derived from the narrowed straight
      // RGB and zero-extended — byte-identical to the direct full-range
      // u16 path's `luma_u16` (which narrows to u8 before luma). Skipped
      // under `NATIVE_LUMA16` (native-precision luma_u16 computed above)
      // and `NATIVE_Y_LUMA` (native binned Y from the Y stream below).
      if narrowed_luma_u16 && let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  // Native-Y luma feed (`Ya16`): a native-depth single-channel area bin of
  // the host-native Y plane — alpha- and range-independent by construction.
  // The binned Y is the direct `ya16_to_luma_u16_row` output (host-native
  // pass-through); `binned_y >> 8` is the direct `ya16_to_luma_row` output.
  // Fed at the same `idx` as the color stream, so the two stay in lockstep.
  if need_y_luma {
    let stream = y_luma_stream_u16
      .as_mut()
      .expect("created in the preflight");
    stream.feed_row(idx, y_src, use_simd, |oy, binned_y| {
      if let Some(buf) = luma_u16.as_deref_mut() {
        buf[oy * ow..(oy + 1) * ow].copy_from_slice(binned_y);
      }
      if let Some(buf) = luma.as_deref_mut() {
        let dst = &mut buf[oy * ow..(oy + 1) * ow];
        for (d, &y) in dst.iter_mut().zip(binned_y.iter()) {
          *d = (y >> 8) as u8;
        }
      }
    })?;
  }
  Ok(())
}

/// Resets the three row-stage area streams (u8 color / native-u16 color
/// / native-u16 luma) and drops the frozen output set for a new frame —
/// the high-bit **planar** YUV 4:4:4 / 4:2:2 sinks' (`yuv-planar`) and the
/// high-bit **semi-planar** P-format sinks' (`yuv-semi-planar`)
/// `begin_frame` body (the streams are lazily created in `process`, so a
/// direct-`process` caller that skips `begin_frame` still gets a correctly
/// initialized first frame). Mirrors the packed high-bit 4:4:4 / 4:2:2
/// sinks' inline resets; factored out because the planar family has eight
/// `begin_frame` impls (Yuv444p / Yuv422p × 10/12/14/16) plus the nine
/// P-format impls share it.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
pub(super) fn reset_high_bit_yuv_streams<F: SourceFormat, R>(sink: &mut MixedSinker<'_, F, R>) {
  if let Some(stream) = sink.rgb_stream.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.rgb_stream_u16.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.luma_stream_u16.as_mut() {
    stream.reset();
  }
  // The filter trio (lazily created when a `Filter` plan drives the sink)
  // shares the same per-frame restart as the area trio above; whichever
  // route the plan kind picks, its streams restart for the new frame.
  if let Some(stream) = sink.rgb_filter_stream.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.rgb_filter_stream_u16.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.luma_filter_stream_u16.as_mut() {
    stream.reset();
  }
  // The high-bit planar 4:2:0 native join (when present) shares the
  // frame-restart contract: restart its plane streams for the new frame.
  // The native join and its route guard exist only under `yuv-planar`
  // (the native tier's home — a yuv-semi-planar-solo build always takes
  // the row-stage tail, so there is no route to clear there).
  #[cfg(feature = "yuv-planar")]
  if let Some(join) = sink.native_420_u16.as_mut() {
    join.reset();
  }
  // The high-bit non-4:2:0 planar native join (when present) shares the
  // frame-restart contract too — restart its plane streams for the new frame.
  #[cfg(feature = "yuv-planar")]
  if let Some(join) = sink.native_planar_u16.as_mut() {
    join.reset();
  }
  // Clear the per-frame frozen native/row-stage route and averaging domain so
  // the next frame may pick either tier / any domain (the dispatch re-freezes
  // each on its first resampled row); a mid-frame flip within a frame stays
  // rejected. The RFC #238 frozen 4:2:2 chroma siting is cleared here too, so
  // the next frame may pick either phase while a mid-frame flip stays rejected.
  #[cfg(feature = "yuv-planar")]
  {
    sink.frozen_native_route = None;
    sink.frozen_domain = None;
    sink.frozen_chroma_centered = None;
    // RFC #238 S6d: clear the frozen VERTICAL (`Bottom`) chroma phase too, so
    // the next frame may pick Center or Bottom while a mid-frame Center ⇆ Bottom
    // flip stays rejected; and invalidate the high-bit bottom-sited vertical
    // lookback (`chroma_prev_u16`, tracked by the shared `chroma_prev_row` tag)
    // so frame N+1's first even row can never box-blend frame N's last chroma
    // row. Both stay `None` / untouched for every non-`Bottom` high-bit format.
    sink.frozen_chroma_bottom_v = None;
    sink.chroma_prev_row = None;
    // RFC #238 Top: clear the frozen VERTICAL (`Top`, `v = 0`) chroma phase so
    // the next frame may pick Top while a mid-frame flip to / from Top stays
    // rejected, and drop any held Top forward-delay odd row so frame N's last
    // deferred colour row never emits into frame N+1 (its `chroma_top_y_u16`
    // bytes are left as-is, re-overwritten before any trusted read). Both stay
    // `None` for every non-`Top` high-bit format.
    sink.frozen_chroma_top_v = None;
    sink.chroma_top_pending = None;
  }
  sink.resample_outputs = None;
}

/// Endian-normalizes one high-bit planar Y row into `scratch`, masking each
/// sample to the source's native depth `(1 << BITS) - 1` (a no-op at
/// `BITS = 16`, and a no-op for every in-range sample).
///
/// The depth-masking de-interleave for the high-bit planar **native-Y luma**
/// resample streams — the row-stage / filter `luma` + `luma_u16` feeds and the
/// native fast tier's binned-Y luma — shared across the plain high-bit planar
/// `Yuv4xxpN` arms and the `Yuva*pN*` arms. `Yuv*pN*Frame::try_new` is
/// geometry-only, so a malformed-but-accepted frame can carry out-of-range Y
/// (e.g. `0x1000` at 12-bit); binning it unmasked publishes a `luma` /
/// `luma_u16` *clamped* to the native max instead of *masked*, inconsistent
/// with the `(1 << BITS) - 1`-masked Y the colour RGB/RGBA kernels decode from
/// the same row (#334). Masking at de-interleave restores that consistency; on
/// the native tier the Y is shared with the colour decode, which re-masks via
/// `extract_hb`, so the pre-mask never changes an in-range colour binning.
/// (The native fast tiers also mask U / V through this: they area-bin chroma in
/// the YUV domain BEFORE the kernel re-masks, where `avg(dirty) & mask` can
/// differ from `avg(dirty & mask)`; the row-stage path converts each sample
/// before its RGB bin, so its chroma needs no pre-mask.)
#[cfg(feature = "yuv-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn deinterleave_y_high_bit_masked<const BITS: u32, const BE: bool>(
  y: &[u16],
  scratch: &mut [u16],
  width: usize,
) {
  let mask = ((1u32 << BITS) - 1) as u16;
  for (dst, &s) in scratch[..width].iter_mut().zip(y.iter()) {
    *dst = (if BE { u16::from_be(s) } else { u16::from_le(s) }) & mask;
  }
}

/// Row-stage fused downscale for the high-bit packed 4:4:4 YUV family
/// (`V30X` / `V410` / `Xv36`). Unlike the 8-bit packed-YUV-4:2:2 path,
/// which carries a single u8 colour binning, this routes **three
/// independent area streams** because the direct path's per-output
/// conversions round and scale independently and luma is taken from the
/// native Y — so no single binning can reproduce every output:
///
/// 1. **u8 colour (rgb / rgba / hsv):** `convert_rgb_u8` stages the
///    direct u8 YUV→RGB row into a source-width u8 scratch; that row
///    bins through the shared u8 packed-RGB tail
///    ([`packed_rgb_resample_emit`]), which emits rgb / rgba / hsv.
///    Luma is *not* derived here (it is taken from Y), so `&mut None`
///    is passed for the tail's luma / luma_u16 outputs.
/// 2. **u16 colour (rgb_u16 / rgba_u16):** `convert_rgb_u16` stages the
///    native-depth u16 YUV→RGB row into a source-width u16 scratch; that
///    row bins through the shared u16 packed-RGB tail
///    ([`packed_rgb_u16_resample_emit`]) at `SRC_BITS`, emitting only
///    rgb_u16 / rgba_u16 (every narrowed output is `&mut None`, so the
///    tail's narrow scratch is never sized).
/// 3. **luma (luma / luma_u16):** `deinterleave_y` stages the native Y
///    into a source-width u16 scratch; a 1-channel `AreaStream<u16>`
///    bins it at native depth. `luma_u16` is the host-native binned Y;
///    `luma` is `binned_y >> (SRC_BITS - 8)`.
///
/// Colour outputs are byte-identical to the area-bin of the direct
/// full-resolution conversion (convert-then-bin — the fused form of
/// converting at full resolution then area-downscaling the RGB); luma is
/// the area-mean of the native Y. A uniform-gray downscale leaves every
/// colour output unchanged — the regression a single narrowed binning
/// would silently break.
///
/// Atomic preflight: a single [`frozen_outputs_check`] over the full
/// output set, then a single sequence check **before any allocation**
/// (so an out-of-sequence row is rejected without allocating and
/// `AllocationFailed` never masks `OutOfSequenceRow`), then every stream
/// and every source-width scratch are created — all before the first
/// feed — so a failure mutates no caller output. A no-output call has no
/// stream to sequence and stays a no-op regardless of the row index.
///
/// NOTE (#263 follow-up): HSV-only (no RGB / RGBA) still stages the
/// source-width u8 RGB row here and derives HSV per OUTPUT row off the
/// binned RGB stream — the direct + native fast tiers go RGB-free, but
/// the row-stage resample's `AreaStream` is keyed on the 3-channel RGB
/// row, so an RGB-free resample+HSV-only is deferred. The HSV output is
/// still correct; only the source-width RGB scratch remains.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv444_triple_resample<const SRC_BITS: u32>(
  rgb_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  rgb_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  luma_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  luma_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgb_u8: impl FnOnce(&mut [u8]),
  convert_rgb_u16: impl FnOnce(&mut [u16]),
  deinterleave_y: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  // Area entry: a `Filter` plan is dispatched by the caller to
  // [`packed_yuv444_triple_filter_resample`] (V410 / V30X) or rejected
  // there (the not-yet-routed 4:4:4 callers), so reject a filter plan
  // that reaches this area body before any work.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16]"
    )
  };
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_u8_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_u16_color = rgb_u16.is_some() || rgba_u16.is_some();

  // Single sequence check before any allocation. The canonical sequence
  // counter is whichever attached stream is fed every row; all attached
  // streams advance in lockstep, so checking one rejects an
  // out-of-sequence row for all without allocating any stream or scratch.
  let expected = if need_luma {
    luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if need_u8_color {
    rgb_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_u16_color {
    rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else {
    return Ok(());
  };
  // On the first row of a frame nothing is frozen yet, so reject an
  // out-of-sequence row here — before the freeze — so a rejected first row
  // never stores a snapshot that would poison a retry. On a later row the
  // freeze runs first (below) so a mid-frame output-set change is reported
  // as ResampleOutputsChanged rather than masked by a freshly-attached
  // stream's row-0 sequence mismatch.
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  // Single freeze over the full output set (luma_f32 is never produced
  // by this family, so it is frozen as absent). A mid-frame output change
  // trips ResampleOutputsChanged.
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  // Create the streams (post-sequence-check). Each plane runs the full
  // output grid against its own source grid (width `w`, height
  // `plan.src_h()`); the colour streams carry 3 interleaved channels, the
  // luma stream 1.
  if need_u8_color && rgb_stream.is_none() {
    *rgb_stream = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_u16_color && rgb_stream_u16.is_none() {
    *rgb_stream_u16 = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_luma && luma_stream_u16.is_none() {
    *luma_stream_u16 = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }

  // Stage + feed + emit. Shared with the filter path
  // ([`packed_yuv444_triple_filter_resample`]) so the area and filter
  // arms run the identical convert-then-resample tail (the only
  // difference is the stream kind built above).
  packed_yuv444_triple_feed_emit::<_, _, _, SRC_BITS>(
    rgb_stream.as_mut(),
    rgb_stream_u16.as_mut(),
    luma_stream_u16.as_mut(),
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    rgb_scratch_u16,
    luma_scratch_u16,
    w,
    plan,
    idx,
    use_simd,
    matrix,
    full_range,
    convert_rgb_u8,
    convert_rgb_u16,
    deinterleave_y,
  )
}

/// Shared stage-then-feed tail for the packed 4:4:4 YUV (no-alpha)
/// family, used by both [`packed_yuv444_triple_resample`] (area) and
/// [`packed_yuv444_triple_filter_resample`] (filter). The two paths
/// differ only in the resampler kind built by the caller — the
/// convert-then-resample staging and per-output emit are identical, so
/// they live here behind the [`RowResampler`](crate::resample::RowResampler)
/// trait (which both [`AreaStream`](crate::resample::AreaStream) and
/// [`FilterStream`](crate::resample::FilterStream) implement). Keeping
/// the emit byte-identical between the arms is what makes the filter
/// output match the area output up to the kernel weights.
///
/// Each stream is `Some` exactly when its output group is attached (the
/// caller created it post-sequence-check). Staging runs every fallible
/// source-width growth before the first feed so a failure mutates no
/// caller output; the three scratches are distinct fields and never
/// alias.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "y2xx",
  feature = "yuv-semi-planar"
))]
#[allow(clippy::too_many_arguments)]
fn packed_yuv444_triple_feed_emit<U8S, U16S, Y16S, const SRC_BITS: u32>(
  rgb_stream: Option<&mut U8S>,
  rgb_stream_u16: Option<&mut U16S>,
  luma_stream_u16: Option<&mut Y16S>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  luma_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgb_u8: impl FnOnce(&mut [u8]),
  convert_rgb_u16: impl FnOnce(&mut [u16]),
  deinterleave_y: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError>
where
  U8S: crate::resample::RowResampler<u8>,
  U16S: crate::resample::RowResampler<u16>,
  Y16S: crate::resample::RowResampler<u16>,
{
  let ow = plan.out_w();
  // Stage every source-width row (all fallible growths run before the
  // first feed, keeping the call atomic). The three scratches are
  // distinct fields and never alias.
  let u8_color_row = if rgb_stream.is_some() {
    let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb_u8(scratch);
    Some(scratch)
  } else {
    None
  };
  let u16_color_row = if rgb_stream_u16.is_some() {
    let scratch = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
    convert_rgb_u16(scratch);
    Some(scratch)
  } else {
    None
  };
  let y_row = if luma_stream_u16.is_some() {
    let scratch = source_luma_u16_scratch(luma_scratch_u16, w, plan)?;
    deinterleave_y(scratch);
    Some(scratch)
  } else {
    None
  };

  // Feed each stream and emit. The u8 tail emits rgb / rgba / hsv only
  // (luma comes from Y, so its luma / luma_u16 are `&mut None`); the u16
  // tail emits rgb_u16 / rgba_u16 only (every narrowed output is
  // `&mut None`, so its narrow scratch is never sized — `rgb_scratch`,
  // already consumed by the u8 feed above, is passed as the unused
  // placeholder).
  if let Some(scratch) = u8_color_row {
    let stream = rgb_stream.expect("staged only when present");
    packed_rgb_resample_emit(
      stream, plan, rgb, rgba, &mut None, &mut None, hsv, scratch, matrix, full_range, idx,
      use_simd,
    )?;
  }
  if let Some(scratch) = u16_color_row {
    let stream = rgb_stream_u16.expect("staged only when present");
    packed_rgb_u16_resample_emit::<SRC_BITS, false>(
      stream,
      plan,
      &mut None,
      &mut None,
      &mut None,
      rgb_u16,
      rgba_u16,
      &mut None,
      &mut None,
      scratch,
      rgb_scratch,
      matrix,
      full_range,
      idx,
      use_simd,
    )?;
  }
  if let Some(scratch) = y_row {
    let stream = luma_stream_u16.expect("staged only when present");
    // A signed filter kernel (CatmullRom / Lanczos3) overshoots a legal
    // edge, so a finalized binned native-Y sample can exceed the source's
    // native max `(1 << SRC_BITS) - 1` even though the `FilterStream`
    // clamps it to the full `u16` range. The de-interleaved native-Y luma
    // bypasses the colour path's `packed_rgb_u16_resample_emit` clamp, so
    // we clip here — the luma twin of that helper's `native_max` clamp.
    // Without it the native-depth `luma_u16` would publish a value above
    // the documented `[0, (1 << SRC_BITS) - 1]` range, and the u8 luma
    // narrowing (`>> (SRC_BITS - 8)`) would wrap a clipped-high Y edge to a
    // small value instead of `255`. For `SRC_BITS == 16` the native max is
    // the u16 max, so this is a value no-op; the area path never
    // overshoots, so it is a no-op there too.
    let native_max: u16 = ((1u32 << SRC_BITS) - 1) as u16;
    stream.feed_row(idx, scratch, use_simd, |oy, binned_y| {
      // luma_u16: host-native pass-through of the binned native Y, clamped
      // to the native max for a sub-16-bit source.
      if let Some(buf) = luma_u16.as_deref_mut() {
        let out = &mut buf[oy * ow..(oy + 1) * ow];
        if SRC_BITS < 16 {
          for (dst, &src) in out.iter_mut().zip(binned_y) {
            *dst = src.min(native_max);
          }
        } else {
          out.copy_from_slice(binned_y);
        }
      }
      // luma: narrow the binned native Y to u8 (`>> (SRC_BITS - 8)`),
      // clamping to the native max first so a sub-16-bit overshoot clips to
      // `255` instead of wrapping.
      if let Some(buf) = luma.as_deref_mut() {
        if SRC_BITS < 16 {
          for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(binned_y) {
            *dst = (src.min(native_max) >> (SRC_BITS - 8)) as u8;
          }
        } else {
          for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(binned_y) {
            *dst = (src >> (SRC_BITS - 8)) as u8;
          }
        }
      }
    })?;
  }

  Ok(())
}

/// Separable-filter fused resize for the **packed 4:4:4 YUV (no-alpha)**
/// family — the [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_yuv444_triple_resample`]. It mirrors the area path exactly:
/// the YUV is first converted to RGB by the **same** `convert_rgb_u8` /
/// `convert_rgb_u16` closures, then the RGB is resampled — only the
/// resampler kind differs (signed-coefficient [`FilterStream`] instead
/// of [`AreaStream`]). The staged RGB feeds the same emit
/// ([`packed_yuv444_triple_feed_emit`]), so the resampled colour output
/// equals the equivalent packed-RGB filter resample of the converted
/// pixels, and (because the area path converts-then-bins the same RGB)
/// matches the area output up to the kernel weights.
///
/// Luma is the native Y filter-resampled (the filter twin of the area
/// path's native-Y bin): `deinterleave_y` stages the native Y into a
/// source-width `u16` scratch and a 1-channel
/// [`FilterStream<u16>`](crate::resample::FilterStream)
/// (`luma_filter_stream_u16`) resamples it at native depth — luma is
/// taken from Y, never colour-derived, so it stays byte-exact to the
/// direct `*_to_luma*` kernels' filter resample. `luma_u16` is the
/// host-native resampled Y; `luma` is `resampled_y >> (SRC_BITS - 8)`.
///
/// Both the colour `u16` outputs and the native-Y luma carry the source's
/// native depth (`SRC_BITS`, `10` for `V410` / `V30X`, low-bit-packed
/// `[0, (1 << SRC_BITS) - 1]`): a signed kernel (CatmullRom / Lanczos3)
/// can overshoot a legal edge. For colour, [`packed_rgb_u16_resample_emit`]
/// clamps every resampled sample to the native max before any `u16` copy /
/// RGBA expansion / u8 narrowing (the same `SRC_BITS < 16` clamp the
/// high-bit GBR / planar filter routes use). The de-interleaved native-Y
/// luma bypasses that colour helper, so [`packed_yuv444_triple_feed_emit`]
/// applies the matching clamp to the binned Y before the `luma_u16` copy
/// and the u8 narrowing — keeping `luma_u16 <= (1 << SRC_BITS) - 1` and a
/// clipped-high Y edge mapping to `255` rather than wrapping. Both clamps
/// match the in-range area path's output exactly.
///
/// Atomic preflight (mirrors [`packed_yuv444_triple_resample`]): a single
/// [`frozen_outputs_check`] over the full output set, then a single
/// sequence check on whichever stream is fed every row **before any
/// allocation** (an out-of-sequence first row is rejected before the
/// freeze, storing no snapshot to poison a retry; on a later row the
/// freeze runs first so a mid-frame output change trips
/// `ResampleOutputsChanged`), then every stream and source-width scratch
/// is created before the first feed — so a failure mutates no caller
/// output. A no-output call has no stream to sequence and stays a no-op.
///
/// NOTE (#263 follow-up): HSV-only (no RGB / RGBA) still stages the
/// source-width u8 RGB row here and derives HSV per OUTPUT row off the
/// filtered RGB stream — the direct + native fast tiers go RGB-free, but
/// the row-stage filter resample's stream is keyed on the 3-channel RGB
/// row, so an RGB-free resample+HSV-only is deferred. The HSV output is
/// still correct; only the source-width RGB scratch remains.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv444_triple_filter_resample<const SRC_BITS: u32>(
  rgb_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  rgb_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  luma_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  luma_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgb_u8: impl FnOnce(&mut [u8]),
  convert_rgb_u16: impl FnOnce(&mut [u16]),
  deinterleave_y: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16]"
    )
  };
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_u8_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_u16_color = rgb_u16.is_some() || rgba_u16.is_some();

  let (fh, fv) = (
    plan
      .filter_h()
      .expect("filter plan carries horizontal windows"),
    plan
      .filter_v()
      .expect("filter plan carries vertical windows"),
  );

  // Single sequence check before any allocation, on whichever attached
  // stream is fed every row (all attached streams advance in lockstep).
  let expected = if need_luma {
    luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if need_u8_color {
    rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_u16_color {
    rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else {
    return Ok(());
  };
  // First row: reject an out-of-sequence row before the freeze so a
  // rejected first row stores no snapshot that would poison a retry.
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  // Create the filter streams (post-sequence-check). Each plane runs the
  // full output grid against its own source grid; the colour streams
  // carry 3 interleaved channels, the luma stream 1.
  if need_u8_color && rgb_filter_stream.is_none() {
    *rgb_filter_stream = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_u16_color && rgb_filter_stream_u16.is_none() {
    *rgb_filter_stream_u16 = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_luma && luma_filter_stream_u16.is_none() {
    *luma_filter_stream_u16 = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }

  packed_yuv444_triple_feed_emit::<_, _, _, SRC_BITS>(
    rgb_filter_stream.as_mut(),
    rgb_filter_stream_u16.as_mut(),
    luma_filter_stream_u16.as_mut(),
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    rgb_scratch_u16,
    luma_scratch_u16,
    w,
    plan,
    idx,
    use_simd,
    matrix,
    full_range,
    convert_rgb_u8,
    convert_rgb_u16,
    deinterleave_y,
  )
}

/// Shared stage-then-feed tail for the packed 4:4:4 YUV-**with-alpha**
/// family, used by both [`packed_yuva444_resample`] (area) and
/// [`packed_yuva444_filter_resample`] (filter). The two paths differ only
/// in the resampler kind built by the caller — the convert-then-resample
/// staging and per-output emit are identical, so they live here behind the
/// [`RowResampler`](crate::resample::RowResampler) trait (which both
/// [`AreaStream`](crate::resample::AreaStream) and
/// [`FilterStream`](crate::resample::FilterStream) implement). Keeping the
/// emit byte-identical between the arms is what makes the filter output
/// match the area output up to the kernel weights.
///
/// Each stream is `Some` exactly when its output group is attached (the
/// caller created it post-sequence-check). Staging runs every fallible
/// out-width resolve growth and source-width conversion before the first
/// feed so a failure mutates no caller output; the five scratches are
/// distinct fields and never alias.
///
/// `premult` selects the alpha resolve (the area path passes
/// `alpha_mode.is_premultiplied()`; the filter path is straight-only — a
/// premultiplied packed YUVA stays on the area path, which un-premultiplies
/// — so it passes `false`). When `premult` is set the source rows are
/// premultiplied in place before binning and each finalized row is
/// un-premultiplied on the way out.
///
/// `SRC_BITS` is the source's native depth (`8` for `Vuya` / `Vuyx`, `16`
/// for `Ayuv64`). A signed filter kernel (CatmullRom / Lanczos3) overshoots
/// a legal edge, so a finalized binned sample can exceed the source's
/// native max `(1 << SRC_BITS) - 1` even though the `FilterStream` clamps it
/// to the full element range. For `SRC_BITS < 16` every binned native-u16
/// colour channel (alpha too) and the binned native Y are clamped to the
/// native max before any u16 copy / RGB drop / u8 narrowing — the same
/// native-depth clamp the high-bit packed-RGBA and no-alpha 4:4:4 filter
/// tails apply, so `luma_u16` / `rgb_u16` / `rgba_u16` stay within
/// `[0, (1 << SRC_BITS) - 1]` and a clipped-high edge narrows to `255`
/// instead of wrapping. The area path never overshoots, so the clamp is a
/// value no-op there; at `SRC_BITS == 16` the native max is the element max,
/// so it is a no-op for `Ayuv64` too.
///
/// `ZEXT_U16_COLOR` (the packed 8-bit `Ya8` gray+alpha filter caller only):
/// when set, the u16 colour outputs are NOT an independent native-u16 binning
/// — the caller passes no u16 colour stream (`rgba_stream_u16 == None`, so
/// binning 2 below never runs and no native-u16 colour scratch is sized) and
/// the u8 colour binning (binning 1) emits `rgb_u16` / `rgba_u16` as the
/// straight binned u8 colour zero-extended (`s as u16`). This is correct only
/// for an 8-bit source, whose native-depth colour IS the u8 colour widened
/// (`rgba_u16 == rgba as u16`), matching the area path's contract. Every
/// other caller passes `false` and keeps the independent native-u16 binning
/// byte-for-byte. Straight alpha only on this path (the `Ya8` filter caller
/// never premultiplies).
#[cfg(any(feature = "yuv-444-packed", feature = "yuva", feature = "gray"))]
#[allow(clippy::too_many_arguments)]
fn packed_yuva444_feed_emit<C8S, C16S, Y16S, const SRC_BITS: u32, const ZEXT_U16_COLOR: bool>(
  rgba_stream: Option<&mut C8S>,
  rgba_stream_u16: Option<&mut C16S>,
  luma_stream_u16: Option<&mut Y16S>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<u8>,
  rgb_drop_scratch: &mut Vec<u8>,
  rgba_scratch_u16: &mut Vec<u16>,
  color_scratch_u16: &mut Vec<u16>,
  luma_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  premult: bool,
  convert_rgba_u8: impl FnOnce(&mut [u8]),
  convert_rgba_u16: impl FnOnce(&mut [u16]),
  deinterleave_y: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError>
where
  C8S: crate::resample::RowResampler<u8>,
  C16S: crate::resample::RowResampler<u16>,
  Y16S: crate::resample::RowResampler<u16>,
{
  // `1 << 16` does not overflow u32; governs the u16 premultiply rounding,
  // the un-premultiply clamp, and the sub-16-bit native-max clip. The u8
  // colour stream always uses 255.
  let max_u16: u32 = (1u32 << SRC_BITS) - 1;
  let native_max: u16 = max_u16 as u16;
  let ow = plan.out_w();
  let need_colour_u8 = rgba_stream.is_some();
  let need_colour_u16 = rgba_stream_u16.is_some();
  // `ZEXT_U16_COLOR` (the packed 8-bit `Ya8` gray+alpha caller): the u16
  // colour outputs are the zero-extension of the binned u8 colour (the
  // source is 8-bit, so its native-depth colour is the u8 colour widened —
  // `rgba_u16 == rgba as u16`, exactly the area path's contract), NOT an
  // independent native-u16 binning. So the caller passes no u16 colour
  // stream (`rgba_stream_u16 == None`, `need_colour_u16 == false`: binning 2
  // below never runs, no native-u16 scratch is sized) and the u8 binning
  // emits the widened u16 outputs inline. Every other caller passes
  // `false` and keeps the independent native-u16 binning byte-for-byte.
  let zext_u16 = ZEXT_U16_COLOR && (rgb_u16.is_some() || rgba_u16.is_some());
  debug_assert!(
    !ZEXT_U16_COLOR || !need_colour_u16,
    "ZEXT_U16_COLOR derives the u16 colour from the u8 binning, so no u16 colour stream is fed"
  );
  // The u8 colour stream's rgb / hsv outputs (and, under `ZEXT_U16_COLOR`,
  // rgb_u16) need a packed RGB row (the per-mode binned colour with α
  // dropped), sized to the out-width RGB row only when one of those is
  // attached so an rgba-only sink never grows it.
  let need_rgb_drop = rgb.is_some() || hsv.is_some() || (zext_u16 && rgb_u16.is_some());

  // Stage every source-width row and grow every out-width resolve scratch
  // before the first feed (all fallible growths precede it, keeping the
  // call atomic). The five scratches are distinct fields and never alias.
  let rgb_drop: &mut [u8] = if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?
  } else {
    &mut []
  };
  let colour_u16: &mut [u16] = if need_colour_u16 {
    source_rgba_u16_scratch(color_scratch_u16, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba_u8 = if need_colour_u8 {
    let scratch = source_rgba_scratch(rgba_scratch, w, plan)?;
    convert_rgba_u8(scratch);
    if premult {
      premultiply_rgba_row_in_place::<u8>(scratch, w, 255);
    }
    Some(scratch)
  } else {
    None
  };
  let src_rgba_u16 = if need_colour_u16 {
    let scratch = source_rgba_u16_scratch(rgba_scratch_u16, w, plan)?;
    convert_rgba_u16(scratch);
    if premult {
      premultiply_rgba_row_in_place::<u16>(scratch, w, max_u16);
    }
    Some(scratch)
  } else {
    None
  };
  let y_row = if luma_stream_u16.is_some() {
    let scratch = source_luma_u16_scratch(luma_scratch_u16, w, plan)?;
    deinterleave_y(scratch);
    Some(scratch)
  } else {
    None
  };

  // Binning 1 — u8 colour. Resolve the per-mode straight RGBA per output
  // row, then emit rgba (straight RGBA), rgb / hsv (straight RGB). luma is
  // native Y (binning 3), so it is never derived here. The u8 filter /
  // area stream already clamps to `[0, 255]`, so no native-max clip is
  // needed on this path.
  if let Some(scratch) = src_rgba_u8 {
    let stream = rgba_stream.expect("staged only when present");
    stream.feed_row(idx, scratch, use_simd, |oy, binned| {
      if let Some(buf) = rgba.as_deref_mut() {
        let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        if premult {
          unpremultiply_binned_rgba_into::<u8>(binned, dst, ow, 255);
        } else {
          dst.copy_from_slice(binned);
        }
      }
      // `ZEXT_U16_COLOR`: rgba_u16 is the straight binned u8 RGBA
      // zero-extended (`s as u16`), matching the area path's degenerate-YUVA
      // u16 contract. Straight alpha only (this caller never premultiplies).
      if zext_u16 && let Some(buf) = rgba_u16.as_deref_mut() {
        let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        for (d, &s) in dst.iter_mut().zip(binned.iter()) {
          *d = s as u16;
        }
      }
      if need_rgb_drop {
        let nrow = &mut rgb_drop[..3 * ow];
        if premult {
          unpremultiply_binned_rgb_into::<u8>(binned, nrow, ow, 255);
        } else {
          drop_alpha_rgba_to_rgb_row(binned, nrow, ow);
        }
        if let Some(buf) = rgb.as_deref_mut() {
          buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
        }
        // `ZEXT_U16_COLOR`: rgb_u16 zero-extends the straight binned u8 RGB
        // (α dropped) — the same `nrow` the u8 rgb / hsv use.
        if zext_u16 && let Some(buf) = rgb_u16.as_deref_mut() {
          let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
          for (d, &s) in dst.iter_mut().zip(nrow.iter()) {
            *d = s as u16;
          }
        }
        if let Some(hsv) = hsv.as_mut() {
          let (h, s, v) = hsv.hsv();
          crate::row::rgb_to_hsv_row(
            nrow,
            &mut h[oy * ow..(oy + 1) * ow],
            &mut s[oy * ow..(oy + 1) * ow],
            &mut v[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
      }
    })?;
  }

  // Binning 2 — u16 colour at native depth, INDEPENDENT of binning 1.
  // Resolve the per-mode straight native RGBA per output row, then emit
  // rgba_u16 (straight) and rgb_u16 (drop α). A signed-kernel overshoot of
  // a sub-16-bit edge is clipped to the native max before any output is
  // derived (no-op at 16-bit — the `FilterStream`'s `0..=65535` clamp is
  // already the native clamp, and the area path never overshoots).
  if let Some(scratch) = src_rgba_u16 {
    let stream = rgba_stream_u16.expect("staged only when present");
    stream.feed_row(idx, scratch, use_simd, |oy, binned| {
      let colour = &mut colour_u16[..4 * ow];
      if premult {
        unpremultiply_binned_rgba_into::<u16>(binned, colour, ow, max_u16);
      } else if SRC_BITS < 16 {
        for (d, &s) in colour.iter_mut().zip(binned.iter()) {
          *d = s.min(native_max);
        }
      } else {
        colour.copy_from_slice(binned);
      }
      if let Some(buf) = rgba_u16.as_deref_mut() {
        buf[oy * 4 * ow..(oy + 1) * 4 * ow].copy_from_slice(colour);
      }
      if let Some(buf) = rgb_u16.as_deref_mut() {
        let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        drop_alpha_rgba_to_rgb_row(colour, dst, ow);
      }
    })?;
  }

  // Binning 3 — native Y through the 1-channel u16 luma stream. The binned
  // row is host-native; luma_u16 is its pass-through, luma the
  // `>> (SRC_BITS - 8)` narrowing (`>> 0` for an 8-bit source). Alpha- and
  // range-independent by construction. A sub-16-bit signed-kernel overshoot
  // is clipped to the native max before the luma_u16 copy AND the u8
  // narrow, so luma_u16 stays `<= (1 << SRC_BITS) - 1` and a clipped-high Y
  // edge narrows to `255` instead of wrapping (no-op at 16-bit / area).
  if let Some(y_row) = y_row {
    let stream = luma_stream_u16.expect("staged only when present");
    stream.feed_row(idx, y_row, use_simd, |oy, binned_y| {
      if let Some(buf) = luma_u16.as_deref_mut() {
        let out = &mut buf[oy * ow..(oy + 1) * ow];
        if SRC_BITS < 16 {
          for (dst, &y) in out.iter_mut().zip(binned_y) {
            *dst = y.min(native_max);
          }
        } else {
          out.copy_from_slice(binned_y);
        }
      }
      if let Some(buf) = luma.as_deref_mut() {
        if SRC_BITS < 16 {
          for (dst, &y) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(binned_y) {
            *dst = (y.min(native_max) >> (SRC_BITS - 8)) as u8;
          }
        } else {
          for (dst, &y) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(binned_y) {
            *dst = (y >> (SRC_BITS - 8)) as u8;
          }
        }
      }
    })?;
  }

  Ok(())
}

/// Row-stage fused downscale for the **packed 4:4:4 YUV-with-alpha**
/// family (`Vuya` 8-bit and `Ayuv64` 16-bit) — the alpha-aware analogue
/// of [`packed_yuv444_triple_resample`]. Packed YUVA is the most
/// demanding alpha family: it must reproduce a direct convert-then-bin
/// for **four** outputs that each round independently, so this routes up
/// to **four** independent area binnings rather than reusing the
/// packed-RGBA tails (whose u8 outputs are a `>> (SRC_BITS - 8)`
/// narrowing of the u16 bin — correct for an RGB source, but **wrong**
/// for YUV, whose u8 and u16 `YUV→RGB` kernels round and scale
/// independently). The four binnings:
///
/// 1. **u8 colour (rgb / rgba / hsv):** `convert_rgba_u8` stages the
///    direct u8 `YUV→RGB` row **with real source alpha** as a canonical
///    source-width `R, G, B, A` u8 row (`*_to_rgba_row`); the 4-channel
///    [`AreaStream<u8>`](crate::resample::AreaStream) (`rgba_stream`)
///    bins all four channels so resampled alpha is a real area mean.
///    Per finalized output row the binned RGBA resolves to its straight
///    form (a copy in [`AlphaMode::Straight`], an un-premultiply in
///    [`AlphaMode::Premultiplied`]) and emits rgba; rgb / hsv drop alpha.
/// 2. **u16 colour (rgb_u16 / rgba_u16):** `convert_rgba_u16` stages the
///    **independent** native-depth u16 `YUV→RGB` row with source alpha
///    (`*_to_rgba_u16_row`); the 4-channel [`AreaStream<u16>`]
///    (`rgba_stream_u16`) bins at native depth and emits rgba_u16 /
///    rgb_u16 from its own straight resolve — never a narrowing of (1).
/// 3. **luma (luma / luma_u16):** `deinterleave_y` stages the native Y
///    plane into a source-width u16 scratch; a 1-channel
///    [`AreaStream<u16>`] (`luma_stream_u16`) bins it at native depth.
///    Luma is **native Y**, NOT derived from either colour stream —
///    byte-exact to the direct `*_to_luma*` kernels for every matrix,
///    both ranges, AND every alpha mode. Under
///    [`AlphaMode::Premultiplied`] each colour stream collapses to
///    `mean(Y·A)/mean(A)`, but native Y stays `mean(Y)`; deriving luma
///    from colour would be wrong (the bug the `Ya` family fixed —
///    [`packed_rgba_resample`]'s `NATIVE_Y_LUMA`). luma_u16 is the
///    host-native binned Y; luma is `binned_Y >> (SRC_BITS - 8)` (an
///    8-bit `Vuya` is `>> 0`, a zero-extension; `Ayuv64` is `>> 8`).
///
/// `SRC_BITS` is the source's native Y / colour depth (`8` for `Vuya`,
/// `16` for `Ayuv64`): it governs the luma narrowing shift and the u16
/// premultiply / un-premultiply maximum `(1 << SRC_BITS) - 1` (so an
/// 8-bit source never builds the u16 colour stream — `Vuya` exposes no
/// u16 outputs, leaving `need_colour_u16` always false). The u8 colour
/// stream's premultiply maximum is always `255`.
///
/// This is an internal `pub(super)` tail, kept separate from
/// [`packed_yuv444_triple_resample`] so the no-alpha 4:4:4 callers
/// (`V30X` / `V410` / `Xv36`, and `Vuyx` whose padding byte forces α
/// opaque) stay byte-identical, and from the packed-RGBA tails so the
/// independent u8/u16 YUV colour rounding is preserved. The alpha
/// arithmetic reuses the shared [`premultiply_rgba_row_in_place`] /
/// [`unpremultiply_binned_rgba_into`] / [`unpremultiply_binned_rgb_into`]
/// helpers, so straight / premultiplied semantics are byte-identical to
/// the packed-RGBA family.
///
/// Atomic preflight (mirrors [`packed_yuv444_triple_resample`]): a
/// no-output call returns before any freeze; a single
/// [`frozen_outputs_check`] runs, then a single sequence check on
/// whichever stream is fed every row (all active streams advance in
/// lockstep) **before any allocation** — an out-of-sequence first row is
/// rejected before the freeze (storing no snapshot to poison a retry),
/// and on a later row the freeze runs first (a mid-frame output change
/// trips `ResampleOutputsChanged` rather than being masked by a
/// freshly-attached stream's row-0 mismatch). Every stream and every
/// (distinct, non-aliasing) scratch is created after the sequence check
/// and before the first feed, so a failure mutates no caller output. The
/// alpha mode is snapshotted at `begin_frame` and checked by the caller
/// (via [`check_frozen_alpha_mode`]) before this tail runs.
#[cfg(any(feature = "yuv-444-packed", feature = "yuva"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuva444_resample<const SRC_BITS: u32>(
  rgba_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  rgba_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  luma_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<u8>,
  rgb_drop_scratch: &mut Vec<u8>,
  rgba_scratch_u16: &mut Vec<u16>,
  color_scratch_u16: &mut Vec<u16>,
  luma_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  convert_rgba_u8: impl FnOnce(&mut [u8]),
  convert_rgba_u16: impl FnOnce(&mut [u16]),
  deinterleave_y: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (packed YUVA 4:4:4 is not routed to the filter path):
  // reject a filter plan before any work.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16] for packed YUVA 4:4:4"
    )
  };
  let premult = alpha_mode.is_premultiplied();
  let need_colour_u8 = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_colour_u16 = rgb_u16.is_some() || rgba_u16.is_some();
  let need_luma = luma.is_some() || luma_u16.is_some();

  // Single sequence check before any allocation, on whichever stream is
  // fed every row (all active streams advance in lockstep against the
  // frozen output set). A no-output call has no stream to sequence and
  // stays a no-op regardless of the row index.
  let expected = if need_colour_u8 {
    rgba_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_colour_u16 {
    rgba_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if need_luma {
    luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else {
    return Ok(());
  };
  // First row: reject an out-of-sequence row before the freeze so a
  // rejected first row stores no snapshot that would poison a retry. On a
  // later row the freeze runs first (below).
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  // Create the streams (post-sequence-check), each running the full
  // output grid against its own source grid. The colour streams carry 4
  // interleaved channels, the luma stream 1.
  if need_colour_u8 && rgba_stream.is_none() {
    *rgba_stream = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_colour_u16 && rgba_stream_u16.is_none() {
    *rgba_stream_u16 = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_luma && luma_stream_u16.is_none() {
    *luma_stream_u16 = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }

  // Stage + feed + emit. Shared with the filter path
  // ([`packed_yuva444_filter_resample`]) so the area and filter arms run
  // the identical convert-then-resolve-then-emit tail (the only difference
  // is the stream kind built above). The native-max clamp inside is a value
  // no-op for the area path (area binning never overshoots). The area path
  // keeps the independent native-u16 binning (`ZEXT_U16_COLOR = false`): a
  // genuine 16-bit packed YUVA (`Ayuv64`) needs it, and `Ya8`'s area path
  // already zero-extends via [`packed_rgba_resample`], not this tail.
  packed_yuva444_feed_emit::<_, _, _, SRC_BITS, false>(
    rgba_stream.as_mut(),
    rgba_stream_u16.as_mut(),
    luma_stream_u16.as_mut(),
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgba_scratch,
    rgb_drop_scratch,
    rgba_scratch_u16,
    color_scratch_u16,
    luma_scratch_u16,
    w,
    plan,
    idx,
    use_simd,
    premult,
    convert_rgba_u8,
    convert_rgba_u16,
    deinterleave_y,
  )
}

/// Separable-filter fused resize for the **packed 4:4:4 YUV-with-alpha**
/// family — the [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_yuva444_resample`], and the 4-channel analogue of
/// [`packed_yuv444_triple_filter_resample`]. It mirrors the area path
/// exactly: the YUVA is first converted to a canonical `R, G, B, A` row
/// (real source alpha) by the **same** `convert_rgba_u8` / `convert_rgba_u16`
/// closures the area path uses, then resampled — only the resampler kind
/// differs (signed-coefficient [`FilterStream`] instead of [`AreaStream`]).
/// The staged RGBA feeds the same emit ([`packed_yuva444_feed_emit`]), so
/// each resampled colour output equals the equivalent packed-RGBA filter
/// resample of the converted pixels (`rgba` == the 8-bit `Rgba` filter; on
/// the independent-u16 path `rgba_u16` == the `Rgba64` filter clamped to the
/// native max, on the `ZEXT_U16_COLOR` path `rgba_u16` == `rgba as u16` — see
/// the const below), and matches the area output up to the kernel weights.
///
/// **Straight alpha only.** PIL filters R, G, B, A independently with no
/// premultiplication; a premultiplied packed YUVA source stays on the area
/// path (which un-premultiplies), so this is reached only for straight
/// alpha and passes `premult = false` to the shared emit. `Vuyx` reaches
/// here too: its padding byte forces α opaque, so `convert_rgba_u8` writes a
/// constant `0xFF` alpha plane — a constant channel filters to itself
/// (partition of unity), so the 4-channel filter reproduces the no-alpha
/// 3-channel result with α pinned opaque, no separate path needed.
///
/// Luma is the native Y filter-resampled (the filter twin of the area
/// path's native-Y bin) — luma is taken from Y, never colour-derived, so it
/// stays byte-exact to the direct `*_to_luma*` kernels' filter resample. The
/// const `NATIVE_LUMA_U8` selects which native-Y stream carries it:
///
/// - **`false` (the packed `Vuya` / `Vuyx` callers):** `deinterleave_y`
///   stages the native Y into a source-width `u16` scratch and a 1-channel
///   [`FilterStream<u16>`](crate::resample::FilterStream)
///   (`luma_filter_stream_u16`) resamples it at native depth (the shared
///   [`packed_yuva444_feed_emit`] owns the luma binning). The packed Y bytes
///   live interleaved in the source row, so there is no contiguous native-Y
///   plane to feed a `u8` stream directly.
/// - **`true` (the 8-bit planar `Yuva420p` / `Yuva422p` / `Yuva444p`
///   callers):** the contiguous native-Y plane `y_row_u8` feeds a 1-channel
///   [`FilterStream<u8>`](crate::resample::FilterStream)
///   (`luma_filter_stream`) directly — the **same** `u8` luma stream the
///   merged 8-bit planar YUV formats use
///   ([`planar_dual_filter_resample`](planar_resample::planar_dual_filter_resample)),
///   then `luma_u16` zero-extends each resampled Y byte. This keeps the
///   8-bit planar YUVA `luma` / `luma_u16` **byte-identical** to the no-alpha
///   `Yuv420p` / `Yuv422p` / `Yuv444p` filter of the same Y plane: a signed
///   kernel rounds and clamps `[0, 255]` on the 8bpc coefficient grid rather
///   than overshooting at full `u16` precision, so merely attaching an alpha
///   plane cannot change the luma. The 8-bit luma is handled inline here (the
///   shared emit is called with the luma outputs detached) so the shared
///   `u16` binning stays untouched for the packed callers.
///
/// `SRC_BITS` is the source's native Y / colour depth (`8` for `Vuya` /
/// `Vuyx` and the 8-bit planar YUVA). A signed kernel (CatmullRom /
/// Lanczos3) overshoots a legal edge, so for the `u16`-luma path the shared
/// emit clamps both the native-u16 colour and the native Y to the native max
/// `(1 << SRC_BITS) - 1` for a sub-16-bit source before any u16 copy / RGBA
/// expansion / u8 narrowing — keeping `rgb_u16` / `rgba_u16` /
/// `luma_u16 <= (1 << SRC_BITS) - 1` and a clipped-high edge mapping to `255`
/// rather than wrapping. (The 8-bit planar YUVA `u8`-luma path needs no such
/// clamp — the `FilterStream<u8>` already finalizes to the native `[0, 255]`
/// range. `Vuya` / `Vuyx` expose no u16 colour outputs, so the colour clamp
/// is exercised only by the native-Y luma there; both clamps match the
/// in-range area path exactly.)
///
/// The const `ZEXT_U16_COLOR` selects how the u16 colour outputs
/// (`rgb_u16` / `rgba_u16`) are derived:
///
/// - **`false` (every genuine-native-u16 caller — `Vuya` / `Vuyx` expose no
///   u16 colour, and the high-bit `Ayuv64` / `Ya16` are truly 16-bit):** the
///   shared emit bins an **independent** native-u16 colour stream
///   (`convert_rgba_u16` → `rgba_filter_stream_u16`), so `rgba_u16` is the
///   native-u16 filter, clamped to the native max — never a widening of the
///   u8 colour. `Ya16` keeps this: it is genuinely 16-bit, so its native u16
///   colour IS the native u16 filter (matching its area path).
/// - **`true` (the packed 8-bit `Ya8` gray+alpha caller only):** `Ya8` is an
///   8-bit source, so its native-depth colour is the binned **u8** colour
///   zero-extended — `rgba_u16 == rgba as u16`, `rgb_u16 == rgb as u16`,
///   byte-for-byte the contract `Ya8`'s area path
///   ([`packed_rgba_resample`]) emits (it likewise zero-extends the binned
///   u8, never an independent native-u16 bin). So **no** independent u16
///   colour stream is created or fed (`need_colour_u16` is forced false; the
///   `convert_rgba_u16` closure is dead and no native-u16 colour scratch is
///   sized) — the u8 colour binning emits the widened u16 outputs inline.
///   This makes the `Ya8` **filter** u16 colour byte-identical to its **area**
///   u16 colour, and to `rgba as u16`, closing a filter-vs-area /
///   u8-vs-u16 contract split (the independent u16 filter could differ from
///   `rgba as u16` by up to 1 LSB near a signed-kernel overshoot, since the
///   u8 and u16 streams clamp their H-pass intermediates differently). The
///   const-asserts `SRC_BITS == 8`; only valid for an 8-bit source.
///
/// Atomic preflight (mirrors [`packed_yuva444_resample`]): a no-output call
/// returns before any freeze; a single [`frozen_outputs_check`] over the
/// full output set, then a single sequence check on whichever stream is fed
/// every row **before any allocation** (an out-of-sequence first row is
/// rejected before the freeze, storing no snapshot to poison a retry; on a
/// later row the freeze runs first so a mid-frame output change trips
/// `ResampleOutputsChanged`), then every stream and source-width scratch is
/// created before the first feed — so a failure mutates no caller output.
///
/// The `NATIVE_LUMA_U8` u8-luma path takes its native Y either from the
/// caller's contiguous plane (`y_row_u8` non-empty — the planar YUVA callers,
/// whose Y plane is already contiguous, zero copy) or, when `y_row_u8` is
/// empty (the **packed** `Ya8` gray+alpha caller, whose Y is interleaved in
/// `[Y, A]`), by de-interleaving into the caller's `y_u8_scratch` via
/// `deinterleave_y_u8` — staged after the sequence check, so the staging
/// allocation cannot mask an out-of-sequence first row.
#[cfg(any(feature = "yuv-444-packed", feature = "yuva", feature = "gray"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuva444_filter_resample<
  const SRC_BITS: u32,
  const NATIVE_LUMA_U8: bool,
  const ZEXT_U16_COLOR: bool,
>(
  rgba_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  rgba_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  luma_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  luma_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<u8>,
  rgb_drop_scratch: &mut Vec<u8>,
  rgba_scratch_u16: &mut Vec<u16>,
  color_scratch_u16: &mut Vec<u16>,
  luma_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  // The contiguous native-Y plane fed to the `u8` luma stream when
  // `NATIVE_LUMA_U8` is set and the source has one (the 8-bit planar YUVA
  // callers — zero copy). Empty for the packed `Ya8` caller (whose Y is
  // interleaved): the u8-luma path then de-interleaves into `y_u8_scratch`
  // via `deinterleave_y_u8` instead. Ignored entirely on the `u16`-luma path
  // (`NATIVE_LUMA_U8 = false`, the packed `Vuya` / `Vuyx` / `Ya16` callers),
  // which routes luma through `deinterleave_y` + the `u16` stream.
  y_row_u8: &[u8],
  // Source-width `u8` staging for the packed `NATIVE_LUMA_U8` caller's native
  // Y (`Ya8`): used only when `y_row_u8` is empty, sized and filled after the
  // sequence check. `None` for every caller with a contiguous `y_row_u8` or
  // on the `u16`-luma path.
  y_u8_scratch: Option<&mut Vec<u8>>,
  convert_rgba_u8: impl FnOnce(&mut [u8]),
  convert_rgba_u16: impl FnOnce(&mut [u16]),
  deinterleave_y: impl FnOnce(&mut [u16]),
  // De-interleaves the packed source's native Y into `y_u8_scratch` for the
  // `NATIVE_LUMA_U8` u8-luma path when `y_row_u8` is empty (`Ya8`). A no-op
  // for every caller that supplies a contiguous `y_row_u8` or rides the
  // `u16`-luma stream.
  deinterleave_y_u8: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16] for packed YUVA 4:4:4"
    )
  };
  const {
    assert!(
      !NATIVE_LUMA_U8 || SRC_BITS == 8,
      "NATIVE_LUMA_U8 (the u8 native-Y luma stream) is only valid for an 8-bit source"
    )
  };
  const {
    assert!(
      !ZEXT_U16_COLOR || SRC_BITS == 8,
      "ZEXT_U16_COLOR (u16 colour = zero-extended u8 colour) is only valid for an 8-bit source"
    )
  };
  // `ZEXT_U16_COLOR` (the packed 8-bit `Ya8` gray+alpha caller): the source
  // is 8-bit, so its native-depth colour is the binned u8 colour
  // zero-extended (`rgba_u16 == rgba as u16`) — matching the area path's
  // contract exactly — NOT an independent native-u16 binning. So the u16
  // colour outputs ride the u8 colour stream (the u8 binning emits them
  // widened inline in the shared emit) and the independent native-u16 stream
  // is never created or fed. Every other caller has `ZEXT_U16_COLOR == false`
  // and keeps the independent native-u16 binning byte-for-byte.
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change. A
  // `Yuva420p` sink can carry a `Bicublin` resampler, so this fence is live.
  plan.ensure_single_kernel_filter()?;
  let zext_u16_color = ZEXT_U16_COLOR && (rgb_u16.is_some() || rgba_u16.is_some());
  let need_colour_u8 = rgb.is_some() || rgba.is_some() || hsv.is_some() || zext_u16_color;
  let need_colour_u16 = (rgb_u16.is_some() || rgba_u16.is_some()) && !ZEXT_U16_COLOR;
  let need_luma = luma.is_some() || luma_u16.is_some();

  let (fh, fv) = (
    plan
      .filter_h()
      .expect("filter plan carries horizontal windows"),
    plan
      .filter_v()
      .expect("filter plan carries vertical windows"),
  );

  // Single sequence check before any allocation, on whichever attached
  // stream is fed every row (all active streams advance in lockstep). For the
  // 8-bit planar YUVA path luma rides the `u8` stream; for the packed path it
  // rides the `u16` stream.
  let expected = if need_colour_u8 {
    rgba_filter_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_colour_u16 {
    rgba_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if need_luma {
    if NATIVE_LUMA_U8 {
      luma_filter_stream.as_ref().map_or(0, |s| s.next_y())
    } else {
      luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
    }
  } else {
    return Ok(());
  };
  // First row: reject an out-of-sequence row before the freeze so a
  // rejected first row stores no snapshot that would poison a retry. On a
  // later row the freeze runs first (below).
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  // Create the filter streams (post-sequence-check), each running the full
  // output grid against its own source grid. The colour streams carry 4
  // interleaved channels, the luma stream 1. For the 8-bit planar YUVA path
  // (`NATIVE_LUMA_U8`) luma is a `u8` stream over the contiguous native-Y
  // plane — identical to the merged no-alpha planar YUV formats; for the
  // packed path it is a `u16` stream over the de-interleaved native Y.
  if need_colour_u8 && rgba_filter_stream.is_none() {
    *rgba_filter_stream = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_colour_u16 && rgba_filter_stream_u16.is_none() {
    *rgba_filter_stream_u16 = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_luma {
    if NATIVE_LUMA_U8 {
      if luma_filter_stream.is_none() {
        *luma_filter_stream = Some({
          let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
          crate::resample::try_box(stream).map_err(|_| {
            MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
              crate::resample::PlanGeometry::new(
                plan.src_w(),
                plan.src_h(),
                plan.out_w(),
                plan.out_h(),
              ),
            ))
          })?
        });
      }
    } else if luma_filter_stream_u16.is_none() {
      *luma_filter_stream_u16 = Some({
        let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
        crate::resample::try_box(stream).map_err(|_| {
          MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
            crate::resample::PlanGeometry::new(
              plan.src_w(),
              plan.src_h(),
              plan.out_w(),
              plan.out_h(),
            ),
          ))
        })?
      });
    }
  }

  // Resolve the `NATIVE_LUMA_U8` native-Y source in the preflight (before any
  // feed, so its allocation stays in the atomic preflight): the caller's
  // contiguous plane when supplied (planar YUVA, zero copy), else the packed
  // `Ya8` caller's interleaved Y de-interleaved into `y_u8_scratch`. Held and
  // fed after the colour emit below.
  let native_y_u8: Option<&[u8]> = if NATIVE_LUMA_U8 && need_luma {
    if y_row_u8.is_empty() {
      let scratch = source_luma_scratch(
        y_u8_scratch.expect("packed NATIVE_LUMA_U8 caller supplies a u8 Y scratch"),
        w,
        plan,
      )?;
      deinterleave_y_u8(scratch);
      Some(&scratch[..w])
    } else {
      Some(&y_row_u8[..w])
    }
  } else {
    None
  };

  // Stage + feed + emit the colour outputs. Straight alpha only
  // (`premult = false`); shared with the area path so the
  // convert-then-resolve-then-emit tail is identical (the only difference is
  // the stream kind built above). The native-max clamp inside clips a
  // sub-16-bit signed-kernel overshoot. For the 8-bit planar YUVA path the
  // luma is detached here (passed as `&mut None`) and handled inline below on
  // the `u8` stream; for the packed path the shared emit owns the `u16` luma
  // binning.
  if NATIVE_LUMA_U8 {
    packed_yuva444_feed_emit::<_, _, crate::resample::FilterStream<u16>, SRC_BITS, ZEXT_U16_COLOR>(
      rgba_filter_stream.as_mut(),
      rgba_filter_stream_u16.as_mut(),
      None,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      &mut None,
      &mut None,
      hsv,
      rgba_scratch,
      rgb_drop_scratch,
      rgba_scratch_u16,
      color_scratch_u16,
      luma_scratch_u16,
      w,
      plan,
      idx,
      use_simd,
      false,
      convert_rgba_u8,
      convert_rgba_u16,
      deinterleave_y,
    )?;

    // Native-Y luma on the `u8` stream — the Y row resolved in the preflight
    // (the caller's contiguous plane, or the de-interleaved packed Y) is fed
    // here, so the only fallible step was the stream creation + scratch sizing
    // above, keeping the call atomic. The emit mirrors the merged no-alpha
    // planar YUV path exactly: `luma` copies each finalized `u8` Y row,
    // `luma_u16` zero-extends it. An 8-bit `u8` stream finalizes to the full
    // `u8` range — the native range — so no clamp is needed (unlike the `u16`
    // luma path's native-max clip).
    if let Some(y_src) = native_y_u8 {
      let stream = luma_filter_stream
        .as_mut()
        .expect("created in the preflight when need_luma");
      let ow = plan.out_w();
      stream.feed_row(idx, y_src, use_simd, |oy, out_row| {
        if let Some(buf) = luma.as_deref_mut() {
          buf[oy * ow..(oy + 1) * ow].copy_from_slice(out_row);
        }
        if let Some(buf) = luma_u16.as_deref_mut() {
          for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(out_row) {
            *dst = src as u16;
          }
        }
      })?;
    }
    Ok(())
  } else {
    packed_yuva444_feed_emit::<_, _, _, SRC_BITS, ZEXT_U16_COLOR>(
      rgba_filter_stream.as_mut(),
      rgba_filter_stream_u16.as_mut(),
      luma_filter_stream_u16.as_mut(),
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgba_scratch,
      rgb_drop_scratch,
      rgba_scratch_u16,
      color_scratch_u16,
      luma_scratch_u16,
      w,
      plan,
      idx,
      use_simd,
      false,
      convert_rgba_u8,
      convert_rgba_u16,
      deinterleave_y,
    )
  }
}

/// Resets the packed-YUVA area streams (`rgba_stream`, `rgba_stream_u16`,
/// `luma_stream_u16`) AND the filter streams (`rgba_filter_stream`,
/// `rgba_filter_stream_u16`, `luma_filter_stream_u16`) and clears the frozen
/// output / alpha-mode snapshots at the start of a new frame for an
/// alpha-aware high-bit planar YUVA sink. Both stream kinds are reset because
/// the plan kind picks which is fed (`Area` bins, `Filter` filters), so a
/// frame switching kind reuses the same fields; the unused kind's streams stay
/// `None` and the resets no-op. The alpha-mode snapshot is re-armed to the
/// sink's current mode so a per-frame `set_alpha_mode` change is accepted (and
/// a mid-frame change is rejected by [`check_frozen_alpha_mode`]).
#[cfg(feature = "yuva")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn reset_high_bit_yuva_streams<F: SourceFormat, R>(sink: &mut MixedSinker<'_, F, R>) {
  if let Some(stream) = sink.rgba_stream.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.rgba_stream_u16.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.luma_stream_u16.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.rgba_filter_stream.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.rgba_filter_stream_u16.as_mut() {
    stream.reset();
  }
  if let Some(stream) = sink.luma_filter_stream_u16.as_mut() {
    stream.reset();
  }
  sink.resample_outputs = None;
  // RFC #238 S6c: clear the per-frame frozen 4:2:0 horizontal chroma siting so
  // the next frame may pick either phase while a mid-frame flip stays rejected
  // (mirrors the planar `reset_high_bit_yuv_streams`). Untouched by the 4:2:2 /
  // 4:4:4 YUVA families that share this reset (they never set it).
  sink.frozen_chroma_centered = None;
  // RFC #238 S6f: clear the frozen VERTICAL (`Bottom`) chroma phase too, so the
  // next frame may pick Center or Bottom while a mid-frame Center ⇆ Bottom flip
  // stays rejected; and invalidate the high-bit YUVA bottom-sited vertical
  // lookback via its validity tag (`chroma_prev_row`) so frame N+1's first even
  // row can never box-blend frame N's last chroma row. The `chroma_prev_u16`
  // bytes are re-overwritten before any trusted read, so only the tag is
  // load-bearing (mirrors the planar `reset_high_bit_yuv_streams`). Both stay
  // `None` for the non-`Bottom` / 4:2:2 / 4:4:4 YUVA families that share this
  // reset (they never set them).
  sink.frozen_chroma_bottom_v = None;
  sink.chroma_prev_row = None;
  // RFC #238 Top: clear the frozen VERTICAL (`Top`, `v = 0`) chroma phase so the
  // next frame may pick Top while a mid-frame flip to / from Top stays rejected,
  // and drop any held Top forward-delay odd row so frame N's last deferred colour
  // row never emits into frame N+1 (its held `chroma_top_y_u16` / `chroma_top_a_u16`
  // bytes are left as-is, re-overwritten before any trusted read). Both stay
  // `None` for the non-`Top` / 4:2:2 / 4:4:4 YUVA families that share this reset.
  sink.frozen_chroma_top_v = None;
  sink.chroma_top_pending = None;
  sink.frozen_alpha_mode = Some(sink.alpha_mode);
}

// Test-only allocation failpoint for the wrapper-owned Y / U / V de-pack
// scratch grow in `y2xx_process_native`. Armed, the FIRST (Y) scratch grow of
// an output-bearing row returns the crate's recoverable `AllocationFailed`
// WITHOUT growing — so the atomicity regressions can prove the join's pre-feed
// preflight (out-of-sequence / frozen-output) runs BEFORE this fallible grow.
// Mirrors the high-bit semi-planar `FORCE_P2XX_ALLOC_FAILURE`. Strictly
// test-only — the non-test build compiles this away entirely.
#[cfg(all(test, feature = "std", feature = "y2xx", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_Y2XX_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the wrapper de-pack scratch allocation failpoint for the **next**
/// output-bearing high-bit packed 4:2:2 native row on the current thread. The
/// flag is consumed (take-on-read) by the first fallible scratch grow that row
/// reaches, so it fires exactly once and cannot leak into a later test.
/// Test-only.
#[cfg(all(test, feature = "std", feature = "y2xx", feature = "yuv-planar"))]
pub(crate) fn arm_y2xx_alloc_failure() {
  FORCE_Y2XX_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Grows a wrapper-owned de-pack scratch to `len` `u16` under the planner's
/// recoverable-allocation contract, optionally firing the test-only failpoint
/// (`fail = true` only on the FIRST grow of an output-bearing row). Runs after
/// the join's preflight clears, so a rejected row never reaches it. The
/// high-bit packed 4:2:2 twin of the semi-planar `grow_depack_scratch`.
#[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn grow_y2xx_depack_scratch(
  scratch: &mut Vec<u16>,
  len: usize,
  fail: bool,
  w: usize,
  h: usize,
  plan: &crate::resample::ResamplePlan,
) -> Result<(), MixedSinkerError> {
  // `fail` is consumed by the caller; on the non-test build it is `false` and
  // the whole branch compiles away.
  let _ = fail;
  if scratch.len() < len {
    #[cfg(all(test, feature = "std", feature = "y2xx", feature = "yuv-planar"))]
    if fail && FORCE_Y2XX_ALLOC_FAILURE.with(|f| f.take()) {
      return Err(MixedSinkerError::Resample(
        crate::resample::ResampleError::AllocationFailed(crate::resample::PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )),
      ));
    }
    scratch
      .try_reserve_exact(len - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        ))
      })?;
    scratch.resize(len, 0);
  }
  Ok(())
}

/// Native fast-tier decimator for the **high-bit packed 4:2:2** YUV family
/// ([`Y210`](crate::source::Y210) / [`Y212`](crate::source::Y212) /
/// [`Y216`](crate::source::Y216)): bins the native Y / U / V planes straight to
/// the output grid and converts once per output row at output resolution. The
/// PACKED, MSB-aligned analog of the high-bit semi-planar 4:2:2
/// [`p2xx_process_native`](subsampled_4_2_2_high_bit) — same reuse of the
/// high-bit non-4:2:0 PLANAR join verbatim
/// ([`planar_high_bit_native::yuv_planar16_process_native`]) — composed with the
/// packed de-pack of the 8-bit packed 4:2:2
/// [`packed_yuv_8bit`]'s `packed_yuv422_process_native`.
///
/// THE COMPOSITION, per row: wire-decode (`from_le` / `from_be`) AND de-pack the
/// MSB alignment (`>> (16 - BITS)`) AND de-interleave the YUYV-ordered u16 words
/// into wrapper-owned host-native LOGICAL u16 scratch, then delegate with
/// `BE = HOST_NATIVE_BE` so the delegate's own decode is a no-op load on every
/// host. Y2xx packs four u16 words per 2-pixel group: `Y₀` at word 0, `U` at
/// word 1, `Y₁` at word 2, `V` at word 3 (YUYV order). The de-pack hits Y AND U
/// AND V; at `BITS = 16` (`Y216`) the shift is `>> 0` (a harmless no-op — the
/// 10/12 tests guard the live shift).
///
/// 4:2:2 geometry: the chroma plane is `w/2 × h` (horizontal-only subsample,
/// vertical cadence `chroma_vsub = 1`), so a chroma row feeds EVERY colour Y row.
/// The delegate builds its chroma grid against the same output geometry via the
/// `area_chroma_422(w/2, h, out_w, out_h, chroma_h_phase, 0.0)` closure.
///
/// `chroma_h_phase` is the horizontal chroma sampling phase folded into the
/// chroma area weights (`ResamplePlan::area_chroma_422`): `0.25` for the centered
/// 4:2:2 group (`chroma_422_center_sited_h`), `0.0` for co-sited / unspecified. At
/// phase `0.0` the folded plan is byte-identical to the plain `area` plan, so the
/// co-sited output is untouched.
///
/// Native-depth clamp + luma-only lazy chroma both carry through the reused join
/// (the #222 fixes): the join clamps luma `src.min((1 << BITS) - 1) >> (BITS - 8)`
/// for the u8 `luma`, emits the SAME clamped binned Y as the native-depth
/// `luma_u16` (host-native u16, not narrowed), and the colour emit clamps to
/// native max — and the wrapper converts MSB→LSB BEFORE that clamp, so an
/// MSB-packed sample (capped at `MASK` by the de-pack) never wraps; under
/// `need_color == false` only Y is de-packed and the join is handed empty U / V
/// slices, so a luma-only / `luma_u16`-only sink plans no chroma state. Because
/// the join now emits `luma_u16`, attaching it routes through this native tier
/// (no row-stage fallback), keeping the rgb colour semantics native.
///
/// Atomicity (the nv12 / high-bit lesson): the join's COMPLETE pre-feed preflight
/// runs FIRST — `Ok(false)` no-op short-circuit, first-row out-of-sequence,
/// frozen-output — BEFORE any fallible scratch grow, so a rejected row returns
/// its deterministic typed error (`OutOfSequenceRow` / `ResampleOutputsChanged`),
/// never `AllocationFailed`, and touches no caller output. The de-pack into
/// scratch is infallible and happens only after the preflight clears; the
/// delegate re-runs the identical preflight (idempotent) and owns the binning +
/// conversion.
#[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn y2xx_process_native<const BITS: u32, const BE: bool>(
  plan: &crate::resample::ResamplePlan,
  native_planar_u16: &mut Option<std::boxed::Box<NativePlanarYuvU16>>,
  y_scratch: &mut Vec<u16>,
  u_scratch: &mut Vec<u16>,
  v_scratch: &mut Vec<u16>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  packed: &[u16],
  chroma_h_phase: f64,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      BITS > 8 && BITS <= 16,
      "BITS must be in (8, 16] for high-bit packed 4:2:2 Y2xx"
    )
  };
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
  // 4:2:2 chroma is half-width, full-height: `chroma_w = w / 2`, a chroma row
  // per Y row (`chroma_vsub = 1`).
  let cw = w / 2;

  // Run the planar join's COMPLETE compare-only pre-feed rejection preflight
  // FIRST — no-output short-circuit, first-row out-of-sequence, AND output-set
  // compare (mid-frame output change) — BEFORE any fallible scratch grow below,
  // so every rejection returns its deterministic typed error and leaves the
  // wrapper scratch untouched (the crate's preflight-atomicity contract).
  // Compare-only (no output-set freeze): the de-pack grows below AND the
  // delegate's own colour / source scratch grows all stay pre-commit, and the
  // delegate performs the SINGLE freeze once every pre-feed allocation has
  // succeeded — so a LATER grow (here or in the delegate) failing leaves
  // `resample_outputs` untouched and the row retryable.
  if !native_planar_hb_preflight_check_only(
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // Grow the wrapper de-pack scratch under the planner's recoverable contract —
  // Y always, U / V only on a colour row (4:2:2: every Y row is a chroma row
  // when colour is wanted). All grows precede the infallible de-pack and the
  // delegate call. The failpoint fires on the FIRST (Y) grow only.
  grow_y2xx_depack_scratch(y_scratch, w, true, w, h, plan)?;
  if need_color {
    grow_y2xx_depack_scratch(u_scratch, cw, false, w, h, plan)?;
    grow_y2xx_depack_scratch(v_scratch, cw, false, w, h, plan)?;
  }

  // De-pack the interleaved YUYV-ordered MSB-aligned u16 words into host-native
  // LOGICAL scratch. Each 2-pixel group is four words `Y₀, U, Y₁, V`: decode the
  // wire endianness, then shift the active high `BITS` down to the low `BITS`
  // (`>> (16 - BITS)`; `>> 0` at BITS = 16). Y is always de-packed (the join
  // bins Y for both luma and colour); U / V only on a colour row. Everything
  // past here is infallible.
  let depack = |s: u16| -> u16 {
    let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
    logical >> (16 - BITS)
  };
  for (i, group) in packed.chunks_exact(4).enumerate() {
    y_scratch[i * 2] = depack(group[0]);
    y_scratch[i * 2 + 1] = depack(group[2]);
  }
  if need_color {
    for (i, group) in packed.chunks_exact(4).enumerate() {
      u_scratch[i] = depack(group[1]);
      v_scratch[i] = depack(group[3]);
    }
  }

  // Delegate to the planar high-bit non-4:2:0 join with `BE = HOST_NATIVE_BE`
  // so its internal decode is a no-op on the already-native scratch, at the
  // 4:2:2 chroma geometry (`chroma_vsub = 1`, `chroma_w = w / 2`). Empty U / V
  // on luma-only rows (the join reads chroma only under colour). Unlike the
  // high-bit semi-planar wrapper, Y2xx threads its real `luma_u16` buffer: the
  // join now emits the native-depth `luma_u16` (the clamped binned Y), so
  // attaching `luma_u16` no longer falls the whole pipeline back to row-stage —
  // the route stays native and rgb keeps native colour semantics.
  let (u_plane, v_plane): (&[u16], &[u16]) = if need_color {
    (&u_scratch[..cw], &v_scratch[..cw])
  } else {
    (&[], &[])
  };
  planar_high_bit_native::yuv_planar16_process_native::<BITS, HIGH_BIT_HOST_NATIVE_BE>(
    plan,
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    rgb_scratch_u16,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    cw,
    || {
      crate::resample::ResamplePlan::area_chroma_422(
        cw,
        h,
        plan.out_w(),
        plan.out_h(),
        chroma_h_phase,
        0.0,
      )
    },
    use_simd,
  )
}

/// Fallible pre-feed reserve for the packed Y2xx centered-siting de-interleave
/// scratch: grows the full-width Y buffer (`width`) and the half-width U / V
/// buffers (`width / 2` each) so the later infallible de-interleave + 4:4:4
/// decode reuse already-sized buffers. The `u16` sibling of the 8-bit packed
/// [`packed_yuv_8bit`]'s `reserve_packed_center_chroma`. Split from the
/// de-interleave so it runs before any output row is written — an allocator
/// refusal must leave the output frame untouched, never partially mutated — and
/// reuses the native tier's de-pack scratch (`width` Y + `width / 2` U / V), so a
/// centered colour row and an area native row never both run in one `process`
/// call. A grow refusal is the typed, recoverable
/// [`ResampleError::AllocationFailed`]; `height` feeds the payload.
///
/// [`ResampleError::AllocationFailed`]: crate::resample::ResampleError::AllocationFailed
#[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
fn reserve_y2xx_center_planes(
  y_full: &mut Vec<u16>,
  u_half: &mut Vec<u16>,
  v_half: &mut Vec<u16>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  let grow = |scratch: &mut Vec<u16>, len: usize| -> Result<(), MixedSinkerError> {
    if scratch.len() < len {
      scratch
        .try_reserve_exact(len - scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
            crate::resample::PlanGeometry::new(width, height, width, height),
          ))
        })?;
      scratch.resize(len, 0);
    }
    Ok(())
  };
  grow(y_full, width)?;
  let cw = width / 2;
  grow(u_half, cw)?;
  grow(v_half, cw)?;
  Ok(())
}

/// Reconstructs full-width chroma for a packed Y2xx centered-siting colour row.
/// De-interleaves the packed `[Y0 U Y1 V]` words into a full-width Y plane and
/// half-width U / V planes, shifting each MSB-aligned sample down to the low
/// `BITS` (`>> (16 - BITS)`) while keeping the source's wire byte order — so the
/// planes match the LSB-aligned low-packed layout the [`reconstruct_chroma`]
/// upsample and the high-bit 4:4:4 decode expect, both of which re-decode the
/// endianness (`be`) themselves. Then phase-0.5 upsamples U / V into `chroma_full`,
/// returning the `(u_full, v_full)` slices the 4:4:4 decode reads (the de-packed
/// `y_full` is left in the caller's scratch, read as `&y_full[..w]`). Every
/// fallible grow (chroma scratch + de-pack planes) precedes the infallible
/// de-interleave, so an allocator refusal leaves the output frame untouched (the
/// crate's preflight-ordering atomicity contract). Reuses the native tier's
/// de-pack scratch, so a centered colour row and an area native row never both run
/// in one `process` call.
#[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn y2xx_center_reconstruct<'c, const BITS: u32>(
  packed: &[u16],
  y_full: &mut Vec<u16>,
  u_half: &mut Vec<u16>,
  v_half: &mut Vec<u16>,
  chroma_full: &'c mut Vec<u16>,
  w: usize,
  h: usize,
  be: bool,
  use_simd: bool,
) -> Result<(&'c [u16], &'c [u16]), MixedSinkerError> {
  let cw = w / 2;
  subsampled_4_2_0_high_bit::reserve_420_chroma_full_u16(chroma_full, w, h)?;
  reserve_y2xx_center_planes(y_full, u_half, v_half, w, h)?;
  // Shift the active high `BITS` down to the low `BITS`, then re-encode in the
  // source wire byte order so the reconstruct + 4:4:4 decode (which apply `be`)
  // see LSB-aligned low-packed samples. At `BITS = 16` this is a no-op round-trip.
  let repack = |wire: u16| -> u16 {
    let logical = (if be {
      u16::from_be(wire)
    } else {
      u16::from_le(wire)
    }) >> (16 - BITS);
    if be { logical.to_be() } else { logical.to_le() }
  };
  for (i, group) in packed.chunks_exact(4).take(cw).enumerate() {
    y_full[2 * i] = repack(group[0]);
    y_full[2 * i + 1] = repack(group[2]);
    u_half[i] = repack(group[1]);
    v_half[i] = repack(group[3]);
  }
  Ok(reconstruct_chroma(
    ChromaU16::<BITS> { big_endian: be },
    chroma_full,
    &u_half[..cw],
    &v_half[..cw],
    w,
    use_simd,
  ))
}

// Test-only allocation failpoint for the wrapper-owned Y / U / V de-pack scratch
// grow in `v210_process_native`. Armed, the FIRST (Y) scratch grow of an
// output-bearing row returns the crate's recoverable `AllocationFailed` WITHOUT
// growing — so the atomicity regressions can prove the join's pre-feed preflight
// (out-of-sequence / frozen-output) runs BEFORE this fallible grow. Mirrors
// `FORCE_Y2XX_ALLOC_FAILURE`. Strictly test-only — the non-test build compiles
// this away entirely.
#[cfg(all(test, feature = "std", feature = "v210", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_V210_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the wrapper de-pack scratch allocation failpoint for the **next**
/// output-bearing V210 native row on the current thread. The flag is consumed
/// (take-on-read) by the first fallible scratch grow that row reaches, so it
/// fires exactly once and cannot leak into a later test. Test-only.
#[cfg(all(test, feature = "std", feature = "v210", feature = "yuv-planar"))]
pub(crate) fn arm_v210_alloc_failure() {
  FORCE_V210_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Grows a wrapper-owned de-pack scratch to `len` `u16` under the planner's
/// recoverable-allocation contract, optionally firing the test-only failpoint
/// (`fail = true` only on the FIRST grow of an output-bearing row). Runs after
/// the join's preflight clears, so a rejected row never reaches it. The V210 twin
/// of [`grow_y2xx_depack_scratch`].
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn grow_v210_depack_scratch(
  scratch: &mut Vec<u16>,
  len: usize,
  fail: bool,
  w: usize,
  h: usize,
  plan: &crate::resample::ResamplePlan,
) -> Result<(), MixedSinkerError> {
  // `fail` is consumed by the caller; on the non-test build it is `false` and
  // the whole branch compiles away.
  let _ = fail;
  if scratch.len() < len {
    #[cfg(all(test, feature = "std", feature = "v210", feature = "yuv-planar"))]
    if fail && FORCE_V210_ALLOC_FAILURE.with(|f| f.take()) {
      return Err(MixedSinkerError::Resample(
        crate::resample::ResampleError::AllocationFailed(crate::resample::PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )),
      ));
    }
    scratch
      .try_reserve_exact(len - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        ))
      })?;
    scratch.resize(len, 0);
  }
  Ok(())
}

/// Native fast-tier decimator for the **exotic 10-bit packed 4:2:2** YUV format
/// ([`V210`](crate::source::V210)): bins the native Y / U / V planes straight to
/// the output grid and converts once per output row at output resolution. The
/// fixed-`BITS = 10` sibling of the high-bit packed 4:2:2
/// [`y2xx_process_native`] — same reuse of the high-bit non-4:2:0 PLANAR join
/// verbatim ([`planar_high_bit_native::yuv_planar16_process_native`]) at the
/// 4:2:2 chroma geometry (`chroma_w = w / 2`, `chroma_vsub = 1`) — composed with
/// the V210 word de-pack in place of the Y2xx YUYV de-pack.
///
/// THE COMPOSITION, per row: wire-decode (`from_le` / `from_be`) AND bit-extract
/// the V210 word packing into wrapper-owned host-native LOGICAL u16 scratch, then
/// delegate with `BE = HOST_NATIVE_BE` so the delegate's own decode is a no-op
/// load on every host. V210 packs 12 × 10-bit samples per 16-byte word = four
/// 32-bit LE/BE words = 6 pixels: word 0 `[Cb0, Y0, Cr0]`, word 1 `[Y1, Cb1, Y2]`,
/// word 2 `[Cr1, Y3, Cb2]`, word 3 `[Y4, Cr2, Y5]` — each sample in the low 10
/// bits of its lane (`bits[9:0]`, `[19:10]`, `[29:20]`). The de-pack MIRRORS the
/// row-stage `unpack_v210_word` bit positions EXACTLY, so the binned planes are
/// the same logical samples the row-stage tier converts. Y is always de-packed
/// (the join bins Y for both luma and colour); U / V only on a colour row. The
/// 10-bit fields are `<= 1023 = (1 << 10) - 1` by construction, so the join's
/// native-depth clamp is a value no-op on encodable input.
///
/// Even-width + partial-last-word: the configured width must be even (4:2:2
/// chroma pair; the caller rejects odd widths as
/// [`WidthAlignment`](MixedSinkerError::WidthAlignment) before this runs). A width
/// not divisible by 6 ends on a partial word carrying only 2 or 4 valid pixels
/// (1 or 2 valid chroma pairs); the de-pack reads every word but writes only the
/// valid prefix (`y_scratch[..w]`, `u/v_scratch[..w/2]`), matching the row-stage
/// `tail_pixels` handling — the unused high samples in the partial word are never
/// binned.
///
/// Native-depth clamp + `luma_u16` emit + luma-only lazy chroma all carry through
/// the reused join: the join clamps `src.min(1023) >> 2` for the u8 `luma`, emits
/// the SAME clamped binned Y as the native-depth `luma_u16` (host-native u16, not
/// narrowed), and the colour emit clamps to 1023; under `need_color == false`
/// only Y is de-packed and the join is handed empty U / V slices, so a luma-only /
/// `luma_u16`-only sink plans no chroma state. Because the join now emits
/// `luma_u16`, attaching it routes through this native tier (no row-stage
/// fallback), keeping the rgb colour semantics native.
///
/// `chroma_h_phase` is the horizontal chroma sampling phase folded into the
/// chroma area weights (`ResamplePlan::area_chroma_422`): `0.25` for the centered
/// 4:2:2 group (`chroma_422_center_sited_h`), `0.0` for co-sited / unspecified. At
/// phase `0.0` the folded plan is byte-identical to the plain `area` plan, so the
/// co-sited output is untouched.
///
/// Atomicity mirrors [`y2xx_process_native`]: the join's COMPLETE pre-feed
/// preflight runs FIRST — `Ok(false)` no-op short-circuit, first-row
/// out-of-sequence, frozen-output — BEFORE any fallible scratch grow, so a
/// rejected row returns its deterministic typed error
/// (`OutOfSequenceRow` / `ResampleOutputsChanged`), never `AllocationFailed`, and
/// touches no caller output. The de-pack into scratch is infallible and happens
/// only after the preflight clears; the delegate re-runs the identical preflight
/// (idempotent) and owns the binning + conversion.
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn v210_process_native<const BE: bool>(
  plan: &crate::resample::ResamplePlan,
  native_planar_u16: &mut Option<std::boxed::Box<NativePlanarYuvU16>>,
  y_scratch: &mut Vec<u16>,
  u_scratch: &mut Vec<u16>,
  v_scratch: &mut Vec<u16>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  packed: &[u8],
  chroma_h_phase: f64,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  // V210 is fixed at 10-bit; the reused join is const-generic over BITS.
  const BITS: u32 = 10;
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
  // 4:2:2 chroma is half-width, full-height: `chroma_w = w / 2`, a chroma row per
  // Y row (`chroma_vsub = 1`).
  let cw = w / 2;

  // Run the planar join's COMPLETE compare-only pre-feed rejection preflight
  // FIRST — no-output short-circuit, first-row out-of-sequence, AND output-set
  // compare (mid-frame output change) — BEFORE any fallible scratch grow below,
  // so every rejection returns its deterministic typed error and leaves the
  // wrapper scratch untouched (the crate's preflight-atomicity contract).
  // Compare-only (no output-set freeze): the de-pack grows below AND the
  // delegate's own colour / source scratch grows all stay pre-commit, and the
  // delegate performs the SINGLE freeze once every pre-feed allocation has
  // succeeded — so a LATER grow (here or in the delegate) failing leaves
  // `resample_outputs` untouched and the row retryable.
  if !native_planar_hb_preflight_check_only(
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // Grow the wrapper de-pack scratch under the planner's recoverable contract —
  // Y always, U / V only on a colour row (4:2:2: every Y row is a chroma row when
  // colour is wanted). All grows precede the infallible de-pack and the delegate
  // call. The failpoint fires on the FIRST (Y) grow only.
  grow_v210_depack_scratch(y_scratch, w, true, w, h, plan)?;
  if need_color {
    grow_v210_depack_scratch(u_scratch, cw, false, w, h, plan)?;
    grow_v210_depack_scratch(v_scratch, cw, false, w, h, plan)?;
  }

  // De-pack the V210 word packing into host-native LOGICAL u16 scratch. Each
  // 16-byte word is four 32-bit LE/BE lanes carrying three 10-bit samples each:
  //   word 0 `[Cb0, Y0, Cr0]`  word 1 `[Y1, Cb1, Y2]`
  //   word 2 `[Cr1, Y3, Cb2]`  word 3 `[Y4, Cr2, Y5]`
  // (sample in `bits[9:0]` / `[19:10]` / `[29:20]` of its lane) — MIRRORING the
  // row-stage `unpack_v210_word` exactly. Decode the wire endianness per lane,
  // then mask `& 0x3FF`. Each full word emits 6 Y + 3 U + 3 V; a final partial
  // word (width not a multiple of 6) writes only its valid prefix — `take_words`
  // caps the per-word output index at `w` (Y) / `cw` (U / V), so the unused high
  // samples are never binned. Y is always de-packed; U / V only on a colour row.
  // Everything past here is infallible.
  let load = |word: &[u8], lane: usize| -> u32 {
    let bytes = [
      word[lane * 4],
      word[lane * 4 + 1],
      word[lane * 4 + 2],
      word[lane * 4 + 3],
    ];
    if BE {
      u32::from_be_bytes(bytes)
    } else {
      u32::from_le_bytes(bytes)
    }
  };
  for (wi, word) in packed.chunks_exact(16).enumerate() {
    let px = wi * 6;
    let w0 = load(word, 0);
    let w1 = load(word, 1);
    let w2 = load(word, 2);
    let w3 = load(word, 3);
    // The six Y samples across the four lanes (`unpack_v210_word` order).
    let ys = [
      ((w0 >> 10) & 0x3FF) as u16, // Y0
      (w1 & 0x3FF) as u16,         // Y1
      ((w1 >> 20) & 0x3FF) as u16, // Y2
      ((w2 >> 10) & 0x3FF) as u16, // Y3
      (w3 & 0x3FF) as u16,         // Y4
      ((w3 >> 20) & 0x3FF) as u16, // Y5
    ];
    for (k, &y) in ys.iter().enumerate() {
      if px + k < w {
        y_scratch[px + k] = y;
      }
    }
    if need_color {
      let cu = px / 2;
      // Cb0 / Cb1 / Cb2 and Cr0 / Cr1 / Cr2 — one chroma pair per 2-pixel pair.
      let us = [
        (w0 & 0x3FF) as u16,         // Cb0
        ((w1 >> 10) & 0x3FF) as u16, // Cb1
        ((w2 >> 20) & 0x3FF) as u16, // Cb2
      ];
      let vs = [
        ((w0 >> 20) & 0x3FF) as u16, // Cr0
        (w2 & 0x3FF) as u16,         // Cr1
        ((w3 >> 10) & 0x3FF) as u16, // Cr2
      ];
      for k in 0..3 {
        if cu + k < cw {
          u_scratch[cu + k] = us[k];
          v_scratch[cu + k] = vs[k];
        }
      }
    }
  }

  // Delegate to the planar high-bit non-4:2:0 join with `BE = HOST_NATIVE_BE` so
  // its internal decode is a no-op on the already-native scratch, at the 4:2:2
  // chroma geometry (`chroma_vsub = 1`, `chroma_w = w / 2`). Empty U / V on
  // luma-only rows (the join reads chroma only under colour). V210 threads its
  // real `luma_u16` buffer: the join emits the native-depth `luma_u16` (the
  // clamped binned Y), so attaching `luma_u16` keeps the native route and rgb
  // keeps native colour semantics.
  let (u_plane, v_plane): (&[u16], &[u16]) = if need_color {
    (&u_scratch[..cw], &v_scratch[..cw])
  } else {
    (&[], &[])
  };
  planar_high_bit_native::yuv_planar16_process_native::<BITS, HIGH_BIT_HOST_NATIVE_BE>(
    plan,
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    rgb_scratch_u16,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    cw,
    || {
      crate::resample::ResamplePlan::area_chroma_422(
        cw,
        h,
        plan.out_w(),
        plan.out_h(),
        chroma_h_phase,
        0.0,
      )
    },
    use_simd,
  )
}

/// Fallible pre-feed reserve for the packed V210 centered-siting de-pack scratch:
/// grows the full-width Y buffer (`width`) and the half-width U / V buffers
/// (`width / 2` each) so the later infallible de-pack + 4:4:4 decode reuse
/// already-sized buffers. The V210 twin of [`reserve_y2xx_center_planes`]. Split
/// from the de-pack so it runs before any output row is written — an allocator
/// refusal must leave the output frame untouched, never partially mutated — and
/// reuses the native tier's de-pack scratch (`width` Y + `width / 2` U / V), so a
/// centered colour row and an area native row never both run in one `process`
/// call. A grow refusal is the typed, recoverable
/// [`ResampleError::AllocationFailed`]; `height` feeds the payload.
///
/// [`ResampleError::AllocationFailed`]: crate::resample::ResampleError::AllocationFailed
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
fn reserve_v210_center_planes(
  y_full: &mut Vec<u16>,
  u_half: &mut Vec<u16>,
  v_half: &mut Vec<u16>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  let grow = |scratch: &mut Vec<u16>, len: usize| -> Result<(), MixedSinkerError> {
    if scratch.len() < len {
      scratch
        .try_reserve_exact(len - scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
            crate::resample::PlanGeometry::new(width, height, width, height),
          ))
        })?;
      scratch.resize(len, 0);
    }
    Ok(())
  };
  grow(y_full, width)?;
  let cw = width / 2;
  grow(u_half, cw)?;
  grow(v_half, cw)?;
  Ok(())
}

/// Reconstructs full-width chroma for a packed V210 centered-siting colour row.
/// De-packs the V210 word packing (four 32-bit lanes per 16-byte word, each
/// carrying three low-10-bit samples in the `unpack_v210_word` order) into a
/// full-width Y plane and half-width U / V planes, re-encoding each already
/// logical 10-bit sample into the source's wire byte order (`be`) — so the planes
/// match the LSB-aligned low-packed layout the [`reconstruct_chroma`] upsample and
/// the 10-bit 4:4:4 decode expect, both of which re-decode the endianness (`be`)
/// themselves. Then phase-0.5 upsamples U / V into `chroma_full`, returning the
/// `(u_full, v_full)` slices the 4:4:4 decode reads (the de-packed `y_full` is
/// left in the caller's scratch, read as `&y_full[..w]`). Every fallible grow
/// (chroma scratch + de-pack planes) precedes the infallible de-pack, so an
/// allocator refusal leaves the output frame untouched (the crate's
/// preflight-ordering atomicity contract). Reuses the native tier's de-pack
/// scratch, so a centered colour row and an area native row never both run in one
/// `process` call. The V210 twin of [`y2xx_center_reconstruct`] — the same
/// reconstruct-and-decode, differing only in the word de-pack (the native tier's
/// bit-extract in place of the YUYV MSB-align).
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn v210_center_reconstruct<'c>(
  packed: &[u8],
  y_full: &mut Vec<u16>,
  u_half: &mut Vec<u16>,
  v_half: &mut Vec<u16>,
  chroma_full: &'c mut Vec<u16>,
  w: usize,
  h: usize,
  be: bool,
  use_simd: bool,
) -> Result<(&'c [u16], &'c [u16]), MixedSinkerError> {
  const BITS: u32 = 10;
  let cw = w / 2;
  subsampled_4_2_0_high_bit::reserve_420_chroma_full_u16(chroma_full, w, h)?;
  reserve_v210_center_planes(y_full, u_half, v_half, w, h)?;
  // Each bit-extracted 10-bit sample is already logical; re-encode it in the
  // source wire byte order so the reconstruct + 4:4:4 decode (which apply `be`)
  // see the LSB-aligned low-packed samples.
  let encode = |logical: u16| -> u16 { if be { logical.to_be() } else { logical.to_le() } };
  let load = |word: &[u8], lane: usize| -> u32 {
    let bytes = [
      word[lane * 4],
      word[lane * 4 + 1],
      word[lane * 4 + 2],
      word[lane * 4 + 3],
    ];
    if be {
      u32::from_be_bytes(bytes)
    } else {
      u32::from_le_bytes(bytes)
    }
  };
  for (wi, word) in packed.chunks_exact(16).enumerate() {
    let px = wi * 6;
    let w0 = load(word, 0);
    let w1 = load(word, 1);
    let w2 = load(word, 2);
    let w3 = load(word, 3);
    // The six Y samples across the four lanes (`unpack_v210_word` order).
    let ys = [
      ((w0 >> 10) & 0x3FF) as u16, // Y0
      (w1 & 0x3FF) as u16,         // Y1
      ((w1 >> 20) & 0x3FF) as u16, // Y2
      ((w2 >> 10) & 0x3FF) as u16, // Y3
      (w3 & 0x3FF) as u16,         // Y4
      ((w3 >> 20) & 0x3FF) as u16, // Y5
    ];
    for (k, &y) in ys.iter().enumerate() {
      if px + k < w {
        y_full[px + k] = encode(y);
      }
    }
    let cu = px / 2;
    // Cb0 / Cb1 / Cb2 and Cr0 / Cr1 / Cr2 — one chroma pair per 2-pixel pair.
    let us = [
      (w0 & 0x3FF) as u16,         // Cb0
      ((w1 >> 10) & 0x3FF) as u16, // Cb1
      ((w2 >> 20) & 0x3FF) as u16, // Cb2
    ];
    let vs = [
      ((w0 >> 20) & 0x3FF) as u16, // Cr0
      (w2 & 0x3FF) as u16,         // Cr1
      ((w3 >> 10) & 0x3FF) as u16, // Cr2
    ];
    for k in 0..3 {
      if cu + k < cw {
        u_half[cu + k] = encode(us[k]);
        v_half[cu + k] = encode(vs[k]);
      }
    }
  }
  Ok(reconstruct_chroma(
    ChromaU16::<BITS> { big_endian: be },
    chroma_full,
    &u_half[..cw],
    &v_half[..cw],
    w,
    use_simd,
  ))
}

/// The de-pack writes host-native LOGICAL u16 into the wrapper scratch BEFORE
/// the planar delegate, so the delegate's own `from_le` / `from_be` decode must
/// be a no-op load on every host: pass `BE = HOST_NATIVE_BE` (= `from_ne`).
/// Forwarding the source wire `BE` here would byte-swap the already-native
/// scratch on a big-endian target. Mirrors the high-bit semi-planar `p2xx`.
/// Shared by the high-bit packed 4:2:2 (`y2xx`) and 4:4:4 (`yuv-444-packed`
/// V410 / Xv36) native wrappers — both de-pack to host-native LOGICAL u16
/// before delegating. The exotic 10-bit packed 4:2:2 `v210` wrapper shares it
/// (it likewise bit-extracts to host-native LOGICAL u16 before delegating).
#[cfg(all(
  any(feature = "y2xx", feature = "yuv-444-packed", feature = "v210"),
  feature = "yuv-planar"
))]
const HIGH_BIT_HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// Test-only allocation failpoint for the wrapper-owned Y / U / V de-pack
// scratch grow in the packed 4:4:4 native wrappers (8-bit `packed_vuyx_process_native`
// and high-bit `packed_yuv444_hb_process_native`). Armed, the FIRST (Y) scratch
// grow of an output-bearing row returns the crate's recoverable `AllocationFailed`
// WITHOUT growing — so the atomicity regressions can prove the join's pre-feed
// preflight (out-of-sequence / frozen-output) runs BEFORE this fallible grow.
// Mirrors `FORCE_Y2XX_ALLOC_FAILURE`. A given sink runs exactly one element-type
// grow (u8 for Vuyx, u16 for V410 / Xv36), so the single flag is shared by both
// `grow_packed_444_*_scratch` helpers. Strictly test-only — the non-test build
// compiles this away entirely.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-444-packed",
  feature = "yuv-planar"
))]
std::thread_local! {
  static FORCE_PACKED_444_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the wrapper de-pack scratch allocation failpoint for the **next**
/// output-bearing packed 4:4:4 native row on the current thread. The flag is
/// consumed (take-on-read) by the first fallible scratch grow that row reaches,
/// so it fires exactly once and cannot leak into a later test. Test-only.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-444-packed",
  feature = "yuv-planar"
))]
pub(crate) fn arm_packed_444_alloc_failure() {
  FORCE_PACKED_444_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Grows a wrapper-owned **u8** de-pack scratch (the 8-bit Vuyx tier) to `len`
/// under the planner's recoverable-allocation contract, optionally firing the
/// test-only failpoint (`fail = true` only on the FIRST grow of an output-bearing
/// row). Runs after the join's preflight clears, so a rejected row never reaches
/// it. The 8-bit packed 4:4:4 twin of `grow_y2xx_depack_scratch`.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn grow_packed_444_u8_scratch(
  scratch: &mut Vec<u8>,
  len: usize,
  fail: bool,
  w: usize,
  h: usize,
  plan: &crate::resample::ResamplePlan,
) -> Result<(), MixedSinkerError> {
  let _ = fail;
  if scratch.len() < len {
    #[cfg(all(
      test,
      feature = "std",
      feature = "yuv-444-packed",
      feature = "yuv-planar"
    ))]
    if fail && FORCE_PACKED_444_ALLOC_FAILURE.with(|f| f.take()) {
      return Err(MixedSinkerError::Resample(
        crate::resample::ResampleError::AllocationFailed(crate::resample::PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )),
      ));
    }
    scratch
      .try_reserve_exact(len - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        ))
      })?;
    scratch.resize(len, 0);
  }
  Ok(())
}

/// Grows a wrapper-owned **u16** de-pack scratch (the high-bit V410 / Xv36 tier)
/// to `len` under the planner's recoverable-allocation contract, with the same
/// failpoint + ordering contract as [`grow_packed_444_u8_scratch`]. The high-bit
/// packed 4:4:4 twin of `grow_y2xx_depack_scratch`.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn grow_packed_444_u16_scratch(
  scratch: &mut Vec<u16>,
  len: usize,
  fail: bool,
  w: usize,
  h: usize,
  plan: &crate::resample::ResamplePlan,
) -> Result<(), MixedSinkerError> {
  let _ = fail;
  if scratch.len() < len {
    #[cfg(all(
      test,
      feature = "std",
      feature = "yuv-444-packed",
      feature = "yuv-planar"
    ))]
    if fail && FORCE_PACKED_444_ALLOC_FAILURE.with(|f| f.take()) {
      return Err(MixedSinkerError::Resample(
        crate::resample::ResampleError::AllocationFailed(crate::resample::PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )),
      ));
    }
    scratch
      .try_reserve_exact(len - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        ))
      })?;
    scratch.resize(len, 0);
  }
  Ok(())
}

/// Native fast-tier decimator for the **8-bit packed 4:4:4** non-alpha YUV
/// format ([`Vuyx`](crate::source::Vuyx) — bytes `V U Y X` per pixel, `X`
/// padding): de-PACKS the fully-interleaved source row into the sink's separate
/// Y / U / V scratch planes, then reuses the planar twin's non-4:2:0 join
/// verbatim ([`planar_8bit::yuv_planar_process_native`]) at
/// [`Yuv444p`](crate::source::Yuv444p) geometry. The 4:4:4 (full-width chroma,
/// `chroma_vsub = 1`) analog of the packed 4:2:2
/// [`packed_yuv_8bit`]'s `packed_yuv422_process_native`, so every output is
/// byte-identical to a [`Yuv444p`](crate::source::Yuv444p) native conversion of
/// those de-packed planes, and within ±1 LSB of the packed row-stage tier. Luma
/// is bit-identical (both bin the same native Y).
///
/// Vuyx packs four bytes per pixel `V U Y X`: V at byte 0, U at byte 1, Y at
/// byte 2, X (padding) at byte 3 — ignored. There is NO chroma subsampling
/// (4:4:4), so the de-pack writes FULL-width Y / U / V (`chroma_w = w`) and the
/// chroma plan is a plain `area(w, h, ..)`. Y is always de-packed (the join bins
/// Y for both luma and colour); U / V only on a colour row — on luma-only /
/// no-colour rows the join never reads chroma, the scratch is left as-is and the
/// join gets empty U / V slices (the lazy-chroma contract).
///
/// 8-bit source, so no native-depth clamp is needed (the source's native range
/// is the full `u8` range and the join's averaging keeps every sample in range).
///
/// Atomicity mirrors the packed 4:2:2 wrapper: the join's COMPLETE pre-feed
/// rejection preflight runs FIRST (via
/// [`planar_8bit::native_planar_preflight_check_only`]), before the fallible
/// Y / U / V scratch grow, so a rejected row returns its deterministic typed
/// error (`OutOfSequenceRow` / `ResampleOutputsChanged`), never
/// `AllocationFailed`, and grows no sink state. It is compare-only (no output-set
/// freeze), so the scratch grow stays a pre-feed step ahead of the delegate's
/// commit; the de-pack writes only the private scratch, so no caller output is
/// touched until the delegate clears.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn packed_vuyx_process_native(
  plan: &crate::resample::ResamplePlan,
  native_planar: &mut Option<std::boxed::Box<planar_8bit::NativePlanarYuv>>,
  y_scratch: &mut Vec<u8>,
  u_scratch: &mut Vec<u8>,
  v_scratch: &mut Vec<u8>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  packed: &[u8],
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || rgba.is_some() || hsv.is_some();

  // Run the join's COMPLETE pre-feed rejection preflight FIRST — before the
  // fallible Y / U / V de-pack scratch grow — so EVERY rejection case returns
  // its deterministic typed error, never AllocationFailed under allocation
  // pressure, and leaves the scratch untouched. Compare-only (no output-set
  // freeze), so the scratch grow stays a pre-feed step ahead of the delegate's
  // commit. `Ok(false)` is the no-output no-op. The delegate re-runs this
  // identical compare harmlessly and owns the commit.
  if !planar_8bit::native_planar_preflight_check_only(
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // De-pack the interleaved `V U Y X` row into the private Y / U / V scratch.
  // 4:4:4: chroma is full-width (`w`). Y is always de-packed; U / V only on a
  // colour row. The failpoint fires on the FIRST (Y) grow only.
  grow_packed_444_u8_scratch(y_scratch, w, true, w, h, plan)?;
  for (i, group) in packed.chunks_exact(4).enumerate() {
    y_scratch[i] = group[2];
  }
  if need_color {
    grow_packed_444_u8_scratch(u_scratch, w, false, w, h, plan)?;
    grow_packed_444_u8_scratch(v_scratch, w, false, w, h, plan)?;
    for (i, group) in packed.chunks_exact(4).enumerate() {
      u_scratch[i] = group[1];
      v_scratch[i] = group[0];
    }
  }

  let (u_plane, v_plane): (&[u8], &[u8]) = if need_color {
    (&u_scratch[..w], &v_scratch[..w])
  } else {
    (&[], &[])
  };
  planar_8bit::yuv_planar_process_native(
    plan,
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    || crate::resample::ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
    use_simd,
  )
}

/// Native fast-tier decimator for the **8-bit packed 4:4:4** non-alpha YUV
/// format [`Vyu444`](crate::source::Vyu444) — bytes `V Y U` per pixel, **3
/// bytes per pixel (24bpp), no alpha**. Identical to
/// [`packed_vuyx_process_native`] except for the source layout: VYU444 packs
/// three bytes per pixel `V Y U` (V at byte 0, Y at byte 1, U at byte 2 —
/// there is no fourth byte), so the de-pack steps the interleaved row in
/// 3-byte groups (`chunks_exact(3)`) rather than 4. Every other step — the
/// pre-feed preflight ordering, the lazy-chroma contract, the reuse of the
/// [`Yuv444p`](crate::source::Yuv444p) planar join at full-width chroma — is
/// shared verbatim, so each output is byte-identical to a `Yuv444p` native
/// conversion of the de-packed planes and within ±1 LSB of the packed
/// row-stage tier. Luma is bit-identical (both bin the same native Y).
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn packed_vyu444_process_native(
  plan: &crate::resample::ResamplePlan,
  native_planar: &mut Option<std::boxed::Box<planar_8bit::NativePlanarYuv>>,
  y_scratch: &mut Vec<u8>,
  u_scratch: &mut Vec<u8>,
  v_scratch: &mut Vec<u8>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  packed: &[u8],
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || rgba.is_some() || hsv.is_some();

  // Run the join's COMPLETE pre-feed rejection preflight FIRST — before the
  // fallible Y / U / V de-pack scratch grow — so EVERY rejection case returns
  // its deterministic typed error, never AllocationFailed, and leaves the
  // scratch untouched. Compare-only (no output-set freeze), so the scratch grow
  // stays a pre-feed step ahead of the delegate's commit. The delegate re-runs
  // this identical compare harmlessly and owns the commit.
  if !planar_8bit::native_planar_preflight_check_only(
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // De-pack the interleaved `V Y U` row (3 bytes/pixel) into the private
  // Y / U / V scratch. 4:4:4: chroma is full-width (`w`). Y is always
  // de-packed; U / V only on a colour row. The failpoint fires on the FIRST
  // (Y) grow only.
  grow_packed_444_u8_scratch(y_scratch, w, true, w, h, plan)?;
  for (i, group) in packed.chunks_exact(3).enumerate() {
    y_scratch[i] = group[1]; // Y at byte 1
  }
  if need_color {
    grow_packed_444_u8_scratch(u_scratch, w, false, w, h, plan)?;
    grow_packed_444_u8_scratch(v_scratch, w, false, w, h, plan)?;
    for (i, group) in packed.chunks_exact(3).enumerate() {
      u_scratch[i] = group[2]; // U at byte 2
      v_scratch[i] = group[0]; // V at byte 0
    }
  }

  let (u_plane, v_plane): (&[u8], &[u8]) = if need_color {
    (&u_scratch[..w], &v_scratch[..w])
  } else {
    (&[], &[])
  };
  planar_8bit::yuv_planar_process_native(
    plan,
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    || crate::resample::ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
    use_simd,
  )
}

/// Native fast-tier decimator for the **high-bit packed 4:4:4** non-alpha YUV
/// formats ([`V410`](crate::source::V410) 10-bit / [`Xv36`](crate::source::Xv36)
/// 12-bit): de-PACKS each format's OWN wire layout into the sink's separate
/// host-native LOGICAL u16 Y / U / V scratch planes, then reuses the HIGH-BIT
/// non-4:2:0 PLANAR join verbatim
/// ([`planar_high_bit_native::yuv_planar16_process_native`]) at
/// [`Yuv444p10`](crate::source::Yuv444p10) /
/// [`Yuv444p12`](crate::source::Yuv444p12) geometry. The 4:4:4 (full-width
/// chroma, `chroma_vsub = 1`) analog of the high-bit packed 4:2:2
/// [`y2xx_process_native`].
///
/// The de-pack is format-specific (V410 bit-extracts a 10-bit U / Y / V from one
/// 32-bit LE/BE word; Xv36 reads four MSB-aligned 12-bit u16 slots `U Y V A` and
/// drops the 4 low padding bits), so it is supplied by the caller as two
/// closures: `fill_y` de-packs the Y plane (always run, the join bins Y for both
/// luma and colour), `fill_uv` de-packs the U / V planes (run only on a colour
/// row). Each writes host-native LOGICAL u16 (a bit-field / MSB extraction is
/// inherently `<= (1 << BITS) - 1`, so it is already in the native range — the
/// join's clamp is a value no-op on encodable input). 4:4:4 has no chroma
/// subsampling, so the de-pack width is `w` and the chroma plan is a plain
/// `area(w, h, ..)`.
///
/// The reused join delegates with `BE = HOST_NATIVE_BE` (= `from_ne`) so its
/// internal decode is a no-op load on the already-native scratch — forwarding the
/// source wire `BE` would byte-swap the native value on a big-endian host. The
/// join now emits BOTH u8 `luma` and the native-depth `luma_u16` (the clamped
/// binned Y), so attaching `luma_u16` keeps the native route (no row-stage
/// fallback) — V410 / Xv36 thread their real `luma_u16` buffer. Under
/// `need_color == false` only Y is de-packed and the join is handed empty U / V
/// slices, so a luma-only / `luma_u16`-only sink plans no chroma state.
///
/// Atomicity mirrors [`y2xx_process_native`]: the join's COMPLETE compare-only
/// pre-feed preflight runs FIRST (via
/// [`planar_high_bit_native::native_planar_hb_preflight_check_only`]), before any
/// fallible scratch grow, so a rejected row returns its deterministic typed
/// error, never `AllocationFailed`, and touches no caller output. It commits no
/// output-set freeze — the de-pack grows here AND the delegate's own colour /
/// source scratch grows all stay pre-commit, and the delegate performs the SINGLE
/// freeze once every pre-feed allocation has succeeded, so a LATER grow failing
/// leaves `resample_outputs` untouched and the row retryable. The de-pack into
/// scratch is infallible and runs only after the preflight clears.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn packed_yuv444_hb_process_native<const BITS: u32>(
  plan: &crate::resample::ResamplePlan,
  native_planar_u16: &mut Option<std::boxed::Box<NativePlanarYuvU16>>,
  y_scratch: &mut Vec<u16>,
  u_scratch: &mut Vec<u16>,
  v_scratch: &mut Vec<u16>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  fill_y: impl FnOnce(&mut [u16]),
  fill_uv: impl FnOnce(&mut [u16], &mut [u16]),
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      BITS > 8 && BITS <= 16,
      "BITS must be in (8, 16] for high-bit packed 4:4:4 YUV"
    )
  };
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();

  // Run the planar join's COMPLETE compare-only pre-feed rejection preflight
  // FIRST — before any fallible scratch grow below, so every rejection returns
  // its deterministic typed error and leaves the wrapper scratch untouched.
  // Compare-only (no output-set freeze): the de-pack grows below AND the
  // delegate's own colour / source scratch grows all stay pre-commit, and the
  // delegate performs the SINGLE freeze once every pre-feed allocation has
  // succeeded — so a LATER grow (here or in the delegate) failing leaves
  // `resample_outputs` untouched and the row retryable.
  if !native_planar_hb_preflight_check_only(
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // Grow the wrapper de-pack scratch under the planner's recoverable contract —
  // Y always, U / V only on a colour row (4:4:4: full-width chroma). All grows
  // precede the infallible de-pack and the delegate call. The failpoint fires on
  // the FIRST (Y) grow only.
  grow_packed_444_u16_scratch(y_scratch, w, true, w, h, plan)?;
  if need_color {
    grow_packed_444_u16_scratch(u_scratch, w, false, w, h, plan)?;
    grow_packed_444_u16_scratch(v_scratch, w, false, w, h, plan)?;
  }

  // De-pack the wire layout into host-native LOGICAL u16 scratch via the
  // format-specific closures. Y is always de-packed; U / V only on a colour row
  // (`u_scratch` / `v_scratch` are distinct Vecs, so the two `&mut` borrows do
  // not alias). Everything past here is infallible.
  fill_y(&mut y_scratch[..w]);
  if need_color {
    fill_uv(&mut u_scratch[..w], &mut v_scratch[..w]);
  }

  // Delegate to the planar high-bit non-4:2:0 join with `BE = HOST_NATIVE_BE` so
  // its internal decode is a no-op on the already-native scratch, at the 4:4:4
  // chroma geometry (`chroma_vsub = 1`, `chroma_w = w`). Empty U / V on luma-only
  // rows. V410 / Xv36 thread their real `luma_u16` buffer (the join emits the
  // native-depth `luma_u16`), so attaching it stays on the native route.
  let (u_plane, v_plane): (&[u16], &[u16]) = if need_color {
    (&u_scratch[..w], &v_scratch[..w])
  } else {
    (&[], &[])
  };
  planar_high_bit_native::yuv_planar16_process_native::<BITS, HIGH_BIT_HOST_NATIVE_BE>(
    plan,
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    rgb_scratch_u16,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    w,
    || crate::resample::ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
    use_simd,
  )
}

/// Row-stage fused downscale for the **high-bit packed 4:2:2 YUV**
/// family (`Y210` / `Y212` / `Y216`, plus the exotic 10-bit `V210` word
/// packing) — the 4:2:2 analogue of the high-bit 4:4:4 route, with
/// **three** independent native-precision binnings.
///
/// High-bit packed YUV needs three binnings because the u8 and u16
/// YUV→RGB kernels (`range_params_n::<BITS, 8>` vs `::<BITS, BITS>`)
/// round and scale *independently*, and luma is native Y. Narrowing the
/// u16 bin to u8 would change a uniform-gray downscale's colour — a real
/// parity bug — so each output group bins its own native-precision
/// conversion:
/// 1. **u8 colour (rgb / rgba / hsv):** `convert_rgb_u8` fills a
///    source-width u8 RGB row (the format's `*_to_rgb_row` kernel —
///    chroma de-interleave + 4:2:2 upsample in-register), binned through
///    the u8 packed-RGB tail and fanned out to rgb / rgba / hsv.
/// 2. **u16 colour (rgb_u16 / rgba_u16):** `convert_rgb_u16` fills a
///    source-width native u16 RGB row (`*_to_rgb_u16_row`, source wire
///    `BE`), binned at native depth through the u16 tail with
///    `NATIVE_LUMA16 = false` and **only** rgb_u16 / rgba_u16 attached
///    (every narrowed u8 output passed as `&mut None`, so the tail's
///    narrow scratch is never sized).
/// 3. **luma (luma / luma_u16):** `deinterleave_y_u16` fills a
///    source-width host-native u16 Y row (`*_to_luma_u16_row`), binned
///    through the 1-channel u16 luma stream; luma_u16 is the host-native
///    binned Y, luma is `binned_Y >> (SRC_BITS - 8)`.
///
/// Colour outputs are byte-identical to the area-bin of the direct
/// full-resolution conversion (convert-then-bin — the fused form of
/// converting at full resolution then area-downscaling the RGB); luma is
/// the area-mean of the native Y. Atomic preflight: a single
/// [`frozen_outputs_check`], then a sequence check before any allocation
/// (so an out-of-sequence row is rejected without allocating and
/// `AllocationFailed` never masks `OutOfSequenceRow`), then the three
/// distinct, non-aliasing scratches grow and the three source rows stage
/// — all before the first feed, so a failure mutates no caller output. A
/// no-output call is a true no-op regardless of the row index.
///
/// NOTE (#263 follow-up): when ONLY `with_hsv()` is attached (no RGB /
/// RGBA), this row-stage resample path still stages the source-width u8
/// RGB row (group 1) and derives HSV per OUTPUT row off the binned RGB
/// stream. The high-bit planar direct + native fast tiers go RGB-free
/// for HSV-only, but the row-stage resample does not yet — the
/// `AreaStream` is keyed on the 3-channel RGB row, so an RGB-free
/// resample+HSV-only (resample the HSV planes, or resample Y/U/V then
/// convert per output row) is larger than this PR's scope and is
/// deferred. The HSV output is still correct; only the source-width RGB
/// scratch allocation remains.
#[cfg(any(
  feature = "y2xx",
  feature = "v210",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv422_triple_resample<const SRC_BITS: u32>(
  luma_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  rgb_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  rgb_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_scratch_u16: &mut Vec<u16>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  deinterleave_y_u16: impl FnOnce(&mut [u16]),
  convert_rgb_u8: impl FnOnce(&mut [u8]),
  convert_rgb_u16: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (high-bit packed YUV 4:2:2 is not routed to the filter
  // path): reject a filter plan before any work.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  const {
    assert!(
      SRC_BITS > 8 && SRC_BITS <= 16,
      "SRC_BITS must be in (8, 16] for high-bit packed 4:2:2 YUV"
    )
  };
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color_u8 = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_color_u16 = rgb_u16.is_some() || rgba_u16.is_some();

  // Sequence-check before allocating: every active stream started at row
  // 0 and the frozen output set keeps the active group fixed for the
  // frame, so they advance in lockstep — any active stream gives the
  // expected row. A no-output call has no stream to sequence and stays a
  // no-op regardless of the row index.
  let expected = if need_luma {
    luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if need_color_u8 {
    rgb_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_color_u16 {
    rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else {
    return Ok(());
  };
  // On the first row of a frame nothing is frozen yet, so reject an
  // out-of-sequence row here — before the freeze — so a rejected first row
  // never stores a snapshot that would poison a retry. On a later row the
  // freeze runs first (below) so a mid-frame output-set change is reported
  // as ResampleOutputsChanged rather than masked by a freshly-attached
  // stream's row-0 sequence mismatch.
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  if need_luma && luma_stream_u16.is_none() {
    *luma_stream_u16 = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_color_u8 && rgb_stream.is_none() {
    *rgb_stream = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_color_u16 && rgb_stream_u16.is_none() {
    *rgb_stream_u16 = Some({
      let stream =
        crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  // Stage the three source-width rows into their own distinct,
  // non-aliasing scratches (all fallible growths precede the feeds,
  // keeping the call atomic).
  let luma_row = if need_luma {
    let scratch = source_luma_u16_scratch(luma_scratch_u16, w, plan)?;
    deinterleave_y_u16(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_u8_row = if need_color_u8 {
    let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb_u8(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_u16_row = if need_color_u16 {
    let scratch = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
    convert_rgb_u16(scratch);
    Some(scratch)
  } else {
    None
  };

  // Binning 3 — native Y through the 1-channel u16 luma stream. A finalized
  // binned native-Y sample can exceed the source's native max on a clipped-high
  // edge, so clamp to `(1 << SRC_BITS) - 1` before the `luma_u16` pass-through
  // and the `>> (SRC_BITS - 8)` u8 narrowing (the luma twin of the colour
  // tail's native_max clamp; without it the narrow would wrap a clipped-high Y
  // to a low value). Mirrors `packed_yuv444_triple_feed_emit`.
  if let Some(y_row) = luma_row {
    let native_max: u16 = ((1u32 << SRC_BITS) - 1) as u16;
    let stream = luma_stream_u16.as_mut().expect("created in the preflight");
    stream.feed_row(idx, y_row, use_simd, |oy, binned_y| {
      if let Some(buf) = luma_u16.as_deref_mut() {
        for (dst, &y) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(binned_y) {
          *dst = y.min(native_max);
        }
      }
      if let Some(buf) = luma.as_deref_mut() {
        for (dst, &y) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(binned_y) {
          *dst = (y.min(native_max) >> (SRC_BITS - 8)) as u8;
        }
      }
    })?;
  }

  // Binning 1 — u8 colour through the shared u8 packed-RGB tail (luma /
  // luma_u16 handled by binning 3, so they are `&mut None` here).
  if let Some(scratch) = color_u8_row {
    let stream = rgb_stream.as_mut().expect("created in the preflight");
    packed_rgb_resample_emit(
      stream, plan, rgb, rgba, &mut None, &mut None, hsv, scratch, matrix, full_range, idx,
      use_simd,
    )?;
  }

  // Binning 2 — u16 colour through the shared u16 packed-RGB tail at
  // native depth. Only rgb_u16 / rgba_u16 are emitted; every narrowed u8
  // output is `&mut None`, so the tail's narrow scratch is never sized.
  if let Some(scratch) = color_u16_row {
    let stream = rgb_stream_u16.as_mut().expect("created in the preflight");
    packed_rgb_u16_resample_emit::<SRC_BITS, false>(
      stream,
      plan,
      &mut None,
      &mut None,
      &mut None,
      rgb_u16,
      rgba_u16,
      &mut None,
      &mut None,
      scratch,
      rgb_scratch,
      matrix,
      full_range,
      idx,
      use_simd,
    )?;
  }

  Ok(())
}

/// Separable-filter fused resize for the **high-bit packed 4:2:2 YUV
/// (no-alpha)** family (`Y210` 10-bit / `Y212` 12-bit / `Y216` 16-bit) —
/// the [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_yuv422_triple_resample`]. It mirrors the area path exactly: the
/// YUV is first converted to RGB by the **same** `convert_rgb_u8` /
/// `convert_rgb_u16` closures (which upsample the 4:2:2 chroma), then the
/// RGB is resampled — only the resampler kind differs (signed-coefficient
/// [`FilterStream`] instead of [`AreaStream`]). The staged RGB feeds the
/// **same** emit as the 4:4:4 path
/// ([`packed_yuv444_triple_feed_emit`], generic over
/// [`RowResampler`](crate::resample::RowResampler)) — the convert-then-RGB
/// resample tail is identical once the YUV is in RGB space, so the
/// resampled colour output equals the equivalent packed-RGB filter resample
/// of the converted pixels, and (because the area path converts-then-bins
/// the same RGB) matches the area output up to the kernel weights.
///
/// Luma is the native Y filter-resampled (the filter twin of the area
/// path's native-Y bin): `deinterleave_y` stages the native Y into a
/// source-width `u16` scratch and a 1-channel
/// [`FilterStream<u16>`](crate::resample::FilterStream)
/// (`luma_filter_stream_u16`) resamples it at native depth — luma is taken
/// from Y, never colour-derived, so it stays byte-exact to the direct
/// `*_to_luma*` kernels' filter resample. `luma_u16` is the host-native
/// resampled Y; `luma` is `resampled_y >> (SRC_BITS - 8)`.
///
/// Both the colour `u16` outputs and the native-Y luma carry the source's
/// native depth (`SRC_BITS`, `10` for `Y210`, `12` for `Y212`,
/// low-bit-packed `[0, (1 << SRC_BITS) - 1]`): a signed kernel (CatmullRom /
/// Lanczos3) can overshoot a legal edge. For colour,
/// [`packed_rgb_u16_resample_emit`] clamps every resampled sample to the
/// native max before any `u16` copy / RGBA expansion / u8 narrowing. The
/// de-interleaved native-Y luma bypasses that colour helper, so
/// [`packed_yuv444_triple_feed_emit`] applies the matching clamp to the
/// binned Y before the `luma_u16` copy and the u8 narrowing — keeping
/// `luma_u16 <= (1 << SRC_BITS) - 1` and a clipped-high Y edge mapping to
/// `255` rather than wrapping. For `Y216` (`SRC_BITS == 16`) the native max
/// is the u16 max, so both clamps are value no-ops; the area path never
/// overshoots, so they are no-ops there too. Both clamps match the in-range
/// area path's output exactly.
///
/// Atomic preflight (mirrors [`packed_yuv422_triple_resample`]): a single
/// sequence check on whichever stream is fed every row **before any
/// allocation** (an out-of-sequence first row is rejected before the freeze,
/// storing no snapshot to poison a retry; on a later row the freeze runs
/// first so a mid-frame output change trips `ResampleOutputsChanged`), then
/// every stream and source-width scratch is created before the first feed —
/// so a failure mutates no caller output. A no-output call has no stream to
/// sequence and stays a no-op.
///
/// NOTE (#263 follow-up): HSV-only (no RGB / RGBA) still stages the
/// source-width u8 RGB row here and derives HSV per OUTPUT row off the
/// filtered RGB stream — the direct + native fast tiers go RGB-free, but
/// the row-stage filter resample's stream is keyed on the 3-channel RGB
/// row, so an RGB-free resample+HSV-only is deferred. The HSV output is
/// still correct; only the source-width RGB scratch remains.
#[cfg(any(feature = "y2xx", feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv422_triple_filter_resample<const SRC_BITS: u32>(
  luma_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  rgb_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  rgb_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_scratch_u16: &mut Vec<u16>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  deinterleave_y_u16: impl FnOnce(&mut [u16]),
  convert_rgb_u8: impl FnOnce(&mut [u8]),
  convert_rgb_u16: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS > 8 && SRC_BITS <= 16,
      "SRC_BITS must be in (8, 16] for high-bit packed 4:2:2 YUV"
    )
  };
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color_u8 = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_color_u16 = rgb_u16.is_some() || rgba_u16.is_some();

  let (fh, fv) = (
    plan
      .filter_h()
      .expect("filter plan carries horizontal windows"),
    plan
      .filter_v()
      .expect("filter plan carries vertical windows"),
  );

  // Single sequence check before any allocation, on whichever attached
  // stream is fed every row (all attached streams advance in lockstep).
  let expected = if need_luma {
    luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if need_color_u8 {
    rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_color_u16 {
    rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else {
    return Ok(());
  };
  // First row: reject an out-of-sequence row before the freeze so a
  // rejected first row stores no snapshot that would poison a retry.
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  // Create the filter streams (post-sequence-check). Each plane runs the
  // full output grid against its own source grid; the colour streams carry
  // 3 interleaved channels, the luma stream 1.
  if need_luma && luma_filter_stream_u16.is_none() {
    *luma_filter_stream_u16 = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_color_u8 && rgb_filter_stream.is_none() {
    *rgb_filter_stream = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_color_u16 && rgb_filter_stream_u16.is_none() {
    *rgb_filter_stream_u16 = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }

  // Shared convert-then-resample tail (the 4:4:4 feed_emit, generic over
  // `RowResampler`): identical once the YUV is in RGB space, and it applies
  // the native-max clamp to both the u16 colour (via
  // `packed_rgb_u16_resample_emit`) and the native-Y luma for a sub-16-bit
  // source. The 4:2:2 arguments are reordered into the 4:4:4 feed_emit's
  // (rgb, rgb_u16, luma) stream / scratch / closure order.
  packed_yuv444_triple_feed_emit::<_, _, _, SRC_BITS>(
    rgb_filter_stream.as_mut(),
    rgb_filter_stream_u16.as_mut(),
    luma_filter_stream_u16.as_mut(),
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    rgb_scratch_u16,
    luma_scratch_u16,
    w,
    plan,
    idx,
    use_simd,
    matrix,
    full_range,
    convert_rgb_u8,
    convert_rgb_u16,
    deinterleave_y_u16,
  )
}

/// Source-width `f32` RGB staging for packed-float-RGB resampling: the
/// wire row converts here (host-native f32, lossless) before feeding
/// [`AreaStream<f32>`]. Grows `scratch` to `3 * width` `f32` under the
/// planner's recoverable-allocation contract. Mirrors
/// [`source_rgb_u16_scratch`] for the float element path.
#[cfg(any(
  all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
  feature = "gbr"
))]
pub(super) fn source_rgb_f32_scratch<'s>(
  scratch: &'s mut Vec<f32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [f32], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a `f32` source-staging OOM AFTER a successful stream
  // build leaves the stream field `None` AND `resample_outputs` uncommitted
  // (retryable). The float twin of the `source_rgb_u32_scratch` failpoint, so a
  // 3-ch float atomicity test can fail this specific pre-commit grow.
  #[cfg(all(
    test,
    feature = "std",
    any(
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
      feature = "gbr"
    )
  ))]
  if FORCE_SOURCE_RGB_F32_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0.0);
  }
  Ok(&mut scratch[..row])
}

// Single-shot source-RGB-`f32`-scratch grow failpoint for the RFC #238
// tail-collapse transactional-preflight tests. Gated on the packed-float-RGB /
// planar-GBR families that stage a source-width `f32` RGB row pre-commit;
// `source_rgb_f32_scratch` consults it under the same gate. Separate from the
// `try_box` box-alloc failpoint so a 3-channel float test can fail the SCRATCH
// grow specifically (after a successful stream construction).
#[cfg(all(
  test,
  feature = "std",
  any(
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
    feature = "gbr"
  )
))]
std::thread_local! {
  static FORCE_SOURCE_RGB_F32_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_rgb_f32_scratch`]
/// grow refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape of [`packed_rgb_f32_resample`] (a source-staging
/// OOM after a successful stream build leaves the stream field `None` AND
/// `resample_outputs` uncommitted, so the row is retryable). Test-only.
#[cfg(all(
  test,
  feature = "std",
  any(
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
    feature = "gbr"
  )
))]
pub(crate) fn arm_source_rgb_f32_scratch_failure() {
  FORCE_SOURCE_RGB_F32_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Out-width G/B/R `f32` plane staging for the [`Gbrpf32`](crate::source::Gbrpf32)
/// arm of the float packed-RGB tail: the dedicated `gbr` emit de-interleaves
/// each binned packed row back into three planes so it can run the exact
/// direct `gbrpf32_*` kernels for every output. Grows `scratch` to
/// `3 * width` `f32` — three contiguous planes — under the planner's
/// recoverable-allocation contract. Only the `gbr` emit consumes it; the
/// `rgb-float` ([`Rgbf32`](crate::source::Rgbf32)) caller never allocates it.
#[cfg(feature = "gbr")]
pub(super) fn rgb_plane_f32_scratch<'s>(
  scratch: &'s mut Vec<f32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [f32], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0.0);
  }
  Ok(&mut scratch[..row])
}

/// Out-width G/B/R `half::f16` plane staging for the
/// [`Gbrpf16`](crate::source::Gbrpf16) arm of the float packed-RGB
/// tail. There is no `AreaStream<f16>`, so binning runs in `f32`; this
/// emit de-interleaves each binned packed row into `f32` planes, rounds
/// them to `half::f16` here, and runs the exact direct `gbrpf16_*`
/// kernels for every output. Grows `scratch` to `3 * width`
/// `half::f16` — three contiguous planes — under the planner's
/// recoverable-allocation contract. Only the f16 emit consumes it.
#[cfg(feature = "gbr")]
pub(super) fn rgb_plane_f16_scratch<'s>(
  scratch: &'s mut Vec<half::f16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [half::f16], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, half::f16::ZERO);
  }
  Ok(&mut scratch[..row])
}

/// RFC #238 tail-collapse — the transactional packed-float-RGB **area** resample,
/// the float element-family twin of [`packed_rgb_u32_resample`]. Folds the split
/// preflight + area-stream build + per-format emit into one call behind the shared
/// compare-before / commit-after preflight
/// ([`resample_preflight_check_only`](resample_preflight_check_only)):
/// the missing area stream is built into a LOCAL and the source-width `f32` RGB
/// staging scratch is grown with NO field mutation, so the stream field insert and
/// the output-set freeze COMMIT TOGETHER only after they both succeed. A first-row
/// allocation failure therefore leaves `rgb_stream_f32` `None` AND `resample_outputs`
/// uncommitted (never a partial commit) — state-atomic and retryable even with a
/// changed output attachment (the #180 discipline). A pure reordering of the
/// pre-collapse split path: identical bytes, only deferring the field mutations past
/// the fallible allocations.
///
/// The four packed-float-RGB / planar-GBR 3-channel arms (`Rgbf32` / `Rgbf16` /
/// `Gbrpf32` / `Gbrpf16`) share this driver, the shared `f32` area stream, and this
/// preflight, differing only in the final `emit` (each format passes its own of the
/// four 3-channel emits — the f16 arms reuse the f32 stream, rounding to f16 in
/// their emit) and the `convert` that stages the source row — so the driver takes
/// both as parameters. `convert` stages the source-width host-native `f32` RGB row
/// into the grown scratch; `emit` feeds it and derives every output. The emit's own
/// out-width scratch grows after the freeze exactly as the pre-collapse split path
/// did (they are the emit's, not this driver's, so they are not pre-grown here).
#[cfg(any(
  all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
  feature = "gbr"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_f32_resample(
  rgb_stream_f32: &mut Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch_f32: &mut Vec<f32>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  convert: impl FnOnce(&mut [f32]),
  mut emit: impl FnMut(
    Option<&mut crate::resample::AreaStream<f32>>,
    &[f32],
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [half::f16]>,
    &mut Option<&mut [half::f16]>,
    &mut Option<HsvFrameMut<'_>>,
  ) -> Result<(), MixedSinkerError>,
) -> Result<(), MixedSinkerError> {
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the area stream is the sole
  // per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op.
  let expected = if need_any {
    Some(rgb_stream_f32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream alloc + scratch growth below succeed, so a first-row
  // allocation failure leaves `resample_outputs` uncommitted (retryable with a
  // changed output attachment). A no-output sink short-circuits to a no-op.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Area-only sink: reject a filter plan before any allocation. Runs AFTER the
  // compare-only preflight, so a no-output sink stays a no-op even under a filter
  // plan.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  // Build the missing area stream into a LOCAL and grow the source-width `f32` RGB
  // staging scratch with NO field mutation, so the stream field insert and the
  // freeze commit TOGETHER only after they both succeed (the `planar_dual_resample`
  // atomic shape: a scratch OOM leaves `rgb_stream_f32` `None` AND `resample_outputs`
  // uncommitted, never a partial commit).
  let new_stream = if rgb_stream_f32.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  source_rgb_f32_scratch(rgb_scratch_f32, w, plan)?;
  // Pre-grow the emit-owned out-width scratch BEFORE the freeze (a `None` stream
  // grows it and returns), so — like the merged u16 / u32 helpers — every fallible
  // pre-feed alloc precedes the commit and NOTHING fallible runs after the freeze.
  emit(
    None,
    &[],
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
  )?;
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream and
  // freeze the output set together (the compare above already caught any later
  // mid-frame output change; the freeze stores the snapshot on the first row).
  if let Some(s) = new_stream {
    *rgb_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  // Stage the source-width host-native `f32` RGB row (the caller's conversion into
  // the grown scratch, the grow re-run a no-op) and run the per-format emit.
  let stream = rgb_stream_f32.as_mut().expect("created above");
  let src_f32 = source_rgb_f32_scratch(rgb_scratch_f32, w, plan)?;
  convert(src_f32);
  emit(
    Some(stream),
    src_f32,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
  )
}

/// RFC #238 tail-collapse — the transactional packed-float-RGB **filter**
/// resample, the [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_rgb_f32_resample`] over a 3-channel `f32`
/// [`FilterStream`](crate::resample::FilterStream). Identical compare-before /
/// commit-after discipline (the filter stream is built into a local and its field
/// insert + the freeze + the pre-grown source scratch commit together only after
/// every allocation) and the identical per-format `emit` tail (the emits are
/// generic over the resampler kind), so the resampled colour equals the equivalent
/// area resample up to the signed kernel weights while remaining byte-identical to
/// the pre-collapse split filter path. Gated like [`packed_rgb_f32_resample`]: the
/// packed-float-RGB consumers (`rgb-float`, engine fenced in) AND `gbr` — the float
/// planar GBR sources scatter their G/B/R planes into the same packed `f32` RGB row
/// and filter through this stream.
#[cfg(any(
  all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
  feature = "gbr"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_f32_resample_filter(
  rgb_filter_stream_f32: &mut Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch_f32: &mut Vec<f32>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  convert: impl FnOnce(&mut [f32]),
  mut emit: impl FnMut(
    Option<&mut crate::resample::FilterStream<f32>>,
    &[f32],
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [half::f16]>,
    &mut Option<&mut [half::f16]>,
    &mut Option<HsvFrameMut<'_>>,
  ) -> Result<(), MixedSinkerError>,
) -> Result<(), MixedSinkerError> {
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgb_filter_stream_f32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_rgb_f32_resample`.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Single-kernel filter: reject a BICUBLIN plan before any allocation (its chroma
  // windows are read only by the `Yuv420p` per-plane route, so filtering every
  // channel with the luma kernel would be wrong output).
  plan.ensure_single_kernel_filter()?;
  // Build the missing filter stream into a LOCAL and grow the source scratch with
  // NO field mutation, so the stream field insert and the freeze commit TOGETHER
  // (the `planar_dual_resample` atomic shape: a scratch OOM leaves
  // `rgb_filter_stream_f32` `None` AND `resample_outputs` uncommitted).
  let new_stream = if rgb_filter_stream_f32.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  source_rgb_f32_scratch(rgb_scratch_f32, w, plan)?;
  // Pre-grow the emit-owned out-width scratch BEFORE the freeze (a `None` stream
  // grows it and returns), so every fallible pre-feed alloc precedes the commit and
  // NOTHING fallible runs after the freeze.
  emit(
    None,
    &[],
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
  )?;
  if let Some(s) = new_stream {
    *rgb_filter_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgb_filter_stream_f32.as_mut().expect("created above");
  let src_f32 = source_rgb_f32_scratch(rgb_scratch_f32, w, plan)?;
  convert(src_f32);
  emit(
    Some(stream),
    src_f32,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
  )
}

/// Feeds the prepared source-width `f32` RGB row into the (already
/// sequence-checked) stream and derives every attached output from each
/// finalized output row. Binning runs in float; the `rgb_f32` output
/// copies it losslessly, and every integer output mirrors the direct
/// [`Rgbf32`](crate::source::Rgbf32) path's clamp+scale kernels run over
/// the binned row. The binned row is already host-native f32 (the wire
/// converted via `rgbf32_to_rgb_f32_row::<BE>` before feeding), so the
/// emit kernels use `::<false>` — no further byte swap. `narrow_scratch`
/// is sized to the out-width u8 RGB row only when one of the outputs
/// that stage through it (`rgb` / `luma` / `luma_u16` / `hsv`) is
/// attached, so an f32-only or native-u16-only sink neither grows it nor
/// risks its allocation failure.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_f32_resample_emit(
  stream: Option<&mut impl crate::resample::RowResampler<f32>>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_f32: &[f32],
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  // The u8 RGB / luma / luma_u16 / hsv outputs stage through a u8 RGB
  // narrowing of the binned float row (exactly the direct path's
  // `rgbf32_to_rgb_row` scratch); an f32-only or native-u16-only sink
  // never touches it, so the out-width u8 scratch is sized — and its
  // allocation failure risked — only when one of those outputs is
  // attached. The predicate gates both the sizing here and the use in
  // the closure, so they cannot drift. `rgba` (u8) derives directly
  // from the float source via `rgbf32_to_rgba_row`, mirroring the
  // direct path, so it does not need the narrow row.
  let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  // The binned row is host-native f32 (the scatter decoded the source to
  // host order before binning), so the `rgbf32_*` kernels — which take a
  // wire-endian const and `load_f32` accordingly — must be told the data
  // is already host-order, else a big-endian target byte-swaps it and
  // corrupts every derived output.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  // Pregrow phase (RFC #238 tail-collapse): a `None` stream grows the emit-owned
  // out-width scratch above and returns, so the combined helper can pre-grow it
  // BEFORE the freeze; the real feed passes `Some`. Nothing fallible runs after the
  // freeze.
  let Some(stream) = stream else {
    return Ok(());
  };
  stream.feed_row(idx, src_f32, use_simd, |oy, binned| {
    // Lossless float pass-through — copy the binned row verbatim
    // (mirrors the direct path's `rgbf32_to_rgb_f32_row`; the binned
    // row is already host-native, so this is a plain copy).
    if let Some(buf) = rgb_f32.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(binned);
    }
    // u16 outputs — direct float→u16 clamp+scale (no narrowing stage),
    // exactly as the direct Rgbf32 path derives them from the source.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      crate::row::rgbf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
        binned,
        &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      crate::row::rgbf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        binned,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
      );
    }
    // u8 RGBA — direct float→u8 clamp+scale, alpha 0xFF (the direct
    // path emits RGBA straight from the float source, not via an
    // expand of the u8 RGB row).
    if let Some(buf) = rgba.as_deref_mut() {
      crate::row::rgbf32_to_rgba_row::<HOST_NATIVE_BE>(
        binned,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
      );
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      // Stage the u8 RGB row once via the direct path's float→u8
      // clamp+scale; rgb / luma / luma_u16 / hsv all read it, matching
      // the direct Rgbf32 source-of-truth ordering exactly.
      crate::row::rgbf32_to_rgb_row::<HOST_NATIVE_BE>(binned, nrow, ow, use_simd);
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Out-width **packed** `R, G, B` `half::f16` staging for the
/// [`Rgbf16`](crate::source::Rgbf16) arm of the float packed-RGB tail.
/// There is no `AreaStream<f16>`, so binning runs in `f32`; this emit
/// rounds each binned packed `f32` element to `half::f16` into this row
/// and runs the exact direct `rgbf16_*` kernels for every output. Unlike
/// the planar [`Gbrpf16`](crate::source::Gbrpf16) scratch this row stays
/// **packed** (no de-interleave). Grows `scratch` to `3 * width`
/// `half::f16` under the planner's recoverable-allocation contract. Only
/// the packed f16 emit consumes it.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
pub(super) fn rgb_packed_f16_scratch<'s>(
  scratch: &'s mut Vec<half::f16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [half::f16], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, half::f16::ZERO);
  }
  Ok(&mut scratch[..row])
}

/// Feeds the prepared source-width packed `R, G, B` `f32` row (the
/// [`Rgbf16`](crate::source::Rgbf16) wire widened f16 -> host-native f32)
/// into the f32 stream and derives every attached output from each
/// finalized output row. `stream` is generic over
/// [`RowResampler<f32>`](crate::resample::RowResampler), so the **same**
/// emit serves both the float area path ([`AreaStream<f32>`]) and the
/// signed-coefficient filter path ([`FilterStream<f32>`]) — the span kind
/// is chosen at the call site; the round-then-`rgbf16_*` tail is identical.
///
/// There is no `AreaStream<f16>`/`FilterStream<f16>`, so binning runs in
/// `f32` for precision.
/// Per finalized output row this tail **rounds the binned packed `f32` row
/// to `half::f16`** ([`rgb_packed_f16_scratch`]) and runs the **exact
/// direct `rgbf16_*` kernels** over that packed f16 row. The result is
/// therefore byte-identical to a direct full-resolution `Rgbf16`
/// conversion of the frame whose per-pixel f16 `R, G, B` is the `f32` area
/// mean rounded to f16 (the parity oracle) — because the emit performs the
/// identical round-then-`rgbf16_*`. The f16-native kernels
/// (`rgbf16_to_rgb_f16_row` / `..._u16_row` / `..._rgba_u16_row` /
/// `..._row` / `..._rgba_row`) consume the rounded packed f16 row directly;
/// the lossless `rgb_f32` output — which the direct path derives by
/// *widening* the f16 source to f32 — is reproduced by widening the
/// **rounded** packed f16 row back to f32 (`rgbf16_to_rgb_f32_row`), so it
/// too matches the f16-rounded oracle, not the raw f32 bin. The u8 RGB /
/// luma / luma_u16 / hsv outputs stage through a u8 RGB narrowing of the
/// rounded packed f16 row (`rgbf16_to_rgb_row`, exactly the direct path's
/// scratch); `rgba` (u8) derives directly from the rounded packed f16 row
/// via `rgbf16_to_rgba_row`, mirroring the direct path.
///
/// The rounded packed f16 row holds **host-native** `half::f16` (rounded
/// from host-native binned f32), so every `rgbf16_*` kernel — which takes a
/// wire-endian const and byte-swaps when it differs from the host — is
/// invoked with `HOST_NATIVE_BE` to make its load a no-op on every host.
/// `packed_scratch_f16` (the rounded packed row) is sized — and its
/// allocation failure risked — only when an output is attached;
/// `narrow_scratch` (the u8 RGB row) only when an output that stages
/// through it (`rgb` / `luma` / `luma_u16` / `hsv`) is attached. So an
/// `rgb_f32`-only sink sizes only the packed f16 row, and a no-output sink
/// sizes neither.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_f16_resample_emit(
  stream: Option<&mut impl crate::resample::RowResampler<f32>>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_f32: &[f32],
  packed_scratch_f16: &mut Vec<half::f16>,
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  // The rounded packed f16 row holds host-native data — the binned row was
  // decoded to host order during scatter, then rounded with `from_f32`,
  // which yields host-native `half::f16`. The `rgbf16_*` kernels take a
  // wire-endian const and byte-swap when it differs from the host, so pass
  // the host's own endianness to make every load a no-op; otherwise a
  // big-endian target would corrupt every output.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

  let ow = plan.out_w();
  // Every output derives from the rounded packed f16 row; even `rgb_f32`
  // does, because the direct `Rgbf16` path widens the f16 source to f32 (so
  // the oracle's `rgb_f32` is the f32 bin rounded to f16, then widened —
  // not the raw f32 bin). The predicate gates both the sizing here and the
  // round in the closure, so they cannot drift; a sink with no output sizes
  // nothing.
  let need_round = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgb_f16.is_some()
    || hsv.is_some();
  // The u8 RGB / luma / luma_u16 / hsv outputs stage through a u8 RGB
  // narrowing of the rounded packed f16 row (exactly the direct path's
  // `rgbf16_to_rgb_row` scratch); an f32-/f16-/native-u16-only sink never
  // touches it, so the out-width u8 scratch is sized — and its allocation
  // failure risked — only when one of those outputs is attached. `rgba`
  // (u8) derives directly from the rounded f16 row, so it does not need the
  // narrow row.
  let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  // Allocate both scratch rows up front (recoverable) before the stream's
  // closure writes any caller buffer, so an allocation refusal never leaves
  // a partially written output.
  let packed_f16: &mut [half::f16] = if need_round {
    rgb_packed_f16_scratch(packed_scratch_f16, ow, plan)?
  } else {
    &mut []
  };
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  // Pregrow phase (RFC #238 tail-collapse): a `None` stream grows the emit-owned
  // scratch above and returns, so the combined helper can pre-grow it BEFORE the
  // freeze; the real feed passes `Some`. Nothing fallible runs after the freeze.
  let Some(stream) = stream else {
    return Ok(());
  };
  stream.feed_row(idx, src_f32, use_simd, |oy, binned| {
    if need_round {
      // Round the binned packed `R, G, B` `f32` row to the packed f16 row
      // — the exact layout the direct `rgbf16_*` kernels consume, holding
      // the f32 block mean rounded to f16. (The f32-derived `rgb_f32`
      // output widens this rounded row back, never the raw bin.)
      let prow = &mut packed_f16[..3 * ow];
      for (dst, &src) in prow.iter_mut().zip(binned.iter()) {
        *dst = half::f16::from_f32(src);
      }
      let prow = &prow[..3 * ow];

      // ---- f16-native kernels: consume the rounded packed f16 row
      // directly, exactly as the direct `Rgbf16` path does ---------------
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::rgbf16_to_rgb_f16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      // Lossless f32 widen of the **rounded** f16 row — the direct path
      // widens its f16 source to f32, so the oracle's `rgb_f32` is the
      // bin rounded to f16 then widened (NOT the raw f32 bin).
      if let Some(buf) = rgb_f32.as_deref_mut() {
        crate::row::rgbf16_to_rgb_f32_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_u16.as_deref_mut() {
        crate::row::rgbf16_to_rgb_u16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_u16.as_deref_mut() {
        crate::row::rgbf16_to_rgba_u16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      // u8 RGBA — direct f16->u8 clamp+scale, alpha 0xFF (the direct path
      // emits RGBA straight from the f16 source, not via an expand of the
      // u8 RGB row).
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::rgbf16_to_rgba_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if need_narrow {
        let nrow = &mut narrow[..3 * ow];
        // Stage the u8 RGB row once via the direct path's f16->u8
        // clamp+scale; rgb / luma / luma_u16 / hsv all read it, matching
        // the direct Rgbf16 source-of-truth ordering exactly.
        crate::row::rgbf16_to_rgb_row::<HOST_NATIVE_BE>(prow, nrow, ow, use_simd);
        if let Some(buf) = rgb.as_deref_mut() {
          buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
        }
        if let Some(buf) = luma.as_deref_mut() {
          crate::row::rgb_to_luma_row(
            nrow,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            matrix,
            full_range,
            use_simd,
          );
        }
        if let Some(buf) = luma_u16.as_deref_mut() {
          crate::row::rgb_to_luma_u16_row(
            nrow,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            matrix,
            full_range,
            use_simd,
          );
        }
        if let Some(hsv) = hsv.as_mut() {
          let (h, s, v) = hsv.hsv();
          crate::row::rgb_to_hsv_row(
            nrow,
            &mut h[oy * ow..(oy + 1) * ow],
            &mut s[oy * ow..(oy + 1) * ow],
            &mut v[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
      }
    }
  })?;
  Ok(())
}

// ---- Packed float RGBA (Rgbaf32 / Rgbaf16) resample tails --------------
//
// The packed-float RGBA sources stage their wire row as a canonical
// source-width packed `R, G, B, A` f32 row and bin all four channels
// (straight alpha — a real per-channel area / filter mean, never forced
// opaque) on the 4-channel `AreaStream<f32>` / `FilterStream<f32>`. Per
// finalized output row every output is derived from the binned packed RGBA
// row by the **exact direct `rgbaf32_*` / `rgbaf16_*` kernels**, so the
// resampled frame is byte-identical to a direct `Rgbaf32` / `Rgbaf16`
// conversion of the binned frame (the parity oracle). These mirror the
// 3-channel `packed_rgb_f32_resample_emit` / `packed_rgb_f16_resample_emit`
// exactly, extended with the alpha channel and the `rgba_f32` / `rgba_f16`
// lossless outputs. Luma / HSV stage through a u8 RGB narrowing of the
// binned row (dropping alpha), matching the `Rgbf32` / `Rgbf16` sources of
// truth.

/// RFC #238 tail-collapse — the transactional packed-RGBA-**float** **area**
/// resample, the 4-channel packed twin of [`packed_rgb_f32_resample`] (`_packed`
/// because [`packed_rgba_f32_resample`] is the planar-GBR-alpha 4-channel path).
/// Folds the split preflight + area-stream build + per-format emit into one call
/// behind the shared compare-before / commit-after preflight: the missing area
/// stream is built into a LOCAL and the source-width `f32` RGBA staging scratch is
/// grown with NO field mutation, so the stream field insert and the output-set
/// freeze COMMIT TOGETHER only after they both succeed. A first-row allocation
/// failure therefore leaves `rgba_stream_f32` `None` AND `resample_outputs`
/// uncommitted (the #180 discipline) — byte-identical to the pre-collapse split
/// path. The [`Rgbaf32`](crate::source::Rgbaf32) and [`Rgbaf16`](crate::source::Rgbaf16)
/// arms share this driver, the shared `f32` area stream, and this preflight,
/// differing only in the final `emit` (the f16 arm reuses the f32 stream, rounding
/// to f16 in its emit) and the `convert` that stages the source row.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_f32_packed_resample(
  rgba_stream_f32: &mut Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch_f32: &mut Vec<f32>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  convert: impl FnOnce(&mut [f32]),
  mut emit: impl FnMut(
    Option<&mut crate::resample::AreaStream<f32>>,
    &[f32],
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [half::f16]>,
    &mut Option<&mut [half::f16]>,
    &mut Option<HsvFrameMut<'_>>,
  ) -> Result<(), MixedSinkerError>,
) -> Result<(), MixedSinkerError> {
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgba_stream_f32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_rgb_f32_resample`.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Area-only: reject a filter plan before any allocation (the straight-alpha
  // filter path takes `packed_rgba_f32_packed_filter`).
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  // Build the missing area stream into a LOCAL and grow the source-width `f32` RGBA
  // staging scratch with NO field mutation, so the stream field insert and the
  // freeze commit TOGETHER (a scratch OOM leaves `rgba_stream_f32` `None` AND
  // `resample_outputs` uncommitted, never a partial commit). The emit's own
  // out-width scratch grows after the freeze as the pre-collapse split path did.
  let new_stream = if rgba_stream_f32.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  source_rgba_f32_scratch(rgba_scratch_f32, w, plan)?;
  // Pre-grow the emit-owned out-width scratch BEFORE the freeze (a `None` stream
  // grows it and returns), so every fallible pre-feed alloc precedes the commit and
  // NOTHING fallible runs after the freeze.
  emit(
    None,
    &[],
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
  )?;
  if let Some(s) = new_stream {
    *rgba_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgba_stream_f32.as_mut().expect("created above");
  let src_rgba = source_rgba_f32_scratch(rgba_scratch_f32, w, plan)?;
  convert(src_rgba);
  emit(
    Some(stream),
    src_rgba,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
  )
}

/// RFC #238 tail-collapse — the transactional packed-RGBA-**float** **filter**
/// resample, the [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_rgba_f32_packed_resample`] over a 4-channel `f32`
/// [`FilterStream`](crate::resample::FilterStream). PIL filters R, G, B, A
/// independently with no premultiplication, so the four interleaved channels bin
/// through one 4-channel stream and resampled alpha is a real filtered channel;
/// the same compare-before / commit-after discipline as the area path keeps a
/// first-row failure from leaving `rgba_filter_stream_f32` `Some` or the freeze
/// committed.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_f32_packed_filter(
  rgba_filter_stream_f32: &mut Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch_f32: &mut Vec<f32>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  convert: impl FnOnce(&mut [f32]),
  mut emit: impl FnMut(
    Option<&mut crate::resample::FilterStream<f32>>,
    &[f32],
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u8]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [u16]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [f32]>,
    &mut Option<&mut [half::f16]>,
    &mut Option<&mut [half::f16]>,
    &mut Option<HsvFrameMut<'_>>,
  ) -> Result<(), MixedSinkerError>,
) -> Result<(), MixedSinkerError> {
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgba_filter_stream_f32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_rgb_f32_resample`.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  plan.ensure_single_kernel_filter()?;
  // Build the missing filter stream into a LOCAL and grow the source scratch with
  // NO field mutation, so the stream field insert and the freeze commit TOGETHER.
  let new_stream = if rgba_filter_stream_f32.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  source_rgba_f32_scratch(rgba_scratch_f32, w, plan)?;
  // Pre-grow the emit-owned out-width scratch BEFORE the freeze (a `None` stream
  // grows it and returns), so every fallible pre-feed alloc precedes the commit and
  // NOTHING fallible runs after the freeze.
  emit(
    None,
    &[],
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
  )?;
  if let Some(s) = new_stream {
    *rgba_filter_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgba_filter_stream_f32.as_mut().expect("created above");
  let src_rgba = source_rgba_f32_scratch(rgba_scratch_f32, w, plan)?;
  convert(src_rgba);
  emit(
    Some(stream),
    src_rgba,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
  )
}

/// Feeds the prepared source-width packed `R, G, B, A` `f32` row into the
/// (already sequence-checked) 4-channel stream and derives every attached
/// output from each finalized output row. The 4-channel twin of
/// [`packed_rgb_f32_resample_emit`]: `rgba_f32` copies the binned row
/// losslessly, `rgb_f32` drops alpha, and every integer / u8 output mirrors
/// the direct [`Rgbaf32`](crate::source::Rgbaf32) clamp+scale kernels over
/// the binned row. The binned row is host-native f32 (the wire converted via
/// `rgbaf32_to_rgba_f32_row::<BE>` before feeding), so the emit kernels use
/// `::<HOST_NATIVE_BE>`. `narrow_scratch` (the out-width u8 RGB row) is sized
/// only when an output that stages through it (`rgb` / `luma` / `luma_u16` /
/// `hsv`) is attached.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_f32_resample_emit(
  stream: Option<&mut impl crate::resample::RowResampler<f32>>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_rgba: &[f32],
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  // Pregrow phase (RFC #238 tail-collapse): a `None` stream grows the emit-owned
  // scratch above and returns, so the combined helper can pre-grow it BEFORE the
  // freeze; the real feed passes `Some`. Nothing fallible runs after the freeze.
  let Some(stream) = stream else {
    return Ok(());
  };
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // Lossless packed `f32` RGBA pass-through — copy the binned 4-channel
    // row verbatim (already host-native).
    if let Some(buf) = rgba_f32.as_deref_mut() {
      buf[oy * 4 * ow..(oy + 1) * 4 * ow].copy_from_slice(binned);
    }
    // Lossless RGB f32 (drop alpha).
    if let Some(buf) = rgb_f32.as_deref_mut() {
      crate::row::rgbaf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
        binned,
        &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
        ow,
        use_simd,
      );
    }
    // u16 outputs — direct float→u16 clamp+scale over the binned row.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      crate::row::rgbaf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
        binned,
        &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      crate::row::rgbaf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        binned,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
      );
    }
    // u8 RGBA — real binned alpha, clamp+scale.
    if let Some(buf) = rgba.as_deref_mut() {
      crate::row::rgbaf32_to_rgba_row::<HOST_NATIVE_BE>(
        binned,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
      );
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      // Stage the u8 RGB row once (drop alpha) via the direct path's
      // float→u8 clamp+scale; rgb / luma / luma_u16 / hsv all read it.
      crate::row::rgbaf32_to_rgb_row::<HOST_NATIVE_BE>(binned, nrow, ow, use_simd);
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Out-width **packed** `R, G, B, A` `half::f16` staging for the
/// [`Rgbaf16`](crate::source::Rgbaf16) resample tail — the 4-channel twin
/// of [`rgb_packed_f16_scratch`]. Grows `scratch` to `4 * width`
/// `half::f16` under the planner's recoverable-allocation contract.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
pub(super) fn rgba_packed_f16_scratch<'s>(
  scratch: &'s mut Vec<half::f16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [half::f16], MixedSinkerError> {
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
      4,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, half::f16::ZERO);
  }
  Ok(&mut scratch[..row])
}

/// Feeds the prepared source-width packed `R, G, B, A` `f32` row (the
/// [`Rgbaf16`](crate::source::Rgbaf16) wire widened f16 -> host-native f32)
/// into the 4-channel `f32` stream and derives every attached output by
/// **rounding** the binned packed `f32` row to `half::f16`
/// ([`rgba_packed_f16_scratch`]) and running the exact direct `rgbaf16_*`
/// kernels — so every output is byte-identical to a direct `Rgbaf16`
/// conversion of the `f32` block-mean rounded to f16 (the parity oracle).
/// The 4-channel twin of [`packed_rgb_f16_resample_emit`], extended with the
/// `rgba_f16` / `rgba_f32` outputs. The f32-derived `rgb_f32` / `rgba_f32`
/// outputs widen the **rounded** f16 row back (matching the direct path's
/// f16->f32 widen). The rounded packed f16 row is host-native, so every
/// `rgbaf16_*` kernel runs `::<HOST_NATIVE_BE>`.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_f16_resample_emit(
  stream: Option<&mut impl crate::resample::RowResampler<f32>>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_rgba: &[f32],
  packed_scratch_f16: &mut Vec<half::f16>,
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  let ow = plan.out_w();
  let need_round = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  let packed_f16: &mut [half::f16] = if need_round {
    rgba_packed_f16_scratch(packed_scratch_f16, ow, plan)?
  } else {
    &mut []
  };
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  // Pregrow phase (RFC #238 tail-collapse): a `None` stream grows the emit-owned
  // scratch above and returns, so the combined helper can pre-grow it BEFORE the
  // freeze; the real feed passes `Some`. Nothing fallible runs after the freeze.
  let Some(stream) = stream else {
    return Ok(());
  };
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    if need_round {
      // Round the binned packed `R, G, B, A` `f32` row to the packed f16
      // row — the exact layout the direct `rgbaf16_*` kernels consume.
      let prow = &mut packed_f16[..4 * ow];
      for (dst, &src) in prow.iter_mut().zip(binned.iter()) {
        *dst = half::f16::from_f32(src);
      }
      let prow = &prow[..4 * ow];

      // f16-native lossless outputs (consume the rounded packed f16 row).
      if let Some(buf) = rgba_f16.as_deref_mut() {
        crate::row::rgbaf16_to_rgba_f16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::rgbaf16_to_rgb_f16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      // f32 widen of the **rounded** f16 row (matches the direct path's
      // f16->f32 widen of its source, NOT the raw f32 bin).
      if let Some(buf) = rgba_f32.as_deref_mut() {
        crate::row::rgbaf16_to_rgba_f32_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_f32.as_deref_mut() {
        crate::row::rgbaf16_to_rgb_f32_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_u16.as_deref_mut() {
        crate::row::rgbaf16_to_rgb_u16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_u16.as_deref_mut() {
        crate::row::rgbaf16_to_rgba_u16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::rgbaf16_to_rgba_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if need_narrow {
        let nrow = &mut narrow[..3 * ow];
        crate::row::rgbaf16_to_rgb_row::<HOST_NATIVE_BE>(prow, nrow, ow, use_simd);
        if let Some(buf) = rgb.as_deref_mut() {
          buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
        }
        if let Some(buf) = luma.as_deref_mut() {
          crate::row::rgb_to_luma_row(
            nrow,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            matrix,
            full_range,
            use_simd,
          );
        }
        if let Some(buf) = luma_u16.as_deref_mut() {
          crate::row::rgb_to_luma_u16_row(
            nrow,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            matrix,
            full_range,
            use_simd,
          );
        }
        if let Some(hsv) = hsv.as_mut() {
          let (h, s, v) = hsv.hsv();
          crate::row::rgb_to_hsv_row(
            nrow,
            &mut h[oy * ow..(oy + 1) * ow],
            &mut s[oy * ow..(oy + 1) * ow],
            &mut v[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
      }
    }
  })?;
  Ok(())
}

/// Feeds the prepared source-width packed `R, G, B` `f32` row (the
/// [`Gbrpf32`](crate::source::Gbrpf32) planes scattered into RGB order)
/// into the prepared float stream — area or signed-coefficient filter,
/// both [`RowResampler<f32>`](crate::resample::RowResampler) — and derives
/// every attached output from each finalized output row. The `rgb-float`
/// ([`Rgbf32`]) tail's
/// per-row `rgbf32_*` clamp/scale kernels are not compiled in a `gbr`
/// build, so this tail de-interleaves each binned packed row back into
/// G/B/R planes once and runs the **exact direct `gbrpf32_*` kernels** —
/// every output, `luma_u16` included, is therefore byte-identical to a
/// direct full-resolution `Gbrpf32` conversion of the binned frame (the
/// parity oracle). The binned row is host-native f32, so the kernels run
/// `::<false>`. `plane_scratch` holds the out-width G/B/R planes
/// (`[0..ow]` = G, `[ow..2ow]` = B, `[2ow..3ow]` = R); it is sized (and
/// its allocation failure risked) only when an output that reads the
/// per-row `rgbf32_*` clamp/scale kernels are not compiled in a `gbr`
/// build, so this tail de-interleaves each binned packed row back into
/// G/B/R planes once and runs the **exact direct `gbrpf32_*` kernels** —
/// every output, `luma_u16` included, is therefore byte-identical to a
/// direct full-resolution `Gbrpf32` conversion of the binned frame (the
/// parity oracle). The binned row is host-native f32, so the kernels run
/// `::<false>`. `plane_scratch` holds the out-width G/B/R planes
/// (`[0..ow]` = G, `[ow..2ow]` = B, `[2ow..3ow]` = R); it is sized (and
/// its allocation failure risked) only when an output that reads the
/// planes is attached, so an `rgb_f32`-only sink neither grows it nor
/// risks its allocation. `rgb_f32` copies the binned row directly,
/// bypassing the planes. The lossless `rgba_f32` and the half-float
/// `rgb_f16` / `rgba_f16` outputs derive from the same de-interleaved
/// G/B/R planes via the direct `gbrpf32_to_rgba_f32_row` /
/// `gbrpf32_to_rgb_f16_row` / `gbrpf32_to_rgba_f16_row` kernels, so they
/// too are byte-identical to a direct `Gbrpf32` conversion of the binned
/// frame.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
pub(super) fn planar_gbr_f32_resample_emit(
  stream: Option<&mut impl crate::resample::RowResampler<f32>>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_f32: &[f32],
  plane_scratch: &mut Vec<f32>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  // Every output but `rgb_f32` derives from the de-interleaved G/B/R
  // planes; an `rgb_f32`-only sink copies the binned row and never sizes
  // the plane scratch. The predicate gates both the sizing here and the
  // de-interleave in the closure, so they cannot drift. `rgba_f32` and
  // the f16 outputs run their direct `gbrpf32_*` kernels over the same
  // planes (byte-identical to the direct path), so they join the gate.
  let need_planes = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let planes: &mut [f32] = if need_planes {
    rgb_plane_f32_scratch(plane_scratch, ow, plan)?
  } else {
    &mut []
  };
  // Pregrow phase (RFC #238 tail-collapse): a `None` stream grows the emit-owned
  // plane scratch above and returns, so the combined helper can pre-grow it BEFORE
  // the freeze; the real feed passes `Some`. Nothing fallible runs after the freeze.
  let Some(stream) = stream else {
    return Ok(());
  };
  stream.feed_row(idx, src_f32, use_simd, |oy, binned| {
    // Lossless float pass-through — copy the binned packed row verbatim
    // (the direct path's `gbrpf32_to_rgb_f32_row` over host-native data
    // is a plain interleave; the binned row is already that interleave).
    if let Some(buf) = rgb_f32.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(binned);
    }
    if need_planes {
      // De-interleave the binned packed `R, G, B` row into G/B/R planes
      // — the exact plane layout the direct `gbrpf32_*` kernels consume.
      let (g, rest) = planes.split_at_mut(ow);
      let (b, r) = rest.split_at_mut(ow);
      for x in 0..ow {
        r[x] = binned[x * 3];
        g[x] = binned[x * 3 + 1];
        b[x] = binned[x * 3 + 2];
      }
      let g = &g[..ow];
      let b = &b[..ow];
      let r = &r[..ow];
      // The de-interleaved planes hold host-native f32 (the binned row was
      // decoded to host order during scatter). The `gbrpf32_*` kernels take
      // a wire-endian const and byte-swap when it differs from the host, so
      // pass the host's own endianness to make the load a no-op — otherwise
      // a big-endian target would corrupt every plane-derived output.
      const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
      if let Some(buf) = rgb_u16.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_u16.as_deref_mut() {
        crate::row::gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      // Lossless packed `f32` RGBA — alpha forced to 1.0 (the direct
      // `gbrpf32_to_rgba_f32_row`, which the binned planes feed verbatim).
      if let Some(buf) = rgba_f32.as_deref_mut() {
        crate::row::gbrpf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      // f16 RGB / RGBA — fused f32->f16 narrow + interleave, exactly the
      // direct `gbrpf32_to_rgb_f16_row` / `gbrpf32_to_rgba_f16_row`.
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_f16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_f16.as_deref_mut() {
        crate::row::gbrpf32_to_rgba_f16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Feeds the prepared source-width packed `R, G, B` `f32` row (the
/// [`Gbrpf16`](crate::source::Gbrpf16) planes widened f16 -> host-native
/// f32 and scattered into RGB order) into the prepared float stream — area
/// or signed-coefficient filter, both
/// [`RowResampler<f32>`](crate::resample::RowResampler) — and derives every
/// attached output from each finalized output row.
///
/// There is no `AreaStream<f16>` / `FilterStream<f16>`, so binning runs in
/// `f32` for precision. Per finalized output row this tail de-interleaves the
/// binned packed row into `f32` G/B/R planes
/// ([`rgb_plane_f32_scratch`]), **rounds each element to `half::f16`**
/// (`half::f16::from_f32`) into the f16 planes ([`rgb_plane_f16_scratch`]:
/// `[0..ow]` = G, `[ow..2ow]` = B, `[2ow..3ow]` = R), then runs the
/// **exact direct `gbrpf16_*` kernels** over those f16 planes. The
/// result is therefore byte-identical to a direct full-resolution
/// `Gbrpf16` conversion of the frame whose per-pixel f16 G/B/R is the
/// `f32` area mean rounded to f16 (the parity oracle) — because the emit
/// performs the identical round-then-`gbrpf16_*`. The f16-native kernels
/// (`gbrpf16_to_rgb_f16_row` / `..._u16_row` / `..._row`) consume the
/// f16 planes directly; the outputs the direct path derives by widening
/// f16 -> f32 (`rgb_f32` / `rgba_f32` / `luma` / `luma_u16` / `hsv`) are
/// reproduced by widening the **rounded** f16 planes back to f32 (the
/// same lossless widen the direct path applies) and running the direct
/// `gbrpf32_*` kernels — so they too match the f16-rounded oracle, not
/// the raw f32 bin.
///
/// The rounded f16 planes hold **host-native** `half::f16` (rounded from
/// host-native binned f32), so every `gbrpf16_*` kernel — which takes a
/// wire-endian const and byte-swaps when it differs from the host — is
/// invoked with `HOST_NATIVE_BE` to make its load a no-op on every host;
/// the widen of those planes back to f32 likewise uses `HOST_NATIVE_BE`.
/// Only the f16 planes are staged: the f32-derived outputs must consume
/// the **rounded** values (widened back from f16), not the raw f32 bin,
/// so no f32 plane scratch is needed — the round writes directly from the
/// binned packed row. `plane_scratch_f16` is sized (and its allocation
/// failure risked) only when an output is attached, so a no-output sink
/// never grows it.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
pub(super) fn planar_gbr_f16_resample_emit(
  stream: Option<&mut impl crate::resample::RowResampler<f32>>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_f32: &[f32],
  plane_scratch_f16: &mut Vec<half::f16>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  use crate::row::scalar::planar_gbr_f16::widen_f16_be_to_host_f32;

  // The rounded f16 planes (and the f32 planes they round from) hold
  // host-native data — the binned row was decoded to host order during
  // scatter, then rounded with `from_f32`, which yields host-native
  // `half::f16`. The `gbrpf16_*` kernels take a wire-endian const and
  // byte-swap when it differs from the host, so pass the host's own
  // endianness to make every plane load a no-op; otherwise a big-endian
  // target would corrupt every output. The widen-back to f32 for the
  // f32-derived outputs uses the same const for the same reason.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  // Chunk size for the f16 -> f32 widen-back of the rounded planes, used
  // only by the outputs whose direct path widens f16 -> f32 (rgb_f32 /
  // rgba_f32 / luma / luma_u16 / hsv). Matches the dispatch layer's
  // widening chunk so the f32 staging is stack-resident.
  const WIDEN_CHUNK: usize = 64;

  let ow = plan.out_w();
  // Every output derives from the rounded f16 planes; even `rgb_f32`
  // does, because the direct `Gbrpf16` path widens the f16 source to f32
  // (so the oracle's `rgb_f32` is the f32 bin rounded to f16, then
  // widened — not the raw f32 bin). The predicate gates both the sizing
  // here and the de-interleave/round in the closure, so they cannot
  // drift; a sink with no output sizes nothing.
  let need_planes = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  // Allocate the f16 plane scratch up front (recoverable) before the
  // stream's closure writes any caller buffer, so an allocation refusal
  // never leaves a partially written output.
  let planes_f16: &mut [half::f16] = if need_planes {
    rgb_plane_f16_scratch(plane_scratch_f16, ow, plan)?
  } else {
    &mut []
  };
  // Pregrow phase (RFC #238 tail-collapse): a `None` stream grows the emit-owned f16
  // plane scratch above and returns, so the combined helper can pre-grow it BEFORE
  // the freeze; the real feed passes `Some`. Nothing fallible runs after the freeze.
  let Some(stream) = stream else {
    return Ok(());
  };
  stream.feed_row(idx, src_f32, use_simd, |oy, binned| {
    if need_planes {
      // De-interleave the binned packed `R, G, B` row directly into the
      // G/B/R f16 planes, **rounding** each element to `half::f16` — the
      // exact plane layout the direct `gbrpf16_*` kernels consume,
      // holding the f32 block mean rounded to f16. (No f32 plane stage:
      // the f32-derived outputs must consume the rounded values, so they
      // widen these f16 planes back, never the raw bin.)
      let (g16, rest_f16) = planes_f16.split_at_mut(ow);
      let (b16, r16) = rest_f16.split_at_mut(ow);
      for x in 0..ow {
        r16[x] = half::f16::from_f32(binned[x * 3]);
        g16[x] = half::f16::from_f32(binned[x * 3 + 1]);
        b16[x] = half::f16::from_f32(binned[x * 3 + 2]);
      }
      let g16 = &g16[..ow];
      let b16 = &b16[..ow];
      let r16 = &r16[..ow];

      // ---- f16-native kernels: consume the rounded f16 planes directly,
      // exactly as the direct `Gbrpf16` path does ------------------------
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::gbrpf16_to_rgb_f16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_f16.as_deref_mut() {
        crate::row::gbrpf16_to_rgba_f16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_u16.as_deref_mut() {
        crate::row::gbrpf16_to_rgb_u16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_u16.as_deref_mut() {
        crate::row::gbrpf16_to_rgba_u16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb.as_deref_mut() {
        crate::row::gbrpf16_to_rgb_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::gbrpf16_to_rgba_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }

      // ---- f32-derived outputs: the direct `Gbrpf16` path widens the
      // f16 source planes to f32 and runs the `gbrpf32_*` kernels, so
      // reproduce that exactly by widening the **rounded** f16 planes
      // back to f32 (chunked, stack-resident) and running the same
      // kernels — byte-identical to the f16-rounded oracle ---------------
      let need_wide_back = rgb_f32.is_some()
        || rgba_f32.is_some()
        || luma.is_some()
        || luma_u16.is_some()
        || hsv.is_some();
      if need_wide_back {
        let mut gw = [0.0f32; WIDEN_CHUNK];
        let mut bw = [0.0f32; WIDEN_CHUNK];
        let mut rw = [0.0f32; WIDEN_CHUNK];
        let mut off = 0;
        while off < ow {
          let n = (ow - off).min(WIDEN_CHUNK);
          // The rounded f16 planes are host-native; widen with the
          // host's own endianness so the bit-normalize is a no-op, then
          // run the `gbrpf32_*` kernels with `HOST_NATIVE_BE` (the same
          // post-widen routing the direct path uses).
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(g16, off, &mut gw, n);
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(b16, off, &mut bw, n);
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(r16, off, &mut rw, n);
          let gwn = &gw[..n];
          let bwn = &bw[..n];
          let rwn = &rw[..n];
          let cps = oy * ow + off;
          let cpe = cps + n;
          if let Some(buf) = rgb_f32.as_deref_mut() {
            crate::row::gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps * 3..cpe * 3],
              n,
              use_simd,
            );
          }
          if let Some(buf) = rgba_f32.as_deref_mut() {
            crate::row::gbrpf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps * 4..cpe * 4],
              n,
              use_simd,
            );
          }
          if let Some(buf) = luma.as_deref_mut() {
            crate::row::gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps..cpe],
              n,
              matrix,
              full_range,
              use_simd,
            );
          }
          if let Some(buf) = luma_u16.as_deref_mut() {
            crate::row::gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps..cpe],
              n,
              matrix,
              full_range,
              use_simd,
            );
          }
          if let Some(hsv) = hsv.as_mut() {
            let (h, s, v) = hsv.hsv();
            crate::row::gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut h[cps..cpe],
              &mut s[cps..cpe],
              &mut v[cps..cpe],
              n,
              use_simd,
            );
          }
          off += n;
        }
      }
    }
  })?;
  Ok(())
}

// ---- Float planar GBR+alpha (Gbrapf32 / Gbrapf16) resample tails -------
//
// The float planar GBR+alpha sources scatter their G/B/R/A planes into a
// canonical source-width packed `R, G, B, A` f32 row and bin all four
// channels in float on a dedicated 4-channel `AreaStream<f32>` — the float
// analogue of the integer `packed_rgba_resample` / `packed_rgba_u16_resample`
// alpha tails. Per finalized output row the binned packed row is resolved to
// its straight form (a copy in `Straight`, an un-premultiply in
// `Premultiplied`), de-interleaved into G/B/R/A planes, and the exact direct
// `gbrapf32_*` (RGBA, real source α) / `gbrpf32_*` (RGB / luma / hsv, α
// dropped) kernels run — so every output is byte-identical to a direct
// `Gbrapf32` conversion of the binned frame (the parity oracle). The
// `rgb-float` (`Rgbf32`) tail's packed `rgbf32_*` kernels are not compiled in
// a `gbr` build, hence the dedicated planar emit. These are GBR-only (there
// is no packed-float RGBA source), so they are gated to `gbr`.

/// Source-width canonical `R, G, B, A` `f32` staging for the float planar
/// GBR+alpha resample tail. Grows `scratch` to `4 * width` `f32` under the
/// planner's recoverable-allocation contract. Mirrors
/// [`source_rgb_f32_scratch`] for the 4-channel RGBA row.
#[cfg(any(
  feature = "gbr",
  all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
))]
pub(super) fn source_rgba_f32_scratch<'s>(
  scratch: &'s mut Vec<f32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [f32], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as
  // a host OOM would, proving a 4-channel `f32` scratch OOM AFTER a successful
  // stream build leaves the stream field `None` AND `resample_outputs` uncommitted
  // (retryable). The 4-channel twin of the `source_rgb_f32_scratch` failpoint, so a
  // packed-RGBA-float atomicity test can fail this specific pre-commit grow.
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "gbr",
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
    )
  ))]
  if FORCE_SOURCE_RGBA_F32_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      4,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0.0);
  }
  Ok(&mut scratch[..row])
}

// Single-shot source-RGBA-`f32`-scratch grow failpoint for the RFC #238
// tail-collapse transactional-preflight tests. Gated on the packed-RGBA-float /
// planar-GBR-alpha families that stage a source-width `f32` RGBA row pre-commit;
// `source_rgba_f32_scratch` consults it under the same gate. Separate from the
// `try_box` box-alloc failpoint so a 4-channel float test can fail the SCRATCH
// grow specifically (after a successful stream construction, before the
// stream-field insert).
#[cfg(all(
  test,
  feature = "std",
  any(
    feature = "gbr",
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
  )
))]
std::thread_local! {
  static FORCE_SOURCE_RGBA_F32_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_rgba_f32_scratch`]
/// grow refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape of the packed-RGBA-float area / filter tails (a
/// source-staging OOM after a successful stream build leaves the stream field
/// `None` AND `resample_outputs` uncommitted, so the row is retryable). Test-only.
#[cfg(all(
  test,
  feature = "std",
  any(
    feature = "gbr",
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
  )
))]
pub(crate) fn arm_source_rgba_f32_scratch_failure() {
  FORCE_SOURCE_RGBA_F32_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Out-width canonical `R, G, B, A` `f32` staging for the straight-color
/// resolve of the float planar GBR+alpha tail (a copy in `Straight`, an
/// un-premultiply in `Premultiplied`). Grows `scratch` to `4 * out_width`
/// `f32`. Mirrors [`source_rgba_f32_scratch`] sized to the output width.
#[cfg(feature = "gbr")]
pub(super) fn out_rgba_f32_scratch<'s>(
  scratch: &'s mut Vec<f32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [f32], MixedSinkerError> {
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
      4,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0.0);
  }
  Ok(&mut scratch[..row])
}

/// Out-width G/B/R/A `f32` plane staging for the [`Gbrapf32`](crate::source::Gbrapf32)
/// arm of the float planar GBR+alpha tail: the resolved straight packed
/// row de-interleaves into four contiguous planes (`[0..ow]` = G,
/// `[ow..2ow]` = B, `[2ow..3ow]` = R, `[3ow..4ow]` = A) so the exact direct
/// `gbrapf32_*` / `gbrpf32_*` kernels can run. Grows `scratch` to
/// `4 * width` `f32` under the planner's recoverable-allocation contract.
#[cfg(feature = "gbr")]
pub(super) fn rgba_plane_f32_scratch<'s>(
  scratch: &'s mut Vec<f32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [f32], MixedSinkerError> {
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
      4,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0.0);
  }
  Ok(&mut scratch[..row])
}

/// Out-width G/B/R/A `half::f16` plane staging for the
/// [`Gbrapf16`](crate::source::Gbrapf16) arm of the float planar GBR+alpha
/// tail. There is no `AreaStream<f16>`, so binning runs in `f32`; this emit
/// de-interleaves the resolved straight packed `f32` row into `f32`,
/// **rounds** each element to `half::f16` into these planes (same layout as
/// [`rgba_plane_f32_scratch`]), and runs the exact direct `gbrapf16_*` /
/// `gbrpf16_*` kernels. Grows `scratch` to `4 * width` `half::f16`.
#[cfg(feature = "gbr")]
pub(super) fn rgba_plane_f16_scratch<'s>(
  scratch: &'s mut Vec<half::f16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [half::f16], MixedSinkerError> {
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
      4,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, half::f16::ZERO);
  }
  Ok(&mut scratch[..row])
}

/// Premultiplies one canonical `R, G, B, A` `f32` row in place: each color
/// channel becomes `c * α` (α the raw plane value, normalized 0..1 by the
/// source); α is left unchanged. The float analogue of
/// [`premultiply_rgba_row_in_place`] — the exact op the
/// [`AlphaMode::Premultiplied`] float oracle mirrors (`R' = R * A`), so the
/// binned-then-un-premultiplied output matches the convert-then-bin oracle.
#[cfg(feature = "gbr")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn premultiply_rgba_f32_row_in_place(row: &mut [f32], width: usize) {
  for px in row[..width * 4].chunks_exact_mut(4) {
    let a = px[3];
    for c in &mut px[..3] {
      *c *= a;
    }
  }
}

/// Un-premultiplied straight color channel for one premultiplied binned
/// `f32` value: `pm / α`, or `0.0` when `α == 0` (a fully-transparent binned
/// pixel exposes no color, so it cannot bleed). The float inverse of
/// [`unpremultiply_channel`].
#[cfg(feature = "gbr")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpremultiply_channel_f32(pm: f32, a: f32) -> f32 {
  if a == 0.0 { 0.0 } else { pm / a }
}

/// Resolves one binned canonical `R, G, B, A` `f32` row to its straight form
/// in `dst` (α copied through): a verbatim copy under [`AlphaMode::Straight`],
/// an un-premultiply (`R = pm / α`, `α == 0 -> 0`) under
/// [`AlphaMode::Premultiplied`]. The float twin of
/// [`unpremultiply_binned_rgba_into`], used as the single straight-color row
/// every output then reads.
#[cfg(feature = "gbr")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn resolve_straight_rgba_f32_into(binned: &[f32], dst: &mut [f32], width: usize, premult: bool) {
  if !premult {
    dst[..width * 4].copy_from_slice(&binned[..width * 4]);
    return;
  }
  for (out_px, in_px) in dst[..width * 4]
    .chunks_exact_mut(4)
    .zip(binned[..width * 4].chunks_exact(4))
  {
    let a = in_px[3];
    for c in 0..3 {
      out_px[c] = unpremultiply_channel_f32(in_px[c], a);
    }
    out_px[3] = a;
  }
}

/// Row-stage fused downscale for the float planar GBR+alpha family
/// ([`Gbrapf32`](crate::source::Gbrapf32)) — the alpha-aware 4-channel f32
/// analogue of the 3-channel [`planar_gbr_f32_resample_emit`]. `convert_rgba`
/// stages the G/B/R/A planes as a canonical source-width packed `R, G, B, A`
/// f32 row (lossless interleave, host-native); this tail bins all four
/// channels so resampled alpha is a real area mean, then per finalized output
/// row resolves the straight color and de-interleaves it into G/B/R/A planes,
/// running the exact direct `gbrapf32_*` (RGBA) / `gbrpf32_*` (RGB / luma /
/// hsv, α dropped) kernels — every output byte-identical to a direct
/// `Gbrapf32` conversion of the binned frame.
///
/// Under [`AlphaMode::Premultiplied`] the staged row is premultiplied in
/// place (`R' = R * A`) before binning and un-premultiplied per output row
/// (`R = mean(R*A) / mean(A)`, `α == 0 -> RGB = 0`), so color outputs are
/// alpha-weighted and transparent pixels never bleed.
///
/// Atomic conditional-ordering preflight identical to
/// [`packed_rgba_resample`]: a no-output call returns before any freeze; an
/// out-of-sequence first row is rejected before the freeze; a later-row
/// output change trips `ResampleOutputsChanged`; the stream and every scratch
/// are created after the sequence check and before the single feed. The
/// alpha-mode freeze itself is checked by the caller's
/// [`check_frozen_alpha_mode`] before route selection (mirroring the integer
/// alpha tails). The binned row is host-native f32 (the scatter decoded the
/// source to host order), so the emit kernels run `::<HOST_NATIVE_BE>` — no
/// further byte swap.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_f32_resample(
  rgba_stream_f32: &mut Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<f32>,
  color_scratch: &mut Vec<f32>,
  plane_scratch: &mut Vec<f32>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [f32]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (the planar-GBR-alpha float RGBA is not routed to the filter
  // path): reject a filter plan before any work. A premultiplied filter plan is
  // routed here by the caller (the filter engine cannot un-premultiply), so it
  // surfaces the typed `UnsupportedFilter` here; straight filter plans take the
  // dedicated `packed_rgba_f32_filter_resample`.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgba_stream_f32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream alloc + scratch growth below succeed (RFC #238
  // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
  // uncommitted — retryable with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  let premult = alpha_mode.is_premultiplied();
  // Every output but `rgba_f32` reads the de-interleaved G/B/R/A planes;
  // an `rgba_f32`-only sink copies the resolved straight row directly and
  // sizes no plane scratch. The RGB-only outputs (rgb / rgb_u16 / rgb_f16 /
  // luma / luma_u16 / hsv) drop α via the `gbrpf32_*` kernels over the G/B/R
  // planes; the RGBA outputs (rgba / rgba_u16 / rgba_f16) run `gbrapf32_*`.
  let need_planes = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  // Build the missing stream into a LOCAL and grow every pre-feed scratch (the
  // always-sized out-width straight-colour row, the conditional out-width G/B/R/A
  // plane row, and the source-width RGBA staging) with NO field mutation, so the
  // stream field insert and the freeze commit TOGETHER only after they all succeed
  // (the `planar_dual_resample` atomic shape: a scratch OOM leaves `rgba_stream_f32`
  // `None` AND `resample_outputs` uncommitted, never a partial commit). The emit's
  // grows re-run as no-ops after these pre-commit grows.
  let new_stream = if rgba_stream_f32.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  out_rgba_f32_scratch(color_scratch, ow, plan)?;
  if need_planes {
    rgba_plane_f32_scratch(plane_scratch, ow, plan)?;
  }
  source_rgba_f32_scratch(rgba_scratch, w, plan)?;
  if let Some(s) = new_stream {
    *rgba_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgba_stream_f32.as_mut().expect("created above");
  packed_rgba_f32_emit(
    stream,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
    rgba_scratch,
    color_scratch,
    plane_scratch,
    w,
    plan,
    idx,
    use_simd,
    premult,
    need_planes,
    matrix,
    full_range,
    convert_rgba,
  )
}

/// Shared per-output emit for the 4-channel float planar GBR+alpha tail —
/// the body both [`packed_rgba_f32_resample`] (area) and
/// [`packed_rgba_f32_filter_resample`] (filter) run after creating their
/// kind-appropriate [`RowResampler<f32>`](crate::resample::RowResampler).
/// Stages the canonical source-width `R, G, B, A` `f32` row (`convert_rgba`),
/// premultiplies it in place when `premult` (area-only; the filter path is
/// straight-only and passes `premult = false`), feeds it once, and per
/// finalized output row resolves the straight color and de-interleaves it
/// into G/B/R/A planes to run the exact direct `gbrapf32_*` (RGBA) /
/// `gbrpf32_*` (RGB / luma / hsv, α dropped) kernels — so every output is
/// byte-identical between the area and filter paths' shared derivation.
///
/// The binned row is host-native f32 (the scatter decoded the source to host
/// order), so the emit kernels run `::<HOST_NATIVE_BE>`. `need_planes`
/// (computed identically by both callers) gates the plane scratch sizing and
/// the de-interleave, so an `rgba_f32`-only sink copies the resolved straight
/// row directly and sizes no plane scratch.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
fn packed_rgba_f32_emit(
  stream: &mut impl crate::resample::RowResampler<f32>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<f32>,
  color_scratch: &mut Vec<f32>,
  plane_scratch: &mut Vec<f32>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  premult: bool,
  need_planes: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [f32]),
) -> Result<(), MixedSinkerError> {
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  let ow = plan.out_w();
  // Resolved straight RGBA color row (per output row); always sized when any
  // output is attached so every output reads one canonical straight row.
  let color = out_rgba_f32_scratch(color_scratch, ow, plan)?;
  let planes: &mut [f32] = if need_planes {
    rgba_plane_f32_scratch(plane_scratch, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_f32_scratch(rgba_scratch, w, plan)?;
  convert_rgba(src_rgba);
  if premult {
    premultiply_rgba_f32_row_in_place(src_rgba, w);
  }
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // Resolve the per-mode straight RGBA once (copy for straight,
    // un-premultiply for premult), then derive every output from it.
    let color = &mut color[..4 * ow];
    resolve_straight_rgba_f32_into(binned, color, ow, premult);
    // Lossless packed `f32` RGBA — copy the resolved straight row verbatim
    // (the direct `gbrapf32_to_rgba_f32_row` over host-native planes is a
    // plain interleave; the resolved row is already that interleave).
    if let Some(buf) = rgba_f32.as_deref_mut() {
      buf[oy * 4 * ow..(oy + 1) * 4 * ow].copy_from_slice(color);
    }
    if need_planes {
      // De-interleave the resolved straight `R, G, B, A` row into G/B/R/A
      // planes — the exact plane layout the direct kernels consume.
      let (g, rest) = planes.split_at_mut(ow);
      let (b, rest) = rest.split_at_mut(ow);
      let (r, a) = rest.split_at_mut(ow);
      for x in 0..ow {
        r[x] = color[x * 4];
        g[x] = color[x * 4 + 1];
        b[x] = color[x * 4 + 2];
        a[x] = color[x * 4 + 3];
      }
      let g = &g[..ow];
      let b = &b[..ow];
      let r = &r[..ow];
      let a = &a[..ow];
      // RGBA outputs carry the resolved straight α via the `gbrapf32_*`
      // kernels (real source α, clamp+scale for the integer/u8 forms,
      // lossless for f16).
      if let Some(buf) = rgba_u16.as_deref_mut() {
        crate::row::gbrapf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          a,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_f16.as_deref_mut() {
        crate::row::gbrapf32_to_rgba_f16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          a,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::gbrapf32_to_rgba_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          a,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      // RGB / luma / hsv outputs drop α via the `gbrpf32_*` kernels over the
      // G/B/R planes — identical to the 3-channel emit's source-of-truth.
      if let Some(buf) = rgb_u16.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_f16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_f32.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    } else if rgb_f32.as_deref().is_some() {
      // rgb_f32 with no other plane-derived output: drop α straight from the
      // resolved color (the direct `gbrpf32_to_rgb_f32_row` is a plain
      // interleave of host-native planes; here a strided copy of R/G/B).
      let buf = rgb_f32.as_deref_mut().unwrap();
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      for x in 0..ow {
        dst[x * 3] = color[x * 4];
        dst[x * 3 + 1] = color[x * 4 + 1];
        dst[x * 3 + 2] = color[x * 4 + 2];
      }
    }
  })?;
  Ok(())
}

/// Separable-filter fused resize for the float planar GBR+alpha family
/// ([`Gbrapf32`](crate::source::Gbrapf32)) — the
/// [`SpanKind::Filter`](crate::resample::SpanKind) twin of the area
/// [`packed_rgba_f32_resample`], and the 4-channel float analogue of the
/// 3-channel filter path. PIL filters R, G, B, A **independently with no
/// premultiplication**, so the staged canonical `R, G, B, A` `f32` row bins
/// through one 4-channel [`FilterStream<f32>`](crate::resample::FilterStream)
/// and resampled alpha is a real filtered channel (never forced opaque).
/// Each finalized output row then resolves the straight color and runs the
/// exact direct `gbrapf32_*` / `gbrpf32_*` kernels through the shared
/// [`packed_rgba_f32_emit`], so every output is byte-identical to the area
/// path's derivation of the same binned color.
///
/// There is **no native-depth clamp**: `f32` is full-range, so the lossless
/// `rgb_f32` / `rgba_f32` outputs preserve a signed-coefficient overshoot
/// (HDR > 1 and negatives) by design, exactly as the area path and `Rgbf32`
/// do, while the integer / f16 outputs clamp `[0, 1]` in their own narrows —
/// so an overshoot cannot wrap them. (Contrast the high-bit u16 filter tail,
/// which clamps to the native max before any narrow.)
///
/// There is **no premultiplied route** — the filter engine cannot
/// un-premultiply, so a premultiplied frame stays on the area
/// [`packed_rgba_f32_resample`] (which un-premultiplies); this tail is reached
/// only for straight alpha and passes `premult = false`.
///
/// Sequence-check precedes every allocation (the 4-channel stream creation
/// runs after the no-output and out-of-sequence rejections), keeping the call
/// atomic: a rejected first row stores no frozen-output snapshot and
/// `AllocationFailed` never masks `OutOfSequenceRow` — identical to
/// [`packed_rgba_u16_filter_resample`]. The alpha-mode freeze itself is
/// checked by the caller's [`check_frozen_alpha_mode`] before route selection.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_f32_filter_resample(
  rgba_filter_stream_f32: &mut Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<f32>,
  color_scratch: &mut Vec<f32>,
  plane_scratch: &mut Vec<f32>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [f32]),
) -> Result<(), MixedSinkerError> {
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgba_filter_stream_f32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER
  // the fallible stream alloc + scratch growth below succeed (RFC #238
  // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
  // uncommitted — retryable with a changed output attachment.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Every output but `rgba_f32` reads the de-interleaved G/B/R/A planes
  // (computed identically to the area path); an `rgba_f32`-only sink copies
  // the resolved straight row directly and sizes no plane scratch.
  let need_planes = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  // Build the missing filter stream into a LOCAL and grow every pre-feed scratch
  // (the always-sized out-width straight-colour row, the conditional out-width
  // plane row, and the source-width RGBA staging) with NO field mutation, so the
  // stream field insert and the freeze commit TOGETHER (a scratch OOM leaves
  // `rgba_filter_stream_f32` `None` AND `resample_outputs` uncommitted).
  let new_stream = if rgba_filter_stream_f32.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  out_rgba_f32_scratch(color_scratch, ow, plan)?;
  if need_planes {
    rgba_plane_f32_scratch(plane_scratch, ow, plan)?;
  }
  source_rgba_f32_scratch(rgba_scratch, w, plan)?;
  if let Some(s) = new_stream {
    *rgba_filter_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgba_filter_stream_f32.as_mut().expect("created above");
  // Straight alpha only (no premult route): bin → resolve (a verbatim copy) →
  // derive every output, sharing the area path's per-output emit so the two
  // paths' derivations cannot drift.
  packed_rgba_f32_emit(
    stream,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
    rgba_scratch,
    color_scratch,
    plane_scratch,
    w,
    plan,
    idx,
    use_simd,
    false,
    need_planes,
    matrix,
    full_range,
    convert_rgba,
  )
}

/// Row-stage fused downscale for the half-float planar GBR+alpha family
/// ([`Gbrapf16`](crate::source::Gbrapf16)) — the alpha-aware 4-channel
/// analogue of the 3-channel [`planar_gbr_f16_resample_emit`]. `convert_rgba`
/// stages the G/B/R/A planes (widened f16 -> host-native f32) as a canonical
/// source-width packed `R, G, B, A` f32 row; this tail bins all four channels
/// in `f32` (there is no `AreaStream<f16>`), then per finalized output row
/// resolves the straight color, de-interleaves it into G/B/R/A `half::f16`
/// planes **rounding** each element, and runs the exact direct `gbrapf16_*` /
/// `gbrpf16_*` kernels. The f32-derived outputs (rgb_f32 / rgba_f32 / luma /
/// luma_u16 / hsv) widen the **rounded** f16 planes back to f32, exactly as
/// the direct `Gbrapf16` path widens its f16 source — so every output is
/// byte-identical to a direct `Gbrapf16` conversion of the f32 block-mean
/// rounded to f16 (the parity oracle).
///
/// Premultiply / un-premultiply, the freeze ordering, the endian handling,
/// and the GBR-only gating match [`packed_rgba_f32_resample`]; the only
/// difference is the per-output round-to-f16 and widen-back.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_f16_resample(
  rgba_stream_f32: &mut Option<std::boxed::Box<crate::resample::AreaStream<f32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<f32>,
  color_scratch: &mut Vec<f32>,
  plane_scratch_f16: &mut Vec<half::f16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [f32]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink: reject a filter plan before any work. A premultiplied filter
  // plan is routed here by the caller (the filter engine cannot un-premultiply), so
  // it surfaces the typed `UnsupportedFilter` here; straight filter plans take the
  // dedicated `packed_rgba_f16_filter_resample`.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgba_stream_f32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_rgba_f32_resample`.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  let premult = alpha_mode.is_premultiplied();
  // Every output derives from the rounded f16 planes (even rgb_f32 / rgba_f32,
  // because the direct `Gbrapf16` path widens its f16 source to f32).
  let need_planes = need_any;
  // Build the missing stream into a LOCAL and grow every pre-feed scratch (the
  // always-sized out-width straight-colour row, the conditional out-width f16 plane
  // row, and the source-width RGBA staging) with NO field mutation, so the stream
  // field insert and the freeze commit TOGETHER (a scratch OOM leaves
  // `rgba_stream_f32` `None` AND `resample_outputs` uncommitted).
  let new_stream = if rgba_stream_f32.is_none() {
    let stream =
      crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  out_rgba_f32_scratch(color_scratch, ow, plan)?;
  if need_planes {
    rgba_plane_f16_scratch(plane_scratch_f16, ow, plan)?;
  }
  source_rgba_f32_scratch(rgba_scratch, w, plan)?;
  if let Some(s) = new_stream {
    *rgba_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgba_stream_f32.as_mut().expect("created above");
  packed_rgba_f16_emit(
    stream,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
    rgba_scratch,
    color_scratch,
    plane_scratch_f16,
    w,
    plan,
    idx,
    use_simd,
    premult,
    need_planes,
    matrix,
    full_range,
    convert_rgba,
  )
}

/// Shared per-output emit for the half-float planar GBR+alpha tail — the body
/// both [`packed_rgba_f16_resample`] (area) and
/// [`packed_rgba_f16_filter_resample`] (filter) run after creating their
/// kind-appropriate [`RowResampler<f32>`](crate::resample::RowResampler).
/// There is no `AreaStream<f16>` / `FilterStream<f16>`, so binning runs in
/// `f32`; per finalized output row this resolves the straight color,
/// de-interleaves it into G/B/R/A `half::f16` planes **rounding** each element,
/// and runs the exact direct `gbrapf16_*` / `gbrpf16_*` kernels (the f32-derived
/// outputs widen the **rounded** f16 planes back to f32, exactly as the direct
/// `Gbrapf16` path widens its f16 source) — so every output is byte-identical
/// between the area and filter paths and to a direct `Gbrapf16` conversion of
/// the f32 block-mean rounded to f16. The area path passes `premult` from the
/// frame's alpha mode; the filter path is straight-only and passes `false`.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
fn packed_rgba_f16_emit(
  stream: &mut impl crate::resample::RowResampler<f32>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<f32>,
  color_scratch: &mut Vec<f32>,
  plane_scratch_f16: &mut Vec<half::f16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  premult: bool,
  need_planes: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [f32]),
) -> Result<(), MixedSinkerError> {
  use crate::row::scalar::planar_gbr_f16::widen_f16_be_to_host_f32;

  // The rounded f16 planes (and the f32 they widen back to) hold host-native
  // data — the binned row was decoded to host order during scatter, then
  // rounded with `from_f32`, which yields host-native `half::f16`. The
  // `gbrpf16_*` / `gbrapf16_*` kernels (and the widen-back `gbrpf32_*` /
  // `gbrapf32_*`) take a wire-endian const and byte-swap when it differs
  // from the host, so pass the host's own endianness to make every load a
  // no-op; otherwise a big-endian target would corrupt every output.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  // Chunk size for the f16 -> f32 widen-back of the rounded planes, matching
  // the dispatch layer's widening chunk so the f32 staging is stack-resident.
  const WIDEN_CHUNK: usize = 64;

  let ow = plan.out_w();
  let color = out_rgba_f32_scratch(color_scratch, ow, plan)?;
  let planes_f16: &mut [half::f16] = if need_planes {
    rgba_plane_f16_scratch(plane_scratch_f16, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_f32_scratch(rgba_scratch, w, plan)?;
  convert_rgba(src_rgba);
  if premult {
    premultiply_rgba_f32_row_in_place(src_rgba, w);
  }
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    if need_planes {
      // Resolve the per-mode straight RGBA, then de-interleave it into the
      // G/B/R/A f16 planes, **rounding** each element to `half::f16` — the
      // exact plane layout the direct `gbrapf16_*` / `gbrpf16_*` kernels
      // consume, holding the f32 block mean rounded to f16.
      let color = &mut color[..4 * ow];
      resolve_straight_rgba_f32_into(binned, color, ow, premult);
      let (g16, rest) = planes_f16.split_at_mut(ow);
      let (b16, rest) = rest.split_at_mut(ow);
      let (r16, a16) = rest.split_at_mut(ow);
      for x in 0..ow {
        r16[x] = half::f16::from_f32(color[x * 4]);
        g16[x] = half::f16::from_f32(color[x * 4 + 1]);
        b16[x] = half::f16::from_f32(color[x * 4 + 2]);
        a16[x] = half::f16::from_f32(color[x * 4 + 3]);
      }
      let g16 = &g16[..ow];
      let b16 = &b16[..ow];
      let r16 = &r16[..ow];
      let a16 = &a16[..ow];

      // ---- f16-native kernels: the outputs the direct `Gbrapf16` path
      // derives straight from the f16 source (no widen) — `rgb_f16` /
      // `rgba_f16` (lossless / fused-narrow) and the u8 `rgb` / `rgba` RGB
      // (the α byte of `rgba` is overwritten from the widened α below,
      // mirroring the direct path's `gbrpf16_to_rgba_row` + α scatter) ---
      if let Some(buf) = rgba_f16.as_deref_mut() {
        crate::row::gbrapf16_to_rgba_f16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          a16,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::gbrpf16_to_rgb_f16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        // RGB from the f16 source (α = 0xFF stub); the real α byte is
        // written from the widened α plane below, exactly as the direct
        // `Gbrapf16` path does (`gbrpf16_to_rgba_row` + α scatter).
        crate::row::gbrpf16_to_rgba_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb.as_deref_mut() {
        crate::row::gbrpf16_to_rgb_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }

      // ---- widen-back f32 outputs: the direct `Gbrapf16` path widens the
      // f16 source to f32 and runs the `gbrapf32_*` / `gbrpf32_*` kernels,
      // so reproduce that by widening the **rounded** f16 planes back to f32
      // (chunked, stack-resident) and running the same kernels —
      // byte-identical to the f16-rounded oracle. The `rgb_u16` / `rgba_u16`
      // and the u8 `rgba` α byte come from this same widened source (no
      // `gbrapf16_to_rgba_u16` / `..._row` kernel exists). -----------------
      let need_wide_back = rgb_u16.is_some()
        || rgba_u16.is_some()
        || rgb_f32.is_some()
        || rgba_f32.is_some()
        || rgba.is_some()
        || luma.is_some()
        || luma_u16.is_some()
        || hsv.is_some();
      if need_wide_back {
        let mut gw = [0.0f32; WIDEN_CHUNK];
        let mut bw = [0.0f32; WIDEN_CHUNK];
        let mut rw = [0.0f32; WIDEN_CHUNK];
        let mut aw = [0.0f32; WIDEN_CHUNK];
        let mut off = 0;
        while off < ow {
          let n = (ow - off).min(WIDEN_CHUNK);
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(g16, off, &mut gw, n);
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(b16, off, &mut bw, n);
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(r16, off, &mut rw, n);
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(a16, off, &mut aw, n);
          let gwn = &gw[..n];
          let bwn = &bw[..n];
          let rwn = &rw[..n];
          let awn = &aw[..n];
          let cps = oy * ow + off;
          let cpe = cps + n;
          if let Some(buf) = rgba_u16.as_deref_mut() {
            crate::row::gbrapf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              awn,
              &mut buf[cps * 4..cpe * 4],
              n,
              use_simd,
            );
          }
          if let Some(buf) = rgb_u16.as_deref_mut() {
            crate::row::gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps * 3..cpe * 3],
              n,
              use_simd,
            );
          }
          // Overwrite the u8 `rgba` α byte from the widened α plane — the
          // same clamp/scale `copy_alpha_plane_f32_to_u8` the direct path's
          // `widen_and_scatter_f16_alpha_to_u8` applies (host-native source).
          if let Some(buf) = rgba.as_deref_mut() {
            crate::row::scalar::alpha_extract::copy_alpha_plane_f32_to_u8::<HOST_NATIVE_BE>(
              awn,
              &mut buf[cps * 4..cpe * 4],
              n,
            );
          }
          if let Some(buf) = rgba_f32.as_deref_mut() {
            crate::row::gbrapf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              awn,
              &mut buf[cps * 4..cpe * 4],
              n,
              use_simd,
            );
          }
          if let Some(buf) = rgb_f32.as_deref_mut() {
            crate::row::gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps * 3..cpe * 3],
              n,
              use_simd,
            );
          }
          if let Some(buf) = luma.as_deref_mut() {
            crate::row::gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps..cpe],
              n,
              matrix,
              full_range,
              use_simd,
            );
          }
          if let Some(buf) = luma_u16.as_deref_mut() {
            crate::row::gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps..cpe],
              n,
              matrix,
              full_range,
              use_simd,
            );
          }
          if let Some(hsv) = hsv.as_mut() {
            let (h, s, v) = hsv.hsv();
            crate::row::gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut h[cps..cpe],
              &mut s[cps..cpe],
              &mut v[cps..cpe],
              n,
              use_simd,
            );
          }
          off += n;
        }
      }
    }
  })?;
  Ok(())
}

/// Separable-filter fused resize for the half-float planar GBR+alpha family
/// ([`Gbrapf16`](crate::source::Gbrapf16)) — the
/// [`SpanKind::Filter`](crate::resample::SpanKind) twin of the area
/// [`packed_rgba_f16_resample`]. `convert_rgba` widens the G/B/R/A f16 planes
/// to host-native f32 and stages a canonical source-width packed `R, G, B, A`
/// f32 row; PIL filters the four channels independently (no premultiplication),
/// so they bin through one 4-channel
/// [`FilterStream<f32>`](crate::resample::FilterStream) (there is no
/// `FilterStream<f16>`) and resampled alpha is a real filtered channel. Each
/// finalized output row resolves the straight color, rounds the de-interleaved
/// G/B/R/A planes to `half::f16`, and runs the exact direct `gbrapf16_*` /
/// `gbrpf16_*` kernels through the shared [`packed_rgba_f16_emit`] — so every
/// output is byte-identical to the area path's derivation of the same binned
/// color rounded to f16.
///
/// No native-depth clamp (the f16 narrows clamp `[0, 1]`; the `rgb_f32` /
/// `rgba_f32` outputs are the lossless widen of the rounded f16, full-range by
/// design) and no premultiplied route (a premultiplied frame stays on the area
/// [`packed_rgba_f16_resample`]; this tail is reached only for straight alpha,
/// passing `premult = false`). The atomic conditional-ordering preflight is
/// identical to [`packed_rgba_f32_filter_resample`].
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_f16_filter_resample(
  rgba_filter_stream_f32: &mut Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<f32>,
  color_scratch: &mut Vec<f32>,
  plane_scratch_f16: &mut Vec<half::f16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [f32]),
) -> Result<(), MixedSinkerError> {
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let expected = if need_any {
    Some(rgba_filter_stream_f32.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_rgba_f32_resample`.
  if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }
  // Every output derives from the rounded f16 planes (even rgb_f32 / rgba_f32,
  // because the direct `Gbrapf16` path widens its f16 source to f32).
  let need_planes = need_any;
  // Build the missing filter stream into a LOCAL and grow every pre-feed scratch
  // (the always-sized out-width straight-colour row, the conditional out-width f16
  // plane row, and the source-width RGBA staging) with NO field mutation, so the
  // stream field insert and the freeze commit TOGETHER (a scratch OOM leaves
  // `rgba_filter_stream_f32` `None` AND `resample_outputs` uncommitted).
  let new_stream = if rgba_filter_stream_f32.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 4)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  out_rgba_f32_scratch(color_scratch, ow, plan)?;
  if need_planes {
    rgba_plane_f16_scratch(plane_scratch_f16, ow, plan)?;
  }
  source_rgba_f32_scratch(rgba_scratch, w, plan)?;
  if let Some(s) = new_stream {
    *rgba_filter_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  let stream = rgba_filter_stream_f32.as_mut().expect("created above");
  // Straight alpha only (no premult route): the shared f16 emit bins in f32,
  // resolves a verbatim straight copy, rounds to f16, and derives every output.
  packed_rgba_f16_emit(
    stream,
    rgb,
    rgba,
    luma,
    rgb_u16,
    rgba_u16,
    luma_u16,
    rgb_f32,
    rgba_f32,
    rgb_f16,
    rgba_f16,
    hsv,
    rgba_scratch,
    color_scratch,
    plane_scratch_f16,
    w,
    plan,
    idx,
    use_simd,
    false,
    need_planes,
    matrix,
    full_range,
    convert_rgba,
  )
}

// ---- Xyz12 (linear-light area mean) resample tail ----------------------
//
// The `Xyz12` source decodes a 2.6-gamma-encoded CIE-XYZ wire sample
// through SMPTE ST 428-1 §8 inverse-OETF -> linear XYZ -> 3x3 gamut
// matrix -> sRGB OETF -> narrow. Area-resampling must average in LINEAR
// light, so the wire row is converted to **linear XYZ** (`xyz12_to_
// xyz_f32_row`, post-OETF / pre-matrix), binned in float, and the
// non-linear tail (matrix + gamma + clamp/scale) is applied per
// finalized output row. Because the bin is a linear combination and the
// matrix is linear, `M . mean(xyz) == mean(M . xyz)` exactly — the
// matrix commutes with the bin — so every derived output equals the
// direct DCP pipeline applied to a frame whose per-pixel linear XYZ is
// the area mean of the source linear XYZ (the linear-light oracle). The
// OETF / narrow are per-pixel and run AFTER the matrix, exactly as the
// direct path does, so they need not commute with the bin.

/// Source-width **linear-XYZ** `f32` staging for the `Xyz12` resample
/// path: the wire row converts here (inverse-OETF only, no matrix)
/// before feeding [`AreaStream<f32>`]. Grows `scratch` to `3 * width`
/// `f32` under the planner's recoverable-allocation contract. Mirrors
/// [`source_rgb_f32_scratch`] for the linear-XYZ element path.
#[cfg(feature = "xyz")]
pub(super) fn source_xyz_f32_scratch<'s>(
  scratch: &'s mut Vec<f32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [f32], MixedSinkerError> {
  // Single-shot grow failpoint (RFC #238 tail-collapse tests): refuse this grow as a
  // host OOM would, proving a scratch OOM AFTER a successful stream build leaves the
  // `Xyz12` stream field `None` AND `resample_outputs` uncommitted (retryable).
  // Separate from the `try_box` box-alloc failpoint so a test can fail the SCRATCH
  // grow specifically.
  #[cfg(all(test, feature = "xyz", feature = "std"))]
  if FORCE_SOURCE_XYZ_F32_SCRATCH_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
    )));
  }
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0.0);
  }
  Ok(&mut scratch[..row])
}

// Single-shot source-linear-XYZ-`f32`-scratch grow failpoint for the RFC #238
// tail-collapse transactional-preflight tests. Gated like the `Xyz12` resample test
// module; `source_xyz_f32_scratch` consults it under the same gate. Separate from
// the `try_box` box-alloc failpoint so a test can fail the SCRATCH grow specifically
// (after a successful stream construction).
#[cfg(all(test, feature = "xyz", feature = "std"))]
std::thread_local! {
  static FORCE_SOURCE_XYZ_F32_SCRATCH_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`source_xyz_f32_scratch`] grow
/// refuse with `AllocationFailed`, exactly as a host OOM would. Proves the
/// commit-together atomic shape (a scratch OOM after a successful `Xyz12` stream
/// build leaves the stream field `None` AND `resample_outputs` uncommitted, so the
/// row is retryable). Test-only.
#[cfg(all(test, feature = "xyz", feature = "std"))]
pub(crate) fn arm_source_xyz_f32_scratch_failure() {
  FORCE_SOURCE_XYZ_F32_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Feeds the prepared source-width **linear-XYZ** `f32` row into the
/// (already sequence-checked) stream and derives every attached output
/// from each finalized binned linear-XYZ output row. The stream is the
/// kind-appropriate engine — an [`AreaStream`](crate::resample::AreaStream)
/// or a [`FilterStream`](crate::resample::FilterStream) — built by the
/// caller (the `Xyz12` sink's resample route) from the plan's
/// [`SpanKind`](crate::resample::SpanKind); both are
/// [`RowResampler`](crate::resample::RowResampler)s that bin the same staged
/// linear-XYZ `f32` row and run the identical per-output derivation.
///
/// `xyz_f32` copies the binned linear XYZ verbatim. Every other output
/// applies the direct DCP path's math to the binned XYZ: the gamut
/// matrix yields linear RGB (`rgb_f32`); the sRGB OETF + clamp/scale +
/// narrow yield the integer / f16 outputs (`rgb` / `rgba` / `rgb_u16` /
/// `rgba_u16` / `rgb_f16` / `rgba_f16`); and `luma` / `luma_u16` / `hsv`
/// derive from the staged u8 RGB row — exactly the direct path's
/// source-of-truth ordering. The u8 RGB staging row is sized only when
/// one of the outputs that reads it (`rgb` / `rgba` / `luma` /
/// `luma_u16` / `hsv`) is attached, so an f32-/f16-only sink neither
/// grows it nor risks its allocation failure.
///
/// `target_gamut` selects the XYZ -> RGB matrix; `luma_q15` carries the
/// gamut-matched Q15 luma weights (both ride [`Xyz12Row`] on the direct
/// path). These bind the entire frame via [`FrozenOutputs`] +
/// `xyz12_to`'s per-frame constants, so they cannot drift mid-frame.
#[cfg(feature = "xyz")]
#[allow(clippy::too_many_arguments)]
pub(super) fn xyz12_resample_emit(
  stream: &mut impl crate::resample::RowResampler<f32>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  xyz_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_xyz: &[f32],
  narrow_scratch: &mut Vec<u8>,
  target_gamut: crate::DcpTargetGamut,
  luma_q15: (i32, i32, i32),
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  use crate::row::scalar::{
    xyz12::{matmul3_xyz_rgb, narrow_unit_to_u8, narrow_unit_to_u16, oetf_srgb},
    xyz12_constants::xyz_to_rgb_matrix,
  };
  let ow = plan.out_w();
  let m = xyz_to_rgb_matrix(target_gamut);
  let one_f16 = half::f16::from_f32(1.0);
  // The u8 RGB / luma / luma_u16 / hsv outputs stage through a u8 RGB
  // narrowing of the binned linear XYZ (matrix + OETF + clamp/x255);
  // an f32-/f16-/native-u16-only sink never touches it, so the
  // out-width u8 scratch is sized — and its allocation failure risked —
  // only when one of those outputs is attached. The predicate gates
  // both the sizing here and the use in the closure, so they cannot
  // drift. `rgba` (u8) derives directly from the binned XYZ (matrix +
  // OETF + narrow + alpha, exactly the direct `xyz12_to_rgba_row`), so
  // it does not need the narrow row.
  let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_xyz, use_simd, |oy, binned| {
    // Lossless linear-XYZ pass-through — copy the binned row verbatim.
    if let Some(buf) = xyz_f32.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(binned);
    }
    // Linear RGB (matrix only, no OETF / clamp) — out-of-gamut negatives
    // and HDR > 1 preserved bit-exact, mirroring `xyz12_to_rgb_f32_row`
    // but over the already-inverse-OETF'd binned XYZ.
    if let Some(buf) = rgb_f32.as_deref_mut() {
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      for x in 0..ow {
        let i = x * 3;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[i] = rgb_lin[0];
        dst[i + 1] = rgb_lin[1];
        dst[i + 2] = rgb_lin[2];
      }
    }
    // f16 RGB / RGBA — matrix + OETF + clamp [0, 1] + IEEE-754 RNE
    // narrow, exactly as `xyz12_to_rgb_f16_row` / `xyz12_to_rgba_f16_row`.
    if let Some(buf) = rgb_f16.as_deref_mut() {
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      for x in 0..ow {
        let i = x * 3;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[i] = half::f16::from_f32(oetf_srgb(rgb_lin[0]).clamp(0.0, 1.0));
        dst[i + 1] = half::f16::from_f32(oetf_srgb(rgb_lin[1]).clamp(0.0, 1.0));
        dst[i + 2] = half::f16::from_f32(oetf_srgb(rgb_lin[2]).clamp(0.0, 1.0));
      }
    }
    if let Some(buf) = rgba_f16.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      for x in 0..ow {
        let i = x * 3;
        let o = x * 4;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[o] = half::f16::from_f32(oetf_srgb(rgb_lin[0]).clamp(0.0, 1.0));
        dst[o + 1] = half::f16::from_f32(oetf_srgb(rgb_lin[1]).clamp(0.0, 1.0));
        dst[o + 2] = half::f16::from_f32(oetf_srgb(rgb_lin[2]).clamp(0.0, 1.0));
        dst[o + 3] = one_f16;
      }
    }
    // u16 RGB / RGBA — matrix + OETF + clamp + x65535 + round-half-up,
    // exactly as `xyz12_to_rgb_u16_row` / `xyz12_to_rgba_u16_row`.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      for x in 0..ow {
        let i = x * 3;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[i] = narrow_unit_to_u16(oetf_srgb(rgb_lin[0]));
        dst[i + 1] = narrow_unit_to_u16(oetf_srgb(rgb_lin[1]));
        dst[i + 2] = narrow_unit_to_u16(oetf_srgb(rgb_lin[2]));
      }
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      for x in 0..ow {
        let i = x * 3;
        let o = x * 4;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[o] = narrow_unit_to_u16(oetf_srgb(rgb_lin[0]));
        dst[o + 1] = narrow_unit_to_u16(oetf_srgb(rgb_lin[1]));
        dst[o + 2] = narrow_unit_to_u16(oetf_srgb(rgb_lin[2]));
        dst[o + 3] = 0xFFFF;
      }
    }
    // u8 RGBA — matrix + OETF + clamp + x255 + round-half-up, alpha
    // 0xFF (exactly `xyz12_to_rgba_row`), derived directly from the
    // binned XYZ rather than expanded from the staged u8 RGB row.
    if let Some(buf) = rgba.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      for x in 0..ow {
        let i = x * 3;
        let o = x * 4;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[o] = narrow_unit_to_u8(oetf_srgb(rgb_lin[0]));
        dst[o + 1] = narrow_unit_to_u8(oetf_srgb(rgb_lin[1]));
        dst[o + 2] = narrow_unit_to_u8(oetf_srgb(rgb_lin[2]));
        dst[o + 3] = 0xFF;
      }
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      // Stage the u8 RGB row once via the direct path's matrix + OETF +
      // clamp + x255; rgb / rgba / luma / luma_u16 / hsv all read it.
      for x in 0..ow {
        let i = x * 3;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        nrow[i] = narrow_unit_to_u8(oetf_srgb(rgb_lin[0]));
        nrow[i + 1] = narrow_unit_to_u8(oetf_srgb(rgb_lin[1]));
        nrow[i + 2] = narrow_unit_to_u8(oetf_srgb(rgb_lin[2]));
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::xyz12_rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          luma_q15,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::xyz12_rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          luma_q15,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
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
/// Used by Bayer / Bayer16 / Pal8 [`MixedSinker`] paths whose source
/// has no native luma plane to memcpy from. YUV source impls take
/// their luma directly off the Y plane and don't go through this
/// helper, so they don't need a configurable coefficient set —
/// the source's `ColorMatrix` already fixed it at encode time.
#[cfg(any(feature = "bayer", feature = "mono"))]
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
  // existing `frame_elems` validation in caller paths makes the
  // product fit, a future caller passing a raw slice with no such
  // upstream check could trigger a `usize` overflow inside the
  // assert message itself (panic before the assertion runs).
  // Failing the assert on overflow yields a clean diagnostic.
  debug_assert!(
    luma
      .len()
      .checked_mul(3)
      .is_some_and(|need| rgb.len() >= need),
    "rgb_row_to_luma_row: rgb.len()={} but need {} (= 3 x luma.len()={})",
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

/// Same as [`rgb_row_to_luma_row`] but widens the luma byte to `u16` via
/// `(y << 8) | y` (`0 → 0x0000`, `255 → 0xFFFF`).
///
/// Used by format sinker paths that expose a `with_luma_u16` output channel
/// (e.g. `MixedSinker<Pal8>`).
#[cfg(feature = "mono")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgb_row_to_luma_u16_row(
  rgb: &[u8],
  luma_u16: &mut [u16],
  coeffs_q8: (u32, u32, u32),
) {
  debug_assert!(
    luma_u16
      .len()
      .checked_mul(3)
      .is_some_and(|need| rgb.len() >= need),
    "rgb_row_to_luma_u16_row: rgb.len()={} but need {} (= 3 x luma_u16.len()={})",
    rgb.len(),
    luma_u16.len().saturating_mul(3),
    luma_u16.len(),
  );
  let (cr, cg, cb) = coeffs_q8;
  for (i, dst) in luma_u16.iter_mut().enumerate() {
    let r = rgb[3 * i] as u32;
    let g = rgb[3 * i + 1] as u32;
    let b = rgb[3 * i + 2] as u32;
    let y = ((cr * r + cg * g + cb * b + 128) >> 8).min(255) as u16;
    *dst = (y << 8) | y;
  }
}

// ---- Format-specific impl blocks (split out of mod.rs) ------------------
//
// Each child module hosts the `MixedSinker<'_, F>` impl blocks for a
// related family of source formats. mod.rs keeps only the shared
// prelude (errors, types, struct, generic impls, helpers) and the
// `LumaCoefficients` API. Per-format `with_rgba` / `set_rgba` builders
// and `PixelSink` impls live in the child modules below.

#[cfg(feature = "yuv-444-packed")]
mod ayuv;
#[cfg(feature = "yuv-444-packed")]
mod ayuv64;
#[cfg(feature = "bayer")]
mod bayer;
#[cfg(feature = "yuv-planar")]
mod chroma_reconstruct;
#[cfg(feature = "gray")]
mod gray;
#[cfg(feature = "rgb-legacy")]
mod legacy_rgb;
#[cfg(feature = "mono")]
mod mono1bit;
#[cfg(feature = "rgb")]
mod packed_rgb_10bit;
#[cfg(feature = "rgb")]
mod packed_rgb_16bit;
#[cfg(feature = "rgb")]
mod packed_rgb_32bit;
#[cfg(feature = "rgb")]
mod packed_rgb_8bit;
#[cfg(feature = "rgb-float")]
mod packed_rgb_f16;
#[cfg(feature = "rgb-float")]
mod packed_rgb_float;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_4_1_1;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_8bit;
#[cfg(feature = "mono")]
mod pal8;
#[cfg(feature = "yuv-planar")]
mod planar_8bit;
// The chroma failpoint is armed only by the non-4:2:0 native suites that own a
// colour oracle, and each reuses the planar join so each also pulls in
// `yuv-planar`: the packed-8-bit / packed-4:4:4 / 4:1:1 natives, plus the
// semi-planar 8-bit suite's `yuv-planar`-gated `native_tier` (its colour oracle
// adds `rgb`). So its re-export is dead in a `yuv-planar`-without-any-consumer
// build (and a `yuv-planar`-less build has no `planar_8bit` module to import).
#[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
pub(crate) use planar_8bit::arm_native_rgb_scratch_failure;
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  any(
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    all(feature = "yuv-semi-planar", feature = "rgb")
  )
))]
pub(crate) use planar_8bit::arm_planar_native_chroma_failure;
// RFC #238 Phase 2 — the linear-light averaging path for the planar 8-bit
// YUV family. Produces RGB/RGBA, so it needs both `yuv-planar` (the source
// formats) and `rgb` (the output kernels).
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod linear_light;
#[cfg(feature = "yuv-planar")]
mod planar_high_bit_native;
// Bring the high-bit non-4:2:0 planar native join + entry point (and its
// test-only alloc failpoint) into the `mixed` scope so the three per-format
// high-bit sinks — `subsampled_4_2_2_high_bit::{yuv422p, yuv440p}` and
// `subsampled_4_4_4_high_bit::yuv444p`, which span two parent modules —
// reach them by the same `super::super::{..}` path they use for the shared
// row-stage tails (`packed_yuv4*_triple_resample`).
#[cfg(feature = "yuv-planar")]
use planar_high_bit_native::yuv_planar16_process_native;
// The high-bit semi-planar non-4:2:0 P-format native wrappers
// (`subsampled_4_2_2_high_bit::p2xx` / `subsampled_4_4_4_high_bit::p4xx`) reuse
// the planar join + its pre-feed preflight after de-interleaving + de-packing
// the packed UV plane — the `u16` twin of the 8-bit semi-planar non-4:2:0
// reuse. The high-bit PACKED 4:2:2 native wrapper (`y2xx_process_native`, the
// `Y210` / `Y212` / `Y216` tier) reuses the same join + preflight after
// de-packing + de-interleaving the YUYV-ordered MSB-aligned u16 words. Bring
// the join type + preflight into the `mixed` scope so those sinks reach them by
// `super::{..}` / `super::super::{..}`. The MSB-aligned planar 4:4:4 native
// wrapper (`subsampled_4_4_4_high_bit::yuv444p_msb`) is a `yuv-planar`-only
// consumer (it de-packs the high-bit samples then reuses the same join +
// preflight as the low-bit twin), so the re-export tracks `yuv-planar` alone —
// matching the [`yuv_planar16_process_native`] re-export above; the
// semi-planar / `y2xx` / packed-4:4:4 wrappers are additional consumers.
#[cfg(feature = "yuv-planar")]
use planar_high_bit_native::{NativePlanarYuvU16, native_planar_hb_preflight_check_only};
// The source-scratch + u16-colour-scratch alloc failpoints are armed only by the
// high-bit planar native suite, which is gated on `rgb` (its colour oracle), so
// these re-exports are dead in a `yuv-planar`-without-`rgb` build.
#[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
pub(crate) use planar_high_bit_native::{
  arm_planar_hb_native_alloc_failure, arm_planar_hb_native_alloc_failure_after,
  arm_planar_hb_native_rgb_u16_scratch_failure,
};
// The chroma failpoint is armed only by the high-bit non-4:2:0 native suites
// that own a colour oracle (V210 / Y2xx / P2xx / P4xx / packed-4:4:4, each
// pulling in `yuv-planar` for the join, plus the `rgb`-gated planar high-bit
// native suite), so its re-export is dead in a `yuv-planar`-without-any-consumer
// build.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  any(
    feature = "rgb",
    feature = "v210",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-semi-planar"
  )
))]
pub(crate) use planar_high_bit_native::arm_planar_hb_native_chroma_failure;
#[cfg(feature = "gbr")]
mod planar_gbr_32bit;
#[cfg(feature = "gbr")]
mod planar_gbr_8bit;
#[cfg(feature = "gbr")]
mod planar_gbr_f16;
#[cfg(feature = "gbr")]
mod planar_gbr_float;
#[cfg(feature = "gbr")]
mod planar_gbr_high_bit;
#[cfg(feature = "gbr")]
mod planar_gbr_msb;
/// Format-agnostic planar-YUV resample helpers (the 4:2:0 native /
/// row-stage join and the shared row-stage path), reused by the 8-bit
/// planar family, the semi-planar family, and the packed YUV family
/// (which de-interleaves Y into a scratch first via
/// [`planar_resample::packed_yuv_dual_resample`]). The shared element-type-free
/// compare-only preflight ([`resample_preflight_check_only`]) that every routed
/// family runs lives beside [`frozen_outputs_check`] in this module, so this
/// planar submodule is gated only to its own consumers.
#[cfg(any(
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuv-packed"
))]
mod planar_resample;
#[cfg(feature = "yuv-semi-planar")]
mod semi_planar_8bit;
// Each `subsampled_4_*_high_bit` parent hosts BOTH the high-bit planar
// `Yuv4*p` sinks (`yuv-planar`) and the high-bit semi-planar P-format
// sinks (`yuv-semi-planar`); the per-submodule gates inside each `mod`
// keep the two families separable, so the parent compiles whenever
// either family is on. Mirrors how `row::dispatch::yuv420` /
// `row::dispatch::pn` host the matching row layer.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod subsampled_4_2_0_high_bit;
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod subsampled_4_2_2_high_bit;
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod subsampled_4_4_4_high_bit;
#[cfg(feature = "yuv-444-packed")]
mod uyva;
#[cfg(feature = "v210")]
mod v210;
#[cfg(feature = "yuv-444-packed")]
mod v30x;
#[cfg(feature = "yuv-444-packed")]
mod v410;
#[cfg(feature = "yuv-444-packed")]
mod vuya;
#[cfg(feature = "yuv-444-packed")]
mod vuyx;
#[cfg(feature = "yuv-444-packed")]
mod vyu444;
#[cfg(feature = "yuv-444-packed")]
mod xv36;
#[cfg(feature = "yuv-444-packed")]
mod xv48;
#[cfg(feature = "xyz")]
mod xyz12;
#[cfg(feature = "y2xx")]
mod y210;
#[cfg(feature = "y2xx")]
mod y212;
#[cfg(feature = "y2xx")]
mod y216;
#[cfg(feature = "yuva")]
mod yuva_4_2_0;
#[cfg(feature = "yuva")]
mod yuva_4_2_2;
#[cfg(feature = "yuva")]
mod yuva_4_4_4;

#[cfg(all(test, feature = "std"))]
mod tests;

#[cfg(all(test, feature = "std"))]
mod api_smoke_tests {
  use super::*;

  #[cfg(feature = "v210")]
  #[test]
  fn mixed_sinker_default_does_not_produce_luma_u16() {
    // Use the currently available V210 source format marker for this smoke test.
    let sink: MixedSinker<'_, crate::source::V210> = MixedSinker::new(6, 1);
    assert!(!sink.produces_luma_u16());
  }

  #[test]
  fn luma_u16_buffer_too_short_error_displays() {
    let err = MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(100, 50));
    let msg = format!("{err}");
    assert!(msg.contains("100"));
    assert!(msg.contains("50"));
  }

  /// `MixedSinker` is routinely constructed as a stack local (tests, the
  /// crate's own callers), so its inline footprint must stay small enough to
  /// fit a constrained thread stack — the Windows (~1 MiB) and SDE-AVX-512
  /// test threads overflowed once the per-format scratch, the native-tier
  /// joins, and the per-plane area / filter streams accumulated inline. The
  /// large sub-structures (every lazily-created stream, every native join) are
  /// held behind a `Box`, so only a pointer sits inline; this ceiling fails the
  /// test if a future field reintroduces the bloat. The threshold sits well
  /// above the current size (a few KiB) yet far below any plausible thread
  /// stack budget.
  #[cfg(feature = "yuv-planar")]
  #[test]
  fn mixed_sinker_inline_size_stays_small() {
    use crate::{resample::AreaResampler, source::Yuv420p};
    // The largest practical monomorphization — a resampling Yuv420p sink with
    // every output and every native / filter join reachable.
    let area = core::mem::size_of::<MixedSinker<'_, Yuv420p, AreaResampler>>();
    let noop = core::mem::size_of::<MixedSinker<'_, Yuv420p>>();
    assert!(
      area < 16384,
      "MixedSinker<Yuv420p, AreaResampler> inline size {area} exceeds the 16 KiB \
       stack-footprint ceiling; box the largest new field(s) (mirror the existing \
       Option<Box<…>> stream / native-join fields)",
    );
    assert!(
      noop < 16384,
      "MixedSinker<Yuv420p, NoopResampler> inline size {noop} exceeds the 16 KiB ceiling",
    );
  }
}
