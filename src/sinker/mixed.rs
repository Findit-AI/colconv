//! [`MixedSinker`] — the common "I want some subset of {RGB, Luma, HSV}
//! written into my own buffers" consumer.
//!
//! Generic over the source format via an `F: SourceFormat` type
//! parameter. One `PixelSink` impl per supported format; v0.1 ships
//! the [`Yuv420p`](crate::yuv::Yuv420p),
//! [`Nv12`](crate::yuv::Nv12), and [`Nv21`](crate::yuv::Nv21) impls.
//! All configuration and processing methods are fallible — no panics
//! under normal contract violations — so the sink is usable on
//! `panic = "abort"` targets.

use core::marker::PhantomData;

use std::vec::Vec;

use derive_more::{Display, IsVariant};
use thiserror::Error;

use crate::{
  HsvBuffers, PixelSink, SourceFormat,
  row::{
    nv12_to_rgb_row, nv21_to_rgb_row, rgb_to_hsv_row, yuv_420_to_rgb_row, yuv420p10_to_rgb_row,
    yuv420p10_to_rgb_u16_row,
  },
  yuv::{
    Nv12, Nv12Row, Nv12Sink, Nv21, Nv21Row, Nv21Sink, Yuv420p, Yuv420p10, Yuv420p10Row,
    Yuv420p10Sink, Yuv420pRow, Yuv420pSink,
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
  #[error(
    "MixedSinker row shape mismatch at row {row}: {which} slice has {actual} bytes, expected {expected}"
  )]
  RowShapeMismatch {
    /// Which slice mismatched. See [`RowSlice`] for variants.
    which: RowSlice,
    /// Row index reported by the offending row.
    row: usize,
    /// Expected slice length in bytes (given the sink's configured width).
    expected: usize,
    /// Actual slice length supplied by the row.
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
/// `Bgr24`, etc. Each format provides its own
/// `impl PixelSink for MixedSinker<'_, F>`. v0.1 ships impls for
/// [`Yuv420p`](crate::yuv::Yuv420p), [`Nv12`](crate::yuv::Nv12), and
/// [`Nv21`](crate::yuv::Nv21).
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
  _fmt: PhantomData<F>,
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

  /// Attaches a packed **`u16`** RGB output buffer sized in elements,
  /// not bytes. For the 10‑bit source path
  /// ([`Yuv420p10`](crate::yuv::Yuv420p10)) each element carries a
  /// 10‑bit value in the **low** 10 bits (upper 6 bits zero), matching
  /// FFmpeg's `yuv420p10le` convention. This is **not** the `p010`
  /// layout (which stores samples in the high 10 bits); callers
  /// feeding a p010 consumer must shift the output left by 6.
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if
  /// `buf.len() < width × height × 3` `u16` elements, or
  /// `Err(GeometryOverflow)` on 32‑bit targets when the product
  /// overflows.
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

// ---- Yuv420p10 impl -----------------------------------------------------

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

/// Returns `Ok(())` iff the walker's frame dimensions exactly match
/// the sinker's configured dimensions. Called from
/// [`PixelSink::begin_frame`] on both `MixedSinker<Yuv420p>` and
/// `MixedSinker<Nv12>`.
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

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{
    ColorMatrix,
    frame::{Nv12Frame, Nv21Frame, Yuv420p10Frame, Yuv420pFrame},
    yuv::{nv12_to, nv21_to, yuv420p_to, yuv420p10_to},
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
  fn yuv420p_odd_width_sink_returns_err_at_begin_frame() {
    // A sink configured with an odd width would later panic inside
    // `yuv_420_to_rgb_row` (which asserts `width & 1 == 0`). The
    // fallible API surfaces this as `OddWidth` at frame start — no
    // rows are processed, no panic. Width=15, height=8 — matching
    // frame so `DimensionMismatch` can't fire first.
    let w = 15usize;
    let h = 8usize;
    let y = std::vec![0u8; w * h];
    let u = std::vec![128u8; ((w + 1) / 2) * h / 2 + 8]; // any valid size
    let v = std::vec![128u8; ((w + 1) / 2) * h / 2 + 8];
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
  fn nv12_odd_width_sink_returns_err_at_begin_frame() {
    let w = 15usize;
    let h = 8usize;
    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Nv12>::new(w, h).with_rgb(&mut rgb).unwrap();
    let err = sink.begin_frame(w as u32, h as u32).err().unwrap();
    assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
  }

  #[test]
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
  fn yuv420p10_rgb_u16_too_short_returns_err() {
    let mut rgb = std::vec![0u16; 10]; // Way too small.
    let err = MixedSinker::<Yuv420p10>::new(16, 8)
      .with_rgb_u16(&mut rgb)
      .err()
      .unwrap();
    assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
  }

  #[test]
  fn yuv420p10_with_simd_false_matches_with_simd_true() {
    // Until real SIMD kernels land, the stubs call into scalar —
    // but the SIMD toggle still exercises the dispatch path. The
    // two results must be byte‑identical on both outputs.
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
}
