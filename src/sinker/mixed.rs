//! [`MixedSinker`] — the common "I want some subset of {RGB, Luma, HSV}
//! written into my own buffers" consumer.
//!
//! Generic over the source format via an `F: SourceFormat` type
//! parameter. One `PixelSink` impl per supported format; v0.1 ships
//! the [`Yuv420p`](crate::yuv::Yuv420p) and [`Nv12`](crate::yuv::Nv12)
//! impls. All configuration and processing methods are fallible — no
//! panics under normal contract violations — so the sink is usable on
//! `panic = "abort"` targets.

use core::marker::PhantomData;

use std::vec::Vec;

use derive_more::{Display, IsVariant};
use thiserror::Error;

use crate::{
  HsvBuffers, PixelSink, SourceFormat,
  row::{nv12_to_rgb_row, rgb_to_hsv_row, yuv_420_to_rgb_row},
  yuv::{Nv12, Nv12Row, Nv12Sink, Yuv420p, Yuv420pRow, Yuv420pSink},
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
/// support for — NV21 (VU instead of UV), YUV422p / YUV444p (full‑width
/// chroma), P010 / P016 (10/16‑bit planes), etc. — will add its own
/// variant. Pattern matches from downstream code should include a
/// `_ => …` arm.
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
/// `F` identifies the source format — `Yuv420p`, `Nv12`, `Bgr24`, etc.
/// Each format provides its own `impl PixelSink for MixedSinker<'_, F>`
/// (the only `impl` landed in v0.1 is for [`Yuv420p`]).
pub struct MixedSinker<'a, F: SourceFormat> {
  rgb: Option<&'a mut [u8]>,
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
      luma: None,
      hsv: None,
      width,
      height,
      rgb_scratch: Vec::new(),
      simd: true,
      _fmt: PhantomData,
    }
  }

  /// Returns `true` iff the sinker will write RGB.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb(&self) -> bool {
    self.rgb.is_some()
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
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth: `begin_frame` already validated frame‑level
    // dimensions, so these checks are unreachable from the walker.
    // They guard direct `process` callers (hand-crafted rows, row
    // replay) from handing a wrong-shaped row or out-of-range index
    // to unsafe SIMD kernels. Report the offending slice length and
    // row index directly — don't reuse `DimensionMismatch`, whose
    // `frame_w` / `frame_h` fields would be meaningless here.
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

    // With preflight validation in place, `idx < h`, and `w × h × N`
    // guaranteed not to overflow, every row-slice range below is
    // in-bounds by construction — no per-row size checks needed.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let rgb_plane_start = one_plane_start * 3;
    let rgb_plane_end = one_plane_end * 3;
    let rgb_row_bytes = w * 3;

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
    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => &mut buf[rgb_plane_start..rgb_plane_end],
      None => {
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
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Nv12Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth shape check (see Yuv420p impl above). An NV12
    // UV row is `width` bytes of interleaved U / V payload — same
    // length as Y — so both slices must equal `self.width`.
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

    // Row slices are in-bounds by construction: preflight validation
    // guaranteed `width × height × N ≤ buffer.len()`, and `idx < h`
    // holds per the assert above.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let rgb_plane_start = one_plane_start * 3;
    let rgb_plane_end = one_plane_end * 3;
    let rgb_row_bytes = w * 3;

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
      Some(buf) => &mut buf[rgb_plane_start..rgb_plane_end],
      None => {
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
    frame::{Nv12Frame, Yuv420pFrame},
    yuv::{nv12_to, yuv420p_to},
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
}
