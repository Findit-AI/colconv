//! P016 — semi‑planar 4:2:0, 16‑bit (`AV_PIX_FMT_P016LE`).
//!
//! Storage is identical to [`super::P010`] / [`super::P012`]: one
//! full‑size Y plane plus one interleaved UV plane at half width and
//! half height. At 16 bits there is no high‑vs‑low distinction — the
//! full `u16` range is active, so `P016` and a hypothetical
//! `yuv420p16le`‑shaped `PnFrame<16>` are numerically identical (the
//! layout difference is only in the plane count / interleave, not
//! sample packing).
//!
//! Runs on the **parallel i64 kernel family** —
//! [`crate::row::p016_to_rgb_row`] dispatches to
//! `scalar::p16_to_rgb_*` plus the matching per-backend SIMD kernels,
//! which widen the chroma matrix multiply to i64.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::P016Frame, sealed::Sealed};

/// Zero‑sized marker for the P016 source format. Used as the `F` type
/// parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct P016;

impl Sealed for P016 {}
impl SourceFormat for P016 {}

/// One output row of a P016 source handed to a [`P016Sink`].
///
/// Shape matches [`super::P010Row`] / [`super::P012Row`]: full‑width
/// Y plus interleaved `U0,V0,U1,V1,…` UV.
#[derive(Debug, Clone, Copy)]
pub struct P016Row<'a> {
  y: &'a [u16],
  uv_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> P016Row<'a> {
  /// Bundles one row of a P016 source for a [`P016Sink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u16],
    uv_half: &'a [u16],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      uv_half,
      row,
      matrix,
      full_range,
    }
  }

  /// Full‑width Y (luma) row — `width` `u16` samples, full 16‑bit
  /// value in each element.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }

  /// Interleaved UV row — `width` `u16` elements laid out as
  /// `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv_half(&self) -> &'a [u16] {
    self.uv_half
  }

  /// Output row index within the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// YUV → RGB matrix carried through from the kernel call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }

  /// `true` iff Y uses the full sample range (`[0, 65535]` for
  /// 16‑bit); `false` for limited range.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume P016 rows.
pub trait P016Sink: for<'a> PixelSink<Input<'a> = P016Row<'a>> {}

/// Converts a P016 frame by walking its rows and feeding each one to
/// the [`P016Sink`].
pub fn p016_to<S: P016Sink>(
  src: &P016Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let uv_stride = src.uv_stride() as usize;
  let uv_row_elems = w;

  let y_plane = src.y();
  let uv_plane = src.uv();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    let chroma_row = row / 2;
    let uv_start = chroma_row * uv_stride;
    let uv_half = &uv_plane[uv_start..uv_start + uv_row_elems];

    sink.process(P016Row::new(y, uv_half, row, matrix, full_range))?;
  }
  Ok(())
}
