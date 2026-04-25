//! YUV 4:2:2 planar 10‑bit (`AV_PIX_FMT_YUV422P10LE`).
//!
//! Same `u16`‑backed layout as [`super::Yuv420p10`] with 4:2:2 chroma
//! (half‑width, **full‑height**). Per‑row kernel reuses the 4:2:0
//! family — [`crate::row::yuv420p10_to_rgb_row`] — verbatim. See
//! [`super::Yuv422p`] for the axis‑difference rationale.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv422p10Frame, sealed::Sealed};

/// Zero‑sized marker for the YUV 4:2:2 **10‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv422p10;

impl Sealed for Yuv422p10 {}
impl SourceFormat for Yuv422p10 {}

/// One output row of a [`Yuv422p10`] source handed to a [`Yuv422p10Sink`].
#[derive(Debug, Clone, Copy)]
pub struct Yuv422p10Row<'a> {
  y: &'a [u16],
  u_half: &'a [u16],
  v_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv422p10Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u16],
    u_half: &'a [u16],
    v_half: &'a [u16],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u_half,
      v_half,
      row,
      matrix,
      full_range,
    }
  }
  /// Full‑width Y row — `width` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Half‑width U row — `width / 2` samples for *this* Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u16] {
    self.u_half
  }
  /// Half‑width V row — `width / 2` samples for *this* Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v_half(&self) -> &'a [u16] {
    self.v_half
  }
  /// Output row index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// YUV → RGB matrix.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }
  /// `true` for full-range Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Yuv422p10Row`].
pub trait Yuv422p10Sink: for<'a> PixelSink<Input<'a> = Yuv422p10Row<'a>> {}

/// Walks a [`Yuv422p10Frame`] row by row into the sink. Chroma
/// advances every row (vs 4:2:0's `row / 2`).
pub fn yuv422p10_to<S: Yuv422p10Sink>(
  src: &Yuv422p10Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let u_stride = src.u_stride() as usize;
  let v_stride = src.v_stride() as usize;
  let chroma_width = w / 2;

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];
    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let u_half = &u_plane[u_start..u_start + chroma_width];
    let v_half = &v_plane[v_start..v_start + chroma_width];

    sink.process(Yuv422p10Row::new(
      y, u_half, v_half, row, matrix, full_range,
    ))?;
  }
  Ok(())
}
