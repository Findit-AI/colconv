//! P412 — semi‑planar 4:4:4, 12‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P412LE`, FFmpeg 5.0+).
//!
//! Same layout as [`super::P410`] but with 12 active bits in the high
//! 12 positions of each `u16` (low 4 bits zero). Per-row kernel reuses
//! the 4:4:4 `p_n_444_to_rgb_*<12>` family; chroma is full-width × full-height.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::P412Frame, sealed::Sealed};

/// Zero‑sized marker for the P412 source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct P412;

impl Sealed for P412 {}
impl SourceFormat for P412 {}

/// One output row of a P412 source handed to a [`P412Sink`].
#[derive(Debug, Clone, Copy)]
pub struct P412Row<'a> {
  y: &'a [u16],
  uv_full: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> P412Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u16],
    uv_full: &'a [u16],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      uv_full,
      row,
      matrix,
      full_range,
    }
  }
  /// Full‑width Y row — `width` `u16` samples, high‑bit‑packed (12 bits).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Full-width interleaved UV row — `2 * width` `u16` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv_full(&self) -> &'a [u16] {
    self.uv_full
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
  /// Full-range flag.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`P412Row`].
pub trait P412Sink: for<'a> PixelSink<Input<'a> = P412Row<'a>> {}

/// Walks a [`P412Frame`] row by row into the sink.
pub fn p412_to<S: P412Sink>(
  src: &P412Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let uv_stride = src.uv_stride() as usize;
  let uv_row_elems = 2 * w;

  let y_plane = src.y();
  let uv_plane = src.uv();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    let uv_start = row * uv_stride;
    let uv_full = &uv_plane[uv_start..uv_start + uv_row_elems];

    sink.process(P412Row::new(y, uv_full, row, matrix, full_range))?;
  }
  Ok(())
}
