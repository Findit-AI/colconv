//! P216 — semi‑planar 4:2:2, 16‑bit (`AV_PIX_FMT_P216LE`).
//!
//! 4:2:2 twin of [`super::P016`]. At 16 bits the high-vs-low packing
//! distinction degenerates — every bit is active. Per-row kernel
//! reuses the 4:2:0 `p16_to_rgb_*` parallel i64-chroma family
//! verbatim; only the walker reads chroma row `r` instead of `r / 2`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::P216Frame, sealed::Sealed};

/// Zero‑sized marker for the P216 source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct P216;

impl Sealed for P216 {}
impl SourceFormat for P216 {}

/// One output row of a P216 source handed to a [`P216Sink`].
#[derive(Debug, Clone, Copy)]
pub struct P216Row<'a> {
  y: &'a [u16],
  uv_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> P216Row<'a> {
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
  /// Full‑width Y row — `width` `u16` samples, full 16-bit range.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Half-width interleaved UV row — `width` `u16` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv_half(&self) -> &'a [u16] {
    self.uv_half
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

/// Sinks that consume [`P216Row`].
pub trait P216Sink: for<'a> PixelSink<Input<'a> = P216Row<'a>> {}

/// Walks a [`P216Frame`] row by row into the sink.
pub fn p216_to<S: P216Sink>(
  src: &P216Frame<'_>,
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

    let uv_start = row * uv_stride;
    let uv_half = &uv_plane[uv_start..uv_start + uv_row_elems];

    sink.process(P216Row::new(y, uv_half, row, matrix, full_range))?;
  }
  Ok(())
}
