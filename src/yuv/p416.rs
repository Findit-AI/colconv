//! P416 — semi‑planar 4:4:4, 16‑bit (`AV_PIX_FMT_P416LE`).
//!
//! 4:4:4 twin of [`super::P016`]. At 16 bits every bit is active.
//! Per-row kernel uses the parallel i64-chroma `p_n_444_16_to_rgb_*`
//! family (chroma matrix multiply-add overflows i32 at 16 bits, same
//! rationale as `p16_to_rgb_*` and `yuv444p16_to_rgb_*`).

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::P416Frame, sealed::Sealed};

/// Zero‑sized marker for the P416 source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct P416;

impl Sealed for P416 {}
impl SourceFormat for P416 {}

/// One output row of a P416 source handed to a [`P416Sink`].
#[derive(Debug, Clone, Copy)]
pub struct P416Row<'a> {
  y: &'a [u16],
  uv_full: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> P416Row<'a> {
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
  /// Full‑width Y row — `width` `u16` samples, full 16-bit range.
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

/// Sinks that consume [`P416Row`].
pub trait P416Sink: for<'a> PixelSink<Input<'a> = P416Row<'a>> {}

/// Walks a [`P416Frame`] row by row into the sink.
pub fn p416_to<S: P416Sink>(
  src: &P416Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let uv_stride = src.uv_stride() as usize;
  // 4:4:4 semi-planar: full-width × 2 elements per pair. See
  // P410's walker for the rationale.
  let uv_row_elems = 2 * w;

  let y_plane = src.y();
  let uv_plane = src.uv();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    let uv_start = row * uv_stride;
    let uv_full = &uv_plane[uv_start..uv_start + uv_row_elems];

    sink.process(P416Row::new(y, uv_full, row, matrix, full_range))?;
  }
  Ok(())
}
