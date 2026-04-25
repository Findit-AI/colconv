//! YUV 4:4:0 planar 12‑bit (`AV_PIX_FMT_YUV440P12LE`).
//!
//! Full-width × half-height chroma at 12 bits per sample. Reuses
//! the const-generic `yuv_444p_n_to_rgb_*<12>` kernel family.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv440p12Frame, sealed::Sealed};

/// Zero‑sized marker for the YUV 4:4:0 **12‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv440p12;

impl Sealed for Yuv440p12 {}
impl SourceFormat for Yuv440p12 {}

/// One output row of a [`Yuv440p12`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuv440p12Row<'a> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv440p12Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u,
      v,
      row,
      matrix,
      full_range,
    }
  }
  /// Full‑width Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Full‑width U row (the half-height chroma row).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u(&self) -> &'a [u16] {
    self.u
  }
  /// Full‑width V row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v(&self) -> &'a [u16] {
    self.v
  }
  /// Row index.
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

/// Sinks that consume [`Yuv440p12Row`].
pub trait Yuv440p12Sink: for<'a> PixelSink<Input<'a> = Yuv440p12Row<'a>> {}

/// Walks a [`Yuv440p12Frame`] row by row into the sink.
pub fn yuv440p12_to<S: Yuv440p12Sink>(
  src: &Yuv440p12Frame<'_>,
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

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    let chroma_row = row / 2;
    let u_start = chroma_row * u_stride;
    let v_start = chroma_row * v_stride;
    let u = &u_plane[u_start..u_start + w];
    let v = &v_plane[v_start..v_start + w];

    sink.process(Yuv440p12Row::new(y, u, v, row, matrix, full_range))?;
  }
  Ok(())
}
