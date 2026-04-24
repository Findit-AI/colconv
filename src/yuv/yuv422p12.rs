//! YUV 4:2:2 planar 12‑bit (`AV_PIX_FMT_YUV422P12LE`). See
//! [`super::Yuv422p10`] for the 4:2:2 family structure.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv422p12Frame, sealed::Sealed};

/// Zero‑sized marker for the YUV 4:2:2 **12‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv422p12;

impl Sealed for Yuv422p12 {}
impl SourceFormat for Yuv422p12 {}

/// One output row of a [`Yuv422p12`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuv422p12Row<'a> {
  y: &'a [u16],
  u_half: &'a [u16],
  v_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv422p12Row<'a> {
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
  /// Full‑width Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Half‑width U row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u16] {
    self.u_half
  }
  /// Half‑width V row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v_half(&self) -> &'a [u16] {
    self.v_half
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

/// Sinks that consume [`Yuv422p12Row`].
pub trait Yuv422p12Sink: for<'a> PixelSink<Input<'a> = Yuv422p12Row<'a>> {}

/// Walks a [`Yuv422p12Frame`] row by row into the sink.
pub fn yuv422p12_to<S: Yuv422p12Sink>(
  src: &Yuv422p12Frame<'_>,
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

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &src.y()[y_start..y_start + w];
    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let u_half = &src.u()[u_start..u_start + chroma_width];
    let v_half = &src.v()[v_start..v_start + chroma_width];

    sink.process(Yuv422p12Row::new(
      y, u_half, v_half, row, matrix, full_range,
    ))?;
  }
  Ok(())
}
