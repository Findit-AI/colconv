//! YUV 4:4:4 planar 16‑bit (`AV_PIX_FMT_YUV444P16LE`). The HW→SW
//! download target for CUDA / NVDEC 4:4:4 HDR content.
//!
//! Uses a **parallel i64 kernel family** for the u16‑output path —
//! same rationale as [`super::Yuv420p16`] (`coeff × u_d` at 16 bits
//! overflows i32 for Bt2020 blue).

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv444p16Frame, sealed::Sealed};

/// Zero‑sized marker for the YUV 4:4:4 **16‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv444p16;

impl Sealed for Yuv444p16 {}
impl SourceFormat for Yuv444p16 {}

/// One output row of a [`Yuv444p16`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuv444p16Row<'a> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv444p16Row<'a> {
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
  /// Full‑width U row (1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u(&self) -> &'a [u16] {
    self.u
  }
  /// Full‑width V row (1:1 with Y).
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

/// Sinks that consume [`Yuv444p16Row`].
pub trait Yuv444p16Sink: for<'a> PixelSink<Input<'a> = Yuv444p16Row<'a>> {}

/// Walks a [`Yuv444p16Frame`] row by row into the sink.
pub fn yuv444p16_to<S: Yuv444p16Sink>(
  src: &Yuv444p16Frame<'_>,
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

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &src.y()[y_start..y_start + w];
    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let u = &src.u()[u_start..u_start + w];
    let v = &src.v()[v_start..v_start + w];

    sink.process(Yuv444p16Row::new(y, u, v, row, matrix, full_range))?;
  }
  Ok(())
}
