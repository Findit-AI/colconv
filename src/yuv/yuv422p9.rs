//! YUV 4:2:2 planar 9‑bit (`AV_PIX_FMT_YUV422P9LE`).
//!
//! Same `u16`-backed layout as [`super::Yuv422p10`] with 9 active
//! bits in the low 9 of each element. Niche format — AVC High 9
//! profile only. Per-row kernel reuses the 4:2:0 family at
//! `BITS = 9` (`yuv_420p_n_to_rgb_row::<9>` and friends, internal
//! to `crate::row`) verbatim — same shape (half-width chroma per
//! row), only the vertical walk differs.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv422p9Frame, sealed::Sealed};

/// Zero‑sized marker for the YUV 4:2:2 **9‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv422p9;

impl Sealed for Yuv422p9 {}
impl SourceFormat for Yuv422p9 {}

/// One output row of a [`Yuv422p9`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuv422p9Row<'a> {
  y: &'a [u16],
  u_half: &'a [u16],
  v_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv422p9Row<'a> {
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
  /// Full-range flag.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Yuv422p9Row`].
pub trait Yuv422p9Sink: for<'a> PixelSink<Input<'a> = Yuv422p9Row<'a>> {}

/// Walks a [`Yuv422p9Frame`] row by row into the sink.
pub fn yuv422p9_to<S: Yuv422p9Sink>(
  src: &Yuv422p9Frame<'_>,
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

    sink.process(Yuv422p9Row::new(y, u_half, v_half, row, matrix, full_range))?;
  }
  Ok(())
}
