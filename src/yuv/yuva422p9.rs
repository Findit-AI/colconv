//! YUVA 4:2:2 planar 9‑bit (`AV_PIX_FMT_YUVA422P9LE`).
//!
//! Storage mirrors [`super::Yuv422p9`] (Y full-width × full-height,
//! U / V half-width × full-height) plus a fourth full-resolution
//! alpha plane (1:1 with Y; only chroma is subsampled in 4:2:2).
//! Sample width is **`u16`** (9 active bits in the low bits of each
//! element).
//!
//! Per-row dispatcher reuses
//! `yuv_420p_n_to_rgba*_with_alpha_src_row::<9>` (in `crate::row`) at
//! the row level — chroma layout for any single Y row is identical to
//! 4:2:0 (half-width U/V); the 4:2:0 vs 4:2:2 difference is purely in
//! the vertical walker.

use crate::{
  ColorMatrix, PixelSink, SourceFormat,
  frame::{Yuva422p9Frame, Yuva422pFrame16},
  sealed::Sealed,
};

/// Zero‑sized marker for the YUVA 4:2:2 **9‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuva422p9;

impl Sealed for Yuva422p9 {}
impl SourceFormat for Yuva422p9 {}

/// One output row of a [`Yuva422p9`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuva422p9Row<'a> {
  y: &'a [u16],
  u_half: &'a [u16],
  v_half: &'a [u16],
  a: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuva422p9Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u16],
    u_half: &'a [u16],
    v_half: &'a [u16],
    a: &'a [u16],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u_half,
      v_half,
      a,
      row,
      matrix,
      full_range,
    }
  }
  /// Full‑width Y row — `width` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Half‑width U row — `width / 2` `u16` samples for *this* Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u16] {
    self.u_half
  }
  /// Half‑width V row — `width / 2` `u16` samples for *this* Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v_half(&self) -> &'a [u16] {
    self.v_half
  }
  /// Full‑width alpha row — `width` `u16` samples, low‑bit‑packed at
  /// 9 bits (1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn a(&self) -> &'a [u16] {
    self.a
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
  /// Full‑range flag.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Yuva422p9Row`].
pub trait Yuva422p9Sink: for<'a> PixelSink<Input<'a> = Yuva422p9Row<'a>> {}

/// Walks a [`Yuva422p9Frame`] row by row into the sink.
pub fn yuva422p9_to<S: Yuva422p9Sink>(
  src: &Yuva422p9Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  yuva422p9_walker::<9, S>(src, full_range, matrix, sink)
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva422p9_walker<const BITS: u32, S: Yuva422p9Sink>(
  src: &Yuva422pFrame16<'_, BITS>,
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
  let a_stride = src.a_stride() as usize;
  let chroma_width = w / 2;

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();
  let a_plane = src.a();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:2:2: chroma is full-height (one chroma row per Y row).
    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let u_half = &u_plane[u_start..u_start + chroma_width];
    let v_half = &v_plane[v_start..v_start + chroma_width];

    let a_start = row * a_stride;
    let a = &a_plane[a_start..a_start + w];

    sink.process(Yuva422p9Row::new(
      y, u_half, v_half, a, row, matrix, full_range,
    ))?;
  }
  Ok(())
}
