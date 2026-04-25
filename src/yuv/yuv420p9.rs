//! YUV 4:2:0 planar 9‑bit (`AV_PIX_FMT_YUV420P9LE`).
//!
//! Niche format — used by AVC High 9 Profile only; HEVC / VP9 / AV1
//! don't produce 9-bit. Reuses the same Q15 i32 kernel family as the
//! 10/12/14-bit siblings (`yuv_420p_n_to_rgb_*<BITS>`); the only
//! per-call difference is the const-generic `BITS = 9`, which fixes
//! the AND-mask to `0x1FF` and the Q15 scale via
//! `range_params_n::<9, _>`.

use crate::{
  ColorMatrix, PixelSink, SourceFormat,
  frame::{Yuv420p9Frame, Yuv420pFrame16},
  sealed::Sealed,
};

/// Zero‑sized marker for the YUV 4:2:0 **9‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv420p9;

impl Sealed for Yuv420p9 {}
impl SourceFormat for Yuv420p9 {}

/// One output row of a 9‑bit YUV 4:2:0 source.
#[derive(Debug, Clone, Copy)]
pub struct Yuv420p9Row<'a> {
  y: &'a [u16],
  u_half: &'a [u16],
  v_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv420p9Row<'a> {
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
  /// Full‑width Y row — `width` `u16` samples (low 9 bits active).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Half‑width U row — `width / 2` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u16] {
    self.u_half
  }
  /// Half‑width V row — `width / 2` `u16` samples.
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
  /// Full-range flag — `[0, 511]` Y, `[0, 511]` UV centered at 256.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume 9‑bit YUV 4:2:0 rows.
pub trait Yuv420p9Sink: for<'a> PixelSink<Input<'a> = Yuv420p9Row<'a>> {}

/// Walks a [`Yuv420p9Frame`] row by row into the sink.
pub fn yuv420p9_to<S: Yuv420p9Sink>(
  src: &Yuv420p9Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  yuv420p9_walker::<9, S>(src, full_range, matrix, sink)
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn yuv420p9_walker<const BITS: u32, S: Yuv420p9Sink>(
  src: &Yuv420pFrame16<'_, BITS>,
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

    let chroma_row = row / 2;
    let u_start = chroma_row * u_stride;
    let v_start = chroma_row * v_stride;
    let u_half = &u_plane[u_start..u_start + chroma_width];
    let v_half = &v_plane[v_start..v_start + chroma_width];

    sink.process(Yuv420p9Row::new(y, u_half, v_half, row, matrix, full_range))?;
  }
  Ok(())
}
