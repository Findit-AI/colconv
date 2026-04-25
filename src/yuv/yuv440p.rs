//! YUV 4:4:0 planar 8-bit (`AV_PIX_FMT_YUV440P` / `AV_PIX_FMT_YUVJ440P`).
//!
//! Full-width chroma, **half-height** — the axis-flipped counterpart
//! to [`super::Yuv422p`]. Mostly seen from JPEG decoders that
//! subsample vertically only.
//!
//! Per-row kernel reuses [`super::Yuv444p`]'s `yuv_444_to_rgb_row`:
//! per-row math is identical (full-width chroma, no horizontal
//! duplication); only the walker reads chroma row `r / 2` instead
//! of `r`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv440pFrame, sealed::Sealed};

/// Zero‑sized marker for the YUV 4:4:0 source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv440p;

impl Sealed for Yuv440p {}
impl SourceFormat for Yuv440p {}

/// One output row of a [`Yuv440p`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuv440pRow<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv440pRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
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
  pub fn y(&self) -> &'a [u8] {
    self.y
  }
  /// Full‑width U row (the chroma row shared with the previous /
  /// next Y row).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u(&self) -> &'a [u8] {
    self.u
  }
  /// Full‑width V row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v(&self) -> &'a [u8] {
    self.v
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
  /// Full-range flag (`yuvj440p` ⇔ `true`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Yuv440pRow`].
pub trait Yuv440pSink: for<'a> PixelSink<Input<'a> = Yuv440pRow<'a>> {}

/// Walks a [`Yuv440pFrame`] row by row into the sink. Y row `r`
/// reads chroma row `r / 2` (half-height vertical subsampling).
pub fn yuv440p_to<S: Yuv440pSink>(
  src: &Yuv440pFrame<'_>,
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

    sink.process(Yuv440pRow::new(y, u, v, row, matrix, full_range))?;
  }
  Ok(())
}
