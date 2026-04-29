//! Packed YUV 4:2:2 source (`AV_PIX_FMT_YUYV422`, also known as
//! YUY2). One plane, byte order `Y0, U0, Y1, V0` per 2-pixel
//! block — Y in even byte positions, U/V in odd positions with
//! U preceding V.
//!
//! Common output of older codecs (M-JPEG, DV), Windows DirectShow /
//! V4L2 webcams, and 8-bit SDI capture in YUY2 mode.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline.
//! - `with_luma` — copies the Y bytes from even positions of the row.
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.
//!
//! The companion [`super::Uyvy422`] format swaps Y and UV positions;
//! [`super::Yvyu422`] swaps U and V relative to YUYV. All three reuse
//! the same const-generic `yuv422_packed_to_rgb_or_rgba_row` template
//! across scalar + every SIMD backend.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuyv422Frame, sealed::Sealed};

/// Zero‑sized marker for the packed **YUYV422** source format. Used
/// as the `F` type parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuyv422;

impl Sealed for Yuyv422 {}
impl SourceFormat for Yuyv422 {}

/// One output row of a [`Yuyv422`] source — `2 * width` packed
/// `Y0, U0, Y1, V0, …` bytes.
#[derive(Debug, Clone, Copy)]
pub struct Yuyv422Row<'a> {
  yuyv: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuyv422Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(yuyv: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      yuyv,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `Y0, U0, Y1, V0, …` row — `2 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn yuyv(&self) -> &'a [u8] {
    self.yuyv
  }
  /// Row index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// YUV → RGB matrix carried through from the kernel call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }
  /// `true` iff Y ∈ `[0, 255]` (full range); `false` for limited.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Yuyv422Row`].
pub trait Yuyv422Sink: for<'a> PixelSink<Input<'a> = Yuyv422Row<'a>> {}

/// Walks a [`Yuyv422Frame`] row by row into the sink.
pub fn yuyv422_to<S: Yuyv422Sink>(
  src: &Yuyv422Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 2;
  let plane = src.yuyv();

  for row in 0..h {
    let start = row * stride;
    let yuyv = &plane[start..start + row_bytes];
    sink.process(Yuyv422Row::new(yuyv, row, matrix, full_range))?;
  }
  Ok(())
}
