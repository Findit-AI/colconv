//! Packed RGB24 source (`AV_PIX_FMT_RGB24`) — 8 bits per channel,
//! byte order `R, G, B`.
//!
//! Unlike every other source format in this crate, the input is
//! already **RGB**, not YUV — there's no chroma matrix work. Outputs
//! are produced by:
//! - `with_rgb` — identity copy (RGB in → RGB out).
//! - `with_rgba` — `expand_rgb_to_rgba_row` with constant `0xFF` alpha.
//! - `with_luma` — `rgb_to_luma_row` (BT.709 / 601 / etc. coefficients).
//! - `with_hsv` — existing `rgb_to_hsv_row` kernel.
//!
//! The companion [`super::Bgr24`] format swaps R↔B at the row level
//! before reusing the same RGB-input kernels.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Rgb24Frame, sealed::Sealed};

/// Zero‑sized marker for the packed **RGB24** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Rgb24;

impl Sealed for Rgb24 {}
impl SourceFormat for Rgb24 {}

/// One output row of an [`Rgb24`] source — `width * 3` packed
/// `R, G, B` bytes.
#[derive(Debug, Clone, Copy)]
pub struct Rgb24Row<'a> {
  rgb: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Rgb24Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(rgb: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      rgb,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `R, G, B, R, G, B, …` row — `3 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn rgb(&self) -> &'a [u8] {
    self.rgb
  }
  /// Row index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// Color matrix (used when sinks derive luma / convert to YUV-space).
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

/// Sinks that consume [`Rgb24Row`].
pub trait Rgb24Sink: for<'a> PixelSink<Input<'a> = Rgb24Row<'a>> {}

/// Walks an [`Rgb24Frame`] row by row into the sink.
pub fn rgb24_to<S: Rgb24Sink>(
  src: &Rgb24Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 3;
  let plane = src.rgb();

  for row in 0..h {
    let start = row * stride;
    let rgb = &plane[start..start + row_bytes];
    sink.process(Rgb24Row::new(rgb, row, matrix, full_range))?;
  }
  Ok(())
}
