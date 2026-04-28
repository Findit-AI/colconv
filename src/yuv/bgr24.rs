//! Packed BGR24 source (`AV_PIX_FMT_BGR24`) — 8 bits per channel,
//! byte order `B, G, R`. Storage and validation mirror
//! [`super::Rgb24`]; only the channel order at the row level differs.
//!
//! Outputs:
//! - `with_rgb` — `bgr_to_rgb_row` (B↔R swap during the copy).
//! - `with_rgba` — swap then append `0xFF` alpha (sinker calls
//!   `bgr_to_rgb_row` into a scratch buffer first).
//! - `with_luma` — swap then `rgb_to_luma_row` (RGB-input kernel).
//! - `with_hsv` — swap then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Bgr24Frame, sealed::Sealed};

/// Zero‑sized marker for the packed **BGR24** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bgr24;

impl Sealed for Bgr24 {}
impl SourceFormat for Bgr24 {}

/// One output row of a [`Bgr24`] source — `width * 3` packed
/// `B, G, R` bytes.
#[derive(Debug, Clone, Copy)]
pub struct Bgr24Row<'a> {
  bgr: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Bgr24Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(bgr: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      bgr,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `B, G, R, B, G, R, …` row — `3 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn bgr(&self) -> &'a [u8] {
    self.bgr
  }
  /// Row index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// Color matrix (used when sinks derive luma).
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

/// Sinks that consume [`Bgr24Row`].
pub trait Bgr24Sink: for<'a> PixelSink<Input<'a> = Bgr24Row<'a>> {}

/// Walks a [`Bgr24Frame`] row by row into the sink.
pub fn bgr24_to<S: Bgr24Sink>(
  src: &Bgr24Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 3;
  let plane = src.bgr();

  for row in 0..h {
    let start = row * stride;
    let bgr = &plane[start..start + row_bytes];
    sink.process(Bgr24Row::new(bgr, row, matrix, full_range))?;
  }
  Ok(())
}
