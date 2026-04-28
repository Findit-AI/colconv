//! Packed ARGB source (`AV_PIX_FMT_ARGB`) — 8 bits per channel,
//! byte order `A, R, G, B`. The 1st byte is real alpha (not
//! padding); leading-alpha layout is the only difference from
//! [`super::Rgba`].
//!
//! Outputs (Ship 9c):
//! - `with_rgb` — `argb_to_rgb_row` (drop leading alpha).
//! - `with_rgba` — `argb_to_rgba_row` (rotate alpha to trailing).
//! - `with_luma` — drop leading alpha into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::ArgbFrame, sealed::Sealed};

/// Zero‑sized marker for the packed **ARGB** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Argb;

impl Sealed for Argb {}
impl SourceFormat for Argb {}

/// One output row of an [`Argb`] source — `width * 4` packed
/// `A, R, G, B` bytes.
#[derive(Debug, Clone, Copy)]
pub struct ArgbRow<'a> {
  argb: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> ArgbRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(argb: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      argb,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `A, R, G, B, A, R, G, B, …` row — `4 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn argb(&self) -> &'a [u8] {
    self.argb
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

/// Sinks that consume [`ArgbRow`].
pub trait ArgbSink: for<'a> PixelSink<Input<'a> = ArgbRow<'a>> {}

/// Walks an [`ArgbFrame`] row by row into the sink.
pub fn argb_to<S: ArgbSink>(
  src: &ArgbFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.argb();

  for row in 0..h {
    let start = row * stride;
    let argb = &plane[start..start + row_bytes];
    sink.process(ArgbRow::new(argb, row, matrix, full_range))?;
  }
  Ok(())
}
