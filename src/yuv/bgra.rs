//! Packed BGRA source (`AV_PIX_FMT_BGRA`) — 8 bits per channel,
//! byte order `B, G, R, A`. The 4th byte is real alpha (not
//! padding); only the channel order on the first three bytes
//! distinguishes this from [`super::Rgba`].
//!
//! Outputs (Ship 9b):
//! - `with_rgb` — `bgra_to_rgb_row` (R↔B swap + drop alpha).
//! - `with_rgba` — `bgra_to_rgba_row` (R↔B swap, alpha preserved).
//! - `with_luma` — `bgra_to_rgb_row` into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::BgraFrame, sealed::Sealed};

/// Zero‑sized marker for the packed **BGRA** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bgra;

impl Sealed for Bgra {}
impl SourceFormat for Bgra {}

/// One output row of a [`Bgra`] source — `width * 4` packed
/// `B, G, R, A` bytes.
#[derive(Debug, Clone, Copy)]
pub struct BgraRow<'a> {
  bgra: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> BgraRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(bgra: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      bgra,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `B, G, R, A, B, G, R, A, …` row — `4 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn bgra(&self) -> &'a [u8] {
    self.bgra
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

/// Sinks that consume [`BgraRow`].
pub trait BgraSink: for<'a> PixelSink<Input<'a> = BgraRow<'a>> {}

/// Walks a [`BgraFrame`] row by row into the sink.
pub fn bgra_to<S: BgraSink>(
  src: &BgraFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.bgra();

  for row in 0..h {
    let start = row * stride;
    let bgra = &plane[start..start + row_bytes];
    sink.process(BgraRow::new(bgra, row, matrix, full_range))?;
  }
  Ok(())
}
