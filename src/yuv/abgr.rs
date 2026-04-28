//! Packed ABGR source (`AV_PIX_FMT_ABGR`) — 8 bits per channel,
//! byte order `A, B, G, R`. Leading alpha + reversed RGB order
//! relative to [`super::Argb`].
//!
//! Outputs (Ship 9c):
//! - `with_rgb` — `abgr_to_rgb_row` (drop alpha + R↔B swap).
//! - `with_rgba` — `abgr_to_rgba_row` (full byte reverse: alpha
//!   rotates to trailing AND inner three bytes flip).
//! - `with_luma` — same swap path into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::AbgrFrame, sealed::Sealed};

/// Zero‑sized marker for the packed **ABGR** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Abgr;

impl Sealed for Abgr {}
impl SourceFormat for Abgr {}

/// One output row of an [`Abgr`] source — `width * 4` packed
/// `A, B, G, R` bytes.
#[derive(Debug, Clone, Copy)]
pub struct AbgrRow<'a> {
  abgr: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> AbgrRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(abgr: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      abgr,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `A, B, G, R, A, B, G, R, …` row — `4 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn abgr(&self) -> &'a [u8] {
    self.abgr
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

/// Sinks that consume [`AbgrRow`].
pub trait AbgrSink: for<'a> PixelSink<Input<'a> = AbgrRow<'a>> {}

/// Walks an [`AbgrFrame`] row by row into the sink.
pub fn abgr_to<S: AbgrSink>(
  src: &AbgrFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.abgr();

  for row in 0..h {
    let start = row * stride;
    let abgr = &plane[start..start + row_bytes];
    sink.process(AbgrRow::new(abgr, row, matrix, full_range))?;
  }
  Ok(())
}
