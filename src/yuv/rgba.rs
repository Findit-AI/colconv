//! Packed RGBA source (`AV_PIX_FMT_RGBA`) — 8 bits per channel,
//! byte order `R, G, B, A`. The 4th byte is real alpha (not
//! padding).
//!
//! Outputs (Ship 9b):
//! - `with_rgb` — `rgba_to_rgb_row` (drop alpha; identity copy of
//!   the first 3 bytes per pixel).
//! - `with_rgba` — identity row copy (input == output layout).
//! - `with_luma` — drop alpha into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — drop alpha into `rgb_scratch`, then
//!   `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::RgbaFrame, sealed::Sealed};

/// Zero‑sized marker for the packed **RGBA** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Rgba;

impl Sealed for Rgba {}
impl SourceFormat for Rgba {}

/// One output row of an [`Rgba`] source — `width * 4` packed
/// `R, G, B, A` bytes.
#[derive(Debug, Clone, Copy)]
pub struct RgbaRow<'a> {
  rgba: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> RgbaRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(rgba: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      rgba,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `R, G, B, A, R, G, B, A, …` row — `4 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn rgba(&self) -> &'a [u8] {
    self.rgba
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

/// Sinks that consume [`RgbaRow`].
pub trait RgbaSink: for<'a> PixelSink<Input<'a> = RgbaRow<'a>> {}

/// Walks an [`RgbaFrame`] row by row into the sink.
pub fn rgba_to<S: RgbaSink>(
  src: &RgbaFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.rgba();

  for row in 0..h {
    let start = row * stride;
    let rgba = &plane[start..start + row_bytes];
    sink.process(RgbaRow::new(rgba, row, matrix, full_range))?;
  }
  Ok(())
}
