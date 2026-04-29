//! Packed **X2RGB10** source (`AV_PIX_FMT_X2RGB10LE`) — 10 bits per
//! channel, 32-bit little-endian word with `(MSB) 2X | 10R | 10G |
//! 10B (LSB)`. The 2 leading bits are **ignored padding**.
//!
//! Outputs (Ship 9e):
//! - `with_rgb` — `x2rgb10_to_rgb_row` (down-shift each 10-bit
//!   channel to 8 bits and pack as `R, G, B`).
//! - `with_rgba` — `x2rgb10_to_rgba_row` (same down-shift + force
//!   alpha to `0xFF`).
//! - `with_rgb_u16` — `x2rgb10_to_rgb_u16_row` (native 10-bit
//!   precision, low-bit aligned in `u16`; max value `1023`).
//! - `with_luma` — drop padding into the u8 RGB scratch, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::X2Rgb10Frame, sealed::Sealed};

/// Zero‑sized marker for the packed **X2RGB10** (LE) source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct X2Rgb10;

impl Sealed for X2Rgb10 {}
impl SourceFormat for X2Rgb10 {}

/// One output row of an [`X2Rgb10`] source — `width * 4` bytes
/// laid out as `width` little-endian `u32` pixels with packing
/// `(MSB) 2X | 10R | 10G | 10B (LSB)`.
#[derive(Debug, Clone, Copy)]
pub struct X2Rgb10Row<'a> {
  x2rgb10: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> X2Rgb10Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(x2rgb10: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      x2rgb10,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed X2RGB10 row bytes — `4 * width` bytes (width LE u32
  /// words).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn x2rgb10(&self) -> &'a [u8] {
    self.x2rgb10
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

/// Sinks that consume [`X2Rgb10Row`].
pub trait X2Rgb10Sink: for<'a> PixelSink<Input<'a> = X2Rgb10Row<'a>> {}

/// Walks an [`X2Rgb10Frame`] row by row into the sink.
pub fn x2rgb10_to<S: X2Rgb10Sink>(
  src: &X2Rgb10Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.x2rgb10();

  for row in 0..h {
    let start = row * stride;
    let x2rgb10 = &plane[start..start + row_bytes];
    sink.process(X2Rgb10Row::new(x2rgb10, row, matrix, full_range))?;
  }
  Ok(())
}
