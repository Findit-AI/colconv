//! Packed **X2BGR10** source (`AV_PIX_FMT_X2BGR10LE`) — 10 bits per
//! channel, 32-bit little-endian word with `(MSB) 2X | 10B | 10G |
//! 10R (LSB)`. Channel positions reversed relative to
//! [`super::X2Rgb10`].
//!
//! Outputs (Ship 9e):
//! - `with_rgb` — `x2bgr10_to_rgb_row` (extract the 10-bit channels
//!   from the swapped positions, down-shift to 8 bits, output
//!   `R, G, B`).
//! - `with_rgba` — `x2bgr10_to_rgba_row` (same extraction + force
//!   alpha to `0xFF`).
//! - `with_rgb_u16` — `x2bgr10_to_rgb_u16_row` (native 10-bit
//!   precision, low-bit aligned).
//! - `with_luma` / `with_hsv` — same scratch path as `X2Rgb10`,
//!   reusing the existing `rgb_to_luma_row` / `rgb_to_hsv_row`
//!   kernels.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::X2Bgr10Frame, sealed::Sealed};

/// Zero‑sized marker for the packed **X2BGR10** (LE) source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct X2Bgr10;

impl Sealed for X2Bgr10 {}
impl SourceFormat for X2Bgr10 {}

/// One output row of an [`X2Bgr10`] source — `width * 4` bytes
/// laid out as `width` little-endian `u32` pixels with packing
/// `(MSB) 2X | 10B | 10G | 10R (LSB)`.
#[derive(Debug, Clone, Copy)]
pub struct X2Bgr10Row<'a> {
  x2bgr10: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> X2Bgr10Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(x2bgr10: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      x2bgr10,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed X2BGR10 row bytes — `4 * width` bytes (width LE u32
  /// words).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn x2bgr10(&self) -> &'a [u8] {
    self.x2bgr10
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

/// Sinks that consume [`X2Bgr10Row`].
pub trait X2Bgr10Sink: for<'a> PixelSink<Input<'a> = X2Bgr10Row<'a>> {}

/// Walks an [`X2Bgr10Frame`] row by row into the sink.
pub fn x2bgr10_to<S: X2Bgr10Sink>(
  src: &X2Bgr10Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.x2bgr10();

  for row in 0..h {
    let start = row * stride;
    let x2bgr10 = &plane[start..start + row_bytes];
    sink.process(X2Bgr10Row::new(x2bgr10, row, matrix, full_range))?;
  }
  Ok(())
}
