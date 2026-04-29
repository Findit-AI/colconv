//! Packed YUV 4:2:2 10-bit `v210` source — pro-broadcast 10-bit SDI
//! capture format. Each 16-byte word holds 6 pixels (12 × 10-bit
//! samples). See [`crate::frame::V210Frame`] for layout details.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=10, downshifted to u8.
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16`.
//! - `with_luma` — extracts the 6 Y values from each v210 word and
//!   downshifts via `>> 2`.
//! - `with_luma_u16` — extracts the 10-bit Y values into u16
//!   (low-bit-packed).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::V210Frame, sealed::Sealed};

/// Zero-sized marker for the packed **v210** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct V210;

impl Sealed for V210 {}
impl SourceFormat for V210 {}

/// One row of a [`V210`] source — `(width / 6) * 16` packed bytes.
#[derive(Debug, Clone, Copy)]
pub struct V210Row<'a> {
  v210: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> V210Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(v210: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      v210,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed v210 row — `(width / 6) * 16` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v210(&self) -> &'a [u8] {
    self.v210
  }
  /// Row index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// YUV → RGB matrix carried through.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }
  /// `true` iff Y ∈ `[0, 1023]` full range (10-bit). Limited range
  /// is `[64, 940]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`V210Row`].
pub trait V210Sink: for<'a> PixelSink<Input<'a> = V210Row<'a>> {}

/// Walks a [`V210Frame`] row by row into the sink.
pub fn v210_to<S: V210Sink>(
  src: &V210Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = (src.width() as usize / 6) * 16;
  let plane = src.v210();

  for row in 0..h {
    let start = row * stride;
    let v210 = &plane[start..start + row_bytes];
    sink.process(V210Row::new(v210, row, matrix, full_range))?;
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::V210Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = V210Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: V210Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.v210().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl V210Sink for CountingSink {}

  #[test]
  fn v210_walker_visits_every_row_once() {
    let buf = std::vec![0u8; 16 * 4];
    let frame = V210Frame::new(&buf, 6, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    v210_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 16);
    assert_eq!(sink.last_row_idx, 3);
  }
}
