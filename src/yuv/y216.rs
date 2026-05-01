//! Packed YUV 4:2:2 16-bit `Y216` source — high-bit-depth packed
//! capture format. Each row is a sequence of YUYV-shaped u16
//! quadruples (`Y₀, U, Y₁, V`); all 16 bits per sample are active.
//! See [`crate::frame::Y216Frame`] for layout details.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=16, downshifted to u8.
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   16-bit depth, full-range u16.
//! - `with_luma` — extracts the Y values from each Y216 quadruple
//!   and downshifts via `>> 8` (16-bit → u8).
//! - `with_luma_u16` — extracts the 16-bit Y values into u16
//!   (direct memcpy of the Y values; full 16-bit fidelity).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Y216Frame, sealed::Sealed};

/// Zero-sized marker for the packed **Y216** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Y216;

impl Sealed for Y216 {}
impl SourceFormat for Y216 {}

/// One row of a [`Y216`] source — `width × 2` u16 elements
/// (`Y₀, U, Y₁, V` quadruples per 2-pixel block).
#[derive(Debug, Clone, Copy)]
pub struct Y216Row<'a> {
  packed: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Y216Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(packed: &'a [u16], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      packed,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed Y216 row — `width × 2` u16 elements (`Y₀, U, Y₁, V`,
  /// all 16 bits active per sample).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn packed(&self) -> &'a [u16] {
    self.packed
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
  /// `true` iff Y ∈ `[0, 65535]` full range (16-bit). Limited range
  /// is `[4096, 60160]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Y216Row`].
pub trait Y216Sink: for<'a> PixelSink<Input<'a> = Y216Row<'a>> {}

/// Walks a [`Y216Frame`] row by row into the sink.
pub fn y216_to<S: Y216Sink>(
  src: &Y216Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_elems = src.width() as usize * 2;
  let plane = src.packed();

  for row in 0..h {
    let start = row * stride;
    let packed = &plane[start..start + row_elems];
    sink.process(Y216Row::new(packed, row, matrix, full_range))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Y216Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Y216Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Y216Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Y216Sink for CountingSink {}

  #[test]
  fn y216_walker_visits_every_row_once() {
    let buf = std::vec![0u16; 8 * 4];
    let frame = Y216Frame::new(&buf, 4, 4, 8);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    y216_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 8);
    assert_eq!(sink.last_row_idx, 3);
  }
}
