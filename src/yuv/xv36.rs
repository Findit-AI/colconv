//! Packed YUV 4:4:4 12-bit `XV36` source — high-bit-depth packed
//! capture format (FFmpeg `AV_PIX_FMT_XV36LE`). Each pixel is a u16
//! quadruple `U(16) ‖ Y(16) ‖ V(16) ‖ A(16)` with each channel using
//! the high 12 bits (low 4 bits zero, MSB-aligned at 12-bit). The
//! `X` prefix means the A slot is padding — read but discarded;
//! RGBA outputs force α = max. See [`crate::frame::Xv36Frame`] for
//! layout details.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=12, downshifted to u8; RGBA α = `0xFF` (XV36 has no alpha
//!   channel — A slot is padding).
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   12-bit depth, low-bit-packed in `u16` (high 4 bits zero); RGBA
//!   α = `0x0FFF` (12-bit max).
//! - `with_luma` — extracts Y values from each XV36 quadruple and
//!   downshifts via `>> 8` (12-bit MSB-aligned → u8 — equivalent to
//!   `>> 4` to drop padding then `>> 4` to bring 12-bit to 8-bit).
//! - `with_luma_u16` — extracts the 12-bit Y values via `>> 4`
//!   (drops padding) into u16 (low-bit-packed at 12-bit).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Xv36Frame, sealed::Sealed};

/// Zero-sized marker for the packed **XV36** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Xv36;

impl Sealed for Xv36 {}
impl SourceFormat for Xv36 {}

/// One row of an [`Xv36`] source — `width × 4` u16 elements (4
/// channels per pixel: U, Y, V, A; the A slot is padding).
///
/// Each u16 channel holds a 12-bit MSB-aligned sample with the low 4
/// bits zero. Channel layout per pixel:
///
/// | u16 slot | Field | Active bits           |
/// |----------|-------|-----------------------|
/// | 0        | U     | bits\[15:4\] (12-bit) |
/// | 1        | Y     | bits\[15:4\] (12-bit) |
/// | 2        | V     | bits\[15:4\] (12-bit) |
/// | 3        | A     | bits\[15:4\] (padding)|
///
/// Full range: `[0, 4095]` (12-bit). Limited range Y: `[256, 3760]`,
/// limited range chroma: `[256, 3840]`.
#[derive(Debug, Clone, Copy)]
pub struct Xv36Row<'a> {
  packed: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Xv36Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(packed: &'a [u16], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      packed,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed XV36 row — `width × 4` u16 elements (4 channels per
  /// pixel: U, Y, V, A; the A slot is padding).
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
  /// `true` iff Y ∈ `[0, 4095]` full range (12-bit). Limited range
  /// is Y `[256, 3760]`, chroma `[256, 3840]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Xv36Row`].
pub trait Xv36Sink: for<'a> PixelSink<Input<'a> = Xv36Row<'a>> {}

/// Walks an [`Xv36Frame`] row by row into the sink.
pub fn xv36_to<S: Xv36Sink>(
  src: &Xv36Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_elems = (src.width() as usize) * 4;
  let plane = src.packed();

  for row in 0..h {
    let start = row * stride;
    let packed = &plane[start..start + row_elems];
    sink.process(Xv36Row::new(packed, row, matrix, full_range))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Xv36Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Xv36Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Xv36Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Xv36Sink for CountingSink {}

  #[test]
  fn xv36_walker_visits_every_row_once() {
    let buf = std::vec![0u16; 4 * 4 * 4]; // 4 px × 4 channels × 4 rows = 64 u16 elements
    let frame = Xv36Frame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    xv36_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 16); // width × 4 u16 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
