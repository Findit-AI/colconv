//! Packed YUV 4:4:4 8-bit `VUYA` source — display-compose capture
//! format (FFmpeg `AV_PIX_FMT_VUYA`). Each pixel is a u8 quadruple
//! `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)` where the A byte is **real alpha**
//! (source transparency). See [`crate::frame::VuyaFrame`] for layout
//! details.
//!
//! Outputs are produced via:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; source α
//!   passed through from byte 3 of each pixel.
//! - `with_luma` — extracts the Y byte (byte 2 of each pixel)
//!   directly.
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.
//!
//! VUYA has no u16 output paths — it is an 8-bit source.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::VuyaFrame, sealed::Sealed};

/// Zero-sized marker for the packed **VUYA** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Vuya;

impl Sealed for Vuya {}
impl SourceFormat for Vuya {}

/// One row of a [`Vuya`] source — `width × 4` bytes (4 channels per
/// pixel: V, U, Y, A; the A byte is real source alpha).
///
/// Byte layout per pixel:
///
/// | Byte offset | Field |
/// |-------------|-------|
/// | 0           | V     |
/// | 1           | U     |
/// | 2           | Y     |
/// | 3           | A     |
///
/// The walker does not interpret the bytes — it passes the raw packed
/// slice to the sink. Byte-level channel extraction happens in the
/// row-kernel layer.
#[derive(Debug, Clone, Copy)]
pub struct VuyaRow<'a> {
  packed: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> VuyaRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(packed: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      packed,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed VUYA row — `width × 4` bytes (4 channels per pixel:
  /// V, U, Y, A).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn packed(&self) -> &'a [u8] {
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
  /// `true` iff Y ∈ `[0, 255]` full range (8-bit). Limited range is
  /// Y `[16, 235]`, chroma `[16, 240]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`VuyaRow`].
pub trait VuyaSink: for<'a> PixelSink<Input<'a> = VuyaRow<'a>> {}

/// Walks a [`VuyaFrame`] row by row into the sink.
pub fn vuya_to<S: VuyaSink>(
  src: &VuyaFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = (src.width() as usize) * 4;
  let plane = src.packed();

  for row in 0..h {
    let start = row * stride;
    let packed = &plane[start..start + row_bytes];
    sink.process(VuyaRow::new(packed, row, matrix, full_range))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::VuyaFrame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_packed_len: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = VuyaRow<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: VuyaRow<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_packed_len = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl VuyaSink for CountingSink {}

  #[test]
  fn vuya_walker_visits_every_row_once() {
    // 4 px × 4 channels × 4 rows = 64 bytes
    let buf = std::vec![0u8; 4 * 4 * 4];
    let frame = VuyaFrame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_packed_len: 0,
      last_row_idx: 0,
    };
    vuya_to(&frame, false, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_packed_len, 16); // width × 4 bytes per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
