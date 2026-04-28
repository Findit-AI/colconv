//! Packed BGRX source (`AV_PIX_FMT_BGR0`) — 8 bits per channel,
//! byte order `B, G, R, X`. Trailing padding + reversed RGB order
//! relative to [`super::Rgbx`].
//!
//! Outputs (Ship 9d):
//! - `with_rgb` — `bgra_to_rgb_row` (drop trailing byte + R↔B swap;
//!   identical to the [`Bgra`](super::Bgra) RGB path because both
//!   ignore byte 3).
//! - `with_rgba` — `bgrx_to_rgba_row` (R↔B swap + force alpha to
//!   `0xFF`).
//! - `with_luma` — same swap+drop path into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::BgrxFrame, sealed::Sealed};

/// Zero‑sized marker for the packed **BGRX** (a.k.a. `bgr0`) source
/// format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bgrx;

impl Sealed for Bgrx {}
impl SourceFormat for Bgrx {}

/// One output row of a [`Bgrx`] source — `width * 4` packed
/// `B, G, R, X` bytes.
#[derive(Debug, Clone, Copy)]
pub struct BgrxRow<'a> {
  bgrx: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> BgrxRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(bgrx: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      bgrx,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `B, G, R, X, B, G, R, X, …` row — `4 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn bgrx(&self) -> &'a [u8] {
    self.bgrx
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

/// Sinks that consume [`BgrxRow`].
pub trait BgrxSink: for<'a> PixelSink<Input<'a> = BgrxRow<'a>> {}

/// Walks a [`BgrxFrame`] row by row into the sink.
pub fn bgrx_to<S: BgrxSink>(
  src: &BgrxFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.bgrx();

  for row in 0..h {
    let start = row * stride;
    let bgrx = &plane[start..start + row_bytes];
    sink.process(BgrxRow::new(bgrx, row, matrix, full_range))?;
  }
  Ok(())
}
