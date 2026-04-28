//! Packed XRGB source (`AV_PIX_FMT_0RGB`) — 8 bits per channel,
//! byte order `X, R, G, B`. The 1st byte is **ignored padding**
//! (not real alpha — see [`super::Argb`] for the alpha-bearing
//! analogue).
//!
//! Outputs (Ship 9d):
//! - `with_rgb` — `argb_to_rgb_row` (drop leading byte; identical to
//!   the [`Argb`](super::Argb) RGB path because both ignore byte 0).
//! - `with_rgba` — `xrgb_to_rgba_row` (drop padding + force alpha to
//!   `0xFF`).
//! - `with_luma` — drop padding into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::XrgbFrame, sealed::Sealed};

/// Zero‑sized marker for the packed **XRGB** (a.k.a. `0rgb`) source
/// format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Xrgb;

impl Sealed for Xrgb {}
impl SourceFormat for Xrgb {}

/// One output row of an [`Xrgb`] source — `width * 4` packed
/// `X, R, G, B` bytes.
#[derive(Debug, Clone, Copy)]
pub struct XrgbRow<'a> {
  xrgb: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> XrgbRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(xrgb: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      xrgb,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `X, R, G, B, X, R, G, B, …` row — `4 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn xrgb(&self) -> &'a [u8] {
    self.xrgb
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

/// Sinks that consume [`XrgbRow`].
pub trait XrgbSink: for<'a> PixelSink<Input<'a> = XrgbRow<'a>> {}

/// Walks an [`XrgbFrame`] row by row into the sink.
pub fn xrgb_to<S: XrgbSink>(
  src: &XrgbFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.xrgb();

  for row in 0..h {
    let start = row * stride;
    let xrgb = &plane[start..start + row_bytes];
    sink.process(XrgbRow::new(xrgb, row, matrix, full_range))?;
  }
  Ok(())
}
