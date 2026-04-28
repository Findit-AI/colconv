//! Packed RGBX source (`AV_PIX_FMT_RGB0`) — 8 bits per channel,
//! byte order `R, G, B, X`. The 4th byte is **ignored padding**
//! (not real alpha — see [`super::Rgba`] for the alpha-bearing
//! analogue).
//!
//! Outputs (Ship 9d):
//! - `with_rgb` — `rgba_to_rgb_row` (drop trailing byte; identical to
//!   the [`Rgba`](super::Rgba) RGB path because both ignore byte 3).
//! - `with_rgba` — `rgbx_to_rgba_row` (memcpy first 3 bytes + force
//!   alpha to `0xFF`).
//! - `with_luma` — drop padding into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::RgbxFrame, sealed::Sealed};

/// Zero‑sized marker for the packed **RGBX** (a.k.a. `rgb0`) source
/// format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Rgbx;

impl Sealed for Rgbx {}
impl SourceFormat for Rgbx {}

/// One output row of an [`Rgbx`] source — `width * 4` packed
/// `R, G, B, X` bytes.
#[derive(Debug, Clone, Copy)]
pub struct RgbxRow<'a> {
  rgbx: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> RgbxRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(rgbx: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      rgbx,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `R, G, B, X, R, G, B, X, …` row — `4 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn rgbx(&self) -> &'a [u8] {
    self.rgbx
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

/// Sinks that consume [`RgbxRow`].
pub trait RgbxSink: for<'a> PixelSink<Input<'a> = RgbxRow<'a>> {}

/// Walks an [`RgbxFrame`] row by row into the sink.
pub fn rgbx_to<S: RgbxSink>(
  src: &RgbxFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.rgbx();

  for row in 0..h {
    let start = row * stride;
    let rgbx = &plane[start..start + row_bytes];
    sink.process(RgbxRow::new(rgbx, row, matrix, full_range))?;
  }
  Ok(())
}
