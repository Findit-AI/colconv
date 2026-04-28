//! Packed XBGR source (`AV_PIX_FMT_0BGR`) — 8 bits per channel,
//! byte order `X, B, G, R`. Leading padding + reversed RGB order
//! relative to [`super::Xrgb`].
//!
//! Outputs (Ship 9d):
//! - `with_rgb` — `abgr_to_rgb_row` (drop leading byte + R↔B swap;
//!   identical to the [`Abgr`](super::Abgr) RGB path because both
//!   ignore byte 0).
//! - `with_rgba` — `xbgr_to_rgba_row` (drop padding + R↔B swap +
//!   force alpha to `0xFF`).
//! - `with_luma` — same swap+drop path into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::XbgrFrame, sealed::Sealed};

/// Zero‑sized marker for the packed **XBGR** (a.k.a. `0bgr`) source
/// format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Xbgr;

impl Sealed for Xbgr {}
impl SourceFormat for Xbgr {}

/// One output row of an [`Xbgr`] source — `width * 4` packed
/// `X, B, G, R` bytes.
#[derive(Debug, Clone, Copy)]
pub struct XbgrRow<'a> {
  xbgr: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> XbgrRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(xbgr: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      xbgr,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `X, B, G, R, X, B, G, R, …` row — `4 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn xbgr(&self) -> &'a [u8] {
    self.xbgr
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

/// Sinks that consume [`XbgrRow`].
pub trait XbgrSink: for<'a> PixelSink<Input<'a> = XbgrRow<'a>> {}

/// Walks an [`XbgrFrame`] row by row into the sink.
pub fn xbgr_to<S: XbgrSink>(
  src: &XbgrFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 4;
  let plane = src.xbgr();

  for row in 0..h {
    let start = row * stride;
    let xbgr = &plane[start..start + row_bytes];
    sink.process(XbgrRow::new(xbgr, row, matrix, full_range))?;
  }
  Ok(())
}
