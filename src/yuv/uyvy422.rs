//! Packed YUV 4:2:2 source (`AV_PIX_FMT_UYVY422`). One plane, byte
//! order `U0, Y0, V0, Y1` per 2-pixel block — Y in odd byte
//! positions, U/V in even positions.
//!
//! De-facto SDI capture format on Apple QuickTime / VideoToolbox
//! 8-bit paths, also widely emitted by professional capture cards
//! in 8-bit mode.
//!
//! Reuses the same const-generic packed-YUV-422 → RGB kernel
//! template as [`super::Yuyv422`] / [`super::Yvyu422`]; the only
//! difference is Y / UV byte positions, selected at compile time.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Uyvy422Frame, sealed::Sealed};

/// Zero‑sized marker for the packed **UYVY422** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Uyvy422;

impl Sealed for Uyvy422 {}
impl SourceFormat for Uyvy422 {}

/// One output row of a [`Uyvy422`] source — `2 * width` packed
/// `U0, Y0, V0, Y1, …` bytes.
#[derive(Debug, Clone, Copy)]
pub struct Uyvy422Row<'a> {
  uyvy: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Uyvy422Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(uyvy: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      uyvy,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `U0, Y0, V0, Y1, …` row — `2 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uyvy(&self) -> &'a [u8] {
    self.uyvy
  }
  /// Row index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// YUV → RGB matrix carried through from the kernel call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }
  /// `true` iff Y ∈ `[0, 255]` (full range); `false` for limited.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Uyvy422Row`].
pub trait Uyvy422Sink: for<'a> PixelSink<Input<'a> = Uyvy422Row<'a>> {}

/// Walks a [`Uyvy422Frame`] row by row into the sink.
pub fn uyvy422_to<S: Uyvy422Sink>(
  src: &Uyvy422Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 2;
  let plane = src.uyvy();

  for row in 0..h {
    let start = row * stride;
    let uyvy = &plane[start..start + row_bytes];
    sink.process(Uyvy422Row::new(uyvy, row, matrix, full_range))?;
  }
  Ok(())
}
