//! Packed YUV 4:2:2 source (`AV_PIX_FMT_YVYU422`). One plane, byte
//! order `Y0, V0, Y1, U0` per 2-pixel block — same Y positions
//! as YUYV422 but with V/U swapped (V precedes U).
//!
//! Common on Android camera HAL outputs and a small handful of
//! older capture devices.
//!
//! Reuses the same const-generic packed-YUV-422 → RGB kernel
//! template as [`super::Yuyv422`] / [`super::Uyvy422`]; the only
//! difference from YUYV is the UV byte order, selected via the
//! `SWAP_UV` const generic at compile time.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yvyu422Frame, sealed::Sealed};

/// Zero‑sized marker for the packed **YVYU422** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yvyu422;

impl Sealed for Yvyu422 {}
impl SourceFormat for Yvyu422 {}

/// One output row of a [`Yvyu422`] source — `2 * width` packed
/// `Y0, V0, Y1, U0, …` bytes.
#[derive(Debug, Clone, Copy)]
pub struct Yvyu422Row<'a> {
  yvyu: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yvyu422Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(yvyu: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      yvyu,
      row,
      matrix,
      full_range,
    }
  }
  /// Packed `Y0, V0, Y1, U0, …` row — `2 * width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn yvyu(&self) -> &'a [u8] {
    self.yvyu
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

/// Sinks that consume [`Yvyu422Row`].
pub trait Yvyu422Sink: for<'a> PixelSink<Input<'a> = Yvyu422Row<'a>> {}

/// Walks a [`Yvyu422Frame`] row by row into the sink.
pub fn yvyu422_to<S: Yvyu422Sink>(
  src: &Yvyu422Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_bytes = w * 2;
  let plane = src.yvyu();

  for row in 0..h {
    let start = row * stride;
    let yvyu = &plane[start..start + row_bytes];
    sink.process(Yvyu422Row::new(yvyu, row, matrix, full_range))?;
  }
  Ok(())
}
