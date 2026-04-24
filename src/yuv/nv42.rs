//! NV42 — semi‑planar 4:4:4 (`AV_PIX_FMT_NV42`), VU‑ordered.
//!
//! Layout: one full‑size Y plane + one interleaved VU plane at full
//! width and full height. Each VU row is `V0, U0, V1, U1, …` — the
//! byte‑order twin of NV24's UV ordering. Shares per‑row kernel math
//! with [`super::Nv24`]; only the chroma‑byte parity differs (swapped
//! inside the SIMD/scalar kernel via a `SWAP_UV` const generic).

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Nv42Frame, sealed::Sealed};

/// Zero‑sized marker for the NV42 source format. Used as the `F` type
/// parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nv42;

impl Sealed for Nv42 {}
impl SourceFormat for Nv42 {}

/// One output row of an NV42 source handed to an [`Nv42Sink`].
#[derive(Debug, Clone, Copy)]
pub struct Nv42Row<'a> {
  y: &'a [u8],
  vu: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Nv42Row<'a> {
  /// Bundles one row of an NV42 source for an [`Nv42Sink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u8],
    vu: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      vu,
      row,
      matrix,
      full_range,
    }
  }

  /// Full‑width Y (luma) row — `width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u8] {
    self.y
  }

  /// Interleaved VU row — `2 * width` bytes laid out as
  /// `V0, U0, V1, U1, …, V_{w-1}, U_{w-1}` (byte order swapped
  /// relative to [`super::Nv24Row`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn vu(&self) -> &'a [u8] {
    self.vu
  }

  /// Output row index within the frame.
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

/// Sinks that consume NV42 rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to [`Nv42Row`].
pub trait Nv42Sink: for<'a> PixelSink<Input<'a> = Nv42Row<'a>> {}

/// Converts an NV42 frame by walking its rows and feeding each one to
/// the [`Nv42Sink`].
pub fn nv42_to<S: Nv42Sink>(
  src: &Nv42Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let vu_stride = src.vu_stride() as usize;
  let vu_row_bytes = 2 * w;

  let y_plane = src.y();
  let vu_plane = src.vu();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    let vu_start = row * vu_stride;
    let vu = &vu_plane[vu_start..vu_start + vu_row_bytes];

    sink.process(Nv42Row::new(y, vu, row, matrix, full_range))?;
  }
  Ok(())
}
