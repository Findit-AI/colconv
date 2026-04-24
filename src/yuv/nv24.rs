//! NV24 — semi‑planar 4:4:4 (`AV_PIX_FMT_NV24`).
//!
//! Layout: one full‑size Y plane + one interleaved UV plane at **full
//! width and full height**. Each UV row is `U0, V0, U1, V1, …` —
//! 2·width bytes of payload per Y row. One UV pair per Y pixel, no
//! chroma upsampling.
//!
//! Compared to [`super::Nv12`] / [`super::Nv16`]: same interleaved‑UV
//! structure, zero subsampling. Width has no parity constraint.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Nv24Frame, sealed::Sealed};

/// Zero‑sized marker for the NV24 source format. Used as the `F` type
/// parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nv24;

impl Sealed for Nv24 {}
impl SourceFormat for Nv24 {}

/// One output row of an NV24 source handed to an [`Nv24Sink`].
///
/// Accessors:
/// - [`y`](Self::y) — full‑width Y row (`width` bytes).
/// - [`uv`](Self::uv) — **interleaved, full‑width** UV row
///   (`2 * width` bytes = `width` U / V pairs). 1:1 with Y.
/// - [`row`](Self::row) — output row index (`0 ..= frame.height() - 1`).
/// - [`matrix`](Self::matrix), [`full_range`](Self::full_range) — carried
///   through from the kernel call so the Sink can use them when calling
///   row primitives.
#[derive(Debug, Clone, Copy)]
pub struct Nv24Row<'a> {
  y: &'a [u8],
  uv: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Nv24Row<'a> {
  /// Bundles one row of an NV24 source for an [`Nv24Sink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u8],
    uv: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      uv,
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

  /// Interleaved UV row — `2 * width` bytes laid out as
  /// `U0, V0, U1, V1, …, U_{w-1}, V_{w-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv(&self) -> &'a [u8] {
    self.uv
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

/// Sinks that consume NV24 rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to [`Nv24Row`].
pub trait Nv24Sink: for<'a> PixelSink<Input<'a> = Nv24Row<'a>> {}

/// Converts an NV24 frame by walking its rows and feeding each one to
/// the [`Nv24Sink`].
pub fn nv24_to<S: Nv24Sink>(
  src: &Nv24Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let uv_stride = src.uv_stride() as usize;
  // 4:4:4: UV payload is `2 * width` bytes per row (one pair per pixel).
  let uv_row_bytes = 2 * w;

  let y_plane = src.y();
  let uv_plane = src.uv();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:4:4: chroma row index matches the Y row (no subsampling).
    let uv_start = row * uv_stride;
    let uv = &uv_plane[uv_start..uv_start + uv_row_bytes];

    sink.process(Nv24Row::new(y, uv, row, matrix, full_range))?;
  }
  Ok(())
}
