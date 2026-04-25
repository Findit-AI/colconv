//! P410 — semi‑planar 4:4:4, 10‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P410LE`).
//!
//! 4:4:4 twin of [`super::P010`]: same high-bit-packed `u16`
//! convention (10 active bits in the high 10 positions), but chroma
//! is **full-width × full-height** (1:1 with Y, no subsampling).
//! Each chroma row holds `2 * width` `u16` elements (= `width`
//! interleaved `U, V` pairs). NVDEC / CUDA HDR 4:4:4 download target.
//!
//! Per-row kernel: a dedicated 4:4:4 high-bit-depth semi-planar
//! family `p_n_444_to_rgb_*<10>` (full-width interleaved UV, no
//! horizontal duplication step). Differs from the 4:2:0 / 4:2:2
//! `p_n_to_rgb_*<10>` family in the chroma layout only.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::P410Frame, sealed::Sealed};

/// Zero‑sized marker for the P410 source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct P410;

impl Sealed for P410 {}
impl SourceFormat for P410 {}

/// One output row of a P410 source handed to a [`P410Sink`].
#[derive(Debug, Clone, Copy)]
pub struct P410Row<'a> {
  y: &'a [u16],
  uv_full: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> P410Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u16],
    uv_full: &'a [u16],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      uv_full,
      row,
      matrix,
      full_range,
    }
  }
  /// Full‑width Y row — `width` `u16` samples, high‑bit‑packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Full-width interleaved UV row — `2 * width` `u16` elements laid
  /// out as `U0, V0, U1, V1, …, U_{w-1}, V_{w-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv_full(&self) -> &'a [u16] {
    self.uv_full
  }
  /// Output row index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// YUV → RGB matrix.
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

/// Sinks that consume [`P410Row`].
pub trait P410Sink: for<'a> PixelSink<Input<'a> = P410Row<'a>> {}

/// Walks a [`P410Frame`] row by row into the sink.
pub fn p410_to<S: P410Sink>(
  src: &P410Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let uv_stride = src.uv_stride() as usize;
  // 4:4:4 semi-planar: full-width × 2 elements per pair. The
  // PnFrame444 validator already rejects geometries where `2 * width`
  // overflows, so a plain multiplication is safe here.
  let uv_row_elems = 2 * w;

  let y_plane = src.y();
  let uv_plane = src.uv();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    let uv_start = row * uv_stride;
    let uv_full = &uv_plane[uv_start..uv_start + uv_row_elems];

    sink.process(P410Row::new(y, uv_full, row, matrix, full_range))?;
  }
  Ok(())
}
