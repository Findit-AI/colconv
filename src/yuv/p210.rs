//! P210 — semi‑planar 4:2:2, 10‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P210LE`).
//!
//! 4:2:2 twin of [`super::P010`]: same Y + interleaved-UV plane shape
//! and same high-bit-packed `u16` convention (10 active bits in the
//! high 10 positions, low 6 zero), but chroma is **full-height** —
//! one chroma row per Y row instead of one per two. NVDEC / CUDA HDR
//! 4:2:2 download target and some QSV configurations.
//!
//! Per-row kernel reuses the 4:2:0 `p_n_to_rgb_*<10>` family verbatim
//! (the per-row UV layout is identical to P010 — half-width
//! interleaved); only the walker reads chroma row `r` instead of
//! `r / 2`.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::P210Frame, sealed::Sealed};

/// Zero‑sized marker for the P210 source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct P210;

impl Sealed for P210 {}
impl SourceFormat for P210 {}

/// One output row of a P210 source handed to a [`P210Sink`].
#[derive(Debug, Clone, Copy)]
pub struct P210Row<'a> {
  y: &'a [u16],
  uv_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> P210Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u16],
    uv_half: &'a [u16],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      uv_half,
      row,
      matrix,
      full_range,
    }
  }
  /// Full‑width Y (luma) row — `width` `u16` samples, high‑bit‑packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Half-width interleaved UV row — `width` `u16` elements laid out
  /// as `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv_half(&self) -> &'a [u16] {
    self.uv_half
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

/// Sinks that consume [`P210Row`].
pub trait P210Sink: for<'a> PixelSink<Input<'a> = P210Row<'a>> {}

/// Walks a [`P210Frame`] row by row into the sink. Each Y row has its
/// own corresponding UV row (4:2:2 — full-height chroma).
pub fn p210_to<S: P210Sink>(
  src: &P210Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let uv_stride = src.uv_stride() as usize;
  let uv_row_elems = w; // half-width × 2 elements per pair = `width` u16 elements

  let y_plane = src.y();
  let uv_plane = src.uv();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:2:2: one chroma row per Y row.
    let uv_start = row * uv_stride;
    let uv_half = &uv_plane[uv_start..uv_start + uv_row_elems];

    sink.process(P210Row::new(y, uv_half, row, matrix, full_range))?;
  }
  Ok(())
}
