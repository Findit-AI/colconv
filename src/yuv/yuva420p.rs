//! YUVA 4:2:0 planar (`AV_PIX_FMT_YUVA420P`).
//!
//! Storage mirrors [`super::Yuv420p`] — three planes for Y / U / V at
//! the standard 4:2:0 layout (Y full-size, U / V half-width × half-
//! height) — plus a fourth full-resolution alpha plane (1:1 with Y;
//! only chroma is subsampled in 4:2:0).
//!
//! Tranche 8b‑2a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_420_to_rgba_with_alpha_src_row` scalar path. Per‑arch SIMD
//! wiring lands in 8b‑2b (`u8` RGBA).

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuva420pFrame, sealed::Sealed};

/// Zero‑sized marker for the YUVA 4:2:0 **8‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuva420p;

impl Sealed for Yuva420p {}
impl SourceFormat for Yuva420p {}

/// One output row of a [`Yuva420p`] source.
///
/// Y / U / V follow the 4:2:0 chroma-pair convention (two consecutive
/// Y rows share one U/V row); A is full-resolution (one alpha row per
/// Y row).
#[derive(Debug, Clone, Copy)]
pub struct Yuva420pRow<'a> {
  y: &'a [u8],
  u_half: &'a [u8],
  v_half: &'a [u8],
  a: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuva420pRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u8],
    u_half: &'a [u8],
    v_half: &'a [u8],
    a: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u_half,
      v_half,
      a,
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
  /// Half‑width U (Cb) row — `width / 2` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u8] {
    self.u_half
  }
  /// Half‑width V (Cr) row — `width / 2` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v_half(&self) -> &'a [u8] {
    self.v_half
  }
  /// Full‑width alpha row — `width` bytes (1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn a(&self) -> &'a [u8] {
    self.a
  }
  /// Row index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// YUV → RGB matrix.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }
  /// Full‑range flag.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume [`Yuva420pRow`].
pub trait Yuva420pSink: for<'a> PixelSink<Input<'a> = Yuva420pRow<'a>> {}

/// Walks a [`Yuva420pFrame`] row by row into the sink.
pub fn yuva420p_to<S: Yuva420pSink>(
  src: &Yuva420pFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let u_stride = src.u_stride() as usize;
  let v_stride = src.v_stride() as usize;
  let a_stride = src.a_stride() as usize;
  let chroma_width = w / 2;

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();
  let a_plane = src.a();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:2:0 chroma subsampling: two consecutive Y rows share one
    // chroma row.
    let chroma_row = row / 2;
    let u_start = chroma_row * u_stride;
    let v_start = chroma_row * v_stride;
    let u_half = &u_plane[u_start..u_start + chroma_width];
    let v_half = &v_plane[v_start..v_start + chroma_width];

    // Alpha is full-resolution (1:1 with Y).
    let a_start = row * a_stride;
    let a = &a_plane[a_start..a_start + w];

    sink.process(Yuva420pRow::new(
      y, u_half, v_half, a, row, matrix, full_range,
    ))?;
  }
  Ok(())
}
