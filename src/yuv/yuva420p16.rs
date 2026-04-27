//! YUVA 4:2:0 planar 16‑bit (`AV_PIX_FMT_YUVA420P16LE`).
//!
//! Storage mirrors [`super::Yuv420p16`] — three planes for Y / U / V
//! at the standard 4:2:0 layout (Y full-size, U / V half-width × half-
//! height) — plus a fourth full-resolution alpha plane (1:1 with Y;
//! only chroma is subsampled in 4:2:0). Sample width is **`u16`**.
//! At 16 bits there is no upper-bit-zero slack; the full `u16` range
//! is active.
//!
//! Runs on the **parallel i64 kernel family** for the u16 RGBA path
//! (Q15 chroma sum overflows i32 at 16 bits); the u8 RGBA path stays
//! on the i32 pipeline (output-range scaling keeps `coeff × u_d`
//! inside i32). 9/10‑bit YUVA siblings use the Q15 i32 family for
//! both u8 and u16 outputs.
//!
//! Tranche 8b‑2a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_420p16_to_rgba*_with_alpha_src_row` scalar paths. Per‑arch
//! SIMD wiring lands in 8b‑2b (`u8` RGBA) and 8b‑2c (`u16` RGBA).

use crate::{
  ColorMatrix, PixelSink, SourceFormat,
  frame::{Yuva420p16Frame, Yuva420pFrame16},
  sealed::Sealed,
};

/// Zero‑sized marker for the YUVA 4:2:0 **16‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuva420p16;

impl Sealed for Yuva420p16 {}
impl SourceFormat for Yuva420p16 {}

/// One output row of a [`Yuva420p16`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuva420p16Row<'a> {
  y: &'a [u16],
  u_half: &'a [u16],
  v_half: &'a [u16],
  a: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuva420p16Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u16],
    u_half: &'a [u16],
    v_half: &'a [u16],
    a: &'a [u16],
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
  /// Full‑width Y row — `width` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Half‑width U row — `width / 2` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u16] {
    self.u_half
  }
  /// Half‑width V row — `width / 2` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v_half(&self) -> &'a [u16] {
    self.v_half
  }
  /// Full‑width alpha row — `width` `u16` samples (full range,
  /// 1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn a(&self) -> &'a [u16] {
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

/// Sinks that consume [`Yuva420p16Row`].
pub trait Yuva420p16Sink: for<'a> PixelSink<Input<'a> = Yuva420p16Row<'a>> {}

/// Walks a [`Yuva420p16Frame`] row by row into the sink.
pub fn yuva420p16_to<S: Yuva420p16Sink>(
  src: &Yuva420p16Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  yuva420p16_walker::<16, S>(src, full_range, matrix, sink)
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva420p16_walker<const BITS: u32, S: Yuva420p16Sink>(
  src: &Yuva420pFrame16<'_, BITS>,
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

    let chroma_row = row / 2;
    let u_start = chroma_row * u_stride;
    let v_start = chroma_row * v_stride;
    let u_half = &u_plane[u_start..u_start + chroma_width];
    let v_half = &v_plane[v_start..v_start + chroma_width];

    let a_start = row * a_stride;
    let a = &a_plane[a_start..a_start + w];

    sink.process(Yuva420p16Row::new(
      y, u_half, v_half, a, row, matrix, full_range,
    ))?;
  }
  Ok(())
}
