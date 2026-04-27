//! YUVA 4:4:4 planar 10‑bit (`AV_PIX_FMT_YUVA444P10LE`).
//!
//! Full‑resolution chroma + an alpha plane, 1:1 with Y. Mirrors
//! [`super::Yuv444p10`] but additionally carries a per‑row alpha slice
//! (also `width` `u16` samples, low‑bit‑packed at 10 bits).
//!
//! Tranche 8b‑1a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_444p_n_to_rgba*_with_alpha_src_row::<10>` scalar path. Per‑arch
//! SIMD wiring lands in 8b‑1b (`u8` RGBA) and 8b‑1c (`u16` RGBA).

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuva444p10Frame, sealed::Sealed};

/// Zero‑sized marker for the YUVA 4:4:4 **10‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuva444p10;

impl Sealed for Yuva444p10 {}
impl SourceFormat for Yuva444p10 {}

/// One output row of a [`Yuva444p10`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuva444p10Row<'a> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  a: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuva444p10Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u,
      v,
      a,
      row,
      matrix,
      full_range,
    }
  }
  /// Full‑width Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Full‑width U row — `width` samples, 1:1 with Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u(&self) -> &'a [u16] {
    self.u
  }
  /// Full‑width V row — `width` samples, 1:1 with Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v(&self) -> &'a [u16] {
    self.v
  }
  /// Full‑width alpha row — `width` `u16` samples, low‑bit‑packed at
  /// 10 bits. 1:1 with Y / U / V.
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

/// Sinks that consume [`Yuva444p10Row`].
pub trait Yuva444p10Sink: for<'a> PixelSink<Input<'a> = Yuva444p10Row<'a>> {}

/// Walks a [`Yuva444p10Frame`] row by row into the sink.
pub fn yuva444p10_to<S: Yuva444p10Sink>(
  src: &Yuva444p10Frame<'_>,
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

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();
  let a_plane = src.a();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];
    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let a_start = row * a_stride;
    let u = &u_plane[u_start..u_start + w];
    let v = &v_plane[v_start..v_start + w];
    let a = &a_plane[a_start..a_start + w];

    sink.process(Yuva444p10Row::new(y, u, v, a, row, matrix, full_range))?;
  }
  Ok(())
}
