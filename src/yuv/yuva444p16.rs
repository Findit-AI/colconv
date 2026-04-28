//! YUVA 4:4:4 planar 16‑bit (`AV_PIX_FMT_YUVA444P16LE`).
//!
//! Storage mirrors [`super::Yuv444p16`] (Y / U / V each full-resolution,
//! `u16` samples — at 16 bits there is no upper-bit-zero slack; the
//! full `u16` range is active) plus a fourth full-resolution alpha
//! plane (1:1 with Y).
//!
//! For the native-depth `u16` output path, this uses the **dedicated
//! i64 4:4:4 kernel family** because the Q15 chroma sum overflows
//! i32 at 16 bits. The `u8` output path stays on the scaled Q15 i32
//! route (output-target scaling keeps `coeff × u_d` inside i32).
//! Either way it sits separate from the BITS-generic Q15 i32 template
//! that covers `BITS ∈ {9, 10, 12, 14}`. Mirrors the 4:2:0 sibling
//! [`super::Yuva420p16`].
//!
//! Tranche 8b‑5a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_444p16_to_rgba*_with_alpha_src_row` scalar paths. Per‑arch
//! SIMD wiring lands in 8b‑5b (`u8` RGBA) and 8b‑5c (`u16` RGBA).

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuva444p16Frame, sealed::Sealed};

/// Zero‑sized marker for the YUVA 4:4:4 **16‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuva444p16;

impl Sealed for Yuva444p16 {}
impl SourceFormat for Yuva444p16 {}

/// One output row of a [`Yuva444p16`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuva444p16Row<'a> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  a: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuva444p16Row<'a> {
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
  /// Full‑width alpha row — `width` `u16` samples (full range, 1:1
  /// with Y / U / V).
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

/// Sinks that consume [`Yuva444p16Row`].
pub trait Yuva444p16Sink: for<'a> PixelSink<Input<'a> = Yuva444p16Row<'a>> {}

/// Walks a [`Yuva444p16Frame`] row by row into the sink.
pub fn yuva444p16_to<S: Yuva444p16Sink>(
  src: &Yuva444p16Frame<'_>,
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

    sink.process(Yuva444p16Row::new(y, u, v, a, row, matrix, full_range))?;
  }
  Ok(())
}
