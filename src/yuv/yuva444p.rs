//! YUVA 4:4:4 planar (`AV_PIX_FMT_YUVA444P`) — 8 bit per sample.
//!
//! Storage mirrors [`super::Yuv444p`] (Y / U / V each full-resolution
//! `u8`) plus a fourth full-resolution alpha plane (1:1 with Y).
//!
//! Per-row dispatcher hands the alpha source straight through to the
//! `yuv_444_to_rgba_with_alpha_src_row` SIMD/scalar paths — same shape
//! as the 4:2:0 sibling [`super::Yuva420p`]. Per-arch SIMD coverage is
//! shipped together with the format wiring (Ship 8b‑6).

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuva444pFrame, sealed::Sealed};

/// Zero‑sized marker for the YUVA 4:4:4 **8‑bit** source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuva444p;

impl Sealed for Yuva444p {}
impl SourceFormat for Yuva444p {}

/// One output row of a [`Yuva444p`] source.
#[derive(Debug, Clone, Copy)]
pub struct Yuva444pRow<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  a: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuva444pRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    a: &'a [u8],
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
  /// Full‑width Y (luma) row — `width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u8] {
    self.y
  }
  /// Full‑width U (Cb) row — `width` bytes (1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u(&self) -> &'a [u8] {
    self.u
  }
  /// Full‑width V (Cr) row — `width` bytes (1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v(&self) -> &'a [u8] {
    self.v
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

/// Sinks that consume [`Yuva444pRow`].
pub trait Yuva444pSink: for<'a> PixelSink<Input<'a> = Yuva444pRow<'a>> {}

/// Walks a [`Yuva444pFrame`] row by row into the sink.
pub fn yuva444p_to<S: Yuva444pSink>(
  src: &Yuva444pFrame<'_>,
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
    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let a_start = row * a_stride;
    let y = &y_plane[y_start..y_start + w];
    let u = &u_plane[u_start..u_start + w];
    let v = &v_plane[v_start..v_start + w];
    let a = &a_plane[a_start..a_start + w];

    sink.process(Yuva444pRow::new(y, u, v, a, row, matrix, full_range))?;
  }
  Ok(())
}
