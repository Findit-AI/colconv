//! YUV 4:4:4 planar (`AV_PIX_FMT_YUV444P`, `yuvj444p`).
//!
//! Three planes, all full-size. One UV pair per Y pixel, no chroma
//! subsampling. Per-row kernel math is the same 4:4:4 arithmetic
//! used by [`super::Nv24`] / [`super::Nv42`] — one `u` sample and
//! one `v` sample per pixel — but U and V come from separate planes
//! instead of an interleaved UV / VU plane.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv444pFrame, sealed::Sealed};

/// Zero-sized marker for the YUV 4:4:4 source format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv444p;

impl Sealed for Yuv444p {}
impl SourceFormat for Yuv444p {}

/// One output row of a YUV 4:4:4 source handed to a [`Yuv444pSink`].
///
/// Accessors:
/// - [`y`](Self::y) — full-width Y row (`width` bytes).
/// - [`u`](Self::u), [`v`](Self::v) — full-width chroma rows
///   (`width` bytes each). 1:1 with Y — no subsampling.
#[derive(Debug, Clone, Copy)]
pub struct Yuv444pRow<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv444pRow<'a> {
  /// Bundles one row of a 4:4:4 source for a [`Yuv444pSink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u,
      v,
      row,
      matrix,
      full_range,
    }
  }

  /// Full-width Y (luma) row — `width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u8] {
    self.y
  }

  /// Full-width U (Cb) row — `width` bytes. 1:1 with Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u(&self) -> &'a [u8] {
    self.u
  }

  /// Full-width V (Cr) row — `width` bytes. 1:1 with Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v(&self) -> &'a [u8] {
    self.v
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

/// Sinks that consume YUV 4:4:4 rows.
pub trait Yuv444pSink: for<'a> PixelSink<Input<'a> = Yuv444pRow<'a>> {}

/// Converts a YUV 4:4:4 frame by walking its rows and feeding each
/// one to the [`Yuv444pSink`].
pub fn yuv444p_to<S: Yuv444pSink>(
  src: &Yuv444pFrame<'_>,
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

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:4:4: U and V are full-width, 1:1 with Y.
    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let u = &u_plane[u_start..u_start + w];
    let v = &v_plane[v_start..v_start + w];

    sink.process(Yuv444pRow::new(y, u, v, row, matrix, full_range))?;
  }
  Ok(())
}
