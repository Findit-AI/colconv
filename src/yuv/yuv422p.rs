//! YUV 4:2:2 planar (`AV_PIX_FMT_YUV422P`, `yuvj422p`).
//!
//! Three planes: full-size Y + half-width, **full-height** U/V.
//! The per-row kernel is identical to [`super::Yuv420p`]'s — the
//! 4:2:0 → 4:2:2 difference is purely vertical: YUV420p reads chroma
//! row `r / 2`, YUV422p reads chroma row `r`. The sinker calls
//! [`crate::row::yuv_420_to_rgb_row`] directly.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv422pFrame, sealed::Sealed};

/// Zero-sized marker for the YUV 4:2:2 source format. Used as the
/// `F` type parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv422p;

impl Sealed for Yuv422p {}
impl SourceFormat for Yuv422p {}

/// One output row of a YUV 4:2:2 source handed to a [`Yuv422pSink`].
///
/// Accessors:
/// - [`y`](Self::y) — full-width Y row (`width` bytes).
/// - [`u_half`](Self::u_half), [`v_half`](Self::v_half) — half-width
///   (`width / 2` bytes) chroma rows. Unlike 4:2:0, **no two Y rows
///   share a chroma row** — the walker advances U/V every row.
/// - [`row`](Self::row), [`matrix`](Self::matrix),
///   [`full_range`](Self::full_range) — carried through from the
///   kernel call.
#[derive(Debug, Clone, Copy)]
pub struct Yuv422pRow<'a> {
  y: &'a [u8],
  u_half: &'a [u8],
  v_half: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv422pRow<'a> {
  /// Bundles one row of a 4:2:2 source for a [`Yuv422pSink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u8],
    u_half: &'a [u8],
    v_half: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u_half,
      v_half,
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

  /// Half-width U (Cb) row — `width / 2` bytes for *this* Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u8] {
    self.u_half
  }

  /// Half-width V (Cr) row — `width / 2` bytes for *this* Y row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v_half(&self) -> &'a [u8] {
    self.v_half
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

/// Sinks that consume YUV 4:2:2 rows.
pub trait Yuv422pSink: for<'a> PixelSink<Input<'a> = Yuv422pRow<'a>> {}

/// Converts a YUV 4:2:2 frame by walking its rows and feeding each
/// one to the [`Yuv422pSink`]. Chroma advances every row (vs 4:2:0's
/// `row / 2`).
pub fn yuv422p_to<S: Yuv422pSink>(
  src: &Yuv422pFrame<'_>,
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
  let chroma_width = w / 2;

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:2:2: chroma row index matches the Y row.
    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let u_half = &u_plane[u_start..u_start + chroma_width];
    let v_half = &v_plane[v_start..v_start + chroma_width];

    sink.process(Yuv422pRow::new(y, u_half, v_half, row, matrix, full_range))?;
  }
  Ok(())
}
