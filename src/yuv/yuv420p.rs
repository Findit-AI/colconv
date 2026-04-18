//! YUV 4:2:0 planar (`AV_PIX_FMT_YUV420P`, `yuvj420p`, `yuv420p9/10/…`
//! once we parameterize depth).
//!
//! See the module docs in [`super`] for the Sink-based conversion
//! model. At 4:2:0 the kernel reads one chroma row per *two* Y rows;
//! both Y rows of a pair receive the same chroma row when the kernel
//! hands them to the Sink.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Yuv420pFrame, sealed::Sealed};

/// Zero-sized marker for the YUV 4:2:0 source format. Used as the
/// `F` type parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv420p;

impl Sealed for Yuv420p {}
impl SourceFormat for Yuv420p {}

/// One output row of a YUV 4:2:0 source handed to a [`Yuv420pSink`].
///
/// Accessors:
/// - [`y`](Self::y) — full-width Y row (`width` bytes).
/// - [`u_half`](Self::u_half), [`v_half`](Self::v_half) — **half-width**
///   (`width / 2` bytes) chroma samples as they appear in the source,
///   without upsampling. Sinks that need full-width chroma upsample
///   inline via the crate's fused row primitives (e.g. the MixedSinker
///   for YUV does nearest-neighbor upsample inside `yuv_420_to_rgb_row`).
/// - [`row`](Self::row) — output row index (`0 ..= frame.height() - 1`).
/// - [`matrix`](Self::matrix), [`full_range`](Self::full_range) — carried
///   through from the kernel call so the Sink can use them when calling
///   row primitives.
#[derive(Debug, Clone, Copy)]
pub struct Yuv420pRow<'a> {
  y: &'a [u8],
  u_half: &'a [u8],
  v_half: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv420pRow<'a> {
  /// Bundles one row of a 4:2:0 source for a [`Yuv420pSink`].
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

  /// Half-width U (Cb) row — `width / 2` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u8] {
    self.u_half
  }

  /// Half-width V (Cr) row — `width / 2` bytes.
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

/// Sinks that consume YUV 4:2:0 rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to
/// [`Yuv420pRow`]. Implementors get `process(&mut self, row: Yuv420pRow<'_>)`
/// via the supertrait.
pub trait Yuv420pSink: for<'a> PixelSink<Input<'a> = Yuv420pRow<'a>> {}

/// Converts a YUV 4:2:0 frame by walking its rows and feeding each one
/// to the [`Yuv420pSink`].
///
/// The kernel is a pure row walker — no color arithmetic happens here.
/// Slice math picks the Y row and the correct chroma row for each
/// output row (`chroma_row = row / 2` for 4:2:0) and hands borrows to
/// the Sink. The Sink decides what to derive and where to write.
///
/// `matrix` and `full_range` are passed through each [`Yuv420pRow`] so
/// the Sink has them available when calling row primitives.
pub fn yuv420p_to<S: Yuv420pSink>(
  src: &Yuv420pFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) {
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

    // 4:2:0 chroma subsampling: two consecutive Y rows share one
    // chroma row.
    let chroma_row = row / 2;
    let u_start = chroma_row * u_stride;
    let v_start = chroma_row * v_stride;
    let u_half = &u_plane[u_start..u_start + chroma_width];
    let v_half = &v_plane[v_start..v_start + chroma_width];

    sink.process(Yuv420pRow::new(y, u_half, v_half, row, matrix, full_range));
  }
}
