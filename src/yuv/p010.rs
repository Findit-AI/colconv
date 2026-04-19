//! P010 — semi‑planar 4:2:0, 10‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P010LE`).
//!
//! Storage is a 2‑plane layout: one full‑size Y plane plus one
//! interleaved UV plane at half width and half height. Sample width
//! is `u16` with the 10 active bits in the **high** 10 positions of
//! each element (`sample = value << 6`), low 6 bits zero. This is
//! Microsoft's P010 convention and what every HDR hardware decoder
//! emits — Apple VideoToolbox, VA‑API, NVDEC, D3D11VA, Intel QSV.
//!
//! Conversion semantics mirror [`super::Nv12`] on the layout side and
//! [`super::Yuv420p10`] on the Q‑math side: two consecutive Y rows
//! share one UV row (4:2:0), chroma is nearest‑neighbor upsampled in
//! registers inside the row primitive, and every SIMD backend shifts
//! each `u16` load right by 6 to extract the 10‑bit value before
//! running the same Q15 pipeline used by [`super::Yuv420p10`].

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::P010Frame, sealed::Sealed};

/// Zero‑sized marker for the P010 source format. Used as the `F` type
/// parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct P010;

impl Sealed for P010 {}
impl SourceFormat for P010 {}

/// One output row of a P010 source handed to a [`P010Sink`].
///
/// Accessors:
/// - [`y`](Self::y) — full‑width Y row (`width` `u16` samples, high‑
///   bit‑packed).
/// - [`uv_half`](Self::uv_half) — **interleaved, half‑width** UV row
///   (`width` `u16` elements = `width / 2` U/V pairs, U first). The
///   row primitive deinterleaves and upsamples in‑register.
/// - [`row`](Self::row) — output row index (`0 ..= frame.height() - 1`).
/// - [`matrix`](Self::matrix), [`full_range`](Self::full_range) —
///   carried through from the kernel call.
#[derive(Debug, Clone, Copy)]
pub struct P010Row<'a> {
  y: &'a [u16],
  uv_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> P010Row<'a> {
  /// Bundles one row of a P010 source for a [`P010Sink`].
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

  /// Full‑width Y (luma) row — `width` `u16` samples, high‑bit‑packed
  /// (10 active bits in the high 10 of each element).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }

  /// Interleaved UV row — `width` `u16` elements laid out as
  /// `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`. Each element is
  /// high‑bit‑packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv_half(&self) -> &'a [u16] {
    self.uv_half
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

  /// `true` iff Y uses the full sample range (`[0, 1023]` for 10‑bit,
  /// scaled into the high 10 bits of each `u16`); `false` for limited
  /// range.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume P010 rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to
/// [`P010Row`]. Implementors get
/// `process(&mut self, row: P010Row<'_>) -> Result<(), Self::Error>`
/// via the supertrait.
pub trait P010Sink: for<'a> PixelSink<Input<'a> = P010Row<'a>> {}

/// Converts a P010 frame by walking its rows and feeding each one to
/// the [`P010Sink`].
///
/// The kernel is a pure row walker — no color arithmetic happens
/// here. Slice math picks the Y row and the correct UV row for each
/// output row (`chroma_row = row / 2` for 4:2:0) and hands borrows to
/// the Sink. The Sink decides what to derive and where to write.
pub fn p010_to<S: P010Sink>(
  src: &P010Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let uv_stride = src.uv_stride() as usize;
  // UV row payload is `width` `u16` elements — `width / 2` interleaved
  // U/V pairs.
  let uv_row_elems = w;

  let y_plane = src.y();
  let uv_plane = src.uv();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:2:0 chroma subsampling: two consecutive Y rows share one UV
    // row.
    let chroma_row = row / 2;
    let uv_start = chroma_row * uv_stride;
    let uv_half = &uv_plane[uv_start..uv_start + uv_row_elems];

    sink.process(P010Row::new(y, uv_half, row, matrix, full_range))?;
  }
  Ok(())
}
