//! NV16 — semi‑planar 4:2:2 (`AV_PIX_FMT_NV16`).
//!
//! Layout: one full‑size Y plane + one interleaved UV plane at half
//! width and **full height**. Each UV row is `U0, V0, U1, V1, …`
//! (U at even byte offsets, V at odd).
//!
//! Relationship to [`super::Nv12`]: identical per‑row kernel contract
//! — the 4:2:0 vs 4:2:2 axis is purely vertical, so
//! [`crate::row::nv12_to_rgb_row`] (and the NEON / SSE4.1 / AVX2 /
//! AVX‑512 / wasm simd128 backends it dispatches to) converts an NV16
//! row without modification. Only the walker differs: NV16 advances
//! chroma every row (`chroma_row = row`), whereas NV12 reads the same
//! chroma row twice (`chroma_row = row / 2`). The [`MixedSinker`]
//! impl for NV16 calls the shared NV12 row primitive directly.
//!
//! [`MixedSinker`]: crate::sinker::MixedSinker

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Nv16Frame, sealed::Sealed};

/// Zero‑sized marker for the NV16 source format. Used as the `F` type
/// parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nv16;

impl Sealed for Nv16 {}
impl SourceFormat for Nv16 {}

/// One output row of an NV16 source handed to an [`Nv16Sink`].
///
/// Accessors:
/// - [`y`](Self::y) — full‑width Y row (`width` bytes).
/// - [`uv`](Self::uv) — **interleaved, half‑width** UV row
///   (`width` bytes = `width / 2` U / V pairs) for the current Y row.
///   Unlike NV12, no two Y rows share a UV row.
/// - [`row`](Self::row) — output row index (`0 ..= frame.height() - 1`).
/// - [`matrix`](Self::matrix), [`full_range`](Self::full_range) — carried
///   through from the kernel call so the Sink can use them when calling
///   row primitives.
#[derive(Debug, Clone, Copy)]
pub struct Nv16Row<'a> {
  y: &'a [u8],
  uv: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Nv16Row<'a> {
  /// Bundles one row of an NV16 source for an [`Nv16Sink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u8],
    uv: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      uv,
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

  /// Interleaved UV row — `width` bytes laid out as
  /// `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv(&self) -> &'a [u8] {
    self.uv
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

/// Sinks that consume NV16 rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to [`Nv16Row`].
/// Implementors get
/// `process(&mut self, row: Nv16Row<'_>) -> Result<(), Self::Error>`
/// via the supertrait.
pub trait Nv16Sink: for<'a> PixelSink<Input<'a> = Nv16Row<'a>> {}

/// Converts an NV16 frame by walking its rows and feeding each one to
/// the [`Nv16Sink`].
///
/// The kernel is a pure row walker — no color arithmetic happens here.
/// Slice math picks the Y row and the matching UV row (1:1, since
/// 4:2:2 chroma is full‑height) and hands borrows to the Sink. The
/// Sink decides what to derive and where to write.
///
/// `matrix` and `full_range` are passed through each [`Nv16Row`] so the
/// Sink has them available when calling row primitives.
pub fn nv16_to<S: Nv16Sink>(
  src: &Nv16Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let uv_stride = src.uv_stride() as usize;
  // UV row payload is `width` bytes — `width / 2` interleaved U/V pairs.
  let uv_row_bytes = w;

  let y_plane = src.y();
  let uv_plane = src.uv();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:2:2: chroma advances every row (vs. 4:2:0's `row / 2`).
    let uv_start = row * uv_stride;
    let uv = &uv_plane[uv_start..uv_start + uv_row_bytes];

    sink.process(Nv16Row::new(y, uv, row, matrix, full_range))?;
  }
  Ok(())
}
