//! NV12 — semi‑planar 4:2:0 (`AV_PIX_FMT_NV12`).
//!
//! Layout: one full‑size Y plane + one interleaved UV plane at half
//! width and half height. Each UV row is `U0, V0, U1, V1, …` (U at even
//! byte offsets, V at odd). This is the canonical 8‑bit output of
//! Apple VideoToolbox, VA‑API, NVDEC, D3D11VA, and Android MediaCodec.
//!
//! Conversion semantics mirror [`super::Yuv420p`]: two consecutive Y
//! rows share one UV row (4:2:0), chroma is nearest‑neighbor upsampled
//! **in registers** inside the row primitive — no intermediate U / V
//! scratch plane.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Nv12Frame, sealed::Sealed};

/// Zero‑sized marker for the NV12 source format. Used as the `F` type
/// parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nv12;

impl Sealed for Nv12 {}
impl SourceFormat for Nv12 {}

/// One output row of an NV12 source handed to an [`Nv12Sink`].
///
/// Accessors:
/// - [`y`](Self::y) — full‑width Y row (`width` bytes).
/// - [`uv_half`](Self::uv_half) — **interleaved, half‑width** UV row
///   (`width` bytes = `width / 2` U / V pairs) as it appears in the
///   source, without deinterleaving or upsampling. Row primitives do
///   both fused in‑register.
/// - [`row`](Self::row) — output row index (`0 ..= frame.height() - 1`).
/// - [`matrix`](Self::matrix), [`full_range`](Self::full_range) — carried
///   through from the kernel call so the Sink can use them when calling
///   row primitives.
#[derive(Debug, Clone, Copy)]
pub struct Nv12Row<'a> {
  y: &'a [u8],
  uv_half: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Nv12Row<'a> {
  /// Bundles one row of an NV12 source for an [`Nv12Sink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u8],
    uv_half: &'a [u8],
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

  /// Full‑width Y (luma) row — `width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u8] {
    self.y
  }

  /// Interleaved UV row — `width` bytes laid out as
  /// `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn uv_half(&self) -> &'a [u8] {
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

  /// `true` iff Y ∈ `[0, 255]` (full range); `false` for limited.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume NV12 rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to [`Nv12Row`].
/// Implementors get `process(&mut self, row: Nv12Row<'_>)` via the
/// supertrait.
pub trait Nv12Sink: for<'a> PixelSink<Input<'a> = Nv12Row<'a>> {}

/// Converts an NV12 frame by walking its rows and feeding each one to
/// the [`Nv12Sink`].
///
/// The kernel is a pure row walker — no color arithmetic happens here.
/// Slice math picks the Y row and the correct UV row for each output
/// row (`chroma_row = row / 2` for 4:2:0) and hands borrows to the
/// Sink. The Sink decides what to derive and where to write.
///
/// `matrix` and `full_range` are passed through each [`Nv12Row`] so the
/// Sink has them available when calling row primitives.
pub fn nv12_to<S: Nv12Sink>(
  src: &Nv12Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  // Per-frame preflight (see [`PixelSink::begin_frame`]). Any error
  // here propagates before row 0 is touched.
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

    // 4:2:0 chroma subsampling: two consecutive Y rows share one UV row.
    let chroma_row = row / 2;
    let uv_start = chroma_row * uv_stride;
    let uv_half = &uv_plane[uv_start..uv_start + uv_row_bytes];

    // `?` short-circuits the walk on the first sink error.
    sink.process(Nv12Row::new(y, uv_half, row, matrix, full_range))?;
  }
  Ok(())
}
