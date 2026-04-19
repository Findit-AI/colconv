//! NV21 — semi‑planar 4:2:0 with VU-ordered chroma (`AV_PIX_FMT_NV21`).
//!
//! Storage is identical to [`super::Nv12`] — one full-size Y plane
//! plus one interleaved chroma plane at half width and half height —
//! but the chroma bytes are **VU-ordered**: `V0, U0, V1, U1, …`
//! instead of NV12's `U0, V0, U1, V1, …`. Android MediaCodec's
//! default output for 8-bit decoded frames and some iOS camera
//! configurations emit NV21.
//!
//! Conversion semantics mirror [`super::Nv12`]: two consecutive Y
//! rows share one VU row (4:2:0), chroma is nearest-neighbor
//! upsampled in registers inside the row primitive — no intermediate
//! U / V scratch plane.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::Nv21Frame, sealed::Sealed};

/// Zero-sized marker for the NV21 source format. Used as the `F`
/// type parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nv21;

impl Sealed for Nv21 {}
impl SourceFormat for Nv21 {}

/// One output row of an NV21 source handed to an [`Nv21Sink`].
///
/// Accessors:
/// - [`y`](Self::y) — full-width Y row (`width` bytes).
/// - [`vu_half`](Self::vu_half) — **interleaved, half-width** VU row
///   (`width` bytes = `width / 2` V / U pairs) as it appears in the
///   source, **V-first**. The row primitive deinterleaves and
///   upsamples in-register.
/// - [`row`](Self::row) — output row index (`0 ..= frame.height() - 1`).
/// - [`matrix`](Self::matrix), [`full_range`](Self::full_range) —
///   carried through from the kernel call.
#[derive(Debug, Clone, Copy)]
pub struct Nv21Row<'a> {
  y: &'a [u8],
  vu_half: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Nv21Row<'a> {
  /// Bundles one row of an NV21 source for an [`Nv21Sink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    y: &'a [u8],
    vu_half: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      vu_half,
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

  /// Interleaved VU row — `width` bytes laid out as
  /// `V0, U0, V1, U1, …, V_{w/2-1}, U_{w/2-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn vu_half(&self) -> &'a [u8] {
    self.vu_half
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

/// Sinks that consume NV21 rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to
/// [`Nv21Row`]. Implementors get
/// `process(&mut self, row: Nv21Row<'_>) -> Result<(), Self::Error>`
/// via the supertrait.
pub trait Nv21Sink: for<'a> PixelSink<Input<'a> = Nv21Row<'a>> {}

/// Converts an NV21 frame by walking its rows and feeding each one
/// to the [`Nv21Sink`].
///
/// The kernel is a pure row walker — no color arithmetic happens
/// here. Slice math picks the Y row and the correct VU row for each
/// output row (`chroma_row = row / 2` for 4:2:0) and hands borrows
/// to the Sink. The Sink decides what to derive and where to write.
pub fn nv21_to<S: Nv21Sink>(
  src: &Nv21Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  // Per-frame preflight (see [`PixelSink::begin_frame`]).
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let vu_stride = src.vu_stride() as usize;
  // VU row payload is `width` bytes — `width / 2` interleaved V/U pairs.
  let vu_row_bytes = w;

  let y_plane = src.y();
  let vu_plane = src.vu();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    // 4:2:0 chroma subsampling: two consecutive Y rows share one VU row.
    let chroma_row = row / 2;
    let vu_start = chroma_row * vu_stride;
    let vu_half = &vu_plane[vu_start..vu_start + vu_row_bytes];

    sink.process(Nv21Row::new(y, vu_half, row, matrix, full_range))?;
  }
  Ok(())
}
