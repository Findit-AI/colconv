//! YUV 4:2:0 planar 10‑bit (`AV_PIX_FMT_YUV420P10LE`).
//!
//! Storage mirrors [`super::Yuv420p`] — three planes, Y at full size
//! plus U / V at half width and half height — but sample width is
//! **`u16`** (10 active bits in the low bits of each element). The
//! [`Yuv420p10Frame`] type alias pins the bit depth; the underlying
//! [`Yuv420pFrame16`] struct is const‑generic over `BITS` so 12‑bit
//! and 14‑bit variants can be added by relaxing its validator without
//! changing kernel math.
//!
//! Ships in colconv v0.2 as the first high‑bit‑depth format (HDR /
//! 10‑bit SDR keystone). Kernel semantics match [`super::Yuv420p`]:
//! two consecutive Y rows share one chroma row (4:2:0), chroma is
//! nearest‑neighbor upsampled in registers inside the row primitive.

use crate::{
  ColorMatrix, PixelSink, SourceFormat,
  frame::{Yuv420p10Frame, Yuv420pFrame16},
  sealed::Sealed,
};

/// Zero‑sized marker for the YUV 4:2:0 **10‑bit** source format. Used
/// as the `F` type parameter on [`crate::sinker::MixedSinker`].
///
/// Ship 2 ships only the 10‑bit specialization; 12‑ and 14‑bit will
/// arrive as separate markers (`Yuv420p12`, `Yuv420p14`) referring to
/// the same underlying [`Yuv420pFrame16`] struct with different
/// `BITS` values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv420p10;

impl Sealed for Yuv420p10 {}
impl SourceFormat for Yuv420p10 {}

/// One output row of a 10‑bit YUV 4:2:0 source handed to a
/// [`Yuv420p10Sink`]. Structurally identical to [`super::Yuv420pRow`],
/// just `u16` samples.
#[derive(Debug, Clone, Copy)]
pub struct Yuv420p10Row<'a> {
  y: &'a [u16],
  u_half: &'a [u16],
  v_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv420p10Row<'a> {
  /// Bundles one row of a 10‑bit 4:2:0 source for a [`Yuv420p10Sink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u16],
    u_half: &'a [u16],
    v_half: &'a [u16],
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

  /// Full‑width Y (luma) row — `width` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn y(&self) -> &'a [u16] {
    self.y
  }

  /// Half‑width U (Cb) row — `width / 2` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn u_half(&self) -> &'a [u16] {
    self.u_half
  }

  /// Half‑width V (Cr) row — `width / 2` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn v_half(&self) -> &'a [u16] {
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

  /// `true` iff Y uses the full sample range (`[0, 1023]` for 10‑bit);
  /// `false` for limited range (`[64, 940]` luma, `[64, 960]` chroma).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume 10‑bit YUV 4:2:0 rows.
pub trait Yuv420p10Sink: for<'a> PixelSink<Input<'a> = Yuv420p10Row<'a>> {}

/// Converts a 10‑bit YUV 4:2:0 frame by walking its rows and feeding
/// each one to the [`Yuv420p10Sink`]. See [`super::yuv420p_to`] for
/// the shared design rationale — kernel is a pure row walker, all
/// color arithmetic happens inside the Sink via the crate's row
/// primitives.
pub fn yuv420p10_to<S: Yuv420p10Sink>(
  src: &Yuv420p10Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  // Delegates to the const‑generic implementation. Once 12‑ and
  // 14‑bit markers ship, each gets a thin wrapper that pins `BITS`.
  yuv420p_n_to::<10, S>(src, full_range, matrix, sink)
}

/// Shared walker for high‑bit‑depth YUV 4:2:0 sources. `BITS` is the
/// active sample depth (10 for Ship 2; 12 / 14 reuse the same walker
/// once the corresponding markers and Sink subtraits are defined).
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuv420p_n_to<const BITS: u32, S: Yuv420p10Sink>(
  src: &Yuv420pFrame16<'_, BITS>,
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

    let chroma_row = row / 2;
    let u_start = chroma_row * u_stride;
    let v_start = chroma_row * v_stride;
    let u_half = &u_plane[u_start..u_start + chroma_width];
    let v_half = &v_plane[v_start..v_start + chroma_width];

    sink.process(Yuv420p10Row::new(
      y, u_half, v_half, row, matrix, full_range,
    ))?;
  }
  Ok(())
}
