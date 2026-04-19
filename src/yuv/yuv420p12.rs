//! YUV 4:2:0 planar 12‑bit (`AV_PIX_FMT_YUV420P12LE`).
//!
//! Storage mirrors [`super::Yuv420p10`] — three planes, Y at full size
//! plus U / V at half width and half height — with **`u16`** samples
//! (12 active bits in the **low** 12 of each element, upper 4 zero).
//! The [`Yuv420p12Frame`] type alias pins the bit depth; the underlying
//! [`Yuv420pFrame16`] struct is const‑generic over `BITS`, so the same
//! Q15 scalar + SIMD kernel family that powers `Yuv420p10` runs
//! unchanged against the 12‑bit instantiation.
//!
//! Ships in colconv v0.2a alongside [`super::Yuv420p14`] and
//! [`super::P012`]. Kernel semantics match [`super::Yuv420p10`]: two
//! consecutive Y rows share one chroma row (4:2:0), chroma is
//! nearest‑neighbor upsampled in registers inside the row primitive,
//! and Q15 intermediates stay in i32 (chroma_sum < 10⁹ < i32 max at 12
//! bits — verified against the scalar reference per SIMD backend).

use crate::{
  ColorMatrix, PixelSink, SourceFormat,
  frame::{Yuv420p12Frame, Yuv420pFrame16},
  sealed::Sealed,
};

/// Zero‑sized marker for the YUV 4:2:0 **12‑bit** source format. Used
/// as the `F` type parameter on [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Yuv420p12;

impl Sealed for Yuv420p12 {}
impl SourceFormat for Yuv420p12 {}

/// One output row of a 12‑bit YUV 4:2:0 source handed to a
/// [`Yuv420p12Sink`]. Structurally identical to [`super::Yuv420p10Row`],
/// just with values in `[0, 4095]` instead of `[0, 1023]`.
#[derive(Debug, Clone, Copy)]
pub struct Yuv420p12Row<'a> {
  y: &'a [u16],
  u_half: &'a [u16],
  v_half: &'a [u16],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> Yuv420p12Row<'a> {
  /// Bundles one row of a 12‑bit 4:2:0 source for a [`Yuv420p12Sink`].
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

  /// `true` iff Y uses the full sample range (`[0, 4095]` for 12‑bit);
  /// `false` for limited range (`[256, 3760]` luma, `[256, 3840]`
  /// chroma — the 8‑bit `[16, 235]` / `[16, 240]` ranges scaled by 16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume 12‑bit YUV 4:2:0 rows.
pub trait Yuv420p12Sink: for<'a> PixelSink<Input<'a> = Yuv420p12Row<'a>> {}

/// Converts a 12‑bit YUV 4:2:0 frame by walking its rows and feeding
/// each one to the [`Yuv420p12Sink`]. Mirrors [`super::yuv420p10_to`] —
/// pure row walker, all color arithmetic happens inside the Sink via
/// the crate's row primitives instantiated at `BITS == 12`.
pub fn yuv420p12_to<S: Yuv420p12Sink>(
  src: &Yuv420p12Frame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  yuv420p12_walker::<12, S>(src, full_range, matrix, sink)
}

/// Row walker for the 12‑bit YUV 4:2:0 source. `BITS` is a const
/// generic so [`Yuv420pFrame16<BITS>`] geometry reads (stride, plane
/// slicing) are monomorphized; the row/sink types bound below are
/// still pinned to the 12‑bit variants.
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuv420p12_walker<const BITS: u32, S: Yuv420p12Sink>(
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

    sink.process(Yuv420p12Row::new(
      y, u_half, v_half, row, matrix, full_range,
    ))?;
  }
  Ok(())
}
