//! 8-bit Bayer (`AV_PIX_FMT_BAYER_BGGR8` / `RGGB8` / `GRBG8` /
//! `GBRG8`) — single-plane mosaic source.
//!
//! Walker hands each output row to a [`BayerSink`] together with
//! the three row-aligned slices the demosaic kernel needs (`above`,
//! `mid`, `below`) and the fused `M = CCM · diag(wb)` transform.
//! The kernel does the bilinear demosaic and the 3×3 matmul in one
//! pass; the sink owns the RGB output buffer.

use crate::{
  PixelSink, SourceFormat,
  frame::BayerFrame,
  raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, fuse_wb_ccm},
  sealed::Sealed,
};

/// Zero-sized marker for the 8-bit Bayer source format. Used as the
/// `F` type parameter on [`crate::sinker::MixedSinker`] (once
/// integrated).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bayer;

impl Sealed for Bayer {}
impl SourceFormat for Bayer {}

/// One output row of a Bayer source handed to a [`BayerSink`].
///
/// Carries the three row-aligned slices the demosaic kernel needs,
/// the row index, the pattern, the demosaic algorithm, and the
/// fused 3×3 transform.
///
/// **Boundary contract: mirror-by-2.** At the top edge (row 0) the
/// walker supplies `above = mid_row(1)`, and at the bottom edge
/// (row `h - 1`) it supplies `below = mid_row(h - 2)` — *not* a
/// replicate clamp. This preserves CFA parity across the row
/// boundary because Bayer tiles in 2×2: skipping two rows lands on
/// the same color the missing-tap site would have provided.
/// Falls back to replicate when `height < 2`. Custom sinks must
/// honor this convention; calling [`crate::row::bayer_to_rgb_row`]
/// from a sink that supplies replicate-clamped row borrows will
/// produce different border pixels than [`super::bayer_to`] does.
///
/// Sinks call into [`crate::row::bayer_to_rgb_row`] (or directly
/// the scalar / SIMD primitive of their choice) with these slices to
/// produce one row of packed RGB output.
#[derive(Debug, Clone, Copy)]
pub struct BayerRow<'a> {
  above: &'a [u8],
  mid: &'a [u8],
  below: &'a [u8],
  row: usize,
  pattern: BayerPattern,
  demosaic: BayerDemosaic,
  m: [[f32; 3]; 3],
}

impl<'a> BayerRow<'a> {
  /// Bundles one row of an 8-bit Bayer source for a [`BayerSink`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    above: &'a [u8],
    mid: &'a [u8],
    below: &'a [u8],
    row: usize,
    pattern: BayerPattern,
    demosaic: BayerDemosaic,
    m: [[f32; 3]; 3],
  ) -> Self {
    Self {
      above,
      mid,
      below,
      row,
      pattern,
      demosaic,
      m,
    }
  }

  /// Row above `mid` per the **mirror-by-2** boundary contract:
  /// for an interior row this is `mid_row(row - 1)`; at the top
  /// edge (`row == 0`) it is `mid_row(1)`. Falls back to `mid` when
  /// `height < 2`. Same length as [`Self::mid`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn above(&self) -> &'a [u8] {
    self.above
  }

  /// The row currently being produced — `width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn mid(&self) -> &'a [u8] {
    self.mid
  }

  /// Row below `mid` per the **mirror-by-2** boundary contract:
  /// for an interior row this is `mid_row(row + 1)`; at the bottom
  /// edge (`row == h - 1`) it is `mid_row(h - 2)`. Falls back to
  /// `mid` when `height < 2`. Same length as [`Self::mid`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn below(&self) -> &'a [u8] {
    self.below
  }

  /// Output row index within the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Row parity (`row & 1`) — needed by the demosaic kernel to pick
  /// which Bayer site each pixel sits on.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row_parity(&self) -> u32 {
    (self.row & 1) as u32
  }

  /// The Bayer pattern this frame uses.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pattern(&self) -> BayerPattern {
    self.pattern
  }

  /// The demosaic algorithm requested by the caller.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn demosaic(&self) -> BayerDemosaic {
    self.demosaic
  }

  /// Borrow the fused `M = CCM · diag(wb)` transform.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn m(&self) -> &[[f32; 3]; 3] {
    &self.m
  }
}

/// Sinks that consume 8-bit Bayer rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to
/// [`BayerRow`].
pub trait BayerSink: for<'a> PixelSink<Input<'a> = BayerRow<'a>> {}

/// Walks an 8-bit [`BayerFrame`] row by row, handing each row to the
/// sink along with the precomputed `M = CCM · diag(wb)` transform.
///
/// **Boundary contract.** `above` / `below` use **mirror-by-2** at
/// the top and bottom edges (`row 0 → above = row 1`, `row h-1 →
/// below = row h-2`); see [`BayerRow`] for the full discussion.
///
/// **Allocation profile.** Zero per-row and zero per-frame heap
/// allocation. The walker computes `M` once on the stack at entry,
/// slices three row borrows into the source plane, and hands them
/// to the sink. The sink owns the RGB output buffer.
pub fn bayer_to<S: BayerSink>(
  src: &BayerFrame<'_>,
  pattern: BayerPattern,
  demosaic: BayerDemosaic,
  wb: WhiteBalance,
  ccm: ColorCorrectionMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let m = fuse_wb_ccm(&wb, &ccm);

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let plane = src.data();

  for row in 0..h {
    // **Mirror-by-2** row clamp at the top / bottom edges. See the
    // [`scalar::bayer_to_rgb_row`] kernel docs for the rationale
    // (preserves CFA parity across the boundary; replicate clamp
    // would mix wrong-color samples into the missing-channel
    // averages). Falls back to replicate when `h < 2`.
    let above_row = if row == 0 {
      if h >= 2 { 1 } else { 0 }
    } else {
      row - 1
    };
    let below_row = if row + 1 == h {
      if h >= 2 { h - 2 } else { h - 1 }
    } else {
      row + 1
    };

    let above = &plane[above_row * stride..above_row * stride + w];
    let mid = &plane[row * stride..row * stride + w];
    let below = &plane[below_row * stride..below_row * stride + w];

    sink.process(BayerRow::new(above, mid, below, row, pattern, demosaic, m))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests;
