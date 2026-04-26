//! 10 / 12 / 14 / 16-bit Bayer — single-plane mosaic source
//! carrying **low-packed** `u16` samples.
//!
//! Shape mirrors [`super::bayer`] for the 8-bit case but with a
//! `u16` plane and a `BITS` const generic. Sinks consume
//! [`BayerRow16<'_, BITS>`] (different row type from the 8-bit
//! [`super::BayerRow`] so the type system pins the input bit depth
//! at the sink boundary).
//!
//! Sample convention is **low-packed**: active samples occupy the
//! low `BITS` bits of each `u16`, valid range
//! `[0, (1 << BITS) - 1]`. This matches the planar
//! [`Yuv420pFrame16`](crate::frame::Yuv420pFrame16) family in
//! packing (low bits) but not validation cost: Bayer16's
//! [`crate::frame::BayerFrame16::try_new`] validates every active
//! sample's range as part of construction, so the
//! [`bayer16_to`] walker is fully fallible — no data-dependent
//! panic surface. **Note:** this is the opposite of
//! [`PnFrame`](crate::frame::PnFrame) (high-bit-packed semi-planar
//! `u16`); if your upstream provides high-bit-packed Bayer,
//! right-shift by `(16 - BITS)` before constructing
//! [`BayerFrame16`](crate::frame::BayerFrame16).

use crate::{
  PixelSink, SourceFormat,
  frame::BayerFrame16,
  raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, fuse_wb_ccm},
  sealed::Sealed,
};

/// Zero-sized marker for the high-bit-depth Bayer source family.
/// Parameterized on the active bit depth `BITS` ∈ {10, 12, 14, 16}.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bayer16<const BITS: u32>;

impl<const BITS: u32> Sealed for Bayer16<BITS> {}
impl<const BITS: u32> SourceFormat for Bayer16<BITS> {}

/// Type-aliased markers for readability at call sites.
pub type Bayer10 = Bayer16<10>;
/// 12-bit Bayer source marker.
pub type Bayer12 = Bayer16<12>;
/// 14-bit Bayer source marker.
pub type Bayer14 = Bayer16<14>;
/// 16-bit Bayer source marker.
pub type Bayer16Bit = Bayer16<16>;

/// One output row of a high-bit-depth Bayer source handed to a
/// [`BayerSink16<BITS>`].
///
/// Carries `&[u16]` slices for `above` / `mid` / `below`, the row
/// index, the pattern, the demosaic algorithm, and the **unscaled**
/// fused `M = CCM · diag(wb)` 3×3. Output-bit-depth scaling
/// (multiply by `255 / ((1 << BITS) - 1)` for u8 output; identity
/// for low-packed u16 output) is the kernel's job.
///
/// **Boundary contract: mirror-by-2** — see [`super::BayerRow`]
/// for the full discussion. Top edge supplies `above = mid_row(1)`,
/// bottom edge supplies `below = mid_row(h - 2)`; replicate
/// fallback applies only when `height < 2`. Custom sinks must
/// honor this convention.
#[derive(Debug, Clone, Copy)]
pub struct BayerRow16<'a, const BITS: u32> {
  above: &'a [u16],
  mid: &'a [u16],
  below: &'a [u16],
  row: usize,
  pattern: BayerPattern,
  demosaic: BayerDemosaic,
  m: [[f32; 3]; 3],
}

impl<'a, const BITS: u32> BayerRow16<'a, BITS> {
  /// Bundles one row of a high-bit-depth Bayer source for a
  /// [`BayerSink16<BITS>`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    above: &'a [u16],
    mid: &'a [u16],
    below: &'a [u16],
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
  /// `mid_row(row - 1)` for interior rows; `mid_row(1)` at the top
  /// edge. See [`super::BayerRow::above`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn above(&self) -> &'a [u16] {
    self.above
  }

  /// The row currently being produced — `width` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn mid(&self) -> &'a [u16] {
    self.mid
  }

  /// Row below `mid` per the **mirror-by-2** boundary contract:
  /// `mid_row(row + 1)` for interior rows; `mid_row(h - 2)` at the
  /// bottom edge. See [`super::BayerRow::below`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn below(&self) -> &'a [u16] {
    self.below
  }

  /// Output row index within the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Row parity (`row & 1`).
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

  /// Borrow the fused `M = CCM · diag(wb)` transform. Unscaled —
  /// kernels apply the input/output bit-depth scaling themselves.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn m(&self) -> &[[f32; 3]; 3] {
    &self.m
  }

  /// Active bit depth — 10, 12, 14, or 16.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// Sinks that consume high-bit-depth Bayer rows at a fixed `BITS`.
pub trait BayerSink16<const BITS: u32>:
  for<'a> PixelSink<Input<'a> = BayerRow16<'a, BITS>>
{
}

/// Walks a [`BayerFrame16<BITS>`] row by row, handing each row to
/// the sink along with the precomputed `M = CCM · diag(wb)` 3×3.
///
/// **Fully fallible.** The walker performs no data-dependent
/// validation — every panic surface that previously existed has
/// been moved to [`BayerFrame16::try_new`], which validates
/// dimensions *and* every active sample's range at construction.
/// Once you hold a `BayerFrame16<BITS>`, the conversion can only
/// fail through `S::Error` (sink-side I/O, geometry-mismatch,
/// etc.); bad sample data is reported as
/// [`crate::frame::BayerFrame16Error::SampleOutOfRange`] from the
/// frame constructor instead of as a runtime panic here.
///
/// **Allocation profile.** Zero per-row and zero per-frame heap
/// allocation, identical to the 8-bit [`super::bayer_to`].
pub fn bayer16_to<const BITS: u32, S: BayerSink16<BITS>>(
  src: &BayerFrame16<'_, BITS>,
  pattern: BayerPattern,
  demosaic: BayerDemosaic,
  wb: WhiteBalance,
  ccm: ColorCorrectionMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let plane = src.data();

  sink.begin_frame(src.width(), src.height())?;

  let m = fuse_wb_ccm(&wb, &ccm);

  for row in 0..h {
    // Mirror-by-2 row clamp; see [`super::bayer::bayer_to`] for
    // the rationale (CFA-parity preservation at boundaries).
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

    sink.process(BayerRow16::<BITS>::new(
      above, mid, below, row, pattern, demosaic, m,
    ))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests;
