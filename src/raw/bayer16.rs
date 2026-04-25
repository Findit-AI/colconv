//! 10 / 12 / 14 / 16-bit Bayer (`AV_PIX_FMT_BAYER_*16LE` after MSB
//! shift) — single-plane mosaic source carrying `u16` samples
//! MSB-aligned at the declared bit depth.
//!
//! Shape mirrors [`super::bayer`] for the 8-bit case but with a
//! `u16` plane and a `BITS` const generic. Sinks consume
//! [`BayerRow16<'_, BITS>`] (different row type from the 8-bit
//! [`super::BayerRow`] so the type system pins the input bit depth
//! at the sink boundary).
//!
//! Sample convention follows the rest of the high-bit-depth crate:
//! the active `BITS` bits live in the high `BITS` of each `u16`,
//! with the low `16 - BITS` bits zero. Use
//! [`crate::frame::BayerFrame16::try_new_checked`] on untrusted
//! input to enforce this.

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
/// Carries `&[u16]` slices for `above` / `mid` / `below` (clamped at
/// the top / bottom edges by the walker), the row index, the
/// pattern, the demosaic algorithm, and the **unscaled** fused
/// `M = CCM · diag(wb)` 3×3. Output-bit-depth scaling (multiply
/// by `255 / max_msb` for u8 output, or `1 / (1 << (16 - BITS))`
/// for low-packed u16 output) is the kernel's job — different
/// kernels for different output types.
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

  /// Row above `mid`, clamped to `mid` when this is the top row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn above(&self) -> &'a [u16] {
    self.above
  }

  /// The row currently being produced — `width` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn mid(&self) -> &'a [u16] {
    self.mid
  }

  /// Row below `mid`, clamped to `mid` when this is the bottom row.
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
  sink.begin_frame(src.width(), src.height())?;

  let m = fuse_wb_ccm(&wb, &ccm);

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let plane = src.data();

  for row in 0..h {
    let above_row = if row == 0 { 0 } else { row - 1 };
    let below_row = if row + 1 == h { h - 1 } else { row + 1 };

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
mod tests {
  use super::*;
  use crate::{
    frame::Bayer12Frame,
    row::{bayer16_to_rgb_row, bayer16_to_rgb_u16_row},
  };
  use core::convert::Infallible;

  /// Sink that walks each `BayerRow16<BITS>` through the public
  /// u8-output dispatcher with SIMD off.
  struct CaptureRgbU8<'a, const BITS: u32> {
    out: &'a mut [u8],
    width: u32,
  }

  impl<const BITS: u32> PixelSink for CaptureRgbU8<'_, BITS> {
    type Input<'b> = BayerRow16<'b, BITS>;
    type Error = Infallible;

    fn begin_frame(&mut self, width: u32, _height: u32) -> Result<(), Self::Error> {
      self.width = width;
      Ok(())
    }

    fn process(&mut self, row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
      let r = row.row();
      let w = self.width as usize;
      let off = r * w * 3;
      let dst = &mut self.out[off..off + 3 * w];
      bayer16_to_rgb_row::<BITS>(
        row.above(),
        row.mid(),
        row.below(),
        row.row_parity(),
        row.pattern(),
        row.demosaic(),
        row.m(),
        dst,
        false,
      );
      Ok(())
    }
  }

  impl<const BITS: u32> BayerSink16<BITS> for CaptureRgbU8<'_, BITS> {}

  /// Sink that walks each `BayerRow16<BITS>` through the public
  /// u16-output dispatcher with SIMD off.
  struct CaptureRgbU16<'a, const BITS: u32> {
    out: &'a mut [u16],
    width: u32,
  }

  impl<const BITS: u32> PixelSink for CaptureRgbU16<'_, BITS> {
    type Input<'b> = BayerRow16<'b, BITS>;
    type Error = Infallible;

    fn begin_frame(&mut self, width: u32, _height: u32) -> Result<(), Self::Error> {
      self.width = width;
      Ok(())
    }

    fn process(&mut self, row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
      let r = row.row();
      let w = self.width as usize;
      let off = r * w * 3;
      let dst = &mut self.out[off..off + 3 * w];
      bayer16_to_rgb_u16_row::<BITS>(
        row.above(),
        row.mid(),
        row.below(),
        row.row_parity(),
        row.pattern(),
        row.demosaic(),
        row.m(),
        dst,
        false,
      );
      Ok(())
    }
  }

  impl<const BITS: u32> BayerSink16<BITS> for CaptureRgbU16<'_, BITS> {}

  /// Build a 12-bit MSB-aligned RGGB Bayer plane from per-channel
  /// nominal values (each 0..=4095). MSB-aligned means each sample
  /// is shifted left by 4 to occupy the high 12 bits of a u16.
  fn solid_rggb_12bit(width: u32, height: u32, r: u16, g: u16, b: u16) -> std::vec::Vec<u16> {
    let w = width as usize;
    let h = height as usize;
    let mut data = std::vec![0u16; w * h];
    let shift = 16 - 12;
    for y in 0..h {
      for x in 0..w {
        let v = match (y & 1, x & 1) {
          (0, 0) => r,
          (0, 1) => g,
          (1, 0) => g,
          (1, 1) => b,
          _ => unreachable!(),
        };
        data[y * w + x] = v << shift;
      }
    }
    data
  }

  fn assert_interior_u8(rgb: &[u8], w: u32, h: u32, expect: (u8, u8, u8)) {
    let w = w as usize;
    let h = h as usize;
    for y in 1..h - 1 {
      for x in 1..w - 1 {
        let i = (y * w + x) * 3;
        assert_eq!(rgb[i], expect.0, "u8 px ({x},{y}) R");
        assert_eq!(rgb[i + 1], expect.1, "u8 px ({x},{y}) G");
        assert_eq!(rgb[i + 2], expect.2, "u8 px ({x},{y}) B");
      }
    }
  }

  fn assert_interior_u16(rgb: &[u16], w: u32, h: u32, expect: (u16, u16, u16)) {
    let w = w as usize;
    let h = h as usize;
    for y in 1..h - 1 {
      for x in 1..w - 1 {
        let i = (y * w + x) * 3;
        assert_eq!(rgb[i], expect.0, "u16 px ({x},{y}) R");
        assert_eq!(rgb[i + 1], expect.1, "u16 px ({x},{y}) G");
        assert_eq!(rgb[i + 2], expect.2, "u16 px ({x},{y}) B");
      }
    }
  }

  #[test]
  fn bayer12_solid_red_rggb_yields_u8_red_interior() {
    // 12-bit max = 4095. R = 4095 (white at this channel), G = B = 0.
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb_12bit(w, h, 4095, 0, 0);
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgbU8::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_interior_u8(&rgb, w, h, (255, 0, 0));
  }

  #[test]
  fn bayer12_solid_red_rggb_yields_u16_red_interior() {
    // 12-bit u16 output: max = 4095 (low-packed), so red site → 4095.
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb_12bit(w, h, 4095, 0, 0);
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u16; (w * h * 3) as usize];
    let mut sink = CaptureRgbU16::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_interior_u16(&rgb, w, h, (4095, 0, 0));
  }

  #[test]
  fn bayer12_uniform_value_yields_uniform_u8_output() {
    // Every sample = 0x8000 (8-bit-ish midgray, MSB-aligned at 12-bit
    // it represents 2048/4095). u8 output should be (2048/4095)*255
    // ≈ 128 everywhere (including edges, since uniform input).
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![0x8000u16; (w * h) as usize];
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgbU8::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    // 0x8000 in 12-bit MSB-aligned = 0x800 = 2048; 2048/4095 * 255 ≈ 127.5 → 128
    for &c in &rgb {
      assert!((c as i32 - 128).abs() <= 1, "got {c}");
    }
  }

  #[test]
  fn bayer12_uniform_value_yields_uniform_u16_output() {
    // Every sample = 0xFFF0 (max 12-bit MSB-aligned). u16 output
    // should be 4095 (low-packed full white).
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![0xFFF0u16; (w * h) as usize];
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u16; (w * h * 3) as usize];
    let mut sink = CaptureRgbU16::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    for &c in &rgb {
      assert_eq!(c, 4095);
    }
  }

  #[test]
  fn bayer12_walker_calls_sink_once_per_row() {
    struct CountSink<const BITS: u32> {
      rows: u32,
    }
    impl<const BITS: u32> PixelSink for CountSink<BITS> {
      type Input<'a> = BayerRow16<'a, BITS>;
      type Error = Infallible;
      fn process(&mut self, _row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
        self.rows += 1;
        Ok(())
      }
    }
    impl<const BITS: u32> BayerSink16<BITS> for CountSink<BITS> {}

    let (w, h) = (8u32, 6u32);
    let raw = std::vec![0u16; (w * h) as usize];
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut sink = CountSink::<12> { rows: 0 };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_eq!(sink.rows, h);
  }
}
