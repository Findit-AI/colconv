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
/// Carries the three row-aligned slices the demosaic kernel needs
/// (top edge: `above` clamped to `mid`; bottom edge: `below`
/// clamped to `mid`), the row index, the pattern, the demosaic
/// algorithm, and the fused 3×3 transform.
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

  /// Row above `mid`, clamped to `mid` when this is the top row.
  /// Same length as [`Self::mid`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn above(&self) -> &'a [u8] {
    self.above
  }

  /// The row currently being produced — `width` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn mid(&self) -> &'a [u8] {
    self.mid
  }

  /// Row below `mid`, clamped to `mid` when this is the bottom row.
  /// Same length as [`Self::mid`].
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
/// **Allocation profile.** Zero per-row and zero per-frame heap
/// allocation. The walker computes `M` once on the stack at entry,
/// slices `above` / `mid` / `below` references into the source
/// plane (clamping at row 0 and `height − 1`), and hands them to
/// the sink. The sink owns the RGB output buffer.
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
    // Top-edge / bottom-edge replicate clamp.
    let above_row = if row == 0 { 0 } else { row - 1 };
    let below_row = if row + 1 == h { h - 1 } else { row + 1 };

    let above = &plane[above_row * stride..above_row * stride + w];
    let mid = &plane[row * stride..row * stride + w];
    let below = &plane[below_row * stride..below_row * stride + w];

    sink.process(BayerRow::new(above, mid, below, row, pattern, demosaic, m))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests {
  use super::*;
  use crate::row::bayer_to_rgb_row;
  use core::convert::Infallible;

  /// Test sink that captures every output row into a single packed
  /// RGB buffer the test owns. Calls the public dispatcher with
  /// SIMD turned off (only scalar is wired up today).
  struct CaptureRgb<'a> {
    out: &'a mut [u8],
    width: u32,
  }

  impl PixelSink for CaptureRgb<'_> {
    type Input<'b> = BayerRow<'b>;
    type Error = Infallible;

    fn begin_frame(&mut self, width: u32, _height: u32) -> Result<(), Self::Error> {
      self.width = width;
      Ok(())
    }

    fn process(&mut self, row: BayerRow<'_>) -> Result<(), Self::Error> {
      let row_idx = row.row();
      let w = self.width as usize;
      let off = row_idx * w * 3;
      let dst = &mut self.out[off..off + 3 * w];
      bayer_to_rgb_row(
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

  impl BayerSink for CaptureRgb<'_> {}

  /// Build an RGGB Bayer plane from per-channel solid values. Pattern:
  /// row 0 = R G R G ..., row 1 = G B G B ..., row 2 = R G R G, ...
  fn solid_rggb(width: u32, height: u32, r: u8, g: u8, b: u8) -> std::vec::Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut data = std::vec![0u8; w * h];
    for y in 0..h {
      for x in 0..w {
        data[y * w + x] = match (y & 1, x & 1) {
          (0, 0) => r,
          (0, 1) => g,
          (1, 0) => g,
          (1, 1) => b,
          _ => unreachable!(),
        };
      }
    }
    data
  }

  /// Walk only the interior (rows `[1, h-2]`, cols `[1, w-2]`)
  /// where edge-clamp artifacts don't apply, and assert the
  /// expected RGB triple at every site. Documented edge behavior:
  /// at edges, the clamp replicates the sampled color and
  /// contaminates the demosaic average, which is the standard
  /// bilinear-edge tradeoff every libdc1394 / OpenCV / RawTherapee
  /// scalar reference also exhibits.
  fn assert_interior(rgb: &[u8], w: u32, h: u32, expect: (u8, u8, u8)) {
    let w = w as usize;
    let h = h as usize;
    for y in 1..h - 1 {
      for x in 1..w - 1 {
        let i = (y * w + x) * 3;
        assert_eq!(rgb[i], expect.0, "px ({x},{y}) R");
        assert_eq!(rgb[i + 1], expect.1, "px ({x},{y}) G");
        assert_eq!(rgb[i + 2], expect.2, "px ({x},{y}) B");
      }
    }
  }

  #[test]
  fn bayer_solid_red_rggb_neutral_wb_identity_ccm_yields_red_interior() {
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb(w, h, 255, 0, 0);
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgb {
      out: &mut rgb,
      width: 0,
    };
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_interior(&rgb, w, h, (255, 0, 0));
  }

  #[test]
  fn bayer_solid_green_rggb_yields_green_interior() {
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb(w, h, 0, 255, 0);
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgb {
      out: &mut rgb,
      width: 0,
    };
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_interior(&rgb, w, h, (0, 255, 0));
  }

  #[test]
  fn bayer_solid_blue_rggb_yields_blue_interior() {
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb(w, h, 0, 0, 255);
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgb {
      out: &mut rgb,
      width: 0,
    };
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_interior(&rgb, w, h, (0, 0, 255));
  }

  #[test]
  fn bayer_uniform_byte_yields_uniform_output() {
    // Every byte = 200; every demosaic site reads 200 in every
    // neighbor (clamps included), so all output channels = 200
    // even at edges. Smoke test for the kernel arithmetic itself,
    // independent of pattern phase.
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![200u8; (w * h) as usize];
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgb {
      out: &mut rgb,
      width: 0,
    };
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    for &b in &rgb {
      assert_eq!(b, 200);
    }
  }

  #[test]
  fn bayer_pattern_swap_red_to_blue_interior() {
    // RGGB plane filled to look red, decoded with BGGR pattern,
    // should come out blue at interior sites (R↔B swap).
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb(w, h, 255, 0, 0);
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgb {
      out: &mut rgb,
      width: 0,
    };
    bayer_to(
      &frame,
      BayerPattern::Bggr,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_interior(&rgb, w, h, (0, 0, 255));
  }

  #[test]
  fn bayer_walker_calls_sink_once_per_row() {
    struct CountSink {
      rows: u32,
    }
    impl PixelSink for CountSink {
      type Input<'a> = BayerRow<'a>;
      type Error = Infallible;
      fn process(&mut self, _row: BayerRow<'_>) -> Result<(), Self::Error> {
        self.rows += 1;
        Ok(())
      }
    }
    impl BayerSink for CountSink {}

    let (w, h) = (8u32, 6u32);
    let raw = std::vec![0u8; (w * h) as usize];
    let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
    let mut sink = CountSink { rows: 0 };
    bayer_to(
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
