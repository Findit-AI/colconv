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

  /// Assert every output pixel — **including the borders** —
  /// matches the expected RGB triple. Mirror-by-2 boundary handling
  /// preserves CFA parity, so a solid-channel Bayer mosaic stays
  /// solid across the full frame (no clamp-induced channel bleed
  /// at the edges or corners).
  fn assert_full_frame(rgb: &[u8], w: u32, h: u32, expect: (u8, u8, u8)) {
    let w = w as usize;
    let h = h as usize;
    for y in 0..h {
      for x in 0..w {
        let i = (y * w + x) * 3;
        assert_eq!(rgb[i], expect.0, "px ({x},{y}) R");
        assert_eq!(rgb[i + 1], expect.1, "px ({x},{y}) G");
        assert_eq!(rgb[i + 2], expect.2, "px ({x},{y}) B");
      }
    }
  }

  #[test]
  fn bayer_solid_red_rggb_neutral_wb_identity_ccm_yields_red() {
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
    assert_full_frame(&rgb, w, h, (255, 0, 0));
  }

  #[test]
  fn bayer_solid_green_rggb_yields_green() {
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
    assert_full_frame(&rgb, w, h, (0, 255, 0));
  }

  #[test]
  fn bayer_solid_blue_rggb_yields_blue() {
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
    assert_full_frame(&rgb, w, h, (0, 0, 255));
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
  fn bayer_pattern_swap_red_to_blue() {
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
    assert_full_frame(&rgb, w, h, (0, 0, 255));
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

  #[test]
  fn bayer_walker_handles_odd_width_and_height_full_frame() {
    // 15×7 RGGB-tiled solid red. Mirror-by-2 boundary handling
    // means every output pixel — interior and border — should
    // match the expected channel.
    let (w, h) = (15u32, 7u32);
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
    assert_full_frame(&rgb, w, h, (255, 0, 0));
  }

  #[test]
  fn bayer_walker_handles_2x2_minimum_tile() {
    // 2×2 RGGB-filled red. Smallest frame that still has a
    // complete CFA tile. Mirror-by-2 maps `row -1 → row 1` and
    // `row 2 → row 0`, so each row of the 2-row frame uses the
    // other row as both `above` and `below`. Same for columns.
    // Full frame should be solid red.
    let raw = solid_rggb(2, 2, 255, 0, 0);
    let frame = BayerFrame::try_new(&raw, 2, 2, 2).unwrap();
    let mut rgb = std::vec![0u8; 2 * 2 * 3];
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
    assert_full_frame(&rgb, 2, 2, (255, 0, 0));
  }

  #[test]
  fn bayer_walker_handles_1x1() {
    // 1×1 corner case — every "neighbor" clamps to the single
    // sample. Demosaic must run without panicking.
    let raw = std::vec![123u8];
    let frame = BayerFrame::try_new(&raw, 1, 1, 1).unwrap();
    let mut rgb = std::vec![0u8; 3];
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
    // Single R-site (RGGB at (0,0) = R): output R = 123, G/B
    // averaged from the same sample = 123.
    assert_eq!(rgb, std::vec![123, 123, 123]);
  }

  /// Asserts the documented mirror-by-2 boundary contract: at the
  /// top edge `above` is `mid_row(1)`, at the bottom edge `below`
  /// is `mid_row(h - 2)`. A custom sink that captures the row
  /// borrows directly can verify this without re-running the
  /// kernel.
  #[test]
  fn bayer_walker_supplies_mirror_by_2_row_borrows() {
    /// Captures the first byte of `above` and `below` for each row.
    struct EdgeCapture {
      above_first: std::vec::Vec<u8>,
      below_first: std::vec::Vec<u8>,
    }
    impl PixelSink for EdgeCapture {
      type Input<'a> = BayerRow<'a>;
      type Error = Infallible;
      fn process(&mut self, row: BayerRow<'_>) -> Result<(), Self::Error> {
        self.above_first.push(row.above()[0]);
        self.below_first.push(row.below()[0]);
        Ok(())
      }
    }
    impl BayerSink for EdgeCapture {}

    // 4×4 plane where every row's first byte is the row index. So
    // mid_row(r)[0] == r, and mirror-by-2 should produce
    // above_first = [1, 0, 1, 2] and below_first = [1, 2, 3, 2].
    let raw: std::vec::Vec<u8> = (0..16u8).map(|i| i / 4).collect();
    let frame = BayerFrame::try_new(&raw, 4, 4, 4).unwrap();
    let mut sink = EdgeCapture {
      above_first: std::vec::Vec::new(),
      below_first: std::vec::Vec::new(),
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
    // row 0: above = mid_row(1), below = mid_row(1)
    // row 1: above = mid_row(0), below = mid_row(2)
    // row 2: above = mid_row(1), below = mid_row(3)
    // row 3: above = mid_row(2), below = mid_row(2)  (mirror-by-2)
    assert_eq!(sink.above_first, std::vec![1u8, 0, 1, 2]);
    assert_eq!(sink.below_first, std::vec![1u8, 2, 3, 2]);
  }

  /// Same contract test for `height < 2` — falls back to replicate
  /// (no mirror partner exists).
  #[test]
  fn bayer_walker_falls_back_to_replicate_when_height_below_2() {
    struct EdgeCapture {
      above_first: std::vec::Vec<u8>,
      below_first: std::vec::Vec<u8>,
    }
    impl PixelSink for EdgeCapture {
      type Input<'a> = BayerRow<'a>;
      type Error = Infallible;
      fn process(&mut self, row: BayerRow<'_>) -> Result<(), Self::Error> {
        self.above_first.push(row.above()[0]);
        self.below_first.push(row.below()[0]);
        Ok(())
      }
    }
    impl BayerSink for EdgeCapture {}

    let raw = std::vec![42u8; 4]; // 4 columns, 1 row
    let frame = BayerFrame::try_new(&raw, 4, 1, 4).unwrap();
    let mut sink = EdgeCapture {
      above_first: std::vec::Vec::new(),
      below_first: std::vec::Vec::new(),
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
    // h=1: replicate fallback. above = below = mid = 42.
    assert_eq!(sink.above_first, std::vec![42u8]);
    assert_eq!(sink.below_first, std::vec![42u8]);
  }
}
