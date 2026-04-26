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
