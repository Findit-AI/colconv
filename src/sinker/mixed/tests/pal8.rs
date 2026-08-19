use super::*;
use crate::{
  frame::Pal8Frame,
  raw::{Pal8, Pal8Row, pal8_to},
  sinker::MixedSinker,
};

// ---- Palette helpers -------------------------------------------------------

/// Returns a palette where every entry is `[b, g, r, a]` (FFmpeg PAL8 order).
fn solid_palette(b: u8, g: u8, r: u8, a: u8) -> [[u8; 4]; 256] {
  let mut p = [[0u8; 4]; 256];
  for entry in p.iter_mut() {
    *entry = [b, g, r, a];
  }
  p
}

/// Returns a palette with unique per-entry values so different index values
/// can be distinguished: entry `i` holds `[i as u8, i as u8, i as u8, 255]`.
fn identity_palette() -> [[u8; 4]; 256] {
  let mut p = [[0u8; 4]; 256];
  for (i, entry) in p.iter_mut().enumerate() {
    let v = i as u8;
    *entry = [v, v, v, 255];
  }
  p
}

/// Builds a valid 1-row `Pal8Frame` from a slice of pixel indices.
fn make_frame<'a>(indices: &'a [u8], palette: &'a [[u8; 4]; 256], width: u32) -> Pal8Frame<'a> {
  Pal8Frame::try_new(indices, palette, width, 1, width).unwrap()
}

// ---- Test 1: RGB channel order (BGRA → RGB) --------------------------------

#[test]
fn pal8_with_rgb_correct_channel_order() {
  // Palette entry 0: B=10, G=20, R=30, A=40.
  // All pixels index 0 → RGB output = [R=30, G=20, B=10].
  let palette = solid_palette(10, 20, 30, 40);
  let indices = [0u8; 4];
  let frame = make_frame(&indices, &palette, 4);
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Pal8>::new(4, 1).with_rgb(&mut rgb).unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgb[i * 3], 30, "px {i} R");
    assert_eq!(rgb[i * 3 + 1], 20, "px {i} G");
    assert_eq!(rgb[i * 3 + 2], 10, "px {i} B");
  }
}

// ---- Test 2: RGB drops alpha -----------------------------------------------

#[test]
fn pal8_with_rgb_drops_alpha() {
  // RGB output is 3 bytes/pixel regardless of palette alpha.
  let palette = solid_palette(0, 0, 0, 128);
  let indices = [0u8; 4];
  let frame = make_frame(&indices, &palette, 4);
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Pal8>::new(4, 1).with_rgb(&mut rgb).unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  // Buffer length must equal width * 3; no alpha byte present.
  assert_eq!(rgb.len(), 4 * 3);
  // All zero because B=0,G=0,R=0 → RGB=[0,0,0].
  assert!(rgb.iter().all(|&b| b == 0));
}

// ---- Test 3: RGBA passes alpha through from palette ------------------------

#[test]
fn pal8_with_rgba_passes_alpha() {
  // Palette entry 0: B=0, G=0, R=0, A=200.
  let palette = solid_palette(0, 0, 0, 200);
  let indices = [0u8; 4];
  let frame = make_frame(&indices, &palette, 4);
  let mut rgba = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Pal8>::new(4, 1).with_rgba(&mut rgba).unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgba[i * 4 + 3], 200, "px {i} alpha");
  }
}

// ---- Test 4: rgb_u16 full-range widening -----------------------------------

#[test]
fn pal8_with_rgb_u16_full_range() {
  // Palette entry 0: B=0, G=0, R=255, A=255.
  // rgb_u16 output: R=0xFFFF, G=0x0000, B=0x0000.
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [0, 0, 255, 255]; // [B, G, R, A]
  let indices = [0u8; 2];
  let frame = make_frame(&indices, &palette, 2);
  let mut rgb_u16 = std::vec![0u16; 2 * 3];
  let mut sink = MixedSinker::<Pal8>::new(2, 1)
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  // Pixel 0: R
  assert_eq!(rgb_u16[0], 0xFFFF, "R=255 → 0xFFFF");
  assert_eq!(rgb_u16[1], 0x0000, "G=0 → 0x0000");
  assert_eq!(rgb_u16[2], 0x0000, "B=0 → 0x0000");
  // Pixel 1: same values.
  assert_eq!(rgb_u16[3], 0xFFFF);
}

// ---- Test 5: rgba_u16 alpha widening ----------------------------------------

#[test]
fn pal8_with_rgba_u16_passes_alpha() {
  // Palette entry 0: B=0, G=0, R=0, A=128.
  // A=128 → (128<<8)|128 = 0x8080.
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [0, 0, 0, 128];
  let indices = [0u8; 1];
  let frame = make_frame(&indices, &palette, 1);
  let mut rgba_u16 = std::vec![0u16; 4];
  let mut sink = MixedSinker::<Pal8>::new(1, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  assert_eq!(rgba_u16[3], 0x8080, "A=128 → 0x8080");
}

// ---- Test 6: Strategy A+ — rgb + rgba combo --------------------------------

#[test]
fn pal8_rgb_and_rgba_combo_strategy_a_plus() {
  // Verify that the Strategy A+ combo path (one palette lookup) produces the
  // same output as two independent sinkers (one with_rgb, one with_rgba).
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [10, 20, 30, 200]; // B=10, G=20, R=30, A=200
  palette[1] = [50, 100, 150, 80]; // B=50, G=100, R=150, A=80

  let indices = [0u8, 1u8, 0u8, 1u8];
  let width = 4u32;
  let frame = Pal8Frame::try_new(&indices, &palette, width, 1, width).unwrap();

  // Combo sinker.
  let mut rgb_combo = std::vec![0u8; 4 * 3];
  let mut rgba_combo = std::vec![0u8; 4 * 4];
  let mut sink_combo = MixedSinker::<Pal8>::new(4, 1)
    .with_rgb(&mut rgb_combo)
    .unwrap()
    .with_rgba(&mut rgba_combo)
    .unwrap();
  pal8_to(&frame, &mut sink_combo).unwrap();

  // Independent rgb-only sinker.
  let mut rgb_only = std::vec![0u8; 4 * 3];
  let mut sink_rgb = MixedSinker::<Pal8>::new(4, 1)
    .with_rgb(&mut rgb_only)
    .unwrap();
  pal8_to(&frame, &mut sink_rgb).unwrap();

  // Independent rgba-only sinker.
  let mut rgba_only = std::vec![0u8; 4 * 4];
  let mut sink_rgba = MixedSinker::<Pal8>::new(4, 1)
    .with_rgba(&mut rgba_only)
    .unwrap();
  pal8_to(&frame, &mut sink_rgba).unwrap();

  assert_eq!(rgb_combo, rgb_only, "rgb combo must equal rgb-only");
  assert_eq!(rgba_combo, rgba_only, "rgba combo must equal rgba-only");
}

// ---- Test 7: u16 combo strategy A+ ----------------------------------------

#[test]
fn pal8_rgb_u16_and_rgba_u16_combo() {
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [10, 20, 30, 200];
  palette[1] = [50, 100, 150, 80];

  let indices = [0u8, 1u8, 0u8, 1u8];
  let width = 4u32;
  let frame = Pal8Frame::try_new(&indices, &palette, width, 1, width).unwrap();

  // Combo sinker.
  let mut rgb_u16_combo = std::vec![0u16; 4 * 3];
  let mut rgba_u16_combo = std::vec![0u16; 4 * 4];
  let mut sink_combo = MixedSinker::<Pal8>::new(4, 1)
    .with_rgb_u16(&mut rgb_u16_combo)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_combo)
    .unwrap();
  pal8_to(&frame, &mut sink_combo).unwrap();

  // Independent rgb_u16-only.
  let mut rgb_u16_only = std::vec![0u16; 4 * 3];
  let mut sink_rgb = MixedSinker::<Pal8>::new(4, 1)
    .with_rgb_u16(&mut rgb_u16_only)
    .unwrap();
  pal8_to(&frame, &mut sink_rgb).unwrap();

  // Independent rgba_u16-only.
  let mut rgba_u16_only = std::vec![0u16; 4 * 4];
  let mut sink_rgba = MixedSinker::<Pal8>::new(4, 1)
    .with_rgba_u16(&mut rgba_u16_only)
    .unwrap();
  pal8_to(&frame, &mut sink_rgba).unwrap();

  assert_eq!(
    rgb_u16_combo, rgb_u16_only,
    "rgb_u16 combo must equal rgb_u16-only"
  );
  assert_eq!(
    rgba_u16_combo, rgba_u16_only,
    "rgba_u16 combo must equal rgba_u16-only"
  );
}

// ---- Test 8: luma known value (BT.709) -------------------------------------

#[test]
fn pal8_with_luma_known_value() {
  // Pure red: R=255, G=0, B=0. Palette entry 0: [B=0, G=0, R=255, A=255].
  // BT.709 Q8 coefficients: (cr=54, cg=183, cb=19).
  // luma = (54*255 + 183*0 + 19*0 + 128) >> 8 = (13770 + 128) >> 8 = 13898 >> 8 = 54.
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [0, 0, 255, 255]; // B=0, G=0, R=255
  let indices = [0u8; 4];
  let frame = make_frame(&indices, &palette, 4);
  let mut luma = std::vec![0u8; 4];
  let mut sink = MixedSinker::<Pal8>::new(4, 1).with_luma(&mut luma).unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  // BT.709: Y = (54*R + 183*G + 19*B + 128) >> 8 = (13770 + 128) >> 8 = 54.
  assert_eq!(luma[0], 54, "BT.709 luma for pure red = 54");
  assert!(luma.iter().all(|&y| y == 54), "all pixels luma == 54");
}

// ---- Test 9: luma_u16 widening ---------------------------------------------

#[test]
fn pal8_with_luma_u16_widening() {
  // Choose a palette entry that gives luma ≈ 128.
  // To get exact luma=128: need (54*R + 183*G + 19*B + 128) >> 8 == 128.
  // Use all-grey: R=G=B=128. luma = (54*128 + 183*128 + 19*128 + 128) >> 8
  //   = (6912 + 23424 + 2432 + 128) >> 8 = 32896 >> 8 = 128.
  // Palette [B=128, G=128, R=128, A=255].
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [128, 128, 128, 255];
  let indices = [0u8; 1];
  let frame = make_frame(&indices, &palette, 1);
  let mut luma_u16 = std::vec![0u16; 1];
  let mut sink = MixedSinker::<Pal8>::new(1, 1)
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  // luma = 128 → luma_u16 = (128 << 8) | 128 = 0x8080.
  assert_eq!(luma_u16[0], 0x8080, "luma=128 → 0x8080");
}

// ---- Test 10: HSV for saturated red ----------------------------------------

#[test]
fn pal8_with_hsv_saturated_red() {
  // Pure red R=255, G=0, B=0. Palette [B=0, G=0, R=255, A=255].
  // Expected: H≈0 (wraps around; within ±1 of 0), S=255, V=255.
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [0, 0, 255, 255];
  let indices = [0u8; 1];
  let frame = make_frame(&indices, &palette, 1);
  let mut h = std::vec![0u8; 1];
  let mut s = std::vec![0u8; 1];
  let mut v = std::vec![0u8; 1];
  let mut sink = MixedSinker::<Pal8>::new(1, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  // For pure red: S and V must be 255; H is at the red wrap-around.
  assert_eq!(s[0], 255, "S=255 for saturated red");
  assert_eq!(v[0], 255, "V=255 for max-value red");
  // H is in [0,255]; pure red sits at H≈0 or H≈255 depending on convention.
  // Accept both ends of the range (within ±1).
  assert!(
    h[0] <= 1 || h[0] >= 254,
    "H for pure red should be near 0 or 255, got {}",
    h[0]
  );
}

// ---- Test 11: walker row index is ascending --------------------------------

#[test]
fn pal8_walker_row_index_ascending() {
  // Stub sink that records row indices.
  use crate::{PixelSink, raw::Pal8Sink};

  struct IndexRecorder {
    indices: std::vec::Vec<usize>,
  }
  impl PixelSink for IndexRecorder {
    type Input<'r> = Pal8Row<'r>;
    type Error = std::convert::Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Self::Error> {
      Ok(())
    }
    fn process(&mut self, row: Pal8Row<'_>) -> Result<(), Self::Error> {
      self.indices.push(row.idx());
      Ok(())
    }
  }
  impl Pal8Sink for IndexRecorder {}

  let palette = identity_palette();
  let (w, h) = (4u32, 5u32);
  let indices_data = std::vec![0u8; (w * h) as usize];
  let frame = Pal8Frame::try_new(&indices_data, &palette, w, h, w).unwrap();
  let mut rec = IndexRecorder {
    indices: std::vec::Vec::new(),
  };
  pal8_to(&frame, &mut rec).unwrap();

  let expected: std::vec::Vec<usize> = (0..h as usize).collect();
  assert_eq!(
    rec.indices, expected,
    "row indices must be 0..height in order"
  );
}

// ---- Test 12: walker stride slices correctly --------------------------------

#[test]
fn pal8_walker_stride_slices_correctly() {
  // Frame with stride > width: walker must deliver rows of `width` elements.
  use crate::{PixelSink, raw::Pal8Sink};

  struct RowLenRecorder {
    lengths: std::vec::Vec<usize>,
  }
  impl PixelSink for RowLenRecorder {
    type Input<'r> = Pal8Row<'r>;
    type Error = std::convert::Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Self::Error> {
      Ok(())
    }
    fn process(&mut self, row: Pal8Row<'_>) -> Result<(), Self::Error> {
      self.lengths.push(row.row().len());
      Ok(())
    }
  }
  impl Pal8Sink for RowLenRecorder {}

  let palette = identity_palette();
  let (w, h, stride) = (4u32, 3u32, 8u32);
  let indices_data = std::vec![0u8; (stride * h) as usize];
  let frame = Pal8Frame::try_new(&indices_data, &palette, w, h, stride).unwrap();
  let mut rec = RowLenRecorder {
    lengths: std::vec::Vec::new(),
  };
  pal8_to(&frame, &mut rec).unwrap();

  assert_eq!(rec.lengths.len(), h as usize, "one call per row");
  for (i, &len) in rec.lengths.iter().enumerate() {
    assert_eq!(len, w as usize, "row {i}: len must equal width");
  }
}

// ---- Test 13: error path — RowShapeMismatch --------------------------------

#[test]
fn pal8_error_row_shape_mismatch() {
  use crate::sinker::mixed::MixedSinkerError;

  let palette = identity_palette();
  let width = 4usize;
  let mut rgb = std::vec![0u8; width * 3];
  let mut sink = MixedSinker::<Pal8>::new(width, 1)
    .with_rgb(&mut rgb)
    .unwrap();

  // `begin_frame` must succeed with matching dimensions.
  sink.begin_frame(width as u32, 1).unwrap();

  // Construct a row with the wrong length (2 instead of 4).
  let wrong_indices = [0u8; 2];
  let row = Pal8Row::for_tests(&wrong_indices, &palette, 0);
  let err = sink.process(row).unwrap_err();

  match err {
    MixedSinkerError::RowShapeMismatch(e) => {
      assert!(
        e.which().is_pal_8_index_row(),
        "which must be Pal8IndexRow, got {}",
        e.which()
      );
      assert_eq!(e.expected(), width);
      assert_eq!(e.actual(), 2);
    }
    other => panic!("expected RowShapeMismatch, got {other:?}"),
  }
}
