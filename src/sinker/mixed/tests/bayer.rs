use super::*;

// ---- Bayer + Bayer16 MixedSinker integration tests ----------------------

/// Build a solid-channel RGGB Bayer plane (8-bit) so every R site
/// holds `r`, every B site holds `b`, and both G sites hold `g`.
fn solid_rggb8(width: u32, height: u32, r: u8, g: u8, b: u8) -> std::vec::Vec<u8> {
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

/// Build a 12-bit low-packed RGGB Bayer plane.
fn solid_rggb12(width: u32, height: u32, r: u16, g: u16, b: u16) -> std::vec::Vec<u16> {
  let w = width as usize;
  let h = height as usize;
  let mut data = std::vec![0u16; w * h];
  for y in 0..h {
    for x in 0..w {
      let v = match (y & 1, x & 1) {
        (0, 0) => r,
        (0, 1) => g,
        (1, 0) => g,
        (1, 1) => b,
        _ => unreachable!(),
      };
      data[y * w + x] = v;
    }
  }
  data
}

#[test]
fn bayer_mixed_sinker_with_rgb_red_interior() {
  use crate::{
    frame::BayerFrame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb8(w, h, 255, 0, 0);
  let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap();
  bayer_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  // Interior should be exactly red.
  let wu = w as usize;
  for y in 0..(h as usize) {
    for x in 0..wu {
      let i = (y * wu + x) * 3;
      assert_eq!(rgb[i], 255, "px ({x},{y}) R");
      assert_eq!(rgb[i + 1], 0, "px ({x},{y}) G");
      assert_eq!(rgb[i + 2], 0, "px ({x},{y}) B");
    }
  }
}

#[test]
fn bayer_mixed_sinker_with_luma_uniform_byte() {
  use crate::{
    frame::BayerFrame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  };
  // Uniform byte → uniform RGB → uniform luma at the same value.
  let (w, h) = (8u32, 6u32);
  let raw = std::vec![200u8; (w * h) as usize];
  let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
  let mut luma = std::vec![0u8; (w * h) as usize];
  let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
    .with_luma(&mut luma)
    .unwrap();
  bayer_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  // BT.709 luma of (200, 200, 200) = 200 (within 1 LSB rounding).
  for &y in &luma {
    assert!((y as i32 - 200).abs() <= 1, "luma got {y}");
  }
}

#[test]
fn bayer_mixed_sinker_with_hsv_solid_red_interior() {
  use crate::{
    frame::BayerFrame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb8(w, h, 255, 0, 0);
  let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
  let mut hh = std::vec![0u8; (w * h) as usize];
  let mut ss = std::vec![0u8; (w * h) as usize];
  let mut vv = std::vec![0u8; (w * h) as usize];
  let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
  bayer_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  // Pure red at interior → H = 0 (red), S = 255 (max), V = 255.
  let wu = w as usize;
  for y in 0..(h as usize) {
    for x in 0..wu {
      let i = y * wu + x;
      assert_eq!(hh[i], 0, "px ({x},{y}) H");
      assert_eq!(ss[i], 255, "px ({x},{y}) S");
      assert_eq!(vv[i], 255, "px ({x},{y}) V");
    }
  }
}

#[test]
fn bayer16_mixed_sinker_with_rgb_red_interior() {
  use crate::{
    frame::Bayer12Frame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb12(w, h, 4095, 0, 0);
  let raw = raw
    .iter()
    .map(|&v| le_wire_u16(v))
    .collect::<std::vec::Vec<u16>>();
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap();
  bayer16_to::<12, _>(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  let wu = w as usize;
  for y in 0..(h as usize) {
    for x in 0..wu {
      let i = (y * wu + x) * 3;
      assert_eq!(rgb[i], 255, "px ({x},{y}) R");
      assert_eq!(rgb[i + 1], 0, "px ({x},{y}) G");
      assert_eq!(rgb[i + 2], 0, "px ({x},{y}) B");
    }
  }
}

#[test]
fn bayer16_mixed_sinker_with_rgb_u16_red_interior() {
  use crate::{
    frame::Bayer12Frame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb12(w, h, 4095, 0, 0);
  let raw = raw
    .iter()
    .map(|&v| le_wire_u16(v))
    .collect::<std::vec::Vec<u16>>();
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u16; (w * h * 3) as usize];
  let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  bayer16_to::<12, _>(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  // Low-packed 12-bit white = 4095 at interior.
  let wu = w as usize;
  for y in 0..(h as usize) {
    for x in 0..wu {
      let i = (y * wu + x) * 3;
      assert_eq!(rgb[i], 4095, "px ({x},{y}) R");
      assert_eq!(rgb[i + 1], 0, "px ({x},{y}) G");
      assert_eq!(rgb[i + 2], 0, "px ({x},{y}) B");
    }
  }
}

#[test]
fn bayer16_mixed_sinker_dual_rgb_and_rgb_u16() {
  use crate::{
    frame::Bayer12Frame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
  };
  // Both u8 RGB and u16 RGB attached — both kernels run.
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb12(w, h, 4095, 0, 0);
  let raw = raw
    .iter()
    .map(|&v| le_wire_u16(v))
    .collect::<std::vec::Vec<u16>>();
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb_u8 = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb_u16 = std::vec![0u16; (w * h * 3) as usize];
  let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  bayer16_to::<12, _>(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  let wu = w as usize;
  for y in 0..(h as usize) {
    for x in 0..wu {
      let i = (y * wu + x) * 3;
      assert_eq!(rgb_u8[i], 255);
      assert_eq!(rgb_u16[i], 4095);
    }
  }
}

#[test]
fn bayer_mixed_sinker_returns_row_shape_mismatch_on_bad_above() {
  use crate::raw::{BayerDemosaic, BayerPattern, BayerRow};
  let mut rgb = std::vec![0u8; 8 * 6 * 3];
  let mut sinker = MixedSinker::<Bayer>::new(8, 6).with_rgb(&mut rgb).unwrap();
  sinker.begin_frame(8, 6).unwrap();
  let mid = std::vec![0u8; 8];
  let below = std::vec![0u8; 8];
  let bad_above = std::vec![0u8; 7]; // wrong length
  let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
  let row = BayerRow::for_tests(
    &bad_above,
    &mid,
    &below,
    0,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    m,
  );
  let err = sinker.process(row).unwrap_err();
  assert!(matches!(
    &err,
    MixedSinkerError::RowShapeMismatch(e)
      if matches!(e.which(), RowSlice::BayerAbove) && e.expected() == 8 && e.actual() == 7
  ));
}

#[test]
fn bayer16_mixed_sinker_returns_row_shape_mismatch_on_bad_mid() {
  use crate::raw::{BayerDemosaic, BayerPattern, BayerRow16};
  let mut rgb = std::vec![0u8; 8 * 6 * 3];
  let mut sinker = MixedSinker::<Bayer16<12>>::new(8, 6)
    .with_rgb(&mut rgb)
    .unwrap();
  sinker.begin_frame(8, 6).unwrap();
  let above = std::vec![0u16; 8];
  let bad_mid = std::vec![0u16; 7]; // wrong length
  let below = std::vec![0u16; 8];
  let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
  let row = BayerRow16::<12>::for_tests(
    &above,
    &bad_mid,
    &below,
    0,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    m,
  );
  let err = sinker.process(row).unwrap_err();
  assert!(matches!(
    &err,
    MixedSinkerError::RowShapeMismatch(e)
      if matches!(e.which(), RowSlice::Bayer16Mid) && e.expected() == 8 && e.actual() == 7
  ));
}

// ---- Bayer luma-coefficients tests --------------------------------------
//
// Cover the gap that earlier `bayer_mixed_sinker_with_luma_uniform_byte`
// missed: every coefficient set agrees on gray, so a hard-coded BT.709
// path could go undetected. The non-gray cases below force the rows
// apart — solid red goes through `cr` only, so each variant produces a
// distinct luma value.

/// Resolve a [`LumaCoefficients`] preset and run a solid-red 8-bit
/// Bayer frame through it; return the `cr` actually applied.
fn bayer8_solid_red_luma(coeffs: LumaCoefficients) -> u8 {
  use crate::{
    frame::BayerFrame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb8(w, h, 255, 0, 0);
  let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
  let mut luma = std::vec![0u8; (w * h) as usize];
  let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_coefficients(coeffs);
  bayer_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  let center = luma[(h as usize / 2) * (w as usize) + (w as usize / 2)];
  for (i, &y) in luma.iter().enumerate() {
    assert_eq!(
      y, center,
      "luma not uniform at idx {i}: {y} vs center {center}"
    );
  }
  center
}

#[test]
fn bayer_with_luma_coefficients_solid_red_differs_by_preset() {
  // Solid red after demosaic is `(255, 0, 0)` everywhere
  // (`bayer_mixed_sinker_with_rgb_red_interior` proves this).
  // Luma reduces to `(cr * 255 + 128) >> 8` for each preset, so
  // each coefficient set must produce a different value. A
  // hard-coded BT.709 implementation would make these all 54.
  let bt709 = bayer8_solid_red_luma(LumaCoefficients::Bt709);
  let bt2020 = bayer8_solid_red_luma(LumaCoefficients::Bt2020);
  let bt601 = bayer8_solid_red_luma(LumaCoefficients::Bt601);
  let dcip3 = bayer8_solid_red_luma(LumaCoefficients::DciP3);
  let aces = bayer8_solid_red_luma(LumaCoefficients::AcesAp1);

  assert_eq!(bt709, 54, "BT.709 red luma");
  assert_eq!(bt2020, 67, "BT.2020 red luma");
  assert_eq!(bt601, 77, "BT.601 red luma");
  assert_eq!(dcip3, 59, "DCI-P3 red luma");
  assert_eq!(aces, 70, "ACES AP1 red luma");

  // Distinct values guard against silent collapse to the default.
  let mut all = std::vec![bt709, bt2020, bt601, dcip3, aces];
  all.sort_unstable();
  all.dedup();
  assert_eq!(all.len(), 5, "presets collapsed to fewer values: {all:?}");
}

#[test]
fn bayer_with_luma_coefficients_custom_round_trips_to_q8() {
  // Custom weights `(1.0, 0.0, 0.0)` → Q8 `(256, 0, 0)`. Solid red
  // 255 then reduces to `(256 * 255 + 128) >> 8 = 255` (clamped).
  let custom = LumaCoefficients::try_custom(1.0, 0.0, 0.0).unwrap();
  let red = bayer8_solid_red_luma(custom);
  assert_eq!(red, 255, "Custom (1.0, 0.0, 0.0) on red 255 → 255");
}

#[test]
fn bayer_with_luma_coefficients_default_is_bt709() {
  // No `with_luma_coefficients` call → default (BT.709). Same red
  // input must produce the BT.709 value (54). This pins the
  // public default so a future refactor can't silently change it.
  use crate::{
    frame::BayerFrame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb8(w, h, 255, 0, 0);
  let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
  let mut luma = std::vec![0u8; (w * h) as usize];
  let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
    .with_luma(&mut luma)
    .unwrap();
  bayer_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  for (i, &y) in luma.iter().enumerate() {
    assert_eq!(y, 54, "default red luma at idx {i}");
  }
  assert_eq!(LumaCoefficients::default(), LumaCoefficients::Bt709);
}

#[test]
fn bayer_with_luma_coefficients_uniform_gray_invariant() {
  // The reverse of the above: gray content *must* be invariant
  // under any preset (this is the property the original
  // `*_with_luma_uniform_byte` test relied on, and the reason
  // the hard-coded BT.709 bug was invisible there).
  use crate::{
    frame::BayerFrame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = std::vec![200u8; (w * h) as usize];
  let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
  let presets = [
    LumaCoefficients::Bt709,
    LumaCoefficients::Bt2020,
    LumaCoefficients::Bt601,
    LumaCoefficients::DciP3,
    LumaCoefficients::AcesAp1,
  ];
  for preset in presets {
    let mut luma = std::vec![0u8; (w * h) as usize];
    let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_coefficients(preset);
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    for &y in &luma {
      assert!(
        (y as i32 - 200).abs() <= 1,
        "{preset:?} on gray 200 → {y} (expected ~200)"
      );
    }
  }
}

#[test]
fn bayer16_with_luma_coefficients_solid_red_differs_by_preset() {
  // Mirror of the 8-bit test for the high-bit-depth path
  // (`MixedSinker<Bayer16<BITS>>`). 12-bit white = 4095 →
  // demosaic produces `(255, 0, 0)` u8 RGB after CCM identity
  // and right-shift to u8 (the bayer16→u8 path reduces samples
  // before the luma kernel).
  use crate::{
    frame::Bayer12Frame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb12(w, h, 4095, 0, 0);
  let raw = raw
    .iter()
    .map(|&v| le_wire_u16(v))
    .collect::<std::vec::Vec<u16>>();
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();

  let run = |coeffs: LumaCoefficients| -> u8 {
    let mut luma = std::vec![0u8; (w * h) as usize];
    let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_coefficients(coeffs);
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    let center = luma[(h as usize / 2) * (w as usize) + (w as usize / 2)];
    for (i, &y) in luma.iter().enumerate() {
      assert_eq!(y, center, "luma not uniform at idx {i}");
    }
    center
  };

  let bt709 = run(LumaCoefficients::Bt709);
  let bt2020 = run(LumaCoefficients::Bt2020);
  let bt601 = run(LumaCoefficients::Bt601);
  let dcip3 = run(LumaCoefficients::DciP3);
  let aces = run(LumaCoefficients::AcesAp1);

  assert_eq!(bt709, 54, "BT.709 red luma (Bayer16<12>)");
  assert_eq!(bt2020, 67, "BT.2020 red luma (Bayer16<12>)");
  assert_eq!(bt601, 77, "BT.601 red luma (Bayer16<12>)");
  assert_eq!(dcip3, 59, "DCI-P3 red luma (Bayer16<12>)");
  assert_eq!(aces, 70, "ACES AP1 red luma (Bayer16<12>)");

  let mut all = std::vec![bt709, bt2020, bt601, dcip3, aces];
  all.sort_unstable();
  all.dedup();
  assert_eq!(all.len(), 5, "Bayer16 presets collapsed: {all:?}");
}

#[test]
fn luma_coefficients_to_q8_presets_sum_to_256() {
  // Round-to-nearest of the published weights for each preset
  // must still sum to exactly 256 — the rgb_row_to_luma_row
  // kernel divides by 256 implicitly via `>> 8`, so any preset
  // that drifts from 256 produces a brightness-scaled luma plane.
  for preset in [
    LumaCoefficients::Bt709,
    LumaCoefficients::Bt2020,
    LumaCoefficients::Bt601,
    LumaCoefficients::DciP3,
    LumaCoefficients::AcesAp1,
  ] {
    let (cr, cg, cb) = preset.to_q8();
    assert_eq!(cr + cg + cb, 256, "{preset:?} Q8 weights don't sum to 256");
  }
}

// ---- CustomLumaCoefficients validation tests ----------------------------
//
// The kernel multiplies these weights into a `u32` accumulator
// after a saturating `f32 → u32` cast. Without validation, NaN
// / negative / ±∞ / very-large finite weights would silently
// corrupt every Bayer luma plane (NaN → 0, +∞ → u32::MAX,
// negative → 0, large finite → debug-panic on multiply or
// wrapping in release). `try_new` rejects all four classes
// upfront so the kernel can stay branchless.

#[test]
fn custom_luma_coefficients_accepts_valid_weights() {
  // Standard BT.709 weights pass through cleanly.
  let c = CustomLumaCoefficients::try_new(0.2126, 0.7152, 0.0722).unwrap();
  assert_eq!(c.r(), 0.2126);
  assert_eq!(c.g(), 0.7152);
  assert_eq!(c.b(), 0.0722);

  // Zeroes are allowed (zero a channel out — degenerate but valid).
  let z = CustomLumaCoefficients::try_new(0.0, 1.0, 0.0).unwrap();
  assert_eq!(z.r(), 0.0);

  // Boundary: exactly `MAX_COEFFICIENT` is allowed (`<=`, not `<`).
  let edge =
    CustomLumaCoefficients::try_new(CustomLumaCoefficients::MAX_COEFFICIENT, 0.0, 0.0).unwrap();
  assert_eq!(edge.r(), CustomLumaCoefficients::MAX_COEFFICIENT);
}

#[test]
fn custom_luma_coefficients_rejects_nan() {
  for (channel, r, g, b) in [
    (LumaChannel::R, f32::NAN, 1.0, 0.0),
    (LumaChannel::G, 0.0, f32::NAN, 0.0),
    (LumaChannel::B, 0.5, 0.5, f32::NAN),
  ] {
    let err = CustomLumaCoefficients::try_new(r, g, b).unwrap_err();
    assert!(
      matches!(err, LumaCoefficientsError::NonFinite { channel: ch, .. } if ch == channel),
      "expected NonFinite for {channel:?}, got {err:?}"
    );
  }
}

#[test]
fn custom_luma_coefficients_rejects_infinity() {
  // Both +∞ and -∞ caught by `is_finite`. The earlier
  // `as u32` saturating cast would turn +∞ into `u32::MAX`,
  // overflowing `cr * 255` in debug builds.
  for inf in [f32::INFINITY, f32::NEG_INFINITY] {
    let err_r = CustomLumaCoefficients::try_new(inf, 0.0, 0.0).unwrap_err();
    let err_g = CustomLumaCoefficients::try_new(0.0, inf, 0.0).unwrap_err();
    let err_b = CustomLumaCoefficients::try_new(0.0, 0.0, inf).unwrap_err();
    for (err, channel) in [
      (err_r, LumaChannel::R),
      (err_g, LumaChannel::G),
      (err_b, LumaChannel::B),
    ] {
      assert!(
        matches!(err, LumaCoefficientsError::NonFinite { channel: ch, .. } if ch == channel),
        "expected NonFinite for {channel:?} with inf={inf}, got {err:?}"
      );
    }
  }
}

#[test]
fn custom_luma_coefficients_rejects_negative() {
  for (channel, r, g, b) in [
    (LumaChannel::R, -0.001, 1.0, 0.0),
    (LumaChannel::G, 0.0, -1.0, 0.0),
    (LumaChannel::B, 0.5, 0.5, -42.0),
  ] {
    let err = CustomLumaCoefficients::try_new(r, g, b).unwrap_err();
    assert!(
      matches!(err, LumaCoefficientsError::Negative { channel: ch, .. } if ch == channel),
      "expected Negative for {channel:?}, got {err:?}"
    );
  }
}

#[test]
fn custom_luma_coefficients_rejects_oversized() {
  let over = CustomLumaCoefficients::MAX_COEFFICIENT + 1.0;
  for (channel, r, g, b) in [
    (LumaChannel::R, over, 0.0, 0.0),
    (LumaChannel::G, 0.0, over, 0.0),
    (LumaChannel::B, 0.0, 0.0, over),
  ] {
    let err = CustomLumaCoefficients::try_new(r, g, b).unwrap_err();
    assert!(
      matches!(
        err,
        LumaCoefficientsError::OutOfBounds { channel: ch, .. } if ch == channel
      ),
      "expected OutOfBounds for {channel:?}, got {err:?}"
    );
  }

  // Pathological value that previously caused saturation:
  // `1e9_f32 * 256.0 ≈ 2.56e11` saturates `as u32` to
  // `u32::MAX`, then `cr * 255` overflows.
  let err = CustomLumaCoefficients::try_new(1.0e9, 0.0, 0.0).unwrap_err();
  assert!(matches!(err, LumaCoefficientsError::OutOfBounds { .. }));
}

#[test]
fn luma_coefficients_try_custom_routes_through_validation() {
  // Convenience constructor surfaces the same errors as
  // `CustomLumaCoefficients::try_new` and yields the wrapped
  // variant on success.
  let ok = LumaCoefficients::try_custom(0.5, 0.4, 0.1).unwrap();
  assert!(ok.is_custom());

  let err = LumaCoefficients::try_custom(f32::NAN, 0.0, 0.0).unwrap_err();
  assert!(matches!(err, LumaCoefficientsError::NonFinite { .. }));
}

#[test]
#[should_panic(expected = "invalid CustomLumaCoefficients")]
fn custom_luma_coefficients_new_panics_on_invalid() {
  // The `::new` and `LumaCoefficients::custom` panicking
  // constructors are intended for compile-time-known weights;
  // hostile input must blow up loudly, not silently corrupt
  // downstream luma.
  let _ = CustomLumaCoefficients::new(f32::NAN, 0.0, 0.0);
}

#[test]
fn custom_luma_coefficients_at_max_does_not_overflow_kernel() {
  // End-to-end proof that `MAX_COEFFICIENT` is conservative:
  // even worst-case (all three channels at max, all pixels at
  // 255) the per-row accumulator stays well under `u32::MAX`,
  // and the final `>> 8 / .min(255)` clamps cleanly to 255.
  use crate::{
    frame::BayerFrame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  };
  let (w, h) = (8u32, 6u32);
  let raw = std::vec![255u8; (w * h) as usize];
  let frame = BayerFrame::try_new(&raw, w, h, w).unwrap();
  let mut luma = std::vec![0u8; (w * h) as usize];
  let max = CustomLumaCoefficients::MAX_COEFFICIENT;
  let mut sinker = MixedSinker::<Bayer>::new(w as usize, h as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_coefficients(LumaCoefficients::try_custom(max, max, max).unwrap());
  bayer_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  for &y in &luma {
    assert_eq!(
      y, 255,
      "max-weight saturated luma should clamp to 255, got {y}"
    );
  }
}

// ---- Bayer16 big-endian (BE) tests --------------------------------------
//
// `Bayer16<BITS, true>` (the `Bayer{10,12,14}Be` / `Bayer16BitBe`
// aliases) reads BE-encoded `u16` plane samples; the kernel byte-swaps
// each sample before the demosaic, so a BE plane carrying the *same
// logical samples* as an LE plane must produce *identical* RGB. These
// tests build the BE plane host-independently (`to_be_bytes` →
// `from_ne_bytes`) so they assert real byte-order handling on both
// little- and big-endian hosts, and pin that the LE path is unchanged.

/// Reinterpret a logical low-packed `u16` as the `u16` element a BE
/// wire plane would store: serialize big-endian, then read back in the
/// host's native order. On an LE host this yields the byte-swapped
/// value; on a BE host it is the identity. Host-independent by
/// construction — mirrors the y216 / p2xx BE test convention.
fn be_wire_u16(logical: u16) -> u16 {
  u16::from_ne_bytes(logical.to_be_bytes())
}

/// LE-wire twin of [`be_wire_u16`]: serialize little-endian, then read
/// back in the host's native order. On an LE host it is the identity;
/// on a BE host it byte-swaps. Encoding the LE-oracle plane through
/// this makes the oracle host-independent — the `from_le` wire decode
/// inside the LE frame recovers the original logical samples on every
/// host, instead of mis-reading a host-native plane as LE.
fn le_wire_u16(logical: u16) -> u16 {
  u16::from_ne_bytes(logical.to_le_bytes())
}

/// Build a 12-bit low-packed Bayer plane for an arbitrary `pattern`,
/// placing `r` at R sites, `b` at B sites, and `g` at both G sites.
fn solid_pattern12(
  pattern: crate::raw::BayerPattern,
  width: u32,
  height: u32,
  r: u16,
  g: u16,
  b: u16,
) -> std::vec::Vec<u16> {
  use crate::raw::BayerPattern::*;
  // (R-site, B-site) parities — must match the kernel's `pattern_phases`.
  let (r_par, b_par) = match pattern {
    Rggb => ((0usize, 0usize), (1usize, 1usize)),
    Bggr => ((1, 1), (0, 0)),
    Grbg => ((0, 1), (1, 0)),
    Gbrg => ((1, 0), (0, 1)),
  };
  let w = width as usize;
  let h = height as usize;
  let mut data = std::vec![0u16; w * h];
  for y in 0..h {
    for x in 0..w {
      let par = (y & 1, x & 1);
      data[y * w + x] = if par == r_par {
        r
      } else if par == b_par {
        b
      } else {
        g
      };
    }
  }
  data
}

/// LE 12-bit RGB (u8 out) for the given logical plane, via the
/// `Bayer16<12, false>` path. The oracle the BE output must match.
fn le12_rgb_u8(
  pattern: crate::raw::BayerPattern,
  w: u32,
  h: u32,
  plane: &[u16],
) -> std::vec::Vec<u8> {
  use crate::{
    frame::Bayer12Frame,
    raw::{BayerDemosaic, ColorCorrectionMatrix, WhiteBalance, bayer16_to},
  };
  let frame = Bayer12Frame::try_new(plane, w, h, w).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap();
  bayer16_to::<12, _>(
    &frame,
    pattern,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sinker,
  )
  .unwrap();
  rgb
}

/// A BE 12-bit plane carrying the same logical samples as its LE twin
/// produces byte-identical RGB (u8 out), for every Bayer pattern. The
/// byte-swap is load-bearing: an asymmetric `g` (`0x0ABC`, whose bytes
/// differ) would demosaic to a wildly different value if the kernel
/// skipped the swap.
#[test]
fn bayer16_be_matches_le_u8_all_patterns() {
  use crate::{
    frame::Bayer12BeFrame,
    raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to_endian},
  };
  let (w, h) = (8u32, 6u32);
  // Asymmetric logical samples so the byte order is observable.
  let (r, g, b) = (0x0FFFu16, 0x0ABCu16, 0x0123u16);
  for pattern in [
    BayerPattern::Rggb,
    BayerPattern::Bggr,
    BayerPattern::Grbg,
    BayerPattern::Gbrg,
  ] {
    // Build the logical samples once, then derive both wire planes from
    // them: the LE oracle via `le_wire_u16`, the BE input via
    // `be_wire_u16`. Both decode back to `logical` on any host.
    let logical = solid_pattern12(pattern, w, h, r, g, b);
    let le_plane: std::vec::Vec<u16> = logical.iter().map(|&v| le_wire_u16(v)).collect();
    let expected = le12_rgb_u8(pattern, w, h, &le_plane);

    let be_plane: std::vec::Vec<u16> = logical.iter().map(|&v| be_wire_u16(v)).collect();
    let frame = Bayer12BeFrame::try_new(&be_plane, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sinker = MixedSinker::<Bayer16<12, true>>::new(w as usize, h as usize)
      .with_rgb(&mut rgb)
      .unwrap();
    bayer16_to_endian::<12, true, _>(
      &frame,
      pattern,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sinker,
    )
    .unwrap();
    assert_eq!(rgb, expected, "BE/LE RGB mismatch for pattern {pattern:?}");
  }
}

/// Same parity check for the native-depth `u16` output path.
#[test]
fn bayer16_be_matches_le_u16_output() {
  use crate::{
    frame::{Bayer12BeFrame, Bayer12Frame},
    raw::{
      BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to,
      bayer16_to_endian,
    },
  };
  let (w, h) = (8u32, 6u32);
  let (r, g, b) = (0x0FFFu16, 0x0ABCu16, 0x0123u16);
  // Shared logical samples; both wire planes derive from them.
  let logical = solid_pattern12(BayerPattern::Rggb, w, h, r, g, b);
  let le_plane: std::vec::Vec<u16> = logical.iter().map(|&v| le_wire_u16(v)).collect();

  // LE oracle (u16 out).
  let le_frame = Bayer12Frame::try_new(&le_plane, w, h, w).unwrap();
  let mut le_rgb = std::vec![0u16; (w * h * 3) as usize];
  let mut le_sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
    .with_rgb_u16(&mut le_rgb)
    .unwrap();
  bayer16_to::<12, _>(
    &le_frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut le_sinker,
  )
  .unwrap();

  // BE under test (u16 out) — output is host-native, only the input
  // wire order differs.
  let be_plane: std::vec::Vec<u16> = logical.iter().map(|&v| be_wire_u16(v)).collect();
  let be_frame = Bayer12BeFrame::try_new(&be_plane, w, h, w).unwrap();
  let mut be_rgb = std::vec![0u16; (w * h * 3) as usize];
  let mut be_sinker = MixedSinker::<Bayer16<12, true>>::new(w as usize, h as usize)
    .with_rgb_u16(&mut be_rgb)
    .unwrap();
  bayer16_to_endian::<12, true, _>(
    &be_frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut be_sinker,
  )
  .unwrap();

  assert_eq!(be_rgb, le_rgb, "BE/LE u16 RGB mismatch");
}

/// Proves the byte-swap actually fires: feeding the *BE-encoded* bytes
/// through the LE marker (the wrong byte order) yields different output
/// than the correct BE path, given asymmetric samples. If the BE path
/// silently read host-native (no swap), the two would coincide.
#[test]
fn bayer16_be_byte_swap_is_load_bearing() {
  use crate::{
    frame::{Bayer12BeFrame, Bayer12Frame},
    raw::{
      BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to,
      bayer16_to_endian,
    },
  };
  let (w, h) = (8u32, 6u32);
  // Logical samples bounded to `< 0x10` so that the *same raw BE bytes*
  // read back through the LE marker (i.e. `logical << 8`) also stay
  // inside the 12-bit low-packed range — both frame constructors must
  // accept the plane for the comparison to be meaningful.
  let (r, g, b) = (0x00Fu16, 0x00Au16, 0x003u16);

  // Shared logical samples; the BE plane derives from them host-
  // independently. The "wrong" reading below reinterprets the *same*
  // BE bytes through the LE marker — so it must stay the be_plane, not
  // an le_wire-encoded oracle, for the divergence to be meaningful.
  let logical = solid_pattern12(BayerPattern::Rggb, w, h, r, g, b);

  // Correct BE path.
  let be_plane: std::vec::Vec<u16> = logical.iter().map(|&v| be_wire_u16(v)).collect();
  let be_frame = Bayer12BeFrame::try_new(&be_plane, w, h, w).unwrap();
  let mut be_rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut be_sinker = MixedSinker::<Bayer16<12, true>>::new(w as usize, h as usize)
    .with_rgb(&mut be_rgb)
    .unwrap();
  bayer16_to_endian::<12, true, _>(
    &be_frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut be_sinker,
  )
  .unwrap();

  // Same raw bytes, but interpreted by the LE marker (wrong order).
  // Each sample reads as `v << 8` (bounded < 4096 by `v < 0x10`), so
  // the plane is still low-packed-valid but numerically different.
  let wrong_frame = Bayer12Frame::try_new(&be_plane, w, h, w).unwrap();
  let mut wrong_rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut wrong_sinker = MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
    .with_rgb(&mut wrong_rgb)
    .unwrap();
  bayer16_to::<12, _>(
    &wrong_frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut wrong_sinker,
  )
  .unwrap();

  assert_ne!(
    be_rgb, wrong_rgb,
    "BE byte-swap not applied: BE path coincides with the wrong-endian LE reading"
  );
}

/// Per-depth BE round-trip: 10 / 14 / 16-bit BE planes each match their
/// LE twin (u8 out). Exercises the `BITS`-generic monomorphization of
/// the BE kernel across every depth, not just 12-bit.
#[test]
fn bayer16_be_matches_le_per_depth() {
  use crate::{
    frame::{
      Bayer10BeFrame, Bayer10Frame, Bayer14BeFrame, Bayer14Frame, Bayer16BeFrame, Bayer16Frame,
    },
    raw::{
      BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer16_to,
      bayer16_to_endian,
    },
  };
  let (w, h) = (8u32, 6u32);

  macro_rules! depth_case {
    ($bits:literal, $le_frame:ident, $be_frame:ident, $r:expr, $g:expr, $b:expr) => {{
      // Build the logical samples once (reuse the 12-bit pattern helper
      // shape; the values are within range for this depth), then derive
      // both wire planes from them so the comparison holds on any host.
      let logical = solid_pattern12(BayerPattern::Rggb, w, h, $r, $g, $b);
      let le_plane: std::vec::Vec<u16> = logical.iter().map(|&v| le_wire_u16(v)).collect();

      let le_frame = $le_frame::try_new(&le_plane, w, h, w).unwrap();
      let mut le_rgb = std::vec![0u8; (w * h * 3) as usize];
      let mut le_sinker =
        MixedSinker::<Bayer16<$bits, false>>::new(w as usize, h as usize)
          .with_rgb(&mut le_rgb)
          .unwrap();
      bayer16_to::<$bits, _>(
        &le_frame,
        BayerPattern::Rggb,
        BayerDemosaic::Bilinear,
        WhiteBalance::neutral(),
        ColorCorrectionMatrix::identity(),
        &mut le_sinker,
      )
      .unwrap();

      let be_plane: std::vec::Vec<u16> = logical.iter().map(|&v| be_wire_u16(v)).collect();
      let be_frame = $be_frame::try_new(&be_plane, w, h, w).unwrap();
      let mut be_rgb = std::vec![0u8; (w * h * 3) as usize];
      let mut be_sinker =
        MixedSinker::<Bayer16<$bits, true>>::new(w as usize, h as usize)
          .with_rgb(&mut be_rgb)
          .unwrap();
      bayer16_to_endian::<$bits, true, _>(
        &be_frame,
        BayerPattern::Rggb,
        BayerDemosaic::Bilinear,
        WhiteBalance::neutral(),
        ColorCorrectionMatrix::identity(),
        &mut be_sinker,
      )
      .unwrap();

      assert_eq!(be_rgb, le_rgb, "BE/LE mismatch at {} bits", $bits);
    }};
  }

  // 10-bit: max 1023; 14-bit: max 16383; 16-bit: full u16.
  depth_case!(10, Bayer10Frame, Bayer10BeFrame, 0x03FF, 0x02AB, 0x0123);
  depth_case!(14, Bayer14Frame, Bayer14BeFrame, 0x3FFF, 0x2ABC, 0x1234);
  depth_case!(16, Bayer16Frame, Bayer16BeFrame, 0xFFFF, 0xABCD, 0x1234);
}
