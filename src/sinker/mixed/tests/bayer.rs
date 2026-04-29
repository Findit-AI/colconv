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
  let row = BayerRow::new(
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
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::BayerAbove,
      expected: 8,
      actual: 7,
      ..
    }
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
  let row = BayerRow16::<12>::new(
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
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Bayer16Mid,
      expected: 8,
      actual: 7,
      ..
    }
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
  // each coefficient set must produce a different value. The
  // hard-coded BT.709 bug Codex flagged would make these all 54.
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
