use super::*;

// ---- expand_rgb_to_rgba_row -----------------------------------------

#[test]
fn expand_rgb_to_rgba_row_pads_alpha_and_preserves_rgb() {
  // Each source pixel's R/G/B must land in the matching slot, with
  // alpha forced to 0xFF — Strategy A's correctness depends on this.
  let rgb: std::vec::Vec<u8> = (0..16 * 3).map(|i| i as u8).collect();
  let mut rgba = std::vec![0u8; 16 * 4];
  expand_rgb_to_rgba_row(&rgb, &mut rgba, 16);
  for x in 0..16 {
    assert_eq!(rgba[x * 4], rgb[x * 3], "R at px {x}");
    assert_eq!(rgba[x * 4 + 1], rgb[x * 3 + 1], "G at px {x}");
    assert_eq!(rgba[x * 4 + 2], rgb[x * 3 + 2], "B at px {x}");
    assert_eq!(rgba[x * 4 + 3], 0xFF, "A at px {x}");
  }
}

#[test]
fn expand_rgb_to_rgba_row_only_writes_first_width_pixels() {
  // Caller may pass over-sized RGBA buffers; we must not stomp on
  // the trailing region. Pre-fill 0xAA, expand into the head, and
  // verify the tail still reads 0xAA.
  let rgb: std::vec::Vec<u8> = (0..8 * 3).map(|i| (i + 1) as u8).collect();
  let mut rgba = std::vec![0xAAu8; 16 * 4];
  expand_rgb_to_rgba_row(&rgb, &mut rgba, 8);
  for x in 0..8 {
    assert_eq!(rgba[x * 4], rgb[x * 3]);
    assert_eq!(rgba[x * 4 + 3], 0xFF);
  }
  for &b in &rgba[8 * 4..] {
    assert_eq!(b, 0xAA, "tail must be untouched");
  }
}

#[test]
fn expand_rgb_to_rgba_row_zero_width_is_noop() {
  let rgb: std::vec::Vec<u8> = std::vec::Vec::new();
  let mut rgba = std::vec![0u8; 0];
  expand_rgb_to_rgba_row(&rgb, &mut rgba, 0);
  assert!(rgba.is_empty());
}

// ---- yuv_420_to_rgb_row ----------------------------------------------

#[test]
fn yuv420_rgb_black() {
  // Full-range Y=0, neutral chroma → black.
  let y = [0u8; 4];
  let u = [128u8; 2];
  let v = [128u8; 2];
  let mut rgb = [0u8; 12];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
}

#[test]
fn yuv420_rgb_white_full_range() {
  let y = [255u8; 4];
  let u = [128u8; 2];
  let v = [128u8; 2];
  let mut rgb = [0u8; 12];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
}

#[test]
fn yuv420_rgb_gray_is_gray() {
  let y = [128u8; 4];
  let u = [128u8; 2];
  let v = [128u8; 2];
  let mut rgb = [0u8; 12];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(128) <= 1, "got {r}");
  }
}

#[test]
fn yuv420_rgb_chroma_shared_across_pair() {
  // Two Y values with same chroma: differing Y produces differing
  // luminance but same chroma-driven offsets. Validates that pixel x
  // and x+1 share the upsampled chroma sample.
  let y = [50u8, 200, 50, 200];
  let u = [128u8; 2];
  let v = [128u8; 2];
  let mut rgb = [0u8; 12];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  // With neutral chroma, output is gray = Y.
  assert_eq!(rgb[0], 50);
  assert_eq!(rgb[3], 200);
  assert_eq!(rgb[6], 50);
  assert_eq!(rgb[9], 200);
}

#[test]
fn yuv420_rgb_limited_range_black_and_white() {
  // Y=16 → black, Y=235 → white in limited range.
  let y = [16u8, 16, 235, 235];
  let u = [128u8; 2];
  let v = [128u8; 2];
  let mut rgb = [0u8; 12];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, false);
  for x in 0..2 {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!((r, g, b), (0, 0, 0), "limited-range Y=16 should be black");
  }
  for x in 2..4 {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!(
      (r, g, b),
      (255, 255, 255),
      "limited-range Y=235 should be white"
    );
  }
}

#[test]
fn yuv420_rgb_ycgco_neutral_is_gray() {
  // Y=128, Cg=128 (U), Co=128 (V) — neutral chroma → gray.
  let y = [128u8; 2];
  let u = [128u8; 1]; // Cg
  let v = [128u8; 1]; // Co
  let mut rgb = [0u8; 6];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 2, ColorMatrix::YCgCo, true);
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1, "RGB should be gray, got {rgb:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
fn yuv420_rgb_ycgco_high_cg_is_green() {
  // U plane = Cg; Cg > 128 means green-ward shift.
  // Expected math (Y=128, Cg=200, Co=128):
  //   u_d = 72, v_d = 0
  //   R = 128 - 72 + 0 = 56
  //   G = 128 + 72     = 200
  //   B = 128 - 72 - 0 = 56
  let y = [128u8; 2];
  let u = [200u8; 1]; // Cg = 200 (green-ward)
  let v = [128u8; 1]; // Co neutral
  let mut rgb = [0u8; 6];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 2, ColorMatrix::YCgCo, true);
  for px in rgb.chunks(3) {
    // Allow ±1 for Q15 rounding. RGB order: [R, G, B].
    assert!(px[0].abs_diff(56) <= 1, "expected R≈56, got {rgb:?}");
    assert!(px[1].abs_diff(200) <= 1, "expected G≈200, got {rgb:?}");
    assert!(px[2].abs_diff(56) <= 1, "expected B≈56, got {rgb:?}");
  }
}

#[test]
fn yuv420_rgb_ycgco_high_co_is_red() {
  // V plane = Co; Co > 128 means orange/red-ward shift.
  // Expected (Y=128, Cg=128, Co=200):
  //   u_d = 0, v_d = 72
  //   R = 128 - 0 + 72 = 200
  //   G = 128 + 0      = 128
  //   B = 128 - 0 - 72 = 56
  let y = [128u8; 2];
  let u = [128u8; 1]; // Cg neutral
  let v = [200u8; 1]; // Co = 200 (orange-ward)
  let mut rgb = [0u8; 6];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 2, ColorMatrix::YCgCo, true);
  for px in rgb.chunks(3) {
    // RGB order: [R, G, B].
    assert!(px[0].abs_diff(200) <= 1, "expected R≈200, got {rgb:?}");
    assert!(px[1].abs_diff(128) <= 1, "expected G≈128, got {rgb:?}");
    assert!(px[2].abs_diff(56) <= 1, "expected B≈56, got {rgb:?}");
  }
}

#[test]
fn yuv420_rgb_bt601_vs_bt709_differ_for_chroma() {
  // Moderate chroma (V=200) so the red channel doesn't saturate on
  // either matrix — saturating both and then diffing gives zero.
  let y = [128u8; 2];
  let u = [128u8; 1];
  let v = [200u8; 1];
  let mut b601 = [0u8; 6];
  let mut b709 = [0u8; 6];
  yuv_420_to_rgb_row(&y, &u, &v, &mut b601, 2, ColorMatrix::Bt601, true);
  yuv_420_to_rgb_row(&y, &u, &v, &mut b709, 2, ColorMatrix::Bt709, true);
  // Sum of per-channel absolute differences — robust to which
  // particular channel the two matrices disagree on.
  let sad: i32 = b601
    .iter()
    .zip(b709.iter())
    .map(|(a, b)| (*a as i32 - *b as i32).abs())
    .sum();
  assert!(
    sad > 20,
    "BT.601 vs BT.709 outputs should materially differ: {b601:?} vs {b709:?}"
  );
}

// ---- rgb_to_hsv_row --------------------------------------------------

#[test]
fn hsv_gray_has_no_hue_no_sat() {
  let rgb = [128u8; 3];
  let (mut h, mut s, mut v) = ([0u8; 1], [0u8; 1], [0u8; 1]);
  rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, 1);
  assert_eq!((h[0], s[0], v[0]), (0, 0, 128));
}

#[test]
fn hsv_pure_red_matches_opencv() {
  // OpenCV RGB2HSV: red = (R=255, G=0, B=0) → H = 0, S = 255, V = 255.
  let rgb = [255u8, 0, 0];
  let (mut h, mut s, mut v) = ([0u8; 1], [0u8; 1], [0u8; 1]);
  rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, 1);
  assert_eq!((h[0], s[0], v[0]), (0, 255, 255));
}

#[test]
fn hsv_pure_green_matches_opencv() {
  // Green (R=0, G=255, B=0) → H = 60 in OpenCV 8-bit (120° / 2).
  let rgb = [0u8, 255, 0];
  let (mut h, mut s, mut v) = ([0u8; 1], [0u8; 1], [0u8; 1]);
  rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, 1);
  assert_eq!((h[0], s[0], v[0]), (60, 255, 255));
}

#[test]
fn hsv_pure_blue_matches_opencv() {
  // Blue (R=0, G=0, B=255) → H = 120 (240° / 2).
  let rgb = [0u8, 0, 255];
  let (mut h, mut s, mut v) = ([0u8; 1], [0u8; 1], [0u8; 1]);
  rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, 1);
  assert_eq!((h[0], s[0], v[0]), (120, 255, 255));
}

// ---- yuv_420p_n_to_rgb_row (10-bit → u8) -----------------------------

#[test]
fn yuv420p10_rgb_black_full_range() {
  // Y=0, neutral chroma (512 in 10-bit) → black.
  let y = [0u16; 4];
  let u = [512u16; 2];
  let v = [512u16; 2];
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
}

#[test]
fn yuv420p10_rgb_white_full_range() {
  // 10-bit full-range white is Y=1023.
  let y = [1023u16; 4];
  let u = [512u16; 2];
  let v = [512u16; 2];
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
}

#[test]
fn yuv420p10_rgb_gray_is_gray() {
  // Mid-gray 10-bit Y=512 ↔ 8-bit 128. Within ±1 for Q15 rounding.
  let y = [512u16; 4];
  let u = [512u16; 2];
  let v = [512u16; 2];
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(128) <= 1, "got {r}");
  }
}

#[test]
fn yuv420p10_rgb_limited_range_black_and_white() {
  // 10-bit limited: Y=64 → black, Y=940 → white.
  let y = [64u16, 64, 940, 940];
  let u = [512u16; 2];
  let v = [512u16; 2];
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, false);
  assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
  assert_eq!((rgb[3], rgb[4], rgb[5]), (0, 0, 0));
  assert_eq!((rgb[6], rgb[7], rgb[8]), (255, 255, 255));
  assert_eq!((rgb[9], rgb[10], rgb[11]), (255, 255, 255));
}

#[test]
fn yuv420p10_rgb_chroma_shared_across_pair() {
  // Two 10-bit Y values sharing chroma: output is gray = Y>>2.
  let y = [200u16, 800, 200, 800];
  let u = [512u16; 2];
  let v = [512u16; 2];
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  // Full-range 10→8 scale = 255/1023, so Y=200 → 50, Y=800 → 199.4 → 199.
  // Allow ±1 for Q15 rounding.
  assert!(rgb[0].abs_diff(50) <= 1, "got {}", rgb[0]);
  assert!(rgb[3].abs_diff(199) <= 1, "got {}", rgb[3]);
  assert!(rgb[6].abs_diff(50) <= 1, "got {}", rgb[6]);
  assert!(rgb[9].abs_diff(199) <= 1, "got {}", rgb[9]);
}

// ---- yuv_420p_n_to_rgb_u16_row (10-bit → 10-bit u16) ----------------

#[test]
fn yuv420p10_rgb_u16_black_full_range() {
  let y = [0u16; 4];
  let u = [512u16; 2];
  let v = [512u16; 2];
  let mut rgb = [0u16; 12];
  yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
}

#[test]
fn yuv420p10_rgb_u16_white_full_range() {
  // 10-bit input Y=1023, full-range scale=1 → output Y=1023 on each channel.
  let y = [1023u16; 4];
  let u = [512u16; 2];
  let v = [512u16; 2];
  let mut rgb = [0u16; 12];
  yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 1023), "got {rgb:?}");
}

#[test]
fn yuv420p10_rgb_u16_limited_range_endpoints() {
  // Limited-range: Y=64 → 0, Y=940 → 1023 in 10-bit output.
  let y = [64u16, 940];
  let u = [512u16; 1];
  let v = [512u16; 1];
  let mut rgb = [0u16; 6];
  yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb, 2, ColorMatrix::Bt709, false);
  assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
  assert_eq!((rgb[3], rgb[4], rgb[5]), (1023, 1023, 1023));
}

#[test]
fn yuv420p10_rgb_u16_preserves_full_10bit_precision() {
  // Sanity: the u16 path retains native-depth precision, so two
  // inputs that round to the same u8 are distinguishable in u16.
  // Full-range Y=200 vs Y=201: same u8 output (50 vs 50) but
  // distinct u16 outputs (200 vs 201).
  let y = [200u16, 201];
  let u = [512u16; 1];
  let v = [512u16; 1];
  let mut rgb8 = [0u8; 6];
  let mut rgb16 = [0u16; 6];
  yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb8, 2, ColorMatrix::Bt601, true);
  yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb16, 2, ColorMatrix::Bt601, true);
  assert_eq!(rgb8[0], rgb8[3]);
  assert_ne!(rgb16[0], rgb16[3]);
}

#[test]
fn yuv420p10_bt709_ycgco_differ_for_chroma() {
  // Non-neutral chroma — different matrices produce different RGB.
  let y = [512u16; 2];
  let u = [512u16; 1];
  let v = [800u16; 1];
  let mut bt709 = [0u8; 6];
  let mut ycgco = [0u8; 6];
  yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut bt709, 2, ColorMatrix::Bt709, true);
  yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut ycgco, 2, ColorMatrix::YCgCo, true);
  let sad: i32 = bt709
    .iter()
    .zip(ycgco.iter())
    .map(|(a, b)| (*a as i32 - *b as i32).abs())
    .sum();
  assert!(
    sad > 20,
    "matrices should materially differ: {bt709:?} vs {ycgco:?}"
  );
}

// ---- p010_to_rgb_row (P010 → u8) ---------------------------------------
//
// P010 samples: 10 active bits in the HIGH 10 of each u16.
// White Y = 1023 << 6 = 0xFFC0, neutral UV = 512 << 6 = 0x8000.

#[test]
fn p010_rgb_black_full_range() {
  // Y = 0, neutral UV → black.
  let y = [0u16; 4];
  let uv = [0x8000u16, 0x8000, 0x8000, 0x8000]; // U0 V0 U1 V1
  let mut rgb = [0u8; 12];
  p_n_to_rgb_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
}

#[test]
fn p010_rgb_white_full_range() {
  // Y = 0xFFC0 = 1023 << 6, neutral UV → white.
  let y = [0xFFC0u16; 4];
  let uv = [0x8000u16, 0x8000, 0x8000, 0x8000];
  let mut rgb = [0u8; 12];
  p_n_to_rgb_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
}

#[test]
fn p010_rgb_gray_is_gray() {
  // 10-bit mid-gray Y=512 → P010 Y = 512 << 6 = 0x8000.
  let y = [0x8000u16; 4];
  let uv = [0x8000u16; 4];
  let mut rgb = [0u8; 12];
  p_n_to_rgb_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(128) <= 1, "got {r}");
  }
}

#[test]
fn p010_rgb_limited_range_endpoints() {
  // 10-bit limited black Y=64 → P010 = 64 << 6 = 0x1000.
  // 10-bit limited white Y=940 → P010 = 940 << 6 = 0xEB00.
  let y = [0x1000u16, 0x1000, 0xEB00, 0xEB00];
  let uv = [0x8000u16, 0x8000, 0x8000, 0x8000];
  let mut rgb = [0u8; 12];
  p_n_to_rgb_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, false);
  assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
  assert_eq!((rgb[3], rgb[4], rgb[5]), (0, 0, 0));
  assert_eq!((rgb[6], rgb[7], rgb[8]), (255, 255, 255));
  assert_eq!((rgb[9], rgb[10], rgb[11]), (255, 255, 255));
}

#[test]
fn p010_matches_yuv420p10_when_shifted() {
  // Handing the same logical samples to P010 (high-packed) and
  // yuv420p10 (low-packed) must produce the same RGB output.
  let y_p10 = [200u16, 800, 500, 700]; // 10-bit values
  let u_p10 = [600u16, 400]; // 10-bit values
  let v_p10 = [300u16, 900]; // 10-bit values

  let y_p010: [u16; 4] = core::array::from_fn(|i| y_p10[i] << 6);
  let uv_p010: [u16; 4] = [u_p10[0] << 6, v_p10[0] << 6, u_p10[1] << 6, v_p10[1] << 6];

  let mut rgb_p10 = [0u8; 12];
  let mut rgb_p010 = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10>(
    &y_p10,
    &u_p10,
    &v_p10,
    &mut rgb_p10,
    4,
    ColorMatrix::Bt709,
    true,
  );
  p_n_to_rgb_row::<10>(
    &y_p010,
    &uv_p010,
    &mut rgb_p010,
    4,
    ColorMatrix::Bt709,
    true,
  );
  assert_eq!(rgb_p10, rgb_p010);
}

// ---- p010_to_rgb_u16_row (P010 → native-depth u16) --------------------

#[test]
fn p010_rgb_u16_white_full_range() {
  let y = [0xFFC0u16; 4];
  let uv = [0x8000u16; 4];
  let mut rgb = [0u16; 12];
  p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 1023), "got {rgb:?}");
}

#[test]
fn p010_rgb_u16_limited_range_endpoints() {
  let y = [0x1000u16, 0xEB00];
  let uv = [0x8000u16, 0x8000];
  let mut rgb = [0u16; 6];
  p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb, 2, ColorMatrix::Bt709, false);
  assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
  assert_eq!((rgb[3], rgb[4], rgb[5]), (1023, 1023, 1023));
}
