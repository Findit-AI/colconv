use super::*;

// Helper: re-encode a host-native u16 slice as the LE-encoded byte layout so
// kernels called with `BE = false` recover the intended values via `from_le`
// on both LE (no-op) and BE (byte-swap) hosts.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

// Same idea for `half::f16` slices.
#[cfg(feature = "rgb-float")]
fn as_le_f16(host: &[half::f16]) -> std::vec::Vec<half::f16> {
  host
    .iter()
    .map(|v| half::f16::from_bits(u16::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

// Same idea for `f32` slices.
#[cfg(feature = "rgb-float")]
fn as_le_f32(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

// ---- expand_rgb_to_rgba_row -----------------------------------------

// Same consumer set as the `rgb_expand` re-export in `scalar::mod`:
// `expand_rgb_to_rgba_row` is only re-exported for the families that
// fan an RGB row out to RGBA via Strategy A (Bayer is RGB-only; mono /
// rgb-float / rgb-legacy / xyz never go through the fan-out), so the
// tests for it gate to the same set.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "rgb",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
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

// Same consumer set as the `rgb_expand` re-export in `scalar::mod`:
// `expand_rgb_to_rgba_row` is only re-exported for the families that
// fan an RGB row out to RGBA via Strategy A (Bayer is RGB-only; mono /
// rgb-float / rgb-legacy / xyz never go through the fan-out), so the
// tests for it gate to the same set.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "rgb",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
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

// Same consumer set as the `rgb_expand` re-export in `scalar::mod`:
// `expand_rgb_to_rgba_row` is only re-exported for the families that
// fan an RGB row out to RGBA via Strategy A (Bayer is RGB-only; mono /
// rgb-float / rgb-legacy / xyz never go through the fan-out), so the
// tests for it gate to the same set.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "rgb",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[test]
fn expand_rgb_to_rgba_row_zero_width_is_noop() {
  let rgb: std::vec::Vec<u8> = std::vec::Vec::new();
  let mut rgba = std::vec![0u8; 0];
  expand_rgb_to_rgba_row(&rgb, &mut rgba, 0);
  assert!(rgba.is_empty());
}

// ---- yuv_420_to_rgb_row ----------------------------------------------

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420_rgb_white_full_range() {
  let y = [255u8; 4];
  let u = [128u8; 2];
  let v = [128u8; 2];
  let mut rgb = [0u8; 12];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
}

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

// ---- yuv_411_to_rgb_row (4:1:1 — quarter-width chroma) ---------------

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_black() {
  // Full-range Y=0, neutral chroma → black. width=4 means one chroma
  // sample drives all four Y pixels (the full sub-block).
  let y = [0u8; 4];
  let u = [128u8; 1];
  let v = [128u8; 1];
  let mut rgb = [0u8; 12];
  yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_white_full_range() {
  let y = [255u8; 4];
  let u = [128u8; 1];
  let v = [128u8; 1];
  let mut rgb = [0u8; 12];
  yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_gray_is_gray() {
  let y = [128u8; 4];
  let u = [128u8; 1];
  let v = [128u8; 1];
  let mut rgb = [0u8; 12];
  yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(128) <= 1, "got {r}");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_chroma_shared_across_quartet() {
  // Four Y values with same chroma: differing Y produces differing
  // luminance but same chroma-driven offsets. Validates that pixels
  // x..x+3 share the upsampled chroma sample.
  let y = [50u8, 100, 150, 200];
  let u = [128u8; 1];
  let v = [128u8; 1];
  let mut rgb = [0u8; 12];
  yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  // With neutral chroma, output is gray = Y for each pixel.
  assert_eq!(rgb[0], 50);
  assert_eq!(rgb[3], 100);
  assert_eq!(rgb[6], 150);
  assert_eq!(rgb[9], 200);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_two_chroma_blocks() {
  // 8-pixel row with two different chroma blocks. Sub-block 0 (Y[0..4])
  // gets chroma[0]; sub-block 1 (Y[4..8]) gets chroma[1]. Validates
  // the `c_idx = x / 4` indexing.
  let y = [128u8; 8];
  // First quartet: red-ward (V=200); second: green-ward (V=64).
  let u = [128u8, 128];
  let v = [200u8, 64];
  let mut rgb = [0u8; 24];
  yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, 8, ColorMatrix::Bt601, true);
  // First four pixels: red boost.
  for x in 0..4 {
    let r = rgb[x * 3];
    let b = rgb[x * 3 + 2];
    assert!(r > b, "px {x}: r={r} should be > b={b} for V=200");
  }
  // Last four pixels: blue boost (V=64 → cyan-ish).
  for x in 4..8 {
    let r = rgb[x * 3];
    let b = rgb[x * 3 + 2];
    assert!(r < b, "px {x}: r={r} should be < b={b} for V=64");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgba_alpha_is_opaque() {
  // RGBA wrapper writes four bytes per pixel with constant 0xFF
  // alpha; the first three bytes match yuv_411_to_rgb_row.
  let y = [200u8, 100, 50, 150];
  let u = [128u8; 1];
  let v = [128u8; 1];
  let mut rgba = [0u8; 16];
  yuv_411_to_rgba_row(&y, &u, &v, &mut rgba, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    assert_eq!(rgba[x * 4 + 3], 0xFF, "alpha at px {x}");
  }
  // R/G/B match `yuv_411_to_rgb_row` byte-for-byte.
  let mut rgb = [0u8; 12];
  yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    assert_eq!(rgba[x * 4], rgb[x * 3]);
    assert_eq!(rgba[x * 4 + 1], rgb[x * 3 + 1]);
    assert_eq!(rgba[x * 4 + 2], rgb[x * 3 + 2]);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_limited_range_black_and_white() {
  // Y=16 → black, Y=235 → white in limited range. Two quartets.
  let y = [16u8, 16, 16, 16, 235, 235, 235, 235];
  let u = [128u8; 2];
  let v = [128u8; 2];
  let mut rgb = [0u8; 24];
  yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, 8, ColorMatrix::Bt601, false);
  for x in 0..4 {
    assert_eq!((rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]), (0, 0, 0));
  }
  for x in 4..8 {
    assert_eq!(
      (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]),
      (255, 255, 255),
      "limited-range Y=235 should be white"
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_widths_1_through_3_partial_chroma_only() {
  // Widths 1, 2, 3 produce a single chroma sample serving all luma.
  // FFmpeg ceil-shift: chroma_width = width.div_ceil(4) = 1.
  // With neutral chroma (128) and full-range, R=G=B=Y for each pixel.
  for w in [1usize, 2, 3] {
    let y: std::vec::Vec<u8> = (0..w as u8).map(|i| (i + 1) * 30).collect();
    let u = [128u8; 1];
    let v = [128u8; 1];
    let mut rgb = std::vec![0u8; 3 * w];
    yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, w, ColorMatrix::Bt601, true);
    for x in 0..w {
      let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!(r, g, "width={w} px={x}: r={r} g={g}");
      assert_eq!(g, b, "width={w} px={x}: g={g} b={b}");
      assert!(r.abs_diff(y[x]) <= 1, "width={w} px={x}: r={r} y={}", y[x]);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_widths_5_6_7_partial_tail_uses_last_chroma() {
  // Widths 5, 6, 7 → chroma_width=2; first 4 luma share chroma[0],
  // trailing 1..3 luma share chroma[1] (the partial group). Use
  // distinct chroma to verify the boundary.
  for w in [5usize, 6, 7] {
    let y = std::vec![128u8; w];
    let u = [128u8, 128];
    // First chroma drives red boost (V=200), second drives blue boost (V=64).
    let v = [200u8, 64];
    let mut rgb = std::vec![0u8; 3 * w];
    yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, w, ColorMatrix::Bt601, true);
    // First four pixels: red > blue.
    for x in 0..4 {
      let r = rgb[x * 3];
      let b = rgb[x * 3 + 2];
      assert!(
        r > b,
        "width={w} px={x}: r={r} should be > b={b} (chroma[0])"
      );
    }
    // Trailing pixels (4..w): blue > red (chroma[1] partial group).
    for x in 4..w {
      let r = rgb[x * 3];
      let b = rgb[x * 3 + 2];
      assert!(
        r < b,
        "width={w} px={x}: r={r} should be < b={b} (chroma[1] partial)"
      );
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgba_widths_5_through_7_alpha_opaque() {
  // RGBA tail must still write 0xFF alpha for every pixel, including
  // the partial-chroma trailing 1..3 luma samples.
  for w in [5usize, 6, 7] {
    let y = std::vec![128u8; w];
    let u = std::vec![128u8; 2];
    let v = std::vec![128u8; 2];
    let mut rgba = std::vec![0u8; 4 * w];
    yuv_411_to_rgba_row(&y, &u, &v, &mut rgba, w, ColorMatrix::Bt601, true);
    for x in 0..w {
      assert_eq!(rgba[x * 4 + 3], 0xFF, "width={w} alpha at px {x}");
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv411_rgb_width_641_realistic_cropped() {
  // Realistic non-4-aligned width: 641 → chroma width 161; the SIMD
  // body (when present) skips the trailing 1 pixel which is handled
  // by the partial-chroma scalar path.
  // Verify no panic, output length correct, last pixel matches Y.
  let w = 641usize;
  let y: std::vec::Vec<u8> = (0..w).map(|i| (i % 256) as u8).collect();
  let u = std::vec![128u8; 161];
  let v = std::vec![128u8; 161];
  let mut rgb = std::vec![0u8; 3 * w];
  yuv_411_to_rgb_row(&y, &u, &v, &mut rgb, w, ColorMatrix::Bt601, true);
  // Neutral chroma → R=G=B≈Y. Spot-check first, mid, and last pixel
  // (the last is the partial-tail one — y[640] inside chroma[160]).
  for &x in &[0usize, 320, 639, 640] {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!(r, g, "px {x}");
    assert_eq!(g, b, "px {x}");
    assert!(r.abs_diff(y[x]) <= 1, "px {x}: r={r} y={}", y[x]);
  }
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

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_black_full_range() {
  // Y=0, neutral chroma (512 in 10-bit) → black.
  let y = as_le_u16(&[0u16; 4]);
  let u = as_le_u16(&[512u16; 2]);
  let v = as_le_u16(&[512u16; 2]);
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_white_full_range() {
  // 10-bit full-range white is Y=1023.
  let y = as_le_u16(&[1023u16; 4]);
  let u = as_le_u16(&[512u16; 2]);
  let v = as_le_u16(&[512u16; 2]);
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_gray_is_gray() {
  // Mid-gray 10-bit Y=512 ↔ 8-bit 128. Within ±1 for Q15 rounding.
  let y = as_le_u16(&[512u16; 4]);
  let u = as_le_u16(&[512u16; 2]);
  let v = as_le_u16(&[512u16; 2]);
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(128) <= 1, "got {r}");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_limited_range_black_and_white() {
  // 10-bit limited: Y=64 → black, Y=940 → white.
  let y = as_le_u16(&[64u16, 64, 940, 940]);
  let u = as_le_u16(&[512u16; 2]);
  let v = as_le_u16(&[512u16; 2]);
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, false);
  assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
  assert_eq!((rgb[3], rgb[4], rgb[5]), (0, 0, 0));
  assert_eq!((rgb[6], rgb[7], rgb[8]), (255, 255, 255));
  assert_eq!((rgb[9], rgb[10], rgb[11]), (255, 255, 255));
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_chroma_shared_across_pair() {
  // Two 10-bit Y values sharing chroma: output is gray = Y>>2.
  let y = as_le_u16(&[200u16, 800, 200, 800]);
  let u = as_le_u16(&[512u16; 2]);
  let v = as_le_u16(&[512u16; 2]);
  let mut rgb = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  // Full-range 10→8 scale = 255/1023, so Y=200 → 50, Y=800 → 199.4 → 199.
  // Allow ±1 for Q15 rounding.
  assert!(rgb[0].abs_diff(50) <= 1, "got {}", rgb[0]);
  assert!(rgb[3].abs_diff(199) <= 1, "got {}", rgb[3]);
  assert!(rgb[6].abs_diff(50) <= 1, "got {}", rgb[6]);
  assert!(rgb[9].abs_diff(199) <= 1, "got {}", rgb[9]);
}

// ---- yuv_420p_n_to_rgb_u16_row (10-bit → 10-bit u16) ----------------

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_u16_black_full_range() {
  let y = as_le_u16(&[0u16; 4]);
  let u = as_le_u16(&[512u16; 2]);
  let v = as_le_u16(&[512u16; 2]);
  let mut rgb = [0u16; 12];
  yuv_420p_n_to_rgb_u16_row::<10, false>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_u16_white_full_range() {
  // 10-bit input Y=1023, full-range scale=1 → output Y=1023 on each channel.
  let y = as_le_u16(&[1023u16; 4]);
  let u = as_le_u16(&[512u16; 2]);
  let v = as_le_u16(&[512u16; 2]);
  let mut rgb = [0u16; 12];
  yuv_420p_n_to_rgb_u16_row::<10, false>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 1023), "got {rgb:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_u16_limited_range_endpoints() {
  // Limited-range: Y=64 → 0, Y=940 → 1023 in 10-bit output.
  let y = as_le_u16(&[64u16, 940]);
  let u = as_le_u16(&[512u16; 1]);
  let v = as_le_u16(&[512u16; 1]);
  let mut rgb = [0u16; 6];
  yuv_420p_n_to_rgb_u16_row::<10, false>(&y, &u, &v, &mut rgb, 2, ColorMatrix::Bt709, false);
  assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
  assert_eq!((rgb[3], rgb[4], rgb[5]), (1023, 1023, 1023));
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_rgb_u16_preserves_full_10bit_precision() {
  // Sanity: the u16 path retains native-depth precision, so two
  // inputs that round to the same u8 are distinguishable in u16.
  // Full-range Y=200 vs Y=201: same u8 output (50 vs 50) but
  // distinct u16 outputs (200 vs 201).
  let y = as_le_u16(&[200u16, 201]);
  let u = as_le_u16(&[512u16; 1]);
  let v = as_le_u16(&[512u16; 1]);
  let mut rgb8 = [0u8; 6];
  let mut rgb16 = [0u16; 6];
  yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut rgb8, 2, ColorMatrix::Bt601, true);
  yuv_420p_n_to_rgb_u16_row::<10, false>(&y, &u, &v, &mut rgb16, 2, ColorMatrix::Bt601, true);
  assert_eq!(rgb8[0], rgb8[3]);
  assert_ne!(rgb16[0], rgb16[3]);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_bt709_ycgco_differ_for_chroma() {
  // Non-neutral chroma — different matrices produce different RGB.
  let y = [512u16; 2];
  let u = [512u16; 1];
  let v = [800u16; 1];
  let mut bt709 = [0u8; 6];
  let mut ycgco = [0u8; 6];
  yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut bt709, 2, ColorMatrix::Bt709, true);
  yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut ycgco, 2, ColorMatrix::YCgCo, true);
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

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn p010_rgb_black_full_range() {
  // Y = 0, neutral UV → black.
  let y = as_le_u16(&[0u16; 4]);
  let uv = as_le_u16(&[0x8000u16, 0x8000, 0x8000, 0x8000]); // U0 V0 U1 V1
  let mut rgb = [0u8; 12];
  p_n_to_rgb_row::<10, false>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn p010_rgb_white_full_range() {
  // Y = 0xFFC0 = 1023 << 6, neutral UV → white.
  let y = as_le_u16(&[0xFFC0u16; 4]);
  let uv = as_le_u16(&[0x8000u16, 0x8000, 0x8000, 0x8000]);
  let mut rgb = [0u8; 12];
  p_n_to_rgb_row::<10, false>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn p010_rgb_gray_is_gray() {
  // 10-bit mid-gray Y=512 → P010 Y = 512 << 6 = 0x8000.
  let y = as_le_u16(&[0x8000u16; 4]);
  let uv = as_le_u16(&[0x8000u16; 4]);
  let mut rgb = [0u8; 12];
  p_n_to_rgb_row::<10, false>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(128) <= 1, "got {r}");
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn p010_rgb_limited_range_endpoints() {
  // 10-bit limited black Y=64 → P010 = 64 << 6 = 0x1000.
  // 10-bit limited white Y=940 → P010 = 940 << 6 = 0xEB00.
  let y = as_le_u16(&[0x1000u16, 0x1000, 0xEB00, 0xEB00]);
  let uv = as_le_u16(&[0x8000u16, 0x8000, 0x8000, 0x8000]);
  let mut rgb = [0u8; 12];
  p_n_to_rgb_row::<10, false>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, false);
  assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
  assert_eq!((rgb[3], rgb[4], rgb[5]), (0, 0, 0));
  assert_eq!((rgb[6], rgb[7], rgb[8]), (255, 255, 255));
  assert_eq!((rgb[9], rgb[10], rgb[11]), (255, 255, 255));
}

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[test]
fn p010_matches_yuv420p10_when_shifted() {
  // Handing the same logical samples to P010 (high-packed) and
  // yuv420p10 (low-packed) must produce the same RGB output.
  //
  // Intended host-native samples:
  //   Y per-pixel: [0, 1023, 512, 700]
  //     pixel 0 = full-range black (Y=0 with neutral chroma → RGB 0,0,0)
  //     pixel 1 = full-range white (Y=1023 with neutral chroma → RGB 255,…)
  //     pixel 2 = full-range mid-gray (Y=512, neutral chroma → R=G=B≈128)
  //     pixel 3 = arbitrary mid-luma (Y=700, neutral chroma → R=G=B≈175)
  //   Chroma: U=V=512 on every chroma sample (neutral 10-bit center).
  //
  // The neutral chroma fixes R=G=B for every pixel, so the expected
  // RGB triples are computable from the host-native intended Y values
  // alone — independent of the kernel's matrix coefficients. That
  // turns this from a vacuous self-comparison ("both kernels agree")
  // into an absolute-value check: a BE host that fails to LE-encode
  // these fixtures would decode different (post-byte-swap) Y values
  // and produce different RGB, mismatching the host-native expectation.
  let y_p10_intended = [0u16, 1023, 512, 700];
  let u_p10_intended = [512u16, 512];
  let v_p10_intended = [512u16, 512];

  let y_p010_intended: [u16; 4] = core::array::from_fn(|i| y_p10_intended[i] << 6);
  let uv_p010_intended: [u16; 4] = [
    u_p10_intended[0] << 6,
    v_p10_intended[0] << 6,
    u_p10_intended[1] << 6,
    v_p10_intended[1] << 6,
  ];

  // LE-encode every fixture so `<false>` kernels recover the intended
  // host-native samples on both LE and BE hosts via `from_le`.
  let y_p10 = as_le_u16(&y_p10_intended);
  let u_p10 = as_le_u16(&u_p10_intended);
  let v_p10 = as_le_u16(&v_p10_intended);
  let y_p010 = as_le_u16(&y_p010_intended);
  let uv_p010 = as_le_u16(&uv_p010_intended);

  let mut rgb_p10 = [0u8; 12];
  let mut rgb_p010 = [0u8; 12];
  yuv_420p_n_to_rgb_row::<10, false>(
    &y_p10,
    &u_p10,
    &v_p10,
    &mut rgb_p10,
    4,
    ColorMatrix::Bt709,
    true,
  );
  p_n_to_rgb_row::<10, false>(
    &y_p010,
    &uv_p010,
    &mut rgb_p010,
    4,
    ColorMatrix::Bt709,
    true,
  );
  // Parity: same logical samples, same RGB output regardless of layout.
  assert_eq!(rgb_p10, rgb_p010);

  // Independent expected-output assertion against the intended
  // host-native samples. Neutral chroma forces R=G=B per pixel; with
  // full-range BT.709 Y' = Y/1023, the expected 8-bit channel value
  // for pixel `i` is `round(255 * y_p10_intended[i] / 1023)`.
  for i in 0..4 {
    let expected = ((y_p10_intended[i] as u32 * 255 + 511) / 1023) as u8;
    let (r, g, b) = (rgb_p10[i * 3], rgb_p10[i * 3 + 1], rgb_p10[i * 3 + 2]);
    assert_eq!(r, g, "pixel {i}: R==G under neutral chroma");
    assert_eq!(g, b, "pixel {i}: G==B under neutral chroma");
    assert!(
      r.abs_diff(expected) <= 1,
      "pixel {i} (Y={}): got R={r}, expected≈{expected}",
      y_p10_intended[i],
    );
  }
}

// ---- p010_to_rgb_u16_row (P010 → native-depth u16) --------------------

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn p010_rgb_u16_white_full_range() {
  let y = as_le_u16(&[0xFFC0u16; 4]);
  let uv = as_le_u16(&[0x8000u16; 4]);
  let mut rgb = [0u16; 12];
  p_n_to_rgb_u16_row::<10, false>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
  assert!(rgb.iter().all(|&c| c == 1023), "got {rgb:?}");
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn p010_rgb_u16_limited_range_endpoints() {
  let y = as_le_u16(&[0x1000u16, 0xEB00]);
  let uv = as_le_u16(&[0x8000u16, 0x8000]);
  let mut rgb = [0u16; 6];
  p_n_to_rgb_u16_row::<10, false>(&y, &uv, &mut rgb, 2, ColorMatrix::Bt709, false);
  assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
  assert_eq!((rgb[3], rgb[4], rgb[5]), (1023, 1023, 1023));
}

// ---- yuv_444p_n_to_rgba_row (10-bit → u8 RGBA) ----------------------

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv444p10_rgba_gray_alpha_is_ff() {
  // Mid-gray 10-bit Y=512 ↔ 8-bit ≈128. RGBA stride is 4 bytes/px;
  // alpha must be 0xFF on every pixel.
  let y = as_le_u16(&[512u16; 4]);
  let u = as_le_u16(&[512u16; 4]);
  let v = as_le_u16(&[512u16; 4]);
  let mut rgba = [0u8; 16];
  yuv_444p_n_to_rgba_row::<10, false>(&y, &u, &v, &mut rgba, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b, a) = (
      rgba[x * 4],
      rgba[x * 4 + 1],
      rgba[x * 4 + 2],
      rgba[x * 4 + 3],
    );
    assert_eq!(r, g, "RGB should be gray");
    assert_eq!(g, b, "RGB should be gray");
    assert!(r.abs_diff(128) <= 1, "got R={r}");
    assert_eq!(a, 0xFF, "alpha must be 0xFF at px {x}");
  }
}

// ---- yuv_444p_n_to_rgba_u16_row (10-bit → 10-bit u16 RGBA) ---------

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv444p10_rgba_u16_gray_alpha_is_1023() {
  // 10-bit u16 RGBA: alpha element is `(1 << BITS) - 1 = 1023`.
  let y = as_le_u16(&[512u16; 4]);
  let u = as_le_u16(&[512u16; 4]);
  let v = as_le_u16(&[512u16; 4]);
  let mut rgba = [0u16; 16];
  yuv_444p_n_to_rgba_u16_row::<10, false>(&y, &u, &v, &mut rgba, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b, a) = (
      rgba[x * 4],
      rgba[x * 4 + 1],
      rgba[x * 4 + 2],
      rgba[x * 4 + 3],
    );
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(512) <= 1, "got R={r}");
    assert_eq!(a, 1023, "alpha must be (1 << 10) - 1 at px {x}");
  }
}

// ---- yuv_444p16_to_rgba_row (16-bit → u8 RGBA) ----------------------

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv444p16_rgba_gray_alpha_is_ff() {
  // 16-bit mid-gray Y = 0x8000 → 8-bit ≈128. Alpha = 0xFF.
  let y = as_le_u16(&[0x8000u16; 4]);
  let u = as_le_u16(&[0x8000u16; 4]);
  let v = as_le_u16(&[0x8000u16; 4]);
  let mut rgba = [0u8; 16];
  yuv_444p16_to_rgba_row::<false>(&y, &u, &v, &mut rgba, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b, a) = (
      rgba[x * 4],
      rgba[x * 4 + 1],
      rgba[x * 4 + 2],
      rgba[x * 4 + 3],
    );
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(128) <= 1, "got R={r}");
    assert_eq!(a, 0xFF, "alpha must be 0xFF at px {x}");
  }
}

// ---- yuv_444p16_to_rgba_u16_row (16-bit → 16-bit u16 RGBA) ---------

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv444p16_rgba_u16_gray_alpha_is_ffff() {
  // 16-bit u16 RGBA: alpha element is `0xFFFF`.
  let y = as_le_u16(&[0x8000u16; 4]);
  let u = as_le_u16(&[0x8000u16; 4]);
  let v = as_le_u16(&[0x8000u16; 4]);
  let mut rgba = [0u16; 16];
  yuv_444p16_to_rgba_u16_row::<false>(&y, &u, &v, &mut rgba, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b, a) = (
      rgba[x * 4],
      rgba[x * 4 + 1],
      rgba[x * 4 + 2],
      rgba[x * 4 + 3],
    );
    assert_eq!(r, g);
    assert_eq!(g, b);
    // Y=0x8000 in full-range 16→16 maps near 32768; allow rounding.
    assert!(r.abs_diff(0x8000) <= 1, "got R={r}");
    assert_eq!(a, 0xFFFF, "alpha must be 0xFFFF at px {x}");
  }
}

// ---- p_n_444_to_rgba_row (P410 → u8 RGBA) ---------------------------

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn p410_rgba_gray_alpha_is_ff() {
  // P410: 10 active bits in HIGH 10 of each u16. Mid-gray 10-bit
  // Y=512 → P410 Y = 0x8000. UV interleaved: U V U V ... full width.
  let y = as_le_u16(&[0x8000u16; 4]);
  // 4 pixels x (U,V) per pixel = 8 elements.
  let uv = as_le_u16(&[0x8000u16; 8]);
  let mut rgba = [0u8; 16];
  p_n_444_to_rgba_row::<10, false>(&y, &uv, &mut rgba, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b, a) = (
      rgba[x * 4],
      rgba[x * 4 + 1],
      rgba[x * 4 + 2],
      rgba[x * 4 + 3],
    );
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(128) <= 1, "got R={r}");
    assert_eq!(a, 0xFF, "alpha must be 0xFF at px {x}");
  }
}

// ---- p_n_444_16_to_rgba_u16_row (P416 → 16-bit u16 RGBA) -----------

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn p416_rgba_u16_gray_alpha_is_ffff() {
  // P416: full 16-bit samples. Mid-gray Y=0x8000, neutral UV=0x8000.
  // 16-bit u16 RGBA: alpha element is `0xFFFF`.
  let y = as_le_u16(&[0x8000u16; 4]);
  let uv = as_le_u16(&[0x8000u16; 8]);
  let mut rgba = [0u16; 16];
  p_n_444_16_to_rgba_u16_row::<false>(&y, &uv, &mut rgba, 4, ColorMatrix::Bt601, true);
  for x in 0..4 {
    let (r, g, b, a) = (
      rgba[x * 4],
      rgba[x * 4 + 1],
      rgba[x * 4 + 2],
      rgba[x * 4 + 3],
    );
    assert_eq!(r, g);
    assert_eq!(g, b);
    assert!(r.abs_diff(0x8000) <= 1, "got R={r}");
    assert_eq!(a, 0xFFFF, "alpha must be 0xFFFF at px {x}");
  }
}

// ---- Rgbf16 scalar row kernel parity tests ------------------------------
//
// Each test builds a `half::f16` input from a set of representative f32
// values, calls the `rgbf16_to_*_row` kernel, and then calls the matching
// `rgbf32_to_*_row` kernel with the widened f32 slice.  The outputs must
// be identical, proving that widening is the only difference.
//
// `rgbf16_test_inputs` re-encodes the host-native f16/f32 fixtures as
// LE-encoded byte layouts via `as_le_f16` / `as_le_f32`, so the kernels
// (called with `::<false>` ⇒ LE-encoded input) recover the intended
// values via `u16::from_le` / `u32::from_le` on both LE (no-op) and BE
// (byte-swap) hosts.

/// 9 representative half-float inputs (LE-encoded) plus the matching
/// widened f32 slice (also LE-encoded). Each output channel position
/// sees every input value at some pixel.
#[cfg(feature = "rgb-float")]
fn rgbf16_test_inputs() -> (Vec<half::f16>, Vec<f32>, usize) {
  let inputs_f32: [f32; 9] = [0.0, 1.0, 0.5, 65504.0, 1e-5, -0.5, 2.5, 0.999, 0.001];
  let width = inputs_f32.len();
  let rgb_in_host: Vec<half::f16> = (0..width * 3)
    .map(|i| half::f16::from_f32(inputs_f32[i % width]))
    .collect();
  let widened_host: Vec<f32> = rgb_in_host.iter().map(|&h| h.to_f32()).collect();
  (as_le_f16(&rgb_in_host), as_le_f32(&widened_host), width)
}

#[cfg(feature = "rgb-float")]
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_scalar_rgb_matches_widen_then_rgbf32() {
  let (rgb_in, widened, width) = rgbf16_test_inputs();
  let mut out_f16 = std::vec![0u8; width * 3];
  let mut out_via_f32 = std::vec![0u8; width * 3];
  rgbf16_to_rgb_row::<false>(&rgb_in, &mut out_f16, width);
  rgbf32_to_rgb_row::<false>(&widened, &mut out_via_f32, width);
  assert_eq!(out_f16, out_via_f32, "rgbf16_to_rgb scalar parity");
}

#[cfg(feature = "rgb-float")]
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_scalar_rgba_matches_widen_then_rgbf32() {
  let (rgb_in, widened, width) = rgbf16_test_inputs();
  let mut out_f16 = std::vec![0u8; width * 4];
  let mut out_via_f32 = std::vec![0u8; width * 4];
  rgbf16_to_rgba_row::<false>(&rgb_in, &mut out_f16, width);
  rgbf32_to_rgba_row::<false>(&widened, &mut out_via_f32, width);
  assert_eq!(out_f16, out_via_f32, "rgbf16_to_rgba scalar parity");
}

#[cfg(feature = "rgb-float")]
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_scalar_rgb_u16_matches_widen_then_rgbf32() {
  let (rgb_in, widened, width) = rgbf16_test_inputs();
  let mut out_f16 = std::vec![0u16; width * 3];
  let mut out_via_f32 = std::vec![0u16; width * 3];
  rgbf16_to_rgb_u16_row::<false>(&rgb_in, &mut out_f16, width);
  rgbf32_to_rgb_u16_row::<false>(&widened, &mut out_via_f32, width);
  assert_eq!(out_f16, out_via_f32, "rgbf16_to_rgb_u16 scalar parity");
}

#[cfg(feature = "rgb-float")]
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_scalar_rgba_u16_matches_widen_then_rgbf32() {
  let (rgb_in, widened, width) = rgbf16_test_inputs();
  let mut out_f16 = std::vec![0u16; width * 4];
  let mut out_via_f32 = std::vec![0u16; width * 4];
  rgbf16_to_rgba_u16_row::<false>(&rgb_in, &mut out_f16, width);
  rgbf32_to_rgba_u16_row::<false>(&widened, &mut out_via_f32, width);
  assert_eq!(out_f16, out_via_f32, "rgbf16_to_rgba_u16 scalar parity");
}

#[cfg(feature = "rgb-float")]
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_scalar_rgb_f32_matches_element_wise_widen() {
  let (rgb_in, widened, width) = rgbf16_test_inputs();
  let mut out = std::vec![0.0f32; width * 3];
  rgbf16_to_rgb_f32_row::<false>(&rgb_in, &mut out, width);
  // Each output must equal the bit-exact widening of the input f16.
  // Output is host-native f32; `widened` is LE-encoded f32 — compare
  // host-native values by re-interpreting `widened` through the same
  // LE→host-native decode the kernel applied.
  let widened_host: Vec<f32> = widened
    .iter()
    .map(|v| f32::from_bits(u32::from_le(v.to_bits())))
    .collect();
  assert_eq!(
    out, widened_host,
    "rgbf16_to_rgb_f32 must widen without clamping"
  );
}

#[cfg(feature = "rgb-float")]
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_scalar_rgb_f16_is_copy() {
  let (rgb_in, _widened, width) = rgbf16_test_inputs();
  let mut out = std::vec![half::f16::ZERO; width * 3];
  rgbf16_to_rgb_f16_row::<false>(&rgb_in, &mut out, width);
  // Output is host-native f16; `rgb_in` is LE-encoded f16 — recover the
  // host-native values for the lossless equality check.
  let rgb_in_host: Vec<half::f16> = rgb_in
    .iter()
    .map(|v| half::f16::from_bits(u16::from_le(v.to_bits())))
    .collect();
  assert_eq!(
    out, rgb_in_host,
    "rgbf16_to_rgb_f16 must be a byte-identical copy"
  );
}

// ---- Tier 5.25 packed YUV 4:1:1 — UYYVYY411 -------------------------

/// Builds a single-row UYYVYY411 packed buffer with constant
/// `(Y, U, V)` per 6-byte / 4-pixel block.
#[cfg(feature = "yuv-packed")]
fn uyyvyy411_solid_row(width: usize, y: u8, u: u8, v: u8) -> std::vec::Vec<u8> {
  assert_eq!(width & 3, 0);
  let mut buf = std::vec![0u8; width * 3 / 2];
  for col in (0..width).step_by(4) {
    let blk = (col / 4) * 6;
    buf[blk] = u;
    buf[blk + 1] = y;
    buf[blk + 2] = y;
    buf[blk + 3] = v;
    buf[blk + 4] = y;
    buf[blk + 5] = y;
  }
  buf
}

#[cfg(feature = "yuv-packed")]
#[test]
fn uyyvyy411_to_rgb_row_solid_gray_full_range() {
  // (Y=128, U=V=128) full-range BT.601 → mid-gray RGB (≈128, 128, 128).
  let w = 16;
  let p = uyyvyy411_solid_row(w, 128, 128, 128);
  let mut rgb = std::vec![0u8; w * 3];
  uyyvyy411_to_rgb_row(&p, &mut rgb, w, ColorMatrix::Bt601, true);
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[cfg(feature = "yuv-packed")]
#[test]
fn uyyvyy411_to_rgba_row_solid_gray_alpha_opaque() {
  let w = 16;
  let p = uyyvyy411_solid_row(w, 128, 128, 128);
  let mut rgba = std::vec![0u8; w * 4];
  uyyvyy411_to_rgba_row(&p, &mut rgba, w, ColorMatrix::Bt601, true);
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }
}

#[cfg(feature = "yuv-packed")]
#[test]
fn uyyvyy411_to_luma_row_extracts_y_at_offsets_1_2_4_5() {
  // Hand-crafted 1 block: U=10, Y0=20, Y1=30, V=40, Y2=50, Y3=60.
  let p = std::vec![10u8, 20, 30, 40, 50, 60];
  let mut luma = std::vec![0u8; 4];
  uyyvyy411_to_luma_row(&p, &mut luma, 4);
  assert_eq!(luma, std::vec![20u8, 30, 50, 60]);
}

#[cfg(feature = "yuv-packed")]
#[test]
fn uyyvyy411_to_luma_u16_row_zero_extends_y_bytes() {
  let p = std::vec![10u8, 20, 30, 40, 50, 60];
  let mut luma = std::vec![0u16; 4];
  uyyvyy411_to_luma_u16_row(&p, &mut luma, 4);
  assert_eq!(luma, std::vec![20u16, 30, 50, 60]);
}

#[cfg(feature = "yuv-packed")]
#[test]
fn uyyvyy411_chroma_shared_across_4_pixels_decodes_via_y_only_variation() {
  // With Y0..Y3 distinct but shared (U, V), each output pixel must
  // map back to a Y-distinct grayscale (since U=V=128 zeroes the
  // chroma contribution under BT.601 full-range).
  // Block: U=128, Y0=64, Y1=96, V=128, Y2=160, Y3=224.
  let p = std::vec![128u8, 64, 96, 128, 160, 224];
  let mut rgb = std::vec![0u8; 4 * 3];
  uyyvyy411_to_rgb_row(&p, &mut rgb, 4, ColorMatrix::Bt601, true);
  let lumas = [64u8, 96, 160, 224];
  for (i, expected) in lumas.iter().enumerate() {
    let r = rgb[i * 3];
    let g = rgb[i * 3 + 1];
    let b = rgb[i * 3 + 2];
    assert_eq!(r, g, "px {i} R==G");
    assert_eq!(g, b, "px {i} G==B");
    assert!(
      r.abs_diff(*expected) <= 1,
      "px {i} expected {expected}, got {r}"
    );
  }
}

// ---- BT.601-equivalent matrices (issue #305) -------------------------

// SMPTE 170M (BT.601 525-line) and BT.470 System BG (BT.601 625-line)
// carry the *identical* Kr=0.299/Kb=0.114 coefficients as BT.601, so
// their YUV→RGB and RGB→luma constants must equal BT.601's — not fall
// through to the BT.709 fallback arm.

#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
fn coeffs_tuple(m: ColorMatrix) -> (i32, i32, i32, i32, i32, i32) {
  // `Coefficients` is internal and intentionally not `PartialEq`; compare
  // through its public accessors instead.
  let c = Coefficients::for_matrix(m);
  (c.r_u(), c.r_v(), c.g_u(), c.g_v(), c.b_u(), c.b_v())
}

#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[test]
fn smpte170m_bt470bg_use_bt601_yuv_coefficients() {
  let bt601 = coeffs_tuple(ColorMatrix::Bt601);
  assert_eq!(coeffs_tuple(ColorMatrix::Smpte170M), bt601);
  assert_eq!(coeffs_tuple(ColorMatrix::Bt470Bg), bt601);
  // Regression guard: must no longer match the BT.709 fallback.
  assert_ne!(coeffs_tuple(ColorMatrix::Bt709), bt601);
}

#[test]
fn smpte170m_bt470bg_use_bt601_luma_coefficients() {
  let bt601 = luma_coefficients_q15(ColorMatrix::Bt601);
  assert_eq!(luma_coefficients_q15(ColorMatrix::Smpte170M), bt601);
  assert_eq!(luma_coefficients_q15(ColorMatrix::Bt470Bg), bt601);
  // Regression guard: must no longer match the BT.709 fallback.
  assert_ne!(luma_coefficients_q15(ColorMatrix::Bt709), bt601);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420_smpte170m_decodes_identically_to_bt601() {
  // Chroma-bearing input (V offset from neutral) so the matrix actually
  // affects the result. Smpte170M must reproduce the BT.601 decode and
  // diverge from the (previously wrongly applied) BT.709 fallback.
  let y = [128u8; 4];
  let u = [128u8; 2];
  let v = [200u8; 2];
  let mut rgb_601 = [0u8; 12];
  let mut rgb_170m = [0u8; 12];
  let mut rgb_709 = [0u8; 12];
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_601, 4, ColorMatrix::Bt601, true);
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_170m, 4, ColorMatrix::Smpte170M, true);
  yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_709, 4, ColorMatrix::Bt709, true);
  assert_eq!(rgb_170m, rgb_601, "Smpte170M must match BT.601");
  assert_ne!(rgb_170m, rgb_709, "Smpte170M must differ from BT.709");
}

// ---- ChromaDerivedNcl primaries-derived coefficients (#303) -----------

/// Asserts every Q15 entry of `got` is within `tol` of `want`.
#[cfg(feature = "yuv-planar")]
fn assert_coeffs_within(got: &Coefficients, want: &Coefficients, tol: i32, ctx: &str) {
  for (name, g, w) in [
    ("r_u", got.r_u(), want.r_u()),
    ("r_v", got.r_v(), want.r_v()),
    ("g_u", got.g_u(), want.g_u()),
    ("g_v", got.g_v(), want.g_v()),
    ("b_u", got.b_u(), want.b_u()),
    ("b_v", got.b_v(), want.b_v()),
  ] {
    assert!(
      (g - w).abs() <= tol,
      "{ctx}: {name} got {g} want {w} (tol {tol})"
    );
  }
}

/// THE GATE (BT.709): `ChromaDerivedNcl` with BT.709 primaries must
/// reproduce the hard-coded `Bt709` affine arm within Q15 rounding. The
/// stored arm uses published/rounded coefficients, so a pure chromaticity
/// derivation lands within ~6 LSB — a wrong derivation (or a BT.709 *vs*
/// BT.2020 mix-up) is off by thousands, which `tol = 8` cleanly rejects.
#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_derived_ncl_bt709_matches_hardcoded_bt709() {
  let derived =
    Coefficients::for_matrix_with_primaries(ColorMatrix::ChromaDerivedNcl, Primaries::Bt709);
  let bt709 = Coefficients::for_matrix(ColorMatrix::Bt709);
  assert_coeffs_within(&derived, &bt709, 8, "ChromaDerivedNcl(Bt709) vs Bt709");
  // Discrimination: it consulted the primaries, not a fixed fallback — it
  // must NOT equal the BT.2020 set (r_v differs by >3000 LSB).
  let bt2020 = Coefficients::for_matrix(ColorMatrix::Bt2020Ncl);
  assert!(
    (derived.r_v() - bt2020.r_v()).abs() > 100,
    "ChromaDerivedNcl(Bt709) must differ from Bt2020Ncl"
  );
}

/// THE GATE (BT.2020): `ChromaDerivedNcl` with BT.2020 primaries must
/// reproduce the hard-coded `Bt2020Ncl` arm within Q15 rounding (~5 LSB),
/// and be clearly distinct from BT.709.
#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_derived_ncl_bt2020_matches_hardcoded_bt2020ncl() {
  let derived =
    Coefficients::for_matrix_with_primaries(ColorMatrix::ChromaDerivedNcl, Primaries::Bt2020);
  let bt2020 = Coefficients::for_matrix(ColorMatrix::Bt2020Ncl);
  assert_coeffs_within(
    &derived,
    &bt2020,
    8,
    "ChromaDerivedNcl(Bt2020) vs Bt2020Ncl",
  );
  let bt709 = Coefficients::for_matrix(ColorMatrix::Bt709);
  assert!(
    (derived.r_v() - bt709.r_v()).abs() > 100,
    "ChromaDerivedNcl(Bt2020) must differ from Bt709"
  );
}

/// THE GATE (BT.601 lineage): BT.601's `Kr = 0.299 / Kb = 0.114` originate
/// from the **NTSC-1953** primaries (mediaframe [`Primaries::Bt470M`], white
/// C), so `ChromaDerivedNcl` with those primaries reproduces the BT.601 luma
/// weights. (Note: SMPTE-170M primaries are SMPTE-C and derive to
/// ~0.212/0.087 instead — see the guard test below; the issue's
/// "Smpte170M → Bt601" assumption attributes the coefficients to the wrong
/// primary set.)
#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_derived_ncl_bt470m_weights_are_bt601() {
  let (kr, kg, kb) =
    chroma_derived_luma_weights(Primaries::Bt470M).expect("Bt470M carries primaries");
  assert!((kr - 0.299).abs() < 1.5e-3, "Kr {kr} (want ~0.299)");
  assert!((kg - 0.587).abs() < 1.5e-3, "Kg {kg} (want ~0.587)");
  assert!((kb - 0.114).abs() < 1.5e-3, "Kb {kb} (want ~0.114)");
}

/// Guards the corrected premise: SMPTE-170M primaries (SMPTE-C) derive to
/// the SMPTE-240M-class weights (~0.2124 / 0.0866), NOT BT.601's
/// 0.299 / 0.114.
#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_derived_ncl_smpte170m_primaries_are_smpte_c_not_bt601() {
  let (kr, _kg, kb) =
    chroma_derived_luma_weights(Primaries::Smpte170M).expect("Smpte170M carries primaries");
  assert!(
    (kr - 0.2124).abs() < 1e-3 && (kb - 0.0866).abs() < 1e-3,
    "SMPTE-C weights expected, got Kr={kr} Kb={kb}"
  );
  assert!((kr - 0.299).abs() > 0.05, "must NOT be BT.601 Kr=0.299");
}

/// Independent consistency (no hard-coded coefficient table): the derived
/// weights `S = (Kr, Kg, Kb)` must sum to 1 and satisfy the *defining*
/// linear system `M · S = W_xyz` for every primary set — i.e. they
/// reconstruct the white point's XYZ.
#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_derived_weights_reconstruct_white_point() {
  for p in [
    Primaries::Bt709,
    Primaries::Bt2020,
    Primaries::Smpte170M,
    Primaries::Bt470M,
    Primaries::Bt470Bg,
    Primaries::SmpteRp431,
    Primaries::Film,
  ] {
    let (kr, kg, kb) = chroma_derived_luma_weights(p).expect("carries primaries");
    assert!(
      (kr + kg + kb - 1.0).abs() < 1e-9,
      "{p:?}: weights sum {} != 1",
      kr + kg + kb
    );
    let rgb = p.chromaticities().unwrap();
    let white = p.white_point().unwrap();
    let xyz = |rx: u32, ry: u32| {
      let x = rx as f64 / 50_000.0;
      let y = ry as f64 / 50_000.0;
      [x / y, 1.0, (1.0 - x - y) / y]
    };
    let r = xyz(rgb[0].x(), rgb[0].y());
    let g = xyz(rgb[1].x(), rgb[1].y());
    let b = xyz(rgb[2].x(), rgb[2].y());
    let w = xyz(white.x(), white.y());
    for axis in 0..3 {
      let recon = r[axis] * kr + g[axis] * kg + b[axis] * kb;
      assert!(
        (recon - w[axis]).abs() < 1e-9,
        "{p:?} axis {axis}: reconstructed {recon} != white {}",
        w[axis]
      );
    }
  }
}

/// `ChromaDerivedNcl` without usable primaries (`Unspecified`) falls back to
/// exactly the BT.709 coefficients, preserving the pre-#303 behaviour.
#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_derived_ncl_without_primaries_falls_back_to_bt709() {
  let fallback =
    Coefficients::for_matrix_with_primaries(ColorMatrix::ChromaDerivedNcl, Primaries::Unspecified);
  let bt709 = Coefficients::for_matrix(ColorMatrix::Bt709);
  assert_coeffs_within(
    &fallback,
    &bt709,
    0,
    "ChromaDerivedNcl(Unspecified) == Bt709",
  );
}

/// SMPTE ST 428-1 (#310), FFmpeg-tabulated interpretation (the default):
/// `ChromaDerivedNcl` with `SmpteSt428` primaries derives its luma weights from
/// FFmpeg's tabulated D-Cinema RGB chromaticities — the pre-#310 behaviour,
/// unchanged. Pins the derivation to those tabulated weights (a proper convex
/// set that reconstructs the white point) and confirms it is *not* the
/// degenerate CIE-XYZ identity. The CIE-XYZ interpretation is rejected upstream
/// by the sink guard (`St428Interpretation::CieXyz`), not here.
#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_derived_ncl_smpte_st428_derives_from_tabulated_primaries() {
  let (kr, kg, kb) = chroma_derived_luma_weights(Primaries::SmpteSt428)
    .expect("SmpteSt428 carries FFmpeg's tabulated primaries + white point E");
  assert!(
    (kr + kg + kb - 1.0).abs() < 1e-9,
    "tabulated weights sum {} != 1",
    kr + kg + kb
  );
  assert!(
    kr > 0.0 && kg > 0.0 && kb > 0.0,
    "tabulated D-Cinema weights are all positive: ({kr}, {kg}, {kb})"
  );
  // A degenerate CIE-XYZ reading (channels are X/Y/Z) would collapse all luma
  // onto the Y axis (kg ≈ 1); the tabulated RGB derivation must not.
  assert!(
    kg < 0.999,
    "tabulated derivation must not collapse to the XYZ Y-axis identity (kg={kg})"
  );
  // The wired `ChromaDerivedNcl` derivation resolves through exactly these
  // tabulated weights — byte-identical to the pre-#310 coefficients.
  let derived =
    Coefficients::for_matrix_with_primaries(ColorMatrix::ChromaDerivedNcl, Primaries::SmpteSt428);
  let from_weights = Coefficients::from_luma_weights(kr, kg, kb);
  assert_coeffs_within(
    &derived,
    &from_weights,
    0,
    "ChromaDerivedNcl(SmpteSt428) == tabulated from_luma_weights",
  );
}

/// Every fixed matrix resolves byte-identically through
/// `for_matrix_with_primaries` regardless of the primaries — the BT.601 /
/// BT.709 grouping (and all others) is unaffected by the new parameter.
#[cfg(feature = "yuv-planar")]
#[test]
fn for_matrix_with_primaries_ignores_primaries_for_fixed_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::YCgCo,
    ColorMatrix::Fcc,
    ColorMatrix::Smpte170M,
    ColorMatrix::Bt470Bg,
  ] {
    let base = Coefficients::for_matrix(m);
    for p in [
      Primaries::Bt709,
      Primaries::Bt2020,
      Primaries::Unspecified,
      Primaries::Smpte170M,
    ] {
      let with = Coefficients::for_matrix_with_primaries(m, p);
      assert_coeffs_within(&with, &base, 0, "fixed matrix must ignore primaries");
    }
  }
}

/// End-to-end YUV→RGB row through the wired `ChromaDerivedNcl` dispatcher
/// ([`yuv_420_to_rgb_row_primaries`]) for a **non-standard** primaries set
/// (BT.470BG / PAL), compared against an *independent* `f64` derivation of
/// the same affine. Also pins the derived weights to externally-computed
/// values so the match is not self-referential.
#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_derived_ncl_bt470bg_row_matches_independent_derivation() {
  use crate::row::yuv_420_to_rgb_row_primaries;

  let (kr, kg, kb) =
    chroma_derived_luma_weights(Primaries::Bt470Bg).expect("Bt470Bg carries primaries");
  // Pin to the values computed offline from the BT.470BG chromaticities
  // (EBU/PAL-class, distinct from BT.601/709/2020): an external anchor.
  assert!((kr - 0.222_004).abs() < 1e-5, "Kr {kr}");
  assert!((kg - 0.706_655).abs() < 1e-5, "Kg {kg}");
  assert!((kb - 0.071_341).abs() < 1e-5, "Kb {kb}");

  // Independent continuous-domain affine (NOT via `Coefficients` / Q15).
  let r_v = 2.0 * (1.0 - kr);
  let b_u = 2.0 * (1.0 - kb);
  let g_u = -2.0 * (1.0 - kb) * kb / kg;
  let g_v = -2.0 * (1.0 - kr) * kr / kg;

  // Full-range pixels (no Y offset/scale): two chroma-bearing pairs.
  let y = [128u8, 60, 200, 130];
  let u = [160u8, 90];
  let v = [96u8, 210];
  let mut got = [0u8; 12];
  yuv_420_to_rgb_row_primaries(
    &y,
    &u,
    &v,
    &mut got,
    4,
    ColorMatrix::ChromaDerivedNcl,
    Primaries::Bt470Bg,
    true,
    true,
  );
  let clamp = |x: f64| x.round().clamp(0.0, 255.0) as i32;
  for px in 0..4 {
    let yv = y[px] as f64;
    let ud = u[px / 2] as f64 - 128.0;
    let vd = v[px / 2] as f64 - 128.0;
    let want = [
      clamp(yv + r_v * vd),
      clamp(yv + g_u * ud + g_v * vd),
      clamp(yv + b_u * ud),
    ];
    for ch in 0..3 {
      assert!(
        (got[px * 3 + ch] as i32 - want[ch]).abs() <= 1,
        "px{px} ch{ch}: kernel {} vs independent {}",
        got[px * 3 + ch],
        want[ch]
      );
    }
  }
}

// ---- 4:4:0 bottom-sited vertical chroma reconstruction ----------------------

#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_upsample_440_bottom_v_matches_oracle() {
  // Full-width vertical rounding-average `(prev + cur + 1) >> 1`, across widths
  // that straddle every SIMD chunk boundary the arch backends use. `prev` and
  // `cur` carry OPPOSITE column parity (the even `+ 198` makes `cur ≡ i` mod 2
  // against `prev ≡ i + 1`), so every `prev[j] + cur[j]` is ODD — this
  // exercises the `+ 1` rounding a truncating `(a + b) >> 1` backend gets wrong.
  for width in [1usize, 2, 7, 15, 16, 17, 31, 33, 64, 65, 129] {
    let prev: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let cur: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 198) & 0xFF) as u8).collect();
    let mut out = std::vec![0u8; width];
    chroma_upsample_440_bottom_v(&prev, &cur, &mut out, width);
    for j in 0..width {
      let expected = (((prev[j] as u16) + (cur[j] as u16) + 1) >> 1) as u8;
      assert_eq!(out[j], expected, "width={width} j={j}");
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_upsample_440_bottom_v_prev_equals_cur_is_identity() {
  // The top-edge clamp replicates the previous chroma row from the current one;
  // with `prev == cur` the box average collapses to `cur` exactly.
  let cur: std::vec::Vec<u8> = (0..40).map(|i| ((i * 7 + 3) & 0xFF) as u8).collect();
  let mut out = std::vec![0u8; 40];
  chroma_upsample_440_bottom_v(&cur, &cur, &mut out, 40);
  assert_eq!(out, cur, "prev == cur must reproduce cur exactly");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn chroma_upsample_440_bottom_v_u16_matches_oracle_all_depths() {
  // The average is bit-depth-independent, so the same pure `(prev + cur + 1)
  // >> 1` holds for any sample range; exercise a 10-bit and a full 16-bit one.
  for &mask in &[0x3FFu32, 0xFFFFu32] {
    for width in [1usize, 7, 8, 9, 15, 16, 17, 33, 64, 65, 129] {
      let prev: std::vec::Vec<u16> = (0..width)
        .map(|i| ((i as u32).wrapping_mul(2654435761) & mask) as u16)
        .collect();
      let cur: std::vec::Vec<u16> = (0..width)
        .map(|i| ((i as u32).wrapping_mul(40503).wrapping_add(12345) & mask) as u16)
        .collect();
      let mut out = std::vec![0u16; width];
      chroma_upsample_440_bottom_v_u16(&prev, &cur, &mut out, width);
      for j in 0..width {
        let expected = (((prev[j] as u32) + (cur[j] as u32) + 1) >> 1) as u16;
        assert_eq!(out[j], expected, "mask={mask:#x} width={width} j={j}");
      }
    }
  }
}
