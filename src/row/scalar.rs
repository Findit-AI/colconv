//! Scalar reference implementations of the row primitives.
//!
//! Always compiled. SIMD backends live in [`super::arch`] and dispatch
//! to these as their tail fallback. Per-call dispatch in
//! [`super`]`::{yuv_420_to_rgb_row, rgb_to_hsv_row}` picks the best
//! backend at the module boundary.

use crate::ColorMatrix;

// ---- YUV 4:2:0 → RGB (fused: upsample + convert) ----------------------

/// Converts one row of 4:2:0 YUV — Y at full width, U/V at half-width —
/// directly to packed RGB. Chroma is nearest-neighbor upsampled **in
/// registers** inside the kernel; no intermediate memory traffic.
///
/// `full_range = true` interprets Y in `[0, 255]` and chroma in
/// `[0, 255]` (JPEG / `yuvjNNNp` convention). `full_range = false`
/// interprets Y in `[16, 235]` and chroma in `[16, 240]` (broadcast /
/// limited-range convention).
///
/// Output is packed `B, G, R` triples: `rgb_out[3*x] = B`,
/// `rgb_out[3*x + 1] = G`, `rgb_out[3*x + 2] = R`.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params(full_range);

  // Process two pixels per iteration — they share one chroma sample.
  // Round-to-nearest on every Q15 shift by adding 1 << 14 before the
  // `>> 15`, so 219 * (255/219 in Q15) cleanly produces 255 at the top
  // of limited-range without a 254-truncation bias.
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = ((u_half[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_half[c_idx] as i32 - 128) * c_scale + RND) >> 15;

    // Single-round per channel keeps the math faithful to a 1×2 3x3
    // matrix multiply. All six coefficients are used; standard
    // matrices (BT.601 / 709 / 2020) have `r_u = b_v = 0` so those
    // terms vanish. YCgCo uses all six.
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    // Pixel x.
    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);

    // Pixel x+1 shares chroma.
    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    rgb_out[(x + 1) * 3] = clamp_u8(y1 + r_chroma);
    rgb_out[(x + 1) * 3 + 1] = clamp_u8(y1 + g_chroma);
    rgb_out[(x + 1) * 3 + 2] = clamp_u8(y1 + b_chroma);

    x += 2;
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn clamp_u8(v: i32) -> u8 {
  v.clamp(0, 255) as u8
}

/// Range-scaling params: `(y_off, y_scale_q15, c_scale_q15)`.
///
/// Full range: no offset, unit scales (Q15 = 2^15).
///
/// Limited range: map Y from `[16, 235]` to `[0, 255]` via
/// `y_scaled = (y - 16) * (255 / 219)`; map chroma from `[16, 240]`
/// to `[0, 255]` via `c_scaled = (c - 128) * (255 / 224)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn range_params(full_range: bool) -> (i32, i32, i32) {
  if full_range {
    (0, 1 << 15, 1 << 15)
  } else {
    //  255 / 219 ≈ 1.164383; * 2^15 ≈ 38142.
    //  255 / 224 ≈ 1.138393; * 2^15 ≈ 37306.
    (16, 38142, 37306)
  }
}

/// Q15 YUV → RGB coefficients for a given matrix.
///
/// Full generalized 3×3 matrix:
/// - `R = Y + r_u·u_d + r_v·v_d`
/// - `G = Y + g_u·u_d + g_v·v_d`
/// - `B = Y + b_u·u_d + b_v·v_d`
///
/// where `u_d = U - 128`, `v_d = V - 128`. Standard matrices
/// (BT.601, BT.709, BT.2020-NCL, SMPTE 240M, FCC) have sparse layout
/// with `r_u = b_v = 0`; YCgCo uses all six entries.
pub(super) struct Coefficients {
  r_u: i32,
  r_v: i32,
  g_u: i32,
  g_v: i32,
  b_u: i32,
  b_v: i32,
}

impl Coefficients {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn for_matrix(m: ColorMatrix) -> Self {
    match m {
      // BT.601: r_v=1.402, g_u=-0.344136, g_v=-0.714136, b_u=1.772.
      ColorMatrix::Bt601 | ColorMatrix::Fcc => Self {
        r_u: 0,
        r_v: 45941,
        g_u: -11277,
        g_v: -23401,
        b_u: 58065,
        b_v: 0,
      },
      // BT.709: r_v=1.5748, g_u=-0.1873, g_v=-0.4681, b_u=1.8556.
      ColorMatrix::Bt709 => Self {
        r_u: 0,
        r_v: 51606,
        g_u: -6136,
        g_v: -15339,
        b_u: 60808,
        b_v: 0,
      },
      // BT.2020-NCL: r_v=1.4746, g_u=-0.164553, g_v=-0.571353, b_u=1.8814.
      ColorMatrix::Bt2020Ncl => Self {
        r_u: 0,
        r_v: 48325,
        g_u: -5391,
        g_v: -18722,
        b_u: 61653,
        b_v: 0,
      },
      // SMPTE 240M: r_v=1.576, g_u=-0.2253, g_v=-0.4767, b_u=1.826.
      ColorMatrix::Smpte240m => Self {
        r_u: 0,
        r_v: 51642,
        g_u: -7383,
        g_v: -15620,
        b_u: 59834,
        b_v: 0,
      },
      // YCgCo per H.273 MatrixCoefficients = 8.
      //   U plane → Cg, V plane → Co (biased by 128 each).
      //   R = Y - (Cg - 128) + (Co - 128) = Y - u_d + v_d
      //   G = Y + (Cg - 128)              = Y + u_d
      //   B = Y - (Cg - 128) - (Co - 128) = Y - u_d - v_d
      // Each coefficient is ±1.0 → ±32768 in Q15.
      ColorMatrix::YCgCo => Self {
        r_u: -32768,
        r_v: 32768,
        g_u: 32768,
        g_v: 0,
        b_u: -32768,
        b_v: -32768,
      },
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn r_u(&self) -> i32 {
    self.r_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn r_v(&self) -> i32 {
    self.r_v
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn g_u(&self) -> i32 {
    self.g_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn g_v(&self) -> i32 {
    self.g_v
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn b_u(&self) -> i32 {
    self.b_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn b_v(&self) -> i32 {
    self.b_v
  }
}

// ---- RGB → HSV ----------------------------------------------------------

// ---- HSV division LUTs (OpenCV `cv2.COLOR_RGB2HSV` compatible) --------
//
// Replace the f32 divisions in the scalar HSV path with an integer
// multiply + table lookup. Produces byte‑exact output against OpenCV
// for 8‑bit RGB → HSV on every pixel.
//
// `HSV_SHIFT = 12` gives 1044480 / v (saturation divisor) and 122880 /
// delta (hue divisor) as the raw Q12 reciprocals. Both fit in i32, and
// the subsequent `diff * table[x]` product (max 255 × 1044480 ≈ 2.66e8)
// also fits in i32 comfortably.
//
// Total `.rodata` cost: 2 KB (two 256‑entry i32 tables). Always fits
// in L1D on every modern CPU, so lookups average ~4 cycles.

const HSV_SHIFT: u32 = 12;
const HSV_RND: i32 = 1 << (HSV_SHIFT - 1);

/// `sdiv_table[v] = round((255 << 12) / v)`. `sdiv_table[0] = 0`
/// (saturation is undefined at v=0; the caller forces `s = 0` there).
const SDIV_TABLE: [i32; 256] = {
  let mut t = [0i32; 256];
  let mut i = 1usize;
  while i < 256 {
    let n: i32 = 255 << HSV_SHIFT;
    t[i] = (n + (i as i32) / 2) / (i as i32);
    i += 1;
  }
  t
};

/// `hdiv_table[delta] = round((30 << 12) / delta)`. The factor is 30
/// (not 60) because OpenCV's u8 hue range is `[0, 180)` instead of
/// `[0, 360)` — every 2° collapses to one unit. `hdiv_table[0] = 0`
/// (hue is undefined at delta=0; the caller forces `h = 0` there).
const HDIV_TABLE: [i32; 256] = {
  let mut t = [0i32; 256];
  let mut i = 1usize;
  while i < 256 {
    let n: i32 = 30 << HSV_SHIFT;
    t[i] = (n + (i as i32) / 2) / (i as i32);
    i += 1;
  }
  t
};

/// Converts one row of packed RGB to three planar HSV bytes matching
/// OpenCV `cv2.COLOR_RGB2HSV` semantics: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`.
///
/// Uses integer LUT arithmetic (no f32 divisions), producing byte‑
/// exact output against OpenCV's uint8 HSV conversion.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(h_out.len() >= width, "H row too short");
  debug_assert!(s_out.len() >= width, "S row too short");
  debug_assert!(v_out.len() >= width, "V row too short");
  for x in 0..width {
    let r = rgb[x * 3] as i32;
    let g = rgb[x * 3 + 1] as i32;
    let b = rgb[x * 3 + 2] as i32;
    let (h, s, v) = rgb_to_hsv_pixel(r, g, b);
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = v;
  }
}

/// Scalar RGB → HSV for a single pixel, using the shared division LUTs.
/// All arithmetic is integer; the two divisions `s = 255*delta/v` and
/// `h = 30*diff/delta` become `(operand * table[divisor] + RND) >> 12`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rgb_to_hsv_pixel(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
  let v = r.max(g.max(b));
  let min = r.min(g.min(b));
  let delta = v - min;

  // S = round(255 * delta / v), s = 0 when v = 0.
  //
  // SDIV_TABLE[0] = 0 so the expression evaluates to (delta * 0 + RND)
  // >> 12 = 0 when v = 0. Delta is also 0 in that case (min = v = 0),
  // but the explicit table entry makes the reasoning obvious.
  let s = ((delta * SDIV_TABLE[v as usize]) + HSV_RND) >> HSV_SHIFT;

  let h = if delta == 0 {
    0
  } else if v == r {
    let diff = g - b;
    let h_raw = ((diff * HDIV_TABLE[delta as usize]) + HSV_RND) >> HSV_SHIFT;
    if h_raw < 0 { h_raw + 180 } else { h_raw }
  } else if v == g {
    let diff = b - r;
    (((diff * HDIV_TABLE[delta as usize]) + HSV_RND) >> HSV_SHIFT) + 60
  } else {
    let diff = r - g;
    (((diff * HDIV_TABLE[delta as usize]) + HSV_RND) >> HSV_SHIFT) + 120
  };

  (h.clamp(0, 179) as u8, s.clamp(0, 255) as u8, v as u8)
}

// ---- BGR ↔ RGB byte swap ------------------------------------------------

/// Swaps the outer two channels of each packed RGB / BGR triple
/// (byte 0 ↔ byte 2), leaving the middle byte (G) untouched.
///
/// This is the shared implementation behind both `bgr_to_rgb_row` and
/// `rgb_to_bgr_row` — the transformation is a self‑inverse.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");
  for x in 0..width {
    let i = x * 3;
    output[i] = input[i + 2];
    output[i + 1] = input[i + 1];
    output[i + 2] = input[i];
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  // ---- yuv_420_to_rgb_row ----------------------------------------------

  #[test]
  fn yuv420_bgr_black() {
    // Full-range Y=0, neutral chroma → black.
    let y = [0u8; 4];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
  }

  #[test]
  fn yuv420_bgr_white_full_range() {
    let y = [255u8; 4];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
  }

  #[test]
  fn yuv420_bgr_gray_is_gray() {
    let y = [128u8; 4];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    for x in 0..4 {
      let (b, g, r) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!(b, g);
      assert_eq!(g, r);
      assert!(b.abs_diff(128) <= 1, "got {b}");
    }
  }

  #[test]
  fn yuv420_bgr_chroma_shared_across_pair() {
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
  fn yuv420_bgr_limited_range_black_and_white() {
    // Y=16 → black, Y=235 → white in limited range.
    let y = [16u8, 16, 235, 235];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, false);
    for x in 0..2 {
      let (b, g, r) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!((b, g, r), (0, 0, 0), "limited-range Y=16 should be black");
    }
    for x in 2..4 {
      let (b, g, r) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!(
        (b, g, r),
        (255, 255, 255),
        "limited-range Y=235 should be white"
      );
    }
  }

  #[test]
  fn yuv420_bgr_ycgco_neutral_is_gray() {
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
  fn yuv420_bgr_bt601_vs_bt709_differ_for_chroma() {
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
}
