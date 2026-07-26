use super::*;

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

/// Q15 RGB → luma coefficients `(k_r, k_g, k_b)` for a given color
/// matrix. `k_r + k_g + k_b ≈ 32768` (1.0 in Q15) — minor rounding
/// imbalance is below quantization noise. Used by
/// [`rgb_to_luma_row`] to derive the Y' channel from packed RGB
/// sources (Tier 6 / Ship 9a) — also re-exported via
/// `crate::row::scalar` so the per-arch SIMD backends can hoist the
/// per-matrix constants outside their main loops.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn luma_coefficients_q15(matrix: ColorMatrix) -> (i32, i32, i32) {
  match matrix {
    // BT.601: Kr=0.299, Kg=0.587, Kb=0.114. SMPTE 170M (525-line) and
    // BT.470 System BG (625-line) carry the identical coefficients, so
    // they share this arm; FCC (Kr=0.30/Kb=0.11) is a close approximation.
    ColorMatrix::Bt601 | ColorMatrix::Fcc | ColorMatrix::Smpte170M | ColorMatrix::Bt470Bg => {
      (9798, 19235, 3735)
    }
    // BT.709: Kr=0.2126, Kg=0.7152, Kb=0.0722.
    ColorMatrix::Bt709 => (6966, 23436, 2366),
    // BT.2020-NCL: Kr=0.2627, Kg=0.6780, Kb=0.0593.
    ColorMatrix::Bt2020Ncl => (8607, 22217, 1944),
    // SMPTE 240M: Kr=0.212, Kg=0.701, Kb=0.087.
    ColorMatrix::Smpte240m => (6947, 22971, 2851),
    // YCgCo: Y = 0.25 R + 0.5 G + 0.25 B (lossless integer).
    ColorMatrix::YCgCo => (8192, 16384, 8192),
    // ColorMatrix is #[non_exhaustive] in mediaframe; fall back to BT.709
    // for any future variants added there before pixon is updated.
    _ => (6966, 23436, 2366),
  }
}

/// Derives luma (Y') from packed RGB into a single-channel `u8`
/// plane.
///
/// `matrix` selects the BT.* coefficient set;
/// `full_range = true` produces full-range Y' in `[0, 255]`,
/// `full_range = false` produces limited-range Y' in `[16, 235]`
/// (the standard YUV studio range).
///
/// # Panics
///
/// Panics (in any build profile, not just debug) if
/// `rgb.len() < 3 * width` or `luma_out.len() < width` — the inner
/// loop indexes `rgb[x * 3 + i]` and `luma_out[x]` directly, so
/// undersized slices fault on bounds-check rather than producing
/// undefined output. The `debug_assert!`s below add a clearer
/// message in debug builds; the bounds check is unconditional.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb_to_luma_row(
  rgb: &[u8],
  luma_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  let (k_r, k_g, k_b) = luma_coefficients_q15(matrix);
  const RND: i32 = 1 << 14;

  if full_range {
    for x in 0..width {
      let r = rgb[x * 3] as i32;
      let g = rgb[x * 3 + 1] as i32;
      let b = rgb[x * 3 + 2] as i32;
      // Q15 weighted sum: each k_x ≤ 32768, sample ≤ 255, so each
      // term ≤ 8.4M; sum ≤ 32768 * 255 ≈ 8.4M ≪ i32::MAX.
      let y = (k_r * r + k_g * g + k_b * b + RND) >> 15;
      luma_out[x] = y.clamp(0, 255) as u8;
    }
  } else {
    // Limited range: Y_lim = 16 + (Y_full * 219 / 255).
    // 219 / 255 ≈ 0.85882; * 2^15 ≈ 28142 (Q15).
    // (`round(219 * 32768 / 255)` evaluates to 28142.)
    const LIMITED_SCALE_Q15: i32 = 28142;
    for x in 0..width {
      let r = rgb[x * 3] as i32;
      let g = rgb[x * 3 + 1] as i32;
      let b = rgb[x * 3 + 2] as i32;
      let y_full = (k_r * r + k_g * g + k_b * b + RND) >> 15;
      let y_full_clamped = y_full.clamp(0, 255);
      let y_lim = 16 + ((y_full_clamped * LIMITED_SCALE_Q15 + RND) >> 15);
      luma_out[x] = y_lim.clamp(0, 255) as u8;
    }
  }
}

/// `u16` luma analogue of [`rgb_to_luma_row`]. Y' is computed at
/// 8-bit precision (the source is 8-bit RGB) and zero-extended to
/// `u16`, matching the convention used by the packed-YUV `*_to_luma_u16`
/// kernels (`yuyv422_to_luma_u16_row` etc.) — the `u16` carrier
/// preserves the same dynamic range as the `u8` path, which is the
/// invariant downstream callers expect when consuming a
/// "native-depth" luma plane from an 8-bit-RGB-equivalent source.
///
/// # Panics
///
/// Panics (any build profile) if `rgb.len() < 3 * width` or
/// `luma_out.len() < width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb_to_luma_u16_row(
  rgb: &[u8],
  luma_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  let (k_r, k_g, k_b) = luma_coefficients_q15(matrix);
  const RND: i32 = 1 << 14;

  if full_range {
    for x in 0..width {
      let r = rgb[x * 3] as i32;
      let g = rgb[x * 3 + 1] as i32;
      let b = rgb[x * 3 + 2] as i32;
      let y = (k_r * r + k_g * g + k_b * b + RND) >> 15;
      luma_out[x] = y.clamp(0, 255) as u16;
    }
  } else {
    const LIMITED_SCALE_Q15: i32 = 28142;
    for x in 0..width {
      let r = rgb[x * 3] as i32;
      let g = rgb[x * 3 + 1] as i32;
      let b = rgb[x * 3 + 2] as i32;
      let y_full = (k_r * r + k_g * g + k_b * b + RND) >> 15;
      let y_full_clamped = y_full.clamp(0, 255);
      let y_lim = 16 + ((y_full_clamped * LIMITED_SCALE_Q15 + RND) >> 15);
      luma_out[x] = y_lim.clamp(0, 255) as u16;
    }
  }
}

/// Native-precision `u16` luma from a packed, host-order, native-depth
/// `u16` RGB row (`R,G,B` interleaved, `bits` significant bits per
/// channel). Mirrors `gbr_to_luma_u16_high_bit_row` bit-for-bit — same
/// Q15 coefficients, `RND = 1 << 14`, `i64` intermediates, and native
/// limited-range scaling — but reads the already-host-order binned RGB
/// the resample tail produces, so `bits` is a runtime argument rather
/// than a const generic and no byte-swap is applied. The fused
/// high-bit-GBR path uses this so its resampled `luma_u16` keeps the
/// direct path's native precision; the 8-bit [`rgb_to_luma_u16_row`]
/// above is the narrowed flavor the Rgb48 tail uses.
///
/// # Panics
///
/// Panics if `rgb.len() < 3 * width` or `luma_out.len() < width`.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  ),
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb_to_luma_u16_native_row(
  rgb: &[u16],
  luma_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  bits: u32,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  let (k_r, k_g, k_b) = luma_coefficients_q15(matrix);
  let (k_r, k_g, k_b) = (k_r as i64, k_g as i64, k_b as i64);
  const RND: i64 = 1 << 14;
  let native_max = ((1u32 << bits) - 1) as i64;
  // The binned RGB this consumes can carry a signed-filter overshoot
  // above the native max (the `FilterStream` only clamps to the full u16
  // range), so each channel is clamped to `[0, native_max]` before the
  // luma sum — the documented "clamp source samples to native range, then
  // derive" semantics. A no-op for the in-range area / direct callers.
  if full_range {
    for x in 0..width {
      let r = (rgb[x * 3] as i64).min(native_max);
      let g = (rgb[x * 3 + 1] as i64).min(native_max);
      let b = (rgb[x * 3 + 2] as i64).min(native_max);
      let y = (k_r * r + k_g * g + k_b * b + RND) >> 15;
      luma_out[x] = y.clamp(0, native_max) as u16;
    }
  } else {
    let y_off = 16i64 << (bits - 8);
    let range = 219i64 << (bits - 8);
    let y_max = 235i64 << (bits - 8);
    for x in 0..width {
      let r = (rgb[x * 3] as i64).min(native_max);
      let g = (rgb[x * 3 + 1] as i64).min(native_max);
      let b = (rgb[x * 3 + 2] as i64).min(native_max);
      let y_full = (k_r * r + k_g * g + k_b * b + RND) >> 15;
      let y_full_clamped = y_full.clamp(0, native_max);
      let y_lim = y_off + (y_full_clamped * range + native_max / 2) / native_max;
      luma_out[x] = y_lim.clamp(y_off, y_max) as u16;
    }
  }
}

/// Scalar RGB → HSV for a single pixel, using the shared division LUTs.
/// All arithmetic is integer; the two divisions `s = 255*delta/v` and
/// `h = 30*diff/delta` become `(operand * table[divisor] + RND) >> 12`.
///
/// `pub(crate)` so the direct YUV→HSV kernels (`yuv_420_to_hsv_row`
/// etc.) can feed their per-pixel decoded `(r, g, b)` straight into the
/// HSV quantizer without staging an intermediate packed-RGB row.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb_to_hsv_pixel(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
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
