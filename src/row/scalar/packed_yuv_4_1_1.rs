//! Scalar reference kernels for the Tier 5.25 packed YUV 4:1:1 (8-bit)
//! source: UYYVYY411 (`AV_PIX_FMT_UYYVYY411`).
//!
//! Per-block layout (6 bytes / 4 pixels):
//!
//! `[U, Y0, Y1, V, Y2, Y3]`
//!
//! Each (U, V) chroma pair is shared by 4 adjacent luma samples
//! (horizontal 4:1:1 subsampling). Pipeline mirrors the packed
//! 4:2:2 kernel in [`super::packed_yuv_8bit`]: extract U / Y0..Y3 / V,
//! run the same Q15 chroma / Y / channel math, and store 12 bytes
//! of RGB (or 16 bytes of RGBA when `ALPHA = true`) per 4-pixel
//! block.

use super::*;

/// Scalar UYYVYY411 → packed RGB or RGBA row kernel. Each 6-byte
/// block decodes to four output pixels sharing one (U, V) pair.
///
/// `packed.len() >= width * 3 / 2`. `width` must be a multiple of 4.
/// `out` length is `3 * width` for `ALPHA = false`, `4 * width` for
/// `ALPHA = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyyvyy411_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(
    packed.len() >= width * 3 / 2,
    "packed row too short for 4:1:1"
  );

  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {}bpp", bpp);

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);

  // Round-to-nearest Q15.
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let block = (x / 4) * 6;
    let u = packed[block] as i32;
    let y0 = packed[block + 1] as i32;
    let y1 = packed[block + 2] as i32;
    let v = packed[block + 3] as i32;
    let y2 = packed[block + 4] as i32;
    let y3 = packed[block + 5] as i32;

    let u_d = ((u - 128) * c_scale + RND) >> 15;
    let v_d = ((v - 128) * c_scale + RND) >> 15;

    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    // Each of the 4 pixels in this block applies the shared chroma to
    // its own scaled Y. Unrolled to keep the inner branch on `ALPHA`
    // monomorphized.
    for (i, y) in [y0, y1, y2, y3].iter().enumerate() {
      let y_s = ((y - y_off) * y_scale + RND) >> 15;
      let pos = (x + i) * bpp;
      out[pos] = clamp_u8(y_s + r_chroma);
      out[pos + 1] = clamp_u8(y_s + g_chroma);
      out[pos + 2] = clamp_u8(y_s + b_chroma);
      if ALPHA {
        out[pos + 3] = 0xFF;
      }
    }

    x += 4;
  }
}

/// Scalar UYYVYY411 → packed RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyyvyy411_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  uyyvyy411_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Scalar UYYVYY411 → packed RGBA (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyyvyy411_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  uyyvyy411_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
}

// ---- Packed YUV 4:1:1 (8-bit) → HSV (direct: no RGB scratch) ----------
//
// The display-referred twin of [`uyyvyy411_to_rgb_or_rgba_row`], fused
// with the OpenCV HSV quantizer. Shares the EXACT per-pixel Q15 decode
// (`Coefficients::for_matrix` + `range_params_n::<8, 8>` + the 6-byte /
// 4-pixel `[U, Y0, Y1, V, Y2, Y3]` unpack with one shared (U, V) per 4
// luma) as the `_to_rgb` sibling, then feeds the decoded `(r, g, b)`
// straight into [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes —
// never materializing a packed-RGB row. Byte-identical to
// `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` but allocates no RGB
// intermediate.

/// Scalar UYYVYY411 → planar HSV row (OpenCV `cv2.COLOR_RGB2HSV`
/// encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). Each 6-byte block
/// decodes to four output pixels sharing one (U, V) pair. Byte-identical
/// to `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))`.
///
/// `packed.len() >= width * 3 / 2`. `width` must be a multiple of 4.
/// Each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn uyyvyy411_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(
    packed.len() >= width * 3 / 2,
    "packed row too short for 4:1:1"
  );
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let block = (x / 4) * 6;
    let u = packed[block] as i32;
    let y0 = packed[block + 1] as i32;
    let y1 = packed[block + 2] as i32;
    let v = packed[block + 3] as i32;
    let y2 = packed[block + 4] as i32;
    let y3 = packed[block + 5] as i32;

    let u_d = ((u - 128) * c_scale + RND) >> 15;
    let v_d = ((v - 128) * c_scale + RND) >> 15;
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    for (i, y) in [y0, y1, y2, y3].iter().enumerate() {
      let y_s = ((y - y_off) * y_scale + RND) >> 15;
      let (h, s, vv) = rgb_to_hsv_pixel(
        clamp_u8(y_s + r_chroma) as i32,
        clamp_u8(y_s + g_chroma) as i32,
        clamp_u8(y_s + b_chroma) as i32,
      );
      h_out[x + i] = h;
      s_out[x + i] = s;
      v_out[x + i] = vv;
    }

    x += 4;
  }
}

/// Copies the Y bytes from a packed UYYVYY411 row directly into a
/// `width`-byte luma plane. Y bytes live at offsets 1, 2, 4, 5 of each
/// 6-byte block.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyyvyy411_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(
    packed.len() >= width * 3 / 2,
    "packed row too short for 4:1:1"
  );
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let mut x = 0;
  while x < width {
    let block = (x / 4) * 6;
    luma_out[x] = packed[block + 1];
    luma_out[x + 1] = packed[block + 2];
    luma_out[x + 2] = packed[block + 4];
    luma_out[x + 3] = packed[block + 5];
    x += 4;
  }
}

/// Extract Y as u16 (zero-extended) from packed UYYVYY411.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyyvyy411_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(
    packed.len() >= width * 3 / 2,
    "packed row too short for 4:1:1"
  );
  debug_assert!(out.len() >= width, "out too short");

  let mut x = 0;
  while x < width {
    let block = (x / 4) * 6;
    out[x] = packed[block + 1] as u16;
    out[x + 1] = packed[block + 2] as u16;
    out[x + 2] = packed[block + 4] as u16;
    out[x + 3] = packed[block + 5] as u16;
    x += 4;
  }
}
