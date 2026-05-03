//! Scalar reference kernels for the Tier 3 packed YUV 4:2:2 (8-bit)
//! sources: YUYV422 / UYVY422 / YVYU422.
//!
//! All three layouts share a single per-pixel-pair pipeline: extract
//! `Y0, Y1, U, V` from a 4-byte block, run the same Q15 chroma /
//! Y / channel math as the planar 4:2:0 / 4:2:2 path, and store
//! 6 bytes of RGB (or 8 bytes of RGBA when `ALPHA = true`).
//!
//! The two const generics select **byte positions** at compile time:
//! - `Y_LSB = true`  → bytes `[Y0, c0, Y1, c1]` (YUYV / YVYU layout).
//! - `Y_LSB = false` → bytes `[c0, Y0, c1, Y1]` (UYVY layout).
//! - `SWAP_UV = false` → `c0 = U, c1 = V` (YUYV / UYVY layout).
//! - `SWAP_UV = true`  → `c0 = V, c1 = U` (YVYU layout).
//!
//! The fourth corner `<Y_LSB=false, SWAP_UV=true>` would be VYUY422
//! (not in FFmpeg) and is never instantiated.

use super::*;

/// Generic packed YUV 4:2:2 → RGB / RGBA row kernel. Three formats
/// share this template via the `Y_LSB` and `SWAP_UV` const generics.
///
/// `packed.len() >= 2 * width`. `width` must be even. `out` length is
/// `3 * width` bytes for `ALPHA = false`, `4 * width` for `ALPHA = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv422_packed_to_rgb_or_rgba_row<const Y_LSB: bool, const SWAP_UV: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");

  // Byte positions inside each 4-byte (2-pixel) block:
  // Y_LSB = true  → `[Y0, c0, Y1, c1]`
  // Y_LSB = false → `[c0, Y0, c1, Y1]`
  let (y0_idx, y1_idx, c0_idx, c1_idx) = if Y_LSB { (0, 2, 1, 3) } else { (1, 3, 0, 2) };
  // c0 / c1 are U / V when SWAP_UV = false, V / U when true.
  let (u_idx, v_idx) = if SWAP_UV {
    (c1_idx, c0_idx)
  } else {
    (c0_idx, c1_idx)
  };

  let bpp: usize = if alpha { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);

  // Round-to-nearest Q15.
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let block = (x / 2) * 4;
    let y0 = packed[block + y0_idx] as i32;
    let y1 = packed[block + y1_idx] as i32;
    let u = packed[block + u_idx] as i32;
    let v = packed[block + v_idx] as i32;

    let u_d = ((u - 128) * c_scale + RND) >> 15;
    let v_d = ((v - 128) * c_scale + RND) >> 15;

    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    // Pixel x.
    let y0_s = ((y0 - y_off) * y_scale + RND) >> 15;
    out[x * bpp] = clamp_u8(y0_s + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0_s + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0_s + b_chroma);
    if alpha {
      out[x * bpp + 3] = 0xFF;
    }

    // Pixel x+1 shares chroma.
    let y1_s = ((y1 - y_off) * y_scale + RND) >> 15;
    out[(x + 1) * bpp] = clamp_u8(y1_s + r_chroma);
    out[(x + 1) * bpp + 1] = clamp_u8(y1_s + g_chroma);
    out[(x + 1) * bpp + 2] = clamp_u8(y1_s + b_chroma);
    if alpha {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}

/// Scalar YUYV422 → packed RGB. Byte order `Y0, U0, Y1, V0` per
/// 4-byte / 2-pixel block.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuyv422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv422_packed_to_rgb_or_rgba_row::<true, false>(
    packed, rgb_out, width, matrix, full_range, false,
  );
}

/// Scalar YUYV422 → packed RGBA (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuyv422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv422_packed_to_rgb_or_rgba_row::<true, false>(
    packed, rgba_out, width, matrix, full_range, true,
  );
}

/// Scalar UYVY422 → packed RGB. Byte order `U0, Y0, V0, Y1` per
/// 4-byte / 2-pixel block.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyvy422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv422_packed_to_rgb_or_rgba_row::<false, false>(
    packed, rgb_out, width, matrix, full_range, false,
  );
}

/// Scalar UYVY422 → packed RGBA (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyvy422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv422_packed_to_rgb_or_rgba_row::<false, false>(
    packed, rgba_out, width, matrix, full_range, true,
  );
}

/// Scalar YVYU422 → packed RGB. Byte order `Y0, V0, Y1, U0` per
/// 4-byte / 2-pixel block (UV swapped relative to YUYV).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yvyu422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv422_packed_to_rgb_or_rgba_row::<true, true>(packed, rgb_out, width, matrix, full_range, false);
}

/// Scalar YVYU422 → packed RGBA (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yvyu422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv422_packed_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range, true);
}

/// Copies the Y bytes from a packed YUV 4:2:2 row directly into a
/// `width`-byte luma plane. Avoids the YUV→RGB pipeline entirely
/// when only luma is needed.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv422_packed_to_luma_row<const Y_LSB: bool>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let (y0_idx, y1_idx) = if Y_LSB { (0, 2) } else { (1, 3) };
  let mut x = 0;
  while x < width {
    let block = (x / 2) * 4;
    luma_out[x] = packed[block + y0_idx];
    luma_out[x + 1] = packed[block + y1_idx];
    x += 2;
  }
}

/// Scalar YUYV422 luma extraction.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuyv422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
}

/// Scalar UYVY422 luma extraction.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyvy422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  yuv422_packed_to_luma_row::<false>(packed, luma_out, width);
}

/// Scalar YVYU422 luma extraction (Y positions identical to YUYV).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yvyu422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
}
