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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv422_packed_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range, true);
}

// ---- Packed YUV 4:2:2 (8-bit) → HSV (direct: no RGB scratch) ----------
//
// The display-referred twins of the `*_to_rgb_row` kernels above, fused
// with the OpenCV HSV quantizer. Each shares the EXACT per-pixel Q15
// decode (`Coefficients::for_matrix` + `range_params_n::<8, 8>` + the
// same `Y_LSB` / `SWAP_UV` byte-position selection) as its `_to_rgb`
// sibling, then feeds the decoded `(r, g, b)` straight into
// [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes — never
// materializing a packed-RGB row. They are therefore byte-identical to
// `rgb_to_hsv_row(*_to_rgb_row(...))` but allocate no RGB intermediate.
// Used by the packed 4:2:2 sink's HSV-without-RGB path; the SIMD
// backends mirror them via a small reused-chunk RGB scratch (the chunk
// filler IS the existing SIMD RGB kernel) plus the SIMD `rgb_to_hsv_row`.

/// Generic packed YUV 4:2:2 → planar HSV row kernel (OpenCV
/// `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`).
/// Three formats share this template via the `Y_LSB` / `SWAP_UV` const
/// generics — the same byte-position selection as
/// [`yuv422_packed_to_rgb_or_rgba_row`]. Byte-identical to
/// `rgb_to_hsv_row(yuv422_packed_to_rgb_or_rgba_row::<Y_LSB, SWAP_UV>(...))`.
///
/// `packed.len() >= 2 * width`. `width` must be even. Each of `h_out` /
/// `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv422_packed_to_hsv_row<const Y_LSB: bool, const SWAP_UV: bool>(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // Byte positions inside each 4-byte (2-pixel) block — identical
  // selection to the `_to_rgb` kernel.
  let (y0_idx, y1_idx, c0_idx, c1_idx) = if Y_LSB { (0, 2, 1, 3) } else { (1, 3, 0, 2) };
  let (u_idx, v_idx) = if SWAP_UV {
    (c1_idx, c0_idx)
  } else {
    (c0_idx, c1_idx)
  };

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
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

    let y0_s = ((y0 - y_off) * y_scale + RND) >> 15;
    let (h0, s0, v0) = rgb_to_hsv_pixel(
      clamp_u8(y0_s + r_chroma) as i32,
      clamp_u8(y0_s + g_chroma) as i32,
      clamp_u8(y0_s + b_chroma) as i32,
    );
    h_out[x] = h0;
    s_out[x] = s0;
    v_out[x] = v0;

    let y1_s = ((y1 - y_off) * y_scale + RND) >> 15;
    let (h1, s1, v1) = rgb_to_hsv_pixel(
      clamp_u8(y1_s + r_chroma) as i32,
      clamp_u8(y1_s + g_chroma) as i32,
      clamp_u8(y1_s + b_chroma) as i32,
    );
    h_out[x + 1] = h1;
    s_out[x + 1] = s1;
    v_out[x + 1] = v1;

    x += 2;
  }
}

/// Scalar YUYV422 → planar HSV. Byte order `Y0, U0, Y1, V0` per 4-byte /
/// 2-pixel block. Byte-identical to `rgb_to_hsv_row(yuyv422_to_rgb_row(...))`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuyv422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv422_packed_to_hsv_row::<true, false>(packed, h_out, s_out, v_out, width, matrix, full_range);
}

/// Scalar UYVY422 → planar HSV. Byte order `U0, Y0, V0, Y1` per 4-byte /
/// 2-pixel block. Byte-identical to `rgb_to_hsv_row(uyvy422_to_rgb_row(...))`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn uyvy422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv422_packed_to_hsv_row::<false, false>(packed, h_out, s_out, v_out, width, matrix, full_range);
}

/// Scalar YVYU422 → planar HSV. Byte order `Y0, V0, Y1, U0` per 4-byte /
/// 2-pixel block (UV swapped relative to YUYV). Byte-identical to
/// `rgb_to_hsv_row(yvyu422_to_rgb_row(...))`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yvyu422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv422_packed_to_hsv_row::<true, true>(packed, h_out, s_out, v_out, width, matrix, full_range);
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

/// Extract Y as u16 (zero-extended) from `Yuyv422` packed `[Y, U, Y, V]` layout.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuyv422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(out.len() >= width, "out too short");
  for x in 0..width {
    out[x] = packed[x * 2] as u16;
  }
}

/// Extract Y as u16 from `Uyvy422` packed `[U, Y, V, Y]` layout (Y at offset 1).
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyvy422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(out.len() >= width, "out too short");
  for x in 0..width {
    out[x] = packed[x * 2 + 1] as u16;
  }
}

/// Extract Y as u16 from `Yvyu422` packed `[Y, V, Y, U]` layout.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yvyu422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(out.len() >= width, "out too short");
  for x in 0..width {
    out[x] = packed[x * 2] as u16;
  }
}
