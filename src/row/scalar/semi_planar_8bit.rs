use super::*;

/// NV12 (semi‑planar 4:2:0, UV-ordered) → packed RGB. Thin wrapper
/// over [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
    y, uv_half, rgb_out, width, matrix, full_range,
  );
}

/// NV21 (semi‑planar 4:2:0, VU-ordered) → packed RGB. Thin wrapper
/// over [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
    y, vu_half, rgb_out, width, matrix, full_range,
  );
}

/// NV12 → packed `R, G, B, A` quadruplets with constant `A = 0xFF`.
/// First three bytes per pixel are byte-identical to
/// [`nv12_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
    y, uv_half, rgba_out, width, matrix, full_range,
  );
}

/// NV21 → packed `R, G, B, A` quadruplets with constant `A = 0xFF`.
/// First three bytes per pixel are byte-identical to
/// [`nv21_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
    y, vu_half, rgba_out, width, matrix, full_range,
  );
}

/// Shared scalar kernel for NV12 (`SWAP_UV = false`) / NV21
/// (`SWAP_UV = true`) at 3 bpp (`ALPHA = false`) or 4 bpp + opaque
/// alpha (`ALPHA = true`). Identical math to [`yuv_420_to_rgb_row`];
/// the only differences are chroma byte order in the interleaved
/// plane and the per-pixel store stride. Both `const` generics drive
/// compile-time monomorphization — each wrapper is inlined with both
/// branches eliminated.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `y.len() >= width`, `uv_or_vu_half.len() >= width`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn nv12_or_nv21_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu_half.len() >= width, "chroma row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params(full_range);
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    // NV12: even byte = U, odd byte = V.
    // NV21: even byte = V, odd byte = U.
    let (u_byte, v_byte) = if SWAP_UV {
      (uv_or_vu_half[c_idx * 2 + 1], uv_or_vu_half[c_idx * 2])
    } else {
      (uv_or_vu_half[c_idx * 2], uv_or_vu_half[c_idx * 2 + 1])
    };
    let u_d = ((u_byte as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_byte as i32 - 128) * c_scale + RND) >> 15;

    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    let r0 = clamp_u8(y0 + r_chroma);
    let g0 = clamp_u8(y0 + g_chroma);
    let b0 = clamp_u8(y0 + b_chroma);
    out[x * bpp] = r0;
    out[x * bpp + 1] = g0;
    out[x * bpp + 2] = b0;
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    let r1 = clamp_u8(y1 + r_chroma);
    let g1 = clamp_u8(y1 + g_chroma);
    let b1 = clamp_u8(y1 + b_chroma);
    out[(x + 1) * bpp] = r1;
    out[(x + 1) * bpp + 1] = g1;
    out[(x + 1) * bpp + 2] = b1;
    if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}

/// NV24 (semi-planar 4:4:4, UV-ordered) → packed RGB. Thin wrapper
/// over [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, false>(y, uv, rgb_out, width, matrix, full_range);
}

/// NV42 (semi-planar 4:4:4, VU-ordered) → packed RGB. Thin wrapper
/// over [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, false>(y, vu, rgb_out, width, matrix, full_range);
}

/// NV24 → packed `R, G, B, A` quadruplets with constant `A = 0xFF`.
/// First three bytes per pixel are byte-identical to
/// [`nv24_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv24_to_rgba_row(
  y: &[u8],
  uv: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, true>(y, uv, rgba_out, width, matrix, full_range);
}

/// NV42 → packed `R, G, B, A` quadruplets with constant `A = 0xFF`.
/// First three bytes per pixel are byte-identical to
/// [`nv42_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv42_to_rgba_row(
  y: &[u8],
  vu: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, true>(y, vu, rgba_out, width, matrix, full_range);
}

/// Shared scalar kernel for NV24 (`SWAP_UV = false`) / NV42
/// (`SWAP_UV = true`) at 3 bpp (`ALPHA = false`) or 4 bpp + opaque
/// alpha (`ALPHA = true`). Identical math to [`yuv_444_to_rgb_row`]
/// (4:4:4 — one UV pair per Y pixel, no chroma upsampling); only
/// the per-pixel store stride differs. Both `const` generics drive
/// compile-time monomorphization.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_or_vu.len() >= 2 * width`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn nv24_or_nv42_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu.len() >= 2 * width, "chroma row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params(full_range);
  const RND: i32 = 1 << 14;

  for x in 0..width {
    // 4:4:4: one UV pair per pixel. No upsampling.
    let (u_byte, v_byte) = if SWAP_UV {
      (uv_or_vu[x * 2 + 1], uv_or_vu[x * 2])
    } else {
      (uv_or_vu[x * 2], uv_or_vu[x * 2 + 1])
    };
    let u_d = ((u_byte as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_byte as i32 - 128) * c_scale + RND) >> 15;

    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}
