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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu_half.len() >= width, "chroma row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu.len() >= 2 * width, "chroma row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
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

// ---- Semi-planar 8-bit NV → HSV (direct: no RGB scratch) -------------
//
// The display-referred twins of the `nv*_to_rgb_row` kernels above,
// fused with the OpenCV HSV quantizer. Each shares the EXACT per-pixel
// Q15 decode (`Coefficients::for_matrix` + `range_params_n::<8, 8>` +
// the same interleaved-chroma byte order / upsampling shape) as its
// `_to_rgb` sibling, then feeds the decoded `(r, g, b)` straight into
// [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes — never
// materializing a packed-RGB row. They are therefore byte-identical to
// `rgb_to_hsv_row(nv*_to_rgb_row(...))` but allocate no RGB
// intermediate. Used by the semi-planar 8-bit sink's HSV-without-RGB
// path; the SIMD backends mirror them via a small reused-chunk RGB
// scratch (the chunk filler IS the existing SIMD RGB kernel) plus the
// SIMD `rgb_to_hsv_row` on the chunk.

/// Shared scalar kernel for NV12 (`SWAP_UV = false`) / NV21
/// (`SWAP_UV = true`) → planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV`
/// encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). 4:2:0 chroma is
/// half-width, nearest-neighbor 1→2 upsampled per pixel pair exactly as
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`]. Also serves NV16 (4:2:2 —
/// the same per-row chroma shape). `SWAP_UV` drives compile-time
/// monomorphization.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_or_vu_half.len() >= width`, and each of
///   `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
fn nv12_or_nv21_to_hsv_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu_half.len() >= width, "chroma row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    // NV12: even byte = U, odd byte = V. NV21: even byte = V, odd = U.
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
    let (h0, s0, v0) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h0;
    s_out[x] = s0;
    v_out[x] = v0;

    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    let (h1, s1, v1) = rgb_to_hsv_pixel(
      clamp_u8(y1 + r_chroma) as i32,
      clamp_u8(y1 + g_chroma) as i32,
      clamp_u8(y1 + b_chroma) as i32,
    );
    h_out[x + 1] = h1;
    s_out[x + 1] = s1;
    v_out[x + 1] = v1;

    x += 2;
  }
}

/// NV12 (semi-planar 4:2:0, UV-ordered) → planar HSV bytes. Thin wrapper
/// over [`nv12_or_nv21_to_hsv_row_impl`] with `SWAP_UV = false`.
/// Byte-identical to `rgb_to_hsv_row(nv12_to_rgb_row(...))`. Also serves
/// NV16 (4:2:2 — identical per-row chroma shape).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn nv12_to_hsv_row(
  y: &[u8],
  uv_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_hsv_row_impl::<false>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range);
}

/// NV21 (semi-planar 4:2:0, VU-ordered) → planar HSV bytes. Thin wrapper
/// over [`nv12_or_nv21_to_hsv_row_impl`] with `SWAP_UV = true`.
/// Byte-identical to `rgb_to_hsv_row(nv21_to_rgb_row(...))`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn nv21_to_hsv_row(
  y: &[u8],
  vu_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_hsv_row_impl::<true>(y, vu_half, h_out, s_out, v_out, width, matrix, full_range);
}

/// Shared scalar kernel for NV24 (`SWAP_UV = false`) / NV42
/// (`SWAP_UV = true`) → planar HSV bytes. 4:4:4 — one UV pair per Y
/// pixel, no chroma upsampling, exactly as
/// [`nv24_or_nv42_to_rgb_or_rgba_row_impl`]. `SWAP_UV` drives
/// compile-time monomorphization.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_or_vu.len() >= 2 * width`, and each of
///   `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
fn nv24_or_nv42_to_hsv_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu.len() >= 2 * width, "chroma row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
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
    let (h, s, vv) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = vv;
  }
}

/// NV24 (semi-planar 4:4:4, UV-ordered) → planar HSV bytes. Thin wrapper
/// over [`nv24_or_nv42_to_hsv_row_impl`] with `SWAP_UV = false`.
/// Byte-identical to `rgb_to_hsv_row(nv24_to_rgb_row(...))`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn nv24_to_hsv_row(
  y: &[u8],
  uv: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  nv24_or_nv42_to_hsv_row_impl::<false>(y, uv, h_out, s_out, v_out, width, matrix, full_range);
}

/// NV42 (semi-planar 4:4:4, VU-ordered) → planar HSV bytes. Thin wrapper
/// over [`nv24_or_nv42_to_hsv_row_impl`] with `SWAP_UV = true`.
/// Byte-identical to `rgb_to_hsv_row(nv42_to_rgb_row(...))`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn nv42_to_hsv_row(
  y: &[u8],
  vu: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  nv24_or_nv42_to_hsv_row_impl::<true>(y, vu, h_out, s_out, v_out, width, matrix, full_range);
}

// ---- Semi-planar 8-bit NV → native luma (Y plane copy) ---------------
//
// For 8-bit NV the luma plane IS the Y plane verbatim (the NATIVE-Y
// contract shared with the planar / packed YUV luma kernels). These thin
// `nv*_to_luma_row` kernels give that copy a named row primitive so the
// sink's `with_luma()` path routes through one kernel instead of an
// inline `copy_from_slice`. NV12 / NV16 / NV21 / NV24 / NV42 all share
// the same Y plane shape (`width` bytes per row), so a single kernel
// serves every member.

/// Semi-planar NV (8-bit) → native luma bytes: the Y plane copied
/// verbatim. For every 8-bit NV format the luma plane is exactly the Y
/// plane (no scaling, no matrix), so this is a straight
/// `copy_from_slice` of the first `width` Y samples — bit-identical to
/// the sink's former inline copy. Serves NV12 / NV16 / NV21 / NV24 /
/// NV42 alike.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width` and `luma_out.len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv_to_luma_row(y: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  luma_out[..width].copy_from_slice(&y[..width]);
}
