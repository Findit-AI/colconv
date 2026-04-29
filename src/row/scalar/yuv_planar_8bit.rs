use super::*;

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
/// Output is packed `R, G, B` triples: `rgb_out[3*x] = R`,
/// `rgb_out[3*x + 1] = G`, `rgb_out[3*x + 2] = B`.
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
  yuv_420_to_rgb_or_rgba_row::<false, false>(
    y, u_half, v_half, None, rgb_out, width, matrix, full_range,
  );
}

/// Same as [`yuv_420_to_rgb_row`] but writes packed `R, G, B, A`
/// quadruplets, with `A = 0xFF` (opaque) for every pixel. The first
/// three bytes per pixel are byte-identical to what
/// [`yuv_420_to_rgb_row`] would write — only the per-pixel stride
/// (4 vs 3) and the alpha byte differ. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420_to_rgb_or_rgba_row::<true, false>(
    y, u_half, v_half, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:2:0 8‑bit → packed **8‑bit** **RGBA**. Same numerical
/// contract as [`yuv_420_to_rgba_row`] for R/G/B; the per-pixel alpha
/// byte is sourced from `a_src` (one byte per pixel, full-width)
/// instead of being constant `0xFF`. Used by the YUVA source family
/// ([`crate::yuv::Yuva420p`] in tranche 8b‑2a).
///
/// Thin wrapper over [`yuv_420_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `a_src.len() >= width`,
///   `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420_to_rgba_with_alpha_src_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420_to_rgb_or_rgba_row::<true, true>(
    y,
    u_half,
    v_half,
    Some(a_src),
    rgba_out,
    width,
    matrix,
    full_range,
  );
}

/// Shared scalar kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bytes / pixel), [`yuv_420_to_rgba_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, 4 bytes / pixel — 4th is opaque
/// `0xFF`) and [`yuv_420_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bytes / pixel with source-derived alpha). The
/// math is identical; only the per-pixel store differs. The const
/// generics drive compile-time monomorphization — each public wrapper
/// is inlined with the branches eliminated.
///
/// `a_src` is `None` for both `ALPHA_SRC = false` flavors — reading
/// it is a const-disabled branch in those monomorphizations, so
/// callers pay zero overhead for the strategy parameter. The 8-bit
/// alpha path stores `a_src[x]` directly: there's no `bits_mask` step
/// like the high-bit-depth siblings since `u8` already fits the
/// output.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
/// - When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///   `a_src.unwrap().len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

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
    let r0 = clamp_u8(y0 + r_chroma);
    let g0 = clamp_u8(y0 + g_chroma);
    let b0 = clamp_u8(y0 + b_chroma);
    out[x * bpp] = r0;
    out[x * bpp + 1] = g0;
    out[x * bpp + 2] = b0;
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies the wrapper
      // passed Some(_), validated above by debug_assert. 8-bit input
      // means u8 fits the u8 output directly — no `bits_mask` /
      // depth-conversion shift like the high-bit-depth siblings.
      out[x * bpp + 3] = a_src.as_ref().unwrap()[x];
    } else if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    // Pixel x+1 shares chroma.
    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    let r1 = clamp_u8(y1 + r_chroma);
    let g1 = clamp_u8(y1 + g_chroma);
    let b1 = clamp_u8(y1 + b_chroma);
    out[(x + 1) * bpp] = r1;
    out[(x + 1) * bpp + 1] = g1;
    out[(x + 1) * bpp + 2] = b1;
    if ALPHA_SRC {
      out[(x + 1) * bpp + 3] = a_src.as_ref().unwrap()[x + 1];
    } else if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}
/// YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// One UV pair per Y pixel, U/V from separate planes. Same
/// arithmetic as [`nv24_to_rgb_row`] (4:4:4 semi-planar) but
/// without the deinterleave step — U and V come pre-separated.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444_to_rgb_or_rgba_row::<false, false>(y, u, v, None, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar → packed `R, G, B, A` quadruplets with constant
/// `A = 0xFF`. First three bytes per pixel are byte-identical to
/// [`yuv_444_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444_to_rgb_or_rgba_row::<true, false>(y, u, v, None, rgba_out, width, matrix, full_range);
}

/// YUVA 4:4:4 planar → packed `R, G, B, A` quadruplets with the
/// per-pixel alpha byte sourced from `a_src` instead of constant
/// `0xFF`. R/G/B are byte-identical to [`yuv_444_to_rgb_row`]. Used
/// by the YUVA 4:4:4 source family ([`crate::yuv::Yuva444p`]).
///
/// Thin wrapper over [`yuv_444_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `a_src.len() >= width`, `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444_to_rgba_with_alpha_src_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444_to_rgb_or_rgba_row::<true, true>(
    y,
    u,
    v,
    Some(a_src),
    rgba_out,
    width,
    matrix,
    full_range,
  );
}

/// Shared scalar kernel for [`yuv_444_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp), [`yuv_444_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, 4 bpp + opaque alpha) and
/// [`yuv_444_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp + source-derived alpha). Math is
/// identical; only the per-pixel store stride and alpha byte differ.
/// `const` generic monomorphizes per call site, so the `if ALPHA` /
/// `if ALPHA_SRC` branches are eliminated.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
/// - When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///   `a_src.unwrap().len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params(full_range);
  const RND: i32 = 1 << 14;

  for x in 0..width {
    // 4:4:4: one UV pair per pixel, no subsampling.
    let u_d = ((u[x] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v[x] as i32 - 128) * c_scale + RND) >> 15;

    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
      // 8-bit alpha already fits u8 — no shift, no mask.
      out[x * bpp + 3] = a_src.as_ref().unwrap()[x];
    } else if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}
