use super::*;

// ---- 16-bit YUV 4:2:0 → RGB (parallel kernel family) -------------------
//
// At 16 bits the chroma multiply-add `c_u * u_d + c_v * v_d` splits
// into two regimes by output target:
//
// - **16 → u8**: the Q15 scale knocks `u_d` / `v_d` down to u8 range
//   (max ±150 at limited range, ±128 at full). Products like
//   `60808 * 150 = 9.1M` and their sums stay well within i32, so the
//   i32 pipeline used by 10/12/14 works unchanged at BITS = 16 — the
//   kernels below reuse that structure without widening.
// - **16 → u16**: the Q15 scale is a near-identity (32768 at full
//   range), so `u_d` / `v_d` can reach ±32768. `coeff * u_d` alone
//   reaches ~1.99·10⁹ (close to i32 max); the full chroma sum
//   reaches ~3.68·10⁹ — overflows i32. The u16 kernels below widen
//   the chroma multiply-add to i64 (via [`q15_chroma64`]) and narrow
//   back after the `>> 15`.
//
// All four functions are dedicated 16-bit entry points (not
// const-generic) so each monomorphization picks the right precision
// path without a runtime branch.
/// Converts one row of **16-bit** YUV 4:2:0 (samples in the full
/// `u16` range) to **8-bit** packed RGB. At 16 → u8 the Q15 scale
/// confines chroma to u8 range, so the i32 chroma pipeline used by
/// 10/12/14 applies unchanged here — this kernel is structurally
/// identical to [`yuv_420p_n_to_rgb_row`] at a hypothetical
/// `BITS = 16`, just without the AND-mask (no upper-bit-zero
/// guarantee to enforce at 16 bits).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p16_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420p16_to_rgb_or_rgba_row::<false, false>(
    y, u_half, v_half, None, rgb_out, width, matrix, full_range,
  );
}

/// Converts one row of **16-bit** YUV 4:2:0 to **8-bit** packed
/// **RGBA**. Same numerical contract as [`yuv_420p16_to_rgb_row`];
/// the only differences are the per-pixel stride (4 vs 3) and the
/// alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
//
// Scalar prep for Ship 8 Tranche 5a: the public dispatcher
// `row::yuv420p16_to_rgba_row` lands in the follow-up SIMD/dispatcher
// PR. Until then this thin wrapper has no caller.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p16_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420p16_to_rgb_or_rgba_row::<true, false>(
    y, u_half, v_half, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:2:0 16‑bit → packed **8‑bit** **RGBA**. Same numerical
/// contract as [`yuv_420p16_to_rgba_row`] for R/G/B; the per-pixel
/// alpha byte is sourced from `a_src` (depth-converted by `>> 8`)
/// instead of being constant `0xFF`. Used by the YUVA 4:2:0 source
/// family ([`crate::yuv::Yuva420p16`] in tranche 8b‑2a).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with
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
pub(crate) fn yuv_420p16_to_rgba_with_alpha_src_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420p16_to_rgb_or_rgba_row::<true, true>(
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

/// Shared 16-bit YUV 4:2:0 → 8-bit RGB / RGBA kernel. `ALPHA = false,
/// ALPHA_SRC = false` emits 3 bpp; `ALPHA = true, ALPHA_SRC = false`
/// emits 4 bpp with constant `0xFF` alpha; `ALPHA = true, ALPHA_SRC =
/// true` emits 4 bpp with depth-converted source alpha.
///
/// 16-bit input has no AND-mask (every `u16` is a valid sample) and
/// uses i32 chroma — output-target scaling keeps `u_d * coeff` inside
/// i32 for u8 output (the i64 chroma family lives in
/// [`yuv_420p16_to_rgb_or_rgba_u16_row`]). Source alpha at 16-bit
/// depth-converts to u8 via `>> 8`; no mask is needed since every
/// u16 is in range.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420p16_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  // No AND-mask needed at 16-bit — every u16 is already a valid
  // sample. `q15_chroma` (i32) is enough for u8 output because the
  // output-target scaling keeps `u_d * coeff` well within i32.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(u_half[c_idx] as i32 - bias, c_scale);
    let v_d = q15_scale(v_half[c_idx] as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(y[x] as i32 - y_off, y_scale);
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
      // 16-bit input is full-range u16 — no `bits_mask` step. Depth
      // convert via `>> 8` to fit the u8 output.
      out[x * bpp + 3] = (a_src.as_ref().unwrap()[x] >> 8) as u8;
    } else if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    let y1 = q15_scale(y[x + 1] as i32 - y_off, y_scale);
    out[(x + 1) * bpp] = clamp_u8(y1 + r_chroma);
    out[(x + 1) * bpp + 1] = clamp_u8(y1 + g_chroma);
    out[(x + 1) * bpp + 2] = clamp_u8(y1 + b_chroma);
    if ALPHA_SRC {
      out[(x + 1) * bpp + 3] = (a_src.as_ref().unwrap()[x + 1] >> 8) as u8;
    } else if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}

/// Converts one row of **16-bit** YUV 4:2:0 to **native-depth `u16`**
/// packed RGB — full-range output in `[0, 65535]`. **Runs the
/// chroma matrix multiply in i64** to accommodate the wider
/// `coeff × u_d` product at 16 → 16-bit scaling.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// Same contract as [`yuv_420p16_to_rgb_row`] plus `rgb_out` is
/// measured in `u16` elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p16_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420p16_to_rgb_or_rgba_u16_row::<false, false>(
    y, u_half, v_half, None, rgb_out, width, matrix, full_range,
  );
}

/// Converts one row of **16-bit** YUV 4:2:0 to **native-depth `u16`**
/// packed **RGBA** — alpha element is `0xFFFF` (opaque maximum at
/// 16-bit).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
//
// Scalar prep for Ship 8 Tranche 5b: the public dispatcher
// `row::yuv420p16_to_rgba_u16_row` lands in the follow-up SIMD/dispatcher
// PR. Until then this thin wrapper has no caller.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p16_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420p16_to_rgb_or_rgba_u16_row::<true, false>(
    y, u_half, v_half, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:2:0 16‑bit → packed **native‑depth `u16`** **RGBA**. Same
/// numerical contract as [`yuv_420p16_to_rgba_u16_row`] for R/G/B;
/// the per-pixel alpha element is sourced from `a_src` (already at
/// the source's native bit depth — no shift needed) instead of being
/// the opaque maximum `0xFFFF`. Used by the YUVA 4:2:0 source family
/// ([`crate::yuv::Yuva420p16`] in tranche 8b‑2a).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_u16_row`] with
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
pub(crate) fn yuv_420p16_to_rgba_u16_with_alpha_src_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420p16_to_rgb_or_rgba_u16_row::<true, true>(
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

/// Shared 16-bit YUV 4:2:0 → native-depth `u16` RGB / RGBA kernel.
/// `ALPHA = false, ALPHA_SRC = false` emits 3 bpp; `ALPHA = true,
/// ALPHA_SRC = false` emits 4 bpp with constant `0xFFFF` alpha;
/// `ALPHA = true, ALPHA_SRC = true` emits 4 bpp with native-depth
/// source alpha.
///
/// Uses i64 chroma multiply (same rationale as
/// [`yuv_420p16_to_rgb_u16_row`]). Source alpha at 16-bit is already
/// at native depth (full u16 range, no `bits_mask` needed).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(u_half[c_idx] as i32 - bias, c_scale);
    let v_d = q15_scale(v_half[c_idx] as i32 - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale64(y[x] as i32 - y_off, y_scale);
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
      // 16-bit alpha is already at native depth (full u16 range);
      // no mask needed since every u16 is in range.
      out[x * bpp + 3] = a_src.as_ref().unwrap()[x];
    } else if ALPHA {
      out[x * bpp + 3] = 0xFFFF;
    }

    let y1 = q15_scale64(y[x + 1] as i32 - y_off, y_scale);
    out[(x + 1) * bpp] = (y1 + r_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA_SRC {
      out[(x + 1) * bpp + 3] = a_src.as_ref().unwrap()[x + 1];
    } else if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFFFF;
    }

    x += 2;
  }
}

/// YUV 4:4:4 planar **16‑bit** → packed **8‑bit** RGB. Same i32
/// chroma pipeline as 10/12/14 (output‑range scaling keeps `coeff × u_d`
/// inside i32 for u8 target). 1:1 chroma per Y pixel, no width parity.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p16_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444p16_to_rgb_or_rgba_row::<false, false>(y, u, v, None, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar **16‑bit** → packed **8‑bit** **RGBA**. Same
/// numerical contract as [`yuv_444p16_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p16_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444p16_to_rgb_or_rgba_row::<true, false>(y, u, v, None, rgba_out, width, matrix, full_range);
}

/// YUVA 4:4:4 16‑bit → packed **8‑bit** **RGBA**. Same numerical
/// contract as [`yuv_444p16_to_rgba_row`] for R/G/B; the per-pixel
/// alpha byte is **sourced from `a_src`** (depth-converted via
/// `>> 8` to fit `u8`) instead of being constant `0xFF`. Used by the
/// YUVA 4:4:4 source family ([`crate::yuv::Yuva444p16`] in tranche
/// 8b‑5a).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `a_src.len() >= width`, `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p16_to_rgba_with_alpha_src_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444p16_to_rgb_or_rgba_row::<true, true>(
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

/// Shared 16-bit YUV 4:4:4 → 8-bit RGB / RGBA kernel. `ALPHA = false,
/// ALPHA_SRC = false` emits 3 bpp; `ALPHA = true, ALPHA_SRC = false`
/// emits 4 bpp with constant `0xFF` alpha; `ALPHA = true, ALPHA_SRC =
/// true` emits 4 bpp with depth-converted source alpha.
///
/// 16-bit input has no AND-mask (every `u16` is a valid sample) and
/// uses i32 chroma — output-target scaling keeps `u_d * coeff` inside
/// i32 for u8 output (the i64 chroma family lives in
/// [`yuv_444p16_to_rgb_or_rgba_u16_row`]). Source alpha at 16-bit
/// depth-converts to u8 via `>> 8`; no mask is needed since every
/// u16 is in range.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p16_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let u_d = q15_scale(u[x] as i32 - bias, c_scale);
    let v_d = q15_scale(v[x] as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(y[x] as i32 - y_off, y_scale);
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
      // 16-bit input is full-range u16 — no `bits_mask` step. Depth
      // convert via `>> 8` to fit the u8 output.
      out[x * bpp + 3] = (a_src.as_ref().unwrap()[x] >> 8) as u8;
    } else if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// YUV 4:4:4 planar **16‑bit** → packed **native‑depth `u16`** RGB.
/// Widens chroma matrix multiply to i64 (Bt2020 `b_u × u_d` reaches
/// ~2.31·10⁹ at limited‑range 16→u16 — overflows i32). Y path widens
/// via [`q15_scale64`] to handle unclamped Y samples above the
/// limited‑range nominal max.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p16_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444p16_to_rgb_or_rgba_u16_row::<false, false>(
    y, u, v, None, rgb_out, width, matrix, full_range,
  );
}

/// YUV 4:4:4 planar **16‑bit** → packed **native‑depth `u16`** **RGBA**
/// — alpha element is `0xFFFF` (opaque maximum at 16‑bit).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p16_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444p16_to_rgb_or_rgba_u16_row::<true, false>(
    y, u, v, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:4:4 16‑bit → packed **native‑depth `u16`** **RGBA**. Same
/// numerical contract as [`yuv_444p16_to_rgba_u16_row`] for R/G/B; the
/// per-pixel alpha element is sourced from `a_src` (already at the
/// source's native bit depth — no shift needed) instead of being the
/// opaque maximum `0xFFFF`. Used by the YUVA 4:4:4 source family
/// ([`crate::yuv::Yuva444p16`] in tranche 8b‑5a).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `a_src.len() >= width`, `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p16_to_rgba_u16_with_alpha_src_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444p16_to_rgb_or_rgba_u16_row::<true, true>(
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

/// Shared 16-bit YUV 4:4:4 → native-depth `u16` RGB / RGBA kernel.
/// `ALPHA = false, ALPHA_SRC = false` emits 3 bpp; `ALPHA = true,
/// ALPHA_SRC = false` emits 4 bpp with constant `0xFFFF` alpha;
/// `ALPHA = true, ALPHA_SRC = true` emits 4 bpp with the alpha
/// element copied from `a_src` (16-bit input is full-range — no
/// shift needed).
///
/// Uses i64 chroma multiply (same rationale as
/// [`yuv_444p16_to_rgb_u16_row`]).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  for x in 0..width {
    let u_d = q15_scale(u[x] as i32 - bias, c_scale);
    let v_d = q15_scale(v[x] as i32 - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale64(y[x] as i32 - y_off, y_scale);
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
      // 16-bit native-depth output keeps alpha verbatim — no shift.
      out[x * bpp + 3] = a_src.as_ref().unwrap()[x];
    } else if ALPHA {
      out[x * bpp + 3] = 0xFFFF;
    }
  }
}

/// Converts one row of **P016** (semi-planar 4:2:0 with UV
/// interleaved, full `u16` samples) to **8-bit** packed RGB. At 16
/// bits there is no "high-bit-packed" vs "low-bit-packed" distinction
/// (every bit is active), so this kernel matches
/// [`yuv_420p16_to_rgb_row`] semantically — only the chroma plane
/// layout differs (interleaved vs. two half-width planes). Uses the
/// i32 chroma pipeline (same reasoning as the planar u8 kernel).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p16_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p16_to_rgb_or_rgba_row::<false>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P016** to **8-bit** packed **RGBA**. Same
/// numerical contract as [`p16_to_rgb_row`] except for the per-pixel
/// stride (4 vs 3) and the alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
//
// Scalar prep for Ship 8 Tranche 5a: the public dispatcher
// `row::p016_to_rgba_row` lands in the follow-up SIMD/dispatcher PR.
// Until then this thin wrapper has no caller.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p16_to_rgba_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p16_to_rgb_or_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Shared P016 → 8-bit RGB / RGBA kernel. `ALPHA = false` emits 3 bpp;
/// `ALPHA = true` emits 4 bpp with constant `0xFF` alpha.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = uv_half[c_idx * 2];
    let v_sample = uv_half[c_idx * 2 + 1];
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(y[x] as i32 - y_off, y_scale);
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    let y1 = q15_scale(y[x + 1] as i32 - y_off, y_scale);
    out[(x + 1) * bpp] = clamp_u8(y1 + r_chroma);
    out[(x + 1) * bpp + 1] = clamp_u8(y1 + g_chroma);
    out[(x + 1) * bpp + 2] = clamp_u8(y1 + b_chroma);
    if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}

/// Converts one row of **P016** to **native-depth `u16`** packed
/// RGB — full-range output in `[0, 65535]`. Chroma matrix multiply
/// runs in i64 (same reasoning as [`yuv_420p16_to_rgb_u16_row`]).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p16_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p16_to_rgb_or_rgba_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P016** to **native-depth `u16`** packed
/// **RGBA** — alpha element is `0xFFFF` (opaque maximum at 16-bit).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
//
// Scalar prep for Ship 8 Tranche 5b: the public dispatcher
// `row::p016_to_rgba_u16_row` lands in the follow-up SIMD/dispatcher
// PR. Until then this thin wrapper has no caller.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p16_to_rgba_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p16_to_rgb_or_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Shared P016 → native-depth `u16` RGB / RGBA kernel. `ALPHA = false`
/// emits 3 bpp; `ALPHA = true` emits 4 bpp with constant `0xFFFF`
/// alpha.
///
/// Uses i64 chroma multiply (same rationale as [`yuv_420p16_to_rgb_or_rgba_u16_row`]).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = uv_half[c_idx * 2];
    let v_sample = uv_half[c_idx * 2 + 1];
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale64(y[x] as i32 - y_off, y_scale);
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[x * bpp + 3] = 0xFFFF;
    }

    let y1 = q15_scale64(y[x + 1] as i32 - y_off, y_scale);
    out[(x + 1) * bpp] = (y1 + r_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFFFF;
    }

    x += 2;
  }
}
