use super::*;

// ---- P010 (semi-planar 10-bit, high-bit-packed) → RGB ------------------

/// Converts one row of P010 (semi‑planar 4:2:0 with UV interleaved,
/// `BITS` active bits in the **high** `BITS` of each `u16`) to
/// **8‑bit** packed RGB.
///
/// Structurally identical to [`nv12_to_rgb_row`] plus the per‑sample
/// shift: each `u16` load is extracted to its `BITS`‑bit value via
/// `sample >> (16 - BITS)`, then the same Q15 pipeline as
/// [`yuv_420p_n_to_rgb_row`] runs with the same `BITS`. For `BITS ==
/// 10` this is P010 (`>> 6`); for `BITS == 12` it's P012 (`>> 4`).
/// Mispacked input — e.g. a low‑bit‑packed buffer handed to this
/// kernel — has its active low bits discarded (producing near‑black
/// output), matching every SIMD backend.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_row::<BITS, false>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of high‑bit‑packed semi‑planar 4:2:0 (P010/P012)
/// to **8‑bit** packed **RGBA**. Same numerical contract as
/// [`p_n_to_rgb_row`]; the only differences are the per-pixel stride
/// (4 vs 3) and the alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
//
// Scalar prep for Ship 8 Tranche 5a: the public dispatchers
// `row::p010_to_rgba_row` and `row::p012_to_rgba_row` land in the
// follow-up SIMD/dispatcher PR. Until then this thin wrapper has no
// caller. P016 has its own kernel family
// ([`p16_to_rgb_or_rgba_row`]) — never routed here.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_row::<BITS, true>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Shared kernel for [`p_n_to_rgb_row`] (`ALPHA = false`, 3 bpp store)
/// and [`p_n_to_rgba_row`] (`ALPHA = true`, 4 bpp store with constant
/// `0xFF` alpha).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // High-bit-packed Pn kernels are only defined for BITS in {10, 12}.
  // Outside that set, `16 - BITS` could under/overflow and the Q15
  // coefficient table has no corresponding entry. P016 (BITS=16) has
  // its own dedicated kernel family with i64 chroma multiply — using
  // this i32 path at BITS=16 would silently overflow on high chroma
  // values. The compile-time assertion fails monomorphization for any
  // BITS outside {10, 12}, eliminating that release-build corruption
  // trap.
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let shift = 16 - BITS;

  // Each `u16` load is converted to its `BITS`-bit sample with
  // `>> (16 - BITS)` — 6 for P010, 4 for P012. Extracts the upper
  // bits and leaves the result in `[0, (1 << BITS) - 1]`. If
  // low-packed input (`yuv420p10le`, `yuv420p12le`) is handed to
  // this kernel by mistake, the shift discards the active low bits
  // rather than recovering the intended value. No hot-path cost:
  // one shift per load.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = uv_half[c_idx * 2] >> shift;
    let v_sample = uv_half[c_idx * 2 + 1] >> shift;
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] >> shift) as i32 - y_off, y_scale);
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    let y1 = q15_scale((y[x + 1] >> shift) as i32 - y_off, y_scale);
    out[(x + 1) * bpp] = clamp_u8(y1 + r_chroma);
    out[(x + 1) * bpp + 1] = clamp_u8(y1 + g_chroma);
    out[(x + 1) * bpp + 2] = clamp_u8(y1 + b_chroma);
    if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}

/// Converts one row of high‑bit‑packed semi‑planar 4:2:0
/// (`BITS` ∈ {10, 12}: P010, P012) to **native‑depth `u16`**
/// packed RGB — samples are **low‑bit‑packed** on output
/// (`[0, (1 << BITS) - 1]` in the low bits of each `u16`, upper bits
/// zero), matching the `yuv420p10le` / `yuv420p12le` convention —
/// **not** the P010/P012 high‑bit packing. Callers feeding a P010/
/// P012 consumer must shift the output left by `16 - BITS`.
///
/// Mirrors [`yuv_420p_n_to_rgb_u16_row`] on the math side; the only
/// differences are the input shift (`sample >> (16 - BITS)` to
/// extract the `BITS`-bit value from the high-bit packing) and the
/// interleaved UV layout.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of high‑bit‑packed semi‑planar 4:2:0 (P010/P012)
/// to **native‑depth `u16`** packed **RGBA** — output is low‑bit‑packed
/// to match [`p_n_to_rgb_u16_row`]. Alpha is `(1 << BITS) - 1` (opaque
/// maximum at the input bit depth).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
//
// Scalar prep for Ship 8 Tranche 5b: the public dispatchers
// `row::p010_to_rgba_u16_row` and `row::p012_to_rgba_u16_row` land in
// the follow-up SIMD/dispatcher PR. Until then this thin wrapper has
// no caller. P016 has its own u16 kernel family
// ([`p16_to_rgb_or_rgba_u16_row`]) — never routed here.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_u16_row::<BITS, true>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Shared kernel for [`p_n_to_rgb_u16_row`] (`ALPHA = false`, 3 bpp
/// store) and [`p_n_to_rgba_u16_row`] (`ALPHA = true`, 4 bpp store
/// with opaque alpha = `(1 << BITS) - 1`).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }` (`u16` elements).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // See `p_n_to_rgb_or_rgba_row` for the BITS range rationale. The
  // P016 u16 path lives in [`p16_to_rgb_or_rgba_u16_row`] (i64 chroma
  // multiply); this i32 path would overflow before clamp at 16-bit
  // chroma. Compile-time assertion eliminates the release-build
  // corruption trap.
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let shift = 16 - BITS;
  let alpha_max: u16 = out_max as u16;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = uv_half[c_idx * 2] >> shift;
    let v_sample = uv_half[c_idx * 2 + 1] >> shift;
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] >> shift) as i32 - y_off, y_scale);
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[x * bpp + 3] = alpha_max;
    }

    let y1 = q15_scale((y[x + 1] >> shift) as i32 - y_off, y_scale);
    out[(x + 1) * bpp] = (y1 + r_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[(x + 1) * bpp + 3] = alpha_max;
    }

    x += 2;
  }
}

// ---- Pn 4:4:4 (semi-planar high-bit-packed) → RGB ----------------------
//
// Mirrors `p_n_to_rgb_*<BITS>` but with full-width interleaved UV: one
// `U, V` pair per pixel (= `2 * width` u16 elements per row), no
// horizontal duplication. Same `>> (16 - BITS)` extraction at load
// time. BITS ∈ {10, 12} on the i32 Q15 pipeline; BITS = 16 lives in
// `p_n_444_16_to_rgb_*` because the chroma multiply-add overflows
// i32 at u16 output (same rationale as p16 / yuv_444p16).

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **8-bit** packed RGB. `BITS ∈ {10, 12}`. Each `u16` load is
/// shifted right by `16 - BITS` to extract the active value before
/// running the standard Q15 i32 pipeline.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_to_rgb_or_rgba_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
}

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **8-bit** packed **RGBA**. Same numerical contract as
/// [`p_n_444_to_rgb_row`]; the only differences are the per-pixel
/// stride (4 vs 3) and the alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgba_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_to_rgb_or_rgba_row::<BITS, true>(y, uv_full, rgba_out, width, matrix, full_range);
}

/// Shared kernel for [`p_n_444_to_rgb_row`] (`ALPHA = false`, 3 bpp
/// store) and [`p_n_444_to_rgba_row`] (`ALPHA = true`, 4 bpp store
/// with constant `0xFF` alpha).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let shift = 16 - BITS;

  for x in 0..width {
    // 4:4:4: one UV pair per pixel — uv_full[x*2] = U, uv_full[x*2+1] = V.
    let u_sample = uv_full[x * 2] >> shift;
    let v_sample = uv_full[x * 2 + 1] >> shift;
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] >> shift) as i32 - y_off, y_scale);
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **native-depth `u16`** packed RGB — low-bit-packed output (the
/// `BITS` active bits in the **low** bits of each `u16`, upper bits
/// zero), matching the [`yuv_444p_n_to_rgb_u16_row`] convention.
/// `BITS ∈ {10, 12}`.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
}

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **native-depth `u16`** packed **RGBA** — low-bit-packed output;
/// alpha element is `(1 << BITS) - 1` (opaque maximum at the input
/// bit depth).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_to_rgb_or_rgba_u16_row::<BITS, true>(y, uv_full, rgba_out, width, matrix, full_range);
}

/// Shared kernel for [`p_n_444_to_rgb_u16_row`] (`ALPHA = false`,
/// 3 bpp store) and [`p_n_444_to_rgba_u16_row`] (`ALPHA = true`,
/// 4 bpp store with opaque alpha = `(1 << BITS) - 1`).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }` (`u16` elements).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let shift = 16 - BITS;
  let alpha_max: u16 = out_max as u16;

  for x in 0..width {
    let u_sample = uv_full[x * 2] >> shift;
    let v_sample = uv_full[x * 2 + 1] >> shift;
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] >> shift) as i32 - y_off, y_scale);
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[x * bpp + 3] = alpha_max;
    }
  }
}

/// Converts one row of P416 (semi-planar 4:4:4, 16-bit, full UV) to
/// **8-bit** packed RGB. Y and chroma both stay on i32 — same logic
/// as `p16_to_rgb_row` plus the full-width UV layout.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_16_to_rgb_or_rgba_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
}

/// Converts one row of P416 to **8-bit** packed **RGBA**. Same
/// numerical contract as [`p_n_444_16_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgba_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_16_to_rgb_or_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range);
}

/// Shared P416 → 8-bit RGB / RGBA kernel. `ALPHA = false` emits 3 bpp;
/// `ALPHA = true` emits 4 bpp with constant `0xFF` alpha.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let u_sample = uv_full[x * 2];
    let v_sample = uv_full[x * 2 + 1];
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
  }
}

/// Converts one row of P416 to **native-depth `u16`** packed RGB —
/// full-range output in `[0, 65535]`. Chroma multiply-add runs in i64
/// (same rationale as `p16_to_rgb_u16_row` and
/// `yuv_444p16_to_rgb_u16_row`: `coeff × u_d` overflows i32 at 16
/// bits for the BT.2020 blue coefficient).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_16_to_rgb_or_rgba_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
}

/// Converts one row of P416 to **native-depth `u16`** packed
/// **RGBA** — full-range output `[0, 65535]`; alpha element is
/// `0xFFFF`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgba_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_16_to_rgb_or_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range);
}

/// Shared P416 → native-depth `u16` RGB / RGBA kernel. `ALPHA = false`
/// emits 3 bpp; `ALPHA = true` emits 4 bpp with constant `0xFFFF`
/// alpha. Uses i64 chroma multiply (same rationale as
/// [`p_n_444_16_to_rgb_u16_row`]).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  for x in 0..width {
    let u_sample = uv_full[x * 2];
    let v_sample = uv_full[x * 2 + 1];
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
  }
}
