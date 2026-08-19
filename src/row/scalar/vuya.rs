//! Scalar reference kernels for the VUYA / VUYX packed YUV 4:4:4 8-bit
//! family (FFmpeg `AV_PIX_FMT_VUYA` / `AV_PIX_FMT_VUYX`). Each pixel
//! is a 4-byte quadruple `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)`.
//!
//! VUYA and VUYX share an identical byte stream; they differ only in α
//! semantics:
//! - VUYA: the A byte is a real alpha channel passed through to RGBA output.
//! - VUYX: the A byte is padding, ignored; RGBA output forces α = 0xFF.
//!
//! One shared kernel template (`vuya_to_rgb_or_rgba_row`) covers all
//! RGB / RGBA conversions via `const` generics. Four public thin
//! wrappers expose the concrete monomorphizations and are consumed by
//! the per-arch SIMD tail handlers, the public dispatchers in
//! [`crate::row::dispatch::vuya`] / [`crate::row::dispatch::vuyx`],
//! and the [`MixedSinker<Vuya>`](crate::sinker::MixedSinker) /
//! [`MixedSinker<Vuyx>`](crate::sinker::MixedSinker) impls.

use super::*;

// ---- shared kernel template --------------------------------------------

/// Shared scalar kernel for the packed 8-bit 4:4:4 YUV family parameterized
/// by per-pixel byte offsets. Covers [`vuya_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp), [`vuya_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp + source-derived alpha) and
/// [`vuyx_to_rgba_row`] (`ALPHA = true, ALPHA_SRC = false`, 4 bpp +
/// opaque alpha). Math is identical; only the per-pixel store stride and
/// alpha byte differ. `const` generic monomorphizes per call site, so
/// the `if ALPHA` / `if ALPHA_SRC` branches are eliminated at compile time.
///
/// The channel byte offsets within each 4-byte source pixel are
/// `const` parameters so the same kernel serves every channel re-ordering
/// of the 4-byte packed family:
/// - `VUYA` / `VUYX`: `V=0, U=1, Y=2, A=3`.
/// - `AYUV` (FFmpeg `AV_PIX_FMT_AYUV`): `A=0, Y=1, U=2, V=3`.
/// - `UYVA` (FFmpeg `AV_PIX_FMT_UYVA`): `U=0, Y=1, V=2, A=3`.
///
/// `A_OFF` is read only when `ALPHA_SRC = true`; for the no-alpha sibling
/// (`Vyu444`, 3 bytes per pixel) use [`vyu444_to_rgb_or_rgba_row`] instead —
/// this kernel hard-codes a 4-byte source stride.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 4`.
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn packed444_to_rgb_or_rgba_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
  const A_OFF: usize,
>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  let bias = chroma_bias::<8>();

  for n in 0..width {
    let base = n * 4;
    let v = packed[base + V_OFF] as i32;
    let u = packed[base + U_OFF] as i32;
    let y = packed[base + Y_OFF] as i32;
    let a = packed[base + A_OFF]; // u8; only used when ALPHA_SRC = true

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let off = n * bpp;
    out[off] = clamp_u8(y_s + r_chroma);
    out[off + 1] = clamp_u8(y_s + g_chroma);
    out[off + 2] = clamp_u8(y_s + b_chroma);
    if ALPHA {
      out[off + 3] = if ALPHA_SRC { a } else { 0xFF };
    }
  }
}

/// VUYA / VUYX channel order (`V=0, U=1, Y=2, A=3`) over the shared
/// [`packed444_to_rgb_or_rgba_row`] kernel.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  packed444_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC, 0, 1, 2, 3>(
    packed, out, width, matrix, full_range,
  );
}

// ---- RGB / RGBA thin wrappers ------------------------------------------

/// Scalar VUYA / VUYX → packed **RGB** (3 bpp). Alpha byte in source is
/// discarded — RGB output has no alpha channel. Used by both VUYA and
/// VUYX because the distinction is irrelevant when there is no α store.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  vuya_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
}

/// Scalar VUYA → packed **RGBA** (4 bpp). The source A byte at offset 3
/// of each pixel quadruple is passed through verbatim.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  vuya_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
}

/// Scalar VUYX → packed **RGBA** (4 bpp). The A byte in source is
/// padding and is ignored; output α is forced to `0xFF` (opaque).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuyx_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range);
}

// ---- VUYA / VUYX → HSV (direct: no RGB scratch) ------------------------
//
// The display-referred twin of [`vuya_to_rgb_row`], fused with the
// OpenCV HSV quantizer. It shares the EXACT per-pixel 8-bit Q15 decode
// (`Coefficients::for_matrix` + `range_params_n::<8, 8>` + the
// V/U/Y slot extraction) as its `_to_rgb` sibling, then feeds the
// decoded `(r, g, b)` straight into [`rgb_to_hsv_pixel`] and scatters to
// the H/S/V planes — never materializing a packed-RGB row. The A byte
// (slot 3) is independent of HSV — HSV derives from the COLOR
// (V/U/Y → RGB → HSV) only — so VUYA's real α and VUYX's padding byte
// are both irrelevant here and a single kernel serves both. Byte-
// identical to `rgb_to_hsv_row(vuya_to_rgb_row(...))`, with no RGB
// allocation. VUYX HSV is exposed as a thin re-export
// ([`vuyx_to_hsv_row`]) of this kernel — the byte streams (and thus the
// colour) are identical regardless of α semantics.

/// Scalar VUYA / VUYX → planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV`
/// encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). 4:4:4 (no chroma
/// subsampling): one V/U/Y triple per pixel. The α byte (slot 3) is
/// dropped — HSV is colour-only — so this serves both VUYA (real α) and
/// VUYX (padding). Byte-identical to `rgb_to_hsv_row(vuya_to_rgb_row(...))`.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 4`.
/// - each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn packed444_to_hsv_row<const V_OFF: usize, const U_OFF: usize, const Y_OFF: usize>(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  let bias = chroma_bias::<8>();

  for n in 0..width {
    let base = n * 4;
    let v = packed[base + V_OFF] as i32;
    let u = packed[base + U_OFF] as i32;
    let y = packed[base + Y_OFF] as i32;
    // The 4th slot (A / X) is intentionally discarded — HSV is colour-only.

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let (h, s, vv) = rgb_to_hsv_pixel(
      clamp_u8(y_s + r_chroma) as i32,
      clamp_u8(y_s + g_chroma) as i32,
      clamp_u8(y_s + b_chroma) as i32,
    );
    h_out[n] = h;
    s_out[n] = s;
    v_out[n] = vv;
  }
}

/// VUYA / VUYX channel order (`V=0, U=1, Y=2`) over the shared
/// [`packed444_to_hsv_row`] kernel.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  packed444_to_hsv_row::<0, 1, 2>(packed, h_out, s_out, v_out, width, matrix, full_range);
}

// ---- Luma extraction ---------------------------------------------------

/// Copies only the Y bytes from a packed 4-byte 4:4:4 row into a
/// `width`-byte luma plane. Avoids the YUV→RGB pipeline entirely when
/// only luma is needed. `Y_OFF` is the Y byte's offset within each 4-byte
/// pixel (`2` for VUYA / VUYX, `1` for AYUV / UYVA).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn packed444_to_luma_row<const Y_OFF: usize>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  for n in 0..width {
    luma_out[n] = packed[n * 4 + Y_OFF];
  }
}

/// VUYA / VUYX luma (Y at offset 2) over [`packed444_to_luma_row`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  packed444_to_luma_row::<2>(packed, luma_out, width);
}

/// Extract Y as u16 (zero-extended) from a packed 4-byte 4:4:4 row.
/// `Y_OFF` is the Y byte's offset within each 4-byte pixel (`2` for
/// VUYA / VUYX, `1` for AYUV / UYVA); the other bytes are ignored. Output
/// is `Y_byte as u16` — no shift, just widening.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn packed444_to_luma_u16_row<const Y_OFF: usize>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(out.len() >= width, "out too short");
  for x in 0..width {
    out[x] = packed[x * 4 + Y_OFF] as u16;
  }
}

/// VUYA / VUYX u16 luma (Y at offset 2) over [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  packed444_to_luma_u16_row::<2>(packed, out, width);
}

/// Extract Y as u16 (zero-extended) from a packed VUYX `[V, U, Y, X]` row.
/// Byte-identical to [`vuya_to_luma_u16_row`] — Y is at byte offset 2 of
/// each 4-byte pixel quadruple regardless of α semantics; the X byte is
/// ignored. Output is `Y_byte as u16`.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuyx_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  vuya_to_luma_u16_row(packed, out, width);
}

// ---- AYUV (A=0, Y=1, U=2, V=3) thin wrappers --------------------------

/// Scalar AYUV → packed **RGB** (3 bpp). The source A byte (offset 0) is
/// discarded — RGB output has no alpha channel.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  packed444_to_rgb_or_rgba_row::<false, false, 3, 2, 1, 0>(
    packed, rgb_out, width, matrix, full_range,
  );
}

/// Scalar AYUV → packed **RGBA** (4 bpp). The source A byte at offset 0 of
/// each pixel quadruple is passed through verbatim to RGBA slot 3.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  packed444_to_rgb_or_rgba_row::<true, true, 3, 2, 1, 0>(
    packed, rgba_out, width, matrix, full_range,
  );
}

/// Scalar AYUV → planar HSV bytes (OpenCV encoding). The A byte (offset 0)
/// is dropped — HSV is colour-only.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  packed444_to_hsv_row::<3, 2, 1>(packed, h_out, s_out, v_out, width, matrix, full_range);
}

/// Scalar AYUV → u8 luma (Y at offset 1).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  packed444_to_luma_row::<1>(packed, luma_out, width);
}

/// Scalar AYUV → u16 luma (zero-extended Y at offset 1).
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  packed444_to_luma_u16_row::<1>(packed, out, width);
}

// ---- UYVA (U=0, Y=1, V=2, A=3) thin wrappers --------------------------

/// Scalar UYVA → packed **RGB** (3 bpp). The source A byte (offset 3) is
/// discarded — RGB output has no alpha channel.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyva_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  packed444_to_rgb_or_rgba_row::<false, false, 2, 0, 1, 3>(
    packed, rgb_out, width, matrix, full_range,
  );
}

/// Scalar UYVA → packed **RGBA** (4 bpp). The source A byte at offset 3 of
/// each pixel quadruple is passed through verbatim to RGBA slot 3.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyva_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  packed444_to_rgb_or_rgba_row::<true, true, 2, 0, 1, 3>(
    packed, rgba_out, width, matrix, full_range,
  );
}

/// Scalar UYVA → planar HSV bytes (OpenCV encoding). The A byte (offset 3)
/// is dropped — HSV is colour-only.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyva_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  packed444_to_hsv_row::<2, 0, 1>(packed, h_out, s_out, v_out, width, matrix, full_range);
}

/// Scalar UYVA → u8 luma (Y at offset 1).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyva_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  packed444_to_luma_row::<1>(packed, luma_out, width);
}

/// Scalar UYVA → u16 luma (zero-extended Y at offset 1).
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyva_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  packed444_to_luma_u16_row::<1>(packed, out, width);
}

// ---- VYU444 (V=0, Y=1, U=2; 3 bytes per pixel, no alpha) --------------
//
// VYU444 (FFmpeg `AV_PIX_FMT_VYU444`) packs **three** bytes per pixel
// (`V(8) ‖ Y(8) ‖ U(8)`) — 24bpp, no alpha. The source stride is 3, not
// 4, so it cannot share the 4-byte [`packed444_to_rgb_or_rgba_row`]
// kernel; these dedicated kernels walk the source in 3-byte steps. RGBA
// output always forces α = `0xFF` (there is no source alpha to carry).

/// Shared scalar VYU444 kernel: → packed **RGB** (`ALPHA = false`, 3 bpp)
/// or → packed **RGBA** (`ALPHA = true`, 4 bpp, α forced `0xFF`). The
/// per-pixel decode math is identical to the 4-byte family; only the
/// 3-byte source stride and the V/Y/U byte offsets differ.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 3`.
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vyu444_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  let bias = chroma_bias::<8>();

  for n in 0..width {
    let base = n * 3;
    let v = packed[base] as i32; // V at offset 0
    let y = packed[base + 1] as i32; // Y at offset 1
    let u = packed[base + 2] as i32; // U at offset 2

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let off = n * bpp;
    out[off] = clamp_u8(y_s + r_chroma);
    out[off + 1] = clamp_u8(y_s + g_chroma);
    out[off + 2] = clamp_u8(y_s + b_chroma);
    if ALPHA {
      out[off + 3] = 0xFF;
    }
  }
}

/// Scalar VYU444 → packed **RGB** (3 bpp).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vyu444_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  vyu444_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Scalar VYU444 → packed **RGBA** (4 bpp). α is forced to `0xFF` (the
/// source carries no alpha).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vyu444_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  vyu444_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Scalar VYU444 → planar HSV bytes (OpenCV encoding). 3-byte source.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vyu444_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  let bias = chroma_bias::<8>();

  for n in 0..width {
    let base = n * 3;
    let v = packed[base] as i32;
    let y = packed[base + 1] as i32;
    let u = packed[base + 2] as i32;

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let (h, s, vv) = rgb_to_hsv_pixel(
      clamp_u8(y_s + r_chroma) as i32,
      clamp_u8(y_s + g_chroma) as i32,
      clamp_u8(y_s + b_chroma) as i32,
    );
    h_out[n] = h;
    s_out[n] = s;
    v_out[n] = vv;
  }
}

/// Scalar VYU444 → u8 luma (Y at offset 1, 3-byte stride).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vyu444_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  for n in 0..width {
    luma_out[n] = packed[n * 3 + 1];
  }
}

/// Scalar VYU444 → u16 luma (zero-extended Y at offset 1, 3-byte stride).
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vyu444_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 3, "packed too short");
  debug_assert!(out.len() >= width, "out too short");
  for x in 0..width {
    out[x] = packed[x * 3 + 1] as u16;
  }
}

// ---- Tests -------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::KernelMatrix;

  /// Build a 4-byte VUYA pixel from explicit components.
  fn pack_vuya(v: u8, u: u8, y: u8, a: u8) -> [u8; 4] {
    [v, u, y, a]
  }

  #[test]
  fn vuya_known_pattern_rgb_limited_range() {
    // Limited-range BT.709, neutral chroma U=V=128.
    // Black: Y=16 (limited-range black). White: Y=235 (limited-range white).
    let p_black = pack_vuya(128, 128, 16, 0);
    let p_white = pack_vuya(128, 128, 235, 0);
    let packed: Vec<u8> = [p_black, p_black, p_white, p_white]
      .iter()
      .flatten()
      .copied()
      .collect();
    let mut out = vec![0u8; 4 * 3];
    vuya_to_rgb_row(&packed, &mut out, 4, KernelMatrix::Bt709, false);
    // Black pixels → [0, 0, 0]
    assert_eq!(&out[0..3], &[0u8, 0, 0], "black pixel 0");
    assert_eq!(&out[3..6], &[0u8, 0, 0], "black pixel 1");
    // White pixels → [255, 255, 255]
    assert_eq!(&out[6..9], &[255u8, 255, 255], "white pixel 2");
    assert_eq!(&out[9..12], &[255u8, 255, 255], "white pixel 3");
  }

  #[test]
  fn vuya_rgba_passes_source_alpha() {
    // VUYA: source A bytes 0x42 and 0x99 must appear verbatim in output.
    let p0 = pack_vuya(128, 128, 16, 0x42);
    let p1 = pack_vuya(128, 128, 235, 0x99);
    let packed: Vec<u8> = [p0, p1].iter().flatten().copied().collect();
    let mut out = vec![0u8; 2 * 4];
    vuya_to_rgba_row(&packed, &mut out, 2, KernelMatrix::Bt709, false);
    assert_eq!(out[3], 0x42, "pixel 0 alpha");
    assert_eq!(out[7], 0x99, "pixel 1 alpha");
  }

  #[test]
  fn vuyx_rgba_forces_alpha_max_regardless_of_source() {
    // VUYX: A byte in source is padding; output must be 0xFF for both pixels.
    let p0 = pack_vuya(128, 128, 16, 0x42);
    let p1 = pack_vuya(128, 128, 235, 0x99);
    let packed: Vec<u8> = [p0, p1].iter().flatten().copied().collect();
    let mut out = vec![0u8; 2 * 4];
    vuyx_to_rgba_row(&packed, &mut out, 2, KernelMatrix::Bt709, false);
    assert_eq!(out[3], 0xFF, "pixel 0 alpha should be 0xFF");
    assert_eq!(out[7], 0xFF, "pixel 1 alpha should be 0xFF");
  }

  #[test]
  fn vuya_luma_extract() {
    // Y is at offset 2 of each quadruple; V/U/A are irrelevant.
    let p0 = pack_vuya(0, 0, 0xFF, 0);
    let p1 = pack_vuya(0, 0, 0x40, 0);
    let packed: Vec<u8> = [p0, p1].iter().flatten().copied().collect();
    let mut luma = vec![0u8; 2];
    vuya_to_luma_row(&packed, &mut luma, 2);
    assert_eq!(&luma[..], &[0xFFu8, 0x40]);
  }

  // ---- AYUV / UYVA / VYU444 channel-order parity --------------------------
  //
  // The new formats are byte-for-byte channel re-orderings of the proven
  // VUYA / VUYX kernels. For each, pack the SAME (v, u, y, a) samples in the
  // format's own byte order and assert the colour outputs are byte-identical
  // to the reference VUYA / VUYX result.

  /// Build a 4-byte AYUV pixel (`A, Y, U, V`) from V/U/Y/A samples.
  fn pack_ayuv(v: u8, u: u8, y: u8, a: u8) -> [u8; 4] {
    [a, y, u, v]
  }
  /// Build a 4-byte UYVA pixel (`U, Y, V, A`) from V/U/Y/A samples.
  fn pack_uyva(v: u8, u: u8, y: u8, a: u8) -> [u8; 4] {
    [u, y, v, a]
  }
  /// Build a 3-byte VYU444 pixel (`V, Y, U`) from V/U/Y samples.
  fn pack_vyu444(v: u8, u: u8, y: u8) -> [u8; 3] {
    [v, y, u]
  }

  /// A fixed assortment of (V, U, Y, A) samples spanning limited/full range.
  const SAMPLES: [(u8, u8, u8, u8); 6] = [
    (128, 128, 16, 0x00),
    (128, 128, 235, 0xFF),
    (240, 16, 128, 0x42),
    (16, 240, 200, 0x99),
    (200, 60, 90, 0x10),
    (90, 170, 30, 0xEE),
  ];

  fn pack_n<const N: usize>(f: impl Fn(u8, u8, u8, u8) -> [u8; N]) -> Vec<u8> {
    SAMPLES
      .iter()
      .flat_map(|&(v, u, y, a)| f(v, u, y, a))
      .collect()
  }

  #[test]
  fn ayuv_rgb_matches_vuya_reference() {
    let w = SAMPLES.len();
    let vuya = pack_n(pack_vuya);
    let ayuv = pack_n(pack_ayuv);
    for &fr in &[false, true] {
      for &m in &[
        KernelMatrix::Bt709,
        KernelMatrix::Bt601,
        KernelMatrix::Bt2020Ncl,
      ] {
        let mut a = vec![0u8; w * 3];
        let mut b = vec![0u8; w * 3];
        vuya_to_rgb_row(&vuya, &mut a, w, m, fr);
        ayuv_to_rgb_row(&ayuv, &mut b, w, m, fr);
        assert_eq!(a, b, "AYUV RGB mismatch (full_range={fr}, matrix={m:?})");
      }
    }
  }

  #[test]
  fn ayuv_rgba_passes_source_alpha_and_matches_color() {
    let w = SAMPLES.len();
    let vuya = pack_n(pack_vuya);
    let ayuv = pack_n(pack_ayuv);
    let mut a = vec![0u8; w * 4];
    let mut b = vec![0u8; w * 4];
    vuya_to_rgba_row(&vuya, &mut a, w, KernelMatrix::Bt709, false);
    ayuv_to_rgba_row(&ayuv, &mut b, w, KernelMatrix::Bt709, false);
    assert_eq!(a, b, "AYUV RGBA (incl. source alpha) must match VUYA");
    // Spot-check the alpha bytes are the source A values.
    for (i, &(_, _, _, av)) in SAMPLES.iter().enumerate() {
      assert_eq!(b[i * 4 + 3], av, "AYUV pixel {i} alpha");
    }
  }

  #[test]
  fn ayuv_luma_and_hsv_match_vuya() {
    let w = SAMPLES.len();
    let vuya = pack_n(pack_vuya);
    let ayuv = pack_n(pack_ayuv);
    let mut la = vec![0u8; w];
    let mut lb = vec![0u8; w];
    vuya_to_luma_row(&vuya, &mut la, w);
    ayuv_to_luma_row(&ayuv, &mut lb, w);
    assert_eq!(la, lb, "AYUV luma must match VUYA");
    let (mut ha, mut sa, mut va) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
    let (mut hb, mut sb, mut vb) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
    vuya_to_hsv_row(
      &vuya,
      &mut ha,
      &mut sa,
      &mut va,
      w,
      KernelMatrix::Bt709,
      false,
    );
    ayuv_to_hsv_row(
      &ayuv,
      &mut hb,
      &mut sb,
      &mut vb,
      w,
      KernelMatrix::Bt709,
      false,
    );
    assert_eq!((ha, sa, va), (hb, sb, vb), "AYUV HSV must match VUYA");
  }

  #[test]
  fn uyva_rgb_rgba_match_vuya_reference() {
    let w = SAMPLES.len();
    let vuya = pack_n(pack_vuya);
    let uyva = pack_n(pack_uyva);
    for &fr in &[false, true] {
      for &m in &[
        KernelMatrix::Bt709,
        KernelMatrix::Bt601,
        KernelMatrix::Bt2020Ncl,
      ] {
        let mut a = vec![0u8; w * 3];
        let mut b = vec![0u8; w * 3];
        vuya_to_rgb_row(&vuya, &mut a, w, m, fr);
        uyva_to_rgb_row(&uyva, &mut b, w, m, fr);
        assert_eq!(a, b, "UYVA RGB mismatch (full_range={fr}, matrix={m:?})");
      }
    }
    let mut a = vec![0u8; w * 4];
    let mut b = vec![0u8; w * 4];
    vuya_to_rgba_row(&vuya, &mut a, w, KernelMatrix::Bt709, true);
    uyva_to_rgba_row(&uyva, &mut b, w, KernelMatrix::Bt709, true);
    assert_eq!(a, b, "UYVA RGBA (incl. source alpha) must match VUYA");
  }

  #[test]
  fn uyva_luma_and_hsv_match_vuya() {
    let w = SAMPLES.len();
    let vuya = pack_n(pack_vuya);
    let uyva = pack_n(pack_uyva);
    let mut la = vec![0u8; w];
    let mut lb = vec![0u8; w];
    vuya_to_luma_row(&vuya, &mut la, w);
    uyva_to_luma_row(&uyva, &mut lb, w);
    assert_eq!(la, lb, "UYVA luma must match VUYA");
    let (mut ha, mut sa, mut va) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
    let (mut hb, mut sb, mut vb) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
    vuya_to_hsv_row(
      &vuya,
      &mut ha,
      &mut sa,
      &mut va,
      w,
      KernelMatrix::Bt2020Ncl,
      true,
    );
    uyva_to_hsv_row(
      &uyva,
      &mut hb,
      &mut sb,
      &mut vb,
      w,
      KernelMatrix::Bt2020Ncl,
      true,
    );
    assert_eq!((ha, sa, va), (hb, sb, vb), "UYVA HSV must match VUYA");
  }

  #[test]
  fn vyu444_rgb_matches_vuyx_reference() {
    // VYU444 has no alpha; its RGB / RGBA(α=0xFF) must match VUYX with the
    // same V/U/Y samples (the padding byte in VUYX is irrelevant for RGB).
    let w = SAMPLES.len();
    let vuyx = pack_n(pack_vuya); // VUYX shares the VUYA byte layout
    let vyu = pack_n(|v, u, y, _a| pack_vyu444(v, u, y));
    for &fr in &[false, true] {
      for &m in &[
        KernelMatrix::Bt709,
        KernelMatrix::Bt601,
        KernelMatrix::Bt2020Ncl,
      ] {
        let mut a = vec![0u8; w * 3];
        let mut b = vec![0u8; w * 3];
        vuya_to_rgb_row(&vuyx, &mut a, w, m, fr);
        vyu444_to_rgb_row(&vyu, &mut b, w, m, fr);
        assert_eq!(a, b, "VYU444 RGB mismatch (full_range={fr}, matrix={m:?})");
      }
    }
  }

  #[test]
  fn vyu444_rgba_forces_opaque_alpha_and_matches_color() {
    let w = SAMPLES.len();
    let vuyx = pack_n(pack_vuya);
    let vyu = pack_n(|v, u, y, _a| pack_vyu444(v, u, y));
    let mut a = vec![0u8; w * 4];
    let mut b = vec![0u8; w * 4];
    vuyx_to_rgba_row(&vuyx, &mut a, w, KernelMatrix::Bt709, false);
    vyu444_to_rgba_row(&vyu, &mut b, w, KernelMatrix::Bt709, false);
    assert_eq!(a, b, "VYU444 RGBA must match VUYX (α forced 0xFF)");
    for i in 0..w {
      assert_eq!(b[i * 4 + 3], 0xFF, "VYU444 pixel {i} alpha must be 0xFF");
    }
  }

  #[test]
  fn vyu444_luma_and_hsv_match_vuyx() {
    let w = SAMPLES.len();
    let vuyx = pack_n(pack_vuya);
    let vyu = pack_n(|v, u, y, _a| pack_vyu444(v, u, y));
    let mut la = vec![0u8; w];
    let mut lb = vec![0u8; w];
    vuya_to_luma_row(&vuyx, &mut la, w);
    vyu444_to_luma_row(&vyu, &mut lb, w);
    assert_eq!(la, lb, "VYU444 luma must match VUYX");
    let (mut ha, mut sa, mut va) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
    let (mut hb, mut sb, mut vb) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
    vuya_to_hsv_row(
      &vuyx,
      &mut ha,
      &mut sa,
      &mut va,
      w,
      KernelMatrix::Bt601,
      false,
    );
    vyu444_to_hsv_row(
      &vyu,
      &mut hb,
      &mut sb,
      &mut vb,
      w,
      KernelMatrix::Bt601,
      false,
    );
    assert_eq!((ha, sa, va), (hb, sb, vb), "VYU444 HSV must match VUYX");
  }
}
