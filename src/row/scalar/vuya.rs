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
//! wrappers expose the concrete monomorphizations.
//!
//! The dispatch layer (`src/row/dispatch/vuya.rs`) and sinker
//! implementations are added in later tasks (Tasks 5-13). Until then
//! the `pub(crate)` symbols are forward-declared here.

use super::*;

// ---- shared kernel template --------------------------------------------

/// Shared scalar kernel for [`vuya_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp), [`vuya_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp + source-derived alpha) and
/// [`vuyx_to_rgba_row`] (`ALPHA = true, ALPHA_SRC = false`, 4 bpp +
/// opaque alpha). Math is identical; only the per-pixel store stride and
/// alpha byte differ. `const` generic monomorphizes per call site, so
/// the `if ALPHA` / `if ALPHA_SRC` branches are eliminated at compile time.
///
/// Input layout per pixel `n`: `packed[n*4] = V`, `packed[n*4+1] = U`,
/// `packed[n*4+2] = Y`, `packed[n*4+3] = A`.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 4`.
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
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
    let v = packed[base] as i32;
    let u = packed[base + 1] as i32;
    let y = packed[base + 2] as i32;
    let a = packed[base + 3]; // u8; only used when ALPHA_SRC = true

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

// ---- RGB / RGBA thin wrappers ------------------------------------------

/// Scalar VUYA / VUYX → packed **RGB** (3 bpp). Alpha byte in source is
/// discarded — RGB output has no alpha channel. Used by both VUYA and
/// VUYX because the distinction is irrelevant when there is no α store.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
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
  matrix: ColorMatrix,
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
  matrix: ColorMatrix,
  full_range: bool,
) {
  vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range);
}

// ---- Luma extraction ---------------------------------------------------

/// Copies only the Y bytes from a packed VUYA / VUYX row into a
/// `width`-byte luma plane. Avoids the YUV→RGB pipeline entirely when
/// only luma is needed. Y is at byte offset 2 of each pixel quadruple.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  for n in 0..width {
    luma_out[n] = packed[n * 4 + 2];
  }
}

// ---- Tests -------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

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
    vuya_to_rgb_row(&packed, &mut out, 4, ColorMatrix::Bt709, false);
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
    vuya_to_rgba_row(&packed, &mut out, 2, ColorMatrix::Bt709, false);
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
    vuyx_to_rgba_row(&packed, &mut out, 2, ColorMatrix::Bt709, false);
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
}
