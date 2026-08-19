//! Scalar reference kernels for the Y2xx packed YUV 4:2:2
//! high-bit-depth family — `Y210` (BITS=10) and `Y212` (BITS=12).
//! `Y216` (BITS=16) gets a parallel kernel family in Ship 11d
//! (the i64 chroma path for u16 output is structurally different).
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)`, MSB-aligned
//! with `(16 - BITS)` low bits zero. Each kernel right-shifts at
//! load to bring samples into the BITS-aligned `[0, 2^BITS - 1]`
//! range, then runs the standard Q15 chroma + Y pipeline at
//! `BITS` (mirrors `v210.rs`'s use of `range_params_n` /
//! `chroma_bias` / `q15_scale` / `q15_chroma`, just sourced from
//! Y2xx's u16 packed quadruples rather than v210's 16-byte words).
//!
//! ## Big-endian wire format (`BE = true`)
//!
//! When `BE = true`, each `u16` element in `packed` is stored in
//! big-endian byte order (high byte first). The `<const BE: bool>`
//! const-generic gates `load_endian_u16::<BE>` at each sample read
//! site; on LE targets the `BE = false` path is identical to the
//! previous plain slice index. On LE hosts with `BE = false` the
//! compiler eliminates the branch entirely.

use super::*;

/// Bring the BITS-aligned active samples to `[0, 2^BITS - 1]` by
/// right-shifting `(16 - BITS)`. For BITS=16 this is a no-op (but
/// the Y216 family lives in Ship 11d; this kernel asserts BITS ∈
/// {10, 12}).
#[cfg_attr(not(tarpaulin), inline(always))]
const fn rshift_bits<const BITS: u32>(sample: u16) -> u16 {
  sample >> (16 - BITS)
}

// ---- u8 RGB / RGBA output ----------------------------------------------

/// Y2xx → packed RGB / RGBA u8 path. Const-generic over BITS ∈
/// {10, 12} (BITS=16 uses the parallel `y216` family in Ship 11d).
/// `ALPHA = false` writes 3 bytes per pixel; `ALPHA = true` writes 4
/// bytes per pixel with `α = 0xFF`. Output bit-depth is u8
/// (downshifted from the native BITS Q15 pipeline via
/// `range_params_n::<BITS, 8>`).
///
/// `BE = true` selects big-endian wire decoding for each u16 sample.
///
/// # Panics (debug builds)
/// - `width` must be even.
/// - `packed.len() >= width * 2` (one u16 quadruple per chroma pair).
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y2xx_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_rgb requires BITS in {{10, 12}}"
    );
  }
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2), "Y2xx requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  // One chroma pair (= 2 pixels) per iter.
  let pairs = width / 2;
  // SAFETY: bounds checked by the debug_asserts above; p * 4 + 4 <= width * 2
  // because pairs = width / 2, so p < pairs means p * 4 + 4 <= width * 2.
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2; // byte offset to quadruple p (4 u16 = 8 bytes)
    let y0 = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4)) }) as i32;
    let u = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 2)) }) as i32;
    let y1 = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) }) as i32;
    let v = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 6)) }) as i32;

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    for (k, &y) in [y0, y1].iter().enumerate() {
      let y_s = q15_scale(y - y_off, y_scale);
      let off = (p * 2 + k) * bpp;
      out[off] = clamp_u8(y_s + r_chroma);
      out[off + 1] = clamp_u8(y_s + g_chroma);
      out[off + 2] = clamp_u8(y_s + b_chroma);
      if ALPHA {
        out[off + 3] = 0xFF;
      }
    }
  }
}

// ---- u16 RGB / RGBA native-depth output --------------------------------

/// Y2xx → packed `u16` RGB / RGBA at native BITS depth
/// (low-bit-packed: 10 / 12 active bits in the low N of each
/// `u16`, upper `(16 - BITS)` bits zero — matches `yuv420p10le`'s
/// fidelity-preserving u16 path).
///
/// `ALPHA = true` writes a 4-element-per-pixel output with α =
/// `(1 << BITS) - 1` (opaque maximum at the native depth).
/// `BE = true` selects big-endian wire decoding.
///
/// # Panics (debug builds)
/// - `width` must be even.
/// - `packed.len() >= width * 2`.
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y2xx_n_to_rgb_u16_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_rgb_u16 requires BITS in {{10, 12}}"
    );
  }
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2), "Y2xx requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let alpha_max: u16 = out_max as u16;

  let pairs = width / 2;
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    let y0 = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4)) }) as i32;
    let u = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 2)) }) as i32;
    let y1 = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) }) as i32;
    let v = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 6)) }) as i32;

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    for (k, &y) in [y0, y1].iter().enumerate() {
      let y_s = q15_scale(y - y_off, y_scale);
      let off = (p * 2 + k) * bpp;
      out[off] = (y_s + r_chroma).clamp(0, out_max) as u16;
      out[off + 1] = (y_s + g_chroma).clamp(0, out_max) as u16;
      out[off + 2] = (y_s + b_chroma).clamp(0, out_max) as u16;
      if ALPHA {
        out[off + 3] = alpha_max;
      }
    }
  }
}

// ---- Luma extraction ---------------------------------------------------

/// Y2xx → 8-bit luma. Y values are downshifted from BITS to 8 via
/// `>> (BITS - 8)`. Bypasses the YUV → RGB pipeline entirely.
/// `BE = true` selects big-endian wire decoding.
///
/// # Panics (debug builds)
/// - `width` must be even.
/// - `packed.len() >= width * 2`.
/// - `luma_out.len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y2xx_n_to_luma_row<const BITS: u32, const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_luma requires BITS in {{10, 12}}"
    );
  }
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2), "Y2xx requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let pairs = width / 2;
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    let y0 = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4)) });
    let y1 = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) });
    luma_out[p * 2] = (y0 >> (BITS - 8)) as u8;
    luma_out[p * 2 + 1] = (y1 >> (BITS - 8)) as u8;
  }
}

/// Y2xx → native-depth `u16` luma (low-bit-packed). Each output
/// `u16` carries the source's BITS-bit Y value in its low BITS bits
/// (upper `(16 - BITS)` bits zero). `BE = true` selects big-endian
/// wire decoding.
///
/// # Panics (debug builds)
/// - `width` must be even.
/// - `packed.len() >= width * 2`.
/// - `luma_out.len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y2xx_n_to_luma_u16_row<const BITS: u32, const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_luma_u16 requires BITS in {{10, 12}}"
    );
  }
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2), "Y2xx requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let pairs = width / 2;
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    luma_out[p * 2] = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4)) });
    luma_out[p * 2 + 1] = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) });
  }
}

// ---- Y2xx (BITS ∈ {10, 12}) → HSV (direct: no RGB scratch) ------------
//
// The display-referred twin of [`y2xx_n_to_rgb_or_rgba_row`], fused with
// the OpenCV HSV quantizer. It shares the EXACT per-pixel **8-bit-output**
// Q15 decode (`Coefficients::for_matrix` + `range_params_n::<BITS, 8>` +
// the `>> (16 - BITS)` MSB-aligned de-pack + the packed 4:2:2 `Y₀, U, Y₁,
// V` extraction) as its `_to_rgb` sibling, then feeds the decoded
// `(r, g, b)` straight into [`rgb_to_hsv_pixel`] and scatters to the
// H/S/V planes — never materializing a packed-RGB row. The HSV output is
// 8-bit (`H ∈ [0, 179]`, `S, V ∈ [0, 255]`) regardless of source depth,
// because the existing Y2xx HSV path is `rgb_to_hsv_row` applied to the
// **8-bit** `y2xx_to_rgb` output. This kernel is therefore byte-identical
// to `rgb_to_hsv_row(y2xx_n_to_rgb_or_rgba_row::<BITS, false, BE>(...))`
// but allocates no RGB intermediate. Y216 (BITS = 16) has its own
// [`y216_to_hsv_row`] (parallel to its i32 RGB-u8 path). The SIMD
// backends mirror this via a small reused 8-bit-RGB chunk filled by the
// existing SIMD `y2xx_n_to_rgb_or_rgba_row` plus the SIMD
// `rgb_to_hsv_row`.

/// Y2xx (`BITS ∈ {10, 12}`) → planar HSV bytes (OpenCV
/// `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`).
/// Const-generic over `BITS ∈ {10, 12}` and `BE` (source byte order),
/// exactly like [`y2xx_n_to_rgb_or_rgba_row`]. Chroma is half-width
/// packed `U, V`, nearest-neighbor 1→2 upsampled per pixel pair.
///
/// Byte-identical to
/// `rgb_to_hsv_row(y2xx_n_to_rgb_or_rgba_row::<BITS, false, BE>(...))`.
///
/// # Panics
///
/// - `width` must be even.
/// - `packed.len() >= width * 2` (one u16 quadruple per chroma pair).
/// - Each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn y2xx_n_to_hsv_row<const BITS: u32, const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_hsv requires BITS in {{10, 12}}"
    );
  }
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2), "Y2xx requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  let pairs = width / 2;
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    let y0 = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4)) }) as i32;
    let u = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 2)) }) as i32;
    let y1 = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) }) as i32;
    let v = rshift_bits::<BITS>(unsafe { load_endian_u16::<BE>(base.add(off4 + 6)) }) as i32;

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0_s = q15_scale(y0 - y_off, y_scale);
    let (h0, s0, v0) = rgb_to_hsv_pixel(
      clamp_u8(y0_s + r_chroma) as i32,
      clamp_u8(y0_s + g_chroma) as i32,
      clamp_u8(y0_s + b_chroma) as i32,
    );
    h_out[p * 2] = h0;
    s_out[p * 2] = s0;
    v_out[p * 2] = v0;

    let y1_s = q15_scale(y1 - y_off, y_scale);
    let (h1, s1, v1) = rgb_to_hsv_pixel(
      clamp_u8(y1_s + r_chroma) as i32,
      clamp_u8(y1_s + g_chroma) as i32,
      clamp_u8(y1_s + b_chroma) as i32,
    );
    h_out[p * 2 + 1] = h1;
    s_out[p * 2 + 1] = s1;
    v_out[p * 2 + 1] = v1;
  }
}

// ---- Public Y210 (BITS=10) wrappers ------------------------------------
//
// Ship 11b instantiates BITS=10 only. Ship 11c will add the parallel
// BITS=12 wrappers (`y212_to_*_row`) without further kernel changes.

/// Public Y210 (BITS=10) → packed RGB / RGBA u8 wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y210_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  y2xx_n_to_rgb_or_rgba_row::<10, ALPHA, BE>(packed, out, width, matrix, full_range);
}

/// Public Y210 → packed `u16` RGB / RGBA wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y210_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  y2xx_n_to_rgb_u16_or_rgba_u16_row::<10, ALPHA, BE>(packed, out, width, matrix, full_range);
}

/// Public Y210 → 8-bit luma wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y210_to_luma_row<const BE: bool>(packed: &[u16], luma_out: &mut [u8], width: usize) {
  y2xx_n_to_luma_row::<10, BE>(packed, luma_out, width);
}

/// Public Y210 → native-depth `u16` luma wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y210_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  y2xx_n_to_luma_u16_row::<10, BE>(packed, luma_out, width);
}

/// Public Y210 → planar HSV wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn y210_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  y2xx_n_to_hsv_row::<10, BE>(packed, h_out, s_out, v_out, width, matrix, full_range);
}

// ---- Public Y212 (BITS=12) wrappers ------------------------------------
//
// Ship 11c monomorphizes the `y2xx_n_*` template at BITS=12. No new
// SIMD code — the per-arch backends already accept BITS ∈ {10, 12}.

/// Public Y212 (BITS=12) → packed RGB / RGBA u8 wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y212_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  y2xx_n_to_rgb_or_rgba_row::<12, ALPHA, BE>(packed, out, width, matrix, full_range);
}

/// Public Y212 → packed `u16` RGB / RGBA wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y212_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, ALPHA, BE>(packed, out, width, matrix, full_range);
}

/// Public Y212 → 8-bit luma wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y212_to_luma_row<const BE: bool>(packed: &[u16], luma_out: &mut [u8], width: usize) {
  y2xx_n_to_luma_row::<12, BE>(packed, luma_out, width);
}

/// Public Y212 → native-depth `u16` luma wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y212_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  y2xx_n_to_luma_u16_row::<12, BE>(packed, luma_out, width);
}

/// Public Y212 → planar HSV wrapper.
/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn y212_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  y2xx_n_to_hsv_row::<12, BE>(packed, h_out, s_out, v_out, width, matrix, full_range);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::KernelMatrix;

  /// Build one Y210-shaped u16 quadruple `[Y0, U, Y1, V]` with each
  /// sample shifted to MSB-aligned 10-bit form (low 6 bits zero).
  fn y210_quad(y0: u16, u: u16, y1: u16, v: u16) -> [u16; 4] {
    [
      (y0 & 0x3FF) << 6,
      (u & 0x3FF) << 6,
      (y1 & 0x3FF) << 6,
      (v & 0x3FF) << 6,
    ]
  }

  /// Re-encode a host-native u16 slice as LE-encoded bytes packed back as
  /// `Vec<u16>`. On LE host this is a no-op; on BE host every u16 is byte-
  /// swapped relative to the intended logical value. Kernels called with
  /// `BE = false` recover the intended values via `from_le` on both hosts.
  fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
    host
      .iter()
      .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
      .collect()
  }

  /// Build a `Vec<u16>` Y210 row of `width` pixels with `(Y, U, V)`
  /// repeated, in LE-encoded byte form. Width must be even.
  fn solid_y210(width: usize, y: u16, u: u16, v: u16) -> std::vec::Vec<u16> {
    let mut buf = std::vec::Vec::with_capacity(width * 2);
    for _ in 0..(width / 2) {
      buf.extend_from_slice(&y210_quad(y, u, y, v));
    }
    as_le_u16(&buf)
  }

  /// Re-encode a host-native u16 slice as BE-encoded byte storage. Mirror of
  /// `as_le_u16` for kernels invoked with `BE = true`. Combined with
  /// `as_le_u16`, lets a single host-native `intended` fixture drive both
  /// `<_, false>` and `<_, true>` kernel paths so they decode the same
  /// logical values on every host.
  fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
    host
      .iter()
      .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
      .collect()
  }

  /// Build the host-native intended Y210 row corresponding to `solid_y210`.
  /// Width must be even.
  fn solid_y210_intended(width: usize, y: u16, u: u16, v: u16) -> std::vec::Vec<u16> {
    let mut buf = std::vec::Vec::with_capacity(width * 2);
    for _ in 0..(width / 2) {
      buf.extend_from_slice(&y210_quad(y, u, y, v));
    }
    buf
  }

  // -- Scalar references for the BE-parity tests --
  //
  // Walk host-native `intended` Y2xx buffers (laid out as YUYV `u16`
  // quadruples, BITS-aligned with `(16 - BITS)` low bits zero) and reproduce
  // each kernel's documented behaviour without going through any byte-order
  // conversion. Pinning the LE / BE outputs against these absolute references
  // prevents the parity assertion from passing in lock-step on two equally
  // corrupt decode paths.

  fn ref_y2xx_to_rgb_u8<const BITS: u32>(
    intended: &[u16],
    width: usize,
    matrix: KernelMatrix,
    full_range: bool,
  ) -> std::vec::Vec<u8> {
    let coeffs = Coefficients::for_matrix(matrix);
    let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
    let bias = chroma_bias::<BITS>();
    let pairs = width / 2;
    let mut out = std::vec![0u8; width * 3];
    for p in 0..pairs {
      let off4 = p * 4;
      let y0 = (intended[off4] >> (16 - BITS)) as i32;
      let u = (intended[off4 + 1] >> (16 - BITS)) as i32;
      let y1 = (intended[off4 + 2] >> (16 - BITS)) as i32;
      let v = (intended[off4 + 3] >> (16 - BITS)) as i32;
      let u_d = q15_scale(u - bias, c_scale);
      let v_d = q15_scale(v - bias, c_scale);
      let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
      let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
      let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);
      for (k, &y) in [y0, y1].iter().enumerate() {
        let y_s = q15_scale(y - y_off, y_scale);
        let off = (p * 2 + k) * 3;
        out[off] = clamp_u8(y_s + r_chroma);
        out[off + 1] = clamp_u8(y_s + g_chroma);
        out[off + 2] = clamp_u8(y_s + b_chroma);
      }
    }
    out
  }

  fn ref_y2xx_to_rgb_u16<const BITS: u32>(
    intended: &[u16],
    width: usize,
    matrix: KernelMatrix,
    full_range: bool,
  ) -> std::vec::Vec<u16> {
    let coeffs = Coefficients::for_matrix(matrix);
    let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
    let bias = chroma_bias::<BITS>();
    let out_max: i32 = (1i32 << BITS) - 1;
    let pairs = width / 2;
    let mut out = std::vec![0u16; width * 3];
    for p in 0..pairs {
      let off4 = p * 4;
      let y0 = (intended[off4] >> (16 - BITS)) as i32;
      let u = (intended[off4 + 1] >> (16 - BITS)) as i32;
      let y1 = (intended[off4 + 2] >> (16 - BITS)) as i32;
      let v = (intended[off4 + 3] >> (16 - BITS)) as i32;
      let u_d = q15_scale(u - bias, c_scale);
      let v_d = q15_scale(v - bias, c_scale);
      let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
      let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
      let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);
      for (k, &y) in [y0, y1].iter().enumerate() {
        let y_s = q15_scale(y - y_off, y_scale);
        let off = (p * 2 + k) * 3;
        out[off] = (y_s + r_chroma).clamp(0, out_max) as u16;
        out[off + 1] = (y_s + g_chroma).clamp(0, out_max) as u16;
        out[off + 2] = (y_s + b_chroma).clamp(0, out_max) as u16;
      }
    }
    out
  }

  fn ref_y2xx_to_luma<const BITS: u32>(intended: &[u16], width: usize) -> std::vec::Vec<u8> {
    let pairs = width / 2;
    let mut out = std::vec![0u8; width];
    for p in 0..pairs {
      let off4 = p * 4;
      let y0 = intended[off4] >> (16 - BITS);
      let y1 = intended[off4 + 2] >> (16 - BITS);
      out[p * 2] = (y0 >> (BITS - 8)) as u8;
      out[p * 2 + 1] = (y1 >> (BITS - 8)) as u8;
    }
    out
  }

  fn ref_y2xx_to_luma_u16<const BITS: u32>(intended: &[u16], width: usize) -> std::vec::Vec<u16> {
    let pairs = width / 2;
    let mut out = std::vec![0u16; width];
    for p in 0..pairs {
      let off4 = p * 4;
      out[p * 2] = intended[off4] >> (16 - BITS);
      out[p * 2 + 1] = intended[off4 + 2] >> (16 - BITS);
    }
    out
  }

  #[test]
  fn scalar_y210_to_rgb_gray_is_gray() {
    // Full-range gray: Y=512, U=V=512 (10-bit center) → RGB ~128.
    let buf = solid_y210(8, 512, 512, 512);
    let mut rgb = [0u8; 8 * 3];
    y210_to_rgb_or_rgba_row::<false, false>(&buf, &mut rgb, 8, KernelMatrix::Bt709, true);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  fn scalar_y210_to_rgba_alpha_is_opaque() {
    let buf = solid_y210(8, 512, 512, 512);
    let mut rgba = [0u8; 8 * 4];
    y210_to_rgb_or_rgba_row::<true, false>(&buf, &mut rgba, 8, KernelMatrix::Bt709, true);
    for px in rgba.chunks(4) {
      assert_eq!(px[3], 0xFF);
    }
  }

  #[test]
  fn scalar_y210_to_rgb_u16_native_depth() {
    // Full-range gray Y=512 → ~512 in 10-bit RGB out (out_max = 1023).
    let buf = solid_y210(8, 512, 512, 512);
    let mut rgb = [0u16; 8 * 3];
    y210_to_rgb_u16_or_rgba_u16_row::<false, false>(&buf, &mut rgb, 8, KernelMatrix::Bt709, true);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(512) <= 2, "px expected ~512, got {}", px[0]);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  fn scalar_y210_to_rgba_u16_alpha_is_max() {
    let buf = solid_y210(8, 512, 512, 512);
    let mut rgba = [0u16; 8 * 4];
    y210_to_rgb_u16_or_rgba_u16_row::<true, false>(&buf, &mut rgba, 8, KernelMatrix::Bt709, true);
    for px in rgba.chunks(4) {
      assert_eq!(px[3], 1023, "alpha must be (1 << 10) - 1");
    }
  }

  #[test]
  fn scalar_y210_to_luma_extracts_y_bytes_downshifted() {
    // Build a width=6 row with Y values 100, 200, 300, 400, 500, 600
    // (10-bit). u16 length = width * 2 = 12.
    let mut intended = std::vec![0u16; 12];
    let ys = [100u16, 200, 300, 400, 500, 600];
    for i in 0..3 {
      intended[i * 4] = ys[i * 2] << 6;
      intended[i * 4 + 1] = 128u16 << 6; // U
      intended[i * 4 + 2] = ys[i * 2 + 1] << 6;
      intended[i * 4 + 3] = 128u16 << 6; // V
    }
    let buf = as_le_u16(&intended);
    let mut luma = [0u8; 6];
    y210_to_luma_row::<false>(&buf, &mut luma, 6);
    assert_eq!(luma[0], (100u16 >> 2) as u8);
    assert_eq!(luma[1], (200u16 >> 2) as u8);
    assert_eq!(luma[2], (300u16 >> 2) as u8);
    assert_eq!(luma[3], (400u16 >> 2) as u8);
    assert_eq!(luma[4], (500u16 >> 2) as u8);
    assert_eq!(luma[5], (600u16 >> 2) as u8);
  }

  #[test]
  fn scalar_y210_to_luma_u16_extracts_y_low_bit_packed() {
    let mut intended = std::vec![0u16; 12];
    let ys = [100u16, 200, 300, 400, 500, 600];
    for i in 0..3 {
      intended[i * 4] = ys[i * 2] << 6;
      intended[i * 4 + 1] = 128u16 << 6;
      intended[i * 4 + 2] = ys[i * 2 + 1] << 6;
      intended[i * 4 + 3] = 128u16 << 6;
    }
    let buf = as_le_u16(&intended);
    let mut luma = [0u16; 6];
    y210_to_luma_u16_row::<false>(&buf, &mut luma, 6);
    assert_eq!(luma[0], 100);
    assert_eq!(luma[1], 200);
    assert_eq!(luma[2], 300);
    assert_eq!(luma[3], 400);
    assert_eq!(luma[4], 500);
    assert_eq!(luma[5], 600);
  }

  // ---- BE=true parity tests -------------------------------------------
  //
  // Pattern: build a single host-native `intended` Y210 row, materialise it as
  // LE-encoded bytes via `as_le_u16` and BE-encoded bytes via `as_be_u16`,
  // run both `<_, false>` and `<_, true>` kernels, and pin each output against
  // an absolute scalar reference so the parity assertion cannot pass on two
  // equally corrupt decodes.

  #[test]
  fn scalar_y210_be_rgb_matches_le() {
    let intended = solid_y210_intended(8, 512, 512, 512);
    let le = as_le_u16(&intended);
    let be = as_be_u16(&intended);
    let mut rgb_le = [0u8; 8 * 3];
    let mut rgb_be = [0u8; 8 * 3];
    y210_to_rgb_or_rgba_row::<false, false>(&le, &mut rgb_le, 8, KernelMatrix::Bt709, true);
    y210_to_rgb_or_rgba_row::<false, true>(&be, &mut rgb_be, 8, KernelMatrix::Bt709, true);
    let expected = ref_y2xx_to_rgb_u8::<10>(&intended, 8, KernelMatrix::Bt709, true);
    assert_eq!(
      rgb_le.as_slice(),
      expected,
      "LE path must match scalar reference"
    );
    assert_eq!(
      rgb_be.as_slice(),
      expected,
      "BE path must match scalar reference"
    );
    assert_eq!(rgb_le, rgb_be, "BE and LE outputs must agree");
  }

  #[test]
  fn scalar_y210_be_rgb_u16_matches_le() {
    let intended = solid_y210_intended(8, 512, 512, 512);
    let le = as_le_u16(&intended);
    let be = as_be_u16(&intended);
    let mut out_le = [0u16; 8 * 3];
    let mut out_be = [0u16; 8 * 3];
    y210_to_rgb_u16_or_rgba_u16_row::<false, false>(&le, &mut out_le, 8, KernelMatrix::Bt709, true);
    y210_to_rgb_u16_or_rgba_u16_row::<false, true>(&be, &mut out_be, 8, KernelMatrix::Bt709, true);
    let expected = ref_y2xx_to_rgb_u16::<10>(&intended, 8, KernelMatrix::Bt709, true);
    assert_eq!(
      out_le.as_slice(),
      expected,
      "LE path must match scalar reference"
    );
    assert_eq!(
      out_be.as_slice(),
      expected,
      "BE path must match scalar reference"
    );
    assert_eq!(out_le, out_be, "BE and LE u16 outputs must agree");
  }

  #[test]
  fn scalar_y210_be_luma_matches_le() {
    let intended = solid_y210_intended(8, 512, 512, 512);
    let le = as_le_u16(&intended);
    let be = as_be_u16(&intended);
    let mut luma_le = [0u8; 8];
    let mut luma_be = [0u8; 8];
    y210_to_luma_row::<false>(&le, &mut luma_le, 8);
    y210_to_luma_row::<true>(&be, &mut luma_be, 8);
    let expected = ref_y2xx_to_luma::<10>(&intended, 8);
    assert_eq!(
      luma_le.as_slice(),
      expected,
      "LE path must match scalar reference"
    );
    assert_eq!(
      luma_be.as_slice(),
      expected,
      "BE path must match scalar reference"
    );
    assert_eq!(luma_le, luma_be, "BE and LE luma outputs must agree");
  }

  #[test]
  fn scalar_y210_be_luma_u16_matches_le() {
    let intended = solid_y210_intended(8, 512, 512, 512);
    let le = as_le_u16(&intended);
    let be = as_be_u16(&intended);
    let mut luma_le = [0u16; 8];
    let mut luma_be = [0u16; 8];
    y210_to_luma_u16_row::<false>(&le, &mut luma_le, 8);
    y210_to_luma_u16_row::<true>(&be, &mut luma_be, 8);
    let expected = ref_y2xx_to_luma_u16::<10>(&intended, 8);
    assert_eq!(
      luma_le.as_slice(),
      expected,
      "LE path must match scalar reference"
    );
    assert_eq!(
      luma_be.as_slice(),
      expected,
      "BE path must match scalar reference"
    );
    assert_eq!(luma_le, luma_be, "BE and LE luma_u16 outputs must agree");
  }
}
