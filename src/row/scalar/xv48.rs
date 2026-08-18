//! Scalar reference kernels for the XV48 packed YUV 4:4:4 16-bit
//! family (FFmpeg `AV_PIX_FMT_XV48LE`). Each pixel is a u16 quadruple
//! `[U, Y, V, X]` with every channel using the full 16 bits (no MSB
//! shift — the full-depth sibling of XV36, which is 12-bit MSB-aligned).
//! The `X` slot is padding — read but discarded; RGBA outputs force
//! α = max.
//!
//! 4:4:4 means no chroma deinterleave step — each pixel's U / Y / V
//! are independent. No bit extraction: the sample is the full u16, fed
//! to the standard Q15 chroma + Y pipeline at BITS=16. u8 output uses
//! i32 chroma (output-range scaling keeps within i32); u16 output uses
//! **i64 chroma** via `q15_chroma64` (Q15 sums overflow i32 at
//! BITS=16/16, peak ~3.7e9 for BT.2020) — exactly like the AYUV64
//! 16-bit sibling.
//!
//! `<const BE: bool>` — when `true`, each `u16` element of the input
//! slice is byte-swapped before use. This handles the `XV48BE`
//! big-endian wire format; `BE = false` is the standard LE path.

use super::*;

/// Extract `(u, y, v)` from one XV48 pixel. The `x` slot at index 3
/// is padding and is not returned. No shift — every channel uses the
/// full 16 bits (`[0, 65535]`).
///
/// Samples are passed already endian-corrected by the caller.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn extract_xv48(quad: &[u16]) -> (i32, i32, i32) {
  let u = quad[0] as i32;
  let y = quad[1] as i32;
  let v = quad[2] as i32;
  // quad[3] is padding X — ignored
  (u, y, v)
}

/// Load one XV48 u16 sample, applying a byte-swap for BE wire format
/// when `BE = true`. Uses target-endian aware `u16::from_be`/`u16::from_le`
/// — these are no-ops when the source byte order matches the host, so the
/// helper produces correct samples on both LE and BE hosts (e.g. s390x).
#[cfg_attr(not(tarpaulin), inline(always))]
fn load_xv48_u16<const BE: bool>(v: u16) -> u16 {
  if BE { u16::from_be(v) } else { u16::from_le(v) }
}

// ---- u8 RGB / RGBA output (i32 chroma) ---------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv48_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let base = x * 4;
    let quad = [
      load_xv48_u16::<BE>(packed[base]),
      load_xv48_u16::<BE>(packed[base + 1]),
      load_xv48_u16::<BE>(packed[base + 2]),
      load_xv48_u16::<BE>(packed[base + 3]),
    ];
    let (u, y, v) = extract_xv48(&quad);
    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let off = x * bpp;
    out[off] = clamp_u8(y_s + r_chroma);
    out[off + 1] = clamp_u8(y_s + g_chroma);
    out[off + 2] = clamp_u8(y_s + b_chroma);
    if ALPHA {
      out[off + 3] = 0xFF;
    }
  }
}

// ---- XV48 → HSV (direct: no RGB scratch) -------------------------------
//
// The display-referred twin of [`xv48_to_rgb_or_rgba_row`] (`ALPHA =
// false`), fused with the OpenCV HSV quantizer. It shares the EXACT
// per-pixel **8-bit-output** Q15 decode (`Coefficients::for_matrix` +
// `range_params_n::<16, 8>` + the full-16-bit U/Y/V extraction) as its
// `_to_rgb` sibling, then feeds the decoded `(r, g, b)` straight into
// [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes — never
// materializing a packed-RGB row. The HSV output is 8-bit
// (`H ∈ [0, 179]`, `S, V ∈ [0, 255]`); XV48 is a 16-bit source but its
// existing HSV path is `rgb_to_hsv_row` applied to the **8-bit**
// `xv48_to_rgb` output, so the 8-bit intermediate is reproduced here.
// The X slot (index 3) is padding — read by `extract_xv48` but discarded
// — and HSV derives from the COLOR (U/Y/V → RGB → HSV) only. Byte-
// identical to `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))`,
// with no RGB allocation.

/// Scalar XV48 → planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV` encoding:
/// `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). Const-generic over `BE` (source
/// byte order), exactly like [`xv48_to_rgb_or_rgba_row`]. 4:4:4 (no
/// chroma subsampling): one U/Y/V triple per pixel. The padding X slot
/// is dropped (HSV is colour-only). Byte-identical to
/// `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))`.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 4`.
/// - each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn xv48_to_hsv_row<const BE: bool>(
  packed: &[u16],
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
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let base = x * 4;
    let quad = [
      load_xv48_u16::<BE>(packed[base]),
      load_xv48_u16::<BE>(packed[base + 1]),
      load_xv48_u16::<BE>(packed[base + 2]),
      load_xv48_u16::<BE>(packed[base + 3]),
    ];
    // The X slot (index 3) is padding — dropped by `extract_xv48`; HSV
    // is colour-only.
    let (u, y, v) = extract_xv48(&quad);
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
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = vv;
  }
}

// ---- u16 RGB / RGBA native-depth output (i64 chroma) -------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv48_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let alpha_max: u16 = 0xFFFF;

  for x in 0..width {
    let base = x * 4;
    let quad = [
      load_xv48_u16::<BE>(packed[base]),
      load_xv48_u16::<BE>(packed[base + 1]),
      load_xv48_u16::<BE>(packed[base + 2]),
      load_xv48_u16::<BE>(packed[base + 3]),
    ];
    let (u, y, v) = extract_xv48(&quad);
    // q15_scale returns i32; q15_chroma64 handles the i32→i64 promotion
    // internally — pass i32 values directly (same API as q15_chroma).
    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    // Use q15_scale64 for luma: at BITS=16/16 limited range, the product
    // (y - y_off) * y_scale can just exceed i32::MAX for out-of-range inputs.
    let y_s = q15_scale64(y - y_off, y_scale);
    let off = x * bpp;
    out[off] = (y_s + r_chroma).clamp(0, 0xFFFF) as u16;
    out[off + 1] = (y_s + g_chroma).clamp(0, 0xFFFF) as u16;
    out[off + 2] = (y_s + b_chroma).clamp(0, 0xFFFF) as u16;
    if ALPHA {
      out[off + 3] = alpha_max;
    }
  }
}

// ---- Luma (u8) — `>> 8` (16-bit → 8-bit high byte) --------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv48_to_luma_row<const BE: bool>(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let y = load_xv48_u16::<BE>(packed[x * 4 + 1]) >> 8;
    out[x] = y as u8;
  }
}

// ---- Luma (u16, full 16-bit native — no shift) ------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv48_to_luma_u16_row<const BE: bool>(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    out[x] = load_xv48_u16::<BE>(packed[x * 4 + 1]);
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::KernelMatrix;

  /// Build a 4-u16 XV48 pixel (host-native u16 quadruple) from explicit
  /// U / Y / V / X samples. Each channel uses the full 16 bits — no shift
  /// (unlike XV36, which MSB-aligns 12-bit values).
  fn pack_xv48(u: u16, y: u16, v: u16, x: u16) -> [u16; 4] {
    [u, y, v, x]
  }

  /// Re-encode a slice of host-native u16 values as LE-encoded byte storage,
  /// packed back into `Vec<u16>`. On LE host this is a no-op; on BE host
  /// every u16 is byte-swapped relative to its host-native representation.
  /// Kernels called with `BE = false` recover the intended logical values
  /// via `u16::from_le` on both hosts.
  fn as_le_u16(host: &[u16]) -> Vec<u16> {
    host
      .iter()
      .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
      .collect()
  }

  /// Limited-range BT.709, neutral chroma U=V=32768.
  /// Black:  Y=4096  (limited-range black at 16-bit: 16 * 256 = 4096).
  /// White:  Y=60160 (limited-range white at 16-bit: 235 * 256 = 60160).
  #[test]
  fn xv48_known_pattern_rgb_limited_range() {
    let p_black = pack_xv48(32768, 4096, 32768, 0);
    let p_white = pack_xv48(32768, 60160, 32768, 0);
    let intended: Vec<u16> = [p_black, p_black, p_white, p_white]
      .iter()
      .flatten()
      .copied()
      .collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u8; 4 * 3];
    xv48_to_rgb_or_rgba_row::<false, false>(&packed, &mut out, 4, KernelMatrix::Bt709, false);
    assert_eq!(&out[0..3], &[0u8, 0, 0], "black pixel 0");
    assert_eq!(&out[3..6], &[0u8, 0, 0], "black pixel 1");
    assert_eq!(&out[6..9], &[255u8, 255, 255], "white pixel 2");
    assert_eq!(&out[9..12], &[255u8, 255, 255], "white pixel 3");
  }

  #[test]
  fn xv48_known_pattern_rgba_alpha_max() {
    let p = pack_xv48(32768, 60160, 32768, 0);
    let packed = as_le_u16(&p);
    let mut out = vec![0u8; 4];
    xv48_to_rgb_or_rgba_row::<true, false>(&packed, &mut out, 1, KernelMatrix::Bt709, false);
    // X = padding; RGBA forces α=0xFF regardless of source X sample.
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn xv48_known_pattern_rgba_ignores_source_x_bits() {
    // Source X=0xFFFF (all bits set) — should not leak into RGB or affect α.
    let p = pack_xv48(32768, 60160, 32768, 0xFFFF);
    let packed = as_le_u16(&p);
    let mut out = vec![0u8; 4];
    xv48_to_rgb_or_rgba_row::<true, false>(&packed, &mut out, 1, KernelMatrix::Bt709, false);
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn xv48_luma_extract_u8() {
    // Y = 0xFFFF → 0xFFFF >> 8 = 0xFF (16-bit max); Y = 0x4000 → 0x40.
    let intended: Vec<u16> = [pack_xv48(0, 0xFFFF, 0, 0), pack_xv48(0, 0x4000, 0, 0)]
      .iter()
      .flatten()
      .copied()
      .collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u8; 2];
    xv48_to_luma_row::<false>(&packed, &mut out, 2);
    assert_eq!(&out[..], &[0xFFu8, 0x40]);
  }

  #[test]
  fn xv48_luma_extract_u16_full_depth() {
    // 16-bit native — written direct, no shift.
    let intended: Vec<u16> = [pack_xv48(0, 0xABCD, 0, 0), pack_xv48(0, 0x1234, 0, 0)]
      .iter()
      .flatten()
      .copied()
      .collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u16; 2];
    xv48_to_luma_u16_row::<false>(&packed, &mut out, 2);
    assert_eq!(&out[..], &[0xABCDu16, 0x1234]);
  }

  #[test]
  fn xv48_known_pattern_rgba_u16_alpha_max() {
    let p = pack_xv48(32768, 60160, 32768, 0xFFFF);
    let packed = as_le_u16(&p);
    let mut out = vec![0u16; 4];
    xv48_to_rgb_u16_or_rgba_u16_row::<true, false>(
      &packed,
      &mut out,
      1,
      KernelMatrix::Bt709,
      false,
    );
    // 16-bit alpha max = 0xFFFF; X = padding so source X sample is ignored.
    assert_eq!(out[3], 0xFFFF);
  }

  #[test]
  fn xv48_be_roundtrip_matches_byte_swapped_le() {
    // Construct LE/BE buffers from raw bytes via `to_le_bytes` / `to_be_bytes`
    // so semantics are host-independent: on every host, `le` carries the
    // intended values as LE-encoded bytes and `be` carries the same values as
    // BE-encoded bytes. Both kernels should therefore decode to the same
    // intended host-native values (and produce identical RGB output) on both
    // LE and BE hosts.
    let intended = pack_xv48(40000, 60160, 20000, 0);
    let le_bytes: Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
    let be_bytes: Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
    let le_buf: Vec<u16> = le_bytes
      .chunks_exact(2)
      .map(|b| u16::from_ne_bytes([b[0], b[1]]))
      .collect();
    let be_buf: Vec<u16> = be_bytes
      .chunks_exact(2)
      .map(|b| u16::from_ne_bytes([b[0], b[1]]))
      .collect();
    let mut out_le = vec![0u8; 3];
    let mut out_be = vec![0u8; 3];
    xv48_to_rgb_or_rgba_row::<false, false>(&le_buf, &mut out_le, 1, KernelMatrix::Bt709, false);
    xv48_to_rgb_or_rgba_row::<false, true>(&be_buf, &mut out_be, 1, KernelMatrix::Bt709, false);
    assert_eq!(out_le, out_be, "XV48 BE scalar must match byte-swapped LE");
  }
}
