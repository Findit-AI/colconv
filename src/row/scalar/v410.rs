//! Scalar reference kernels for the V410 packed YUV 4:4:4 10-bit
//! family (FFmpeg `AV_PIX_FMT_V410`). One pixel per 32-bit word;
//! 10-bit U / Y / V channels with 2-bit padding (see
//! [`crate::frame::V410Frame`]). 4:4:4 means no chroma deinterleave
//! step — each word yields a complete `(Y, U, V)` triple.
//!
//! `<const BE: bool>` — when `true`, each `u32` element of the input
//! slice is byte-swapped before field extraction. This handles the
//! `V410BE` big-endian wire format; `BE = false` is the standard LE path.

use super::*;

/// Extract `(u, y, v)` from one V410 word.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn extract_v410(word: u32) -> (i32, i32, i32) {
  let u = (word & 0x3FF) as i32;
  let y = ((word >> 10) & 0x3FF) as i32;
  let v = ((word >> 20) & 0x3FF) as i32;
  (u, y, v)
}

// ---- u8 RGB / RGBA output ----------------------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v410_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u32],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<10, 8>(full_range);
  let bias = chroma_bias::<10>();

  for (x, &raw) in packed[..width].iter().enumerate() {
    let word = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    let (u, y, v) = extract_v410(word);
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

// ---- V410 → HSV (direct: no RGB scratch) -------------------------------

/// Scalar V410 → planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV` encoding:
/// `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). Const-generic over `BE` (source
/// byte order), exactly like [`v410_to_rgb_or_rgba_row`]. 4:4:4 (no chroma
/// subsampling): one U/Y/V triple per word. Shares that kernel's EXACT
/// **8-bit-output** Q15 decode (`range_params_n::<10, 8>` + `extract_v410`),
/// then feeds the clamped `(r, g, b)` straight into [`rgb_to_hsv_pixel`] —
/// byte-identical to `rgb_to_hsv_row(v410_to_rgb_or_rgba_row::<false,
/// BE>(...))`, with no RGB allocation.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width`.
/// - each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn v410_to_hsv_row<const BE: bool>(
  packed: &[u32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<10, 8>(full_range);
  let bias = chroma_bias::<10>();

  for (x, &raw) in packed[..width].iter().enumerate() {
    let word = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    let (u, y, v) = extract_v410(word);
    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let (hh, ss, vv) = rgb_to_hsv_pixel(
      clamp_u8(y_s + r_chroma) as i32,
      clamp_u8(y_s + g_chroma) as i32,
      clamp_u8(y_s + b_chroma) as i32,
    );
    h_out[x] = hh;
    s_out[x] = ss;
    v_out[x] = vv;
  }
}

// ---- u16 RGB / RGBA native-depth output --------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v410_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u32],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<10, 10>(full_range);
  let bias = chroma_bias::<10>();
  let alpha_max: u16 = 0x3FF;
  let out_max: i32 = 0x3FF;

  for (x, &raw) in packed[..width].iter().enumerate() {
    let word = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    let (u, y, v) = extract_v410(word);
    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let off = x * bpp;
    out[off] = (y_s + r_chroma).clamp(0, out_max) as u16;
    out[off + 1] = (y_s + g_chroma).clamp(0, out_max) as u16;
    out[off + 2] = (y_s + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[off + 3] = alpha_max;
    }
  }
}

// ---- Luma (u8) — `>> 2` ------------------------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v410_to_luma_row<const BE: bool>(packed: &[u32], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let word = if BE {
      u32::from_be(packed[x])
    } else {
      u32::from_le(packed[x])
    };
    let y = (word >> 10) & 0x3FF;
    out[x] = (y >> 2) as u8;
  }
}

// ---- Luma (u16, low-bit-packed at 10-bit) ------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v410_to_luma_u16_row<const BE: bool>(packed: &[u32], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let word = if BE {
      u32::from_be(packed[x])
    } else {
      u32::from_le(packed[x])
    };
    let y = (word >> 10) & 0x3FF;
    out[x] = y as u16;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::KernelMatrix;

  /// Pack one V410 word (host-native u32) from explicit U / Y / V samples.
  fn pack_v410(u: u32, y: u32, v: u32) -> u32 {
    debug_assert!(u < 1024 && y < 1024 && v < 1024);
    (v << 20) | (y << 10) | u
  }

  /// Re-encode a slice of host-native u32 values as LE-encoded byte storage,
  /// packed back into `Vec<u32>`. On LE host this is a no-op; on BE host
  /// every u32 is byte-swapped relative to its host-native representation.
  /// Kernels called with `BE = false` recover the intended logical values
  /// via `u32::from_le` on both hosts.
  fn as_le_u32(host: &[u32]) -> std::vec::Vec<u32> {
    host
      .iter()
      .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
      .collect()
  }

  #[test]
  fn v410_known_pattern_rgb() {
    // Limited-range BT.709, gray Y=64 (≈ 0 in [0, 255]) with neutral
    // chroma U=V=512. Both pixels should produce ~black [0, 0, 0]
    // before saturation.
    let p = as_le_u32(&[
      pack_v410(512, 64, 512),
      pack_v410(512, 64, 512),
      pack_v410(512, 940, 512), // Y=940 ≈ 255 (limited-range white)
      pack_v410(512, 940, 512),
    ]);
    let mut out = vec![0u8; 4 * 3];
    v410_to_rgb_or_rgba_row::<false, false>(&p, &mut out, 4, KernelMatrix::Bt709, false);
    // Two black pixels followed by two white pixels.
    assert_eq!(&out[0..3], &[0u8, 0, 0]);
    assert_eq!(&out[3..6], &[0u8, 0, 0]);
    assert_eq!(&out[6..9], &[255u8, 255, 255]);
    assert_eq!(&out[9..12], &[255u8, 255, 255]);
  }

  #[test]
  fn v410_known_pattern_rgba_alpha_max() {
    let p = as_le_u32(&[pack_v410(512, 940, 512)]);
    let mut out = vec![0u8; 4];
    v410_to_rgb_or_rgba_row::<true, false>(&p, &mut out, 1, KernelMatrix::Bt709, false);
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn v410_luma_extract_u8() {
    let p = as_le_u32(&[
      pack_v410(0, 0x3FF, 0), // Y = 0x3FF (10-bit max)
      pack_v410(0, 0x100, 0), // Y = 0x100
    ]);
    let mut out = vec![0u8; 2];
    v410_to_luma_row::<false>(&p, &mut out, 2);
    // 0x3FF >> 2 = 0xFF; 0x100 >> 2 = 0x40.
    assert_eq!(&out[..], &[0xFFu8, 0x40]);
  }

  #[test]
  fn v410_luma_extract_u16_low_bit_packed() {
    let p = as_le_u32(&[pack_v410(0, 0x3FF, 0), pack_v410(0, 0x123, 0)]);
    let mut out = vec![0u16; 2];
    v410_to_luma_u16_row::<false>(&p, &mut out, 2);
    assert_eq!(&out[..], &[0x3FFu16, 0x123]);
  }

  #[test]
  fn v410_known_pattern_rgba_u16_alpha_max() {
    let p = as_le_u32(&[pack_v410(512, 940, 512)]);
    let mut out = vec![0u16; 4];
    v410_to_rgb_u16_or_rgba_u16_row::<true, false>(&p, &mut out, 1, KernelMatrix::Bt709, false);
    // 10-bit alpha max is 0x3FF (low-bit-packed).
    assert_eq!(out[3], 0x3FF);
  }

  #[test]
  fn v410_be_roundtrip_matches_byte_swapped_le() {
    // Construct LE/BE buffers from raw bytes via `to_le_bytes` / `to_be_bytes`
    // so semantics are host-independent: on every host, `le` carries the
    // intended value as LE-encoded bytes and `be` carries the same value as
    // BE-encoded bytes. Both kernels should therefore decode to the same
    // intended host-native value (and produce identical RGB output) on both
    // LE and BE hosts. The earlier `swap_bytes` pattern only validated this
    // on LE hosts and degenerated to equal-but-wrong on BE hosts.
    let intended = pack_v410(200, 500, 800);
    let le_bytes: std::vec::Vec<u8> = intended.to_le_bytes().to_vec();
    let be_bytes: std::vec::Vec<u8> = intended.to_be_bytes().to_vec();
    let le_buf: std::vec::Vec<u32> = le_bytes
      .chunks_exact(4)
      .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
      .collect();
    let be_buf: std::vec::Vec<u32> = be_bytes
      .chunks_exact(4)
      .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
      .collect();
    let mut out_le = vec![0u8; 3];
    let mut out_be = vec![0u8; 3];
    v410_to_rgb_or_rgba_row::<false, false>(&le_buf, &mut out_le, 1, KernelMatrix::Bt709, false);
    v410_to_rgb_or_rgba_row::<false, true>(&be_buf, &mut out_be, 1, KernelMatrix::Bt709, false);
    assert_eq!(out_le, out_be, "V410 BE scalar must match byte-swapped LE");
  }
}
