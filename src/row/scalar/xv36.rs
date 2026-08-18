//! Scalar reference kernels for the XV36 packed YUV 4:4:4 12-bit
//! family (FFmpeg `AV_PIX_FMT_XV36LE`). Each pixel is a u16
//! quadruple `[U, Y, V, A]` MSB-aligned at 12-bit (low 4 bits zero
//! per sample). The A slot is padding — read but discarded; RGBA
//! outputs force α = max.
//!
//! 4:4:4 means no chroma deinterleave step — each pixel's U / Y / V
//! are independent. Bit extraction is `>> 4` to drop the 4 padding
//! LSBs, then the standard Q15 chroma + Y pipeline at BITS=12 (i32
//! chroma — same depth as Y2xx at BITS=12).
//!
//! `<const BE: bool>` — when `true`, each `u16` element of the input
//! slice is byte-swapped before use. This handles the `XV36BE`
//! big-endian wire format; `BE = false` is the standard LE path.

use super::*;

/// Extract `(u, y, v)` from one XV36 pixel. The `a` slot at index 3
/// is padding and is not returned. Each channel is `>> 4` to drop
/// the 4 padding LSBs, bringing the 12-bit MSB-aligned sample to
/// the BITS=12 range `[0, 4095]`.
///
/// Samples are passed already endian-corrected by the caller.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn extract_xv36(quad: &[u16]) -> (i32, i32, i32) {
  let u = (quad[0] >> 4) as i32;
  let y = (quad[1] >> 4) as i32;
  let v = (quad[2] >> 4) as i32;
  // quad[3] is padding A — ignored
  (u, y, v)
}

/// Load one XV36 u16 sample, applying a byte-swap for BE wire format
/// when `BE = true`. Uses target-endian aware `u16::from_be`/`u16::from_le`
/// — these are no-ops when the source byte order matches the host, so the
/// helper produces correct samples on both LE and BE hosts (e.g. s390x).
#[cfg_attr(not(tarpaulin), inline(always))]
fn load_xv36_u16<const BE: bool>(v: u16) -> u16 {
  if BE { u16::from_be(v) } else { u16::from_le(v) }
}

// ---- u8 RGB / RGBA output ----------------------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv36_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
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
  let (y_off, y_scale, c_scale) = range_params_n::<12, 8>(full_range);
  let bias = chroma_bias::<12>();

  for x in 0..width {
    let base = x * 4;
    let quad = [
      load_xv36_u16::<BE>(packed[base]),
      load_xv36_u16::<BE>(packed[base + 1]),
      load_xv36_u16::<BE>(packed[base + 2]),
      load_xv36_u16::<BE>(packed[base + 3]),
    ];
    let (u, y, v) = extract_xv36(&quad);
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

// ---- XV36 → HSV (direct: no RGB scratch) -------------------------------
//
// The display-referred twin of [`xv36_to_rgb_or_rgba_row`] (`ALPHA =
// false`), fused with the OpenCV HSV quantizer. It shares the EXACT
// per-pixel **8-bit-output** Q15 decode (`Coefficients::for_matrix` +
// `range_params_n::<12, 8>` + the `>> 4` 12-bit de-pack) as its `_to_rgb`
// sibling, then feeds the decoded `(r, g, b)` straight into
// [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes — never
// materializing a packed-RGB row. The HSV output is 8-bit
// (`H ∈ [0, 179]`, `S, V ∈ [0, 255]`); XV36 is a 12-bit source but its
// existing HSV path is `rgb_to_hsv_row` applied to the **8-bit**
// `xv36_to_rgb` output, so the 8-bit intermediate is reproduced here.
// The A slot (index 3) is padding — read by `extract_xv36` but discarded
// — and HSV derives from the COLOR (U/Y/V → RGB → HSV) only. Byte-
// identical to `rgb_to_hsv_row(xv36_to_rgb_or_rgba_row::<false, BE>(...))`,
// with no RGB allocation.

/// Scalar XV36 → planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV` encoding:
/// `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). Const-generic over `BE` (source
/// byte order), exactly like [`xv36_to_rgb_or_rgba_row`]. 4:4:4 (no
/// chroma subsampling): one U/Y/V triple per pixel. The padding A slot
/// is dropped (HSV is colour-only). Byte-identical to
/// `rgb_to_hsv_row(xv36_to_rgb_or_rgba_row::<false, BE>(...))`.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 4`.
/// - each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn xv36_to_hsv_row<const BE: bool>(
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
  let (y_off, y_scale, c_scale) = range_params_n::<12, 8>(full_range);
  let bias = chroma_bias::<12>();

  for x in 0..width {
    let base = x * 4;
    let quad = [
      load_xv36_u16::<BE>(packed[base]),
      load_xv36_u16::<BE>(packed[base + 1]),
      load_xv36_u16::<BE>(packed[base + 2]),
      load_xv36_u16::<BE>(packed[base + 3]),
    ];
    // The A slot (index 3) is padding — dropped by `extract_xv36`; HSV
    // is colour-only.
    let (u, y, v) = extract_xv36(&quad);
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

// ---- u16 RGB / RGBA native-depth output --------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv36_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
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
  let (y_off, y_scale, c_scale) = range_params_n::<12, 12>(full_range);
  let bias = chroma_bias::<12>();
  let alpha_max: u16 = 0x0FFF;
  let out_max: i32 = 0x0FFF;

  for x in 0..width {
    let base = x * 4;
    let quad = [
      load_xv36_u16::<BE>(packed[base]),
      load_xv36_u16::<BE>(packed[base + 1]),
      load_xv36_u16::<BE>(packed[base + 2]),
      load_xv36_u16::<BE>(packed[base + 3]),
    ];
    let (u, y, v) = extract_xv36(&quad);
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

// ---- Luma (u8) — `>> 8` (drops 4 padding bits + 4 LSBs) ----------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv36_to_luma_row<const BE: bool>(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let y = load_xv36_u16::<BE>(packed[x * 4 + 1]) >> 8;
    out[x] = y as u8;
  }
}

// ---- Luma (u16, low-bit-packed at 12-bit) ------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv36_to_luma_u16_row<const BE: bool>(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let y = load_xv36_u16::<BE>(packed[x * 4 + 1]) >> 4;
    out[x] = y;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::KernelMatrix;

  /// Pack one XV36 pixel (host-native u16 quadruple) from explicit
  /// U / Y / V / A samples. Each channel value must be in `[0, 0xFFF]`;
  /// the helper shifts each by 4 to MSB-align into the high 12 bits.
  fn pack_xv36(u: u16, y: u16, v: u16, a: u16) -> [u16; 4] {
    debug_assert!(u <= 0xFFF && y <= 0xFFF && v <= 0xFFF && a <= 0xFFF);
    [u << 4, y << 4, v << 4, a << 4]
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

  #[test]
  fn xv36_known_pattern_rgb() {
    // Limited-range BT.709, gray Y=256 (≈ 0 in u8) with neutral
    // chroma U=V=2048. Then white Y=3760 (≈ 255).
    let p0 = pack_xv36(2048, 256, 2048, 0);
    let p1 = pack_xv36(2048, 256, 2048, 0);
    let p2 = pack_xv36(2048, 3760, 2048, 0);
    let p3 = pack_xv36(2048, 3760, 2048, 0);
    let intended: Vec<u16> = [p0, p1, p2, p3].iter().flatten().copied().collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u8; 4 * 3];
    xv36_to_rgb_or_rgba_row::<false, false>(&packed, &mut out, 4, KernelMatrix::Bt709, false);
    assert_eq!(&out[0..3], &[0u8, 0, 0]);
    assert_eq!(&out[3..6], &[0u8, 0, 0]);
    assert_eq!(&out[6..9], &[255u8, 255, 255]);
    assert_eq!(&out[9..12], &[255u8, 255, 255]);
  }

  #[test]
  fn xv36_known_pattern_rgba_alpha_max() {
    let p = pack_xv36(2048, 3760, 2048, 0);
    let packed = as_le_u16(&p);
    let mut out = vec![0u8; 4];
    xv36_to_rgb_or_rgba_row::<true, false>(&packed, &mut out, 1, KernelMatrix::Bt709, false);
    // X = padding; RGBA forces α=0xFF regardless of source A byte.
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn xv36_known_pattern_rgba_ignores_source_alpha_bits() {
    // Source A=0x123 (low 12 bits set) — should not leak into RGB or affect α.
    let p = pack_xv36(2048, 3760, 2048, 0xFFF);
    let packed = as_le_u16(&p);
    let mut out = vec![0u8; 4];
    xv36_to_rgb_or_rgba_row::<true, false>(&packed, &mut out, 1, KernelMatrix::Bt709, false);
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn xv36_luma_extract_u8() {
    // Y = 0xFFF → 0xFFF >> 4 = 0xFF (12-bit max); Y = 0x100 → 0x10
    let intended: Vec<u16> = [pack_xv36(0, 0xFFF, 0, 0), pack_xv36(0, 0x100, 0, 0)]
      .iter()
      .flatten()
      .copied()
      .collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u8; 2];
    xv36_to_luma_row::<false>(&packed, &mut out, 2);
    assert_eq!(&out[..], &[0xFFu8, 0x10]);
  }

  #[test]
  fn xv36_luma_extract_u16_low_bit_packed() {
    let intended: Vec<u16> = [pack_xv36(0, 0xFFF, 0, 0), pack_xv36(0, 0x123, 0, 0)]
      .iter()
      .flatten()
      .copied()
      .collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u16; 2];
    xv36_to_luma_u16_row::<false>(&packed, &mut out, 2);
    assert_eq!(&out[..], &[0xFFFu16, 0x123]);
  }

  #[test]
  fn xv36_known_pattern_rgba_u16_alpha_max() {
    let p = pack_xv36(2048, 3760, 2048, 0xFFF);
    let packed = as_le_u16(&p);
    let mut out = vec![0u16; 4];
    xv36_to_rgb_u16_or_rgba_u16_row::<true, false>(
      &packed,
      &mut out,
      1,
      KernelMatrix::Bt709,
      false,
    );
    // 12-bit alpha max = 0x0FFF; X = padding so source A byte is ignored.
    assert_eq!(out[3], 0x0FFF);
  }

  #[test]
  fn xv36_be_roundtrip_matches_byte_swapped_le() {
    // Construct LE/BE buffers from raw bytes via `to_le_bytes` / `to_be_bytes`
    // so semantics are host-independent: on every host, `le` carries the
    // intended values as LE-encoded bytes and `be` carries the same values as
    // BE-encoded bytes. Both kernels should therefore decode to the same
    // intended host-native values (and produce identical RGB output) on both
    // LE and BE hosts. The earlier `swap_bytes` pattern only validated this
    // on LE hosts and degenerated to equal-but-wrong on BE hosts.
    let intended = pack_xv36(1024, 2048, 512, 0);
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
    xv36_to_rgb_or_rgba_row::<false, false>(&le_buf, &mut out_le, 1, KernelMatrix::Bt709, false);
    xv36_to_rgb_or_rgba_row::<false, true>(&be_buf, &mut out_be, 1, KernelMatrix::Bt709, false);
    assert_eq!(out_le, out_be, "XV36 BE scalar must match byte-swapped LE");
  }
}
