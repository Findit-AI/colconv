//! Scalar reference kernels for the AYUV64 packed YUV 4:4:4 16-bit
//! family (FFmpeg `AV_PIX_FMT_AYUV64LE`). Each pixel is a 4-u16
//! quadruple `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)`.
//!
//! Source α is real (depth-converted u16 → u8 for u8 RGBA output;
//! written direct as u16 for u16 RGBA output). Type-distinct; no
//! α-as-padding sibling in scope.
//!
//! The `<ALPHA, ALPHA_SRC>` const-generic template covers all valid
//! monomorphizations: `<false, false>` (RGB-only, drops α), `<true,
//! true>` (RGBA with source α). `<false, true>` is rejected at
//! monomorphization. `<true, false>` (force-max α) is unused — no
//! AYUV64x sibling.
//!
//! u8 output uses i32 chroma (output-range scaling keeps within i32);
//! u16 output uses **i64 chroma** via `q15_chroma64` (Q15 sums
//! overflow i32 at BITS=16/16, peak ~3.7e9 for BT.2020).
//!
//! `<const BE: bool>` — when `true`, each `u16` element of the input
//! slice is byte-swapped before use. This handles the `AYUV64BE`
//! big-endian wire format; `BE = false` is the standard LE path.

use super::*;

/// Extract `(u, y, v, a)` from one AYUV64 pixel quadruple.
///
/// Channel slot order: A at slot 0, Y at slot 1, U at slot 2, V at slot 3
/// (differs from VUYA which has A at slot 3). No right-shift needed — 16-bit
/// native samples with no padding bits.
///
/// Samples are passed already endian-corrected by the caller.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn extract_ayuv64(quad: &[u16]) -> (i32, i32, i32, u16) {
  let a = quad[0]; // slot 0 = A (source α)
  let y = quad[1] as i32; // slot 1 = Y
  let u = quad[2] as i32; // slot 2 = U
  let v = quad[3] as i32; // slot 3 = V
  (u, y, v, a) // returned as (u, y, v, a) for consistency with chroma pipeline
}

/// Load one AYUV64 u16 sample, applying a byte-swap for BE wire format
/// when `BE = true`. Uses target-endian aware `u16::from_be`/`u16::from_le`
/// — these are no-ops when the source byte order matches the host, so the
/// helper produces correct samples on both LE and BE hosts (e.g. s390x).
#[cfg_attr(not(tarpaulin), inline(always))]
fn load_ayuv64_u16<const BE: bool>(v: u16) -> u16 {
  if BE { u16::from_be(v) } else { u16::from_le(v) }
}

// ---- u8 output (i32 chroma) --------------------------------------------

/// Shared scalar kernel for AYUV64 → packed **RGB** (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp) or → packed **RGBA** (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp + depth-converted source α).
///
/// Input layout per pixel `n`: `packed[n*4] = A`, `packed[n*4+1] = Y`,
/// `packed[n*4+2] = U`, `packed[n*4+3] = V`. All channels are 16-bit
/// native (no padding bits, no shift required).
///
/// Source α is depth-converted u16 → u8 via `>> 8` when `ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 4`.
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv64_to_rgb_or_rgba_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
>(
  packed: &[u16],
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
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let pix_off = x * 4;
    let quad = [
      load_ayuv64_u16::<BE>(packed[pix_off]),
      load_ayuv64_u16::<BE>(packed[pix_off + 1]),
      load_ayuv64_u16::<BE>(packed[pix_off + 2]),
      load_ayuv64_u16::<BE>(packed[pix_off + 3]),
    ];
    let (u, y, v, a) = extract_ayuv64(&quad);
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
      // ALPHA_SRC=true: depth-convert u16 → u8 by taking high byte (>> 8).
      // ALPHA_SRC=false: force opaque (unused — no AYUV64x sibling).
      out[off + 3] = if ALPHA_SRC { (a >> 8) as u8 } else { 0xFF };
    }
  }
}

// ---- RGB / RGBA u8 thin wrappers ----------------------------------------

/// Scalar AYUV64 → packed **RGB** (3 bpp). Source α is discarded.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv64_to_rgb_row<const BE: bool>(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  ayuv64_to_rgb_or_rgba_row::<false, false, BE>(packed, rgb_out, width, matrix, full_range);
}

/// Scalar AYUV64 → packed **RGBA** (4 bpp). The source A u16 at slot 0
/// of each pixel quadruple is depth-converted to u8 via `>> 8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv64_to_rgba_row<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  ayuv64_to_rgb_or_rgba_row::<true, true, BE>(packed, rgba_out, width, matrix, full_range);
}

// ---- AYUV64 → HSV (direct: no RGB scratch) -----------------------------
//
// The display-referred twin of [`ayuv64_to_rgb_row`], fused with the
// OpenCV HSV quantizer. It shares the EXACT per-pixel **8-bit-output**
// Q15 decode (`Coefficients::for_matrix` + `range_params_n::<16, 8>` +
// the slot-1/2/3 Y/U/V extraction) as its `_to_rgb` sibling, then feeds
// the decoded `(r, g, b)` straight into [`rgb_to_hsv_pixel`] and
// scatters to the H/S/V planes — never materializing a packed-RGB row.
// The HSV output is 8-bit (`H ∈ [0, 179]`, `S, V ∈ [0, 255]`); AYUV64 is
// a 16-bit source but its existing HSV path is `rgb_to_hsv_row` applied
// to the **8-bit** `ayuv64_to_rgb_row` output, so the 8-bit intermediate
// is reproduced here. The source α (slot 0) is independent of HSV — HSV
// derives from the COLOR (Y/U/V → RGB → HSV) only — so it is read and
// dropped exactly as the RGB path drops it. Byte-identical to
// `rgb_to_hsv_row(ayuv64_to_rgb_row::<BE>(...))`, with no RGB allocation.

/// Scalar AYUV64 → planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV`
/// encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). Const-generic over
/// `BE` (source byte order), exactly like [`ayuv64_to_rgb_row`]. 4:4:4
/// (no chroma subsampling): one U/Y/V triple per pixel. Source α is
/// dropped (HSV is colour-only). Byte-identical to
/// `rgb_to_hsv_row(ayuv64_to_rgb_row::<BE>(...))`.
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 4`.
/// - each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ayuv64_to_hsv_row<const BE: bool>(
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
    let pix_off = x * 4;
    let quad = [
      load_ayuv64_u16::<BE>(packed[pix_off]),
      load_ayuv64_u16::<BE>(packed[pix_off + 1]),
      load_ayuv64_u16::<BE>(packed[pix_off + 2]),
      load_ayuv64_u16::<BE>(packed[pix_off + 3]),
    ];
    // Source α (slot 0) is intentionally discarded — HSV is colour-only.
    let (u, y, v, _a) = extract_ayuv64(&quad);
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

// ---- u16 output (i64 chroma) -------------------------------------------

/// Shared scalar kernel for AYUV64 → packed **RGB u16** (`ALPHA = false,
/// ALPHA_SRC = false`, 3 x u16 per pixel) or → packed **RGBA u16**
/// (`ALPHA = true, ALPHA_SRC = true`, 4 x u16 per pixel + source α direct).
///
/// Uses **i64 chroma** via `q15_chroma64` because at BITS=16/16 the
/// Q15 chroma sums exceed i32 range (peak ~3.7x10⁹ for BT.2020-NCL at
/// limited range). Source α is written direct as u16 (no conversion).
///
/// # Panics (debug builds)
///
/// - `packed.len() >= width * 4`.
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv64_to_rgb_u16_or_rgba_u16_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };

  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let pix_off = x * 4;
    let quad = [
      load_ayuv64_u16::<BE>(packed[pix_off]),
      load_ayuv64_u16::<BE>(packed[pix_off + 1]),
      load_ayuv64_u16::<BE>(packed[pix_off + 2]),
      load_ayuv64_u16::<BE>(packed[pix_off + 3]),
    ];
    let (u, y, v, a) = extract_ayuv64(&quad);
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
      // ALPHA_SRC=true: write source α u16 direct (no conversion needed).
      // ALPHA_SRC=false: force opaque (unused — no AYUV64x sibling).
      out[off + 3] = if ALPHA_SRC { a } else { 0xFFFF };
    }
  }
}

// ---- RGB / RGBA u16 thin wrappers ---------------------------------------

/// Scalar AYUV64 → packed **RGB u16** (3 x u16 per pixel). Source α discarded.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv64_to_rgb_u16_row<const BE: bool>(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  ayuv64_to_rgb_u16_or_rgba_u16_row::<false, false, BE>(packed, rgb_out, width, matrix, full_range);
}

/// Scalar AYUV64 → packed **RGBA u16** (4 x u16 per pixel). The source A u16
/// at slot 0 of each pixel quadruple is written direct (no conversion).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv64_to_rgba_u16_row<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  ayuv64_to_rgb_u16_or_rgba_u16_row::<true, true, BE>(packed, rgba_out, width, matrix, full_range);
}

// ---- Luma extraction ---------------------------------------------------

/// Copies only the Y u16 from each AYUV64 pixel into a u8 luma plane,
/// extracting the high byte via `>> 8`. Y is at slot 1 of each quadruple.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv64_to_luma_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  for x in 0..width {
    luma_out[x] = (load_ayuv64_u16::<BE>(packed[x * 4 + 1]) >> 8) as u8;
  }
}

/// Copies only the Y u16 from each AYUV64 pixel into a u16 luma plane,
/// direct (no shift — 16-bit native). Y is at slot 1 of each quadruple.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv64_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  for x in 0..width {
    luma_out[x] = load_ayuv64_u16::<BE>(packed[x * 4 + 1]);
  }
}

// ---- Tests -------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::KernelMatrix;

  /// Build a 4-u16 AYUV64 pixel (host-native u16 quadruple) from explicit
  /// components.
  fn pack_ayuv64(a: u16, y: u16, u: u16, v: u16) -> [u16; 4] {
    [a, y, u, v]
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
  fn ayuv64_known_pattern_rgb_limited_range() {
    let p_black = pack_ayuv64(0xFFFF, 4096, 32768, 32768);
    let p_white = pack_ayuv64(0xFFFF, 60160, 32768, 32768);
    let intended: Vec<u16> = [p_black, p_black, p_white, p_white]
      .iter()
      .flatten()
      .copied()
      .collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u8; 4 * 3];
    ayuv64_to_rgb_row::<false>(&packed, &mut out, 4, KernelMatrix::Bt709, false);
    // Black pixels → [0, 0, 0]
    assert_eq!(&out[0..3], &[0u8, 0, 0], "black pixel 0");
    assert_eq!(&out[3..6], &[0u8, 0, 0], "black pixel 1");
    // White pixels → [255, 255, 255]
    assert_eq!(&out[6..9], &[255u8, 255, 255], "white pixel 2");
    assert_eq!(&out[9..12], &[255u8, 255, 255], "white pixel 3");
  }

  /// AYUV64 RGBA u8: source α = 0x42AB / 0x99CD must appear depth-converted
  /// (>> 8) as 0x42 / 0x99 in the output α channel.
  ///
  /// Alpha values are deliberately non-byte-palindromic (low byte ≠ high
  /// byte) so a BE host that fails to LE-encode the fixture would see the
  /// alpha decoded as 0xAB42 / 0xCD99 and the assertion would fail. This
  /// is the test's only sentinel for the alpha pass-through path under
  /// real LE byte storage — palindromic alpha (e.g. 0x4242) would mask a
  /// missing `as_le_u16` wrap because byte-swap is a no-op on palindromes.
  #[test]
  fn ayuv64_rgba_passes_source_alpha_depth_converted() {
    let p0 = pack_ayuv64(0x42AB, 60160, 32768, 32768);
    let p1 = pack_ayuv64(0x99CD, 60160, 32768, 32768);
    let intended: Vec<u16> = [p0, p1].iter().flatten().copied().collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u8; 2 * 4];
    ayuv64_to_rgba_row::<false>(&packed, &mut out, 2, KernelMatrix::Bt709, false);
    assert_eq!(out[3], 0x42, "pixel 0 alpha (0x42AB >> 8 = 0x42)");
    assert_eq!(out[7], 0x99, "pixel 1 alpha (0x99CD >> 8 = 0x99)");
  }

  /// AYUV64 RGBA u16: source α = 0x42AB / 0x99CD must appear direct
  /// (no conversion) in the output α channel.
  ///
  /// Same non-palindromic-alpha rationale as the u8 variant above:
  /// asserting the full u16 alpha (not just one byte) plus LE-encoding
  /// the fixture turns this from a byte-vacuous check on BE into an
  /// end-to-end alpha pass-through verification under real LE storage.
  #[test]
  fn ayuv64_rgba_u16_passes_source_alpha_direct() {
    let p0 = pack_ayuv64(0x42AB, 60160, 32768, 32768);
    let p1 = pack_ayuv64(0x99CD, 60160, 32768, 32768);
    let intended: Vec<u16> = [p0, p1].iter().flatten().copied().collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u16; 2 * 4];
    ayuv64_to_rgba_u16_row::<false>(&packed, &mut out, 2, KernelMatrix::Bt709, false);
    assert_eq!(out[3], 0x42AB, "pixel 0 alpha u16 direct");
    assert_eq!(out[7], 0x99CD, "pixel 1 alpha u16 direct");
  }

  /// Luma u8: Y at slot 1, extracted via >> 8 (high byte only).
  /// Y=0xFFFF → 0xFF; Y=0x4000 → 0x40.
  #[test]
  fn ayuv64_luma_extract_u8_high_byte() {
    let p0 = pack_ayuv64(0, 0xFFFF, 0, 0);
    let p1 = pack_ayuv64(0, 0x4000, 0, 0);
    let intended: Vec<u16> = [p0, p1].iter().flatten().copied().collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u8; 2];
    ayuv64_to_luma_row::<false>(&packed, &mut out, 2);
    assert_eq!(&out[..], &[0xFFu8, 0x40], "luma u8 high-byte extract");
  }

  /// Luma u16: Y at slot 1, written direct (no shift).
  /// Y=0xABCD → 0xABCD; Y=0x1234 → 0x1234.
  #[test]
  fn ayuv64_luma_extract_u16_direct() {
    let p0 = pack_ayuv64(0, 0xABCD, 0, 0);
    let p1 = pack_ayuv64(0, 0x1234, 0, 0);
    let intended: Vec<u16> = [p0, p1].iter().flatten().copied().collect();
    let packed = as_le_u16(&intended);
    let mut out = vec![0u16; 2];
    ayuv64_to_luma_u16_row::<false>(&packed, &mut out, 2);
    assert_eq!(&out[..], &[0xABCDu16, 0x1234], "luma u16 direct extract");
  }

  #[test]
  fn ayuv64_be_roundtrip_matches_byte_swapped_le() {
    // Construct LE/BE buffers from raw bytes via `to_le_bytes` / `to_be_bytes`
    // so semantics are host-independent: on every host, `le` carries the
    // intended values as LE-encoded bytes and `be` carries the same values as
    // BE-encoded bytes. Both kernels should therefore decode to the same
    // intended host-native values (and produce identical RGB output) on both
    // LE and BE hosts. The earlier `swap_bytes` pattern only validated this
    // on LE hosts and degenerated to equal-but-wrong on BE hosts.
    let intended = pack_ayuv64(0xFFFF, 60160, 32768, 32768);
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
    ayuv64_to_rgb_row::<false>(&le_buf, &mut out_le, 1, KernelMatrix::Bt709, false);
    ayuv64_to_rgb_row::<true>(&be_buf, &mut out_be, 1, KernelMatrix::Bt709, false);
    assert_eq!(
      out_le, out_be,
      "AYUV64 BE scalar must match byte-swapped LE"
    );
  }
}
