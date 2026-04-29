//! Scalar reference kernels for v210 (Tier 4 packed YUV 4:2:2
//! 10-bit). One v210 word = 16 bytes = 6 pixels. Three 10-bit
//! samples per 32-bit lane in `(low, mid, high)` order.
//!
//! Layout per spec §5.3.1: each 32-bit little-endian word holds
//! three 10-bit samples in bits `[9:0]`, `[19:10]`, `[29:20]`. The
//! upper two bits are zero. Across four words:
//!   word 0: `[Cb0, Y0, Cr0]`
//!   word 1: `[Y1,  Cb1, Y2]`
//!   word 2: `[Cr1, Y3, Cb2]`
//!   word 3: `[Y4,  Cr2, Y5]`
//!
//! The Q15 chroma pipeline that follows the unpack mirrors
//! `yuv_planar_high_bit.rs`'s `yuv_420p_n_to_rgb_or_rgba_row<10, _, _>`
//! exactly — same `range_params_n` / `chroma_bias` / `q15_scale` /
//! `q15_chroma` helpers, just with samples sourced from the v210
//! unpack rather than three planar buffers.

use super::*;

// ---- Word unpack -------------------------------------------------------

/// Extracts 6 Y + 3 U + 3 V 10-bit samples from one 16-byte v210
/// word. Output samples are 10-bit values in the low 10 bits of
/// each `u16`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpack_v210_word(word: &[u8]) -> ([u16; 6], [u16; 3], [u16; 3]) {
  debug_assert_eq!(word.len(), 16);
  let w0 = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
  let w1 = u32::from_le_bytes([word[4], word[5], word[6], word[7]]);
  let w2 = u32::from_le_bytes([word[8], word[9], word[10], word[11]]);
  let w3 = u32::from_le_bytes([word[12], word[13], word[14], word[15]]);

  // Word 0: [Cb0, Y0, Cr0]
  let cb0 = (w0 & 0x3FF) as u16;
  let y0 = ((w0 >> 10) & 0x3FF) as u16;
  let cr0 = ((w0 >> 20) & 0x3FF) as u16;
  // Word 1: [Y1, Cb1, Y2]
  let y1 = (w1 & 0x3FF) as u16;
  let cb1 = ((w1 >> 10) & 0x3FF) as u16;
  let y2 = ((w1 >> 20) & 0x3FF) as u16;
  // Word 2: [Cr1, Y3, Cb2]
  let cr1 = (w2 & 0x3FF) as u16;
  let y3 = ((w2 >> 10) & 0x3FF) as u16;
  let cb2 = ((w2 >> 20) & 0x3FF) as u16;
  // Word 3: [Y4, Cr2, Y5]
  let y4 = (w3 & 0x3FF) as u16;
  let cr2 = ((w3 >> 10) & 0x3FF) as u16;
  let y5 = ((w3 >> 20) & 0x3FF) as u16;

  ([y0, y1, y2, y3, y4, y5], [cb0, cb1, cb2], [cr0, cr1, cr2])
}

// ---- u8 RGB / RGBA output ----------------------------------------------

/// Scalar v210 → packed RGB / RGBA (u8). Const-generic on `ALPHA`:
/// `false` writes 3 bytes per pixel, `true` writes 4 bytes per
/// pixel with `α = 0xFF`. Output bit-depth is u8 (downshifted from
/// the native 10-bit Q15 pipeline via `range_params_n::<10, 8>`).
///
/// # Panics (debug builds)
/// - `width` must be a multiple of 6.
/// - `packed.len() >= (width / 6) * 16`.
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
//
// Scalar prep for Ship 11a: this kernel and its three siblings below
// are consumed by the per-arch SIMD fallbacks and dispatcher entries
// landing in Tasks 4-9. Until those tasks merge, the only callers
// are this file's inline tests, so the unused-helper warnings would
// fail CI's `-D warnings`. The `dead_code` allow lets this prep PR
// ship the foundation without the eventual call sites.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v210_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(
    width.is_multiple_of(6),
    "v210 requires width divisible by 6"
  );
  let words = width / 6;
  debug_assert!(packed.len() >= words * 16, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<10, 8>(full_range);
  let bias = chroma_bias::<10>();

  for w in 0..words {
    let word = &packed[w * 16..w * 16 + 16];
    let (ys, us, vs) = unpack_v210_word(word);

    // 6 pixels per word; each chroma pair (U[i], V[i]) covers
    // Y[2i] and Y[2i+1].
    for i in 0..3 {
      let u_d = q15_scale(us[i] as i32 - bias, c_scale);
      let v_d = q15_scale(vs[i] as i32 - bias, c_scale);

      let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
      let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
      let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

      for k in 0..2 {
        let y = ys[i * 2 + k] as i32;
        let y_s = q15_scale(y - y_off, y_scale);
        let px = w * 6 + i * 2 + k;
        let off = px * bpp;
        out[off] = clamp_u8(y_s + r_chroma);
        out[off + 1] = clamp_u8(y_s + g_chroma);
        out[off + 2] = clamp_u8(y_s + b_chroma);
        if ALPHA {
          out[off + 3] = 0xFF;
        }
      }
    }
  }
}

// ---- u16 RGB / RGBA native-depth output --------------------------------

/// Scalar v210 → packed `u16` RGB / RGBA at native 10-bit depth
/// (low-bit-packed: each output `u16` has its 10 active bits in
/// the low 10, upper 6 bits zero — matches `yuv420p10le`'s
/// fidelity-preserving u16 path).
///
/// `ALPHA = true` writes a 4-element-per-pixel output with α =
/// `(1 << 10) - 1 = 1023` (opaque maximum at 10-bit).
///
/// # Panics (debug builds)
/// - `width` must be a multiple of 6.
/// - `packed.len() >= (width / 6) * 16`.
/// - `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[allow(dead_code)] // Ship 11a prep — see note on `v210_to_rgb_or_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v210_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(
    width.is_multiple_of(6),
    "v210 requires width divisible by 6"
  );
  let words = width / 6;
  debug_assert!(packed.len() >= words * 16, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<10, 10>(full_range);
  let bias = chroma_bias::<10>();
  // 10-bit output range: `[0, 1023]`.
  let out_max: i32 = (1i32 << 10) - 1;
  let alpha_max: u16 = out_max as u16;

  for w in 0..words {
    let word = &packed[w * 16..w * 16 + 16];
    let (ys, us, vs) = unpack_v210_word(word);

    for i in 0..3 {
      let u_d = q15_scale(us[i] as i32 - bias, c_scale);
      let v_d = q15_scale(vs[i] as i32 - bias, c_scale);

      let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
      let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
      let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

      for k in 0..2 {
        let y = ys[i * 2 + k] as i32;
        let y_s = q15_scale(y - y_off, y_scale);
        let px = w * 6 + i * 2 + k;
        let off = px * bpp;
        out[off] = (y_s + r_chroma).clamp(0, out_max) as u16;
        out[off + 1] = (y_s + g_chroma).clamp(0, out_max) as u16;
        out[off + 2] = (y_s + b_chroma).clamp(0, out_max) as u16;
        if ALPHA {
          out[off + 3] = alpha_max;
        }
      }
    }
  }
}

// ---- Luma extraction ---------------------------------------------------

/// Scalar v210 → 8-bit luma. Y values are downshifted from 10-bit
/// to 8-bit via `>> 2`. Bypasses the YUV → RGB pipeline entirely.
///
/// # Panics (debug builds)
/// - `width` must be a multiple of 6.
/// - `packed.len() >= (width / 6) * 16`.
/// - `luma_out.len() >= width`.
#[allow(dead_code)] // Ship 11a prep — see note on `v210_to_rgb_or_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v210_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(
    width.is_multiple_of(6),
    "v210 requires width divisible by 6"
  );
  let words = width / 6;
  debug_assert!(packed.len() >= words * 16, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  for w in 0..words {
    let word = &packed[w * 16..w * 16 + 16];
    let (ys, _, _) = unpack_v210_word(word);
    for k in 0..6 {
      luma_out[w * 6 + k] = (ys[k] >> 2) as u8;
    }
  }
}

/// Scalar v210 → native-depth `u16` luma (low-bit-packed). Each
/// output `u16` carries the source's 10-bit Y value in its low 10
/// bits (upper 6 bits zero).
///
/// # Panics (debug builds)
/// - `width` must be a multiple of 6.
/// - `packed.len() >= (width / 6) * 16`.
/// - `luma_out.len() >= width`.
#[allow(dead_code)] // Ship 11a prep — see note on `v210_to_rgb_or_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v210_to_luma_u16_row(packed: &[u8], luma_out: &mut [u16], width: usize) {
  debug_assert!(
    width.is_multiple_of(6),
    "v210 requires width divisible by 6"
  );
  let words = width / 6;
  debug_assert!(packed.len() >= words * 16, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  for w in 0..words {
    let word = &packed[w * 16..w * 16 + 16];
    let (ys, _, _) = unpack_v210_word(word);
    luma_out[w * 6..w * 6 + 6].copy_from_slice(&ys);
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  /// Build a v210 word from 12 logical samples in v210 standard
  /// order: `[Cb0, Y0, Cr0, Y1, Cb1, Y2, Cr1, Y3, Cb2, Y4, Cr2, Y5]`.
  /// Each sample is a 10-bit value (`0..=1023`).
  fn pack_v210_word(samples: [u16; 12]) -> [u8; 16] {
    let mut out = [0u8; 16];
    // Word 0: bits [9:0] = Cb0, [19:10] = Y0, [29:20] = Cr0
    let w0 = (samples[0] as u32 & 0x3FF)
      | ((samples[1] as u32 & 0x3FF) << 10)
      | ((samples[2] as u32 & 0x3FF) << 20);
    // Word 1: bits [9:0] = Y1, [19:10] = Cb1, [29:20] = Y2
    let w1 = (samples[3] as u32 & 0x3FF)
      | ((samples[4] as u32 & 0x3FF) << 10)
      | ((samples[5] as u32 & 0x3FF) << 20);
    // Word 2: bits [9:0] = Cr1, [19:10] = Y3, [29:20] = Cb2
    let w2 = (samples[6] as u32 & 0x3FF)
      | ((samples[7] as u32 & 0x3FF) << 10)
      | ((samples[8] as u32 & 0x3FF) << 20);
    // Word 3: bits [9:0] = Y4, [19:10] = Cr2, [29:20] = Y5
    let w3 = (samples[9] as u32 & 0x3FF)
      | ((samples[10] as u32 & 0x3FF) << 10)
      | ((samples[11] as u32 & 0x3FF) << 20);
    out[0..4].copy_from_slice(&w0.to_le_bytes());
    out[4..8].copy_from_slice(&w1.to_le_bytes());
    out[8..12].copy_from_slice(&w2.to_le_bytes());
    out[12..16].copy_from_slice(&w3.to_le_bytes());
    out
  }

  #[test]
  fn scalar_v210_to_rgb_gray_is_gray() {
    // Full-range gray: Y=512, U=V=512 (10-bit center).
    let word = pack_v210_word([512; 12]);
    let mut rgb = [0u8; 6 * 3];
    v210_to_rgb_or_rgba_row::<false>(&word, &mut rgb, 6, ColorMatrix::Bt709, true);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  fn scalar_v210_to_rgba_gray_is_gray_with_opaque_alpha() {
    let word = pack_v210_word([512; 12]);
    let mut rgba = [0u8; 6 * 4];
    v210_to_rgb_or_rgba_row::<true>(&word, &mut rgba, 6, ColorMatrix::Bt709, true);
    for px in rgba.chunks(4) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[3], 0xFF);
    }
  }

  #[test]
  fn scalar_v210_to_rgb_u16_gray_is_gray_native_depth() {
    // Full-range gray Y=512 → ~512 in 10-bit RGB out (out_max = 1023).
    let word = pack_v210_word([512; 12]);
    let mut rgb_u16 = [0u16; 6 * 3];
    v210_to_rgb_u16_or_rgba_u16_row::<false>(&word, &mut rgb_u16, 6, ColorMatrix::Bt709, true);
    for px in rgb_u16.chunks(3) {
      // Gray luma at 512 / full-range produces RGB ~512 in 10-bit.
      assert!(px[0].abs_diff(512) <= 2);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  fn scalar_v210_to_rgba_u16_alpha_is_max() {
    let word = pack_v210_word([512; 12]);
    let mut rgba_u16 = [0u16; 6 * 4];
    v210_to_rgb_u16_or_rgba_u16_row::<true>(&word, &mut rgba_u16, 6, ColorMatrix::Bt709, true);
    for px in rgba_u16.chunks(4) {
      assert_eq!(px[3], 1023, "alpha must be (1 << 10) - 1");
    }
  }

  #[test]
  fn scalar_v210_to_luma_extracts_y_bytes() {
    let samples = [
      100, 200, 100, 300, 100, 400, 100, 500, 100, 600, 100, 700, // Cb0,Y0,Cr0,Y1,...
    ];
    let word = pack_v210_word(samples);
    let mut luma = [0u8; 6];
    v210_to_luma_row(&word, &mut luma, 6);
    // Y values: 200, 300, 400, 500, 600, 700 → 10-bit, downshift >> 2.
    assert_eq!(luma[0], (200u16 >> 2) as u8);
    assert_eq!(luma[1], (300u16 >> 2) as u8);
    assert_eq!(luma[2], (400u16 >> 2) as u8);
    assert_eq!(luma[3], (500u16 >> 2) as u8);
    assert_eq!(luma[4], (600u16 >> 2) as u8);
    assert_eq!(luma[5], (700u16 >> 2) as u8);
  }

  #[test]
  fn scalar_v210_to_luma_u16_extracts_y_low_bit_packed() {
    let samples = [100, 200, 100, 300, 100, 400, 100, 500, 100, 600, 100, 700];
    let word = pack_v210_word(samples);
    let mut luma = [0u16; 6];
    v210_to_luma_u16_row(&word, &mut luma, 6);
    assert_eq!(luma[0], 200);
    assert_eq!(luma[1], 300);
    assert_eq!(luma[2], 400);
    assert_eq!(luma[3], 500);
    assert_eq!(luma[4], 600);
    assert_eq!(luma[5], 700);
  }

  #[test]
  fn scalar_v210_handles_two_words_36_pixels() {
    // width = 12 ⇒ 2 words; every pair of words is independent.
    let samples = [512u16; 12];
    let mut packed = std::vec::Vec::with_capacity(32);
    packed.extend_from_slice(&pack_v210_word(samples));
    packed.extend_from_slice(&pack_v210_word(samples));
    let mut rgb = std::vec![0u8; 12 * 3];
    v210_to_rgb_or_rgba_row::<false>(&packed, &mut rgb, 12, ColorMatrix::Bt709, true);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
    }
  }
}
