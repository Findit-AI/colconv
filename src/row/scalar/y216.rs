//! Scalar reference kernels for the Y216 packed YUV 4:2:2 family
//! (BITS=16, full-range u16 samples). Separated from `y2xx.rs`
//! because the u16 native-depth output path uses i64 chroma to
//! avoid overflow — at BITS=16, `q15_coeff x chroma + q15_coeff
//! x chroma` exceeds i32 range. Mirrors
//! `src/row/scalar/yuv_planar_16bit.rs`'s i64 chroma scalar
//! pattern but sourced from YUYV-shaped u16 quadruples rather
//! than separate Y/U/V planes.
//!
//! ## Big-endian wire format (`BE = true`)
//!
//! When `BE = true`, each `u16` element in `packed` is stored in
//! big-endian byte order. `load_endian_u16::<BE>` handles the
//! conditional byte-swap at each sample site; the unused branch is
//! eliminated at monomorphization.

use super::*;

// ---- u8 RGB / RGBA output (i32 chroma — same as Y210/Y212) -------

/// `BE = true` selects big-endian wire decoding for each u16 sample.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y216_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2), "Y216 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  let pairs = width / 2;
  // SAFETY: bounds validated above; off4 + 6 < packed.len() * 2 for p < pairs.
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    // No right-shift: BITS=16 means samples are already full-width.
    let y0 = unsafe { load_endian_u16::<BE>(base.add(off4)) } as i32;
    let u = unsafe { load_endian_u16::<BE>(base.add(off4 + 2)) } as i32;
    let y1 = unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) } as i32;
    let v = unsafe { load_endian_u16::<BE>(base.add(off4 + 6)) } as i32;

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

// ---- u16 RGB / RGBA native-depth output (i64 chroma) ----------------

/// `BE = true` selects big-endian wire decoding for each u16 sample.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y216_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2), "Y216 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  let pairs = width / 2;
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    let y0 = unsafe { load_endian_u16::<BE>(base.add(off4)) } as i32;
    let u = unsafe { load_endian_u16::<BE>(base.add(off4 + 2)) } as i32;
    let y1 = unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) } as i32;
    let v = unsafe { load_endian_u16::<BE>(base.add(off4 + 6)) } as i32;

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    for (k, &y) in [y0, y1].iter().enumerate() {
      let y_s = q15_scale64(y - y_off, y_scale);
      let off = (p * 2 + k) * bpp;
      out[off] = (y_s + r_chroma).clamp(0, out_max) as u16;
      out[off + 1] = (y_s + g_chroma).clamp(0, out_max) as u16;
      out[off + 2] = (y_s + b_chroma).clamp(0, out_max) as u16;
      if ALPHA {
        out[off + 3] = 0xFFFF;
      }
    }
  }
}

// ---- Luma (u8) — `>> 8` ----------------------------------------------

/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y216_to_luma_row<const BE: bool>(packed: &[u16], out: &mut [u8], width: usize) {
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2));
  assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let pairs = width / 2;
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    let y0 = unsafe { load_endian_u16::<BE>(base.add(off4)) };
    let y1 = unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) };
    out[p * 2] = (y0 >> 8) as u8;
    out[p * 2 + 1] = (y1 >> 8) as u8;
  }
}

// ---- Luma (u16, direct extract) ---------------------------------------

/// `BE = true` selects big-endian wire decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y216_to_luma_u16_row<const BE: bool>(packed: &[u16], out: &mut [u16], width: usize) {
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2));
  assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let pairs = width / 2;
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    // Direct extract — full 16 bits, no shift; byte-swap if BE.
    out[p * 2] = unsafe { load_endian_u16::<BE>(base.add(off4)) };
    out[p * 2 + 1] = unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) };
  }
}

// ---- Y216 (BITS=16) → HSV (direct: no RGB scratch) -------------------
//
// The display-referred twin of [`y216_to_rgb_or_rgba_row`], fused with
// the OpenCV HSV quantizer. It shares the EXACT per-pixel **8-bit-output**
// Q15 decode (`Coefficients::for_matrix` + `range_params_n::<16, 8>` +
// the i32 chroma path + the packed 4:2:2 `Y₀, U, Y₁, V` extraction, no
// shift) as the RGB-u8 sibling, then feeds the decoded `(r, g, b)`
// straight into [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes —
// never materializing a packed-RGB row. Byte-identical to
// `rgb_to_hsv_row(y216_to_rgb_or_rgba_row::<false, BE>(...))` (the 8-bit
// RGB-u8 path uses i32 chroma, so this HSV kernel does too — only the
// native-depth u16 RGB path needs i64). The SIMD backends mirror this via
// a small reused 8-bit-RGB chunk filled by the existing SIMD
// `y216_to_rgb_or_rgba_row` plus the SIMD `rgb_to_hsv_row`.

/// Y216 (BITS=16) → planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV`
/// encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). `BE` selects the source
/// byte order. Chroma is half-width packed `U, V`, nearest-neighbor 1→2
/// upsampled per pixel pair.
///
/// Byte-identical to `rgb_to_hsv_row(y216_to_rgb_or_rgba_row::<false,
/// BE>(...))`.
///
/// # Panics
///
/// - `width` must be even.
/// - `packed.len() >= width * 2` (one u16 quadruple per chroma pair).
/// - Each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn y216_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // assert! (not debug_assert!) — bounds gate `unsafe load_endian_u16`
  // reads below; release-mode check prevents UB on bad inputs.
  assert!(width.is_multiple_of(2), "Y216 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  let pairs = width / 2;
  let base = packed.as_ptr().cast::<u8>();
  for p in 0..pairs {
    let off4 = p * 4 * 2;
    // No right-shift: BITS=16 means samples are already full-width.
    let y0 = unsafe { load_endian_u16::<BE>(base.add(off4)) } as i32;
    let u = unsafe { load_endian_u16::<BE>(base.add(off4 + 2)) } as i32;
    let y1 = unsafe { load_endian_u16::<BE>(base.add(off4 + 4)) } as i32;
    let v = unsafe { load_endian_u16::<BE>(base.add(off4 + 6)) } as i32;

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

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::KernelMatrix;

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

  // Test input layout: two YUYV quadruples (= 4 pixels, width=4), in
  // LE-encoded byte form so kernels with `BE = false` recover the
  // intended logical values on both LE and BE hosts.
  // Pair 0: Y0=4096, U=32768, Y1=32000, V=32768  (neutral chroma, Bt709 limited floor / mid-gray)
  // Pair 1: Y2=0,    U=16384, Y3=65535, V=49152  (non-neutral chroma, below-floor / above-ceil Y)
  fn test_input() -> std::vec::Vec<u16> {
    as_le_u16(&[4096, 32768, 32000, 32768, 0, 16384, 65535, 49152])
  }

  /// Re-encode a host-native u16 slice as BE-encoded byte storage. Mirror of
  /// `as_le_u16` for kernels invoked with `BE = true`. Combined with
  /// `as_le_u16`, lets a single host-native `intended` fixture drive both
  /// `<ALPHA, false>` and `<ALPHA, true>` kernel paths so they decode the
  /// same logical values on every host.
  fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
    host
      .iter()
      .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
      .collect()
  }

  // -- Scalar references for the BE-parity tests --
  //
  // Walk host-native `intended` buffers (laid out as YUYV `u16` quadruples,
  // BITS=16) and reproduce each kernel's documented behaviour without going
  // through any byte-order conversion. Pinning the LE / BE outputs against
  // these absolute references prevents the parity assertion from passing in
  // lock-step on two equally corrupt decode paths.

  /// Reference for `y216_to_rgb_or_rgba_row::<false, _>` (RGB u8 path, i32
  /// chroma at BITS=16). Mirrors the in-source kernel without any endian
  /// conversion.
  fn ref_y216_to_rgb_u8(
    intended: &[u16],
    width: usize,
    matrix: KernelMatrix,
    full_range: bool,
  ) -> std::vec::Vec<u8> {
    let coeffs = Coefficients::for_matrix(matrix);
    let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
    let bias = chroma_bias::<16>();
    let pairs = width / 2;
    let mut out = std::vec![0u8; width * 3];
    for p in 0..pairs {
      let off4 = p * 4;
      let y0 = intended[off4] as i32;
      let u = intended[off4 + 1] as i32;
      let y1 = intended[off4 + 2] as i32;
      let v = intended[off4 + 3] as i32;
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

  /// Reference for `y216_to_rgb_u16_or_rgba_u16_row::<false, _>` (RGB u16
  /// path, i64 chroma at BITS=16).
  fn ref_y216_to_rgb_u16(
    intended: &[u16],
    width: usize,
    matrix: KernelMatrix,
    full_range: bool,
  ) -> std::vec::Vec<u16> {
    let coeffs = Coefficients::for_matrix(matrix);
    let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
    let bias = chroma_bias::<16>();
    let out_max: i32 = 0xFFFF;
    let pairs = width / 2;
    let mut out = std::vec![0u16; width * 3];
    for p in 0..pairs {
      let off4 = p * 4;
      let y0 = intended[off4] as i32;
      let u = intended[off4 + 1] as i32;
      let y1 = intended[off4 + 2] as i32;
      let v = intended[off4 + 3] as i32;
      let u_d = q15_scale(u - bias, c_scale);
      let v_d = q15_scale(v - bias, c_scale);
      let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
      let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
      let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);
      for (k, &y) in [y0, y1].iter().enumerate() {
        let y_s = q15_scale64(y - y_off, y_scale);
        let off = (p * 2 + k) * 3;
        out[off] = (y_s + r_chroma).clamp(0, out_max) as u16;
        out[off + 1] = (y_s + g_chroma).clamp(0, out_max) as u16;
        out[off + 2] = (y_s + b_chroma).clamp(0, out_max) as u16;
      }
    }
    out
  }

  /// Reference for `y216_to_luma_row` (Y `>> 8`).
  fn ref_y216_to_luma(intended: &[u16], width: usize) -> std::vec::Vec<u8> {
    let pairs = width / 2;
    let mut out = std::vec![0u8; width];
    for p in 0..pairs {
      let off4 = p * 4;
      out[p * 2] = (intended[off4] >> 8) as u8;
      out[p * 2 + 1] = (intended[off4 + 2] >> 8) as u8;
    }
    out
  }

  /// Reference for `y216_to_luma_u16_row` (direct Y extract).
  fn ref_y216_to_luma_u16(intended: &[u16], width: usize) -> std::vec::Vec<u16> {
    let pairs = width / 2;
    let mut out = std::vec![0u16; width];
    for p in 0..pairs {
      let off4 = p * 4;
      out[p * 2] = intended[off4];
      out[p * 2 + 1] = intended[off4 + 2];
    }
    out
  }

  /// The host-native `intended` source for `test_input` — pre-LE-encoding.
  /// `test_input()` is the LE-encoded view; `as_be_u16(&intended_test_input())`
  /// is the BE-encoded view.
  fn intended_test_input() -> std::vec::Vec<u16> {
    std::vec![4096, 32768, 32000, 32768, 0, 16384, 65535, 49152]
  }

  /// u8 RGB output — hand-derived expected values for Bt709 limited range.
  ///
  /// Pair 0 (neutral chroma, U=V=32768=bias → u_d=v_d=0 → chroma=0):
  ///   Y0=4096 = y_off → y_s=0 → clamp → R=G=B=0
  ///   Y1=32000 → y_s=127 → R=G=B=127
  /// Pair 1 (U=16384, V=49152 → u_d=-73, v_d=73 → r_c=115, g_c=-21, b_c=-135):
  ///   Y2=0 (below floor, y_s=-19) → R=clamp(-19+115,0,255)=96, G=0, B=0
  ///   Y3=65535 (above ceil, y_s=279) → R=255, G=255, B=clamp(279-135,0,255)=144
  #[test]
  fn y216_known_pattern_rgb() {
    let packed = test_input();
    let mut out = [0u8; 4 * 3];
    y216_to_rgb_or_rgba_row::<false, false>(&packed, &mut out, 4, KernelMatrix::Bt709, false);

    // Pixel 0: Y=4096 (limited-range black), neutral chroma → (0, 0, 0)
    assert_eq!(&out[0..3], &[0, 0, 0], "pixel 0 (Y=4096, neutral chroma)");
    // Pixel 1: Y=32000, neutral chroma → (127, 127, 127)
    assert_eq!(
      &out[3..6],
      &[127, 127, 127],
      "pixel 1 (Y=32000, neutral chroma)"
    );
    // Pixel 2: Y=0 (below floor), U=16384, V=49152 → (96, 0, 0)
    assert_eq!(&out[6..9], &[96, 0, 0], "pixel 2 (Y=0, non-neutral chroma)");
    // Pixel 3: Y=65535 (above ceil), U=16384, V=49152 → (255, 255, 144)
    assert_eq!(
      &out[9..12],
      &[255, 255, 144],
      "pixel 3 (Y=65535, non-neutral chroma)"
    );
  }

  /// RGBA output — same values as RGB plus alpha=0xFF at byte [3].
  #[test]
  fn y216_known_pattern_rgba() {
    let packed = test_input();
    let mut out = [0u8; 4 * 4];
    y216_to_rgb_or_rgba_row::<true, false>(&packed, &mut out, 4, KernelMatrix::Bt709, false);

    assert_eq!(&out[0..4], &[0, 0, 0, 0xFF]);
    assert_eq!(&out[4..8], &[127, 127, 127, 0xFF]);
    assert_eq!(&out[8..12], &[96, 0, 0, 0xFF]);
    assert_eq!(&out[12..16], &[255, 255, 144, 0xFF]);
  }

  /// u16 RGB output (i64 chroma path) — Bt709 limited range.
  ///
  /// Pair 0 neutral chroma:
  ///   Y=4096 (floor) → y_s=0 → R=G=B=0
  ///   Y=32000 → y_s=32618 → R=G=B=32618
  #[test]
  fn y216_known_pattern_rgb_u16() {
    let packed = test_input();
    let mut out = [0u16; 4 * 3];
    y216_to_rgb_u16_or_rgba_u16_row::<false, false>(
      &packed,
      &mut out,
      4,
      KernelMatrix::Bt709,
      false,
    );

    // Pixel 0: Y=4096 = limited-range floor → all channels 0
    assert_eq!(
      &out[0..3],
      &[0, 0, 0],
      "pixel 0 (Y=4096, neutral chroma, u16)"
    );
    // Pixel 1: Y=32000 neutral chroma → 32618 on all channels
    assert_eq!(
      &out[3..6],
      &[32618, 32618, 32618],
      "pixel 1 (Y=32000, neutral chroma, u16)"
    );
    // Pixel 2: non-neutral chroma — pixel-exact i64-path values
    assert_eq!(&out[6..9], &[24702_u16, 0, 0]);
    // Pixel 3
    assert_eq!(&out[9..12], &[65535_u16, 65535, 37073]);
  }

  /// u16 RGBA output — same color values, alpha=0xFFFF.
  #[test]
  fn y216_known_pattern_rgba_u16() {
    let packed = test_input();
    let mut out = [0u16; 4 * 4];
    y216_to_rgb_u16_or_rgba_u16_row::<true, false>(
      &packed,
      &mut out,
      4,
      KernelMatrix::Bt709,
      false,
    );

    assert_eq!(&out[0..4], &[0, 0, 0, 0xFFFF]);
    assert_eq!(&out[4..8], &[32618, 32618, 32618, 0xFFFF]);
    // Pixel 2: non-neutral chroma — pixel-exact i64-path values
    assert_eq!(&out[8..12], &[24702_u16, 0, 0, 0xFFFF]);
    // Pixel 3
    assert_eq!(&out[12..16], &[65535_u16, 65535, 37073, 0xFFFF]);
  }

  /// Luma u8: each Y extracted via `>> 8`. Input quadruple
  /// [0xAB12, 0x4444, 0xCD34, 0x5555] → luma [0xAB, 0xCD].
  #[test]
  fn y216_luma_extract() {
    let packed = as_le_u16(&[0xAB12u16, 0x4444, 0xCD34, 0x5555]);
    let mut out = [0u8; 2];
    y216_to_luma_row::<false>(&packed, &mut out, 2);
    assert_eq!(out[0], 0xAB, "Y0 luma u8");
    assert_eq!(out[1], 0xCD, "Y1 luma u8");
  }

  /// Luma u16: each Y extracted directly (no shift). Input quadruple
  /// [0xAB12, 0x4444, 0xCD34, 0x5555] → luma [0xAB12, 0xCD34].
  #[test]
  fn y216_luma_u16_extract() {
    let packed = as_le_u16(&[0xAB12u16, 0x4444, 0xCD34, 0x5555]);
    let mut out = [0u16; 2];
    y216_to_luma_u16_row::<false>(&packed, &mut out, 2);
    assert_eq!(out[0], 0xAB12, "Y0 luma u16");
    assert_eq!(out[1], 0xCD34, "Y1 luma u16");
  }

  // ---- BE=true parity tests -------------------------------------------
  //
  // Pattern: build a single host-native `intended` fixture, materialise it as
  // LE-encoded bytes via `as_le_u16` and BE-encoded bytes via `as_be_u16`,
  // run both `<_, false>` and `<_, true>` kernels, and pin each output against
  // an absolute scalar reference so the parity assertion cannot pass on two
  // equally corrupt decodes.

  #[test]
  fn y216_be_rgb_matches_le() {
    let intended = intended_test_input();
    let le = as_le_u16(&intended);
    let be = as_be_u16(&intended);
    let mut out_le = [0u8; 4 * 3];
    let mut out_be = [0u8; 4 * 3];
    y216_to_rgb_or_rgba_row::<false, false>(&le, &mut out_le, 4, KernelMatrix::Bt709, false);
    y216_to_rgb_or_rgba_row::<false, true>(&be, &mut out_be, 4, KernelMatrix::Bt709, false);
    let expected = ref_y216_to_rgb_u8(&intended, 4, KernelMatrix::Bt709, false);
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
    assert_eq!(out_le, out_be, "BE and LE RGB outputs must agree");
  }

  #[test]
  fn y216_be_rgb_u16_matches_le() {
    let intended = intended_test_input();
    let le = as_le_u16(&intended);
    let be = as_be_u16(&intended);
    let mut out_le = [0u16; 4 * 3];
    let mut out_be = [0u16; 4 * 3];
    y216_to_rgb_u16_or_rgba_u16_row::<false, false>(
      &le,
      &mut out_le,
      4,
      KernelMatrix::Bt709,
      false,
    );
    y216_to_rgb_u16_or_rgba_u16_row::<false, true>(&be, &mut out_be, 4, KernelMatrix::Bt709, false);
    let expected = ref_y216_to_rgb_u16(&intended, 4, KernelMatrix::Bt709, false);
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
    assert_eq!(out_le, out_be, "BE and LE RGB u16 outputs must agree");
  }

  #[test]
  fn y216_be_luma_matches_le() {
    let intended = intended_test_input();
    let le = as_le_u16(&intended);
    let be = as_be_u16(&intended);
    let mut luma_le = [0u8; 4];
    let mut luma_be = [0u8; 4];
    y216_to_luma_row::<false>(&le, &mut luma_le, 4);
    y216_to_luma_row::<true>(&be, &mut luma_be, 4);
    let expected = ref_y216_to_luma(&intended, 4);
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
  fn y216_be_luma_u16_matches_le() {
    let intended = intended_test_input();
    let le = as_le_u16(&intended);
    let be = as_be_u16(&intended);
    let mut luma_le = [0u16; 4];
    let mut luma_be = [0u16; 4];
    y216_to_luma_u16_row::<false>(&le, &mut luma_le, 4);
    y216_to_luma_u16_row::<true>(&be, &mut luma_be, 4);
    let expected = ref_y216_to_luma_u16(&intended, 4);
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
