//! SSE4.1 Y216 (packed YUV 4:2:2, BITS=16) kernels.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` where each
//! sample occupies the full 16-bit word (no MSB/LSB alignment shift
//! needed — unlike Y210/Y212 which require `>> (16 - BITS)`).
//!
//! ## u8 pipeline (16 px / iter)
//!
//! Per 8-pixel half-iteration, two `_mm_loadu_si128` loads fetch
//! 8 u16 = 4 pixels each:
//!   - `lo` = `[Y0,U0,Y1,V0, Y2,U1,Y3,V1]` (quadruples 0 and 1)
//!   - `hi` = `[Y4,U2,Y5,V2, Y6,U3,Y7,V3]` (quadruples 2 and 3)
//!
//! `_mm_shuffle_epi8` with byte-level masks extracts:
//!   - Y: `[Y0,Y1,Y2,Y3]` (lo) + `[Y4,Y5,Y6,Y7]` (hi) →
//!     combined to `[Y0..Y7]` u16x8 via `_mm_unpacklo_epi64`.
//!   - UV: `[U0,V0,U1,V1]` (lo) + `[U2,V2,U3,V3]` (hi) →
//!     combined to `[U0,V0,U1,V1,U2,V2,U3,V3]` u16x8.
//!   - U / V split: `_mm_shuffle_epi8` with u_idx / v_idx →
//!     `[U0,U1,U2,U3]` (low 4 lanes valid) and `[V0..V3]`.
//!
//! Y is unsigned-widened via `scale_y_u16` (not `scale_y`) to avoid
//! sign-bit corruption for Y values > 32767.
//!
//! ## u16 pipeline (8 px / iter)
//!
//! i64 chroma arithmetic via `chroma_i64x2` + `srai64_15` bias trick.
//! Processes 8 Y pixels (4 chroma pairs) per iteration due to the
//! i64 arithmetic width constraint. Final saturation via
//! `_mm_packus_epi32` (signed i32 → u16).

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- u8 output (i32 chroma, 16 px/iter) ---------------------------------

/// SSE4.1 Y216 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::y216_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn y216_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    // Y216 samples are full u16 [0..65535]; use i32 y_off and
    // scale_y_u16 (unsigned widening) to avoid sign-bit corruption for Y > 32767.
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    // Subtract chroma bias (32768) via wrapping: -32768i16 bits = 0x8000.
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm_set1_epi8(-1);

    // Byte-level shuffle masks for one 8-pixel group (2 loads of 8 u16 each).
    // Each load holds 4 YUYV quadruples = 8 u16 = 16 bytes.
    // Byte layout of one load `[Y0,U0,Y1,V0,Y2,U1,Y3,V1]` (bytes):
    //   0,1 = Y0  2,3 = U0  4,5 = Y1  6,7 = V0
    //   8,9 = Y2  10,11 = U1  12,13 = Y3  14,15 = V1
    // Y (even u16 lanes): bytes [0,1,4,5,8,9,12,13] → low 8 bytes, high zeroed.
    let y_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    // Chroma (odd u16 lanes): bytes [2,3,6,7,10,11,14,15] → low 8 bytes.
    let c_idx = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    // U lanes from interleaved [U,V,U,V,...]: even u16 lanes.
    let u_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    // V lanes: odd u16 lanes.
    let v_idx = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      // --- lo group: pixels x..x+7 (8 pixels, 16 u16 = 2 loads) ------
      // packed[x*2 .. x*2+8] = quadruples 0,1 = pixels x..x+3
      // packed[x*2+8 .. x*2+16] = quadruples 2,3 = pixels x+4..x+7
      let lo = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let hi = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 8).cast());

      // Y extraction: [Y0,Y1,Y2,Y3] from lo and [Y4,Y5,Y6,Y7] from hi.
      let y_lo_half = _mm_shuffle_epi8(lo, y_idx); // [Y0,Y1,Y2,Y3, 0,0,0,0] in u16x8
      let y_hi_half = _mm_shuffle_epi8(hi, y_idx); // [Y4,Y5,Y6,Y7, 0,0,0,0]
      let y_lo_vec = _mm_unpacklo_epi64(y_lo_half, y_hi_half); // [Y0..Y7] u16x8

      // Chroma extraction: interleaved [U,V,U,V,...] per 4-pair group.
      let c_lo_half = _mm_shuffle_epi8(lo, c_idx); // [U0,V0,U1,V1, 0,0,0,0]
      let c_hi_half = _mm_shuffle_epi8(hi, c_idx); // [U2,V2,U3,V3, 0,0,0,0]
      let chroma_lo = _mm_unpacklo_epi64(c_lo_half, c_hi_half); // [U0,V0,U1,V1,U2,V2,U3,V3]

      // Split U and V (4 valid low-half lanes each).
      let u_lo = _mm_shuffle_epi8(chroma_lo, u_idx); // [U0,U1,U2,U3, 0,0,0,0] u16x8
      let v_lo = _mm_shuffle_epi8(chroma_lo, v_idx); // [V0,V1,V2,V3, 0,0,0,0] u16x8

      // Center UV: subtract 32768 wrapping.
      let u_lo_i16 = _mm_sub_epi16(u_lo, bias16_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo, bias16_v);

      // Widen 4 valid i16 chroma lanes to i32x4 for Q15 scale.
      let u_lo_i32 = _mm_cvtepi16_epi32(u_lo_i16); // [U0,U1,U2,U3]
      let v_lo_i32 = _mm_cvtepi16_epi32(v_lo_i16); // [V0,V1,V2,V3]
      // `_mm_cvtepi16_epi32` uses the low 4 lanes; high 4 of u_lo_i16 are
      // 0x8080 garbage from the -1-byte shuffles, but we don't use them.
      // Widen the high half too for `chroma_i16x8` (don't-care input).
      let u_lo_hi = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let v_lo_hi = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_lo_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_hi, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_lo_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_hi, c_scale_v), rnd_v));

      // chroma_i16x8 takes two i32x4 halves (lo=valid lanes 0..3,
      // hi=don't-care lanes 4..7) → produces i16x8 with only lanes 0..3 valid.
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);

      // Duplicate each chroma sample into its Y-pair slot (4:2:2):
      // unpacklo_epi16([c0,c1,c2,c3,...], same) → [c0,c0,c1,c1,c2,c2,c3,c3]
      let r_dup_lo = _mm_unpacklo_epi16(r_chroma_lo, r_chroma_lo);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma_lo, g_chroma_lo);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma_lo, b_chroma_lo);

      // Scale Y: unsigned-widening avoids i16 overflow for Y > 32767.
      let y_lo_scaled = scale_y_u16(y_lo_vec, y_off_v, y_scale_v, rnd_v);

      // Saturating add and narrow to u8.
      let r_lo_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_lo_scaled, r_dup_lo),
        _mm_setzero_si128(),
      );
      let g_lo_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_lo_scaled, g_dup_lo),
        _mm_setzero_si128(),
      );
      let b_lo_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_lo_scaled, b_dup_lo),
        _mm_setzero_si128(),
      );

      // --- hi group: pixels x+8..x+15 ---------------------------------
      let lo2 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 16).cast());
      let hi2 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 24).cast());

      let y_lo2_half = _mm_shuffle_epi8(lo2, y_idx);
      let y_hi2_half = _mm_shuffle_epi8(hi2, y_idx);
      let y_hi_vec = _mm_unpacklo_epi64(y_lo2_half, y_hi2_half); // [Y8..Y15]

      let c_lo2_half = _mm_shuffle_epi8(lo2, c_idx);
      let c_hi2_half = _mm_shuffle_epi8(hi2, c_idx);
      let chroma_hi = _mm_unpacklo_epi64(c_lo2_half, c_hi2_half);

      let u_hi = _mm_shuffle_epi8(chroma_hi, u_idx);
      let v_hi = _mm_shuffle_epi8(chroma_hi, v_idx);

      let u_hi_i16 = _mm_sub_epi16(u_hi, bias16_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi, bias16_v);

      let u_hi_i32 = _mm_cvtepi16_epi32(u_hi_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(v_hi_i16);
      let u_hi_hi = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_hi_hi = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let u_d_hi_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_hi, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));
      let v_d_hi_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_hi, c_scale_v), rnd_v));

      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi, v_d_hi, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi, v_d_hi, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi, v_d_hi, u_d_hi_hi, v_d_hi_hi, rnd_v);

      let r_dup_hi = _mm_unpacklo_epi16(r_chroma_hi, r_chroma_hi);
      let g_dup_hi = _mm_unpacklo_epi16(g_chroma_hi, g_chroma_hi);
      let b_dup_hi = _mm_unpacklo_epi16(b_chroma_hi, b_chroma_hi);

      let y_hi_scaled = scale_y_u16(y_hi_vec, y_off_v, y_scale_v, rnd_v);

      let r_hi_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_hi_scaled, r_dup_hi),
        _mm_setzero_si128(),
      );
      let g_hi_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_hi_scaled, g_dup_hi),
        _mm_setzero_si128(),
      );
      let b_hi_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_hi_scaled, b_dup_hi),
        _mm_setzero_si128(),
      );

      // Combine two 8-pixel groups into 16-pixel output.
      // Each *_lo_u8 / *_hi_u8 holds 8 valid u8 in its low 8 bytes.
      // `_mm_unpacklo_epi64` joins the two low halves → 16 valid u8.
      let r_u8 = _mm_unpacklo_epi64(r_lo_u8, r_hi_u8);
      let g_u8 = _mm_unpacklo_epi64(g_lo_u8, g_hi_u8);
      let b_u8 = _mm_unpacklo_epi64(b_lo_u8, b_hi_u8);

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 output (i64 chroma, 8 px/iter) ---------------------------------

/// SSE4.1 Y216 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x2`) to avoid overflow at 16-bit scales.
/// Byte-identical to `scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// Processes 8 pixels (4 chroma pairs) per SIMD iteration due to the
/// i64 arithmetic width constraint.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn y216_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_v = _mm_set1_epi64x(RND);
    let rnd32_v = _mm_set1_epi32(1 << 14);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    // bias 32768 via wrapping i16 trick
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    // Byte-level shuffle masks (same as u8 path).
    let y_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let c_idx = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let v_idx = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 8 <= width {
      // Two 128-bit loads: each covers 8 u16 = 4 pixels.
      // packed[x*2 .. x*2+8] = [Y0,U0,Y1,V0,Y2,U1,Y3,V1]
      // packed[x*2+8 .. x*2+16] = [Y4,U2,Y5,V2,Y6,U3,Y7,V3]
      let lo = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let hi = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 8).cast());

      // Y: [Y0..Y7] u16x8
      let y_lo_half = _mm_shuffle_epi8(lo, y_idx);
      let y_hi_half = _mm_shuffle_epi8(hi, y_idx);
      let y_vec = _mm_unpacklo_epi64(y_lo_half, y_hi_half);

      // UV interleaved: [U0,V0,U1,V1,U2,V2,U3,V3]
      let c_lo_half = _mm_shuffle_epi8(lo, c_idx);
      let c_hi_half = _mm_shuffle_epi8(hi, c_idx);
      let chroma = _mm_unpacklo_epi64(c_lo_half, c_hi_half);

      // U and V (4 valid low-half lanes each)
      let u_vec4 = _mm_shuffle_epi8(chroma, u_idx); // [U0,U1,U2,U3, 0,0,0,0]
      let v_vec4 = _mm_shuffle_epi8(chroma, v_idx); // [V0,V1,V2,V3, 0,0,0,0]

      // Center UV via wrapping i16 subtraction.
      let u_i16 = _mm_sub_epi16(u_vec4, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec4, bias16_v);

      // Scale UV in i32 (4 valid lanes from low half of u_i16/v_i16).
      let u_i32 = _mm_cvtepi16_epi32(u_i16);
      let v_i32 = _mm_cvtepi16_epi32(v_i16);
      let u_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_i32, c_scale_v), rnd32_v));
      let v_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_i32, c_scale_v), rnd32_v));

      // i64 chroma: _mm_mul_epi32 uses even-indexed i32 lanes.
      let u_d_even = u_d;
      let v_d_even = v_d;
      let u_d_odd = _mm_shuffle_epi32::<0xF5>(u_d); // [1,1,3,3] → odd to even
      let v_d_odd = _mm_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x2(cru, crv, u_d_even, v_d_even, rnd_v);
      let r_ch_odd = chroma_i64x2(cru, crv, u_d_odd, v_d_odd, rnd_v);
      let g_ch_even = chroma_i64x2(cgu, cgv, u_d_even, v_d_even, rnd_v);
      let g_ch_odd = chroma_i64x2(cgu, cgv, u_d_odd, v_d_odd, rnd_v);
      let b_ch_even = chroma_i64x2(cbu, cbv, u_d_even, v_d_even, rnd_v);
      let b_ch_odd = chroma_i64x2(cbu, cbv, u_d_odd, v_d_odd, rnd_v);

      // Reassemble i64x2 pairs (even + odd) → i32x4.
      let r_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_even, r_ch_odd),
        _mm_unpackhi_epi32(r_ch_even, r_ch_odd),
      );
      let g_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_even, g_ch_odd),
        _mm_unpackhi_epi32(g_ch_even, g_ch_odd),
      );
      let b_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_even, b_ch_odd),
        _mm_unpackhi_epi32(b_ch_even, b_ch_odd),
      );

      // Duplicate each chroma value for 2 Y pixels per chroma pair (4:2:2).
      // unpacklo_epi32([r0,r1,r2,r3], same) → [r0,r0,r1,r1] (pixels 0,1,2,3)
      // unpackhi_epi32([r0,r1,r2,r3], same) → [r2,r2,r3,r3] (pixels 4,5,6,7)
      let r_dup_lo = _mm_unpacklo_epi32(r_ch_i32, r_ch_i32);
      let r_dup_hi = _mm_unpackhi_epi32(r_ch_i32, r_ch_i32);
      let g_dup_lo = _mm_unpacklo_epi32(g_ch_i32, g_ch_i32);
      let g_dup_hi = _mm_unpackhi_epi32(g_ch_i32, g_ch_i32);
      let b_dup_lo = _mm_unpacklo_epi32(b_ch_i32, b_ch_i32);
      let b_dup_hi = _mm_unpackhi_epi32(b_ch_i32, b_ch_i32);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      let y_lo_pair = _mm_cvtepu16_epi32(y_vec); // [y0,y1,y2,y3] as i32
      let y_hi_pair = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(y_vec)); // [y4,y5,y6,y7]
      let y_lo_sub = _mm_sub_epi32(y_lo_pair, y_off_v);
      let y_hi_sub = _mm_sub_epi32(y_hi_pair, y_off_v);

      // Even/odd split for _mm_mul_epi32.
      let y_lo_even = scale_y16_i64(y_lo_sub, y_scale_v, rnd_v);
      let y_lo_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_lo_sub), y_scale_v, rnd_v);
      let y_hi_even = scale_y16_i64(y_hi_sub, y_scale_v, rnd_v);
      let y_hi_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_hi_sub), y_scale_v, rnd_v);

      // Reassemble Y i64x2 pairs to i32x4.
      let y_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_lo_even, y_lo_odd),
        _mm_unpackhi_epi32(y_lo_even, y_lo_odd),
      );
      let y_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_hi_even, y_hi_odd),
        _mm_unpackhi_epi32(y_hi_even, y_hi_odd),
      );

      // Add Y + chroma, saturate i32 → u16 via _mm_packus_epi32.
      let r_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, r_dup_lo),
        _mm_add_epi32(y_hi_i32, r_dup_hi),
      );
      let g_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, g_dup_lo),
        _mm_add_epi32(y_hi_i32, g_dup_hi),
      );
      let b_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, b_dup_lo),
        _mm_add_epi32(y_hi_i32, b_dup_hi),
      );

      if ALPHA {
        write_rgba_u16_8(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_8(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (16 px/iter) -----------------------------------------------

/// SSE4.1 Y216 → u8 luma. Extracts Y via `>> 8`.
///
/// Byte-identical to `scalar::y216_to_luma_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn y216_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  unsafe {
    // Pick even u16 lanes (Y samples) into low 8 bytes, zero high bytes.
    let y_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      // Four loads covering 16 pixels (16 u16 per load pair).
      // packed offset x*2 = quadruple-base for pixel x.
      // lo0/hi0 cover pixels x..x+7, lo1/hi1 cover x+8..x+15.
      let lo0 = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let hi0 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 8).cast());
      let lo1 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 16).cast());
      let hi1 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 24).cast());

      // Extract Y lanes into u16x8.
      let y_lo_half = _mm_shuffle_epi8(lo0, y_idx); // [Y0..Y3, 0..]
      let y_hi_half = _mm_shuffle_epi8(hi0, y_idx); // [Y4..Y7, 0..]
      let y_vec_lo = _mm_unpacklo_epi64(y_lo_half, y_hi_half); // [Y0..Y7]

      let y_lo2_half = _mm_shuffle_epi8(lo1, y_idx); // [Y8..Y11, 0..]
      let y_hi2_half = _mm_shuffle_epi8(hi1, y_idx); // [Y12..Y15, 0..]
      let y_vec_hi = _mm_unpacklo_epi64(y_lo2_half, y_hi2_half); // [Y8..Y15]

      // `>> 8` to get u8 luma (high byte of each Y sample).
      let y_lo_shr = _mm_srli_epi16::<8>(y_vec_lo);
      let y_hi_shr = _mm_srli_epi16::<8>(y_vec_hi);
      // Pack 16 × i16 → 16 × u8.
      let y_u8 = _mm_packus_epi16(y_lo_shr, y_hi_shr);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y_u8);

      x += 16;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_row(tail_packed, tail_out, tail_w);
    }
  }
}

// ---- Luma u16 (16 px/iter) ----------------------------------------------

/// SSE4.1 Y216 → u16 luma. Direct copy of Y samples (no shift).
///
/// Byte-identical to `scalar::y216_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn y216_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  unsafe {
    let y_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      let lo0 = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let hi0 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 8).cast());
      let lo1 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 16).cast());
      let hi1 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 24).cast());

      let y_lo_half = _mm_shuffle_epi8(lo0, y_idx);
      let y_hi_half = _mm_shuffle_epi8(hi0, y_idx);
      let y_vec_lo = _mm_unpacklo_epi64(y_lo_half, y_hi_half); // [Y0..Y7]

      let y_lo2_half = _mm_shuffle_epi8(lo1, y_idx);
      let y_hi2_half = _mm_shuffle_epi8(hi1, y_idx);
      let y_vec_hi = _mm_unpacklo_epi64(y_lo2_half, y_hi2_half); // [Y8..Y15]

      // Direct copy — full 16-bit Y values, no shift.
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y_vec_lo);
      _mm_storeu_si128(out.as_mut_ptr().add(x + 8).cast(), y_vec_hi);

      x += 16;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_u16_row(tail_packed, tail_out, tail_w);
    }
  }
}
