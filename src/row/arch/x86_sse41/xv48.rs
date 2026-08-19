//! SSE4.1 kernels for XV48 packed YUV 4:4:4 16-bit family
//! (FFmpeg `AV_PIX_FMT_XV48LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), X(16)]`
//! little-endian, each holding a full 16-bit sample (no padding bits,
//! no right-shift on load — the full-depth sibling of XV36). The `X`
//! slot is **padding** — loaded but discarded. RGBA outputs force
//! α = max (`0xFF` u8 / `0xFFFF` u16).
//!
//! ## u8 pipeline (16 px / iter)
//!
//! Two 8-pixel half-iterations per SIMD block, each using the XV36
//! 3-level unpack cascade (slots U=0, Y=1, V=2, X=3) but with **no**
//! `>> 4` shift (16-bit native). Chroma centered (subtract 32768 via
//! the wrapping `-32768i16` trick), Q15 chroma scale via `chroma_i16x8`
//! (i32 widening — no overflow at BITS=16/8). Y unsigned-widened via
//! `scale_y_u16` (not `scale_y`, which would corrupt Y > 32767).
//!
//! ## u16 pipeline (8 px / iter)
//!
//! i64 chroma arithmetic via `chroma_i64x2` to avoid i32 overflow at
//! BITS=16/16. Y scaled via `scale_y16_i64`.
//!
//! ## Tail
//!
//! `width % block_size` remaining pixels fall through to `scalar::xv48_*`.

use super::{endian, *};
use crate::{KernelMatrix, row::scalar};

// ---- Deinterleave helper ------------------------------------------------

/// Deinterleaves 8 XV48 quadruples (32 u16 = 64 bytes) from `ptr` into
/// `(u_vec, y_vec, v_vec)` — three `__m128i` vectors each holding 8
/// `u16` samples (full 16-bit native — no shift). The X channel (slot 3)
/// is computed but discarded.
///
/// When `BE = true`, each 128-bit load is byte-swapped within every 2-byte
/// lane via `endian::load_endian_u16x8::<true>`.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (32 `u16` elements).
/// Caller's `target_feature` must include SSE4.1.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn deinterleave_xv48<const BE: bool>(ptr: *const u16) -> (__m128i, __m128i, __m128i) {
  unsafe {
    // Load 4 × __m128i (8 pixels × 4 channels × u16 = 64 bytes).
    let raw0 = endian::load_endian_u16x8::<BE>(ptr as *const u8); // U0,Y0,V0,X0,U1,Y1,V1,X1
    let raw1 = endian::load_endian_u16x8::<BE>(ptr.add(8) as *const u8); // U2..X2,U3..X3
    let raw2 = endian::load_endian_u16x8::<BE>(ptr.add(16) as *const u8); // U4..X4,U5..X5
    let raw3 = endian::load_endian_u16x8::<BE>(ptr.add(24) as *const u8); // U6..X6,U7..X7

    // Level 1 unpack (pairs 0-1, pairs 2-3).
    let s1_lo = _mm_unpacklo_epi16(raw0, raw1); // U0,U2,Y0,Y2,V0,V2,X0,X2
    let s1_hi = _mm_unpackhi_epi16(raw0, raw1); // U1,U3,Y1,Y3,V1,V3,X1,X3
    let s2_lo = _mm_unpacklo_epi16(raw2, raw3); // U4,U6,Y4,Y6,V4,V6,X4,X6
    let s2_hi = _mm_unpackhi_epi16(raw2, raw3); // U5,U7,Y5,Y7,V5,V7,X5,X7

    // Level 2 unpack (merge lo/hi within each group).
    let s3_lo = _mm_unpacklo_epi16(s1_lo, s1_hi); // U0,U1,U2,U3,Y0,Y1,Y2,Y3
    let s3_hi = _mm_unpackhi_epi16(s1_lo, s1_hi); // V0,V1,V2,V3,X0,X1,X2,X3
    let s4_lo = _mm_unpacklo_epi16(s2_lo, s2_hi); // U4,U5,U6,U7,Y4,Y5,Y6,Y7
    let s4_hi = _mm_unpackhi_epi16(s2_lo, s2_hi); // V4,V5,V6,V7,X4,X5,X6,X7

    // Level 3: combine the two groups to get full 8-lane channel vectors.
    let u_vec = _mm_unpacklo_epi64(s3_lo, s4_lo); // U0..U7
    let y_vec = _mm_unpackhi_epi64(s3_lo, s4_lo); // Y0..Y7
    let v_vec = _mm_unpacklo_epi64(s3_hi, s4_hi); // V0..V7
    // x_vec = _mm_unpackhi_epi64(s3_hi, s4_hi)  — X0..X7, discarded.

    // No shift — XV48 channels are full 16-bit native.
    (u_vec, y_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (16 px/iter) ----------------------------------

/// SSE4.1 XV48 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv48_to_rgb_or_rgba_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xv48_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    // Y values are full u16 (0..65535); use i32 y_off for scale_y_u16.
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    // Subtract chroma bias (32768) via wrapping i16 trick: -32768i16 == 0x8000.
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // --- lo half: pixels x..x+7 ----------------------------------------
      let (u_lo_u16, y_lo_u16, v_lo_u16) = deinterleave_xv48::<BE>(packed.as_ptr().add(x * 4));

      // Center chroma: subtract 32768 via wrapping i16.
      let u_lo_i16 = _mm_sub_epi16(u_lo_u16, bias16_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_u16, bias16_v);

      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale.
      let y_lo_scaled = scale_y_u16(y_lo_u16, y_off_v, y_scale_v, rnd_v);

      let r_lo_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_lo_scaled, r_chroma_lo),
        _mm_setzero_si128(),
      );
      let g_lo_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_lo_scaled, g_chroma_lo),
        _mm_setzero_si128(),
      );
      let b_lo_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_lo_scaled, b_chroma_lo),
        _mm_setzero_si128(),
      );

      // --- hi half: pixels x+8..x+15 ------------------------------------
      let (u_hi_u16, y_hi_u16, v_hi_u16) = deinterleave_xv48::<BE>(packed.as_ptr().add(x * 4 + 32));

      let u_hi_i16 = _mm_sub_epi16(u_hi_u16, bias16_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_u16, bias16_v);

      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_hi_scaled = scale_y_u16(y_hi_u16, y_off_v, y_scale_v, rnd_v);

      let r_hi_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_hi_scaled, r_chroma_hi),
        _mm_setzero_si128(),
      );
      let g_hi_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_hi_scaled, g_chroma_hi),
        _mm_setzero_si128(),
      );
      let b_hi_u8 = _mm_packus_epi16(
        _mm_adds_epi16(y_hi_scaled, b_chroma_hi),
        _mm_setzero_si128(),
      );

      // Combine lo+hi 8-byte halves into 16-byte vectors for write helpers.
      let r_u8 = _mm_unpacklo_epi64(r_lo_u8, r_hi_u8);
      let g_u8 = _mm_unpacklo_epi64(g_lo_u8, g_hi_u8);
      let b_u8 = _mm_unpacklo_epi64(b_lo_u8, b_hi_u8);

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        // X slot is padding — RGBA forces α = 0xFF.
        let a_vec: __m128i = _mm_set1_epi8(-1i8); // 0xFF
        write_rgba_16(r_u8, g_u8, b_u8, a_vec, out_ptr);
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv48_to_rgb_or_rgba_row::<ALPHA, BE>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (8 px/iter) --------------------

/// SSE4.1 XV48 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x2`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xv48_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16); // 0xFFFF
    let rnd_v = _mm_set1_epi64x(RND);
    let rnd32_v = _mm_set1_epi32(1 << 14);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      // Deinterleave 8 XV48 quadruples → U, Y, V as u16x8.
      let (u_u16, y_vec, v_u16) = deinterleave_xv48::<BE>(packed.as_ptr().add(x * 4));

      // Center chroma via wrapping i16 subtraction.
      let u_i16 = _mm_sub_epi16(u_u16, bias16_v);
      let v_i16 = _mm_sub_epi16(v_u16, bias16_v);

      // Lo half of chroma (pixels 0-3): widen 4 i16 → i32x4.
      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd32_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd32_v));

      // Hi half of chroma (pixels 4-7): widen upper 4 i16 → i32x4.
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd32_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd32_v));

      // i64 chroma via chroma_i64x2 (uses _mm_mul_epi32 on even-indexed lanes).
      // lo half (pixels 0-3):
      let u_d_lo_even = u_d_lo; // lanes 0 and 2
      let v_d_lo_even = v_d_lo;
      let u_d_lo_odd = _mm_shuffle_epi32::<0xF5>(u_d_lo); // lanes 1,3 → even slots
      let v_d_lo_odd = _mm_shuffle_epi32::<0xF5>(v_d_lo);

      let r_ch_lo_even = chroma_i64x2(cru, crv, u_d_lo_even, v_d_lo_even, rnd_v);
      let r_ch_lo_odd = chroma_i64x2(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let g_ch_lo_even = chroma_i64x2(cgu, cgv, u_d_lo_even, v_d_lo_even, rnd_v);
      let g_ch_lo_odd = chroma_i64x2(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let b_ch_lo_even = chroma_i64x2(cbu, cbv, u_d_lo_even, v_d_lo_even, rnd_v);
      let b_ch_lo_odd = chroma_i64x2(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_v);

      // Reassemble i64x2 pairs → i32x4: [ch0, ch1, ch2, ch3].
      let r_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_lo_even, r_ch_lo_odd),
        _mm_unpackhi_epi32(r_ch_lo_even, r_ch_lo_odd),
      );
      let g_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_lo_even, g_ch_lo_odd),
        _mm_unpackhi_epi32(g_ch_lo_even, g_ch_lo_odd),
      );
      let b_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_lo_even, b_ch_lo_odd),
        _mm_unpackhi_epi32(b_ch_lo_even, b_ch_lo_odd),
      );

      // hi half (pixels 4-7):
      let u_d_hi_even = u_d_hi;
      let v_d_hi_even = v_d_hi;
      let u_d_hi_odd = _mm_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_hi_odd = _mm_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_hi_even = chroma_i64x2(cru, crv, u_d_hi_even, v_d_hi_even, rnd_v);
      let r_ch_hi_odd = chroma_i64x2(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let g_ch_hi_even = chroma_i64x2(cgu, cgv, u_d_hi_even, v_d_hi_even, rnd_v);
      let g_ch_hi_odd = chroma_i64x2(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let b_ch_hi_even = chroma_i64x2(cbu, cbv, u_d_hi_even, v_d_hi_even, rnd_v);
      let b_ch_hi_odd = chroma_i64x2(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_v);

      let r_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_hi_even, r_ch_hi_odd),
        _mm_unpackhi_epi32(r_ch_hi_even, r_ch_hi_odd),
      );
      let g_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_hi_even, g_ch_hi_odd),
        _mm_unpackhi_epi32(g_ch_hi_even, g_ch_hi_odd),
      );
      let b_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_hi_even, b_ch_hi_odd),
        _mm_unpackhi_epi32(b_ch_hi_even, b_ch_hi_odd),
      );

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      let y_lo_pair = _mm_cvtepu16_epi32(y_vec); // [Y0,Y1,Y2,Y3] as i32
      let y_hi_pair = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(y_vec)); // [Y4,Y5,Y6,Y7]
      let y_lo_sub = _mm_sub_epi32(y_lo_pair, y_off_v);
      let y_hi_sub = _mm_sub_epi32(y_hi_pair, y_off_v);

      // Even/odd split for _mm_mul_epi32.
      let y_lo_even = scale_y16_i64(y_lo_sub, y_scale_v, rnd_v);
      let y_lo_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_lo_sub), y_scale_v, rnd_v);
      let y_hi_even = scale_y16_i64(y_hi_sub, y_scale_v, rnd_v);
      let y_hi_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_hi_sub), y_scale_v, rnd_v);

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
        _mm_add_epi32(y_lo_i32, r_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, r_ch_hi_i32),
      );
      let g_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, g_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, g_ch_hi_i32),
      );
      let b_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, b_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, b_ch_hi_i32),
      );

      if ALPHA {
        // X slot is padding — RGBA forces α = 0xFFFF.
        write_rgba_u16_8(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_8(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>(
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

/// SSE4.1 XV48 → u8 luma. Y is quadruple element 1; `>> 8` extracts the
/// high byte.
///
/// Byte-identical to `scalar::xv48_to_luma_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xv48_to_luma_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Two deinterleaves for 8 pixels each.
      let (_u_lo, y_lo, _v_lo) = deinterleave_xv48::<BE>(packed.as_ptr().add(x * 4));
      let (_u_hi, y_hi, _v_hi) = deinterleave_xv48::<BE>(packed.as_ptr().add(x * 4 + 32));

      // >> 8 to get u8 luma (high byte of each Y u16 sample).
      let y_lo_shr = _mm_srli_epi16::<8>(y_lo);
      let y_hi_shr = _mm_srli_epi16::<8>(y_hi);
      let y_u8 = _mm_packus_epi16(y_lo_shr, y_hi_shr);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y_u8);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_row::<BE>(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (16 px/iter) ----------------------------------------------

/// SSE4.1 XV48 → u16 luma. Direct copy of Y samples (slot 1, no shift —
/// 16-bit native).
///
/// Byte-identical to `scalar::xv48_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xv48_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Two deinterleaves for 8 pixels each.
      let (_u_lo, y_lo, _v_lo) = deinterleave_xv48::<BE>(packed.as_ptr().add(x * 4));
      let (_u_hi, y_hi, _v_hi) = deinterleave_xv48::<BE>(packed.as_ptr().add(x * 4 + 32));

      // Direct copy — Y samples are 16-bit native (no shift needed).
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y_lo);
      _mm_storeu_si128(out.as_mut_ptr().add(x + 8).cast(), y_hi);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_u16_row::<BE>(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- XV48 → HSV (staged via a reused 8-bit RGB chunk) ----------------
//
// The SIMD twin of the scalar `xv48_to_hsv_row` kernel. Fills a small
// fixed reused **8-bit** RGB scratch (one `HSV_CHUNK`-pixel chunk at a
// time) using the EXISTING `xv48_to_rgb_or_rgba_row::<false, BE>` kernel
// of this file — so the chunk filler IS the production 8-bit RGB kernel —
// then runs the SIMD `rgb_to_hsv_row` on the chunk. Byte-identical to
// `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))` within this
// SIMD tier. The X slot is padding, dropped by the RGB kernel; HSV is
// colour-only.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared SIMD driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb`, then runs the SIMD
/// [`rgb_to_hsv_row`] on that chunk into the H/S/V planes.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// The SIMD feature must be available, and `fill_rgb` must uphold the
/// underlying RGB kernel's safety contract for each chunk. Each of
/// `h_out` / `s_out` / `v_out` must be `>= width`.
#[inline]
unsafe fn xv48_hsv_via_rgb_chunks(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  mut fill_rgb: impl FnMut(usize, usize, &mut [u8]),
) {
  let mut scratch = [0u8; HSV_CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(HSV_CHUNK);
    fill_rgb(offset, n, &mut scratch[..n * 3]);
    // SAFETY: the SIMD feature is verified by the wrapper's
    // `#[target_feature]`; the chunk and the output sub-slices are all
    // length `n`.
    unsafe {
      rgb_to_hsv_row(
        &scratch[..n * 3],
        &mut h_out[offset..offset + n],
        &mut s_out[offset..offset + n],
        &mut v_out[offset..offset + n],
        n,
      );
    }
    offset += n;
  }
}

/// SIMD: XV48 (packed 4:4:4, 16-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the reused-8-bit-RGB-chunk pattern over the
/// SIMD [`xv48_to_rgb_or_rgba_row`] + [`rgb_to_hsv_row`]. Byte-identical
/// to `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))` within
/// this SIMD tier. The padding X slot is dropped (HSV is colour-only).
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn xv48_to_hsv_row<const BE: bool>(
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

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the SIMD XV48 RGB kernel under
  // the same contract (its own scalar tail covers small n).
  unsafe {
    xv48_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      xv48_to_rgb_or_rgba_row::<false, BE>(&packed[offset * 4..], rgb, n, matrix, full_range);
    });
  }
}
