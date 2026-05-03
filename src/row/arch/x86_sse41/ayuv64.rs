#![allow(dead_code)] // dispatcher (Tasks 9-13) not yet wired

//! SSE4.1 kernels for AYUV64 packed YUV 4:4:4 16-bit family
//! (FFmpeg `AV_PIX_FMT_AYUV64LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)`.
//! All channels are 16-bit native — no padding bits, no right-shift on load.
//! Channel slot order at deinterleave output: **A=0, Y=1, U=2, V=3**.
//!
//! ## u8 pipeline (16 px / iter)
//!
//! Two 8-pixel half-iterations per SIMD block. Each half uses the same
//! 3-level unpack cascade as XV36 (adapted for the A/Y/U/V slot order):
//!
//! ```text
//! raw0 = [A0,Y0,U0,V0, A1,Y1,U1,V1]   (pixels 0-1)
//! raw1 = [A2,Y2,U2,V2, A3,Y3,U3,V3]   (pixels 2-3)
//! raw2 = [A4,Y4,U4,V4, A5,Y5,U5,V5]   (pixels 4-5)
//! raw3 = [A6,Y6,U6,V6, A7,Y7,U7,V7]   (pixels 6-7)
//!
//! step1_lo = unpacklo(raw0, raw1) = [A0,A2,Y0,Y2,U0,U2,V0,V2]
//! step1_hi = unpackhi(raw0, raw1) = [A1,A3,Y1,Y3,U1,U3,V1,V3]
//! step2_lo = unpacklo(raw2, raw3) = [A4,A6,Y4,Y6,U4,U6,V4,V6]
//! step2_hi = unpackhi(raw2, raw3) = [A5,A7,Y5,Y7,U5,U7,V5,V7]
//!
//! step3_lo = unpacklo(step1_lo, step1_hi) = [A0,A1,A2,A3,Y0,Y1,Y2,Y3]
//! step3_hi = unpackhi(step1_lo, step1_hi) = [U0,U1,U2,U3,V0,V1,V2,V3]
//! step4_lo = unpacklo(step2_lo, step2_hi) = [A4,A5,A6,A7,Y4,Y5,Y6,Y7]
//! step4_hi = unpackhi(step2_lo, step2_hi) = [U4,U5,U6,U7,V4,V5,V6,V7]
//!
//! a_vec = unpacklo_epi64(step3_lo, step4_lo) = [A0..A7]
//! y_vec = unpackhi_epi64(step3_lo, step4_lo) = [Y0..Y7]
//! u_vec = unpacklo_epi64(step3_hi, step4_hi) = [U0..U7]
//! v_vec = unpackhi_epi64(step3_hi, step4_hi) = [V0..V7]
//! ```
//!
//! Y is unsigned-widened via `scale_y_u16` (not `scale_y`) to avoid
//! sign-bit corruption for Y values > 32767.
//!
//! Source α (A channel): `_mm_srli_epi16::<8>` (u16 → high byte in low byte
//! slot), then `_mm_packus_epi16` to narrow to u8.
//!
//! ## u16 pipeline (8 px / iter)
//!
//! i64 chroma arithmetic via `chroma_i64x2` + `srai64_15`. The same 3-level
//! unpack cascade deinterleaves each 8-pixel block. For 4:4:4 we have 8
//! unique U/V values per block, split into lo (pixels 0-3) and hi (pixels 4-7)
//! groups. Y scaled via `scale_y16_i64` + even/odd reassembly. Final
//! saturation via `_mm_packus_epi32`.
//!
//! ## Tail
//!
//! `width % block_size` remaining pixels fall through to
//! `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>` (or u16 variant).

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Deinterleave helper ------------------------------------------------

/// Deinterleaves 8 AYUV64 quadruples (32 u16 = 64 bytes) from `ptr` into
/// `(a_vec, y_vec, u_vec, v_vec)` — four `__m128i` vectors each holding 8
/// `u16` samples. No shift is applied (16-bit native samples).
///
/// Channel slot order in source: A=0, Y=1, U=2, V=3.
///
/// See module-level doc for the 3-level unpack cascade diagram.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (32 `u16` elements).
/// Caller's `target_feature` must include SSE4.1.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn deinterleave_ayuv64(ptr: *const u16) -> (__m128i, __m128i, __m128i, __m128i) {
  unsafe {
    // Load 4 × __m128i (8 pixels × 4 channels × u16 = 64 bytes).
    let raw0 = _mm_loadu_si128(ptr.cast()); // A0,Y0,U0,V0, A1,Y1,U1,V1
    let raw1 = _mm_loadu_si128(ptr.add(8).cast()); // A2,Y2,U2,V2, A3,Y3,U3,V3
    let raw2 = _mm_loadu_si128(ptr.add(16).cast()); // A4,Y4,U4,V4, A5,Y5,U5,V5
    let raw3 = _mm_loadu_si128(ptr.add(24).cast()); // A6,Y6,U6,V6, A7,Y7,U7,V7

    // Level 1 unpack (pairs 0-1, pairs 2-3).
    let s1_lo = _mm_unpacklo_epi16(raw0, raw1); // A0,A2,Y0,Y2,U0,U2,V0,V2
    let s1_hi = _mm_unpackhi_epi16(raw0, raw1); // A1,A3,Y1,Y3,U1,U3,V1,V3
    let s2_lo = _mm_unpacklo_epi16(raw2, raw3); // A4,A6,Y4,Y6,U4,U6,V4,V6
    let s2_hi = _mm_unpackhi_epi16(raw2, raw3); // A5,A7,Y5,Y7,U5,U7,V5,V7

    // Level 2 unpack (merge lo/hi within each group).
    let s3_lo = _mm_unpacklo_epi16(s1_lo, s1_hi); // A0,A1,A2,A3,Y0,Y1,Y2,Y3
    let s3_hi = _mm_unpackhi_epi16(s1_lo, s1_hi); // U0,U1,U2,U3,V0,V1,V2,V3
    let s4_lo = _mm_unpacklo_epi16(s2_lo, s2_hi); // A4,A5,A6,A7,Y4,Y5,Y6,Y7
    let s4_hi = _mm_unpackhi_epi16(s2_lo, s2_hi); // U4,U5,U6,U7,V4,V5,V6,V7

    // Level 3: combine the two groups to get full 8-lane channel vectors.
    let a_vec = _mm_unpacklo_epi64(s3_lo, s4_lo); // A0..A7
    let y_vec = _mm_unpackhi_epi64(s3_lo, s4_lo); // Y0..Y7
    let u_vec = _mm_unpacklo_epi64(s3_hi, s4_hi); // U0..U7
    let v_vec = _mm_unpackhi_epi64(s3_hi, s4_hi); // V0..V7

    (a_vec, y_vec, u_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (16 px/iter) ----------------------------------

/// SSE4.1 AYUV64 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB (α dropped)
/// - `<true, true>`  — RGBA, source α depth-converted u16 → u8 (`>> 8`)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ayuv64_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
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
      let (a_lo_u16, y_lo_u16, u_lo_u16, v_lo_u16) =
        deinterleave_ayuv64(packed.as_ptr().add(x * 4));

      // Center chroma: subtract 32768 via wrapping i16.
      let u_lo_i16 = _mm_sub_epi16(u_lo_u16, bias16_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_u16, bias16_v);

      // Widen chroma to i32x4 lo/hi for Q15 scale multiply.
      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));

      // 4:4:4 chroma for lo 8 lanes.
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale.
      let y_lo_scaled = scale_y_u16(y_lo_u16, y_off_v, y_scale_v, rnd_v);

      // Saturate-narrow to u8x8 (lo 8 bytes of packus result are valid).
      let r_lo_u8 =
        _mm_packus_epi16(_mm_adds_epi16(y_lo_scaled, r_chroma_lo), _mm_setzero_si128());
      let g_lo_u8 =
        _mm_packus_epi16(_mm_adds_epi16(y_lo_scaled, g_chroma_lo), _mm_setzero_si128());
      let b_lo_u8 =
        _mm_packus_epi16(_mm_adds_epi16(y_lo_scaled, b_chroma_lo), _mm_setzero_si128());

      // --- hi half: pixels x+8..x+15 ------------------------------------
      let (a_hi_u16, y_hi_u16, u_hi_u16, v_hi_u16) =
        deinterleave_ayuv64(packed.as_ptr().add(x * 4 + 32));

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

      let r_hi_u8 =
        _mm_packus_epi16(_mm_adds_epi16(y_hi_scaled, r_chroma_hi), _mm_setzero_si128());
      let g_hi_u8 =
        _mm_packus_epi16(_mm_adds_epi16(y_hi_scaled, g_chroma_hi), _mm_setzero_si128());
      let b_hi_u8 =
        _mm_packus_epi16(_mm_adds_epi16(y_hi_scaled, b_chroma_hi), _mm_setzero_si128());

      // Combine lo+hi 8-byte halves into 16-byte vectors for write helpers.
      let r_u8 = _mm_unpacklo_epi64(r_lo_u8, r_hi_u8);
      let g_u8 = _mm_unpacklo_epi64(g_lo_u8, g_hi_u8);
      let b_u8 = _mm_unpacklo_epi64(b_lo_u8, b_hi_u8);

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        // Depth-convert u16 → u8 via >> 8 (take high byte).
        // _mm_srli_epi16::<8> shifts each u16 right by 8, putting the high byte
        // into the low 8 bits of each 16-bit lane. _mm_packus_epi16 then narrows
        // each lane from i16 to u8 (values 0..255, no saturation needed).
        let a_vec: __m128i = if ALPHA_SRC {
          let a_lo_shr = _mm_srli_epi16::<8>(a_lo_u16); // high byte of A u16 → low byte
          let a_hi_shr = _mm_srli_epi16::<8>(a_hi_u16);
          // Narrow 8+8 u16 lanes to 16 u8 lanes.
          _mm_packus_epi16(a_lo_shr, a_hi_shr)
        } else {
          _mm_set1_epi8(-1i8) // 0xFF — opaque (unused, but allowed)
        };
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
      scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
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

/// SSE4.1 AYUV64 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x2`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>`.
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB u16 (α dropped)
/// - `<true, true>`  — RGBA u16, source α written direct (no conversion)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ayuv64_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
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
    // Subtract chroma bias (32768) via wrapping i16 trick.
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      // Deinterleave 8 AYUV64 quadruples → A, Y, U, V as u16x8.
      let (a_u16, y_vec, u_u16, v_u16) = deinterleave_ayuv64(packed.as_ptr().add(x * 4));

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
      // Each call processes 2 values; we need 4 per half (lo) and 4 per half (hi).
      // Split each i32x4 into even (lanes 0,2) and odd (lanes 1,3) pairs.
      //
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
        // Source α: direct write (no conversion needed for u16 output).
        let a_vec: __m128i = if ALPHA_SRC { a_u16 } else { alpha_u16 };
        write_rgba_u16_8(r_u16, g_u16, b_u16, a_vec, out.as_mut_ptr().add(x * 4));
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
      scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- thin wrappers -------------------------------------------------------

/// SSE4.1 AYUV64 → packed **RGB** (3 bpp). Source α is discarded.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ayuv64_to_rgb_row(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 AYUV64 → packed **RGBA** (4 bpp). Source A u16 is depth-converted
/// to u8 via `>> 8`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ayuv64_to_rgba_row(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// SSE4.1 AYUV64 → packed **RGB u16** (3 × u16 per pixel). Source α discarded.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ayuv64_to_rgb_u16_row(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 AYUV64 → packed **RGBA u16** (4 × u16 per pixel). Source A u16
/// is written direct (no conversion).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ayuv64_to_rgba_u16_row(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- Luma u8 (16 px/iter) -----------------------------------------------

/// SSE4.1 AYUV64 → u8 luma. Y is the second u16 (slot 1) of each pixel
/// quadruple; `>> 8` extracts the high byte.
///
/// Uses two deinterleave calls (8 pixels each) per 16-pixel SIMD block.
///
/// Byte-identical to `scalar::ayuv64_to_luma_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ayuv64_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Two deinterleaves for 8 pixels each.
      let (_a_lo, y_lo, _u_lo, _v_lo) = deinterleave_ayuv64(packed.as_ptr().add(x * 4));
      let (_a_hi, y_hi, _u_hi, _v_hi) = deinterleave_ayuv64(packed.as_ptr().add(x * 4 + 32));

      // >> 8 to get u8 luma (high byte of each Y u16 sample).
      let y_lo_shr = _mm_srli_epi16::<8>(y_lo);
      let y_hi_shr = _mm_srli_epi16::<8>(y_hi);
      // Pack 16 × i16 → 16 × u8.
      let y_u8 = _mm_packus_epi16(y_lo_shr, y_hi_shr);
      _mm_storeu_si128(luma_out.as_mut_ptr().add(x).cast(), y_u8);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_row(&packed[x * 4..width * 4], &mut luma_out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (16 px/iter) ----------------------------------------------

/// SSE4.1 AYUV64 → u16 luma. Direct copy of Y samples (slot 1, no shift —
/// 16-bit native).
///
/// Byte-identical to `scalar::ayuv64_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ayuv64_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Two deinterleaves for 8 pixels each.
      let (_a_lo, y_lo, _u_lo, _v_lo) = deinterleave_ayuv64(packed.as_ptr().add(x * 4));
      let (_a_hi, y_hi, _u_hi, _v_hi) = deinterleave_ayuv64(packed.as_ptr().add(x * 4 + 32));

      // Direct copy — Y samples are 16-bit native (no shift needed).
      _mm_storeu_si128(luma_out.as_mut_ptr().add(x).cast(), y_lo);
      _mm_storeu_si128(luma_out.as_mut_ptr().add(x + 8).cast(), y_hi);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_u16_row(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}
