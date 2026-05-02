//! SSE4.1 XV36 (packed YUV 4:4:4, 12-bit) kernels.
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), A(16)]`
//! little-endian, each holding a 12-bit sample MSB-aligned in the
//! high 12 bits (low 4 bits zero). The `X` prefix means the A slot
//! is **padding** — loaded but discarded. RGBA outputs force α = max
//! (`0xFF` u8 / `0x0FFF` u16).
//!
//! ## Per-iter pipeline (8 px / iter)
//!
//! Four `_mm_loadu_si128` loads fetch 32 u16 lanes (8 pixels × 4
//! channels). A cascade of `_mm_unpacklo_epi16` / `_mm_unpackhi_epi16`
//! deinterleaves them into four channel vectors (U, Y, V, A) of 8 u16
//! lanes each:
//!
//! ```text
//! raw0 = [U0,Y0,V0,A0, U1,Y1,V1,A1]   (pixels 0-1)
//! raw1 = [U2,Y2,V2,A2, U3,Y3,V3,A3]   (pixels 2-3)
//! raw2 = [U4,Y4,V4,A4, U5,Y5,V5,A5]   (pixels 4-5)
//! raw3 = [U6,Y6,V6,A6, U7,Y7,V7,A7]   (pixels 6-7)
//!
//! step1_lo = unpacklo(raw0, raw1) = [U0,U2,Y0,Y2,V0,V2,A0,A2]
//! step1_hi = unpackhi(raw0, raw1) = [U1,U3,Y1,Y3,V1,V3,A1,A3]
//! step2_lo = unpacklo(raw2, raw3) = [U4,U6,Y4,Y6,V4,V6,A4,A6]
//! step2_hi = unpackhi(raw2, raw3) = [U5,U7,Y5,Y7,V5,V7,A5,A7]
//!
//! step3_lo = unpacklo(step1_lo, step1_hi) = [U0,U1,U2,U3,Y0,Y1,Y2,Y3]
//! step3_hi = unpackhi(step1_lo, step1_hi) = [V0,V1,V2,V3,A0,A1,A2,A3]
//! step4_lo = unpacklo(step2_lo, step2_hi) = [U4,U5,U6,U7,Y4,Y5,Y6,Y7]
//! step4_hi = unpackhi(step2_lo, step2_hi) = [V4,V5,V6,V7,A4,A5,A6,A7]
//!
//! u_vec = unpacklo_epi64(step3_lo, step4_lo) = [U0..U7]
//! y_vec = unpackhi_epi64(step3_lo, step4_lo) = [Y0..Y7]
//! v_vec = unpacklo_epi64(step3_hi, step4_hi) = [V0..V7]
//! a_vec = unpackhi_epi64(step3_hi, step4_hi) = [A0..A7]  (discarded)
//! ```
//!
//! Each channel is then right-shifted by 4 (`_mm_srli_epi16::<4>`) to
//! drop the 4 padding LSBs, bringing the 12-bit MSB-aligned sample to
//! `[0, 4095]`. From there the Q15 pipeline at BITS=12 is identical to
//! the NEON and V410 siblings: `chroma_i16x8` (i32 chroma) + `scale_y`.
//!
//! ## 4:4:4 vs. 4:2:2
//!
//! XV36 is 4:4:4 — no chroma duplication is needed. Each pixel has its
//! own unique `(U, Y, V)` triple.
//!
//! ## Tail
//!
//! `width % 8` remaining pixels fall through to `scalar::xv36_*`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Deinterleave helper ------------------------------------------------

/// Deinterleaves 8 XV36 quadruples (32 u16 = 64 bytes) from `ptr` into
/// `(u_vec, y_vec, v_vec)` — three `__m128i` vectors each holding 8
/// `u16` samples **after** the 4-bit right-shift to drop padding LSBs.
/// The A channel is computed but returned separately (caller discards it).
///
/// See module-level doc for the 3-level unpack cascade.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (32 `u16` elements).
/// Caller's `target_feature` must include SSE4.1.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn deinterleave_xv36(ptr: *const u16) -> (__m128i, __m128i, __m128i) {
  unsafe {
    // Load 4 × __m128i (8 pixels × 4 channels × u16 = 64 bytes).
    let raw0 = _mm_loadu_si128(ptr.cast()); // U0,Y0,V0,A0,U1,Y1,V1,A1
    let raw1 = _mm_loadu_si128(ptr.add(8).cast()); // U2,Y2,V2,A2,U3,Y3,V3,A3
    let raw2 = _mm_loadu_si128(ptr.add(16).cast()); // U4,Y4,V4,A4,U5,Y5,V5,A5
    let raw3 = _mm_loadu_si128(ptr.add(24).cast()); // U6,Y6,V6,A6,U7,Y7,V7,A7

    // Level 1 unpack (pairs 0-1, pairs 2-3).
    let s1_lo = _mm_unpacklo_epi16(raw0, raw1); // U0,U2,Y0,Y2,V0,V2,A0,A2
    let s1_hi = _mm_unpackhi_epi16(raw0, raw1); // U1,U3,Y1,Y3,V1,V3,A1,A3
    let s2_lo = _mm_unpacklo_epi16(raw2, raw3); // U4,U6,Y4,Y6,V4,V6,A4,A6
    let s2_hi = _mm_unpackhi_epi16(raw2, raw3); // U5,U7,Y5,Y7,V5,V7,A5,A7

    // Level 2 unpack (merge lo/hi within each group).
    let s3_lo = _mm_unpacklo_epi16(s1_lo, s1_hi); // U0,U1,U2,U3,Y0,Y1,Y2,Y3
    let s3_hi = _mm_unpackhi_epi16(s1_lo, s1_hi); // V0,V1,V2,V3,A0,A1,A2,A3
    let s4_lo = _mm_unpacklo_epi16(s2_lo, s2_hi); // U4,U5,U6,U7,Y4,Y5,Y6,Y7
    let s4_hi = _mm_unpackhi_epi16(s2_lo, s2_hi); // V4,V5,V6,V7,A4,A5,A6,A7

    // Level 3: combine the two groups to get full 8-lane channel vectors.
    let u_raw = _mm_unpacklo_epi64(s3_lo, s4_lo); // U0..U7
    let y_raw = _mm_unpackhi_epi64(s3_lo, s4_lo); // Y0..Y7
    let v_raw = _mm_unpacklo_epi64(s3_hi, s4_hi); // V0..V7
    // a_raw = _mm_unpackhi_epi64(s3_hi, s4_hi)  — A0..A7, discarded.

    // Right-shift by 4 to drop MSB-alignment padding → 12-bit range [0, 4095].
    let u_vec = _mm_srli_epi16::<4>(u_raw);
    let y_vec = _mm_srli_epi16::<4>(y_raw);
    let v_vec = _mm_srli_epi16::<4>(v_raw);

    (u_vec, y_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (8 px/iter) -----------------------------------

/// SSE4.1 XV36 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv36_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xv36_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 8>(full_range);
  let bias = scalar::chroma_bias::<12>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      // Deinterleave 8 XV36 quadruples → U, Y, V as i16x8 in [0, 4095].
      let (u_u16, y_u16, v_u16) = deinterleave_xv36(packed.as_ptr().add(x * 4));

      // Reinterpret as signed i16 (values ≤ 4095 < 32767, safe).
      let u_i16 = u_u16; // u16 values fit in i16 range
      let y_i16 = y_u16;
      let v_i16 = v_u16;

      // Subtract chroma bias (2048 for 12-bit).
      let u_sub = _mm_sub_epi16(u_i16, bias_v);
      let v_sub = _mm_sub_epi16(v_i16, bias_v);

      // Widen to i32x4 lo/hi for Q15 chroma-scale multiply.
      let u_d_lo_i32 = _mm_cvtepi16_epi32(u_sub);
      let u_d_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_sub));
      let v_d_lo_i32 = _mm_cvtepi16_epi32(v_sub);
      let v_d_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_sub));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_d_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_d_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_d_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_d_hi_i32, c_scale_v), rnd_v));

      // 4:4:4 — no chroma duplication; all 8 lanes carry unique U/V per pixel.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Y values ≤ 4095 fit in i16; use scale_y (NOT scale_y_u16).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Saturate-narrow to u8 with saturation. Low 8 bytes per channel
      // hold valid results; high 8 bytes (from setzero) are zero.
      let zero = _mm_setzero_si128();
      let r_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, r_chroma), zero);
      let g_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, g_chroma), zero);
      let b_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, b_chroma), zero);

      // 8-pixel partial store via stack buffer + scalar interleave.
      let mut r_tmp = [0u8; 16];
      let mut g_tmp = [0u8; 16];
      let mut b_tmp = [0u8; 16];
      _mm_storeu_si128(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm_storeu_si128(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm_storeu_si128(b_tmp.as_mut_ptr().cast(), b_u8);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 8 * 4];
        for i in 0..8 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 8 * 3];
        for i in 0..8 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (8 px/iter) ---------------------

/// SSE4.1 XV36 → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 12-bit).
///
/// Byte-identical to `scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xv36_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 12>(full_range);
  let bias = scalar::chroma_bias::<12>();
  const RND: i32 = 1 << 14;
  // 12-bit output max (low-bit-packed): [0, 0x0FFF].
  let out_max: i16 = 0x0FFF;
  let alpha_u16: u16 = 0x0FFF;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      let (u_u16, y_u16, v_u16) = deinterleave_xv36(packed.as_ptr().add(x * 4));

      let u_i16 = u_u16;
      let y_i16 = y_u16;
      let v_i16 = v_u16;

      let u_sub = _mm_sub_epi16(u_i16, bias_v);
      let v_sub = _mm_sub_epi16(v_i16, bias_v);

      let u_d_lo_i32 = _mm_cvtepi16_epi32(u_sub);
      let u_d_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_sub));
      let v_d_lo_i32 = _mm_cvtepi16_epi32(v_sub);
      let v_d_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_sub));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_d_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_d_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_d_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_d_hi_i32, c_scale_v), rnd_v));

      // 12-bit chroma: i32 arithmetic is sufficient (no overflow at 12-bit).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // XV36 Y ≤ 4095 fits in i16 — use scale_y (not scale_y_u16).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x0FFF] (12-bit low-bit-packed output range).
      let r = clamp_u16_max(_mm_adds_epi16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max(_mm_adds_epi16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max(_mm_adds_epi16(y_scaled, b_chroma), zero_v, max_v);

      // 8-pixel u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 8];
      let mut g_tmp = [0u16; 8];
      let mut b_tmp = [0u16; 8];
      _mm_storeu_si128(r_tmp.as_mut_ptr().cast(), r);
      _mm_storeu_si128(g_tmp.as_mut_ptr().cast(), g);
      _mm_storeu_si128(b_tmp.as_mut_ptr().cast(), b);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 8 * 4];
        for i in 0..8 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha_u16;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 8 * 3];
        for i in 0..8 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (8 px/iter) ------------------------------------------------

/// SSE4.1 XV36 → u8 luma. Y is quadruple element 1 (offset 1 in each
/// group of 4 u16). `>> 8` drops the 4 padding LSBs plus 4 more MSB
/// bits → 8-bit (same as scalar `packed[x*4+1] >> 8`).
///
/// Byte-identical to `scalar::xv36_to_luma_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xv36_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      // Deinterleave to get Y channel, then shift >> 8 for u8 luma.
      let (_u_vec, y_vec, _v_vec) = deinterleave_xv36(packed.as_ptr().add(x * 4));

      // y_vec already has >> 4 applied (values in [0, 4095]).
      // Scalar does `packed[x*4+1] >> 8` — that's (MSB-aligned >> 4) >> 4
      // = the 12-bit value >> 4 → 8-bit. Apply one more >> 4 shift.
      let y_shr = _mm_srli_epi16::<4>(y_vec);

      // Narrow to u8 via packus (values ≤ 255, no saturation needed).
      let y_u8 = _mm_packus_epi16(y_shr, _mm_setzero_si128());

      // Store 8 valid bytes (lower half of the 16-byte register).
      let mut tmp = [0u8; 16];
      _mm_storeu_si128(tmp.as_mut_ptr().cast(), y_u8);
      out[x..x + 8].copy_from_slice(&tmp[..8]);

      x += 8;
    }

    // Scalar tail.
    if x < width {
      scalar::xv36_to_luma_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (8 px/iter) -----------------------------------------------

/// SSE4.1 XV36 → u16 luma (low-bit-packed at 12-bit). Y is quadruple
/// element 1; `>> 4` drops the 4 padding LSBs to give a 12-bit value
/// in `[0, 4095]`.
///
/// Byte-identical to `scalar::xv36_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xv36_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      // Deinterleave — y_vec already has >> 4 applied (= 12-bit value).
      let (_u_vec, y_vec, _v_vec) = deinterleave_xv36(packed.as_ptr().add(x * 4));

      // Direct store of 8 × u16 (12-bit values in low bits).
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y_vec);

      x += 8;
    }

    // Scalar tail.
    if x < width {
      scalar::xv36_to_luma_u16_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}
