#![allow(dead_code)] // dispatcher (Tasks 9-13) not yet wired

//! AVX2 kernels for AYUV64 packed YUV 4:4:4 16-bit family
//! (FFmpeg `AV_PIX_FMT_AYUV64LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)`.
//! All channels are 16-bit native — no padding bits, no right-shift on
//! load. Channel slot order at deinterleave output: **A=0, Y=1, U=2,
//! V=3** (differs from XV36's U/Y/V/A).
//!
//! ## Per-iter pipeline (32 px / iter for u8, 16 px / iter for u16)
//!
//! Both paths use the same 16-pixel deinterleave helper. The u8 path
//! runs the helper twice per main-loop iteration (lo half = pixels
//! 0..15, hi half = pixels 16..31) and then narrows to u8x32. The u16
//! path runs the helper once and emits 16 pixels of u16 RGBA/RGB.
//!
//! Per 16-pixel deinterleave, four `_mm256_loadu_si256` loads fetch
//! 64 u16 = 128 bytes (16 pixels of 4-channel u16). The post-fix XV36
//! AVX2 pattern is used: `_mm256_permute2x128_si256` reshapes the four
//! contiguous loads into the strided lane layout the per-128-bit-lane
//! 3-level `_mm256_unpacklo/hi_epi16` cascade expects, then
//! `_mm256_unpacklo/hi_epi64` combines into natural pixel order. **No
//! trailing `_mm256_permute4x64_epi64` is needed** (and adding one
//! would scramble the result — that was the Ship 12b bug).
//!
//! ```text
//! After 4 contiguous loads:
//!   raw_c0 lo=P0..3   hi=...     (pixels 0..3, all 4 channels)
//!   raw_c1 lo=P4..7   hi=...     (pixels 4..7)
//!   raw_c2 lo=P8..11  hi=...     (pixels 8..11)
//!   raw_c3 lo=P12..15 hi=...     (pixels 12..15)
//!
//! After permute2x128 reshape:
//!   raw0 lo=P0..1 hi=P8..9
//!   raw1 lo=P2..3 hi=P10..11
//!   raw2 lo=P4..5 hi=P12..13
//!   raw3 lo=P6..7 hi=P14..15
//!
//! After 3-level unpacklo/hi_epi16 + unpacklo/hi_epi64 cascade:
//!   a_vec lo=[A0..A7] hi=[A8..A15]  (natural pixel order)
//!   y_vec lo=[Y0..Y7] hi=[Y8..Y15]
//!   u_vec lo=[U0..U7] hi=[U8..U15]
//!   v_vec lo=[V0..V7] hi=[V8..V15]
//! ```
//!
//! Lane n of every channel vector is the channel sample from pixel n
//! in *natural* order, by construction of the cross-lane reshape — no
//! lane-fixup `_mm256_permute4x64_epi64` is needed.
//!
//! ## u8 pipeline (32 px / iter)
//!
//! Two halves × 16-pixel deinterleaves. Per half: chroma centered
//! (subtract 32768 via wrapping `-32768i16` trick), Q15 chroma scale via
//! `chroma_i16x16` (i32 widening — no overflow at BITS=16/8). Y scaled
//! via `scale_y_u16_avx2` (unsigned-widened to avoid sign-bit corruption
//! for Y > 32767). Saturating add Y + chroma → narrow to u8x32 via
//! `narrow_u8x32`. Source α: `_mm256_srli_epi16::<8>` (high byte) +
//! `_mm256_packus_epi16` to depth-convert u16 → u8.
//!
//! ## u16 pipeline (16 px / iter)
//!
//! i64 chroma via `chroma_i64x4_avx2` to avoid i32 overflow at
//! BITS=16/16. Y scaled via `scale_y_i32x8_i64`. Per pixel: 4 i32x8
//! halves → reassembled to i32x8 → saturating-narrow to u16 via
//! `_mm256_packus_epi32` + `0xD8` lane fixup. Source α: deinterleaved
//! A vector written direct (no conversion).
//!
//! ## Tail
//!
//! `width % block_size` remaining pixels fall through to
//! `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>` (or u16 variant).

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Deinterleave helper (16 pixels / 64 u16 / 128 bytes) ---------------

/// Deinterleaves 16 AYUV64 quadruples (64 u16 = 128 bytes) from `ptr`
/// into `(a_vec, y_vec, u_vec, v_vec)` — four `__m256i` vectors each
/// holding 16 `u16` samples in **natural pixel order** (lane n = u16
/// from pixel n).
///
/// Channel slot order in source: A=0, Y=1, U=2, V=3 (AYUV64 native).
/// No shift is applied (16-bit native samples).
///
/// ## Strategy (post-fix XV36 pattern)
///
/// 1. Four contiguous `_mm256_loadu_si256` loads fetch 16 pixels'
///    worth of A/Y/U/V (128 bytes).
/// 2. Four `_mm256_permute2x128_si256` calls reshape the contiguous
///    loads into the strided lane layout the per-128-bit-lane unpack
///    cascade expects:
///    `raw0 lo=P0,P1 hi=P8,P9   raw1 lo=P2,P3 hi=P10,P11`
///    `raw2 lo=P4,P5 hi=P12,P13 raw3 lo=P6,P7 hi=P14,P15`
/// 3. Per-lane `_mm256_unpacklo/hi_epi16` cascade (3 levels) produces
///    interleaved channel samples.
/// 4. Final `_mm256_unpacklo/hi_epi64` step combines into full 16-lane
///    channel vectors in natural pixel order.
///
/// Because the cross-lane reshape in step 2 placed pixels 0..7 in the
/// lo 128-bit lane and 8..15 in the hi 128-bit lane of each
/// downstream register, no `_mm256_permute4x64_epi64` lane-fixup is
/// needed at the end. Adding one would scramble the result (this was
/// the Ship 12b second bug).
///
/// # Safety
///
/// `ptr` must point to at least 128 readable bytes (64 `u16`
/// elements). Caller's `target_feature` must include AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_ayuv64_16px_avx2(ptr: *const u16) -> (__m256i, __m256i, __m256i, __m256i) {
  // SAFETY: caller obligation — `ptr` has 128 bytes readable; AVX2 is
  // available.
  unsafe {
    // Load 4 × __m256i contiguously (16 pixels × 4 channels × u16 = 128 bytes).
    //
    // Each load covers 4 contiguous pixels (16 u16 elements = 32 bytes):
    //   raw_c0 lo=A0,Y0,U0,V0,A1,Y1,U1,V1  hi=A2..V2,A3..V3   (pixels 0..3)
    //   raw_c1 lo=A4..V4,A5..V5             hi=A6..V6,A7..V7    (pixels 4..7)
    //   raw_c2 lo=A8..V8,A9..V9             hi=A10..V10,A11..V11 (pixels 8..11)
    //   raw_c3 lo=A12..V12,A13..V13         hi=A14..V14,A15..V15 (pixels 12..15)
    let raw_c0 = _mm256_loadu_si256(ptr.cast());
    let raw_c1 = _mm256_loadu_si256(ptr.add(16).cast());
    let raw_c2 = _mm256_loadu_si256(ptr.add(32).cast());
    let raw_c3 = _mm256_loadu_si256(ptr.add(48).cast());

    // Reshape via cross-lane permute so each register holds the layout
    // the per-128-bit-lane cascade below expects:
    //   raw0 lo=P0,P1 hi=P8,P9
    //   raw1 lo=P2,P3 hi=P10,P11
    //   raw2 lo=P4,P5 hi=P12,P13
    //   raw3 lo=P6,P7 hi=P14,P15
    //
    // `_mm256_permute2x128_si256::<imm>` selects 128-bit halves: imm=0x20
    // picks src1 lo + src2 lo; imm=0x31 picks src1 hi + src2 hi.
    let raw0 = _mm256_permute2x128_si256::<0x20>(raw_c0, raw_c2);
    let raw1 = _mm256_permute2x128_si256::<0x31>(raw_c0, raw_c2);
    let raw2 = _mm256_permute2x128_si256::<0x20>(raw_c1, raw_c3);
    let raw3 = _mm256_permute2x128_si256::<0x31>(raw_c1, raw_c3);

    // Level 1: unpack pairs (0-1, 2-3) and (4-5, 6-7) within each lane.
    // Per-128-bit-lane result (using AYUV order: A=0, Y=1, U=2, V=3):
    //   s1_lo per lane: [A0,A2,Y0,Y2,U0,U2,V0,V2] (lo: px0/2; hi: px8/10)
    //   s1_hi per lane: [A1,A3,Y1,Y3,U1,U3,V1,V3] (lo: px1/3; hi: px9/11)
    //   s2_lo per lane: [A4,A6,Y4,Y6,U4,U6,V4,V6] (lo: px4/6; hi: px12/14)
    //   s2_hi per lane: [A5,A7,Y5,Y7,U5,U7,V5,V7] (lo: px5/7; hi: px13/15)
    let s1_lo = _mm256_unpacklo_epi16(raw0, raw1);
    let s1_hi = _mm256_unpackhi_epi16(raw0, raw1);
    let s2_lo = _mm256_unpacklo_epi16(raw2, raw3);
    let s2_hi = _mm256_unpackhi_epi16(raw2, raw3);

    // Level 2: merge lo/hi within each group.
    // Per-lane:
    //   s3_lo: [A0,A1,A2,A3,Y0,Y1,Y2,Y3] (lo: px0-3; hi: px8-11)
    //   s3_hi: [U0,U1,U2,U3,V0,V1,V2,V3] (lo: px0-3; hi: px8-11)
    //   s4_lo: [A4,A5,A6,A7,Y4,Y5,Y6,Y7] (lo: px4-7; hi: px12-15)
    //   s4_hi: [U4,U5,U6,U7,V4,V5,V6,V7] (lo: px4-7; hi: px12-15)
    let s3_lo = _mm256_unpacklo_epi16(s1_lo, s1_hi);
    let s3_hi = _mm256_unpackhi_epi16(s1_lo, s1_hi);
    let s4_lo = _mm256_unpacklo_epi16(s2_lo, s2_hi);
    let s4_hi = _mm256_unpackhi_epi16(s2_lo, s2_hi);

    // Level 3: combine the two groups via per-lane unpacklo/hi_epi64.
    //
    // Because the load step reshaped via `_mm256_permute2x128_si256` so
    // raw0..raw3 hold strided lanes (raw0: lo=P0,P1 hi=P8,P9; raw1: lo=P2,P3
    // hi=P10,P11; raw2: lo=P4,P5 hi=P12,P13; raw3: lo=P6,P7 hi=P14,P15),
    // the cascade above already accumulates the per-pixel channels into
    // natural [0..15] order:
    //   a_vec lo lane (px 0..7): [A0, A1, A2, A3, A4, A5, A6, A7]
    //         hi lane (px 8..15): [A8..A15]
    //
    // No 4x64 cross-lane permute is needed — applying one would scramble
    // the result to [0..3, 8..11, 4..7, 12..15] (Ship 12b second bug).
    let a_vec = _mm256_unpacklo_epi64(s3_lo, s4_lo);
    let y_vec = _mm256_unpackhi_epi64(s3_lo, s4_lo);
    let u_vec = _mm256_unpacklo_epi64(s3_hi, s4_hi);
    let v_vec = _mm256_unpackhi_epi64(s3_hi, s4_hi);

    (a_vec, y_vec, u_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (32 px/iter) ----------------------------------

/// AVX2 AYUV64 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// Block size: 32 pixels per SIMD iteration (two 16-pixel deinterleaves).
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB (α dropped)
/// - `<true, true>`  — RGBA, source α depth-converted u16 → u8 (`>> 8`)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    // Y values are full u16 (0..65535); use i32 y_off for scale_y_u16_avx2.
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    // Subtract chroma bias (32768) via wrapping i16 trick: -32768i16 == 0x8000.
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    // 0xFF for the (theoretical) opaque path — not emitted by valid
    // monomorphizations, but kept for symmetry with the SSE4.1 sibling.
    let alpha_u8 = _mm256_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 32 <= width {
      // --- lo half: pixels x..x+15 (one 16-pixel deinterleave) ----------
      let (a_lo_u16, y_lo_u16, u_lo_u16, v_lo_u16) =
        deinterleave_ayuv64_16px_avx2(packed.as_ptr().add(x * 4));

      // Center chroma: subtract 32768 via wrapping i16 (-32768i16 == 0x8000).
      let u_lo_i16 = _mm256_sub_epi16(u_lo_u16, bias16_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_u16, bias16_v);

      // Widen each i16x16 chroma into two i32x8 halves for Q15 multiply.
      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));

      // 4:4:4 — no chroma duplication; one chroma sample per Y pixel.
      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);

      // Y: full u16 values → use scale_y_u16_avx2 (NOT scale_y, which
      // would corrupt Y > 32767 by treating as signed).
      let y_lo_scaled = scale_y_u16_avx2(y_lo_u16, y_off_v, y_scale_v, rnd_v);

      // --- hi half: pixels x+16..x+31 (one more 16-pixel deinterleave) --
      let (a_hi_u16, y_hi_u16, u_hi_u16, v_hi_u16) =
        deinterleave_ayuv64_16px_avx2(packed.as_ptr().add(x * 4 + 64));

      let u_hi_i16 = _mm256_sub_epi16(u_hi_u16, bias16_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_u16, bias16_v);

      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_hi_scaled = scale_y_u16_avx2(y_hi_u16, y_off_v, y_scale_v, rnd_v);

      // Saturating add Y + chroma per channel; narrow both halves into
      // u8x32 with natural lane order via `narrow_u8x32` (which applies
      // the post-pack 0xD8 lane fixup).
      let r_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, r_chroma_lo),
        _mm256_adds_epi16(y_hi_scaled, r_chroma_hi),
      );
      let g_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, g_chroma_lo),
        _mm256_adds_epi16(y_hi_scaled, g_chroma_hi),
      );
      let b_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, b_chroma_lo),
        _mm256_adds_epi16(y_hi_scaled, b_chroma_hi),
      );

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        // Source α: depth-convert u16 → u8 via >> 8 (high byte).
        // _mm256_srli_epi16::<8> shifts each u16 right by 8, putting the
        // high byte into the low 8 bits of each 16-bit lane. The lo/hi
        // halves are then narrowed via narrow_u8x32 (which already
        // applies the 0xD8 lane fixup).
        let a_vec: __m256i = if ALPHA_SRC {
          let a_lo_shr = _mm256_srli_epi16::<8>(a_lo_u16);
          let a_hi_shr = _mm256_srli_epi16::<8>(a_hi_u16);
          narrow_u8x32(a_lo_shr, a_hi_shr)
        } else {
          alpha_u8 // 0xFF — opaque (unused, but allowed)
        };
        write_rgba_32(r_u8, g_u8, b_u8, a_vec, out_ptr);
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
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

// ---- u16 RGB / RGBA native-depth output (16 px/iter) --------------------

/// AVX2 AYUV64 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x4_avx2`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>`.
///
/// Block size: 16 pixels per SIMD iteration (one 16-pixel deinterleave).
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB u16 (α dropped)
/// - `<true, true>`  — RGBA u16, source α written direct (no conversion)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16); // 0xFFFF for forced-opaque path.
    let rnd_v = _mm256_set1_epi64x(RND);
    let rnd32_v = _mm256_set1_epi32(1 << 14);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    // Subtract chroma bias (32768) via wrapping i16 trick.
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 AYUV64 quadruples → A, Y, U, V as u16x16 in
      // natural pixel order.
      let (a_u16, y_vec, u_u16, v_u16) = deinterleave_ayuv64_16px_avx2(packed.as_ptr().add(x * 4));

      // Center chroma via wrapping i16 subtraction.
      let u_i16 = _mm256_sub_epi16(u_u16, bias16_v);
      let v_i16 = _mm256_sub_epi16(v_u16, bias16_v);

      // Lo half (pixels 0..7): widen 8 i16 → i32x8.
      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd32_v,
      ));

      // Hi half (pixels 8..15): widen upper 8 i16 → i32x8.
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd32_v,
      ));

      // i64 chroma: even/odd i32 lanes via 0xF5 shuffle.
      // Each `chroma_i64x4_avx2` call processes 4 i64 values from the
      // even-indexed i32 lanes of u_d/v_d. We need 8 per half → two calls
      // per half (even + odd).
      let u_d_lo_odd = _mm256_shuffle_epi32::<0xF5>(u_d_lo);
      let v_d_lo_odd = _mm256_shuffle_epi32::<0xF5>(v_d_lo);
      let u_d_hi_odd = _mm256_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_hi_odd = _mm256_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_even = chroma_i64x4_avx2(cru, crv, u_d_lo, v_d_lo, rnd_v);
      let r_ch_lo_odd = chroma_i64x4_avx2(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let g_ch_lo_even = chroma_i64x4_avx2(cgu, cgv, u_d_lo, v_d_lo, rnd_v);
      let g_ch_lo_odd = chroma_i64x4_avx2(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let b_ch_lo_even = chroma_i64x4_avx2(cbu, cbv, u_d_lo, v_d_lo, rnd_v);
      let b_ch_lo_odd = chroma_i64x4_avx2(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_v);

      let r_ch_hi_even = chroma_i64x4_avx2(cru, crv, u_d_hi, v_d_hi, rnd_v);
      let r_ch_hi_odd = chroma_i64x4_avx2(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let g_ch_hi_even = chroma_i64x4_avx2(cgu, cgv, u_d_hi, v_d_hi, rnd_v);
      let g_ch_hi_odd = chroma_i64x4_avx2(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let b_ch_hi_even = chroma_i64x4_avx2(cbu, cbv, u_d_hi, v_d_hi, rnd_v);
      let b_ch_hi_odd = chroma_i64x4_avx2(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_v);

      // Reassemble each pair of i64x4 → i32x8.
      let r_ch_lo_i32 = reassemble_i64x4_to_i32x8(r_ch_lo_even, r_ch_lo_odd);
      let g_ch_lo_i32 = reassemble_i64x4_to_i32x8(g_ch_lo_even, g_ch_lo_odd);
      let b_ch_lo_i32 = reassemble_i64x4_to_i32x8(b_ch_lo_even, b_ch_lo_odd);
      let r_ch_hi_i32 = reassemble_i64x4_to_i32x8(r_ch_hi_even, r_ch_hi_odd);
      let g_ch_hi_i32 = reassemble_i64x4_to_i32x8(g_ch_hi_even, g_ch_hi_odd);
      let b_ch_hi_i32 = reassemble_i64x4_to_i32x8(b_ch_hi_even, b_ch_hi_odd);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      // y_vec is __m256i with 16 u16 lanes (Y0..Y15).
      let y_lo_u16 = _mm256_castsi256_si128(y_vec);
      let y_hi_u16 = _mm256_extracti128_si256::<1>(y_vec);
      let y_lo_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x8_i64(y_lo_i32, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_i32x8_i64(y_hi_i32, y_scale_v, rnd_v);

      // Add Y + chroma in i32; saturate-narrow to u16 via _mm256_packus_epi32
      // + 0xD8 lane fixup (packus is per-lane; produces lane-split result).
      let r_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, r_ch_lo_i32),
        _mm256_add_epi32(y_hi_scaled, r_ch_hi_i32),
      ));
      let g_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, g_ch_lo_i32),
        _mm256_add_epi32(y_hi_scaled, g_ch_hi_i32),
      ));
      let b_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, b_ch_lo_i32),
        _mm256_add_epi32(y_hi_scaled, b_ch_hi_i32),
      ));

      // Write 16 pixels via two 8-pixel u16 helpers.
      if ALPHA {
        // Source α: direct write (no conversion needed for u16 output).
        let dst = out.as_mut_ptr().add(x * 4);
        let (a_lo, a_hi) = if ALPHA_SRC {
          (
            _mm256_castsi256_si128(a_u16),
            _mm256_extracti128_si256::<1>(a_u16),
          )
        } else {
          (alpha_u16, alpha_u16)
        };
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_u16),
          _mm256_castsi256_si128(g_u16),
          _mm256_castsi256_si128(b_u16),
          a_lo,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_u16),
          _mm256_extracti128_si256::<1>(g_u16),
          _mm256_extracti128_si256::<1>(b_u16),
          a_hi,
          dst.add(32),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_u16),
          _mm256_castsi256_si128(g_u16),
          _mm256_castsi256_si128(b_u16),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_u16),
          _mm256_extracti128_si256::<1>(g_u16),
          _mm256_extracti128_si256::<1>(b_u16),
          dst.add(24),
        );
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
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

/// AVX2 AYUV64 → packed **RGB** (3 bpp). Source α is discarded.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 AYUV64 → packed **RGBA** (4 bpp). Source A u16 is depth-converted
/// to u8 via `>> 8`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 AYUV64 → packed **RGB u16** (3 × u16 per pixel). Source α discarded.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 AYUV64 → packed **RGBA u16** (4 × u16 per pixel). Source A u16
/// is written direct (no conversion).
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 AYUV64 → u8 luma. Y is the second u16 (slot 1) of each pixel
/// quadruple; `>> 8` extracts the high byte.
///
/// Block size: 16 pixels per SIMD iteration (one 16-pixel deinterleave).
/// Reuses the full deinterleave helper and discards A/U/V — the
/// compiler lifts the dead per-channel ops, and keeping the same code
/// path gives the lane-order regression test the strongest possible
/// coverage.
///
/// Byte-identical to `scalar::ayuv64_to_luma_row`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ayuv64_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 pixels and discard A/U/V.
      let (_a, y_vec, _u, _v) = deinterleave_ayuv64_16px_avx2(packed.as_ptr().add(x * 4));

      // y_vec lo lane = [Y0..Y7], hi lane = [Y8..Y15] (16 u16 in natural order).
      // `>> 8` → high byte of each Y u16. Then narrow to u8.
      let y_shr = _mm256_srli_epi16::<8>(y_vec);

      // Narrow to u8x32: only low 16 bytes carry valid data; high 16 from zero.
      let zero = _mm256_setzero_si256();
      let y_u8 = narrow_u8x32(y_shr, zero);

      // Store low 16 bytes (the valid Y values).
      let mut tmp = [0u8; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), y_u8);
      luma_out[x..x + 16].copy_from_slice(&tmp[..16]);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_row(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- Luma u16 (16 px/iter) ----------------------------------------------

/// AVX2 AYUV64 → u16 luma. Direct copy of Y samples (slot 1, no shift —
/// 16-bit native).
///
/// Block size: 16 pixels per SIMD iteration. Reuses the full
/// deinterleave helper and discards A/U/V — compiler lifts dead ops.
///
/// Byte-identical to `scalar::ayuv64_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ayuv64_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_a, y_vec, _u, _v) = deinterleave_ayuv64_16px_avx2(packed.as_ptr().add(x * 4));
      // Direct store — Y samples are 16-bit native, in natural pixel order.
      _mm256_storeu_si256(luma_out.as_mut_ptr().add(x).cast(), y_vec);
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
