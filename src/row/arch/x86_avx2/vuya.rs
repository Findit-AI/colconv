//! AVX2 kernels for the VUYA / VUYX packed YUV 4:4:4 8-bit family.
//!
//! ## Layout
//!
//! Four `u8` elements per pixel: `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)`.
//! VUYA carries a real alpha channel in byte 3. VUYX treats byte 3 as
//! padding and forces output α to `0xFF`.
//!
//! ## Per-iter pipeline (32 px / iter)
//!
//! Four contiguous `_mm256_loadu_si256` loads fetch 128 bytes = 32
//! pixels of `V U Y A`. Each 256-bit register holds 8 pixels (4 in the
//! lo lane, 4 in the hi lane). Four `_mm256_permute2x128_si256` calls
//! reshape the four contiguous registers into the strided layout the
//! per-128-bit-lane shuffle / unpack cascade expects:
//!
//! ```text
//! After contiguous loads:
//!   raw_c0 lo=P0..3   hi=P4..7
//!   raw_c1 lo=P8..11  hi=P12..15
//!   raw_c2 lo=P16..19 hi=P20..23
//!   raw_c3 lo=P24..27 hi=P28..31
//!
//! After permute2x128 reshape (cascade input):
//!   raw0 lo=P0..3   hi=P16..19   (lo halves of c0 and c2)
//!   raw1 lo=P4..7   hi=P20..23   (hi halves of c0 and c2)
//!   raw2 lo=P8..11  hi=P24..27   (lo halves of c1 and c3)
//!   raw3 lo=P12..15 hi=P28..31   (hi halves of c1 and c3)
//! ```
//!
//! Per-lane `_mm256_shuffle_epi8` with masks gathering bytes at offsets
//! 0/4/8/12 (V), 1/5/9/13 (U), 2/6/10/14 (Y), 3/7/11/15 (A) packs each
//! channel's 4 bytes into the low 4 bytes of each 128-bit lane (upper
//! 12 bytes zeroed).
//!
//! After per-lane shuffle (e.g. for V):
//!   v0: lo=[V0,V1,V2,V3, 0..]  hi=[V16,V17,V18,V19, 0..]
//!   v1: lo=[V4..V7, 0..]       hi=[V20..V23, 0..]
//!   v2: lo=[V8..V11, 0..]      hi=[V24..V27, 0..]
//!   v3: lo=[V12..V15, 0..]     hi=[V28..V31, 0..]
//!
//! `_mm256_unpacklo_epi32(v0, v1)` (per-128-bit-lane) interleaves the low
//! two i32s of each lane:
//!   v_01: lo=[V0..V7, 0..0]    hi=[V16..V23, 0..0]
//! `_mm256_unpacklo_epi32(v2, v3)`:
//!   v_23: lo=[V8..V15, 0..0]   hi=[V24..V31, 0..0]
//!
//! `_mm256_unpacklo_epi64(v_01, v_23)` (per-lane) interleaves the low
//! 64-bit chunks:
//!   v_vec: lo=[V0..V15]        hi=[V16..V31]
//!
//! Crucially, lane n of `v_vec` is byte V from pixel n in *natural*
//! order — no trailing `_mm256_permute4x64_epi64` is needed, and adding
//! one would scramble the result. This is the post-fix XV36 pattern
//! lifted from u16 to u8.
//!
//! Each combined channel vector is then zero-extended to two `i16x16`
//! halves via `_mm256_cvtepu8_epi16` on the low/high 128-bit lanes,
//! after which the Q15 pipeline (chroma bias subtract, c_scale, R/G/B
//! coeff multiply, Y scale, saturating add, packus) is byte-identical
//! to the NV24 / packed YUV422 AVX2 siblings.
//!
//! α handling: when `ALPHA && ALPHA_SRC`, the A vector from the
//! deinterleave is passed straight through. When `ALPHA && !ALPHA_SRC`,
//! `_mm256_set1_epi8(-1)` (= 0xFF) is used. RGB output uses
//! `write_rgb_32`; RGBA uses `write_rgba_32`.
//!
//! ## Tail
//!
//! `width % 32` remaining pixels fall through to
//! `scalar::vuya_to_rgb_or_rgba_row`.
use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Deinterleave helper ------------------------------------------------

/// Deinterleaves 32 VUYA quadruples (128 bytes = 32 pixels) from `ptr`
/// into `(v_vec, u_vec, y_vec, a_vec)` — four `__m256i` vectors each
/// holding 32 `u8` samples in **natural pixel order** (lane n = byte
/// from pixel n).
///
/// ## Strategy
///
/// 1. Four contiguous `_mm256_loadu_si256` loads fetch 32 pixels' worth
///    of V/U/Y/A (128 bytes).
/// 2. Four `_mm256_permute2x128_si256` calls reshape the contiguous
///    loads into the strided lane layout the per-lane shuffle / unpack
///    cascade expects (each result holds two 4-pixel groups, but with
///    pixels from different halves of the original 32-pixel block in
///    its lo / hi 128-bit lanes).
/// 3. Per-lane `_mm256_shuffle_epi8` masks extract V/U/Y/A bytes from
///    each 128-bit lane into its low 4 bytes.
/// 4. Per-lane `_mm256_unpacklo_epi32` interleaves the 4-byte chunks
///    from pairs of registers, producing 8 valid bytes per lane.
/// 5. Per-lane `_mm256_unpacklo_epi64` combines the two 8-byte halves
///    into a full 16-byte channel chunk per lane — i.e. 32 bytes of
///    natural-order channel samples per `__m256i`.
///
/// Because the cross-lane reshape in step 2 placed pixels 0..15 in the
/// low 128-bit lane and 16..31 in the high 128-bit lane of each
/// downstream register, no `_mm256_permute4x64_epi64` lane-fixup is
/// needed at the end.
///
/// # Safety
///
/// `ptr` must point to at least 128 readable bytes (32 VUYA quadruples).
/// Caller's `target_feature` must include AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_vuya_avx2(ptr: *const u8) -> (__m256i, __m256i, __m256i, __m256i) {
  // SAFETY: caller obligation — `ptr` has 128 bytes readable; AVX2 is
  // available.
  unsafe {
    // Load 4 × __m256i contiguously (32 pixels × 4 channels × u8 = 128 bytes).
    //
    // Each load covers 8 contiguous pixels:
    //   raw_c0 lo=P0..3   hi=P4..7
    //   raw_c1 lo=P8..11  hi=P12..15
    //   raw_c2 lo=P16..19 hi=P20..23
    //   raw_c3 lo=P24..27 hi=P28..31
    let raw_c0 = _mm256_loadu_si256(ptr.cast());
    let raw_c1 = _mm256_loadu_si256(ptr.add(32).cast());
    let raw_c2 = _mm256_loadu_si256(ptr.add(64).cast());
    let raw_c3 = _mm256_loadu_si256(ptr.add(96).cast());

    // Reshape via cross-lane permute so each register holds the layout
    // the per-128-bit-lane cascade below expects:
    //   raw0 lo=P0..3   hi=P16..19   (lo halves of c0 and c2)
    //   raw1 lo=P4..7   hi=P20..23   (hi halves of c0 and c2)
    //   raw2 lo=P8..11  hi=P24..27   (lo halves of c1 and c3)
    //   raw3 lo=P12..15 hi=P28..31   (hi halves of c1 and c3)
    //
    // `_mm256_permute2x128_si256::<imm>` selects 128-bit halves: imm=0x20
    // picks src1 lo + src2 lo; imm=0x31 picks src1 hi + src2 hi.
    let raw0 = _mm256_permute2x128_si256::<0x20>(raw_c0, raw_c2);
    let raw1 = _mm256_permute2x128_si256::<0x31>(raw_c0, raw_c2);
    let raw2 = _mm256_permute2x128_si256::<0x20>(raw_c1, raw_c3);
    let raw3 = _mm256_permute2x128_si256::<0x31>(raw_c1, raw_c3);

    // Shuffle masks: replicate the per-lane VUYA byte gather across both
    // 128-bit halves. Within each 16-byte lane, gather bytes at the
    // channel's offsets (V at 0/4/8/12; U at 1/5/9/13; Y at 2/6/10/14;
    // A at 3/7/11/15) into the low 4 bytes; -1 zeroes the upper 12.
    let v_mask = _mm256_setr_epi8(
      0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );
    let u_mask = _mm256_setr_epi8(
      1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );
    let y_mask = _mm256_setr_epi8(
      2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );
    let a_mask = _mm256_setr_epi8(
      3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );

    // Apply masks: each result has 4 valid bytes in the low 4 lanes of
    // each 128-bit half, with 12 zero bytes filling the upper 12 of
    // each half.
    //
    // For V (analogous for U/Y/A):
    //   v0: lo=[V0,V1,V2,V3, 0..]  hi=[V16,V17,V18,V19, 0..]
    //   v1: lo=[V4..V7, 0..]       hi=[V20..V23, 0..]
    //   v2: lo=[V8..V11, 0..]      hi=[V24..V27, 0..]
    //   v3: lo=[V12..V15, 0..]     hi=[V28..V31, 0..]
    let v0 = _mm256_shuffle_epi8(raw0, v_mask);
    let v1 = _mm256_shuffle_epi8(raw1, v_mask);
    let v2 = _mm256_shuffle_epi8(raw2, v_mask);
    let v3 = _mm256_shuffle_epi8(raw3, v_mask);

    let u0 = _mm256_shuffle_epi8(raw0, u_mask);
    let u1 = _mm256_shuffle_epi8(raw1, u_mask);
    let u2 = _mm256_shuffle_epi8(raw2, u_mask);
    let u3 = _mm256_shuffle_epi8(raw3, u_mask);

    let y0 = _mm256_shuffle_epi8(raw0, y_mask);
    let y1 = _mm256_shuffle_epi8(raw1, y_mask);
    let y2 = _mm256_shuffle_epi8(raw2, y_mask);
    let y3 = _mm256_shuffle_epi8(raw3, y_mask);

    let a0 = _mm256_shuffle_epi8(raw0, a_mask);
    let a1 = _mm256_shuffle_epi8(raw1, a_mask);
    let a2 = _mm256_shuffle_epi8(raw2, a_mask);
    let a3 = _mm256_shuffle_epi8(raw3, a_mask);

    // Step 1: combine 4-byte chunks via per-lane unpacklo_epi32.
    // `_mm256_unpacklo_epi32(a, b)` per 128-bit lane interleaves the
    // low two i32 chunks of each operand:
    //   v_01 lo = [V0V1V2V3 (i32 from v0 lo), V4V5V6V7 (i32 from v1 lo),
    //              0, 0]
    //          = bytes [V0..V7, 0..0]
    //   v_01 hi = bytes [V16..V23, 0..0]
    //   v_23 lo = bytes [V8..V15, 0..0]
    //   v_23 hi = bytes [V24..V31, 0..0]
    let v_01 = _mm256_unpacklo_epi32(v0, v1);
    let v_23 = _mm256_unpacklo_epi32(v2, v3);
    let u_01 = _mm256_unpacklo_epi32(u0, u1);
    let u_23 = _mm256_unpacklo_epi32(u2, u3);
    let y_01 = _mm256_unpacklo_epi32(y0, y1);
    let y_23 = _mm256_unpacklo_epi32(y2, y3);
    let a_01 = _mm256_unpacklo_epi32(a0, a1);
    let a_23 = _mm256_unpacklo_epi32(a2, a3);

    // Step 2: combine the two 8-byte halves into a full 16-byte channel
    // chunk per 128-bit lane via per-lane unpacklo_epi64. Result has
    // 16 bytes of natural-order channel samples per lane:
    //   v_vec lo = [V0..V15]
    //   v_vec hi = [V16..V31]
    // i.e. lane n of v_vec is V from pixel n.
    //
    // No `_mm256_permute4x64_epi64` lane fixup is needed because the
    // cross-lane reshape at the start placed pixels 0..15 in the lo
    // 128-bit lane and 16..31 in the hi 128-bit lane of each register.
    let v_vec = _mm256_unpacklo_epi64(v_01, v_23);
    let u_vec = _mm256_unpacklo_epi64(u_01, u_23);
    let y_vec = _mm256_unpacklo_epi64(y_01, y_23);
    let a_vec = _mm256_unpacklo_epi64(a_01, a_23);

    (v_vec, u_vec, y_vec, a_vec)
  }
}

// ---- Shared RGB / RGBA kernel (32 px/iter) ------------------------------

/// AVX2 VUYA / VUYX → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// Block size: 32 pixels per SIMD iteration (four `_mm256_loadu_si256`
/// loads, 128 bytes total).
///
/// The three valid monomorphizations are:
/// - `<false, false>` — RGB (drops α)
/// - `<true, true>`  — RGBA, source α pass-through (VUYA)
/// - `<true, false>` — RGBA, force α = `0xFF` (VUYX)
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
pub(crate) unsafe fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  let bias = scalar::chroma_bias::<8>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    // 0xFF for VUYX forced-opaque path.
    let alpha_u8 = _mm256_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 32 <= width {
      // Deinterleave 32 VUYA quadruples → V, U, Y, A as u8x32 in
      // natural pixel order.
      let (v_u8, u_u8, y_u8, a_u8) = deinterleave_vuya_avx2(packed.as_ptr().add(x * 4));

      // Zero-extend each channel to two i16x16 halves (low 16 bytes →
      // pixels 0..15, high 16 bytes → pixels 16..31).
      let v_lo_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_u8));
      let v_hi_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v_u8));
      let u_lo_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_u8));
      let u_hi_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(u_u8));
      let y_lo_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_u8));
      let y_hi_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_u8));

      // Subtract chroma bias (128 for 8-bit).
      let u_lo_sub = _mm256_sub_epi16(u_lo_i16, bias_v);
      let u_hi_sub = _mm256_sub_epi16(u_hi_i16, bias_v);
      let v_lo_sub = _mm256_sub_epi16(v_lo_i16, bias_v);
      let v_hi_sub = _mm256_sub_epi16(v_hi_i16, bias_v);

      // Widen each i16x16 chroma half into two i32x8 halves for Q15
      // multiply: u_lo_a = pixels 0..7, u_lo_b = 8..15, u_hi_a = 16..23,
      // u_hi_b = 24..31.
      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_sub));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_sub));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_sub));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_sub));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_sub));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_sub));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_sub));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_sub));

      // u_d / v_d = (u * c_scale + RND) >> 15.
      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
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
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      // 32 chroma per channel: two `chroma_i16x16` calls per channel
      // (no chroma duplication at 4:4:4 — one chroma sample per Y pixel).
      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      // Y path: scale each i16x16 half independently.
      let y_scaled_lo = scale_y(y_lo_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel, then narrow to u8x32
      // with natural lane order (the `narrow_u8x32` helper applies the
      // `permute4x64<0xD8>` post-pack lane fixup).
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x32(r_lo, r_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let b_u8 = narrow_u8x32(b_lo, b_hi);

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        let a_vec = if ALPHA_SRC { a_u8 } else { alpha_u8 };
        write_rgba_32(r_u8, g_u8, b_u8, a_vec, out_ptr);
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        &packed[x * 4..],
        &mut out[x * bpp..],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Thin wrappers ------------------------------------------------------

/// AVX2 VUYA / VUYX → packed **RGB** (3 bpp). Alpha byte in source is
/// discarded — RGB output has no alpha channel.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 VUYA → packed **RGBA** (4 bpp). Source A byte is passed through
/// verbatim.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuya_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// AVX2 VUYX → packed **RGBA** (4 bpp). Source A byte is padding;
/// output α is forced to `0xFF` (opaque).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuyx_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- Luma extraction (32 px/iter) ---------------------------------------

/// AVX2 VUYA / VUYX → u8 luma. Y is the third byte (offset 2) of each
/// pixel quadruple.
///
/// Byte-identical to `scalar::vuya_to_luma_row`.
///
/// Block size: 32 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      // Reuse the full 4-channel deinterleave and discard V/U/A. The
      // compiler lifts the dead shuffles; keeping the same code path
      // gives the lane-order regression test the strongest possible
      // coverage: any deinterleave bug in the V/U/Y path manifests
      // identically here.
      let (_v, _u, y_vec, _a) = deinterleave_vuya_avx2(packed.as_ptr().add(x * 4));
      _mm256_storeu_si256(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      scalar::vuya_to_luma_row(&packed[x * 4..], &mut luma_out[x..], width - x);
    }
  }
}
