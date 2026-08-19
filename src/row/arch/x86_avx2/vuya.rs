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

use super::*;
use crate::{KernelMatrix, row::scalar};

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
/// Build a per-lane gather mask for one channel at byte offset `OFF`
/// within each 4-byte pixel (`OFF, OFF+4, OFF+8, OFF+12` into the low 4
/// bytes of each 128-bit lane; `-1` zeroes the rest). The const offset
/// makes the mask a compile-time constant.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn chan_mask_avx2<const OFF: usize>() -> __m256i {
  let o = OFF as i8;
  _mm256_setr_epi8(
    o,
    o + 4,
    o + 8,
    o + 12,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1, // low lane
    o,
    o + 4,
    o + 8,
    o + 12,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1, // high lane
  )
}

/// Offset-parameterized AVX2 deinterleave of 32 four-byte pixels into
/// `(v_vec, u_vec, y_vec, a_vec)` (each `__m256i` of 32 natural-order
/// channel bytes). The four byte offsets select which source byte feeds
/// each channel, so the single routine serves every channel re-ordering
/// (VUYA/VUYX `0,1,2,3`; AYUV `3,2,1,0`; UYVA `2,0,1,3`).
///
/// # Safety
///
/// `ptr` must point to at least 128 readable bytes (32 pixels). AVX2 must
/// be available.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_packed444_avx2<
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
  const A_OFF: usize,
>(
  ptr: *const u8,
) -> (__m256i, __m256i, __m256i, __m256i) {
  // SAFETY: caller obligation — `ptr` has 128 bytes readable; AVX2 available.
  unsafe {
    let raw_c0 = _mm256_loadu_si256(ptr.cast());
    let raw_c1 = _mm256_loadu_si256(ptr.add(32).cast());
    let raw_c2 = _mm256_loadu_si256(ptr.add(64).cast());
    let raw_c3 = _mm256_loadu_si256(ptr.add(96).cast());

    let raw0 = _mm256_permute2x128_si256::<0x20>(raw_c0, raw_c2);
    let raw1 = _mm256_permute2x128_si256::<0x31>(raw_c0, raw_c2);
    let raw2 = _mm256_permute2x128_si256::<0x20>(raw_c1, raw_c3);
    let raw3 = _mm256_permute2x128_si256::<0x31>(raw_c1, raw_c3);

    let v_mask = chan_mask_avx2::<V_OFF>();
    let u_mask = chan_mask_avx2::<U_OFF>();
    let y_mask = chan_mask_avx2::<Y_OFF>();
    let a_mask = chan_mask_avx2::<A_OFF>();

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

    let v_01 = _mm256_unpacklo_epi32(v0, v1);
    let v_23 = _mm256_unpacklo_epi32(v2, v3);
    let u_01 = _mm256_unpacklo_epi32(u0, u1);
    let u_23 = _mm256_unpacklo_epi32(u2, u3);
    let y_01 = _mm256_unpacklo_epi32(y0, y1);
    let y_23 = _mm256_unpacklo_epi32(y2, y3);
    let a_01 = _mm256_unpacklo_epi32(a0, a1);
    let a_23 = _mm256_unpacklo_epi32(a2, a3);

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
pub(crate) unsafe fn packed444_to_rgb_or_rgba_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
  const A_OFF: usize,
>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
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
    // 0xFF for the forced-opaque path.
    let alpha_u8 = _mm256_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 32 <= width {
      // Deinterleave 32 quadruples → V, U, Y, A as u8x32 in natural pixel
      // order per the format's byte offsets.
      let (v_u8, u_u8, y_u8, a_u8) =
        deinterleave_packed444_avx2::<V_OFF, U_OFF, Y_OFF, A_OFF>(packed.as_ptr().add(x * 4));

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
      scalar::packed444_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC, V_OFF, U_OFF, Y_OFF, A_OFF>(
        &packed[x * 4..],
        &mut out[x * bpp..],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// VUYA / VUYX byte order (`V=0,U=1,Y=2,A=3`) over the offset-generic AVX2
/// kernel.
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC, 0, 1, 2, 3>(
      packed, out, width, matrix, full_range,
    );
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
pub(crate) unsafe fn packed444_to_luma_row<const Y_OFF: usize>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      // Reuse the full 4-channel deinterleave and keep only Y (at `Y_OFF`).
      // The compiler lifts the dead shuffles; keeping the same code path
      // gives the lane-order regression test the strongest coverage.
      let (_v, _u, y_vec, _a) =
        deinterleave_packed444_avx2::<0, 1, Y_OFF, 3>(packed.as_ptr().add(x * 4));
      _mm256_storeu_si256(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      scalar::packed444_to_luma_row::<Y_OFF>(&packed[x * 4..], &mut luma_out[x..], width - x);
    }
  }
}

/// VUYA / VUYX u8 luma (Y at offset 2) over [`packed444_to_luma_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<2>(packed, luma_out, width);
  }
}

/// AVX2 VUYA → u16 luma (zero-extended Y bytes). Y is the third byte
/// (offset 2) of each pixel quadruple. 16 pixels per SIMD iteration.
///
/// Strategy: reuse the 4-channel deinterleave to get a `__m256i` of 32
/// Y u8 bytes. The low 128-bit lane (pixels 0-15) is zero-extended to
/// u16x16 via `_mm256_cvtepu8_epi16`; the high lane (pixels 16-31) is
/// extracted and widened the same way. Two `_mm256_storeu_si256` writes
/// produce 32 u16 values.
///
/// Byte-identical to `scalar::vuya_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn packed444_to_luma_u16_row<const Y_OFF: usize>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(out.len() >= width, "out too short");

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      // Deinterleave 32 quadruples; keep only Y (at `Y_OFF`), u8x32 natural.
      let (_v, _u, y_u8, _a) =
        deinterleave_packed444_avx2::<0, 1, Y_OFF, 3>(packed.as_ptr().add(x * 4));

      // Widen low 16 Y bytes → u16x16 (pixels 0-15).
      let lo_u16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_u8));
      // Widen high 16 Y bytes → u16x16 (pixels 16-31).
      let hi_u16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_u8));

      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), lo_u16);
      _mm256_storeu_si256(out.as_mut_ptr().add(x + 16).cast(), hi_u16);
      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      scalar::packed444_to_luma_u16_row::<Y_OFF>(&packed[x * 4..], &mut out[x..], width - x);
    }
  }
}

/// VUYA u16 luma (Y at offset 2) over [`packed444_to_luma_u16_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuya_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<2>(packed, out, width);
  }
}

/// AVX2 VUYX → u16 luma (zero-extended Y bytes). Byte-identical to
/// [`vuya_to_luma_u16_row`] — Y is at byte offset 2 of each quadruple
/// regardless of α semantics; the X byte is discarded.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuyx_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    vuya_to_luma_u16_row(packed, out, width);
  }
}

// ---- VUYA → HSV (staged via a reused 8-bit RGB chunk) ----------------
//
// The SIMD twin of the scalar `vuya_to_hsv_row` kernel. Rather than
// re-derive an HSV-specific register pipeline, it fills a small fixed
// reused 8-bit RGB scratch (one `HSV_CHUNK`-pixel chunk at a time)
// using the EXISTING vuya_to_rgb_row kernel of this file — so the chunk
// filler IS the production 8-bit RGB kernel — then runs the SIMD
// `rgb_to_hsv_row` on the chunk. This makes the result byte-identical to
// `rgb_to_hsv_row(vuya_to_rgb_row(...))` within this SIMD tier with no source-width RGB allocation. The
// scalar tail of the underlying RGB kernel handles widths below the SIMD
// block, so no separate tail is needed here. The α byte (slot 3) is
// dropped by the RGB kernel — HSV is colour-only — so a single kernel
// serves both VUYA (real α) and VUYX (padding); `vuyx_to_hsv_row` is a
// thin re-export.
//
// The chunked driver is defined locally (mirroring the semi-planar
// high-bit `pn_hsv_via_rgb_chunks`) and gated `yuv-444-packed` with the
// rest of this file. Only `rgb_to_hsv_row` (ungated) is shared.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared SIMD driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing SIMD RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the SIMD [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. Byte-identical to `rgb_to_hsv_row(vuya_to_rgb_row(...))` within this SIMD tier, with no
/// source-width RGB allocation.
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
unsafe fn vuya_hsv_via_rgb_chunks(
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

/// SIMD: VUYA (packed 4:4:4, 8-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the reused-8-bit-RGB-chunk pattern over the
/// SIMD [`vuya_to_rgb_row`] + [`rgb_to_hsv_row`]. Byte-identical to
/// `rgb_to_hsv_row(vuya_to_rgb_row(...))` within this SIMD tier. The α
/// byte is dropped (HSV is colour-only).
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn packed444_to_hsv_row<
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
>(
  packed: &[u8],
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
  // forwards the per-chunk sub-slices to the offset-generic AVX2 RGB kernel
  // (3-bpp / no alpha) under the same contract. `A_OFF` is unused in the RGB
  // path; reuse `Y_OFF` as the dummy 4th offset.
  unsafe {
    vuya_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      packed444_to_rgb_or_rgba_row::<false, false, V_OFF, U_OFF, Y_OFF, Y_OFF>(
        &packed[offset * 4..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// VUYA / VUYX HSV (`V=0,U=1,Y=2`) over [`packed444_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vuya_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<0, 1, 2>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

// ---- AYUV / UYVA AVX2 wrappers ----------------------------------------

/// AVX2 AYUV (`A=0,Y=1,U=2,V=3`) → packed RGB.
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ayuv_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<false, false, 3, 2, 1, 0>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 AYUV → packed RGBA (source α at offset 0).
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ayuv_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<true, true, 3, 2, 1, 0>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX2 AYUV → planar HSV bytes.
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ayuv_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<3, 2, 1>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// AVX2 AYUV → u8 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ayuv_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<1>(packed, luma_out, width);
  }
}

/// AVX2 AYUV → u16 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ayuv_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<1>(packed, out, width);
  }
}

/// AVX2 UYVA (`U=0,Y=1,V=2,A=3`) → packed RGB.
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyva_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<false, false, 2, 0, 1, 3>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 UYVA → packed RGBA (source α at offset 3).
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyva_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<true, true, 2, 0, 1, 3>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX2 UYVA → planar HSV bytes.
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyva_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<2, 0, 1>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// AVX2 UYVA → u8 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyva_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<1>(packed, luma_out, width);
  }
}

/// AVX2 UYVA → u16 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyva_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<1>(packed, out, width);
  }
}

// ---- VYU444 (V=0, Y=1, U=2; 3 bytes per pixel, no alpha) AVX2 ----------
//
// VYU444 is a 24bpp (3-byte) format. A 3-byte de-interleave does not tile
// onto 256-bit lanes (the 3-byte stride is co-prime with the 32-byte lane
// width), so — exactly like the AVX2 RGB-input HSV / luma kernels, which
// reuse the 16-pixel SSE-width `x86_common::deinterleave_rgb_16` /
// `rgb_to_hsv_16_pixels` helpers — the AVX2 VYU444 entry points dispatch
// to the SSE4.1 16-px 3-byte-de-interleave kernel. AVX2 is a strict
// superset of SSE4.1, so the call is sound; the de-interleave (the
// throughput floor for this layout) runs at its natural 128-bit width.
// Byte-identical to the scalar reference and to the NEON `vld3q_u8` tier.

/// AVX2 VYU444 → packed RGB (via the SSE4.1 16-px 3-byte kernel).
///
/// # Safety
///
/// `packed.len() >= width * 3`; `rgb_out.len() >= width * 3`. AVX2 (⊇
/// SSE4.1) must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vyu444_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    crate::row::arch::x86_sse41::vyu444_to_rgb_row(packed, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 VYU444 → packed RGBA (α forced `0xFF`; via the SSE4.1 kernel).
///
/// # Safety
///
/// `packed.len() >= width * 3`; `rgba_out.len() >= width * 4`. AVX2 (⊇
/// SSE4.1) must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vyu444_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    crate::row::arch::x86_sse41::vyu444_to_rgba_row(packed, rgba_out, width, matrix, full_range);
  }
}

/// AVX2 VYU444 → planar HSV bytes (via the SSE4.1 kernel).
///
/// # Safety
///
/// `packed.len() >= width * 3`; each H/S/V plane `>= width`. AVX2 (⊇
/// SSE4.1) must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vyu444_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    crate::row::arch::x86_sse41::vyu444_to_hsv_row(
      packed, h_out, s_out, v_out, width, matrix, full_range,
    );
  }
}

/// AVX2 VYU444 → u8 luma (Y at offset 1, 3-byte stride; via the SSE4.1
/// kernel).
///
/// # Safety
///
/// `packed.len() >= width * 3`; `luma_out.len() >= width`. AVX2 (⊇ SSE4.1)
/// must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vyu444_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    crate::row::arch::x86_sse41::vyu444_to_luma_row(packed, luma_out, width);
  }
}

/// AVX2 VYU444 → u16 luma (Y at offset 1, 3-byte stride; via the SSE4.1
/// kernel).
///
/// # Safety
///
/// `packed.len() >= width * 3`; `out.len() >= width`. AVX2 (⊇ SSE4.1) must
/// be available.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vyu444_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    crate::row::arch::x86_sse41::vyu444_to_luma_u16_row(packed, out, width);
  }
}
