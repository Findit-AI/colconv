//! AVX-512 kernels for XV48 packed YUV 4:4:4 16-bit family
//! (FFmpeg `AV_PIX_FMT_XV48LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), X(16)]`, each
//! holding a full 16-bit sample (no padding bits, no right-shift on
//! load — the full-depth sibling of XV36). The `X` slot is **padding** —
//! loaded but discarded. RGBA outputs force α = max.
//!
//! ## Per-iter pipeline (64 px / iter for u8, 32 px / iter for u16)
//!
//! Both paths use the same 32-pixel deinterleave helper (two rounds of
//! `_mm512_permutex2var_epi16`, `vpermt2w` from AVX-512BW — slots U=0,
//! Y=1, V=2, X=3). The u8 path runs it twice per main-loop iteration;
//! the u16 path runs it once.
//!
//! - u8 output: chroma centered (subtract 32768 via wrapping
//!   `-32768i16`), Q15 chroma scale via `chroma_i16x32`. Y scaled via
//!   `scale_y_u16_avx512`.
//!
//! - u16 output: i64 chroma via `chroma_i64x8_avx512` to avoid i32
//!   overflow at BITS=16/16. Y scaled via `scale_y_i32x16_i64`.
//!
//! ## Tail
//!
//! `width % block_size` remaining pixels fall through to `scalar::xv48_*`.

use super::{endian, *};
use crate::{KernelMatrix, row::scalar};

// ---- Static permute index tables ----------------------------------------
//
// XV48 layout per pixel: [U, Y, V, X] (4 u16 per pixel). Within one
// __m512i, pixel p occupies lanes [4p, 4p+1, 4p+2, 4p+3] = [U, Y, V, X].
// Channel offsets: U=0, Y=+1, V=+2, X=+3.

// Round-1 index: pick U channel (offset 0 within each pixel group of 4).
#[rustfmt::skip]
static U_FROM_PAIR_IDX: [i16; 32] = [
  // U from v0 (lanes 0..8 of output).
   0,  4,  8, 12, 16, 20, 24, 28,
  // U from v1 (lanes 8..16 of output), idx >= 32.
  32, 36, 40, 44, 48, 52, 56, 60,
  // Don't-care lanes 16..32: safe index 0.
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
];

// Round-1 index: pick Y channel (offset +1 within each pixel group of 4).
#[rustfmt::skip]
static Y_FROM_PAIR_IDX: [i16; 32] = [
  // Y from v0 (lanes 0..8 of output).
   1,  5,  9, 13, 17, 21, 25, 29,
  // Y from v1 (lanes 8..16 of output), idx >= 32.
  33, 37, 41, 45, 49, 53, 57, 61,
  // Don't-care lanes 16..32: safe index 1.
   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
];

// Round-1 index: pick V channel (offset +2 within each pixel group of 4).
#[rustfmt::skip]
static V_FROM_PAIR_IDX: [i16; 32] = [
  // V from v0 (lanes 0..8 of output).
   2,  6, 10, 14, 18, 22, 26, 30,
  // V from v1 (lanes 8..16 of output), idx >= 32.
  34, 38, 42, 46, 50, 54, 58, 62,
  // Don't-care lanes 16..32: safe index 2.
   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
];

// Round-2 index: combine two 16-value half-vectors into a full 32-lane vector.
#[rustfmt::skip]
static COMBINE_IDX: [i16; 32] = [
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
];

// ---- Deinterleave helper (32 pixels / 128 u16 / 256 bytes) --------------

/// Deinterleaves 32 XV48 quadruples (128 u16 = 256 bytes) from `ptr`
/// into `(u_vec, y_vec, v_vec)` — three `__m512i` vectors each holding
/// 32 `u16` samples in **natural pixel order** (lane n = u16 from pixel
/// n). Channel slot order in source: U=0, Y=1, V=2, X=3. No shift
/// (16-bit native). The X channel (slot 3) is not extracted.
///
/// Cross-lane primitive `vpermt2w` is part of AVX-512BW — no
/// AVX-512VBMI required.
///
/// # Safety
///
/// `ptr` must point to at least 256 readable bytes (128 `u16`
/// elements). Caller's `target_feature` must include AVX-512F +
/// AVX-512BW.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn deinterleave_xv48_32px_avx512<const BE: bool>(
  ptr: *const u16,
) -> (__m512i, __m512i, __m512i) {
  // SAFETY: caller obligation — `ptr` has 256 bytes readable; AVX-512F +
  // AVX-512BW are available.
  unsafe {
    let v0 = endian::load_endian_u16x32::<BE>(ptr as *const u8);
    let v1 = endian::load_endian_u16x32::<BE>(ptr.add(32) as *const u8);
    let v2 = endian::load_endian_u16x32::<BE>(ptr.add(64) as *const u8);
    let v3 = endian::load_endian_u16x32::<BE>(ptr.add(96) as *const u8);

    let u_idx = _mm512_loadu_si512(U_FROM_PAIR_IDX.as_ptr().cast());
    let y_idx = _mm512_loadu_si512(Y_FROM_PAIR_IDX.as_ptr().cast());
    let v_idx_tbl = _mm512_loadu_si512(V_FROM_PAIR_IDX.as_ptr().cast());
    let comb_idx = _mm512_loadu_si512(COMBINE_IDX.as_ptr().cast());

    // Round 1: gather U / Y / V from each pair of __m512i vectors.
    let u_01 = _mm512_permutex2var_epi16(v0, u_idx, v1); // U for pixels  0..15
    let u_23 = _mm512_permutex2var_epi16(v2, u_idx, v3); // U for pixels 16..31
    let y_01 = _mm512_permutex2var_epi16(v0, y_idx, v1); // Y for pixels  0..15
    let y_23 = _mm512_permutex2var_epi16(v2, y_idx, v3); // Y for pixels 16..31
    let v_01 = _mm512_permutex2var_epi16(v0, v_idx_tbl, v1); // V for pixels  0..15
    let v_23 = _mm512_permutex2var_epi16(v2, v_idx_tbl, v3); // V for pixels 16..31

    // Round 2: combine the two 16-value half-vectors into 32-lane channel
    // vectors.
    let u_vec = _mm512_permutex2var_epi16(u_01, comb_idx, u_23);
    let y_vec = _mm512_permutex2var_epi16(y_01, comb_idx, y_23);
    let v_vec = _mm512_permutex2var_epi16(v_01, comb_idx, v_23);

    (u_vec, y_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (64 px/iter) ----------------------------------

/// AVX-512 XV48 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv48_to_rgb_or_rgba_row::<ALPHA, BE>`.
///
/// Block size: 64 pixels per SIMD iteration (two 32-pixel deinterleaves).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    // X is padding — RGBA forces α = 0xFF.
    let alpha_u8 = _mm512_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 64 <= width {
      // --- lo half: pixels x..x+31 (one 32-pixel deinterleave) ----------
      let (u_lo_u16, y_lo_u16, v_lo_u16) =
        deinterleave_xv48_32px_avx512::<BE>(packed.as_ptr().add(x * 4));

      let u_lo_i16 = _mm512_sub_epi16(u_lo_u16, bias16_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_u16, bias16_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );

      // Y: full u16 values → use scale_y_u16_avx512 (NOT scale_y).
      let y_lo_scaled = scale_y_u16_avx512(y_lo_u16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // --- hi half: pixels x+32..x+63 (one more 32-pixel deinterleave) --
      let (u_hi_u16, y_hi_u16, v_hi_u16) =
        deinterleave_xv48_32px_avx512::<BE>(packed.as_ptr().add(x * 4 + 128));

      let u_hi_i16 = _mm512_sub_epi16(u_hi_u16, bias16_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_u16, bias16_v);

      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_hi_scaled = scale_y_u16_avx512(y_hi_u16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating add Y + chroma per channel; narrow both halves to u8x64.
      let r_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, r_chroma_lo),
        _mm512_adds_epi16(y_hi_scaled, r_chroma_hi),
        pack_fixup,
      );
      let g_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, g_chroma_lo),
        _mm512_adds_epi16(y_hi_scaled, g_chroma_hi),
        pack_fixup,
      );
      let b_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, b_chroma_lo),
        _mm512_adds_epi16(y_hi_scaled, b_chroma_hi),
        pack_fixup,
      );

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out_ptr);
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 64;
    }

    // Scalar tail — remaining < 64 pixels.
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

// ---- u16 RGB / RGBA native-depth output (32 px/iter) --------------------

/// AVX-512 XV48 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x8_avx512`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>`.
///
/// Block size: 32 pixels per SIMD iteration (one 32-pixel deinterleave).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let alpha_u16_v = _mm_set1_epi16(-1i16); // 0xFFFF
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // Deinterleave 32 XV48 quadruples → U, Y, V as u16x32 (natural order).
      let (u_u16, y_vec, v_u16) = deinterleave_xv48_32px_avx512::<BE>(packed.as_ptr().add(x * 4));

      let u_i16 = _mm512_sub_epi16(u_u16, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_u16, bias16_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_i32_v,
      ));

      // i64 chroma: even/odd i32 lanes via 0xF5 shuffle.
      let u_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(u_d_lo);
      let v_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(v_d_lo);
      let u_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_even = chroma_i64x8_avx512(cru, crv, u_d_lo, v_d_lo, rnd_i64_v);
      let r_ch_lo_odd = chroma_i64x8_avx512(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let g_ch_lo_even = chroma_i64x8_avx512(cgu, cgv, u_d_lo, v_d_lo, rnd_i64_v);
      let g_ch_lo_odd = chroma_i64x8_avx512(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let b_ch_lo_even = chroma_i64x8_avx512(cbu, cbv, u_d_lo, v_d_lo, rnd_i64_v);
      let b_ch_lo_odd = chroma_i64x8_avx512(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);

      let r_ch_hi_even = chroma_i64x8_avx512(cru, crv, u_d_hi, v_d_hi, rnd_i64_v);
      let r_ch_hi_odd = chroma_i64x8_avx512(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let g_ch_hi_even = chroma_i64x8_avx512(cgu, cgv, u_d_hi, v_d_hi, rnd_i64_v);
      let g_ch_hi_odd = chroma_i64x8_avx512(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let b_ch_hi_even = chroma_i64x8_avx512(cbu, cbv, u_d_hi, v_d_hi, rnd_i64_v);
      let b_ch_hi_odd = chroma_i64x8_avx512(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);

      // Reassemble each pair of i64x8 → i32x16.
      let r_ch_lo_i32 = reassemble_i32x16(r_ch_lo_even, r_ch_lo_odd, interleave_idx);
      let g_ch_lo_i32 = reassemble_i32x16(g_ch_lo_even, g_ch_lo_odd, interleave_idx);
      let b_ch_lo_i32 = reassemble_i32x16(b_ch_lo_even, b_ch_lo_odd, interleave_idx);
      let r_ch_hi_i32 = reassemble_i32x16(r_ch_hi_even, r_ch_hi_odd, interleave_idx);
      let g_ch_hi_i32 = reassemble_i32x16(g_ch_hi_even, g_ch_hi_odd, interleave_idx);
      let b_ch_hi_i32 = reassemble_i32x16(b_ch_hi_even, b_ch_hi_odd, interleave_idx);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Add Y + chroma in i32; saturate-narrow to u16 via _mm512_packus_epi32
      // + pack_fixup.
      let r_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, r_ch_lo_i32),
          _mm512_add_epi32(y_hi_scaled, r_ch_hi_i32),
        ),
      );
      let g_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, g_ch_lo_i32),
          _mm512_add_epi32(y_hi_scaled, g_ch_hi_i32),
        ),
      );
      let b_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, b_ch_lo_i32),
          _mm512_add_epi32(y_hi_scaled, b_ch_hi_i32),
        ),
      );

      // Write 32 pixels.
      if ALPHA {
        // X is padding — RGBA forces α = 0xFFFF on every pixel.
        let dst = out.as_mut_ptr().add(x * 4);
        let r0: __m128i = _mm512_castsi512_si128(r_u16);
        let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r_u16);
        let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r_u16);
        let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r_u16);
        let g0: __m128i = _mm512_castsi512_si128(g_u16);
        let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g_u16);
        let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g_u16);
        let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g_u16);
        let b0: __m128i = _mm512_castsi512_si128(b_u16);
        let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b_u16);
        let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b_u16);
        let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b_u16);
        // Each `write_rgba_u16_8` writes 8 pixels × 4 × u16 = 32 u16 elements.
        write_rgba_u16_8(r0, g0, b0, alpha_u16_v, dst);
        write_rgba_u16_8(r1, g1, b1, alpha_u16_v, dst.add(32));
        write_rgba_u16_8(r2, g2, b2, alpha_u16_v, dst.add(64));
        write_rgba_u16_8(r3, g3, b3, alpha_u16_v, dst.add(96));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
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

// ---- Luma u8 (32 px/iter) -----------------------------------------------

/// AVX-512 XV48 → u8 luma. Y is quadruple element 1; `>> 8` extracts the
/// high byte. Reuses the full deinterleave helper and discards U/V.
///
/// Byte-identical to `scalar::xv48_to_luma_row`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xv48_to_luma_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let zero = _mm512_setzero_si512();

    let mut x = 0usize;
    while x + 32 <= width {
      let (_u, y_vec, _v) = deinterleave_xv48_32px_avx512::<BE>(packed.as_ptr().add(x * 4));

      let y_shr = _mm512_srli_epi16::<8>(y_vec);
      let y_u8 = narrow_u8x64(y_shr, zero, pack_fixup);

      // Store low 32 bytes (the valid Y values) via the low 256-bit half.
      _mm256_storeu_si256(
        luma_out.as_mut_ptr().add(x).cast(),
        _mm512_castsi512_si256(y_u8),
      );

      x += 32;
    }

    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_row::<BE>(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- Luma u16 (32 px/iter) ----------------------------------------------

/// AVX-512 XV48 → u16 luma. Direct copy of Y samples (slot 1, no shift —
/// 16-bit native). Reuses the full deinterleave helper and discards U/V.
///
/// Byte-identical to `scalar::xv48_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xv48_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let (_u, y_vec, _v) = deinterleave_xv48_32px_avx512::<BE>(packed.as_ptr().add(x * 4));
      // Direct store — Y samples are 16-bit native, in natural pixel order.
      _mm512_storeu_si512(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 32;
    }

    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_u16_row::<BE>(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- XV48 → HSV (staged via a reused 8-bit RGB chunk) ----------------
//
// The SIMD twin of the scalar `xv48_to_hsv_row` kernel — fills a small
// reused 8-bit RGB scratch via the production `xv48_to_rgb_row` kernel,
// then runs `rgb_to_hsv_row`. The X slot is dropped (HSV is colour-only).

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
#[target_feature(enable = "avx512f,avx512bw")]
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
