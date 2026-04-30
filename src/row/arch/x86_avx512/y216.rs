//! AVX-512 Y216 (packed YUV 4:2:2, BITS=16) kernels.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` where each
//! sample spans the **full 16-bit word** — unlike Y210/Y212 there is no
//! MSB alignment shift.
//!
//! ## u8 pipeline (64 px / iter)
//!
//! Two calls to [`unpack_y216_32px_avx512`] load 32 pixels each (two
//! 512-bit loads → 64 u16 = 128 bytes) and separate Y/U/V using the
//! same `_mm512_permutex2var_epi16` cross-lane trick as `y2xx.rs`, but
//! **without** the right-shift (samples are already at full 16-bit
//! range). Chroma via `chroma_i16x32` (i32 Q15); Y via
//! `scale_y_u16_avx512` (unsigned-widen). Output via `write_rgb_64` /
//! `write_rgba_64`.
//!
//! ## u16 pipeline (32 px / iter)
//!
//! One `unpack_y216_32px_avx512` deinterleave gives 32 Y and 16 UV
//! pairs. Chroma scaled in i64 via `chroma_i64x8_avx512` (native
//! `_mm512_srai_epi64` — no bias trick). Y via `scale_y_i32x16_i64`.
//! Chroma duplicated per `_mm512_permutexvar_epi32`; Y + chroma summed
//! and packed via `_mm512_packus_epi32` + `pack_fixup`. Output via
//! `write_rgb_u16_32` / `write_rgba_u16_32`.
//!
//! ## Luma u8 (64 px / iter)
//!
//! Two 512-bit loads + cross-vector Y permute (`_mm512_permutex2var_epi16`)
//! + `>> 8` + `narrow_u8x64` + 256-bit store.
//!
//! ## Luma u16 (64 px / iter)
//!
//! Two 512-bit loads + cross-vector Y permute + two 512-bit stores.
//!
//! ## Tail
//!
//! `width % 64` (u8/luma) or `width % 32` (u16) → `scalar::y216_*`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Static permute index tables (same layout as y2xx.rs) ---------------

#[rustfmt::skip]
static Y_FROM_YUYV_IDX: [i16; 32] = [
  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
  32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
];

#[rustfmt::skip]
static CHROMA_FROM_YUYV_IDX: [i16; 32] = [
  1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
  33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
];

#[rustfmt::skip]
static U_FROM_UV_IDX: [i16; 32] = [
  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[rustfmt::skip]
static V_FROM_UV_IDX: [i16; 32] = [
  1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// Loads 32 Y216 pixels (64 u16 = 128 bytes) and deinterleaves them
/// into:
/// - `y_vec`:  lanes 0..32 = Y0..Y31 (full 16-bit, no shift).
/// - `u_vec`:  lanes 0..16 = U0..U15 (lanes 16..32 don't-care).
/// - `v_vec`:  lanes 0..16 = V0..V15 (lanes 16..32 don't-care).
///
/// Unlike Y210/Y212 no right-shift is applied — Y216 uses the full u16
/// range.
///
/// # Safety
///
/// `ptr` must have at least 128 readable bytes (64 u16). Caller's
/// `target_feature` must include AVX-512F + AVX-512BW.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn unpack_y216_32px_avx512(ptr: *const u16) -> (__m512i, __m512i, __m512i) {
  // SAFETY: caller obligation.
  unsafe {
    let v0 = _mm512_loadu_si512(ptr.cast());
    let v1 = _mm512_loadu_si512(ptr.add(32).cast());

    let y_idx = _mm512_loadu_si512(Y_FROM_YUYV_IDX.as_ptr().cast());
    let chroma_idx = _mm512_loadu_si512(CHROMA_FROM_YUYV_IDX.as_ptr().cast());

    // Cross-vector u16 gather — no shift for Y216 full-range samples.
    let y_vec = _mm512_permutex2var_epi16(v0, y_idx, v1);
    let chroma = _mm512_permutex2var_epi16(v0, chroma_idx, v1);

    let u_idx = _mm512_loadu_si512(U_FROM_UV_IDX.as_ptr().cast());
    let v_idx = _mm512_loadu_si512(V_FROM_UV_IDX.as_ptr().cast());
    let u_vec = _mm512_permutexvar_epi16(u_idx, chroma);
    let v_vec = _mm512_permutexvar_epi16(v_idx, chroma);

    (y_vec, u_vec, v_vec)
  }
}

// ---- u8 output (64 px / iter) -------------------------------------------

/// AVX-512 Y216 → packed u8 RGB or RGBA. Block size **64 px / iter**.
///
/// Byte-identical to `scalar::y216_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)] // dispatcher lands in follow-up PR (Task 9)
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // range_params_n::<16, 8> — y_off is i32.
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512F + AVX-512BW is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    // Chroma bias: 32768 via wrapping -32768 i16.
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      // --- lo group: pixels x..x+31 (32 pixels) --------------------------
      let (y_lo_vec, u_lo_vec, v_lo_vec) =
        unpack_y216_32px_avx512(packed.as_ptr().add(x * 2));

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias16_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias16_v);

      // Widen 16 valid U/V i16 lanes to two i32x16 halves.
      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(v_lo_b, c_scale_v), rnd_v));

      // chroma_i16x32: 32-lane vector, valid data in lanes 0..16.
      let r_chroma_lo = chroma_i16x32(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup);
      let g_chroma_lo = chroma_i16x32(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup);
      let b_chroma_lo = chroma_i16x32(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup);

      // Duplicate each chroma sample into its 4:2:2 Y-pair slot.
      // 16 valid chroma → lo32 covers all 32 Y lanes.
      let (r_dup_lo, _) = chroma_dup(r_chroma_lo, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, _) = chroma_dup(g_chroma_lo, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, _) = chroma_dup(b_chroma_lo, dup_lo_idx, dup_hi_idx);

      // scale_y_u16_avx512: unsigned-widens Y to avoid i16 overflow for Y > 32767.
      let y_lo_scaled = scale_y_u16_avx512(y_lo_vec, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // --- hi group: pixels x+32..x+63 (32 pixels) ----------------------
      let (y_hi_vec, u_hi_vec, v_hi_vec) =
        unpack_y216_32px_avx512(packed.as_ptr().add(x * 2 + 64));

      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias16_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias16_v);

      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_hi_a = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_hi = chroma_i16x32(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup);
      let g_chroma_hi = chroma_i16x32(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup);
      let b_chroma_hi = chroma_i16x32(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup);

      let (r_dup_hi, _) = chroma_dup(r_chroma_hi, dup_lo_idx, dup_hi_idx);
      let (g_dup_hi, _) = chroma_dup(g_chroma_hi, dup_lo_idx, dup_hi_idx);
      let (b_dup_hi, _) = chroma_dup(b_chroma_hi, dup_lo_idx, dup_hi_idx);

      let y_hi_scaled = scale_y_u16_avx512(y_hi_vec, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating i16 add + narrow to u8x64 per channel.
      let r_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, r_dup_lo),
        _mm512_adds_epi16(y_hi_scaled, r_dup_hi),
        pack_fixup,
      );
      let g_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, g_dup_lo),
        _mm512_adds_epi16(y_hi_scaled, g_dup_hi),
        pack_fixup,
      );
      let b_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, b_dup_lo),
        _mm512_adds_epi16(y_hi_scaled, b_dup_hi),
        pack_fixup,
      );

      if ALPHA {
        let alpha = _mm512_set1_epi8(-1);
        write_rgba_64(r_u8, g_u8, b_u8, alpha, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

    // Scalar tail — remaining < 64 pixels (always even per 4:2:2).
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 output (i64 chroma, 32 px / iter) ------------------------------

/// AVX-512 Y216 → packed native-depth u16 RGB or RGBA. Uses i64 chroma
/// (`chroma_i64x8_avx512`) to avoid overflow at 16-bit scales. Block
/// size **32 px / iter**.
///
/// Byte-identical to `scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[allow(dead_code)] // dispatcher lands in follow-up PR (Task 9)
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512F + AVX-512BW is the caller's obligation.
  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
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

    // Permute indices built once.
    // dup_{lo,hi}_idx: duplicate 16 chroma i32 lanes into 32 slots.
    let dup_lo_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let dup_hi_idx =
      _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    // interleave_idx: even i32x8 + odd i32x8 → i32x16 [e0,o0,e1,o1,...].
    let interleave_idx =
      _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // One deinterleave gives 32 Y + 16 UV pairs.
      let (y_vec, u_vec, v_vec) = unpack_y216_32px_avx512(packed.as_ptr().add(x * 2));

      // Subtract chroma bias (wrapping i16 sub of -32768 = +32768 mod 2^16).
      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

      // Widen 16 valid i16 lanes (low 256 bits) to i32x16 for Q15 scale.
      // High 256 bits of u_vec / v_vec hold don't-care values after the
      // U/V split permute; they won't reach chroma_i64x8_avx512.
      let u_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let v_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));

      // Scale UV in i32: |u_centered| ≤ 32768, |c_scale| ≤ ~38300 →
      // product ≤ ~1.26·10⁹ — fits i32.
      let u_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_i32_v,
      ));

      // i64 chroma: even and odd i32 lanes separately.
      let u_d_odd = _mm512_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm512_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x8_avx512(cru, crv, u_d, v_d, rnd_i64_v);
      let r_ch_odd = chroma_i64x8_avx512(cru, crv, u_d_odd, v_d_odd, rnd_i64_v);
      let g_ch_even = chroma_i64x8_avx512(cgu, cgv, u_d, v_d, rnd_i64_v);
      let g_ch_odd = chroma_i64x8_avx512(cgu, cgv, u_d_odd, v_d_odd, rnd_i64_v);
      let b_ch_even = chroma_i64x8_avx512(cbu, cbv, u_d, v_d, rnd_i64_v);
      let b_ch_odd = chroma_i64x8_avx512(cbu, cbv, u_d_odd, v_d_odd, rnd_i64_v);

      // Reassemble i64x8 pairs → i32x16 [c0..c15].
      let r_ch_i32 = reassemble_i32x16(r_ch_even, r_ch_odd, interleave_idx);
      let g_ch_i32 = reassemble_i32x16(g_ch_even, g_ch_odd, interleave_idx);
      let b_ch_i32 = reassemble_i32x16(b_ch_even, b_ch_odd, interleave_idx);

      // Duplicate 16 chroma values → 32 slots (4:2:2 upsampling).
      let r_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, r_ch_i32);
      let r_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, r_ch_i32);
      let g_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, g_ch_i32);
      let g_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, g_ch_i32);
      let b_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, b_ch_i32);
      let b_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, b_ch_i32);

      // Y: unsigned-widen 32 u16 → two i32x16 halves, subtract y_off, scale i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Y + chroma → pack with unsigned saturation to u16x32.
      let r_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, r_dup_lo),
          _mm512_add_epi32(y_hi_scaled, r_dup_hi),
        ),
      );
      let g_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, g_dup_lo),
          _mm512_add_epi32(y_hi_scaled, g_dup_hi),
        ),
      );
      let b_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, b_dup_lo),
          _mm512_add_epi32(y_hi_scaled, b_dup_hi),
        ),
      );

      if ALPHA {
        write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
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

// ---- Luma u8 (64 px / iter) ----------------------------------------------

/// AVX-512 Y216 → u8 luma. Extracts Y via `>> 8`. Block size
/// **64 px / iter**.
///
/// Byte-identical to `scalar::y216_to_luma_row`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width`.
#[allow(dead_code)] // dispatcher lands in follow-up PR (Task 9)
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn y216_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX-512F + AVX-512BW is the caller's obligation.
  unsafe {
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let y_idx = _mm512_loadu_si512(Y_FROM_YUYV_IDX.as_ptr().cast());

    let mut x = 0usize;
    while x + 64 <= width {
      // lo group: pixels x..x+31
      let v0 = _mm512_loadu_si512(packed.as_ptr().add(x * 2).cast());
      let v1 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 32).cast());
      let y_lo = _mm512_permutex2var_epi16(v0, y_idx, v1);
      let y_lo_shr = _mm512_srli_epi16::<8>(y_lo);

      // hi group: pixels x+32..x+63
      let v2 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 64).cast());
      let v3 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 96).cast());
      let y_hi = _mm512_permutex2var_epi16(v2, y_idx, v3);
      let y_hi_shr = _mm512_srli_epi16::<8>(y_hi);

      // Pack 64 × i16 → 64 × u8 with natural order.
      let y_u8 = narrow_u8x64(y_lo_shr, y_hi_shr, pack_fixup);
      // Store all 64 bytes at once.
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), y_u8);

      x += 64;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_row(tail_packed, tail_out, tail_w);
    }
  }
}

// ---- Luma u16 (64 px / iter) --------------------------------------------

/// AVX-512 Y216 → u16 luma. Direct copy of Y samples (no shift). Block
/// size **64 px / iter**.
///
/// Byte-identical to `scalar::y216_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width`.
#[allow(dead_code)] // dispatcher lands in follow-up PR (Task 9)
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn y216_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX-512F + AVX-512BW is the caller's obligation.
  unsafe {
    let y_idx = _mm512_loadu_si512(Y_FROM_YUYV_IDX.as_ptr().cast());

    let mut x = 0usize;
    while x + 64 <= width {
      // lo group: pixels x..x+31
      let v0 = _mm512_loadu_si512(packed.as_ptr().add(x * 2).cast());
      let v1 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 32).cast());
      let y_lo = _mm512_permutex2var_epi16(v0, y_idx, v1);

      // hi group: pixels x+32..x+63
      let v2 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 64).cast());
      let v3 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 96).cast());
      let y_hi = _mm512_permutex2var_epi16(v2, y_idx, v3);

      // Direct store — full 16-bit Y values, no shift.
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), y_lo);
      _mm512_storeu_si512(out.as_mut_ptr().add(x + 32).cast(), y_hi);

      x += 64;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_u16_row(tail_packed, tail_out, tail_w);
    }
  }
}
