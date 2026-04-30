//! AVX-512 Y2xx (Tier 4 packed YUV 4:2:2 high-bit-depth) kernels for
//! `BITS ∈ {10, 12}`. One iteration processes **32 pixels** = 64 u16
//! samples = 128 bytes via two `_mm512_loadu_si512` + a single
//! `_mm512_permutex2var_epi16` per stream (Y / chroma) for clean
//! cross-vector u16 separation. Doubles AVX2's 16-px-per-iter
//! throughput.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` with the active
//! `BITS` bits sitting in the **high** bits of each `u16` (low
//! `(16 - BITS)` bits are zero, MSB-aligned). Right-shifting by
//! `(16 - BITS)` brings the active samples into `[0, 2^BITS - 1]`.
//!
//! ## Per-iter pipeline (32 px / 64 u16 / 128 bytes)
//!
//! Two `_mm512_loadu_si512` calls fetch 64 u16 lanes split across two
//! 512-bit vectors. AVX-512BW's cross-vector u16 permute
//! (`_mm512_permutex2var_epi16`) selects 32 u16s from `concat(v0, v1)`
//! in one shot:
//! - **Y_FROM_YUYV_IDX** picks even u16 lanes `[0, 2, 4, ..., 62]` →
//!   `y_raw` = `[Y0, Y1, ..., Y31]` (32 valid Y lanes).
//! - **CHROMA_FROM_YUYV_IDX** picks odd u16 lanes `[1, 3, 5, ..., 63]`
//!   → `chroma_raw` = `[U0, V0, U1, V1, ..., U15, V15]` (32 valid
//!   chroma lanes interleaved).
//!
//! Right-shift by `(16 - BITS)` via the runtime-count
//! `_mm512_srl_epi16` to bring MSB-aligned samples into BITS-aligned
//! form. (`_mm512_srli_epi16::<IMM8>` requires a literal const-generic
//! shift, which `16 - BITS` is not on stable Rust.)
//!
//! Split chroma into U / V via two single-source
//! `_mm512_permutexvar_epi16` calls — `U_FROM_UV_IDX` picks even
//! lanes `[0, 2, ..., 30]`, `V_FROM_UV_IDX` picks odd lanes
//! `[1, 3, ..., 31]`. Each result has 16 valid samples in lanes 0..16
//! and 16 don't-care lanes 16..32.
//!
//! From there the kernel mirrors the AVX-512
//! `yuv_planar_high_bit.rs::yuv_420p_n_to_rgb_or_rgba_row<BITS, _, _>`
//! pipeline: subtract chroma bias, Q15-scale chroma to `u_d` / `v_d`,
//! compute `chroma_i16x32` for r/g/b, scale Y, sum + saturate / clamp,
//! write. With 16 valid chroma samples per channel, `chroma_dup`'s
//! `lo32` output covers all 32 Y lanes after duplication (lanes
//! `[c0,c0, c1,c1, ..., c15,c15]`); `hi32` is don't-care.
//!
//! ## Tail
//!
//! Pixels less than the next 32-px multiple fall through to scalar.
//!
//! ## BITS-template runtime shift count
//!
//! `_mm512_srli_epi16::<IMM8>` requires a literal const generic shift,
//! which `16 - BITS` is not on stable Rust. We mirror the SSE4.1 /
//! AVX2 precedent (`y2xx.rs`) and the established
//! `yuv_planar_high_bit.rs` alpha pattern: build the count vector once
//! via `_mm_cvtsi32_si128` and pass it to the runtime-count
//! `_mm512_srl_epi16`.
//!
//! ## Permute index encoding
//!
//! `_mm512_setr_epi16` is **not** available in stable stdarch (Ship
//! 10's lesson, mirrored by Ship 11a v210). Permute indices live in
//! `static [i16; 32]` arrays loaded via `_mm512_loadu_si512(ptr.cast())`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Static permute index tables --------------------------------------
//
// Each table is a 32-lane u16 permute index. For
// `_mm512_permutex2var_epi16`, indices < 32 select from the first
// source vector, indices ≥ 32 select from the second (concatenated u16
// view). For `_mm512_permutexvar_epi16` (single-source), indices are
// modulo 32.

#[rustfmt::skip]
static Y_FROM_YUYV_IDX: [i16; 32] = [
  // Y output u16 lane i ← concat(v0, v1) u16 lane 2i.
  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
  32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
];

#[rustfmt::skip]
static CHROMA_FROM_YUYV_IDX: [i16; 32] = [
  // Chroma output u16 lane i ← concat(v0, v1) u16 lane (2i + 1).
  // Result is interleaved: [U0, V0, U1, V1, ..., U15, V15].
  1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
  33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
];

#[rustfmt::skip]
static U_FROM_UV_IDX: [i16; 32] = [
  // U output u16 lane i ← chroma u16 lane 2i (even = U). Lanes 16..32
  // are don't-care; pick lane 0 (always-valid index for safety).
  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[rustfmt::skip]
static V_FROM_UV_IDX: [i16; 32] = [
  // V output u16 lane i ← chroma u16 lane (2i + 1) (odd = V). Lanes
  // 16..32 are don't-care; pick lane 1.
  1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// Loads 32 Y2xx pixels (64 u16 samples = 128 bytes) and unpacks them
/// into three `__m512i` vectors holding `BITS`-bit samples in their
/// low bits (each lane an i16):
/// - `y_vec`: lanes 0..32 = Y0..Y31 in `[0, 2^BITS - 1]`.
/// - `u_vec`: lanes 0..16 = U0..U15 in `[0, 2^BITS - 1]` (lanes
///   16..32 hold don't-care values).
/// - `v_vec`: lanes 0..16 = V0..V15 in `[0, 2^BITS - 1]` (lanes
///   16..32 hold don't-care values).
///
/// Strategy: two 512-bit loads + two `_mm512_permutex2var_epi16`
/// (cross-vector u16 separation) + two `_mm512_srl_epi16`
/// (runtime-count BITS-aligned shift) + two `_mm512_permutexvar_epi16`
/// (chroma U/V split).
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 128 bytes (64 u16) readable,
/// and `target_feature` includes AVX-512F + AVX-512BW (BW provides the
/// u16 permute ops `vpermt2w` / `vpermw`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn unpack_y2xx_32px_avx512(
  ptr: *const u16,
  shr_count: __m128i,
) -> (__m512i, __m512i, __m512i) {
  // SAFETY: caller obligation — `ptr` has 64 u16 readable; AVX-512F +
  // AVX-512BW are available.
  unsafe {
    // Load 64 u16 = 32 pixels (128 bytes = 2 × __m512i).
    let v0 = _mm512_loadu_si512(ptr.cast());
    let v1 = _mm512_loadu_si512(ptr.add(32).cast());

    // Cross-vector u16 separation in one shot per stream.
    let y_idx = _mm512_loadu_si512(Y_FROM_YUYV_IDX.as_ptr().cast());
    let chroma_idx = _mm512_loadu_si512(CHROMA_FROM_YUYV_IDX.as_ptr().cast());
    let y_raw = _mm512_permutex2var_epi16(v0, y_idx, v1);
    let chroma_raw = _mm512_permutex2var_epi16(v0, chroma_idx, v1);

    // Right-shift by `(16 - BITS)` to bring MSB-aligned samples into
    // the BITS-aligned range. Runtime count via `_mm512_srl_epi16` —
    // see module-level note on the BITS-template constraint.
    let y_vec = _mm512_srl_epi16(y_raw, shr_count);
    let chroma = _mm512_srl_epi16(chroma_raw, shr_count);

    // Split chroma U / V via single-source `_mm512_permutexvar_epi16`
    // — even u16 lanes go to U, odd to V. Output lanes 16..32 are
    // don't-care (only 16 chroma pairs per 32 px).
    let u_idx = _mm512_loadu_si512(U_FROM_UV_IDX.as_ptr().cast());
    let v_idx = _mm512_loadu_si512(V_FROM_UV_IDX.as_ptr().cast());
    let u_vec = _mm512_permutexvar_epi16(u_idx, chroma);
    let v_vec = _mm512_permutexvar_epi16(v_idx, chroma);

    (y_vec, u_vec, v_vec)
  }
}

/// AVX-512 Y2xx → packed RGB / RGBA u8. Const-generic over
/// `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`. Output bit depth is
/// u8 (downshifted from the native BITS Q15 pipeline via
/// `range_params_n::<BITS, 8>`). Block size 32 pixels — doubles
/// AVX2's per-iter throughput.
///
/// Byte-identical to `scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, ALPHA>`
/// for every input.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn y2xx_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_rgb_or_rgba_row requires BITS in {{10, 12}}"
    );
  }
  debug_assert!(width.is_multiple_of(2), "Y2xx requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's
  // obligation; the dispatcher in `crate::row` verifies it. Pointer
  // adds are bounded by the `while x + 32 <= width` loop and the
  // caller-promised slice lengths checked above.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    // Loop-invariant runtime shift count for `_mm512_srl_epi16` — see
    // module-level note.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
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
    while x + 32 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_32px_avx512(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = y_vec;

      // Subtract chroma bias (e.g. 512 for 10-bit) — fits i16 since
      // each chroma sample is ≤ 2^BITS - 1 ≤ 4095. Only lanes 0..16
      // carry valid samples; the bias subtraction on don't-care lanes
      // is harmless since they're discarded by `chroma_dup`'s `hi32`.
      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      // Widen 16-valid-lane i16 chroma to two i32x16 halves so the
      // Q15 multiplies don't overflow. Only lanes 0..16 of `_lo` are
      // valid; `_hi` is entirely don't-care. We feed both halves
      // through `chroma_i16x32` to recycle the helper exactly; the
      // don't-care output lanes 16..32 are discarded by `chroma_dup`'s
      // `hi32` return below (which only consumes lanes 0..16 in its
      // `lo32` return).
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      // i16x32 chroma vectors with valid data in lanes 0..16.
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // Each chroma sample covers 2 Y lanes (4:2:2). `chroma_dup`
      // duplicates each of 32 chroma lanes into its pair slot,
      // splitting across two i16x32 vectors. With 16 valid chroma in
      // lanes 0..16, `lo32` lanes 0..32 are valid (= [c0,c0, c1,c1,
      // ..., c15,c15]); `hi32` is don't-care.
      let (r_dup_lo, _r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, _g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, _b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x32.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Per-channel saturating add (i16x32). All 32 lanes valid.
      let r_sum = _mm512_adds_epi16(y_scaled, r_dup_lo);
      let g_sum = _mm512_adds_epi16(y_scaled, g_dup_lo);
      let b_sum = _mm512_adds_epi16(y_scaled, b_dup_lo);

      // u8 narrow with saturation. `narrow_u8x64(lo, zero, pack_fixup)`
      // packs 32 i16 lanes of `lo` to u8 in the result's first 32
      // bytes (next 32 zero, after the lane-fixup permute).
      let zero = _mm512_setzero_si512();
      let r_u8 = narrow_u8x64(r_sum, zero, pack_fixup);
      let g_u8 = narrow_u8x64(g_sum, zero, pack_fixup);
      let b_u8 = narrow_u8x64(b_sum, zero, pack_fixup);

      // 32-pixel store via two `write_rgb_16` / `write_rgba_16` calls
      // (each writes 16 px = 48 / 64 bytes). `_mm512_extracti32x4_epi32`
      // pulls the two valid 128-bit halves out of the u8x64 result.
      if ALPHA {
        let alpha = _mm_set1_epi8(-1);
        let r0 = _mm512_castsi512_si128(r_u8);
        let r1 = _mm512_extracti32x4_epi32::<1>(r_u8);
        let g0 = _mm512_castsi512_si128(g_u8);
        let g1 = _mm512_extracti32x4_epi32::<1>(g_u8);
        let b0 = _mm512_castsi512_si128(b_u8);
        let b1 = _mm512_extracti32x4_epi32::<1>(b_u8);
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_16(r0, g0, b0, alpha, dst);
        write_rgba_16(r1, g1, b1, alpha, dst.add(64));
      } else {
        let r0 = _mm512_castsi512_si128(r_u8);
        let r1 = _mm512_extracti32x4_epi32::<1>(r_u8);
        let g0 = _mm512_castsi512_si128(g_u8);
        let g1 = _mm512_extracti32x4_epi32::<1>(g_u8);
        let b0 = _mm512_castsi512_si128(b_u8);
        let b1 = _mm512_extracti32x4_epi32::<1>(b_u8);
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_16(r0, g0, b0, dst);
        write_rgb_16(r1, g1, b1, dst.add(48));
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels (always even per 4:2:2).
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 Y2xx → packed `u16` RGB / RGBA at native BITS depth
/// (low-bit-packed: BITS active bits in the low N of each `u16`).
/// Const-generic over `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`.
/// Block size 32 pixels.
///
/// Byte-identical to
/// `scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, ALPHA>`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn y2xx_n_to_rgb_u16_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_rgb_u16_or_rgba_u16_row requires BITS in {{10, 12}}"
    );
  }
  debug_assert!(width.is_multiple_of(2), "Y2xx requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
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
    while x + 32 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_32px_avx512(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = y_vec;
      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, _r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, _g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, _b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Native-depth output: clamp to [0, (1 << BITS) - 1].
      let r = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, r_dup_lo), zero_v, max_v);
      let g = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, g_dup_lo), zero_v, max_v);
      let b = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, b_dup_lo), zero_v, max_v);

      // 32-pixel u16 store via the shared 32-pixel writers.
      if ALPHA {
        let alpha_u16 = _mm_set1_epi16(out_max);
        write_rgba_u16_32(r, g, b, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_32(r, g, b, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 Y2xx → 8-bit luma. Y values are downshifted from BITS to 8
/// via `>> (BITS - 8)` after the `>> (16 - BITS)` MSB-alignment, i.e.
/// a single `>> 8` from the raw u16 sample. Bypasses the YUV → RGB
/// pipeline entirely. Block size 32 pixels.
///
/// Byte-identical to `scalar::y2xx_n_to_luma_row::<BITS>`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn y2xx_n_to_luma_row<const BITS: u32>(
  packed: &[u16],
  luma_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_luma_row requires BITS in {{10, 12}}"
    );
  }
  debug_assert!(width.is_multiple_of(2), "Y2xx requires even width");
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let zero = _mm512_setzero_si512();
    let y_idx = _mm512_loadu_si512(Y_FROM_YUYV_IDX.as_ptr().cast());

    let mut x = 0usize;
    while x + 32 <= width {
      // Load 64 u16 = 32 pixels and pull just the Y lanes via the
      // cross-vector u16 permute. We don't need chroma here.
      let v0 = _mm512_loadu_si512(packed.as_ptr().add(x * 2).cast());
      let v1 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 32).cast());
      let y_raw = _mm512_permutex2var_epi16(v0, y_idx, v1);
      // `>> (16 - BITS)` then `>> (BITS - 8)` collapses to `>> 8` for
      // any BITS ∈ {10, 12} — same single-shift simplification used
      // by NEON / AVX2. `_mm512_srli_epi16::<8>` has a literal const
      // count, so it works without runtime-count helper.
      let y_shr = _mm512_srli_epi16::<8>(y_raw);
      // Pack 32 i16 lanes to u8 — first 32 bytes valid (after pack
      // fixup); next 32 zero from the zero-hi pack source.
      let y_u8 = narrow_u8x64(y_shr, zero, pack_fixup);
      // Store first 32 bytes via the low 256-bit half.
      _mm256_storeu_si256(
        luma_out.as_mut_ptr().add(x).cast(),
        _mm512_castsi512_si256(y_u8),
      );
      x += 32;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y2xx_n_to_luma_row::<BITS>(tail_packed, tail_out, tail_w);
    }
  }
}

/// AVX-512 Y2xx → native-depth `u16` luma (low-bit-packed). Each
/// output `u16` carries the source's BITS-bit Y value in its low BITS
/// bits. Block size 32 pixels. Byte-identical to
/// `scalar::y2xx_n_to_luma_u16_row::<BITS>`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn y2xx_n_to_luma_u16_row<const BITS: u32>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_luma_u16_row requires BITS in {{10, 12}}"
    );
  }
  debug_assert!(width.is_multiple_of(2), "Y2xx requires even width");
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let y_idx = _mm512_loadu_si512(Y_FROM_YUYV_IDX.as_ptr().cast());

    let mut x = 0usize;
    while x + 32 <= width {
      let v0 = _mm512_loadu_si512(packed.as_ptr().add(x * 2).cast());
      let v1 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 32).cast());
      let y_raw = _mm512_permutex2var_epi16(v0, y_idx, v1);
      // Right-shift by `(16 - BITS)` to bring MSB-aligned samples into
      // low-bit-packed form for the native-depth u16 output.
      let y_low = _mm512_srl_epi16(y_raw, shr_count);
      _mm512_storeu_si512(luma_out.as_mut_ptr().add(x).cast(), y_low);
      x += 32;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y2xx_n_to_luma_u16_row::<BITS>(tail_packed, tail_out, tail_w);
    }
  }
}
