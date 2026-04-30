//! AVX2 Y2xx (Tier 4 packed YUV 4:2:2 high-bit-depth) kernels for
//! `BITS ∈ {10, 12}`. One iteration processes **16 pixels** = 32 u16
//! samples = 64 bytes via two `_mm256_loadu_si256` + per-lane shuffle
//! + cross-lane permute. Doubles SSE4.1's 8-px-per-iter throughput.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` with the active
//! `BITS` bits sitting in the **high** bits of each `u16` (low
//! `(16 - BITS)` bits are zero, MSB-aligned). Right-shifting by
//! `(16 - BITS)` brings the active samples into `[0, 2^BITS - 1]`.
//!
//! ## Per-iter pipeline (16 px / 32 u16 / 64 bytes)
//!
//! Two `_mm256_loadu_si256` calls fetch 32 u16 lanes split across two
//! 256-bit vectors. AVX2's per-128-bit-lane shuffle (`_mm256_shuffle_epi8`)
//! gathers Y bytes into the low 8 bytes of each lane and chroma bytes
//! into the high 8 bytes. Cross-lane consolidation via
//! `_mm256_permute4x64_epi64::<0xD8>` (= `[0, 2, 1, 3]`) brings each
//! vector to `[Y_8_lane0, Y_8_lane1, c_8_lane0, c_8_lane1]` (Y in low
//! 128, chroma in high 128). Finally `_mm256_permute2x128_si256` with
//! selectors `0x20` / `0x31` merges the two vectors:
//! - low 128s combined → `y_vec` = `[Y0..Y15]` (16 u16 lanes).
//! - high 128s combined → `chroma_vec` = `[U0,V0,U1,V1, ..., U7,V7]`
//!   (16 u16 lanes interleaved).
//!
//! A second `_mm256_shuffle_epi8` + `_mm256_permute4x64_epi64::<0xD8>`
//! pair separates U / V from `chroma_vec`, leaving 8 valid lanes (low
//! 128) per vector — `_mm_unpacklo_epi16` (via `chroma_dup`) then
//! duplicates each of the 8 chroma samples to its 4:2:2 Y-pair slot.
//!
//! From there the kernel mirrors the AVX2
//! `yuv_planar_high_bit.rs::yuv_420p_n_to_rgb_or_rgba_row<BITS, _, _>`
//! pipeline byte-for-byte: subtract chroma bias, Q15-scale chroma to
//! `u_d` / `v_d`, compute `chroma_i16x16` for r/g/b, scale Y, sum +
//! saturate / clamp, write.
//!
//! ## Tail
//!
//! Pixels less than the next 16-px multiple fall through to scalar.
//!
//! ## BITS-template runtime shift count
//!
//! `_mm256_srli_epi16::<IMM8>` requires a literal const generic shift,
//! which `16 - BITS` is not on stable Rust. We mirror the SSE4.1
//! precedent (`y2xx.rs`) and the established
//! `subsampled_high_bit_pn_4_2_0.rs` / `yuv_planar_high_bit.rs` alpha
//! pattern: build the count vector once via `_mm_cvtsi32_si128` and
//! pass it to the runtime-count `_mm256_srl_epi16`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Loads 16 Y2xx pixels (32 u16 samples = 64 bytes) and unpacks them
/// into three `__m256i` vectors holding `BITS`-bit samples in their
/// low bits (each lane an i16):
/// - `y_vec`: lanes 0..16 = Y0..Y15 in `[0, 2^BITS - 1]`.
/// - `u_vec`: lanes 0..8 = U0..U7 in `[0, 2^BITS - 1]` (lanes 8..15
///   hold zeros, treated as don't-care downstream).
/// - `v_vec`: lanes 0..8 = V0..V7 in `[0, 2^BITS - 1]` (lanes 8..15
///   hold zeros, treated as don't-care downstream).
///
/// Strategy: two 256-bit loads + two `_mm256_shuffle_epi8` per-lane
/// permutes + two `_mm256_permute4x64_epi64::<0xD8>` consolidations +
/// two `_mm256_permute2x128_si256` cross-vector merges + two
/// runtime-count `_mm256_srl_epi16` shifts + chroma U/V split via
/// another shuffle/permute pair.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 64 bytes (32 u16) readable,
/// and `target_feature` includes AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_y2xx_16px_avx2(
  ptr: *const u16,
  shr_count: __m128i,
) -> (__m256i, __m256i, __m256i) {
  // SAFETY: caller obligation — `ptr` has 32 u16 readable; AVX2 is
  // available.
  unsafe {
    // Load 32 u16 = 16 pixels (64 bytes = 2 × __m256i).
    let v0 = _mm256_loadu_si256(ptr.cast());
    // Per 256-bit vector, `v0` = `[Y0,U0,Y1,V0, Y2,U1,Y3,V1, Y4,U2,Y5,V2, Y6,U3,Y7,V3]`
    // (16 u16 lanes; per-128-bit-lane interleaved).
    let v1 = _mm256_loadu_si256(ptr.add(16).cast());
    // `v1` similarly holds Y8..Y15 + chroma 4..7.

    // Per-lane split: in each 128-bit lane, gather even u16 lanes (Y)
    // into the low 8 bytes, odd u16 lanes (chroma) into the high 8
    // bytes. Same byte indices replicated across both lanes.
    let split_idx = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // low lane: Y first, then chroma
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // high lane: same
    );
    let v0s = _mm256_shuffle_epi8(v0, split_idx);
    let v1s = _mm256_shuffle_epi8(v1, split_idx);

    // After per-lane shuffle, each vector is split into 64-bit chunks:
    //   v0s = [Y0..Y3 (lane0 low 64), c0..c3 (lane0 hi 64),
    //          Y4..Y7 (lane1 low 64), c4..c7 (lane1 hi 64)]
    //   v1s = [Y8..Y11, c8..c11, Y12..Y15, c12..c15]
    // Permute 64-bit chunks within each vector (0xD8 = [0, 2, 1, 3]):
    //   pre  = [A, B, C, D]
    //   post = [A, C, B, D] → all Y in low 128, all chroma in high 128.
    let v0p = _mm256_permute4x64_epi64::<0xD8>(v0s);
    let v1p = _mm256_permute4x64_epi64::<0xD8>(v1s);

    // Cross-vector merge: collect Y from low 128 of v0p,v1p; chroma
    // from high 128.
    let y_raw = _mm256_permute2x128_si256::<0x20>(v0p, v1p); // [Y0..Y15]
    let chroma_raw = _mm256_permute2x128_si256::<0x31>(v0p, v1p); // [U0,V0,U1,V1, ..., U7,V7]

    // Right-shift by `(16 - BITS)` to bring MSB-aligned samples into
    // the BITS-aligned range. Runtime count via `_mm256_srl_epi16`
    // (the const-count form is incompatible with a BITS-template
    // kernel since `16 - BITS` is not a stable const generic
    // expression).
    let y_vec = _mm256_srl_epi16(y_raw, shr_count);
    let chroma = _mm256_srl_epi16(chroma_raw, shr_count);

    // Split chroma U / V via `_mm256_shuffle_epi8` (per-lane) +
    // `_mm256_permute4x64_epi64::<0x88>` (lane-cross consolidate).
    // chroma layout per 128-bit lane (8 u16): [U,V,U,V, U,V,U,V].
    // Per-lane shuffle: U bytes → low 8, V bytes → high 8 (or zero).
    let u_idx = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );
    let v_idx = _mm256_setr_epi8(
      2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );
    // After per-lane shuffle:
    //   u_per_lane = [U0..U3 (lane0 low 64), 0 (lane0 hi 64),
    //                 U4..U7 (lane1 low 64), 0 (lane1 hi 64)]
    // We need lane0_low (U0..U3) and lane1_low (U4..U7) packed into
    // the low 128 bits, with the high 128 don't-care.
    // 0x88 = [0, 2, 0, 2] picks 64-bit chunks (lane0_low, lane1_low,
    // lane0_low, lane1_low) → low 128 = [U0..U7], high 128 dup.
    let u_per_lane = _mm256_shuffle_epi8(chroma, u_idx);
    let v_per_lane = _mm256_shuffle_epi8(chroma, v_idx);
    let u_vec = _mm256_permute4x64_epi64::<0x88>(u_per_lane);
    let v_vec = _mm256_permute4x64_epi64::<0x88>(v_per_lane);

    (y_vec, u_vec, v_vec)
  }
}

/// AVX2 Y2xx → packed RGB / RGBA u8. Const-generic over
/// `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`. Output bit depth is
/// u8 (downshifted from the native BITS Q15 pipeline via
/// `range_params_n::<BITS, 8>`). Block size 16 pixels — doubles
/// SSE4.1's per-iter throughput.
///
/// Byte-identical to `scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, ALPHA>`
/// for every input.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 16 <= width` loop and the caller-promised slice
  // lengths checked above.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    // Loop-invariant runtime shift count for `_mm256_srl_epi16` — see
    // module-level note.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_16px_avx2(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = y_vec;

      // Subtract chroma bias (e.g. 512 for 10-bit) — fits i16 since
      // each chroma sample is ≤ 2^BITS - 1 ≤ 4095.
      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

      // Widen 8-valid-lane i16 chroma to two i32x8 halves so the Q15
      // multiplies don't overflow. Only lanes 0..7 of `_lo` are
      // valid; `_hi` is entirely don't-care. We feed both halves
      // through `chroma_i16x16` to recycle the helper exactly; the
      // don't-care output lanes are discarded by the
      // `chroma_dup` step below (which only consumes lanes 0..7 in
      // its `lo16` return).
      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      // 16-lane chroma vectors with valid data in lanes 0..7.
      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Each chroma sample covers 2 Y lanes (4:2:2): duplicate via
      // `chroma_dup` so lanes 0..15 of `_dup_lo` align with Y0..Y15.
      // `_dup_hi` is don't-care (covers Y16..Y31 if input had 32
      // chroma; we have only 8).
      let (r_dup_lo, _r_dup_hi) = chroma_dup(r_chroma);
      let (g_dup_lo, _g_dup_hi) = chroma_dup(g_chroma);
      let (b_dup_lo, _b_dup_hi) = chroma_dup(b_chroma);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x16.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. `narrow_u8x32(lo, hi)` emits 32 u8
      // lanes from 32 i16 lanes; we feed `lo` and zero for `hi` so
      // the low 16 bytes hold the saturated u8 of our 16 valid lanes.
      let zero = _mm256_setzero_si256();
      let r_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, r_dup_lo), zero);
      let g_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, g_dup_lo), zero);
      let b_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, b_dup_lo), zero);

      // 16-pixel partial store: `write_rgb_32` / `write_rgba_32` emit
      // 32-pixel output (96 / 128 bytes) — too wide for our 16-pixel
      // iter. Use the v210-style stack-buffer + scalar interleave
      // pattern. (16 px × 3 = 48 bytes RGB, 16 px × 4 = 64 bytes RGBA.)
      let mut r_tmp = [0u8; 32];
      let mut g_tmp = [0u8; 32];
      let mut b_tmp = [0u8; 32];
      _mm256_storeu_si256(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm256_storeu_si256(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm256_storeu_si256(b_tmp.as_mut_ptr().cast(), b_u8);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 16 * 4];
        for i in 0..16 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 16 * 3];
        for i in 0..16 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels (always even per 4:2:2).
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

/// AVX2 Y2xx → packed `u16` RGB / RGBA at native BITS depth
/// (low-bit-packed: BITS active bits in the low N of each `u16`).
/// Const-generic over `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`.
/// Block size 16 pixels.
///
/// Byte-identical to
/// `scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, ALPHA>`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[inline]
#[target_feature(enable = "avx2")]
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
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let max_v = _mm256_set1_epi16(out_max);
    let zero_v = _mm256_set1_epi16(0);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_16px_avx2(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = y_vec;
      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let (r_dup_lo, _r_dup_hi) = chroma_dup(r_chroma);
      let (g_dup_lo, _g_dup_hi) = chroma_dup(g_chroma);
      let (b_dup_lo, _b_dup_hi) = chroma_dup(b_chroma);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Native-depth output: clamp to [0, (1 << BITS) - 1]. The AVX2
      // `clamp_u16_max_x16` mirrors SSE4.1's `clamp_u16_max`.
      let r = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, r_dup_lo), zero_v, max_v);
      let g = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, g_dup_lo), zero_v, max_v);
      let b = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, b_dup_lo), zero_v, max_v);

      // 16-pixel u16 store: split each i16x16 channel into two
      // 128-bit halves and use the SSE4.1 u16 interleave helpers
      // (`write_rgb_u16_8` / `write_rgba_u16_8`) — same pattern as
      // the AVX2 high-bit YUV planar u16 path.
      if ALPHA {
        let alpha_u16 = _mm_set1_epi16(out_max);
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(
          _mm256_castsi256_si128(r),
          _mm256_castsi256_si128(g),
          _mm256_castsi256_si128(b),
          alpha_u16,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r),
          _mm256_extracti128_si256::<1>(g),
          _mm256_extracti128_si256::<1>(b),
          alpha_u16,
          dst.add(32),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r),
          _mm256_castsi256_si128(g),
          _mm256_castsi256_si128(b),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r),
          _mm256_extracti128_si256::<1>(g),
          _mm256_extracti128_si256::<1>(b),
          dst.add(24),
        );
      }

      x += 16;
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

/// AVX2 Y2xx → 8-bit luma. Y values are downshifted from BITS to 8
/// via `>> (BITS - 8)` after the `>> (16 - BITS)` MSB-alignment, i.e.
/// a single `>> 8` from the raw u16 sample. Bypasses the YUV → RGB
/// pipeline entirely. Block size 16 pixels.
///
/// Byte-identical to `scalar::y2xx_n_to_luma_row::<BITS>`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    // Per-lane Y permute mask: pick even u16 lanes (low byte at [0],
    // high byte at [1]) into the low 8 bytes; high 8 bytes zeroed.
    let split_idx = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );

    let mut x = 0usize;
    while x + 16 <= width {
      let v0 = _mm256_loadu_si256(packed.as_ptr().add(x * 2).cast());
      let v1 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 16).cast());
      let v0s = _mm256_shuffle_epi8(v0, split_idx);
      let v1s = _mm256_shuffle_epi8(v1, split_idx);
      // After per-lane shuffle: each 256-bit vector has 8 valid u16 Y
      // values in its two lanes' low 64 bits. Pack lane0_low and
      // lane1_low into the low 128 bits of each vector via
      // `_mm256_permute4x64_epi64::<0x88>` (= [0, 2, 0, 2]).
      let v0p = _mm256_permute4x64_epi64::<0x88>(v0s);
      let v1p = _mm256_permute4x64_epi64::<0x88>(v1s);
      // Low 128 of v0p = [Y0..Y7] (8 u16 = 16 bytes).
      // Low 128 of v1p = [Y8..Y15].
      // Combine via `_mm256_permute2x128_si256::<0x20>` (low | low).
      let y_vec = _mm256_permute2x128_si256::<0x20>(v0p, v1p);

      // `>> (16 - BITS)` then `>> (BITS - 8)` collapses to `>> 8` for
      // any BITS ∈ {10, 12} — same single-shift simplification used
      // by NEON's `vshrn_n_u16::<8>`. `_mm256_srli_epi16::<8>` has a
      // literal const count, so it works without runtime-count helper.
      let y_shr = _mm256_srli_epi16::<8>(y_vec);
      // Pack 16 i16 lanes to u8 — only low 16 bytes used.
      let y_u8 = narrow_u8x32(y_shr, _mm256_setzero_si256());
      // Store low 16 bytes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), y_u8);
      luma_out[x..x + 16].copy_from_slice(&tmp[..16]);

      x += 16;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y2xx_n_to_luma_row::<BITS>(tail_packed, tail_out, tail_w);
    }
  }
}

/// AVX2 Y2xx → native-depth `u16` luma (low-bit-packed). Each output
/// `u16` carries the source's BITS-bit Y value in its low BITS bits.
/// Block size 16 pixels. Byte-identical to
/// `scalar::y2xx_n_to_luma_u16_row::<BITS>`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let split_idx = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );

    let mut x = 0usize;
    while x + 16 <= width {
      let v0 = _mm256_loadu_si256(packed.as_ptr().add(x * 2).cast());
      let v1 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 16).cast());
      let v0s = _mm256_shuffle_epi8(v0, split_idx);
      let v1s = _mm256_shuffle_epi8(v1, split_idx);
      let v0p = _mm256_permute4x64_epi64::<0x88>(v0s);
      let v1p = _mm256_permute4x64_epi64::<0x88>(v1s);
      let y_vec = _mm256_permute2x128_si256::<0x20>(v0p, v1p);
      // Right-shift by `(16 - BITS)` to bring MSB-aligned samples
      // into low-bit-packed form for the native-depth u16 output.
      let y_low = _mm256_srl_epi16(y_vec, shr_count);
      _mm256_storeu_si256(luma_out.as_mut_ptr().add(x).cast(), y_low);
      x += 16;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y2xx_n_to_luma_u16_row::<BITS>(tail_packed, tail_out, tail_w);
    }
  }
}
