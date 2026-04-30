//! SSE4.1 Y2xx (Tier 4 packed YUV 4:2:2 high-bit-depth) kernels for
//! `BITS ∈ {10, 12}`. One iteration processes **8 pixels** = 16 u16
//! samples = 32 bytes via two `_mm_loadu_si128` + `_mm_shuffle_epi8`
//! deinterleave.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` with the active
//! `BITS` bits sitting in the **high** bits of each `u16` (low
//! `(16 - BITS)` bits are zero, MSB-aligned). Right-shifting by
//! `(16 - BITS)` brings the active samples into `[0, 2^BITS - 1]`.
//!
//! ## Per-iter pipeline (8 px / 16 u16 / 32 bytes)
//!
//! Two `_mm_loadu_si128` calls fetch 16 u16 lanes:
//!   - `lo` = `[Y0, U0, Y1, V0, Y2, U1, Y3, V1]`
//!   - `hi` = `[Y4, U2, Y5, V2, Y6, U3, Y7, V3]`
//!
//! `_mm_shuffle_epi8` with byte-level u16 indices then permutes:
//!   - Y vector = even u16 lanes: `[Y0..Y7]` (8 valid lanes).
//!   - chroma vector = odd u16 lanes: `[U0, V0, U1, V1, U2, V2, U3, V3]`.
//!
//! A second pair of `_mm_shuffle_epi8` separates U / V, leaving 4
//! valid lanes (low half) per chroma vector. The high 4 lanes hold
//! zeros; they're "don't care" because `_mm_unpacklo_epi16` consumes
//! only lanes 0..3 when duplicating each chroma sample to its 4:2:2
//! Y-pair slot.
//!
//! From there the kernel mirrors `yuv_planar_high_bit.rs::
//! yuv_420p_n_to_rgb_or_rgba_row<BITS, _>` byte-for-byte: subtract
//! chroma bias, Q15-scale chroma to `u_d` / `v_d`, compute
//! `chroma_i16x8` for r/g/b, scale Y, sum + saturate / clamp, write.
//!
//! ## Tail
//!
//! Pixels less than the next 8-px multiple fall through to scalar.
//!
//! ## BITS-template runtime shift count
//!
//! `_mm_srli_epi16::<IMM8>` requires a literal const generic shift,
//! which `16 - BITS` is not on stable Rust. We mirror the established
//! `subsampled_high_bit_pn_4_2_0.rs` / `yuv_planar_high_bit.rs` alpha
//! pattern: build the count vector once via `_mm_cvtsi32_si128` and
//! pass it to the runtime-count `_mm_srl_epi16`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Loads 8 Y2xx pixels (16 u16 samples = 32 bytes) and unpacks them
/// into three `__m128i` vectors holding `BITS`-bit samples in their
/// low bits (each lane an i16):
/// - `y_vec`: lanes 0..8 = Y0..Y7 in `[0, 2^BITS - 1]`.
/// - `u_vec`: lanes 0..4 = U0..U3 in `[0, 2^BITS - 1]` (lanes 4..7
///   hold zeros, treated as don't-care downstream).
/// - `v_vec`: lanes 0..4 = V0..V3 in `[0, 2^BITS - 1]` (lanes 4..7
///   hold zeros, treated as don't-care downstream).
///
/// Strategy: two 128-bit loads + four `_mm_shuffle_epi8` permutes +
/// two `_mm_unpacklo_epi64` consolidations + two runtime-count
/// `_mm_srl_epi16` shifts.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 32 bytes (16 u16) readable,
/// and `target_feature` includes SSE4.1 (which implies SSSE3 for
/// `_mm_shuffle_epi8`).
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn unpack_y2xx_8px_sse41(
  ptr: *const u16,
  shr_count: __m128i,
) -> (__m128i, __m128i, __m128i) {
  // SAFETY: caller obligation — `ptr` has 16 u16 readable; SSE4.1
  // (and thus SSSE3) is available.
  unsafe {
    // Load 16 u16 = 8 pixels (32 bytes = 2 × __m128i).
    let lo = _mm_loadu_si128(ptr.cast()); // [Y0, U0, Y1, V0, Y2, U1, Y3, V1]
    let hi = _mm_loadu_si128(ptr.add(8).cast()); // [Y4, U2, Y5, V2, Y6, U3, Y7, V3]

    // Y permute: pick even u16 lanes (bytes [0,1,4,5,8,9,12,13]) into
    // the low 8 bytes of each shuffled vector; high 8 bytes zeroed by
    // the `-1` mask byte (0x80).
    let y_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_lo = _mm_shuffle_epi8(lo, y_idx); // [Y0, Y1, Y2, Y3, _, _, _, _]
    let y_hi = _mm_shuffle_epi8(hi, y_idx); // [Y4, Y5, Y6, Y7, _, _, _, _]
    // `_mm_unpacklo_epi64` puts the low 8 bytes of `y_hi` after the
    // low 8 bytes of `y_lo` → 8 valid Y lanes.
    let y_vec_raw = _mm_unpacklo_epi64(y_lo, y_hi); // [Y0..Y7]

    // Chroma permute: pick odd u16 lanes (bytes [2,3,6,7,10,11,14,15]).
    let c_idx = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let c_lo = _mm_shuffle_epi8(lo, c_idx); // [U0, V0, U1, V1, _, _, _, _]
    let c_hi = _mm_shuffle_epi8(hi, c_idx); // [U2, V2, U3, V3, _, _, _, _]
    let chroma_raw = _mm_unpacklo_epi64(c_lo, c_hi); // [U0, V0, U1, V1, U2, V2, U3, V3]

    // Right-shift by `(16 - BITS)` to bring MSB-aligned samples into
    // the BITS-aligned range. Runtime count via `_mm_srl_epi16` (the
    // `_mm_srli_epi16::<IMM8>` const-count form is incompatible with
    // a BITS-template kernel since `16 - BITS` is not a stable const
    // generic expression).
    let y_vec = _mm_srl_epi16(y_vec_raw, shr_count);
    let chroma = _mm_srl_epi16(chroma_raw, shr_count);

    // Split chroma U / V via `_mm_shuffle_epi8` with byte-level u16
    // lane indices.
    // chroma layout: [U0, V0, U1, V1, U2, V2, U3, V3] (8 × u16)
    // U vector: [U0, U1, U2, U3, _, _, _, _] — bytes [0,1, 4,5, 8,9, 12,13]
    // V vector: [V0, V1, V2, V3, _, _, _, _] — bytes [2,3, 6,7, 10,11, 14,15]
    let u_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let v_idx = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_vec = _mm_shuffle_epi8(chroma, u_idx);
    let v_vec = _mm_shuffle_epi8(chroma, v_idx);
    (y_vec, u_vec, v_vec)
  }
}

/// SSE4.1 Y2xx → packed RGB / RGBA u8. Const-generic over
/// `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`. Output bit depth is
/// u8 (downshifted from the native BITS Q15 pipeline via
/// `range_params_n::<BITS, 8>`).
///
/// Byte-identical to `scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, ALPHA>`
/// for every input.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)] // wired by Task 9 dispatcher
#[inline]
#[target_feature(enable = "sse4.1")]
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

  // SAFETY: SSE4.1 availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 8 <= width` loop and the caller-promised slice
  // lengths checked above.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    // Loop-invariant runtime shift count for `_mm_srl_epi16`, see
    // module-level note.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_8px_sse41(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = y_vec;

      // Subtract chroma bias (e.g. 512 for 10-bit) — fits i16 since
      // each chroma sample is ≤ 2^BITS - 1 ≤ 4095.
      let u_i16 = _mm_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias_v);

      // Widen 8-lane i16 chroma to two i32x4 halves so the Q15
      // multiplies don't overflow. Only lanes 0..3 of `_lo` are
      // valid; `_hi` is entirely don't-care. We feed both halves
      // through `chroma_i16x8` to recycle the helper exactly; the
      // don't-care output lanes are discarded by the
      // `_mm_unpacklo_epi16` duplicate step below (which only consumes
      // lanes 0..3).
      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      // 8-lane chroma vectors with valid data in lanes 0..3.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Each chroma sample covers 2 Y lanes (4:2:2): duplicate via
      // `_mm_unpacklo_epi16` so lanes 0..7 of `r_dup` align with
      // Y0..Y7. Lane order: [c0, c0, c1, c1, c2, c2, c3, c3].
      let r_dup = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let g_dup = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let b_dup = _mm_unpacklo_epi16(b_chroma, b_chroma);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x8.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. `_mm_packus_epi16(lo, hi)` emits
      // 16 u8 lanes from 16 i16 lanes; we feed `lo == hi` (or zero
      // for hi) so the low 8 bytes of the result hold the saturated
      // u8 of the input i16x8. Only the first 8 bytes per channel
      // matter.
      let zero = _mm_setzero_si128();
      let r_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, r_dup), zero);
      let g_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, g_dup), zero);
      let b_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, b_dup), zero);

      // 8-pixel partial store: SSE4.1's `write_rgb_16` / `write_rgba_16`
      // emit 16-pixel output (48 / 64 bytes), so for the 8-px-iter
      // body we use the v210-style stack-buffer + scalar interleave
      // pattern. (8 px × 3 = 24 bytes RGB, 8 px × 4 = 32 bytes RGBA.)
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

    // Scalar tail — remaining < 8 pixels (always even per 4:2:2).
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

/// SSE4.1 Y2xx → packed `u16` RGB / RGBA at native BITS depth
/// (low-bit-packed: BITS active bits in the low N of each `u16`).
/// Const-generic over `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`.
///
/// Byte-identical to
/// `scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, ALPHA>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[allow(dead_code)] // wired by Task 9 dispatcher
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
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
      let (y_vec, u_vec, v_vec) = unpack_y2xx_8px_sse41(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = y_vec;
      let u_i16 = _mm_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let g_dup = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let b_dup = _mm_unpacklo_epi16(b_chroma, b_chroma);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Native-depth output: clamp to [0, (1 << BITS) - 1].
      // `_mm_adds_epi16` saturates at i16 bounds (no-op here since
      // |sum| stays well inside i16 for BITS ≤ 12), then min/max
      // clamps to the BITS range.
      let r = clamp_u16_max(_mm_adds_epi16(y_scaled, r_dup), zero_v, max_v);
      let g = clamp_u16_max(_mm_adds_epi16(y_scaled, g_dup), zero_v, max_v);
      let b = clamp_u16_max(_mm_adds_epi16(y_scaled, b_dup), zero_v, max_v);

      if ALPHA {
        let alpha = _mm_set1_epi16(out_max);
        write_rgba_u16_8(r, g, b, alpha, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_8(r, g, b, out.as_mut_ptr().add(x * 3));
      }

      x += 8;
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

/// SSE4.1 Y2xx → 8-bit luma. Y values are downshifted from BITS to 8
/// via `>> (BITS - 8)` after the `>> (16 - BITS)` MSB-alignment, i.e.
/// a single `>> 8` from the raw u16 sample. Bypasses the YUV → RGB
/// pipeline entirely.
///
/// Byte-identical to `scalar::y2xx_n_to_luma_row::<BITS>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[allow(dead_code)] // wired by Task 9 dispatcher
#[inline]
#[target_feature(enable = "sse4.1")]
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
    // Y permute mask: pick even u16 lanes (low byte at [0], high byte
    // at [1]) into the low 8 bytes; high 8 bytes zeroed.
    let y_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 8 <= width {
      let lo = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let hi = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 8).cast());
      let y_lo = _mm_shuffle_epi8(lo, y_idx); // [Y0..Y3, _, _, _, _]
      let y_hi = _mm_shuffle_epi8(hi, y_idx); // [Y4..Y7, _, _, _, _]
      let y_vec = _mm_unpacklo_epi64(y_lo, y_hi); // [Y0..Y7] MSB-aligned

      // `>> (16 - BITS)` then `>> (BITS - 8)` collapses to `>> 8` for
      // any BITS ∈ {10, 12} — same single-shift simplification used
      // by NEON's `vshrn_n_u16::<8>`.
      // `_mm_srli_epi16::<8>` has a literal const count, so it works
      // here without the runtime-count helper.
      let y_shr = _mm_srli_epi16::<8>(y_vec);
      // Pack 8 i16 lanes to u8 — only low 8 bytes used.
      let y_u8 = _mm_packus_epi16(y_shr, _mm_setzero_si128());
      // Store low 8 bytes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 16];
      _mm_storeu_si128(tmp.as_mut_ptr().cast(), y_u8);
      luma_out[x..x + 8].copy_from_slice(&tmp[..8]);

      x += 8;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y2xx_n_to_luma_row::<BITS>(tail_packed, tail_out, tail_w);
    }
  }
}

/// SSE4.1 Y2xx → native-depth `u16` luma (low-bit-packed). Each output
/// `u16` carries the source's BITS-bit Y value in its low BITS bits.
/// Byte-identical to `scalar::y2xx_n_to_luma_u16_row::<BITS>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[allow(dead_code)] // wired by Task 9 dispatcher
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let y_idx = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 8 <= width {
      let lo = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let hi = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 8).cast());
      let y_lo = _mm_shuffle_epi8(lo, y_idx);
      let y_hi = _mm_shuffle_epi8(hi, y_idx);
      let y_vec = _mm_unpacklo_epi64(y_lo, y_hi);
      // Right-shift by `(16 - BITS)` to bring MSB-aligned samples
      // into low-bit-packed form for the native-depth u16 output.
      let y_low = _mm_srl_epi16(y_vec, shr_count);
      _mm_storeu_si128(luma_out.as_mut_ptr().add(x).cast(), y_low);
      x += 8;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y2xx_n_to_luma_u16_row::<BITS>(tail_packed, tail_out, tail_w);
    }
  }
}
