//! wasm-simd128 Y2xx (Tier 4 packed YUV 4:2:2 high-bit-depth) kernels
//! for `BITS ∈ {10, 12}`. One iteration processes **8 pixels** = 16
//! u16 samples = 32 bytes via two `v128_load` + `u8x16_swizzle`
//! deinterleave — same block size as NEON / SSE4.1 (wasm has 128-bit
//! registers).
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` with the active
//! `BITS` bits sitting in the **high** bits of each `u16` (low
//! `(16 - BITS)` bits are zero, MSB-aligned). Right-shifting by
//! `(16 - BITS)` brings the active samples into `[0, 2^BITS - 1]`.
//!
//! ## Per-iter pipeline (8 px / 16 u16 / 32 bytes)
//!
//! Two `v128_load` calls fetch 16 u16 lanes:
//!   - `lo` = `[Y0, U0, Y1, V0, Y2, U1, Y3, V1]`
//!   - `hi` = `[Y4, U2, Y5, V2, Y6, U3, Y7, V3]`
//!
//! `u8x16_swizzle` with byte-level u16 indices then permutes:
//!   - Y per-source: even u16 lanes (`[0,1,4,5,8,9,12,13]`) into the
//!     low 8 bytes of each shuffled vector; high bytes zeroed via the
//!     `-1` (= `0xFF` ≥ 16) mask byte — same semantics as SSSE3
//!     `_mm_shuffle_epi8`.
//!   - chroma per-source: odd u16 lanes (`[2,3,6,7,10,11,14,15]`).
//!
//! A cross-vector `i8x16_shuffle` then concatenates the low 8 bytes
//! of each per-source result (wasm has no direct `_mm_unpacklo_epi64`
//! analog), yielding `[Y0..Y7]` and `[U0,V0,U1,V1,U2,V2,U3,V3]`. A
//! second pair of `u8x16_swizzle` separates U / V, leaving 4 valid
//! lanes (low half) per chroma vector. The high 4 lanes hold zeros;
//! they're "don't care" because the [`dup_lo`] helper consumes only
//! lanes 0..3 when duplicating each chroma sample to its 4:2:2
//! Y-pair slot.
//!
//! From there the kernel mirrors `yuv_planar_high_bit.rs::
//! yuv_420p_n_to_rgb_or_rgba_row<BITS, _, _>` byte-for-byte: subtract
//! chroma bias, Q15-scale chroma to `u_d` / `v_d`, compute
//! `chroma_i16x8` for r/g/b, scale Y, sum + saturate / clamp, write.
//!
//! ## Tail
//!
//! Pixels less than the next 8-px multiple fall through to scalar.
//!
//! ## BITS-template runtime shift count
//!
//! `u16x8_shr(v, count)` accepts a runtime u32 count, so unlike
//! SSE4.1's `_mm_srli_epi16::<IMM8>` (literal const generic) we don't
//! need a `_mm_cvtsi32_si128`-style scratch vector. The shift amount
//! `(16 - BITS) as u32` is computed once outside the hot loop.

use core::arch::wasm32::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Loads 8 Y2xx pixels (16 u16 samples = 32 bytes) and unpacks them
/// into three `v128` vectors holding `BITS`-bit samples in their low
/// bits (each lane an i16):
/// - `y_vec`: lanes 0..8 = Y0..Y7 in `[0, 2^BITS - 1]`.
/// - `u_vec`: lanes 0..4 = U0..U3 in `[0, 2^BITS - 1]` (lanes 4..7
///   hold zeros, treated as don't-care downstream).
/// - `v_vec`: lanes 0..4 = V0..V3 in `[0, 2^BITS - 1]` (lanes 4..7
///   hold zeros, treated as don't-care downstream).
///
/// Strategy: two 128-bit loads + four `u8x16_swizzle` permutes + two
/// cross-vector `i8x16_shuffle` consolidations + two runtime-count
/// `u16x8_shr` shifts.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 32 bytes (16 u16) readable,
/// and `target_feature` includes `simd128` (verified at compile time
/// on wasm).
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn unpack_y2xx_8px_wasm(ptr: *const u16, shr_count: u32) -> (v128, v128, v128) {
  // SAFETY: caller obligation — `ptr` has 16 u16 readable; simd128 is
  // enabled at compile time.
  unsafe {
    // Load 16 u16 = 8 pixels (32 bytes = 2 × v128).
    let lo = v128_load(ptr.cast()); // [Y0, U0, Y1, V0, Y2, U1, Y3, V1]
    let hi = v128_load(ptr.add(8).cast()); // [Y4, U2, Y5, V2, Y6, U3, Y7, V3]

    // Y permute: pick even u16 lanes (bytes [0,1,4,5,8,9,12,13]) into
    // the low 8 bytes of each shuffled vector; high 8 bytes zeroed by
    // the `-1` mask byte (0xFF ≥ 16 → `u8x16_swizzle` zeros the
    // lane, matching SSSE3 `_mm_shuffle_epi8`).
    let y_idx = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_lo = u8x16_swizzle(lo, y_idx); // [Y0, Y1, Y2, Y3, _, _, _, _]
    let y_hi = u8x16_swizzle(hi, y_idx); // [Y4, Y5, Y6, Y7, _, _, _, _]
    // Concatenate low 8 bytes of `y_lo` and `y_hi` into one vector:
    // `i8x16_shuffle` byte indices 0..15 select from `y_lo`, 16..31
    // from `y_hi`. Picking `[0..7, 16..23]` mirrors
    // `_mm_unpacklo_epi64(y_lo, y_hi)`.
    let y_vec_raw =
      i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_lo, y_hi); // [Y0..Y7]

    // Chroma permute: pick odd u16 lanes (bytes [2,3,6,7,10,11,14,15]).
    let c_idx = i8x16(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let c_lo = u8x16_swizzle(lo, c_idx); // [U0, V0, U1, V1, _, _, _, _]
    let c_hi = u8x16_swizzle(hi, c_idx); // [U2, V2, U3, V3, _, _, _, _]
    let chroma_raw =
      i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(c_lo, c_hi); // [U0,V0,U1,V1,U2,V2,U3,V3]

    // Right-shift by `(16 - BITS)` to bring MSB-aligned samples into
    // the BITS-aligned range. `u16x8_shr` accepts a runtime u32
    // count, so the BITS-template shift works directly without the
    // SSE4.1-style `_mm_cvtsi32_si128` scratch register.
    let y_vec = u16x8_shr(y_vec_raw, shr_count);
    let chroma = u16x8_shr(chroma_raw, shr_count);

    // Split chroma U / V via `u8x16_swizzle` with byte-level u16 lane
    // indices.
    // chroma layout: [U0, V0, U1, V1, U2, V2, U3, V3] (8 × u16)
    // U vector: [U0, U1, U2, U3, _, _, _, _] — bytes [0,1, 4,5, 8,9, 12,13]
    // V vector: [V0, V1, V2, V3, _, _, _, _] — bytes [2,3, 6,7, 10,11, 14,15]
    let u_idx = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let v_idx = i8x16(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_vec = u8x16_swizzle(chroma, u_idx);
    let v_vec = u8x16_swizzle(chroma, v_idx);
    (y_vec, u_vec, v_vec)
  }
}

/// wasm-simd128 Y2xx → packed RGB / RGBA u8. Const-generic over
/// `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`. Output bit depth is
/// u8 (downshifted from the native BITS Q15 pipeline via
/// `range_params_n::<BITS, 8>`).
///
/// Byte-identical to `scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, ALPHA>`
/// for every input.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)] // wired by Task 9 dispatcher
#[inline]
#[target_feature(enable = "simd128")]
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

  // SAFETY: simd128 compile-time availability is the caller's
  // obligation; the dispatcher in `crate::row` verifies it. Pointer
  // adds are bounded by the `while x + 8 <= width` loop and the
  // caller-promised slice lengths checked above.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias_v = i16x8_splat(bias as i16);
    // Loop-invariant runtime shift count for `u16x8_shr`, see
    // module-level note.
    let shr_count: u32 = 16 - BITS;
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_8px_wasm(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = y_vec;

      // Subtract chroma bias (e.g. 512 for 10-bit) — fits i16 since
      // each chroma sample is ≤ 2^BITS - 1 ≤ 4095.
      let u_i16 = i16x8_sub(u_vec, bias_v);
      let v_i16 = i16x8_sub(v_vec, bias_v);

      // Widen 8-lane i16 chroma to two i32x4 halves so the Q15
      // multiplies don't overflow. Only lanes 0..3 of `_lo` are
      // valid; `_hi` is entirely don't-care. We feed both halves
      // through `chroma_i16x8` to recycle the helper exactly; the
      // don't-care output lanes are discarded by the [`dup_lo`]
      // duplicate step below (which only consumes lanes 0..3).
      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      // 8-lane chroma vectors with valid data in lanes 0..3.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Each chroma sample covers 2 Y lanes (4:2:2): duplicate via
      // [`dup_lo`] so lanes 0..7 of `r_dup` align with Y0..Y7. Lane
      // order: [c0, c0, c1, c1, c2, c2, c3, c3].
      let r_dup = dup_lo(r_chroma);
      let g_dup = dup_lo(g_chroma);
      let b_dup = dup_lo(b_chroma);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x8.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. `u8x16_narrow_i16x8(lo, hi)` emits
      // 16 u8 lanes from 16 i16 lanes; we feed `lo == hi` so the low
      // 8 bytes of the result hold the saturated u8 of the input
      // i16x8. Only the first 8 bytes per channel matter.
      let r_sum = i16x8_add_sat(y_scaled, r_dup);
      let g_sum = i16x8_add_sat(y_scaled, g_dup);
      let b_sum = i16x8_add_sat(y_scaled, b_dup);
      let r_u8 = u8x16_narrow_i16x8(r_sum, r_sum);
      let g_u8 = u8x16_narrow_i16x8(g_sum, g_sum);
      let b_u8 = u8x16_narrow_i16x8(b_sum, b_sum);

      // 8-pixel partial store: wasm-simd128's [`write_rgb_16`] /
      // [`write_rgba_16`] emit 16-pixel output (48 / 64 bytes), so
      // for the 8-px-iter body we use the v210-style stack-buffer +
      // scalar interleave pattern. (8 px × 3 = 24 bytes RGB,
      // 8 px × 4 = 32 bytes RGBA.)
      let mut r_tmp = [0u8; 16];
      let mut g_tmp = [0u8; 16];
      let mut b_tmp = [0u8; 16];
      v128_store(r_tmp.as_mut_ptr().cast(), r_u8);
      v128_store(g_tmp.as_mut_ptr().cast(), g_u8);
      v128_store(b_tmp.as_mut_ptr().cast(), b_u8);

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

/// wasm-simd128 Y2xx → packed `u16` RGB / RGBA at native BITS depth
/// (low-bit-packed: BITS active bits in the low N of each `u16`).
/// Const-generic over `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`.
///
/// Byte-identical to
/// `scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[allow(dead_code)] // wired by Task 9 dispatcher
#[inline]
#[target_feature(enable = "simd128")]
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
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias_v = i16x8_splat(bias as i16);
    let shr_count: u32 = 16 - BITS;
    let max_v = i16x8_splat(out_max);
    let zero_v = i16x8_splat(0);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_8px_wasm(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = y_vec;
      let u_i16 = i16x8_sub(u_vec, bias_v);
      let v_i16 = i16x8_sub(v_vec, bias_v);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup = dup_lo(r_chroma);
      let g_dup = dup_lo(g_chroma);
      let b_dup = dup_lo(b_chroma);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Native-depth output: clamp to [0, (1 << BITS) - 1].
      // `i16x8_add_sat` saturates at i16 bounds (no-op here since
      // |sum| stays well inside i16 for BITS ≤ 12), then min/max
      // clamps to the BITS range.
      let r = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, r_dup), zero_v, max_v);
      let g = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, g_dup), zero_v, max_v);
      let b = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, b_dup), zero_v, max_v);

      if ALPHA {
        let alpha = i16x8_splat(out_max);
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

/// wasm-simd128 Y2xx → 8-bit luma. Y values are downshifted from BITS
/// to 8 via `>> (BITS - 8)` after the `>> (16 - BITS)` MSB-alignment,
/// i.e. a single `>> 8` from the raw u16 sample. Bypasses the YUV →
/// RGB pipeline entirely.
///
/// Byte-identical to `scalar::y2xx_n_to_luma_row::<BITS>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[allow(dead_code)] // wired by Task 9 dispatcher
#[inline]
#[target_feature(enable = "simd128")]
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
    let y_idx = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 8 <= width {
      let lo = v128_load(packed.as_ptr().add(x * 2).cast());
      let hi = v128_load(packed.as_ptr().add(x * 2 + 8).cast());
      let y_lo = u8x16_swizzle(lo, y_idx); // [Y0..Y3, _, _, _, _]
      let y_hi = u8x16_swizzle(hi, y_idx); // [Y4..Y7, _, _, _, _]
      // Concatenate low halves: same `_mm_unpacklo_epi64` pattern as
      // the 4:2:2 unpack helper.
      let y_vec =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_lo, y_hi); // [Y0..Y7] MSB-aligned

      // `>> (16 - BITS)` then `>> (BITS - 8)` collapses to `>> 8` for
      // any BITS ∈ {10, 12} — same single-shift simplification used
      // by NEON's `vshrn_n_u16::<8>` and SSE4.1's `_mm_srli_epi16::<8>`.
      let y_shr = u16x8_shr(y_vec, 8);
      // Pack 8 i16 lanes to u8 — only low 8 bytes used.
      let y_u8 = u8x16_narrow_i16x8(y_shr, y_shr);
      // Store low 8 bytes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 16];
      v128_store(tmp.as_mut_ptr().cast(), y_u8);
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

/// wasm-simd128 Y2xx → native-depth `u16` luma (low-bit-packed). Each
/// output `u16` carries the source's BITS-bit Y value in its low BITS
/// bits. Byte-identical to `scalar::y2xx_n_to_luma_u16_row::<BITS>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[allow(dead_code)] // wired by Task 9 dispatcher
#[inline]
#[target_feature(enable = "simd128")]
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
    let shr_count: u32 = 16 - BITS;
    let y_idx = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 8 <= width {
      let lo = v128_load(packed.as_ptr().add(x * 2).cast());
      let hi = v128_load(packed.as_ptr().add(x * 2 + 8).cast());
      let y_lo = u8x16_swizzle(lo, y_idx);
      let y_hi = u8x16_swizzle(hi, y_idx);
      let y_vec =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_lo, y_hi);
      // Right-shift by `(16 - BITS)` to bring MSB-aligned samples
      // into low-bit-packed form for the native-depth u16 output.
      let y_low = u16x8_shr(y_vec, shr_count);
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_low);
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
