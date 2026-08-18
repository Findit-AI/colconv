//! NEON Y216 (packed YUV 4:2:2, BITS=16) kernels.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` where each
//! sample occupies the full 16-bit word (no MSB/LSB alignment shift
//! needed — unlike Y210/Y212 which require `>> (16 - BITS)`).
//!
//! ## Per-iter pipeline (16 px / 32 u16 / 64 bytes)
//!
//! Two `vld2q_u16` reads deinterleave the 32 u16 samples into:
//!   - `pair_lo.0` = `[Y0..Y7]`   (even u16 indices from first 16 u16s)
//!   - `pair_lo.1` = `[U0,V0,U1,V1,U2,V2,U3,V3]` (odd u16 indices)
//!   - `pair_hi.0` = `[Y8..Y15]`
//!   - `pair_hi.1` = `[U4,V4,…,U7,V7]`
//!
//! `vuzp1q_u16(chroma, chroma)` extracts the U lanes (even positions)
//! and `vuzp2q_u16(chroma, chroma)` extracts V (odd positions).
//! Only `vget_low_u16` is used to obtain the 4 valid chroma samples;
//! the high 4 duplicated lanes are discarded.
//!
//! ## Arithmetic
//!
//! u8 output: i32 chroma via `chroma_i16x8` helper (same as Y210/Y212).
//! u16 output: i64 chroma via `chroma_i64x4` helper (same as
//! `yuv_420p16_to_rgb_or_rgba_u16_row`). No load-time right-shift or
//! mask — BITS=16 samples are already full-range u16.

use super::*;
use crate::{KernelMatrix, row::scalar};

/// Host-endian gate for Y216 SIMD bodies.
///
/// `vld2q_u16` deinterleaves using **host-native** u16 reads, so the SIMD
/// body is only correct when the encoded byte order matches the host.
/// Truth table:
///
/// | wire `BE` | host       | `BE == HOST_NATIVE_BE` | path   | correct via    |
/// |-----------|------------|------------------------|--------|----------------|
/// | false     | LE         | true                   | SIMD   | host-native LE |
/// | false     | BE         | false                  | scalar | `from_le`      |
/// | true      | LE         | false                  | scalar | `from_be`      |
/// | true      | BE         | true                   | SIMD   | host-native BE |
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// ---- u8 output (i32 chroma, 16 px/iter) ---------------------------------

/// NEON Y216 → packed u8 RGB or RGBA.
/// The NEON loop runs only when `BE == HOST_NATIVE_BE`; otherwise the
/// scalar kernel handles the full row (cross-endian decode).
///
/// Byte-identical to `scalar::y216_to_rgb_or_rgba_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn y216_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    // SIMD body runs only when the wire byte order matches the host
    // (`BE == HOST_NATIVE_BE`); otherwise scalar handles the full row
    // via `from_le` / `from_be`.
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      let rnd_v = vdupq_n_s32(RND);
      // For the u8 output path: `scale_y_u16_to_i16` takes i32x4 y_off.
      // Y values are full u16 (0..65535), so we must use u16-aware widening
      // rather than reinterpreting as i16 (which would corrupt values > 32767).
      let y_off_v = vdupq_n_s32(y_off);
      let y_scale_v = vdupq_n_s32(y_scale);
      let c_scale_v = vdupq_n_s32(c_scale);
      let bias_v = vdupq_n_s16(bias as i16);
      let cru = vdupq_n_s32(coeffs.r_u());
      let crv = vdupq_n_s32(coeffs.r_v());
      let cgu = vdupq_n_s32(coeffs.g_u());
      let cgv = vdupq_n_s32(coeffs.g_v());
      let cbu = vdupq_n_s32(coeffs.b_u());
      let cbv = vdupq_n_s32(coeffs.b_v());

      while x + 16 <= width {
        // Two vld2q_u16 calls: each deinterleaves 8 px (16 u16).
        // ptr offset x*2 u16 for lo-group, x*2+16 u16 for hi-group.
        let pair_lo = vld2q_u16(packed.as_ptr().add(x * 2));
        let pair_hi = vld2q_u16(packed.as_ptr().add(x * 2 + 16));

        // Extract U and V from interleaved chroma via vuzp.
        // pair_lo.1 = [U0,V0,U1,V1,U2,V2,U3,V3]
        // vuzp1q_u16(c,c) = [U0,U1,U2,U3, U0,U1,U2,U3] — low 4 valid.
        // vuzp2q_u16(c,c) = [V0,V1,V2,V3, V0,V1,V2,V3] — low 4 valid.
        let u_lo_vec = vuzp1q_u16(pair_lo.1, pair_lo.1);
        let v_lo_vec = vuzp2q_u16(pair_lo.1, pair_lo.1);
        let u_hi_vec = vuzp1q_u16(pair_hi.1, pair_hi.1);
        let v_hi_vec = vuzp2q_u16(pair_hi.1, pair_hi.1);

        // Chroma bias subtraction: chroma ∈ [0,65535], bias=32768, so
        // (chroma - bias) ∈ [-32768, 32767] which fits exactly in i16.
        let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(u_lo_vec), bias_v);
        let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(v_lo_vec), bias_v);
        let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(u_hi_vec), bias_v);
        let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(v_hi_vec), bias_v);

        // Widen to i32x4 for Q15 multiply.
        // _0 = low 4 (valid), _1 = high 4 (duplicates; don't-care outputs
        // discarded by vzip1q_s16 below which only uses lanes 0..3).
        let u_lo_i32_0 = vmovl_s16(vget_low_s16(u_lo_i16));
        let u_lo_i32_1 = vmovl_s16(vget_high_s16(u_lo_i16));
        let v_lo_i32_0 = vmovl_s16(vget_low_s16(v_lo_i16));
        let v_lo_i32_1 = vmovl_s16(vget_high_s16(v_lo_i16));
        let u_hi_i32_0 = vmovl_s16(vget_low_s16(u_hi_i16));
        let u_hi_i32_1 = vmovl_s16(vget_high_s16(u_hi_i16));
        let v_hi_i32_0 = vmovl_s16(vget_low_s16(v_hi_i16));
        let v_hi_i32_1 = vmovl_s16(vget_high_s16(v_hi_i16));

        // Q15 chroma scale.
        let u_d_lo_0 = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32_0, c_scale_v), rnd_v));
        let u_d_lo_1 = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32_1, c_scale_v), rnd_v));
        let v_d_lo_0 = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32_0, c_scale_v), rnd_v));
        let v_d_lo_1 = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32_1, c_scale_v), rnd_v));
        let u_d_hi_0 = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32_0, c_scale_v), rnd_v));
        let u_d_hi_1 = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32_1, c_scale_v), rnd_v));
        let v_d_hi_0 = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32_0, c_scale_v), rnd_v));
        let v_d_hi_1 = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32_1, c_scale_v), rnd_v));

        // Build 8-lane chroma vectors (4 valid in lo + 4 duplicate in hi;
        // `chroma_i16x8` produces lanes 0..3 correct, lanes 4..7 don't-care).
        let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_0, v_d_lo_0, u_d_lo_1, v_d_lo_1, rnd_v);
        let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_0, v_d_lo_0, u_d_lo_1, v_d_lo_1, rnd_v);
        let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_0, v_d_lo_0, u_d_lo_1, v_d_lo_1, rnd_v);
        let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_0, v_d_hi_0, u_d_hi_1, v_d_hi_1, rnd_v);
        let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_0, v_d_hi_0, u_d_hi_1, v_d_hi_1, rnd_v);
        let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_0, v_d_hi_0, u_d_hi_1, v_d_hi_1, rnd_v);

        // Duplicate chroma into Y-pair slots (4:2:2):
        // vzip1q_s16([c0,c1,c2,c3, …dup…], same) = [c0,c0,c1,c1,c2,c2,c3,c3]
        let r_dup_lo = vzip1q_s16(r_chroma_lo, r_chroma_lo);
        let g_dup_lo = vzip1q_s16(g_chroma_lo, g_chroma_lo);
        let b_dup_lo = vzip1q_s16(b_chroma_lo, b_chroma_lo);
        let r_dup_hi = vzip1q_s16(r_chroma_hi, r_chroma_hi);
        let g_dup_hi = vzip1q_s16(g_chroma_hi, g_chroma_hi);
        let b_dup_hi = vzip1q_s16(b_chroma_hi, b_chroma_hi);

        // Y scale using u16-aware helper: unsigned-widens u16 → i32, applies
        // (y - y_off) * y_scale Q15, narrows to i16x8.  Avoids the i16
        // overflow that `scale_y` would cause for Y values > 32767.
        let y_lo_scaled = scale_y_u16_to_i16(pair_lo.0, y_off_v, y_scale_v, rnd_v);
        let y_hi_scaled = scale_y_u16_to_i16(pair_hi.0, y_off_v, y_scale_v, rnd_v);

        // Saturating add; narrow to u8x8.
        let r_lo_u8 = vqmovun_s16_compat(vqaddq_s16(y_lo_scaled, r_dup_lo));
        let g_lo_u8 = vqmovun_s16_compat(vqaddq_s16(y_lo_scaled, g_dup_lo));
        let b_lo_u8 = vqmovun_s16_compat(vqaddq_s16(y_lo_scaled, b_dup_lo));
        let r_hi_u8 = vqmovun_s16_compat(vqaddq_s16(y_hi_scaled, r_dup_hi));
        let g_hi_u8 = vqmovun_s16_compat(vqaddq_s16(y_hi_scaled, g_dup_hi));
        let b_hi_u8 = vqmovun_s16_compat(vqaddq_s16(y_hi_scaled, b_dup_hi));

        if ALPHA {
          let alpha = vdup_n_u8(0xFF);
          vst4_u8(
            out.as_mut_ptr().add(x * 4),
            uint8x8x4_t(r_lo_u8, g_lo_u8, b_lo_u8, alpha),
          );
          vst4_u8(
            out.as_mut_ptr().add(x * 4 + 32),
            uint8x8x4_t(r_hi_u8, g_hi_u8, b_hi_u8, alpha),
          );
        } else {
          vst3_u8(
            out.as_mut_ptr().add(x * 3),
            uint8x8x3_t(r_lo_u8, g_lo_u8, b_lo_u8),
          );
          vst3_u8(
            out.as_mut_ptr().add(x * 3 + 24),
            uint8x8x3_t(r_hi_u8, g_hi_u8, b_hi_u8),
          );
        }

        x += 16;
      }
    } // end if BE == HOST_NATIVE_BE

    // Scalar tail — remaining < 16 pixels, or full-row fallback when
    // `BE != HOST_NATIVE_BE` (cross-endian decode on either host).
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_or_rgba_row::<ALPHA, BE>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- u16 output (i64 chroma, 16 px/iter) --------------------------------

/// NEON Y216 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x4`) to avoid overflow at 16-bit scales.
/// The NEON loop runs only when `BE == HOST_NATIVE_BE`; otherwise the
/// scalar kernel handles the full row (cross-endian decode).
/// Byte-identical to `scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>`.
///
/// ## Pipeline
///
/// Two `vld2q_u16` loads give `pair_lo` (Y0..Y7 + 4 UV pairs) and
/// `pair_hi` (Y8..Y15 + 4 UV pairs). `vuzp1q_u16(pair.1, pair.1)`
/// puts the 4 valid U samples in lanes 0..3; `vget_low_u16` extracts
/// them as a clean i32x4 after widening. `u_d_lo` covers chroma for
/// Y0..Y7; `u_d_hi` covers Y8..Y15. Each i32x4 is duplicated via
/// `vzip1q_s32`/`vzip2q_s32` into per-pixel chroma aligned to Y.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn y216_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    // SIMD body runs only when the wire byte order matches the host
    // (`BE == HOST_NATIVE_BE`); otherwise scalar handles the full row
    // via `from_le` / `from_be`.
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      let alpha_u16 = vdupq_n_u16(0xFFFF);
      let rnd_v = vdupq_n_s32(RND);
      let rnd64 = vdupq_n_s64(RND as i64);
      let y_off_v = vdupq_n_s32(y_off);
      let y_scale_d = vdup_n_s32(y_scale); // int32x2_t for vmull_s32
      let c_scale_v = vdupq_n_s32(c_scale);
      let bias_v = vdupq_n_s32(bias);
      let cru = vdupq_n_s32(coeffs.r_u());
      let crv = vdupq_n_s32(coeffs.r_v());
      let cgu = vdupq_n_s32(coeffs.g_u());
      let cgv = vdupq_n_s32(coeffs.g_v());
      let cbu = vdupq_n_s32(coeffs.b_u());
      let cbv = vdupq_n_s32(coeffs.b_v());

      while x + 16 <= width {
        // Two vld2q_u16: each deinterleaves 8 px → 8 Y + [UV…] pairs.
        let pair_lo = vld2q_u16(packed.as_ptr().add(x * 2));
        let pair_hi = vld2q_u16(packed.as_ptr().add(x * 2 + 16));

        // Extract U/V from chroma via vuzp.
        // vuzp1q_u16(c,c) = [U0..U3, U0..U3]; use vget_low for 4 valid.
        let u_lo_raw = vuzp1q_u16(pair_lo.1, pair_lo.1);
        let v_lo_raw = vuzp2q_u16(pair_lo.1, pair_lo.1);
        let u_hi_raw = vuzp1q_u16(pair_hi.1, pair_hi.1);
        let v_hi_raw = vuzp2q_u16(pair_hi.1, pair_hi.1);

        // Widen 4 valid chroma samples, subtract bias, apply c_scale → u_d.
        let u_d_lo = q15_shift(vaddq_s32(
          vmulq_s32(
            vsubq_s32(
              vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_lo_raw))),
              bias_v,
            ),
            c_scale_v,
          ),
          rnd_v,
        ));
        let v_d_lo = q15_shift(vaddq_s32(
          vmulq_s32(
            vsubq_s32(
              vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_lo_raw))),
              bias_v,
            ),
            c_scale_v,
          ),
          rnd_v,
        ));
        let u_d_hi = q15_shift(vaddq_s32(
          vmulq_s32(
            vsubq_s32(
              vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_hi_raw))),
              bias_v,
            ),
            c_scale_v,
          ),
          rnd_v,
        ));
        let v_d_hi = q15_shift(vaddq_s32(
          vmulq_s32(
            vsubq_s32(
              vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_hi_raw))),
              bias_v,
            ),
            c_scale_v,
          ),
          rnd_v,
        ));

        // i64 chroma: 4 values → i32x4 (vmull_s32 widening to avoid i32 overflow).
        let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
        let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
        let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
        let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
        let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
        let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

        // Duplicate 4 chroma values into 8 per-pixel slots (4:2:2).
        // vzip1q_s32([c0,c1,c2,c3], same) = [c0,c0,c1,c1] → Y0,Y1,Y2,Y3
        // vzip2q_s32([c0,c1,c2,c3], same) = [c2,c2,c3,c3] → Y4,Y5,Y6,Y7
        let r_cd_lo0 = vzip1q_s32(r_ch_lo, r_ch_lo);
        let r_cd_lo1 = vzip2q_s32(r_ch_lo, r_ch_lo);
        let g_cd_lo0 = vzip1q_s32(g_ch_lo, g_ch_lo);
        let g_cd_lo1 = vzip2q_s32(g_ch_lo, g_ch_lo);
        let b_cd_lo0 = vzip1q_s32(b_ch_lo, b_ch_lo);
        let b_cd_lo1 = vzip2q_s32(b_ch_lo, b_ch_lo);
        let r_cd_hi0 = vzip1q_s32(r_ch_hi, r_ch_hi);
        let r_cd_hi1 = vzip2q_s32(r_ch_hi, r_ch_hi);
        let g_cd_hi0 = vzip1q_s32(g_ch_hi, g_ch_hi);
        let g_cd_hi1 = vzip2q_s32(g_ch_hi, g_ch_hi);
        let b_cd_hi0 = vzip1q_s32(b_ch_hi, b_ch_hi);
        let b_cd_hi1 = vzip2q_s32(b_ch_hi, b_ch_hi);

        // i64 Y scale: (y - y_off) * y_scale can reach ~2.35x10⁹ at limited range.
        // Split each 8-lane Y into two i32x4 halves for scale_y_u16_i64.
        // y_lo_0 = Y0..Y3, y_lo_1 = Y4..Y7; y_hi_0 = Y8..Y11, y_hi_1 = Y12..Y15.
        let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(pair_lo.0)));
        let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(pair_lo.0)));
        let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(pair_hi.0)));
        let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(pair_hi.0)));
        let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
        let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
        let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
        let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

        // Y + chroma; vqmovun_s32 saturates i32 → u16 (clamps [0, 65535]).
        //
        // Alignment:
        //   ys_lo_0 = [Y0,Y1,Y2,Y3]   r_cd_lo0 = [c0,c0,c1,c1]  → pixels 0..3
        //   ys_lo_1 = [Y4,Y5,Y6,Y7]   r_cd_lo1 = [c2,c2,c3,c3]  → pixels 4..7
        //   ys_hi_0 = [Y8,Y9,Y10,Y11] r_cd_hi0 = [c4,c4,c5,c5]  → pixels 8..11
        //   ys_hi_1 = [Y12..Y15]       r_cd_hi1 = [c6,c6,c7,c7]  → pixels 12..15
        //
        // vcombine_u16(A, B) packs two u16x4 into one u16x8.
        let r_lo_u16 = vcombine_u16(
          vqmovun_s32_compat(vaddq_s32(ys_lo_0, r_cd_lo0)),
          vqmovun_s32_compat(vaddq_s32(ys_lo_1, r_cd_lo1)),
        );
        let g_lo_u16 = vcombine_u16(
          vqmovun_s32_compat(vaddq_s32(ys_lo_0, g_cd_lo0)),
          vqmovun_s32_compat(vaddq_s32(ys_lo_1, g_cd_lo1)),
        );
        let b_lo_u16 = vcombine_u16(
          vqmovun_s32_compat(vaddq_s32(ys_lo_0, b_cd_lo0)),
          vqmovun_s32_compat(vaddq_s32(ys_lo_1, b_cd_lo1)),
        );
        // hi group (Y8..Y15)
        let r_hi_u16 = vcombine_u16(
          vqmovun_s32_compat(vaddq_s32(ys_hi_0, r_cd_hi0)),
          vqmovun_s32_compat(vaddq_s32(ys_hi_1, r_cd_hi1)),
        );
        let g_hi_u16 = vcombine_u16(
          vqmovun_s32_compat(vaddq_s32(ys_hi_0, g_cd_hi0)),
          vqmovun_s32_compat(vaddq_s32(ys_hi_1, g_cd_hi1)),
        );
        let b_hi_u16 = vcombine_u16(
          vqmovun_s32_compat(vaddq_s32(ys_hi_0, b_cd_hi0)),
          vqmovun_s32_compat(vaddq_s32(ys_hi_1, b_cd_hi1)),
        );

        // Each u16x8 covers 8 pixels.  Two stores per format (lo + hi).
        // For ALPHA: each vst4q_u16 writes 8 RGBA pixels (8 x 4 x 2 = 64 bytes).
        //   Offset for lo: x*4 u16. Offset for hi: x*4+32 u16.
        // For RGB:  each vst3q_u16 writes 8 RGB pixels (8 x 3 x 2 = 48 bytes).
        //   Offset for lo: x*3 u16. Offset for hi: x*3+24 u16.
        if ALPHA {
          vst4q_u16(
            out.as_mut_ptr().add(x * 4),
            uint16x8x4_t(r_lo_u16, g_lo_u16, b_lo_u16, alpha_u16),
          );
          vst4q_u16(
            out.as_mut_ptr().add(x * 4 + 32),
            uint16x8x4_t(r_hi_u16, g_hi_u16, b_hi_u16, alpha_u16),
          );
        } else {
          vst3q_u16(
            out.as_mut_ptr().add(x * 3),
            uint16x8x3_t(r_lo_u16, g_lo_u16, b_lo_u16),
          );
          vst3q_u16(
            out.as_mut_ptr().add(x * 3 + 24),
            uint16x8x3_t(r_hi_u16, g_hi_u16, b_hi_u16),
          );
        }

        x += 16;
      }
    } // end if BE == HOST_NATIVE_BE

    // Scalar tail — remaining < 16 pixels, or full-row fallback when
    // `BE != HOST_NATIVE_BE` (cross-endian decode on either host).
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (16 px/iter) -----------------------------------------------

/// NEON Y216 → u8 luma. Extracts Y via `>> 8`.
/// NEON runs only when `BE == HOST_NATIVE_BE`; otherwise the scalar
/// kernel handles the full row.
///
/// Byte-identical to `scalar::y216_to_luma_row::<BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn y216_to_luma_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  unsafe {
    // SIMD body runs only when the wire byte order matches the host
    // (`BE == HOST_NATIVE_BE`); otherwise scalar handles the full row.
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + 16 <= width {
        // Two vld2q_u16: pair.0 = 8 Y lanes each; chroma discarded.
        let pair_lo = vld2q_u16(packed.as_ptr().add(x * 2));
        let pair_hi = vld2q_u16(packed.as_ptr().add(x * 2 + 16));
        // >> 8 narrows u16 → u8 (high byte of each Y sample).
        let y_lo_u8 = vshrn_n_u16::<8>(pair_lo.0);
        let y_hi_u8 = vshrn_n_u16::<8>(pair_hi.0);
        vst1_u8(out.as_mut_ptr().add(x), y_lo_u8);
        vst1_u8(out.as_mut_ptr().add(x + 8), y_hi_u8);
        x += 16;
      }
    }
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_row::<BE>(tail_packed, tail_out, tail_w);
    }
  }
}

// ---- Luma u16 (16 px/iter) ----------------------------------------------

/// NEON Y216 → u16 luma. Direct copy of Y samples (no shift).
/// NEON runs only when `BE == HOST_NATIVE_BE`; otherwise the scalar
/// kernel handles the full row.
///
/// Byte-identical to `scalar::y216_to_luma_u16_row::<BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn y216_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  unsafe {
    // SIMD body runs only when the wire byte order matches the host
    // (`BE == HOST_NATIVE_BE`); otherwise scalar handles the full row.
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + 16 <= width {
        let pair_lo = vld2q_u16(packed.as_ptr().add(x * 2));
        let pair_hi = vld2q_u16(packed.as_ptr().add(x * 2 + 16));
        // Direct copy — Y samples are already full 16-bit (no shift needed).
        vst1q_u16(out.as_mut_ptr().add(x), pair_lo.0);
        vst1q_u16(out.as_mut_ptr().add(x + 8), pair_hi.0);
        x += 16;
      }
    }
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_u16_row::<BE>(tail_packed, tail_out, tail_w);
    }
  }
}

// ---- Y216 (8-bit) → HSV (staged via a reused 8-bit RGB chunk) --------
//
// The NEON twin of the scalar `y216_to_hsv_row` kernel. Fills a small
// reused 8-bit RGB scratch (one chunk at a time) via the EXISTING NEON
// Y216 RGB-u8 kernel, then runs the NEON `rgb_to_hsv_row` on the chunk —
// byte-identical to `rgb_to_hsv_row(y216_to_rgb_or_rgba_row::<false,
// BE>(...))` within the NEON tier. Reuses the family-local
// `super::y2xx::y2xx_hsv_via_rgb_chunks` driver (the shared yuv-planar
// driver is unavailable under `y2xx` alone).

/// NEON: Y216 (BITS=16) → planar HSV bytes (OpenCV encoding), staged via
/// the reused-8-bit-RGB-chunk pattern over the NEON
/// [`y216_to_rgb_or_rgba_row`] + [`rgb_to_hsv_row`]. `BE` selects the
/// source byte order. Byte-identical to
/// `rgb_to_hsv_row(y216_to_rgb_or_rgba_row::<false, BE>(...))` within the
/// NEON tier.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn y216_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: NEON verified; the chunk filler forwards to the NEON Y216
  // RGB-u8 kernel under the same contract (its own scalar tail covers small
  // n). Packed byte offset = `offset * 2` u16.
  unsafe {
    super::y2xx::y2xx_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      y216_to_rgb_or_rgba_row::<false, BE>(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}
