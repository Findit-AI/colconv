//! NEON kernels for XV48 packed YUV 4:4:4 16-bit family
//! (FFmpeg `AV_PIX_FMT_XV48LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), X(16)]`
//! little-endian, each holding a full 16-bit sample (no padding bits,
//! no right-shift on load — the full-depth sibling of XV36). The `X`
//! slot is **padding** — read by `vld4q_u16` but discarded. RGBA
//! outputs force α = max (`0xFF` u8 / `0xFFFF` u16).
//!
//! ## Per-iter pipeline
//!
//! `vld4q_u16` loads 8 quadruples in one call, returning a
//! `uint16x8x4_t` where `.0 = U`, `.1 = Y`, `.2 = V`, `.3 = X`
//! (padding). No chroma duplication needed — 4:4:4 means each pixel has
//! its own U/V.
//!
//! - u8 output: Y values are full 16-bit (0..65535), so
//!   `scale_y_u16_to_i16` is used (not `scale_y`, which would corrupt
//!   values > 32767). i32 chroma via `chroma_i16x8` at BITS=16.
//!
//! - u16 output: i64 chroma via `chroma_i64x4` to avoid i32 overflow at
//!   BITS=16/16. Y scaled via `scale_y_u16_i64`.
//!
//! For BE wire format (`BE = true`), each deinterleaved `uint16x8_t`
//! channel is byte-swapped via `bswap_u16x8_if_be::<true>` after the
//! `vld4q_u16` call.
//!
//! ## Tail
//!
//! `width % 8` (u8) / `width % 16` (u16) remaining pixels fall through
//! to `scalar::xv48_*`.

use super::*;
use crate::{KernelMatrix, row::scalar};

// ---- u8 RGB / RGBA output -----------------------------------------------

/// NEON XV48 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv48_to_rgb_or_rgba_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
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
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    // Y values are full u16 (0..65535); use i32 y_off for scale_y_u16_to_i16.
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    // Subtract chroma bias (32768 for 16-bit); fits i16 as the wrapping
    // -32768 pattern.
    let bias_v = vdupq_n_s16(bias as i16);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 XV48 quadruples (8 x 4 x u16 = 64 bytes).
      // vld4q_u16 deinterleaves: .0=U8, .1=Y8, .2=V8, .3=X8 (padding).
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      let u_u16 = bswap_u16x8_if_be::<BE>(q.0);
      let y_u16 = bswap_u16x8_if_be::<BE>(q.1);
      let v_u16 = bswap_u16x8_if_be::<BE>(q.2);
      // q.3 (X) is padding — discarded (no swap needed).

      // Reinterpret chroma as signed i16 (bias subtraction fits i16:
      // chroma ∈ [0,65535], bias=32768, so (chroma-bias) ∈ [-32768,32767]).
      let u_i16 = vreinterpretq_s16_u16(u_u16);
      let v_i16 = vreinterpretq_s16_u16(v_u16);

      // Subtract chroma bias (32768 for 16-bit).
      let u_sub = vsubq_s16(u_i16, bias_v);
      let v_sub = vsubq_s16(v_i16, bias_v);

      // Widen to i32x4 lo/hi for Q15 chroma-scale multiply.
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_sub));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_sub));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_sub));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_sub));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // 4:4:4 — no chroma duplication; all 8 lanes carry unique U/V per pixel.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Y: full u16 values → use scale_y_u16_to_i16 (NOT scale_y, which
      // would corrupt Y > 32767 by treating as signed).
      let y_scaled = scale_y_u16_to_i16(y_u16, y_off_v, y_scale_v, rnd_v);

      // Saturate-add Y + chroma, narrow to u8 with saturation.
      let r_u8 = vqmovun_s16_compat(vqaddq_s16(y_scaled, r_chroma));
      let g_u8 = vqmovun_s16_compat(vqaddq_s16(y_scaled, g_chroma));
      let b_u8 = vqmovun_s16_compat(vqaddq_s16(y_scaled, b_chroma));

      // Store 8 pixels.
      let off = x * bpp;
      if ALPHA {
        let alpha = vdup_n_u8(0xFF);
        vst4_u8(
          out.as_mut_ptr().add(off),
          uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
        );
      } else {
        vst3_u8(out.as_mut_ptr().add(off), uint8x8x3_t(r_u8, g_u8, b_u8));
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
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

// ---- u16 RGB / RGBA native-depth output ---------------------------------

/// NEON XV48 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x4`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "neon")]
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
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
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

    let mut x = 0usize;
    while x + 16 <= width {
      // Two vld4q_u16 loads: each deinterleaves 8 pixels.
      // Channel order: .0=U, .1=Y, .2=V, .3=X (padding).
      let q_lo = vld4q_u16(packed.as_ptr().add(x * 4));
      let q_hi = vld4q_u16(packed.as_ptr().add(x * 4 + 32));

      let y_lo_u16 = bswap_u16x8_if_be::<BE>(q_lo.1);
      let u_lo_u16 = bswap_u16x8_if_be::<BE>(q_lo.0);
      let v_lo_u16 = bswap_u16x8_if_be::<BE>(q_lo.2);

      let y_hi_u16 = bswap_u16x8_if_be::<BE>(q_hi.1);
      let u_hi_u16 = bswap_u16x8_if_be::<BE>(q_hi.0);
      let v_hi_u16 = bswap_u16x8_if_be::<BE>(q_hi.2);
      // q.3 (X) is padding — discarded.

      // Chroma: widen u16 → i32, subtract bias, apply c_scale (Q15).
      // lo half: pixels 0..7, split into lo0 (0..3) and lo1 (4..7).
      let u_lo0_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_lo_u16)));
      let u_lo1_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_lo_u16)));
      let v_lo0_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_lo_u16)));
      let v_lo1_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_lo_u16)));

      let u_d_lo0 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(u_lo0_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let u_d_lo1 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(u_lo1_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let v_d_lo0 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(v_lo0_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let v_d_lo1 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(v_lo1_i32, bias_v), c_scale_v),
        rnd_v,
      ));

      // hi half: pixels 8..15.
      let u_hi0_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_hi_u16)));
      let u_hi1_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_hi_u16)));
      let v_hi0_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_hi_u16)));
      let v_hi1_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_hi_u16)));

      let u_d_hi0 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(u_hi0_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let u_d_hi1 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(u_hi1_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let v_d_hi0 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(v_hi0_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let v_d_hi1 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(v_hi1_i32, bias_v), c_scale_v),
        rnd_v,
      ));

      // i64 chroma: 4 values each → i32x4 (via vmull_s32 widening).
      // 4:4:4 — no duplication needed; each pixel has unique U/V.
      let r_ch_lo0 = chroma_i64x4(cru, crv, u_d_lo0, v_d_lo0, rnd64);
      let r_ch_lo1 = chroma_i64x4(cru, crv, u_d_lo1, v_d_lo1, rnd64);
      let g_ch_lo0 = chroma_i64x4(cgu, cgv, u_d_lo0, v_d_lo0, rnd64);
      let g_ch_lo1 = chroma_i64x4(cgu, cgv, u_d_lo1, v_d_lo1, rnd64);
      let b_ch_lo0 = chroma_i64x4(cbu, cbv, u_d_lo0, v_d_lo0, rnd64);
      let b_ch_lo1 = chroma_i64x4(cbu, cbv, u_d_lo1, v_d_lo1, rnd64);

      let r_ch_hi0 = chroma_i64x4(cru, crv, u_d_hi0, v_d_hi0, rnd64);
      let r_ch_hi1 = chroma_i64x4(cru, crv, u_d_hi1, v_d_hi1, rnd64);
      let g_ch_hi0 = chroma_i64x4(cgu, cgv, u_d_hi0, v_d_hi0, rnd64);
      let g_ch_hi1 = chroma_i64x4(cgu, cgv, u_d_hi1, v_d_hi1, rnd64);
      let b_ch_hi0 = chroma_i64x4(cbu, cbv, u_d_hi0, v_d_hi0, rnd64);
      let b_ch_hi1 = chroma_i64x4(cbu, cbv, u_d_hi1, v_d_hi1, rnd64);

      // i64 Y scale: split each 8-lane Y into two i32x4 halves.
      let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_lo_u16)));
      let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_lo_u16)));
      let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_hi_u16)));
      let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_hi_u16)));

      let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
      let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
      let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
      let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

      // Y + chroma; vqmovun_s32 saturates i32 → u16 (clamps [0, 65535]).
      let r_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, r_ch_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, r_ch_lo1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, g_ch_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, g_ch_lo1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, b_ch_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, b_ch_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, r_ch_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, r_ch_hi1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, g_ch_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, g_ch_hi1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, b_ch_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, b_ch_hi1)),
      );

      // Store 16 pixels (two vst*q_u16 of 8 pixels each).
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

    // Scalar tail.
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

// ---- Luma u8 (8 px/iter) -----------------------------------------------

/// NEON XV48 → u8 luma. Y is quadruple element 1; `>> 8` brings the
/// 16-bit sample to 8-bit (high byte).
///
/// Byte-identical to `scalar::xv48_to_luma_row::<BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv48_to_luma_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      // Y is q.1. Apply BE byte-swap if needed before the shift.
      let y_raw = bswap_u16x8_if_be::<BE>(q.1);
      // vshrn_n_u16::<8> narrows (u16 >> 8) → u8x8, handling 8 lanes.
      let y_u8 = vshrn_n_u16::<8>(y_raw);
      vst1_u8(out.as_mut_ptr().add(x), y_u8);
      x += 8;
    }
    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_row::<BE>(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (8 px/iter) -----------------------------------------------

/// NEON XV48 → u16 luma (full 16-bit native — no shift). Y is quadruple
/// element 1.
///
/// Byte-identical to `scalar::xv48_to_luma_u16_row::<BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv48_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      // Y is q.1. Apply BE byte-swap if needed, then direct store.
      let y_u16 = bswap_u16x8_if_be::<BE>(q.1);
      vst1q_u16(out.as_mut_ptr().add(x), y_u16);
      x += 8;
    }
    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_u16_row::<BE>(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- XV48 → HSV (staged via a reused 8-bit RGB chunk) ------------------
//
// The NEON twin of the scalar `xv48_to_hsv_row` kernel. Rather than
// re-derive an HSV-specific register pipeline, it fills a small fixed
// reused **8-bit** RGB scratch (one `HSV_CHUNK`-pixel chunk at a time)
// using the EXISTING NEON `xv48_to_rgb_or_rgba_row::<false, BE>` kernel
// of this file — so the chunk filler IS the production 8-bit RGB kernel
// — then runs the NEON `rgb_to_hsv_row` on the chunk. This makes the
// result byte-identical to
// `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))` within the
// NEON tier, with no source-width RGB allocation. The scalar tail of the
// underlying RGB kernel handles widths below the SIMD block. The X slot
// is padding, dropped by the RGB kernel; HSV is colour-only.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared NEON driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing NEON RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the NEON [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. Byte-identical to
/// `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))` within the
/// NEON tier, with no source-width RGB allocation.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// NEON must be available, and `fill_rgb` must uphold the underlying RGB
/// kernel's safety contract for each chunk. Each of `h_out` / `s_out` /
/// `v_out` must be `>= width`.
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
    // SAFETY: NEON verified by the wrapper's `#[target_feature]`; the
    // chunk and the output sub-slices are all length `n`.
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

/// NEON: XV48 (packed 4:4:4, 16-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the reused-8-bit-RGB-chunk pattern over the
/// NEON [`xv48_to_rgb_or_rgba_row`] + [`rgb_to_hsv_row`]. Const-generic
/// over `BE`. Byte-identical to
/// `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))` within the
/// NEON tier. The padding X slot is dropped (HSV is colour-only).
///
/// # Safety
///
/// 1. The NEON feature must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
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
  // forwards the per-chunk sub-slices to the NEON XV48 RGB kernel under
  // the same contract (its own scalar tail covers small n).
  unsafe {
    xv48_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      xv48_to_rgb_or_rgba_row::<false, BE>(&packed[offset * 4..], rgb, n, matrix, full_range);
    });
  }
}
