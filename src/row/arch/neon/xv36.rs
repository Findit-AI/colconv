//! NEON kernels for XV36 packed YUV 4:4:4 12-bit family.
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), A(16)]`
//! little-endian, each holding a 12-bit sample MSB-aligned in the
//! high 12 bits (low 4 bits zero). The `X` prefix means the A slot
//! is **padding** — read by `vld4q_u16` but discarded. RGBA outputs
//! force α = max (`0xFF` u8 / `0x0FFF` u16).
//!
//! ## Per-iter pipeline (8 px / iter)
//!
//! `vld4q_u16` loads 8 quadruples in one call, returning a
//! `uint16x8x4_t` where `.0 = U`, `.1 = Y`, `.2 = V`, `.3 = A`
//! (padding). Each channel is right-shifted by 4 (`vshrq_n_u16::<4>`)
//! to bring the 12-bit value into `[0, 4095]`. No chroma duplication
//! needed — 4:4:4 means each pixel has its own U/V. Y values ≤ 4095
//! fit in i16, so `scale_y` is used (not `scale_y_u16_to_i16`).
//! The Q15 pipeline uses i32 chroma (`chroma_i16x8`) at BITS=12.
//!
//! For BE wire format (`BE = true`), each deinterleaved `uint16x8_t`
//! channel is byte-swapped via `bswap_u16x8_if_be::<true>` after the
//! `vld4q_u16` call.
//!
//! ## Tail
//!
//! `width % 8` remaining pixels fall through to `scalar::xv36_*`.

use super::*;
use crate::{KernelMatrix, row::scalar};

// ---- u8 RGB / RGBA output -----------------------------------------------

/// NEON XV36 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv36_to_rgb_or_rgba_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv36_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 8>(full_range);
  let bias = scalar::chroma_bias::<12>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 XV36 quadruples (8 x 4 x u16 = 64 bytes).
      // vld4q_u16 deinterleaves: .0=U8, .1=Y8, .2=V8, .3=A8 (padding).
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      // Apply BE byte-swap per-channel if needed.
      let u_raw = bswap_u16x8_if_be::<BE>(q.0);
      let y_raw = bswap_u16x8_if_be::<BE>(q.1);
      let v_raw = bswap_u16x8_if_be::<BE>(q.2);
      // q.3 (A) is padding — discarded (no swap needed).

      // Right-shift by 4 to drop the 4 padding LSBs → 12-bit range [0, 4095].
      let u_u16 = vshrq_n_u16::<4>(u_raw); // 8 lanes of U
      let y_u16 = vshrq_n_u16::<4>(y_raw); // 8 lanes of Y
      let v_u16 = vshrq_n_u16::<4>(v_raw); // 8 lanes of V

      // Reinterpret as signed i16 (values ≤ 4095 < 32767, safe).
      let u_i16 = vreinterpretq_s16_u16(u_u16);
      let y_i16 = vreinterpretq_s16_u16(y_u16);
      let v_i16 = vreinterpretq_s16_u16(v_u16);

      // Subtract chroma bias (2048 for 12-bit).
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

      // Y values ≤ 4095 fit in i16; use scale_y (NOT scale_y_u16_to_i16).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

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
      scalar::xv36_to_rgb_or_rgba_row::<ALPHA, BE>(
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

/// NEON XV36 → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 12-bit).
///
/// Byte-identical to `scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv36_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 12>(full_range);
  let bias = scalar::chroma_bias::<12>();
  const RND: i32 = 1 << 14;
  // 12-bit output max (low-bit-packed): [0, 0x0FFF].
  let out_max: i16 = 0x0FFF;
  let alpha_u16: u16 = 0x0FFF;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let alpha_v = vdupq_n_u16(alpha_u16);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      let u_raw = bswap_u16x8_if_be::<BE>(q.0);
      let y_raw = bswap_u16x8_if_be::<BE>(q.1);
      let v_raw = bswap_u16x8_if_be::<BE>(q.2);
      // q.3 (A) is padding — discarded.

      let u_u16 = vshrq_n_u16::<4>(u_raw);
      let y_u16 = vshrq_n_u16::<4>(y_raw);
      let v_u16 = vshrq_n_u16::<4>(v_raw);

      let u_i16 = vreinterpretq_s16_u16(u_u16);
      let y_i16 = vreinterpretq_s16_u16(y_u16);
      let v_i16 = vreinterpretq_s16_u16(v_u16);

      let u_sub = vsubq_s16(u_i16, bias_v);
      let v_sub = vsubq_s16(v_i16, bias_v);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_sub));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_sub));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_sub));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_sub));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x0FFF] (12-bit low-bit-packed output range).
      let r = clamp_u16_max(vqaddq_s16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max(vqaddq_s16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max(vqaddq_s16(y_scaled, b_chroma), zero_v, max_v);

      // Store 8 pixels.
      let off = x * bpp;
      if ALPHA {
        vst4q_u16(out.as_mut_ptr().add(off), uint16x8x4_t(r, g, b, alpha_v));
      } else {
        vst3q_u16(out.as_mut_ptr().add(off), uint16x8x3_t(r, g, b));
      }

      x += 8;
    }

    // Scalar tail.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>(
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

/// NEON XV36 → u8 luma. Y is quadruple element 1; `>> 8` brings the
/// 12-bit MSB-aligned sample to 8-bit (drops 4 padding LSBs + 4 more).
///
/// Byte-identical to `scalar::xv36_to_luma_row::<BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv36_to_luma_row<const BE: bool>(
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
      // Scalar does `packed[x*4+1] >> 8`; apply the same shift.
      // vshrn_n_u16::<8> narrows (u16 >> 8) → u8x8, handling 8 lanes.
      let y_u8 = vshrn_n_u16::<8>(y_raw);
      vst1_u8(out.as_mut_ptr().add(x), y_u8);
      x += 8;
    }
    // Scalar tail.
    if x < width {
      scalar::xv36_to_luma_row::<BE>(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (8 px/iter) -----------------------------------------------

/// NEON XV36 → u16 luma (low-bit-packed at 12-bit). Y is quadruple
/// element 1; `>> 4` drops the 4 padding LSBs to give a 12-bit value
/// in `[0, 4095]`.
///
/// Byte-identical to `scalar::xv36_to_luma_u16_row::<BE>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv36_to_luma_u16_row<const BE: bool>(
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
      // Y is q.1. Apply BE byte-swap if needed, then `>> 4`.
      let y_raw = bswap_u16x8_if_be::<BE>(q.1);
      let y_u16 = vshrq_n_u16::<4>(y_raw);
      vst1q_u16(out.as_mut_ptr().add(x), y_u16);
      x += 8;
    }
    // Scalar tail.
    if x < width {
      scalar::xv36_to_luma_u16_row::<BE>(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- XV36 → HSV (staged via a reused 8-bit RGB chunk) ------------------
//
// The NEON twin of the scalar `xv36_to_hsv_row` kernel. Rather than
// re-derive an HSV-specific register pipeline, it fills a small fixed
// reused **8-bit** RGB scratch (one `HSV_CHUNK`-pixel chunk at a time)
// using the EXISTING NEON `xv36_to_rgb_or_rgba_row::<false, BE>` kernel
// of this file — so the chunk filler IS the production 8-bit RGB kernel
// — then runs the NEON `rgb_to_hsv_row` on the chunk. This makes the
// result byte-identical to
// `rgb_to_hsv_row(xv36_to_rgb_or_rgba_row::<false, BE>(...))` within the
// NEON tier — the same 8-bit RGB intermediate the existing XV36 HSV path
// uses — with no source-width RGB allocation. The scalar tail of the
// underlying RGB kernel handles widths below the SIMD block. The A slot
// is padding, dropped by the RGB kernel; HSV is colour-only.
//
// The chunked driver is defined locally (mirroring the semi-planar
// high-bit `pn_hsv_via_rgb_chunks`) and gated `yuv-444-packed` with the
// rest of this file. Only `rgb_to_hsv_row` (ungated) is shared.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared NEON driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing NEON RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the NEON [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. Byte-identical to
/// `rgb_to_hsv_row(xv36_to_rgb_or_rgba_row::<false, BE>(...))` within the
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
unsafe fn xv36_hsv_via_rgb_chunks(
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

/// NEON: XV36 (packed 4:4:4, 12-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the reused-8-bit-RGB-chunk pattern over the
/// NEON [`xv36_to_rgb_or_rgba_row`] + [`rgb_to_hsv_row`]. Const-generic
/// over `BE`. Byte-identical to
/// `rgb_to_hsv_row(xv36_to_rgb_or_rgba_row::<false, BE>(...))` within the
/// NEON tier. The padding A slot is dropped (HSV is colour-only).
///
/// # Safety
///
/// 1. The NEON feature must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn xv36_to_hsv_row<const BE: bool>(
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
  // forwards the per-chunk sub-slices to the NEON XV36 RGB kernel under
  // the same contract (its own scalar tail covers small n).
  unsafe {
    xv36_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      xv36_to_rgb_or_rgba_row::<false, BE>(&packed[offset * 4..], rgb, n, matrix, full_range);
    });
  }
}
