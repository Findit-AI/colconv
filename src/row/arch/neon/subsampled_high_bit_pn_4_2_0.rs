use core::arch::aarch64::*;

use crate::{ColorMatrix, row::scalar};

use super::*;

/// NEON high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}: P010, P012)
/// → packed **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Differences
/// from [`yuv_420p_n_to_rgb_row`]:
/// - UV is semi‑planar interleaved (`U0, V0, U1, V1, …`), split in
///   one shot via `vld2q_u16` (returns separate U and V vectors).
/// - Each `u16` load is **right‑shifted by `16 - BITS`** — 6 for
///   P010, 4 for P012 — extracting the `BITS` active bits from the
///   high bits of each `u16` and clearing the low bits. The shift
///   runs via `vshlq_u16` with a negative loop‑invariant count so a
///   single kernel serves all supported bit depths.
///
/// After the shift, the rest of the pipeline is identical to the
/// low‑bit‑packed planar path — same `chroma_i16x8` / `scale_y` /
/// `chroma_dup` / `vst3q_u8` write, with `range_params_n::<BITS, 8>`
/// scaling.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] across all
/// supported `BITS` values.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
/// 4. `BITS` must be one of `{10, 12}`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON high-bit-packed semi-planar 4:2:0 → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as [`p_n_to_rgb_row`];
/// 4 bpp store with constant alpha.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON kernel for [`p_n_to_rgb_row`] (`ALPHA = false`,
/// `vst3q_u8`) and [`p_n_to_rgba_row`] (`ALPHA = true`, `vst4q_u8`
/// with constant `0xFF` alpha vector).
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // P016 (BITS=16) routes through `p16_to_rgb_or_rgba_row` (i64 chroma);
  // attempting `::<16, _>` here would silently overflow on high chroma.
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    // `vshlq_u16` performs right shift when the count is negative.
    // Count = -(16 - BITS) extracts the `BITS` active high bits.
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y pixels in two u16x8 loads, right-shifted by 16-BITS to
      // extract the active bits from the high-bit packing.
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);

      // Semi‑planar UV: `vld2q_u16` loads 16 interleaved `u16` elements
      // and returns (evens, odds) = (U, V) in one shot.
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = vshlq_u16(uv_pair.0, shr_count);
      let v_vec = vshlq_u16(uv_pair.1, shr_count);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );

      if ALPHA {
        let rgba = uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8);
        vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
      } else {
        let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
        vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_to_rgba_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_to_rgb_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **native‑depth `u16`** RGB (low‑bit‑packed output,
/// `yuv420p10le` / `yuv420p12le` convention — not P010/P012).
///
/// Same structure as [`super::neon::p_n_to_rgb_row`] up to the
/// chroma compute; the only differences are:
/// - `range_params_n::<BITS, BITS>` → larger scales targeting the
///   native‑depth output range.
/// - Clamp is explicit min/max to `[0, (1 << BITS) - 1]` via
///   [`clamp_u10`](crate::row::arch::neon::clamp_u10) — the helper
///   name is historical; the actual max is derived from `BITS` at
///   the call site (1023 for P010, 4095 for P012).
/// - Writes use two `vst3q_u16` calls per 16‑pixel block.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_u16_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p_n_to_rgba_row`] for native-depth `u16` output.
/// Alpha samples are `(1 << BITS) - 1` (opaque maximum at the input
/// bit depth) — matches `scalar::p_n_to_rgba_u16_row`. P016 has its
/// own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`], plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON Pn → native-depth `u16` kernel. `ALPHA = false` writes
/// RGB triples via `vst3q_u16`; `ALPHA = true` writes RGBA quads via
/// `vst4q_u16` with constant alpha `(1 << BITS) - 1`. P016 has its
/// own kernel family — never routed here.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = vshlq_u16(uv_pair.0, shr_count);
      let v_vec = vshlq_u16(uv_pair.1, shr_count);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      if ALPHA {
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, alpha_u16);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, alpha_u16);
        vst4q_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u16(out.as_mut_ptr().add(x * 4 + 32), rgba_hi);
      } else {
        let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
        let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
        vst3q_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u16(out.as_mut_ptr().add(x * 3 + 24), rgb_hi);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_to_rgba_u16_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_to_rgb_u16_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}
/// NEON P016 (semi-planar 16-bit) → packed 8-bit RGB.
///
/// UV is interleaved (`U0, V0, U1, V1, …`), split via `vld2q_u16`.
/// Byte-identical to [`scalar::p16_to_rgb_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON P016 (semi-planar 4:2:0, full 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgba_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON P016 kernel for [`p16_to_rgb_row`] (`ALPHA = false`,
/// `vst3q_u8`) and [`p16_to_rgba_row`] (`ALPHA = true`, `vst4q_u8`
/// with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = uv_pair.0;
      let v_vec = uv_pair.1;

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y_u16_to_i16(y_vec_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_vec_hi, y_off_v, y_scale_v, rnd_v);

      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );

      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON P016 (semi-planar 16-bit) → packed native-depth u16 RGB.
///
/// Byte-identical to [`scalar::p16_to_rgb_u16_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p16_to_rgba_row`] for native-depth `u16` output.
/// Alpha is `0xFFFF` — matches `scalar::p16_to_rgba_u16_row`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgba_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `vst3q_u16`; `ALPHA = true`
/// writes RGBA quads via `vst4q_u16` with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = vdupq_n_u16(0xFFFF);
    let rnd_v = vdupq_n_s32(RND);
    let rnd64 = vdupq_n_s64(RND as i64);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_d = vdup_n_s32(y_scale);
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
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = uv_pair.0;
      let v_vec = uv_pair.1;

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      let r_cd_lo0 = vzip1q_s32(r_ch_lo, r_ch_lo);
      let r_cd_lo1 = vzip2q_s32(r_ch_lo, r_ch_lo);
      let r_cd_hi0 = vzip1q_s32(r_ch_hi, r_ch_hi);
      let r_cd_hi1 = vzip2q_s32(r_ch_hi, r_ch_hi);
      let g_cd_lo0 = vzip1q_s32(g_ch_lo, g_ch_lo);
      let g_cd_lo1 = vzip2q_s32(g_ch_lo, g_ch_lo);
      let g_cd_hi0 = vzip1q_s32(g_ch_hi, g_ch_hi);
      let g_cd_hi1 = vzip2q_s32(g_ch_hi, g_ch_hi);
      let b_cd_lo0 = vzip1q_s32(b_ch_lo, b_ch_lo);
      let b_cd_lo1 = vzip2q_s32(b_ch_lo, b_ch_lo);
      let b_cd_hi0 = vzip1q_s32(b_ch_hi, b_ch_hi);
      let b_cd_hi1 = vzip2q_s32(b_ch_hi, b_ch_hi);

      let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_lo)));
      let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_lo)));
      let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_hi)));
      let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_hi)));
      let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
      let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
      let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
      let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

      let r_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, r_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, r_cd_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, r_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, r_cd_hi1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, g_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, g_cd_lo1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, g_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, g_cd_hi1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, b_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, b_cd_lo1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, b_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, b_cd_hi1)),
      );

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

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}
