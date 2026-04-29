use core::arch::aarch64::*;

use crate::{ColorMatrix, row::scalar};

use super::*;

/// NEON NV12 → packed RGB (UV-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_or_nv21_to_rgb_or_rgba_row_impl`]:
///
/// 1. **NEON must be available on the current CPU.** Direct callers
///    are responsible for verifying this; the dispatcher in
///    [`crate::row::nv12_to_rgb_row`] checks it.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `uv_half.len() >= width` (interleaved UV bytes, 2 per chroma pair).
/// 5. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON NV21 → packed RGB (VU-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_to_rgb_row`]; `vu_half` carries the same
/// number of bytes (`>= width`) but in V-then-U order per chroma
/// pair.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
      y, vu_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON NV12 → packed RGBA (R, G, B, `0xFF` per pixel). Same
/// contract as [`nv12_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8`. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv12_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes (one extra byte per pixel for the opaque
/// alpha).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON NV21 → packed RGBA (R, G, B, `0xFF` per pixel). Same
/// contract as [`nv21_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8`. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv21_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
      y, vu_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared NEON NV12/NV21 kernel at 3 bpp (RGB) or 4 bpp + opaque
/// alpha (RGBA). `SWAP_UV = false` selects NV12 (even byte = U, odd =
/// V); `SWAP_UV = true` selects NV21 (even = V, odd = U). `ALPHA =
/// true` writes via `vst4q_u8` with constant `0xFF` alpha; `ALPHA =
/// false` writes via `vst3q_u8`. Both const generics drive
/// compile-time monomorphization — branches are eliminated and each
/// of the four wrappers produces byte‑identical output to the scalar
/// reference.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.** The dispatcher
///    verifies this; direct callers are responsible.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (2 × (width / 2) interleaved bytes).
/// 5. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
///
/// Bounds are `debug_assert`-checked; release builds trust the caller
/// because the kernel uses unchecked pointer arithmetic (`vld1q_u8`,
/// `vld2_u8`, `vst3q_u8` / `vst4q_u8`).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn nv12_or_nv21_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu_half.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 16 <= width` loop
  // condition and the caller‑promised slice lengths above.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let mid128 = vdupq_n_s16(128);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      // 16 Y pixels → 8 chroma pairs. `vld2_u8` loads 16 interleaved
      // bytes and splits into (even-offset bytes, odd-offset bytes).
      // For NV12: even=U, odd=V. For NV21: even=V, odd=U, so we
      // swap which lane becomes `u_vec`. The `const SWAP_UV` makes
      // this a compile-time choice — no runtime branch.
      let uv_pair = vld2_u8(uv_or_vu_half.as_ptr().add(x));
      let (u_vec, v_vec) = if SWAP_UV {
        (uv_pair.1, uv_pair.0)
      } else {
        (uv_pair.0, uv_pair.1)
      };

      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_vec)), mid128);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_vec)), mid128);

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

    // Scalar tail for the 0..14 leftover pixels. Dispatch to the
    // matching scalar kernel based on SWAP_UV × ALPHA.
    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu_half[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      match (SWAP_UV, ALPHA) {
        (false, false) => {
          scalar::nv12_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, false) => {
          scalar::nv21_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (false, true) => {
          scalar::nv12_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, true) => {
          scalar::nv21_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
      }
    }
  }
}

/// NEON NV24 → packed RGB (UV-ordered, 4:4:4). Thin wrapper over
/// [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] with
/// `ALPHA = false` (so `out.len() >= width * 3` specializes to
/// `rgb_out.len() >= 3 * width`):
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`.
/// 3. `uv.len() >= 2 * width`.
/// 4. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, false>(y, uv, rgb_out, width, matrix, full_range);
  }
}

/// NEON NV42 → packed RGB (VU-ordered, 4:4:4). Thin wrapper over
/// [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv24_to_rgb_row`]; `vu` carries the same
/// `2 * width` bytes but in V-then-U order per chroma pair.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, false>(y, vu, rgb_out, width, matrix, full_range);
  }
}

/// NEON NV24 → packed RGBA (R, G, B, `0xFF` per pixel). Same
/// contract as [`nv24_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8`. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv24_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv24_to_rgba_row(
  y: &[u8],
  uv: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, true>(y, uv, rgba_out, width, matrix, full_range);
  }
}

/// NEON NV42 → packed RGBA (R, G, B, `0xFF` per pixel). Same
/// contract as [`nv42_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8`. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv42_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv42_to_rgba_row(
  y: &[u8],
  vu: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, true>(y, vu, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON NV24/NV42 kernel (4:4:4 semi-planar) at 3 bpp (RGB)
/// or 4 bpp + opaque alpha (RGBA). Unlike
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`], chroma is not
/// subsampled — one UV pair per Y pixel, so the chroma-duplication
/// step (`vzip*`) disappears: compute 16 chroma values per 16 Y
/// pixels directly.
///
/// `SWAP_UV = false` selects NV24 (even byte = U, odd = V);
/// `SWAP_UV = true` selects NV42 (even = V, odd = U). `ALPHA = true`
/// writes via `vst4q_u8` with constant `0xFF` alpha; `ALPHA = false`
/// writes via `vst3q_u8`. Both const generics drive compile-time
/// monomorphization.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`.
/// 3. `uv_or_vu.len() >= 2 * width` (one UV pair per Y pixel).
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
///
/// No width parity constraint (4:4:4).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn nv24_or_nv42_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 16 <= width` loop and
  // the caller-promised slice lengths.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let mid128 = vdupq_n_s16(128);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      // 16 Y pixels → 16 chroma pairs = 32 bytes. `vld2q_u8`
      // deinterleaves 32 bytes into (even-offset, odd-offset) — 16
      // bytes each.
      let uv_pair = vld2q_u8(uv_or_vu.as_ptr().add(x * 2));
      let (u_vec, v_vec) = if SWAP_UV {
        (uv_pair.1, uv_pair.0)
      } else {
        (uv_pair.0, uv_pair.1)
      };

      // Widen Y, U, V halves to i16x8.
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_vec))), mid128);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u_vec))), mid128);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_vec))), mid128);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_vec))), mid128);

      // Widen each i16x8 to two i32x4 halves for the Q15 multiply.
      let u_lo_a = vmovl_s16(vget_low_s16(u_lo_i16));
      let u_lo_b = vmovl_s16(vget_high_s16(u_lo_i16));
      let u_hi_a = vmovl_s16(vget_low_s16(u_hi_i16));
      let u_hi_b = vmovl_s16(vget_high_s16(u_hi_i16));
      let v_lo_a = vmovl_s16(vget_low_s16(v_lo_i16));
      let v_lo_b = vmovl_s16(vget_high_s16(v_lo_i16));
      let v_hi_a = vmovl_s16(vget_low_s16(v_hi_i16));
      let v_hi_b = vmovl_s16(vget_high_s16(v_hi_i16));

      // u_d / v_d = (u * c_scale + RND) >> 15 — i32x4 lanes.
      let u_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(v_hi_b, c_scale_v), rnd_v));

      // Compute chroma per channel — 8 results covering 8 Y pixels
      // per half (no duplication, since UV is 1:1 with Y).
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
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

    // Scalar tail for 0..15 leftover pixels.
    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu[x * 2..width * 2];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      match (SWAP_UV, ALPHA) {
        (false, false) => {
          scalar::nv24_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, false) => {
          scalar::nv42_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (false, true) => {
          scalar::nv24_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, true) => {
          scalar::nv42_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
      }
    }
  }
}
