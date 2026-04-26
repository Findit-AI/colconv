//! x86_64 SSE4.1 backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher as a fallback when AVX2 is
//! not available. SSE4.1 is a wide baseline on x86 (Penryn and newer,
//! ~2008), so this covers essentially all x86 hardware still in
//! production use that lacks AVX2.
//!
//! The kernel carries `#[target_feature(enable = "sse4.1")]` so its
//! intrinsics execute in an explicitly feature‑enabled context. The
//! shared [`super::x86_common::write_rgb_16`] helper uses SSSE3
//! (`_mm_shuffle_epi8`), which is a subset of SSE4.1 and thus
//! available here.
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON and AVX2 backends.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`_mm_loadu_si128`) + 8 U + 8 V (low 8 bytes of each
//!    via `_mm_loadl_epi64`).
//! 2. Widen U, V to i16x8 (`_mm_cvtepu8_epi16`), subtract 128.
//! 3. Split each i16x8 into two i32x4 halves and apply `c_scale`.
//! 4. Per channel C ∈ {R, G, B}: `(C_u*u_d + C_v*v_d + RND) >> 15` in
//!    i32, narrow‑saturate to i16x8.
//! 5. Nearest‑neighbor chroma upsample: `_mm_unpacklo_epi16` /
//!    `_mm_unpackhi_epi16` duplicate each of 8 chroma lanes into its
//!    pair slot → two i16x8 vectors covering 16 Y lanes. No lane‑
//!    crossing fixups are needed at 128 bits.
//! 6. Y path: widen low/high 8 Y to i16x8, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel.
//! 8. Saturate‑narrow to u8x16 per channel, then interleave via
//!    `super::x86_common::write_rgb_16`.

use core::arch::x86_64::*;

use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{
      rgb_to_hsv_16_pixels, swap_rb_16_pixels, write_rgb_16, write_rgb_u16_8, write_rgba_16,
    },
    scalar,
  },
};

/// SSE4.1 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **SSE4.1 must be available on the current CPU.** The dispatcher
///    in [`crate::row`] verifies this with
///    `is_x86_feature_detected!("sse4.1")` (runtime, std) or
///    `cfg!(target_feature = "sse4.1")` (compile‑time, no‑std).
///    Calling this kernel on a CPU without SSE4.1 triggers an
///    illegal‑instruction trap.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`_mm_loadu_si128`, `_mm_loadl_epi64`,
/// `_mm_storeu_si128` inside `write_rgb_16`).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked SSE4.1 availability + slice bounds — see
  // [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false>(y, u_half, v_half, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 YUV 4:2:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. SSE4.1 must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked SSE4.1 availability + slice bounds — see
  // [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true>(y, u_half, v_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared SSE4.1 kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false`,
/// [`write_rgb_16`]) and [`yuv_420_to_rgba_row`] (`ALPHA = true`,
/// [`write_rgba_16`] with constant `0xFF` alpha). Math is
/// byte-identical to `scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA>`;
/// only the per-block store helper differs. `const` generic
/// monomorphizes per call site, so the `if ALPHA` branches are
/// eliminated.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: SSE4.1 availability is the caller's obligation per the
  // `# Safety` section; the dispatcher in `crate::row` checks it.
  // All pointer adds below are bounded by the `while x + 16 <= width`
  // loop condition and the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let mid128 = _mm_set1_epi16(128);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = _mm_set1_epi8(-1); // 0xFF as i8

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 Y, 8 U, 8 V.
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let u_vec = _mm_loadl_epi64(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm_loadl_epi64(v_half.as_ptr().add(x / 2).cast());

      // Widen U/V to i16x8 and subtract 128.
      let u_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(u_vec), mid128);
      let v_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(v_vec), mid128);

      // Split each i16x8 into two i32x4 halves.
      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      // Per‑channel chroma → i16x8 (8 chroma values per channel).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: duplicate each of 8 chroma lanes
      // into its pair slot → two i16x8 vectors covering 16 Y lanes.
      // At 128 bits there's no lane‑crossing issue, so a plain unpack
      // is correct.
      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      // Y path: widen low/high 8 Y to i16x8, scale.
      let y_low_i16 = _mm_cvtepu8_epi16(y_vec);
      let y_high_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x16 per channel (no lane fixup needed at
      // 128 bits).
      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      if ALPHA {
        // 4‑way interleave → packed RGBA (64 bytes).
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        // 3‑way interleave → packed RGB (48 bytes).
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels.
    if x < width {
      scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut out[x * bpp..width * bpp],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Differences
/// from [`super::x86_sse41::yuv_420p_n_to_rgb_row`]:
/// - Samples are shifted right by `16 - BITS` (`_mm_srl_epi16`, with
///   a shift count computed from `BITS` once per call) instead of
///   AND‑masked — Pn's `BITS` active bits live in the HIGH `BITS` of
///   each `u16`.
/// - Semi‑planar UV is deinterleaved via [`deinterleave_uv_u16`]
///   below (one `_mm_shuffle_epi8` + two 64‑bit unpacks per 16
///   chroma elements).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    // High-bit-packed samples: shift right by `16 - BITS` to extract
    // the BITS-bit value. Loop-invariant, loaded once into the low 64b
    // of `shr_count` for `_mm_srl_epi16`.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // Y: two u16×8 loads, each shifted right by `16 - BITS`.
      let y_low_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), shr_count);

      // UV: two u16×8 loads of interleaved [U0,V0,U1,V1,...], then
      // deinterleave into separate u_vec + v_vec.
      let (u_vec, v_vec) = deinterleave_uv_u16(uv_half.as_ptr().add(x));
      let u_vec = _mm_srl_epi16(u_vec, shr_count);
      let v_vec = _mm_srl_epi16(v_vec, shr_count);

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

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      scalar::p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **native‑depth `u16`** RGB (low‑bit‑packed output, `yuv420pNle`
/// convention).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_u16_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    // High-bit-packed samples: shift right by `16 - BITS` to extract
    // the BITS-bit value.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), shr_count);
      let (u_vec, v_vec) = deinterleave_uv_u16(uv_half.as_ptr().add(x));
      let u_vec = _mm_srl_epi16(u_vec, shr_count);
      let v_vec = _mm_srl_epi16(v_vec, shr_count);

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

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      write_rgb_u16_8(r_lo, g_lo, b_lo, rgb_out.as_mut_ptr().add(x * 3));
      write_rgb_u16_8(r_hi, g_hi, b_hi, rgb_out.as_mut_ptr().add(x * 3 + 24));

      x += 16;
    }

    if x < width {
      scalar::p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// Deinterleaves 16 `u16` elements at `ptr` (`[U0, V0, U1, V1, …,
/// U7, V7]`) into `(u_vec, v_vec)` where each vector holds 8 packed
/// `u16` samples.
///
/// Each of the two 128‑bit loads is byte‑shuffled via
/// `_mm_shuffle_epi8` so that U samples land in the low 64 bits and
/// V samples in the high 64. Then `_mm_unpacklo_epi64` /
/// `_mm_unpackhi_epi64` combine the two halves into full u16×8
/// vectors. 2 loads + 2 shuffles + 2 unpacks = 6 ops.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes (16 `u16`
/// elements). Caller's `target_feature` must include SSSE3 (via
/// SSE4.1 or a superset).
#[inline(always)]
unsafe fn deinterleave_uv_u16(ptr: *const u16) -> (__m128i, __m128i) {
  unsafe {
    // Per‑chunk mask: pack even u16s (U's) into low 8 bytes, odd u16s
    // (V's) into high 8 bytes.
    let split_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    let chunk0 = _mm_loadu_si128(ptr.cast());
    let chunk1 = _mm_loadu_si128(ptr.add(8).cast());
    let s0 = _mm_shuffle_epi8(chunk0, split_mask);
    let s1 = _mm_shuffle_epi8(chunk1, split_mask);
    let u_vec = _mm_unpacklo_epi64(s0, s1);
    let v_vec = _mm_unpackhi_epi64(s0, s1);
    (u_vec, v_vec)
  }
}

/// SSE4.1 NV12 → packed RGB (UV-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = false`.
///
/// # Safety
///
/// SSE4.1 YUV 4:2:0 10‑bit → packed **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Mirrors
/// [`yuv_420_to_rgb_row`] with three structural differences:
/// - Two `_mm_loadu_si128` loads for Y (each pulls 8 `u16` = 16 bytes);
///   U/V each load 8 `u16` via one `_mm_loadu_si128`. No u8 widening —
///   the samples already occupy 16‑bit lanes.
/// - Chroma bias is 512 (10‑bit center).
/// - `range_params_n::<10, 8>` calibrates `y_scale` / `c_scale` to
///   map 10‑bit input directly to 8‑bit output in one Q15 shift.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::yuv_420p_n_to_rgb_row::<10>`].
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: SSE4.1 availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 16 <= width` loop condition and the caller‑
  // promised slice lengths.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let mask_v = _mm_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y = two `u16x8` loads; 8 U + 8 V = one load each. Each
      // load is AND‑masked to the low 10 bits (see matching comment
      // in [`crate::row::scalar::yuv_420p_n_to_rgb_row`]). Valid
      // 10‑bit samples ≤ 1023 pass through unchanged.
      let y_low_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_vec = _mm_and_si128(_mm_loadu_si128(u_half.as_ptr().add(x / 2).cast()), mask_v);
      let v_vec = _mm_and_si128(_mm_loadu_si128(v_half.as_ptr().add(x / 2).cast()), mask_v);

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

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      scalar::yuv_420p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 YUV 4:2:0 10‑bit → packed **10‑bit `u16`** RGB.
///
/// Block size 16 Y pixels per iteration; writes two 8‑pixel u16 RGB
/// chunks via [`write_rgb_u16_8`]. Shares all pre‑write math with the
/// u8 output path; the key differences:
/// - `range_params_n::<10, 10>` → `y_scale` / `c_scale` target the
///   10‑bit output range (values in `[0, 1023]` at Q15 exit).
/// - Clamp is explicit min/max to `[0, 1023]` — `_mm_packus_epi16`
///   would clip to u8, so we can't reuse it here.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<10>`].
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let mask_v = _mm_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // AND‑mask each load to the low 10 bits — critical for the
      // u16 output path since its larger `y_scale` / `c_scale`
      // (32768 for 10→10 full range) would let an out‑of‑range
      // sample push a `coeff * v_d` product past i16 range,
      // triggering information loss in the subsequent
      // `_mm_packs_epi32` narrow step inside `chroma_i16x8`.
      let y_low_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_vec = _mm_and_si128(_mm_loadu_si128(u_half.as_ptr().add(x / 2).cast()), mask_v);
      let v_vec = _mm_and_si128(_mm_loadu_si128(v_half.as_ptr().add(x / 2).cast()), mask_v);

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

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Per‑channel sum + clamp to [0, 1023].
      let r_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      // Two 8‑pixel u16 writes cover the 16‑pixel block.
      write_rgb_u16_8(r_lo, g_lo, b_lo, rgb_out.as_mut_ptr().add(x * 3));
      write_rgb_u16_8(r_hi, g_hi, b_hi, rgb_out.as_mut_ptr().add(x * 3 + 24));

      x += 16;
    }

    if x < width {
      scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// Clamps an i16x8 vector to `[0, max]` for native-depth u16 output
/// paths (10/12/14 bit). `_mm_packus_epi16` would clip to u8, so we
/// use explicit min/max with a caller-provided `max`.
#[inline(always)]
fn clamp_u16_max(v: __m128i, zero_v: __m128i, max_v: __m128i) -> __m128i {
  unsafe { _mm_min_epi16(_mm_max_epi16(v, zero_v), max_v) }
}

/// SSE4.1 YUV 4:4:4 planar 10/12/14-bit → packed **u8** RGB.
/// Const-generic over `BITS ∈ {10, 12, 14}`.
///
/// Block size: 16 pixels per iteration (same as the 4:2:0 sibling).
/// Differs from [`yuv_420p_n_to_rgb_row`] by loading full-width U/V
/// (16 samples each) and computing two chroma-per-Y-half vectors,
/// skipping the horizontal chroma-duplication step (4:4:4 chroma is
/// 1:1 with Y, not paired).
///
/// # Numerical contract
///
/// Byte-identical to [`scalar::yuv_444p_n_to_rgb_row::<BITS>`].
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let mask_v = _mm_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y + 16 U + 16 V per iter. Full-width chroma load (two
      // u16x8 each) — no horizontal duplication needed.
      let y_low_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_lo_vec = _mm_and_si128(_mm_loadu_si128(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm_and_si128(_mm_loadu_si128(u.as_ptr().add(x + 8).cast()), mask_v);
      let v_lo_vec = _mm_and_si128(_mm_loadu_si128(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm_and_si128(_mm_loadu_si128(v.as_ptr().add(x + 8).cast()), mask_v);

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias_v);

      // Widen each i16x8 → two i32x4 (4+4 per half).
      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      // Two chroma_i16x8 calls per channel produce 16 chroma values.
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      scalar::yuv_444p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 YUV 4:4:4 planar 10/12/14-bit → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {10, 12, 14}`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Compile-time guard — `out_max = ((1 << BITS) - 1) as i16` below
  // silently wraps to -1 at BITS=16, corrupting the u16 clamp. The
  // dedicated 16-bit u16-output path is `yuv_444p16_to_rgb_u16_row`.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let mask_v = _mm_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_lo_vec = _mm_and_si128(_mm_loadu_si128(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm_and_si128(_mm_loadu_si128(u.as_ptr().add(x + 8).cast()), mask_v);
      let v_lo_vec = _mm_and_si128(_mm_loadu_si128(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm_and_si128(_mm_loadu_si128(v.as_ptr().add(x + 8).cast()), mask_v);

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      write_rgb_u16_8(r_lo, g_lo, b_lo, rgb_out.as_mut_ptr().add(x * 3));
      write_rgb_u16_8(r_hi, g_hi, b_hi, rgb_out.as_mut_ptr().add(x * 3 + 24));

      x += 16;
    }

    if x < width {
      scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 YUV 4:4:4 planar **16-bit** → packed **u8** RGB. Stays on
/// the i32 Q15 pipeline — output-range scaling keeps `coeff × u_d`
/// within i32 for u8 output.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_444p16_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let y_high = _mm_loadu_si128(y.as_ptr().add(x + 8).cast());
      let u_lo_vec = _mm_loadu_si128(u.as_ptr().add(x).cast());
      let u_hi_vec = _mm_loadu_si128(u.as_ptr().add(x + 8).cast());
      let v_lo_vec = _mm_loadu_si128(v.as_ptr().add(x).cast());
      let v_hi_vec = _mm_loadu_si128(v.as_ptr().add(x + 8).cast());

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias16_v);

      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y_u16(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = _mm_packus_epi16(r_lo, r_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let b_u8 = _mm_packus_epi16(b_lo, b_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }

    if x < width {
      scalar::yuv_444p16_to_rgb_row(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 YUV 4:4:4 planar **16-bit** → packed **u16** RGB.
///
/// i64 chroma arithmetic via `_mm_mul_epi32` + `srai64_15` bias trick.
/// Processes 8 pixels per iteration (i64 width constraint). Final
/// saturation via `_mm_packus_epi32` (signed i32 → u16).
///
/// Differs from [`yuv_420p16_to_rgb_u16_row`] by loading 8 full-width
/// U/V (vs 4 half-width), computing 8 chroma values (vs 4 + dup), and
/// skipping the chroma-duplication step.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_444p16_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi64x(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 pixels per iter. 4:4:4 with 16-bit i64 chroma → load 8 Y,
      // 8 U, 8 V; compute 8 chroma values per channel.
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let u_vec = _mm_loadu_si128(u.as_ptr().add(x).cast());
      let v_vec = _mm_loadu_si128(v.as_ptr().add(x).cast());

      let u_i16 = _mm_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias16_v);

      let rnd32_v = _mm_set1_epi32(1 << 14);
      // Two i32x4 per chroma channel (low 4 + high 4 samples).
      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd32_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd32_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd32_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd32_v));

      // i64 chroma: _mm_mul_epi32 uses even-indexed i32 lanes.
      let u_d_lo_even = u_d_lo;
      let u_d_lo_odd = _mm_shuffle_epi32::<0xF5>(u_d_lo);
      let v_d_lo_even = v_d_lo;
      let v_d_lo_odd = _mm_shuffle_epi32::<0xF5>(v_d_lo);
      let u_d_hi_even = u_d_hi;
      let u_d_hi_odd = _mm_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_hi_even = v_d_hi;
      let v_d_hi_odd = _mm_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_even = chroma_i64x2(cru, crv, u_d_lo_even, v_d_lo_even, rnd_v);
      let r_ch_lo_odd = chroma_i64x2(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let r_ch_hi_even = chroma_i64x2(cru, crv, u_d_hi_even, v_d_hi_even, rnd_v);
      let r_ch_hi_odd = chroma_i64x2(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let g_ch_lo_even = chroma_i64x2(cgu, cgv, u_d_lo_even, v_d_lo_even, rnd_v);
      let g_ch_lo_odd = chroma_i64x2(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let g_ch_hi_even = chroma_i64x2(cgu, cgv, u_d_hi_even, v_d_hi_even, rnd_v);
      let g_ch_hi_odd = chroma_i64x2(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let b_ch_lo_even = chroma_i64x2(cbu, cbv, u_d_lo_even, v_d_lo_even, rnd_v);
      let b_ch_lo_odd = chroma_i64x2(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let b_ch_hi_even = chroma_i64x2(cbu, cbv, u_d_hi_even, v_d_hi_even, rnd_v);
      let b_ch_hi_odd = chroma_i64x2(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_v);

      // Reassemble i64x2 (even + odd) → i32x4. Each chroma_i64x2 pair
      // produces Q15 chroma in the low 32 bits of each i64.
      let r_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_lo_even, r_ch_lo_odd),
        _mm_unpackhi_epi32(r_ch_lo_even, r_ch_lo_odd),
      );
      let r_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_hi_even, r_ch_hi_odd),
        _mm_unpackhi_epi32(r_ch_hi_even, r_ch_hi_odd),
      );
      let g_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_lo_even, g_ch_lo_odd),
        _mm_unpackhi_epi32(g_ch_lo_even, g_ch_lo_odd),
      );
      let g_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_hi_even, g_ch_hi_odd),
        _mm_unpackhi_epi32(g_ch_hi_even, g_ch_hi_odd),
      );
      let b_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_lo_even, b_ch_lo_odd),
        _mm_unpackhi_epi32(b_ch_lo_even, b_ch_lo_odd),
      );
      let b_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_hi_even, b_ch_hi_odd),
        _mm_unpackhi_epi32(b_ch_hi_even, b_ch_hi_odd),
      );

      // Y: 8 pixels, scale_y16_i64 in pairs (even + odd).
      let y_lo_pair = _mm_cvtepu16_epi32(y_vec);
      let y_hi_pair = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(y_vec));
      let y_lo_sub = _mm_sub_epi32(y_lo_pair, y_off_v);
      let y_hi_sub = _mm_sub_epi32(y_hi_pair, y_off_v);
      let y_lo_even = scale_y16_i64(y_lo_sub, y_scale_v, rnd_v);
      let y_lo_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_lo_sub), y_scale_v, rnd_v);
      let y_hi_even = scale_y16_i64(y_hi_sub, y_scale_v, rnd_v);
      let y_hi_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_hi_sub), y_scale_v, rnd_v);
      let y_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_lo_even, y_lo_odd),
        _mm_unpackhi_epi32(y_lo_even, y_lo_odd),
      );
      let y_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_hi_even, y_hi_odd),
        _mm_unpackhi_epi32(y_hi_even, y_hi_odd),
      );

      // Add Y + chroma (no dup — 4:4:4 is 1:1). Saturate to u16.
      let r_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, r_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, r_ch_hi_i32),
      );
      let g_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, g_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, g_ch_hi_i32),
      );
      let b_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, b_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, b_ch_hi_i32),
      );

      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }

    if x < width {
      scalar::yuv_444p16_to_rgb_u16_row(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 NV12 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_or_nv21_to_rgb_or_rgba_row_impl`]:
///
/// 1. **SSE4.1 must be available on the current CPU.** Direct
///    callers are responsible for verifying this; the dispatcher in
///    [`crate::row::nv12_to_rgb_row`] checks it.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `uv_half.len() >= width` (interleaved UV bytes, 2 per chroma pair).
/// 5. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 NV21 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_to_rgb_row`]; `vu_half` carries the same
/// number of bytes (`>= width`) but in V-then-U order per chroma
/// pair.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
      y, vu_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 NV12 → packed RGBA. Same contract as [`nv12_to_rgb_row`]
/// but writes 4 bytes per pixel via [`write_rgba_16`].
/// `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv12_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes (one extra byte per pixel for the opaque
/// alpha).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 NV21 → packed RGBA. Same contract as [`nv21_to_rgb_row`]
/// but writes 4 bytes per pixel via [`write_rgba_16`].
/// `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv21_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
      y, vu_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared SSE4.1 NV12/NV21 kernel at 3 bpp (RGB) or 4 bpp + opaque
/// alpha (RGBA). `SWAP_UV = false` → NV12; `SWAP_UV = true` → NV21.
/// `ALPHA = true` writes via [`write_rgba_16`]; `ALPHA = false` via
/// [`write_rgb_16`]. Both const generics drive compile-time
/// monomorphization.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (2 × (width / 2) interleaved bytes).
/// 5. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn nv12_or_nv21_to_rgb_or_rgba_row_impl<
  const SWAP_UV: bool,
  const ALPHA: bool,
>(
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

  // SAFETY: SSE4.1 availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 16 <= width` condition and
  // the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let mid128 = _mm_set1_epi16(128);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm_set1_epi8(-1); // 0xFF as i8

    // Deinterleave masks: `even_mask` pulls even-offset bytes into
    // lanes 0..7, `odd_mask` pulls odd-offset bytes. For NV12 that's
    // (U, V); for NV21 the roles swap.
    let even_mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let odd_mask = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      // 16 Y pixels correspond to 8 chroma pairs = 16 interleaved
      // bytes at offset `x` in the chroma row.
      let uv_vec = _mm_loadu_si128(uv_or_vu_half.as_ptr().add(x).cast());
      let (u_vec, v_vec) = if SWAP_UV {
        (
          _mm_shuffle_epi8(uv_vec, odd_mask),
          _mm_shuffle_epi8(uv_vec, even_mask),
        )
      } else {
        (
          _mm_shuffle_epi8(uv_vec, even_mask),
          _mm_shuffle_epi8(uv_vec, odd_mask),
        )
      };

      let u_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(u_vec), mid128);
      let v_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(v_vec), mid128);

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

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_low_i16 = _mm_cvtepu8_epi16(y_vec);
      let y_high_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

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

/// SSE4.1 NV24 → packed RGB (UV-ordered, 4:4:4). Thin wrapper over
/// [`nv24_or_nv42_to_rgb_row_impl`] with `SWAP_UV = false`.
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "sse4.1")]
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
    nv24_or_nv42_to_rgb_row_impl::<false>(y, uv, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 NV42 → packed RGB (VU-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "sse4.1")]
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
    nv24_or_nv42_to_rgb_row_impl::<true>(y, vu, rgb_out, width, matrix, full_range);
  }
}

/// Shared SSE4.1 NV24/NV42 kernel (4:4:4 semi-planar). Unlike
/// [`nv12_or_nv21_to_rgb_row_impl`], chroma is not subsampled — one
/// UV pair per Y pixel. Per 16 Y pixels, load 32 UV bytes (two
/// `_mm_loadu_si128`), deinterleave, compute 16 chroma values per
/// channel directly, and skip the `_mm_unpacklo/hi_epi16` chroma
/// duplication. No width parity constraint.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `y.len() >= width`.
/// 3. `uv_or_vu.len() >= 2 * width`.
/// 4. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn nv24_or_nv42_to_rgb_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: SSE4.1 availability is the caller's obligation; all
  // pointer adds below are bounded by the `while x + 16 <= width`
  // loop and the caller-promised slice lengths.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let mid128 = _mm_set1_epi16(128);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    // Shuffle masks to deinterleave 16 UV bytes into 8 U + 8 V (low
    // lanes). The upper 8 lanes are zeroed by `_mm_shuffle_epi8`
    // whenever the mask byte has its high bit set — `-1` (= `0xFF`)
    // written here as a signed `i8` literal triggers that behavior.
    let even_mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let odd_mask = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      // 16 Y pixels → 32 UV bytes.
      let uv_lo = _mm_loadu_si128(uv_or_vu.as_ptr().add(x * 2).cast());
      let uv_hi = _mm_loadu_si128(uv_or_vu.as_ptr().add(x * 2 + 16).cast());
      let (u_lo_bytes, v_lo_bytes, u_hi_bytes, v_hi_bytes) = if SWAP_UV {
        (
          _mm_shuffle_epi8(uv_lo, odd_mask),
          _mm_shuffle_epi8(uv_lo, even_mask),
          _mm_shuffle_epi8(uv_hi, odd_mask),
          _mm_shuffle_epi8(uv_hi, even_mask),
        )
      } else {
        (
          _mm_shuffle_epi8(uv_lo, even_mask),
          _mm_shuffle_epi8(uv_lo, odd_mask),
          _mm_shuffle_epi8(uv_hi, even_mask),
          _mm_shuffle_epi8(uv_hi, odd_mask),
        )
      };

      // Widen U/V halves to i16x8 (cvtepu8_epi16 on low 8 bytes).
      let u_lo_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(u_lo_bytes), mid128);
      let u_hi_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(u_hi_bytes), mid128);
      let v_lo_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(v_lo_bytes), mid128);
      let v_hi_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(v_hi_bytes), mid128);

      // Split each i16x8 into two i32x4 halves.
      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      // u_d / v_d = (u * c_scale + RND) >> 15.
      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      // 16 chroma per channel (two `chroma_i16x8` per channel, no
      // duplication).
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      // Y: widen 16 u8 to two i16x8, apply y_off / y_scale.
      let y_low_i16 = _mm_cvtepu8_epi16(y_vec);
      let y_high_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma, then saturating-narrow to u8x16.
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      if SWAP_UV {
        scalar::nv42_to_rgb_row(
          &y[x..width],
          &uv_or_vu[x * 2..width * 2],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      } else {
        scalar::nv24_to_rgb_row(
          &y[x..width],
          &uv_or_vu[x * 2..width * 2],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      }
    }
  }
}

/// SSE4.1 YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked SSE4.1 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 YUV 4:4:4 planar → packed **RGBA** (8-bit). Same contract
/// as [`yuv_444_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_16`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_444_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked SSE4.1 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared SSE4.1 YUV 4:4:4 kernel for [`yuv_444_to_rgb_row`]
/// (`ALPHA = false`, [`write_rgb_16`]) and [`yuv_444_to_rgba_row`]
/// (`ALPHA = true`, [`write_rgba_16`] with constant `0xFF` alpha).
/// Math is byte-identical to
/// `scalar::yuv_444_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let mid128 = _mm_set1_epi16(128);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm_set1_epi8(-1); // 0xFF as i8

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      // 4:4:4: 16 U + 16 V, one load each. No deinterleave.
      let u_vec = _mm_loadu_si128(u.as_ptr().add(x).cast());
      let v_vec = _mm_loadu_si128(v.as_ptr().add(x).cast());

      // Widen each half of U / V to i16x8, subtract 128.
      let u_lo_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(u_vec), mid128);
      let u_hi_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_srli_si128::<8>(u_vec)), mid128);
      let v_lo_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(v_vec), mid128);
      let v_hi_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_srli_si128::<8>(v_vec)), mid128);

      // Split each i16x8 into two i32x4 halves.
      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_low_i16 = _mm_cvtepu8_epi16(y_vec);
      let y_high_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA {
        scalar::yuv_444_to_rgba_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yuv_444_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- helpers (inlined into the target_feature‑enabled caller) ----------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: __m128i) -> __m128i {
  unsafe { _mm_srai_epi32::<15>(v) }
}

/// Computes one i16x8 chroma channel vector from the 4 × i32x4 chroma
/// inputs. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, then saturating‑packs
/// to i16x8. No lane fixup needed at 128 bits.
#[inline(always)]
fn chroma_i16x8(
  cu: __m128i,
  cv: __m128i,
  u_d_lo: __m128i,
  v_d_lo: __m128i,
  u_d_hi: __m128i,
  v_d_hi: __m128i,
  rnd: __m128i,
) -> __m128i {
  unsafe {
    let lo = _mm_srai_epi32::<15>(_mm_add_epi32(
      _mm_add_epi32(_mm_mullo_epi32(cu, u_d_lo), _mm_mullo_epi32(cv, v_d_lo)),
      rnd,
    ));
    let hi = _mm_srai_epi32::<15>(_mm_add_epi32(
      _mm_add_epi32(_mm_mullo_epi32(cu, u_d_hi), _mm_mullo_epi32(cv, v_d_hi)),
      rnd,
    ));
    _mm_packs_epi32(lo, hi)
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x8 vector,
/// returned as i16x8.
#[inline(always)]
fn scale_y(y_i16: __m128i, y_off_v: __m128i, y_scale_v: __m128i, rnd: __m128i) -> __m128i {
  unsafe {
    let shifted = _mm_sub_epi16(y_i16, y_off_v);
    let lo_i32 = _mm_cvtepi16_epi32(shifted);
    let hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(shifted));
    let lo_scaled = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(lo_i32, y_scale_v), rnd));
    let hi_scaled = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(hi_i32, y_scale_v), rnd));
    _mm_packs_epi32(lo_scaled, hi_scaled)
  }
}

// ===== 16-bit YUV → RGB helpers =========================================

/// `(Y_u16 - y_off) * y_scale + RND >> 15` for full u16 Y samples
/// (unsigned widening via `_mm_cvtepu16_epi32`). Returns i16x8.
#[inline(always)]
fn scale_y_u16(y_u16: __m128i, y_off_v: __m128i, y_scale_v: __m128i, rnd_v: __m128i) -> __m128i {
  unsafe {
    let y_lo_i32 = _mm_sub_epi32(_mm_cvtepu16_epi32(y_u16), y_off_v);
    let y_hi_u16 = _mm_srli_si128::<8>(y_u16);
    let y_hi_i32 = _mm_sub_epi32(_mm_cvtepu16_epi32(y_hi_u16), y_off_v);
    let lo = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(y_lo_i32, y_scale_v), rnd_v));
    let hi = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(y_hi_i32, y_scale_v), rnd_v));
    _mm_packs_epi32(lo, hi)
  }
}

/// `srai64_15(x) = srli64_15(x + 2^32) - 2^17` — arithmetic right-shift
/// by 15 for i64x2. Mathematically valid for `x >= -2^32` (i.e.
/// `x + 2^32 >= 0` so the unsigned shift matches the signed one).
/// No `_mm_srai_epi64` in SSE4.1, so AVX2/AVX-512 u16 paths delegate
/// to the SSE4.1 kernel that uses this helper.
///
/// Callers: both u16 callers stay strictly inside this domain.
/// - **Chroma sum** `c_u * u_d + c_v * v_d + RND` reaches at most
///   `|c|_max * |u_d|_max ≈ 61655 * 37449 ≈ 2.31·10⁹` across all
///   supported matrices at 16-bit limited range (Bt2020Ncl b_u is
///   the tightest case). `|x| ≤ 2.31·10⁹ < 2^32`.
/// - **Y scale** `(y - y_off) * y_scale + RND` reaches at most
///   `61439 * ~38290 ≈ 2.35·10⁹` at 16-bit limited range. Still
///   `|x| < 2^32`.
///
/// The scalar comment's pessimistic `~4.3·10⁹` upper bound
/// overcounts by summing `|c_u|+|c_v|` against the same worst-case
/// chroma; in practice only one of the two is near the peak per
/// output channel.
#[inline(always)]
fn srai64_15(x: __m128i) -> __m128i {
  unsafe {
    // Bias x up by 2^32 so the unsigned shift is correct, then undo the
    // extra 2^17 (= 2^32 >> 15) introduced by the bias.
    let biased = _mm_add_epi64(x, _mm_set1_epi64x(1i64 << 32));
    let shifted = _mm_srli_epi64::<15>(biased);
    _mm_sub_epi64(shifted, _mm_set1_epi64x(1i64 << 17))
  }
}

/// Computes one i64x2 chroma channel from 2 × i64 (u_d, v_d) inputs.
/// Returns i64x2 with [`srai64_15`]-shifted results.
#[inline(always)]
fn chroma_i64x2(cu: __m128i, cv: __m128i, u_d: __m128i, v_d: __m128i, rnd_v: __m128i) -> __m128i {
  unsafe {
    srai64_15(_mm_add_epi64(
      _mm_add_epi64(_mm_mul_epi32(cu, u_d), _mm_mul_epi32(cv, v_d)),
      rnd_v,
    ))
  }
}

/// `(y_minus_off * y_scale + RND) >> 15` in i64 via `_mm_mul_epi32` (even
/// lanes). Caller must supply an i32x4 that is already `Y - y_off`.
/// Returns i64x2 for the two even-indexed lanes.
#[inline(always)]
fn scale_y16_i64(y_minus_off: __m128i, y_scale_v: __m128i, rnd_v: __m128i) -> __m128i {
  unsafe { srai64_15(_mm_add_epi64(_mm_mul_epi32(y_minus_off, y_scale_v), rnd_v)) }
}

// ===== 16-bit planar (YUV420P16) → RGB ===================================

/// SSE4.1 YUV 4:2:0 16-bit → packed **8-bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. i32 chroma
/// arithmetic suffices for the u8 output target (small `c_scale ≈ 146`).
/// Y is unsigned-widened via `_mm_cvtepu16_epi32` (values can exceed 32767).
/// UV centering subtracts 32768 using the `0x8000` wrapping trick
/// (`_mm_sub_epi16(v, _mm_set1_epi16(-32768i16))`).
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_420p16_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    // Subtract 32768 (0x8000) via wrapping: -32768i16 as bits = 0x8000.
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let y_high = _mm_loadu_si128(y.as_ptr().add(x + 8).cast());
      let u_vec = _mm_loadu_si128(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm_loadu_si128(v_half.as_ptr().add(x / 2).cast());

      // Center UV: subtract 32768 (wrapping i16 trick).
      let u_i16 = _mm_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias16_v);

      // Scale UV to u8 space via i32 Q15 arithmetic.
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

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y_u16(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = _mm_packus_epi16(r_lo, r_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let b_u8 = _mm_packus_epi16(b_lo, b_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }

    if x < width {
      scalar::yuv_420p16_to_rgb_row(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 YUV 4:2:0 16-bit → packed **16-bit** RGB.
///
/// i64 chroma arithmetic via `_mm_mul_epi32` + `srai64_15` bias trick.
/// Processes 8 Y pixels (4 chroma pairs) per iteration (i64 width constraint).
/// Final saturation via `_mm_packus_epi32` (signed i32 → u16).
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_row`] but `rgb_out` is `&mut [u16]`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_420p16_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi64x(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 Y and 4 U/V; process 4 chroma pairs → 8 pixels.
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      // Load 4 U and 4 V u16 values into the low 64 bits of each vector.
      let u_vec4 = _mm_loadl_epi64(u_half.as_ptr().add(x / 2).cast());
      let v_vec4 = _mm_loadl_epi64(v_half.as_ptr().add(x / 2).cast());

      // Center UV: subtract 32768 (wrapping i16 trick).
      let u_i16 = _mm_sub_epi16(u_vec4, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec4, bias16_v);

      // Scale UV in i32 (fits: |u_centered| ≤ 32768, c_scale ≤ 38302).
      let rnd32_v = _mm_set1_epi32(1 << 14);
      let u_i32 = _mm_cvtepi16_epi32(u_i16);
      let v_i32 = _mm_cvtepi16_epi32(v_i16);
      let u_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_i32, c_scale_v), rnd32_v));
      let v_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_i32, c_scale_v), rnd32_v));

      // Chroma in i64x2 pairs (even / odd lanes of u_d / v_d).
      // _mm_mul_epi32 uses even-indexed i32 lanes → result is i64x2.
      let u_d_even = u_d; // lanes [0,_,2,_] used by _mm_mul_epi32
      let v_d_even = v_d;
      let u_d_odd = _mm_shuffle_epi32::<0xF5>(u_d); // [1,1,3,3] → odd lanes to even
      let v_d_odd = _mm_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x2(cru, crv, u_d_even, v_d_even, rnd_v);
      let r_ch_odd = chroma_i64x2(cru, crv, u_d_odd, v_d_odd, rnd_v);
      let g_ch_even = chroma_i64x2(cgu, cgv, u_d_even, v_d_even, rnd_v);
      let g_ch_odd = chroma_i64x2(cgu, cgv, u_d_odd, v_d_odd, rnd_v);
      let b_ch_even = chroma_i64x2(cbu, cbv, u_d_even, v_d_even, rnd_v);
      let b_ch_odd = chroma_i64x2(cbu, cbv, u_d_odd, v_d_odd, rnd_v);

      // Reassemble i64x2 pairs to i32x4: unpacklo_epi32 interleaves
      // low 32 bits; unpacklo_epi64 joins the two halves.
      let r_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_even, r_ch_odd),
        _mm_unpackhi_epi32(r_ch_even, r_ch_odd),
      );
      let g_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_even, g_ch_odd),
        _mm_unpackhi_epi32(g_ch_even, g_ch_odd),
      );
      let b_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_even, b_ch_odd),
        _mm_unpackhi_epi32(b_ch_even, b_ch_odd),
      );

      // Duplicate each chroma value for 2 Y pixels per chroma pair.
      // unpacklo_epi32([r0,r1,r2,r3], same) → [r0,r0,r1,r1]
      let r_dup_lo = _mm_unpacklo_epi32(r_ch_i32, r_ch_i32);
      let r_dup_hi = _mm_unpackhi_epi32(r_ch_i32, r_ch_i32);
      let g_dup_lo = _mm_unpacklo_epi32(g_ch_i32, g_ch_i32);
      let g_dup_hi = _mm_unpackhi_epi32(g_ch_i32, g_ch_i32);
      let b_dup_lo = _mm_unpacklo_epi32(b_ch_i32, b_ch_i32);
      let b_dup_hi = _mm_unpackhi_epi32(b_ch_i32, b_ch_i32);

      // Scale Y in i64 via pairs: process pixels 0-1, 2-3, 4-5, 6-7.
      // Load pairs of Y as 32-bit lanes for _mm_mul_epi32.
      let y_lo_pair = _mm_cvtepu16_epi32(y_vec); // [y0,y1,y2,y3] as i32
      let y_hi_pair = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(y_vec)); // [y4,y5,y6,y7]

      let y_lo_sub = _mm_sub_epi32(y_lo_pair, y_off_v);
      let y_hi_sub = _mm_sub_epi32(y_hi_pair, y_off_v);

      // Scale Y pairs in i64 via _mm_mul_epi32 (even lanes).
      // y_lo_sub = [y0-off, y1-off, y2-off, y3-off]
      // even lanes: y0-off and y2-off
      let y_lo_even = scale_y16_i64(y_lo_sub, y_scale_v, rnd_v);
      let y_lo_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_lo_sub), y_scale_v, rnd_v);
      let y_hi_even = scale_y16_i64(y_hi_sub, y_scale_v, rnd_v);
      let y_hi_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_hi_sub), y_scale_v, rnd_v);

      // Reassemble Y i64x2 pairs to i32x4.
      let y_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_lo_even, y_lo_odd),
        _mm_unpackhi_epi32(y_lo_even, y_lo_odd),
      );
      let y_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_hi_even, y_hi_odd),
        _mm_unpackhi_epi32(y_hi_even, y_hi_odd),
      );

      // Add Y + chroma, saturate to u16 via _mm_packus_epi32.
      let r_lo_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, r_dup_lo),
        _mm_add_epi32(y_hi_i32, r_dup_hi),
      );
      let g_lo_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, g_dup_lo),
        _mm_add_epi32(y_hi_i32, g_dup_hi),
      );
      let b_lo_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, b_dup_lo),
        _mm_add_epi32(y_hi_i32, b_dup_hi),
      );

      write_rgb_u16_8(
        r_lo_u16,
        g_lo_u16,
        b_lo_u16,
        rgb_out.as_mut_ptr().add(x * 3),
      );
      x += 8;
    }

    if x < width {
      scalar::yuv_420p16_to_rgb_u16_row(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ===== 16-bit semi-planar (P016) → RGB ===================================

/// SSE4.1 P016 → packed **8-bit** RGB. Thin wrapper: deinterleaves UV
/// via [`deinterleave_uv_u16`] then delegates to the shared planar kernel.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn p16_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let y_high = _mm_loadu_si128(y.as_ptr().add(x + 8).cast());
      let (u_vec, v_vec) = deinterleave_uv_u16(uv_half.as_ptr().add(x));

      let u_i16 = _mm_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias16_v);

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

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y_u16(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = _mm_packus_epi16(r_lo, r_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let b_u8 = _mm_packus_epi16(b_lo, b_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }

    if x < width {
      scalar::p16_to_rgb_row(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 P016 → packed **16-bit** RGB. i64 chroma arithmetic, 8 pixels per iteration.
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgb_out` is `&mut [u16]`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn p16_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi64x(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      // Load 4 UV pairs = 8 u16 = 16 bytes; deinterleave inline.
      // uv_half.len() >= width >= x + 8 guarantees 8 u16 readable.
      let uv_raw = _mm_loadu_si128(uv_half.as_ptr().add(x).cast());
      // [U0,V0,U1,V1,U2,V2,U3,V3] → [U0,U1,U2,U3, V0,V1,V2,V3]
      let split_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
      let uv_split = _mm_shuffle_epi8(uv_raw, split_mask);
      let u_vec4 = uv_split;
      let v_vec4 = _mm_srli_si128::<8>(uv_split);

      let u_i16 = _mm_sub_epi16(u_vec4, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec4, bias16_v);

      let rnd32_v = _mm_set1_epi32(1 << 14);
      let u_i32 = _mm_cvtepi16_epi32(u_i16);
      let v_i32 = _mm_cvtepi16_epi32(v_i16);
      let u_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_i32, c_scale_v), rnd32_v));
      let v_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_i32, c_scale_v), rnd32_v));

      let u_d_even = u_d;
      let v_d_even = v_d;
      let u_d_odd = _mm_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x2(cru, crv, u_d_even, v_d_even, rnd_v);
      let r_ch_odd = chroma_i64x2(cru, crv, u_d_odd, v_d_odd, rnd_v);
      let g_ch_even = chroma_i64x2(cgu, cgv, u_d_even, v_d_even, rnd_v);
      let g_ch_odd = chroma_i64x2(cgu, cgv, u_d_odd, v_d_odd, rnd_v);
      let b_ch_even = chroma_i64x2(cbu, cbv, u_d_even, v_d_even, rnd_v);
      let b_ch_odd = chroma_i64x2(cbu, cbv, u_d_odd, v_d_odd, rnd_v);

      let r_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_even, r_ch_odd),
        _mm_unpackhi_epi32(r_ch_even, r_ch_odd),
      );
      let g_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_even, g_ch_odd),
        _mm_unpackhi_epi32(g_ch_even, g_ch_odd),
      );
      let b_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_even, b_ch_odd),
        _mm_unpackhi_epi32(b_ch_even, b_ch_odd),
      );

      let r_dup_lo = _mm_unpacklo_epi32(r_ch_i32, r_ch_i32);
      let r_dup_hi = _mm_unpackhi_epi32(r_ch_i32, r_ch_i32);
      let g_dup_lo = _mm_unpacklo_epi32(g_ch_i32, g_ch_i32);
      let g_dup_hi = _mm_unpackhi_epi32(g_ch_i32, g_ch_i32);
      let b_dup_lo = _mm_unpacklo_epi32(b_ch_i32, b_ch_i32);
      let b_dup_hi = _mm_unpackhi_epi32(b_ch_i32, b_ch_i32);

      let y_lo_pair = _mm_cvtepu16_epi32(y_vec);
      let y_hi_pair = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(y_vec));
      let y_lo_sub = _mm_sub_epi32(y_lo_pair, y_off_v);
      let y_hi_sub = _mm_sub_epi32(y_hi_pair, y_off_v);

      let y_lo_even = scale_y16_i64(y_lo_sub, y_scale_v, rnd_v);
      let y_lo_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_lo_sub), y_scale_v, rnd_v);
      let y_hi_even = scale_y16_i64(y_hi_sub, y_scale_v, rnd_v);
      let y_hi_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_hi_sub), y_scale_v, rnd_v);

      let y_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_lo_even, y_lo_odd),
        _mm_unpackhi_epi32(y_lo_even, y_lo_odd),
      );
      let y_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_hi_even, y_hi_odd),
        _mm_unpackhi_epi32(y_hi_even, y_hi_odd),
      );

      let r_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, r_dup_lo),
        _mm_add_epi32(y_hi_i32, r_dup_hi),
      );
      let g_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, g_dup_lo),
        _mm_add_epi32(y_hi_i32, g_dup_hi),
      );
      let b_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, b_dup_lo),
        _mm_add_epi32(y_hi_i32, b_dup_hi),
      );

      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }

    if x < width {
      scalar::p16_to_rgb_u16_row(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ===== Pn 4:4:4 (semi-planar high-bit-packed) → RGB =======================
//
// SSE4.1 kernels for `p_n_444_to_rgb_*<BITS>` (BITS ∈ {10, 12}) and
// `p_n_444_16_to_rgb_*` (BITS = 16). Combine the deinterleave of
// `p_n_to_rgb_row` (UV via `deinterleave_uv_u16`) with the 1:1 chroma
// compute of `yuv_444p_n_to_rgb_row` (no duplication step). Block
// size: 16 Y pixels + 32 UV `u16` elements per iter (two
// `deinterleave_uv_u16` calls).

/// SSE4.1 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **u8** RGB.
///
/// # Safety
///
/// 1. SSE4.1 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn p_n_444_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), shr_count);

      // Two deinterleave calls — 32 UV u16 elements (= 16 pairs).
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = _mm_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm_srl_epi16(v_hi_vec, shr_count);

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      scalar::p_n_444_to_rgb_row::<BITS>(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB. Output is low-bit-packed.
///
/// # Safety
///
/// 1. SSE4.1 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn p_n_444_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), shr_count);

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = _mm_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm_srl_epi16(v_hi_vec, shr_count);

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      write_rgb_u16_8(r_lo, g_lo, b_lo, rgb_out.as_mut_ptr().add(x * 3));
      write_rgb_u16_8(r_hi, g_hi, b_hi, rgb_out.as_mut_ptr().add(x * 3 + 24));

      x += 16;
    }

    if x < width {
      scalar::p_n_444_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB. Y +
/// chroma both stay on i32 (output-range scaling keeps `coeff × u_d`
/// within i32 for u8 output). Mirrors `yuv_444p16_to_rgb_row` with
/// full-width interleaved UV via `deinterleave_uv_u16`.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn p_n_444_16_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let y_high = _mm_loadu_si128(y.as_ptr().add(x + 8).cast());

      // 32 UV elements per iter — two deinterleave calls.
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2 + 16));

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias16_v);

      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y_u16(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = _mm_packus_epi16(r_lo, r_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let b_u8 = _mm_packus_epi16(b_lo, b_hi);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }

    if x < width {
      scalar::p_n_444_16_to_rgb_row(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// SSE4.1 P416 → packed **native-depth u16** RGB. i64 chroma via
/// `_mm_mul_epi32` + `srai64_15` bias trick (mirroring
/// `yuv_444p16_to_rgb_u16_row`). 8 pixels per iter.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn p_n_444_16_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi64x(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let rnd32_v = _mm_set1_epi32(1 << 14);

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 pixels per iter (i64 narrows). 16 UV u16 elements (= 8 pairs).
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let (u_vec, v_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2));

      let u_i16 = _mm_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias16_v);

      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd32_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd32_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd32_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd32_v));

      // i64 chroma via even/odd splits — same pattern as
      // yuv_444p16_to_rgb_u16_row.
      let u_d_lo_even = u_d_lo;
      let u_d_lo_odd = _mm_shuffle_epi32::<0xF5>(u_d_lo);
      let v_d_lo_even = v_d_lo;
      let v_d_lo_odd = _mm_shuffle_epi32::<0xF5>(v_d_lo);
      let u_d_hi_even = u_d_hi;
      let u_d_hi_odd = _mm_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_hi_even = v_d_hi;
      let v_d_hi_odd = _mm_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_even = chroma_i64x2(cru, crv, u_d_lo_even, v_d_lo_even, rnd_v);
      let r_ch_lo_odd = chroma_i64x2(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let r_ch_hi_even = chroma_i64x2(cru, crv, u_d_hi_even, v_d_hi_even, rnd_v);
      let r_ch_hi_odd = chroma_i64x2(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let g_ch_lo_even = chroma_i64x2(cgu, cgv, u_d_lo_even, v_d_lo_even, rnd_v);
      let g_ch_lo_odd = chroma_i64x2(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let g_ch_hi_even = chroma_i64x2(cgu, cgv, u_d_hi_even, v_d_hi_even, rnd_v);
      let g_ch_hi_odd = chroma_i64x2(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let b_ch_lo_even = chroma_i64x2(cbu, cbv, u_d_lo_even, v_d_lo_even, rnd_v);
      let b_ch_lo_odd = chroma_i64x2(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let b_ch_hi_even = chroma_i64x2(cbu, cbv, u_d_hi_even, v_d_hi_even, rnd_v);
      let b_ch_hi_odd = chroma_i64x2(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_v);

      let r_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_lo_even, r_ch_lo_odd),
        _mm_unpackhi_epi32(r_ch_lo_even, r_ch_lo_odd),
      );
      let r_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_hi_even, r_ch_hi_odd),
        _mm_unpackhi_epi32(r_ch_hi_even, r_ch_hi_odd),
      );
      let g_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_lo_even, g_ch_lo_odd),
        _mm_unpackhi_epi32(g_ch_lo_even, g_ch_lo_odd),
      );
      let g_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_hi_even, g_ch_hi_odd),
        _mm_unpackhi_epi32(g_ch_hi_even, g_ch_hi_odd),
      );
      let b_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_lo_even, b_ch_lo_odd),
        _mm_unpackhi_epi32(b_ch_lo_even, b_ch_lo_odd),
      );
      let b_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_hi_even, b_ch_hi_odd),
        _mm_unpackhi_epi32(b_ch_hi_even, b_ch_hi_odd),
      );

      let y_lo_pair = _mm_cvtepu16_epi32(y_vec);
      let y_hi_pair = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(y_vec));
      let y_lo_sub = _mm_sub_epi32(y_lo_pair, y_off_v);
      let y_hi_sub = _mm_sub_epi32(y_hi_pair, y_off_v);
      let y_lo_even = scale_y16_i64(y_lo_sub, y_scale_v, rnd_v);
      let y_lo_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_lo_sub), y_scale_v, rnd_v);
      let y_hi_even = scale_y16_i64(y_hi_sub, y_scale_v, rnd_v);
      let y_hi_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_hi_sub), y_scale_v, rnd_v);
      let y_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_lo_even, y_lo_odd),
        _mm_unpackhi_epi32(y_lo_even, y_lo_odd),
      );
      let y_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_hi_even, y_hi_odd),
        _mm_unpackhi_epi32(y_hi_even, y_hi_odd),
      );

      let r_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, r_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, r_ch_hi_i32),
      );
      let g_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, g_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, g_ch_hi_i32),
      );
      let b_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, b_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, b_ch_hi_i32),
      );

      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }

    if x < width {
      scalar::p_n_444_16_to_rgb_u16_row(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ===== BGR ↔ RGB byte swap ==============================================

/// SSE4.1 BGR ↔ RGB byte swap. 16 pixels per iteration via the shared
/// [`super::x86_common::swap_rb_16_pixels`] helper (SSSE3 `_mm_shuffle_epi8`
/// underneath). Drives both conversion directions since the swap is
/// self‑inverse.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  // SAFETY: SSE4.1 is available per caller obligation; SSSE3 (required
  // by `swap_rb_16_pixels`) is a subset. All pointer adds are bounded
  // by the `while x + 16 <= width` condition.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      swap_rb_16_pixels(input.as_ptr().add(x * 3), output.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::bgr_rgb_swap_row(
        &input[x * 3..width * 3],
        &mut output[x * 3..width * 3],
        width - x,
      );
    }
  }
}

// ===== RGB → HSV =========================================================

/// SSE4.1 RGB → planar HSV (OpenCV 8‑bit encoding). 16 pixels per
/// iteration via the shared [`super::x86_common::rgb_to_hsv_16_pixels`]
/// helper.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`.
/// 3. `h_out.len() >= width`, `s_out.len() >= width`, `v_out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      rgb_to_hsv_16_pixels(
        rgb.as_ptr().add(x * 3),
        h_out.as_mut_ptr().add(x),
        s_out.as_mut_ptr().add(x),
        v_out.as_mut_ptr().add(x),
      );
      x += 16;
    }
    if x < width {
      scalar::rgb_to_hsv_row(
        &rgb[x * 3..width * 3],
        &mut h_out[x..width],
        &mut s_out[x..width],
        &mut v_out[x..width],
        width - x,
      );
    }
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  fn check_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_sse41 = std::vec![0u8; width * 3];

    scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_sse41, width, matrix, full_range);
    }

    if rgb_scalar != rgb_sse41 {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_sse41.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "SSE4.1 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} sse41={}",
        rgb_scalar[first_diff], rgb_sse41[first_diff]
      );
    }
  }

  #[test]
  fn sse41_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_matches_scalar_width_32() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    check_equivalence(32, ColorMatrix::Bt601, true);
    check_equivalence(32, ColorMatrix::Bt709, false);
    check_equivalence(32, ColorMatrix::YCgCo, true);
  }

  #[test]
  fn sse41_matches_scalar_width_1920() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    check_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  fn sse41_matches_scalar_odd_tail_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    // Widths that leave a non‑trivial scalar tail (non‑multiple of 16).
    for w in [18usize, 30, 34, 1922] {
      check_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  // ---- yuv_420_to_rgba_row equivalence --------------------------------
  //
  // Direct backend test for the new RGBA path: bypasses the public
  // dispatcher so the SSE4.1 `write_rgba_16` shuffle masks are
  // exercised regardless of what tier the dispatcher would pick on
  // the current runner. Catches lane-order, shuffle-mask, or alpha
  // splat corruption that an AVX2- or AVX-512-routed test would
  // miss.

  fn check_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let mut rgba_scalar = std::vec![0u8; width * 4];
    let mut rgba_sse41 = std::vec![0u8; width * 4];

    scalar::yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_sse41, width, matrix, full_range);
    }

    if rgba_scalar != rgba_sse41 {
      let first_diff = rgba_scalar
        .iter()
        .zip(rgba_sse41.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      let pixel = first_diff / 4;
      let channel = ["R", "G", "B", "A"][first_diff % 4];
      panic!(
        "SSE4.1 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} sse41={}",
        rgba_scalar[first_diff], rgba_sse41[first_diff]
      );
    }
  }

  #[test]
  fn sse41_rgba_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_rgba_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_rgba_matches_scalar_width_32() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    check_rgba_equivalence(32, ColorMatrix::Bt601, true);
    check_rgba_equivalence(32, ColorMatrix::Bt709, false);
    check_rgba_equivalence(32, ColorMatrix::YCgCo, true);
  }

  #[test]
  fn sse41_rgba_matches_scalar_width_1920() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    check_rgba_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  fn sse41_rgba_matches_scalar_odd_tail_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [18usize, 30, 34, 1922] {
      check_rgba_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  // ---- nv12_to_rgb_row equivalence ------------------------------------

  fn check_nv12_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_sse41 = std::vec![0u8; width * 3];

    scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_sse41, width, matrix, full_range);
    }

    if rgb_scalar != rgb_sse41 {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_sse41.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "SSE4.1 NV12 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} sse41={}",
        rgb_scalar[first_diff], rgb_sse41[first_diff]
      );
    }
  }

  fn check_nv12_matches_yuv420p(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

    let mut rgb_yuv420p = std::vec![0u8; width * 3];
    let mut rgb_nv12 = std::vec![0u8; width * 3];
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_yuv420p, width, matrix, full_range);
      nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
    }
    assert_eq!(
      rgb_yuv420p, rgb_nv12,
      "SSE4.1 NV12 ≠ YUV420P for equivalent UV"
    );
  }

  #[test]
  fn sse41_nv12_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv12_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_nv12_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [32usize, 1920, 18, 30, 34, 1922] {
      check_nv12_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn sse41_nv12_matches_yuv420p() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [16usize, 30, 64, 1920] {
      check_nv12_matches_yuv420p(w, ColorMatrix::Bt709, false);
      check_nv12_matches_yuv420p(w, ColorMatrix::YCgCo, true);
    }
  }

  // ---- nv24_to_rgb_row / nv42_to_rgb_row equivalence ------------------

  fn check_nv24_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width)
      .flat_map(|i| {
        [
          ((i * 53 + 23) & 0xFF) as u8, // U_i
          ((i * 71 + 91) & 0xFF) as u8, // V_i
        ]
      })
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_sse41 = std::vec![0u8; width * 3];

    scalar::nv24_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv24_to_rgb_row(&y, &uv, &mut rgb_sse41, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_sse41,
      "SSE4.1 NV24 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_nv42_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vu: std::vec::Vec<u8> = (0..width)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_sse41 = std::vec![0u8; width * 3];

    scalar::nv42_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv42_to_rgb_row(&y, &vu, &mut rgb_sse41, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_sse41,
      "SSE4.1 NV42 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn sse41_nv24_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv24_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_nv24_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    // Odd widths validate the 4:4:4 no-parity contract.
    for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
      check_nv24_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn sse41_nv42_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv42_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_nv42_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
      check_nv42_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  // ---- yuv_444_to_rgb_row equivalence ---------------------------------

  fn check_yuv_444_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
    let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_sse41 = std::vec![0u8; width * 3];

    scalar::yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_sse41, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_sse41,
      "SSE4.1 yuv_444 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn sse41_yuv_444_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_yuv_444_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_yuv_444_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    // Odd widths validate the 4:4:4 no-parity contract.
    for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
      check_yuv_444_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  // ---- yuv_444_to_rgba_row equivalence --------------------------------

  fn check_yuv_444_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
    let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
    let mut rgba_scalar = std::vec![0u8; width * 4];
    let mut rgba_sse41 = std::vec![0u8; width * 4];

    scalar::yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
    unsafe {
      yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_sse41, width, matrix, full_range);
    }

    if rgba_scalar != rgba_sse41 {
      let first_diff = rgba_scalar
        .iter()
        .zip(rgba_sse41.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      let pixel = first_diff / 4;
      let channel = ["R", "G", "B", "A"][first_diff % 4];
      panic!(
        "SSE4.1 yuv_444 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} sse41={}",
        rgba_scalar[first_diff], rgba_sse41[first_diff]
      );
    }
  }

  #[test]
  fn sse41_yuv_444_rgba_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_yuv_444_rgba_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_yuv_444_rgba_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
      check_yuv_444_rgba_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  // ---- yuv_444p_n<BITS> + yuv_444p16 equivalence ----------------------

  fn check_yuv_444p_n_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let max_val = (1u16 << BITS) - 1;
    let y: std::vec::Vec<u16> = (0..width)
      .map(|i| ((i * 37 + 11) as u16) & max_val)
      .collect();
    let u: std::vec::Vec<u16> = (0..width)
      .map(|i| ((i * 53 + 23) as u16) & max_val)
      .collect();
    let v: std::vec::Vec<u16> = (0..width)
      .map(|i| ((i * 71 + 91) as u16) & max_val)
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_sse41 = std::vec![0u8; width * 3];
    let mut u16_scalar = std::vec![0u16; width * 3];
    let mut u16_sse41 = std::vec![0u16; width * 3];

    scalar::yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
      &y,
      &u,
      &v,
      &mut u16_scalar,
      width,
      matrix,
      full_range,
    );
    unsafe {
      yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_sse41, width, matrix, full_range);
      yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut u16_sse41, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_sse41,
      "SSE4.1 yuv_444p_n<{BITS}> u8 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
    assert_eq!(
      u16_scalar, u16_sse41,
      "SSE4.1 yuv_444p_n<{BITS}> u16 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn sse41_yuv_444p9_matches_scalar_all_matrices() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
      for full in [true, false] {
        check_yuv_444p_n_equivalence::<9>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_yuv_444p10_matches_scalar_all_matrices() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_yuv_444p_n_equivalence::<10>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_yuv_444p12_matches_scalar_all_matrices() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
      for full in [true, false] {
        check_yuv_444p_n_equivalence::<12>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_yuv_444p14_matches_scalar_all_matrices() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
      for full in [true, false] {
        check_yuv_444p_n_equivalence::<14>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_yuv_444p_n_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
      check_yuv_444p_n_equivalence::<10>(w, ColorMatrix::Bt709, false);
    }
  }

  fn check_yuv_444p16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u16> = (0..width).map(|i| (i * 2027 + 11) as u16).collect();
    let u: std::vec::Vec<u16> = (0..width).map(|i| (i * 2671 + 23) as u16).collect();
    let v: std::vec::Vec<u16> = (0..width).map(|i| (i * 3329 + 91) as u16).collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_sse41 = std::vec![0u8; width * 3];
    let mut u16_scalar = std::vec![0u16; width * 3];
    let mut u16_sse41 = std::vec![0u16; width * 3];

    scalar::yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    scalar::yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut u16_scalar, width, matrix, full_range);
    unsafe {
      yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_sse41, width, matrix, full_range);
      yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut u16_sse41, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_sse41,
      "SSE4.1 yuv_444p16 u8 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
    assert_eq!(
      u16_scalar, u16_sse41,
      "SSE4.1 yuv_444p16 u16 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn sse41_yuv_444p16_matches_scalar_all_matrices() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_yuv_444p16_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_yuv_444p16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    // The u16 kernel is 8-pixel per iter; the u8 kernel is 16.
    for w in [1usize, 3, 7, 8, 9, 15, 17, 32, 33, 1920, 1921] {
      check_yuv_444p16_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  // ---- bgr_rgb_swap_row equivalence -----------------------------------

  fn check_swap_equivalence(width: usize) {
    let input: std::vec::Vec<u8> = (0..width * 3)
      .map(|i| ((i * 17 + 41) & 0xFF) as u8)
      .collect();
    let mut out_scalar = std::vec![0u8; width * 3];
    let mut out_sse41 = std::vec![0u8; width * 3];

    scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
    unsafe {
      bgr_rgb_swap_row(&input, &mut out_sse41, width);
    }
    assert_eq!(out_scalar, out_sse41, "SSE4.1 swap diverges from scalar");
  }

  #[test]
  fn sse41_swap_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [1usize, 15, 16, 17, 31, 32, 33, 1920, 1921] {
      check_swap_equivalence(w);
    }
  }

  // ---- nv21_to_rgb_row equivalence ------------------------------------

  fn check_nv21_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vu: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_sse41 = std::vec![0u8; width * 3];

    scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv21_to_rgb_row(&y, &vu, &mut rgb_sse41, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_sse41,
      "SSE4.1 NV21 ≠ scalar (width={width}, matrix={matrix:?})"
    );
  }

  fn check_nv21_matches_nv12_swapped(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut vu = std::vec![0u8; width];
    for i in 0..width / 2 {
      vu[2 * i] = uv[2 * i + 1];
      vu[2 * i + 1] = uv[2 * i];
    }

    let mut rgb_nv12 = std::vec![0u8; width * 3];
    let mut rgb_nv21 = std::vec![0u8; width * 3];
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
      nv21_to_rgb_row(&y, &vu, &mut rgb_nv21, width, matrix, full_range);
    }
    assert_eq!(
      rgb_nv12, rgb_nv21,
      "SSE4.1 NV21 ≠ NV12 with byte-swapped chroma"
    );
  }

  #[test]
  fn nv21_sse41_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv21_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn nv21_sse41_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [32usize, 1920, 18, 30, 34, 1922] {
      check_nv21_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn nv21_sse41_matches_nv12_swapped() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [16usize, 30, 64, 1920] {
      check_nv21_matches_nv12_swapped(w, ColorMatrix::Bt709, false);
      check_nv21_matches_nv12_swapped(w, ColorMatrix::YCgCo, true);
    }
  }

  // ---- nv12_to_rgba_row / nv21_to_rgba_row equivalence ----------------

  fn check_nv12_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgba_scalar = std::vec![0u8; width * 4];
    let mut rgba_sse41 = std::vec![0u8; width * 4];

    scalar::nv12_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
    unsafe {
      nv12_to_rgba_row(&y, &uv, &mut rgba_sse41, width, matrix, full_range);
    }

    if rgba_scalar != rgba_sse41 {
      let first_diff = rgba_scalar
        .iter()
        .zip(rgba_sse41.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      let pixel = first_diff / 4;
      let channel = ["R", "G", "B", "A"][first_diff % 4];
      panic!(
        "SSE4.1 NV12 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} sse41={}",
        rgba_scalar[first_diff], rgba_sse41[first_diff]
      );
    }
  }

  fn check_nv21_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vu: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgba_scalar = std::vec![0u8; width * 4];
    let mut rgba_sse41 = std::vec![0u8; width * 4];

    scalar::nv21_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
    unsafe {
      nv21_to_rgba_row(&y, &vu, &mut rgba_sse41, width, matrix, full_range);
    }

    if rgba_scalar != rgba_sse41 {
      let first_diff = rgba_scalar
        .iter()
        .zip(rgba_sse41.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      let pixel = first_diff / 4;
      let channel = ["R", "G", "B", "A"][first_diff % 4];
      panic!(
        "SSE4.1 NV21 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} sse41={}",
        rgba_scalar[first_diff], rgba_sse41[first_diff]
      );
    }
  }

  fn check_nv12_rgba_matches_yuv420p_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

    let mut rgba_yuv420p = std::vec![0u8; width * 4];
    let mut rgba_nv12 = std::vec![0u8; width * 4];
    unsafe {
      yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_yuv420p, width, matrix, full_range);
      nv12_to_rgba_row(&y, &uv, &mut rgba_nv12, width, matrix, full_range);
    }
    assert_eq!(
      rgba_yuv420p, rgba_nv12,
      "SSE4.1 NV12 RGBA must match Yuv420p RGBA for equivalent UV (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn nv12_sse41_rgba_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv12_rgba_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn nv12_sse41_rgba_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [18usize, 30, 34, 1920, 1922] {
      check_nv12_rgba_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  #[test]
  fn nv12_sse41_rgba_matches_yuv420p_rgba_sse41() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [16usize, 30, 64, 1920] {
      check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::Bt709, false);
      check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::YCgCo, true);
    }
  }

  #[test]
  fn nv21_sse41_rgba_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv21_rgba_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn nv21_sse41_rgba_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [18usize, 30, 34, 1920, 1922] {
      check_nv21_rgba_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  // ---- rgb_to_hsv_row equivalence --------------------------------------

  fn check_hsv_equivalence(rgb: &[u8], width: usize) {
    let mut h_s = std::vec![0u8; width];
    let mut s_s = std::vec![0u8; width];
    let mut v_s = std::vec![0u8; width];
    let mut h_k = std::vec![0u8; width];
    let mut s_k = std::vec![0u8; width];
    let mut v_k = std::vec![0u8; width];

    scalar::rgb_to_hsv_row(rgb, &mut h_s, &mut s_s, &mut v_s, width);
    unsafe {
      rgb_to_hsv_row(rgb, &mut h_k, &mut s_k, &mut v_k, width);
    }
    for (i, (a, b)) in h_s.iter().zip(h_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "H divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
    for (i, (a, b)) in s_s.iter().zip(s_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "S divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
    for (i, (a, b)) in v_s.iter().zip(v_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "V divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
  }

  #[test]
  fn sse41_hsv_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let rgb: std::vec::Vec<u8> = (0..1921 * 3)
      .map(|i| ((i * 37 + 11) & 0xFF) as u8)
      .collect();
    for w in [1usize, 15, 16, 17, 31, 1920, 1921] {
      check_hsv_equivalence(&rgb[..w * 3], w);
    }
  }

  // ---- yuv420p10 scalar-equivalence -----------------------------------

  fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
      .collect()
  }

  fn check_p10_u8_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p10_plane(width, 37);
    let u = p10_plane(width / 2, 53);
    let v = p10_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];

    scalar::yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }

    if rgb_scalar != rgb_simd {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_simd.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "SSE4.1 10→u8 diverges at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
        rgb_scalar[first_diff], rgb_simd[first_diff]
      );
    }
  }

  fn check_p10_u16_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p10_plane(width, 37);
    let u = p10_plane(width / 2, 53);
    let v = p10_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];

    scalar::yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }

    if rgb_scalar != rgb_simd {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_simd.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "SSE4.1 10→u16 diverges at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
        rgb_scalar[first_diff], rgb_simd[first_diff]
      );
    }
  }

  #[test]
  fn sse41_p10_u8_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p10_u8_sse41_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p10_u16_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p10_u16_sse41_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p10_matches_scalar_odd_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_p10_u8_sse41_equivalence(w, ColorMatrix::Bt601, false);
      check_p10_u16_sse41_equivalence(w, ColorMatrix::Bt709, true);
    }
  }

  #[test]
  fn sse41_p10_matches_scalar_1920() {
    check_p10_u8_sse41_equivalence(1920, ColorMatrix::Bt709, false);
    check_p10_u16_sse41_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- yuv420p_n<BITS> SSE4.1 scalar-equivalence (BITS=9 coverage) -----

  fn p_n_plane_sse41<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = ((1u32 << BITS) - 1) as u16;
    (0..n)
      .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask)
      .collect()
  }

  fn check_p_n_u8_sse41_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p_n_plane_sse41::<BITS>(width, 37);
    let u = p_n_plane_sse41::<BITS>(width / 2, 53);
    let v = p_n_plane_sse41::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 yuv_420p_n<{BITS}>→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p_n_u16_sse41_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p_n_plane_sse41::<BITS>(width, 37);
    let u = p_n_plane_sse41::<BITS>(width / 2, 53);
    let v = p_n_plane_sse41::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
      &y,
      &u,
      &v,
      &mut rgb_scalar,
      width,
      matrix,
      full_range,
    );
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 yuv_420p_n<{BITS}>→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn sse41_yuv420p9_matches_scalar_all_matrices_and_ranges() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p_n_u8_sse41_equivalence::<9>(16, m, full);
        check_p_n_u16_sse41_equivalence::<9>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_yuv420p9_matches_scalar_tail_and_large_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_p_n_u8_sse41_equivalence::<9>(w, ColorMatrix::Bt601, false);
      check_p_n_u16_sse41_equivalence::<9>(w, ColorMatrix::Bt709, true);
    }
    check_p_n_u8_sse41_equivalence::<9>(1920, ColorMatrix::Bt709, false);
    check_p_n_u16_sse41_equivalence::<9>(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- P010 SSE4.1 scalar-equivalence ----------------------------------

  fn p010_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| (((i * seed + seed * 3) & 0x3FF) as u16) << 6)
      .collect()
  }

  fn p010_uv_interleave(u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
    let pairs = u.len();
    debug_assert_eq!(u.len(), v.len());
    let mut out = std::vec::Vec::with_capacity(pairs * 2);
    for i in 0..pairs {
      out.push(u[i]);
      out.push(v[i]);
    }
    out
  }

  fn check_p010_u8_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p010_plane(width, 37);
    let u = p010_plane(width / 2, 53);
    let v = p010_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "SSE4.1 P010→u8 diverges");
  }

  fn check_p010_u16_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p010_plane(width, 37);
    let u = p010_plane(width / 2, 53);
    let v = p010_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "SSE4.1 P010→u16 diverges");
  }

  #[test]
  fn sse41_p010_u8_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p010_u8_sse41_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p010_u16_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p010_u16_sse41_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p010_matches_scalar_odd_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_p010_u8_sse41_equivalence(w, ColorMatrix::Bt601, false);
      check_p010_u16_sse41_equivalence(w, ColorMatrix::Bt709, true);
    }
  }

  #[test]
  fn sse41_p010_matches_scalar_1920() {
    check_p010_u8_sse41_equivalence(1920, ColorMatrix::Bt709, false);
    check_p010_u16_sse41_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- Generic BITS equivalence (12/14-bit coverage) ------------------
  //
  // The helpers below parameterize over `const BITS: u32` so the same
  // scalar-equivalence scaffolding covers 10/12/14 without duplicating
  // the 16-pixel block seeding + diff harness. `<10>` is already
  // exercised by the dedicated tests above; `<12>` / `<14>` add
  // regression coverage for the new yuv420p12 / yuv420p14 / P012
  // kernels. 14-bit is planar-only (no P014 in Ship 4a).

  fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = (1u32 << BITS) - 1;
    (0..n)
      .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
      .collect()
  }

  fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = (1u32 << BITS) - 1;
    let shift = 16 - BITS;
    (0..n)
      .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
      .collect()
  }

  fn check_planar_u8_sse41_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = planar_n_plane::<BITS>(width, 37);
    let u = planar_n_plane::<BITS>(width / 2, 53);
    let v = planar_n_plane::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];

    scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 planar {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_planar_u16_sse41_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = planar_n_plane::<BITS>(width, 37);
    let u = planar_n_plane::<BITS>(width / 2, 53);
    let v = planar_n_plane::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];

    scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
      &y,
      &u,
      &v,
      &mut rgb_scalar,
      width,
      matrix,
      full_range,
    );
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 planar {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_pn_u8_sse41_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p_n_packed_plane::<BITS>(width, 37);
    let u = p_n_packed_plane::<BITS>(width / 2, 53);
    let v = p_n_packed_plane::<BITS>(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "SSE4.1 Pn {BITS}-bit → u8 diverges");
  }

  fn check_pn_u16_sse41_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p_n_packed_plane::<BITS>(width, 37);
    let u = p_n_packed_plane::<BITS>(width / 2, 53);
    let v = p_n_packed_plane::<BITS>(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "SSE4.1 Pn {BITS}-bit → u16 diverges");
  }

  #[test]
  fn sse41_p12_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_planar_u8_sse41_equivalence_n::<12>(16, m, full);
        check_planar_u16_sse41_equivalence_n::<12>(16, m, full);
        check_pn_u8_sse41_equivalence_n::<12>(16, m, full);
        check_pn_u16_sse41_equivalence_n::<12>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p14_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_planar_u8_sse41_equivalence_n::<14>(16, m, full);
        check_planar_u16_sse41_equivalence_n::<14>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p12_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_planar_u8_sse41_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
      check_planar_u16_sse41_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
      check_pn_u8_sse41_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
      check_pn_u16_sse41_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    }
  }

  #[test]
  fn sse41_p14_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_planar_u8_sse41_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
      check_planar_u16_sse41_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
    }
  }

  // ---- 16-bit (full-range u16 samples) SSE4.1 equivalence -------------

  fn p16_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
      .collect()
  }

  fn check_yuv420p16_u8_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p16_plane(width, 37);
    let u = p16_plane(width / 2, 53);
    let v = p16_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 yuv420p16→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_yuv420p16_u16_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p16_plane(width, 37);
    let u = p16_plane(width / 2, 53);
    let v = p16_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 yuv420p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p16_u8_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p16_plane(width, 37);
    let u = p16_plane(width / 2, 53);
    let v = p16_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 p016→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p16_u16_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p16_plane(width, 37);
    let u = p16_plane(width / 2, 53);
    let v = p16_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 p016→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn sse41_p16_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_yuv420p16_u8_sse41_equivalence(16, m, full);
        check_yuv420p16_u16_sse41_equivalence(16, m, full);
        check_p16_u8_sse41_equivalence(16, m, full);
        check_p16_u16_sse41_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p16_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_yuv420p16_u8_sse41_equivalence(w, ColorMatrix::Bt601, false);
      check_yuv420p16_u16_sse41_equivalence(w, ColorMatrix::Bt709, true);
      check_p16_u8_sse41_equivalence(w, ColorMatrix::Bt601, false);
      check_p16_u16_sse41_equivalence(w, ColorMatrix::Bt2020Ncl, false);
    }
  }

  #[test]
  fn sse41_p16_matches_scalar_1920() {
    check_yuv420p16_u8_sse41_equivalence(1920, ColorMatrix::Bt709, false);
    check_yuv420p16_u16_sse41_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
    check_p16_u8_sse41_equivalence(1920, ColorMatrix::Bt709, false);
    check_p16_u16_sse41_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- Pn 4:4:4 (P410 / P412 / P416) SSE4.1 equivalence ---------------

  fn high_bit_plane_sse41<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = ((1u32 << BITS) - 1) as u16;
    let shift = 16 - BITS;
    (0..n)
      .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
      .collect()
  }

  fn interleave_uv_sse41(u_full: &[u16], v_full: &[u16]) -> std::vec::Vec<u16> {
    debug_assert_eq!(u_full.len(), v_full.len());
    let mut out = std::vec::Vec::with_capacity(u_full.len() * 2);
    for i in 0..u_full.len() {
      out.push(u_full[i]);
      out.push(v_full[i]);
    }
    out
  }

  fn check_p_n_444_u8_sse41_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = high_bit_plane_sse41::<BITS>(width, 37);
    let u = high_bit_plane_sse41::<BITS>(width, 53);
    let v = high_bit_plane_sse41::<BITS>(width, 71);
    let uv = interleave_uv_sse41(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 Pn4:4:4 {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p_n_444_u16_sse41_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = high_bit_plane_sse41::<BITS>(width, 37);
    let u = high_bit_plane_sse41::<BITS>(width, 53);
    let v = high_bit_plane_sse41::<BITS>(width, 71);
    let uv = interleave_uv_sse41(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 Pn4:4:4 {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p_n_444_16_u8_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p16_plane(width, 37);
    let u = p16_plane(width, 53);
    let v = p16_plane(width, 71);
    let uv = interleave_uv_sse41(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 P416 → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p_n_444_16_u16_sse41_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let y = p16_plane(width, 37);
    let u = p16_plane(width, 53);
    let v = p16_plane(width, 71);
    let uv = interleave_uv_sse41(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "SSE4.1 P416 → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn sse41_p410_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p_n_444_u8_sse41_equivalence::<10>(16, m, full);
        check_p_n_444_u16_sse41_equivalence::<10>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p412_matches_scalar_all_matrices() {
    for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
      for full in [true, false] {
        check_p_n_444_u8_sse41_equivalence::<12>(16, m, full);
        check_p_n_444_u16_sse41_equivalence::<12>(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p410_p412_matches_scalar_tail_widths() {
    for w in [1usize, 3, 7, 15, 17, 31, 33, 1920, 1921] {
      check_p_n_444_u8_sse41_equivalence::<10>(w, ColorMatrix::Bt601, false);
      check_p_n_444_u16_sse41_equivalence::<10>(w, ColorMatrix::Bt709, true);
      check_p_n_444_u8_sse41_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    }
  }

  #[test]
  fn sse41_p416_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p_n_444_16_u8_sse41_equivalence(16, m, full);
        check_p_n_444_16_u16_sse41_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_p416_matches_scalar_tail_widths() {
    for w in [1usize, 3, 7, 8, 9, 15, 16, 17, 31, 33, 1920, 1921] {
      check_p_n_444_16_u8_sse41_equivalence(w, ColorMatrix::Bt709, false);
      check_p_n_444_16_u16_sse41_equivalence(w, ColorMatrix::Bt2020Ncl, true);
    }
  }
}
