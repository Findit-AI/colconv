use core::arch::x86_64::*;

use super::*;

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

/// SSE4.1 NV24 → packed RGB (UV-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
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
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, false>(y, uv, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 NV42 → packed RGB (VU-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
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
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, false>(y, vu, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 NV24 → packed RGBA (UV-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 NV42 → packed RGBA (VU-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 NV24/NV42 kernel (4:4:4 semi-planar). Unlike
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`], chroma is not subsampled — one
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
/// 4. `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn nv24_or_nv42_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

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
    let alpha_u8 = _mm_set1_epi8(-1);

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

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
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
