use core::arch::x86_64::*;

use super::*;

/// SSE4.1 YUYV422 → packed RGB. Semantics match
/// [`scalar::yuyv422_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `packed.len() >= 2 * width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuyv422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 YUYV422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuyv422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 UYVY422 → packed RGB.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn uyvy422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 UYVY422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn uyvy422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 YVYU422 → packed RGB.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yvyu422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 YVYU422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yvyu422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// Generic packed YUV 4:2:2 → RGB / RGBA SSE4.1 kernel. 16 pixels /
/// iter; deinterleaves bytes via `_mm_shuffle_epi8` then runs the
/// same Q15 chroma / Y / channel pipeline as `yuv_420`.
///
/// # Safety
///
/// Caller has verified SSE4.1. `packed.len() >= 2 * width`. `width`
/// even. `out.len() >= bpp * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn yuv422_packed_to_rgb_or_rgba_row<
  const Y_LSB: bool,
  const SWAP_UV: bool,
  const ALPHA: bool,
>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: SSE4.1 availability is the caller's obligation.
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

    // Per-block split mask: rearrange a 16-byte (4-block × 4-byte)
    // chunk so that Y bytes land in the low 8 lanes and chroma bytes
    // in the high 8 lanes. The `Y_LSB` const generic picks which set
    // of byte positions is Y vs chroma.
    let split_mask = if Y_LSB {
      // Y at byte positions 0,2,4,...,14; chroma at 1,3,5,...,15.
      _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)
    } else {
      // Y at byte positions 1,3,5,...,15; chroma at 0,2,4,...,14.
      _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14)
    };

    // Mask to split chroma bytes into evens (low 8) and odds (high 8).
    let chroma_split = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 32 packed bytes (covers 16 pixels = 8 chroma pairs).
      let p0 = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let p1 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 16).cast());

      // Per-block split: low 8 bytes = Y of that 16-byte half,
      // high 8 bytes = chroma of that half.
      let p0s = _mm_shuffle_epi8(p0, split_mask);
      let p1s = _mm_shuffle_epi8(p1, split_mask);

      // Combine: low 64 of each → 16 Y bytes; high 64 of each →
      // 16 chroma bytes.
      let y_vec = _mm_unpacklo_epi64(p0s, p1s);
      let chroma_vec = _mm_unpackhi_epi64(p0s, p1s);

      // Split chroma: low 8 = c-evens (positions 0,2,4,...),
      // high 8 = c-odds.
      let chroma_split_v = _mm_shuffle_epi8(chroma_vec, chroma_split);
      // Map to U / V. yuv_420 reads u_vec/v_vec via
      // `_mm_loadl_epi64` → `_mm_cvtepu8_epi16` (low 8 bytes).
      // For SWAP_UV = false (YUYV / UYVY) → c_evens = U,
      // c_odds = V. For SWAP_UV = true (YVYU) → reversed.
      let u_vec = if SWAP_UV {
        _mm_srli_si128::<8>(chroma_split_v) // bring c_odds into low 8
      } else {
        chroma_split_v // c_evens already in low 8
      };
      let v_vec = if SWAP_UV {
        chroma_split_v
      } else {
        _mm_srli_si128::<8>(chroma_split_v)
      };

      // Widen U/V to i16x8 and subtract 128.
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

    // Scalar tail.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        if Y_LSB && !SWAP_UV {
          scalar::yuyv422_to_rgba_row(tail_packed, tail_out, tail_w, matrix, full_range);
        } else if !Y_LSB && !SWAP_UV {
          scalar::uyvy422_to_rgba_row(tail_packed, tail_out, tail_w, matrix, full_range);
        } else {
          scalar::yvyu422_to_rgba_row(tail_packed, tail_out, tail_w, matrix, full_range);
        }
      } else if Y_LSB && !SWAP_UV {
        scalar::yuyv422_to_rgb_row(tail_packed, tail_out, tail_w, matrix, full_range);
      } else if !Y_LSB && !SWAP_UV {
        scalar::uyvy422_to_rgb_row(tail_packed, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yvyu422_to_rgb_row(tail_packed, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// SSE4.1 YUYV422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuyv422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

/// SSE4.1 UYVY422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn uyvy422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<false>(packed, luma_out, width);
  }
}

/// SSE4.1 YVYU422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yvyu422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn yuv422_packed_to_luma_row<const Y_LSB: bool>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let split_mask = if Y_LSB {
      _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)
    } else {
      _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14)
    };

    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let p1 = _mm_loadu_si128(packed.as_ptr().add(x * 2 + 16).cast());
      let p0s = _mm_shuffle_epi8(p0, split_mask);
      let p1s = _mm_shuffle_epi8(p1, split_mask);
      let y_vec = _mm_unpacklo_epi64(p0s, p1s);
      _mm_storeu_si128(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 16;
    }
    if x < width {
      if Y_LSB {
        scalar::yuyv422_to_luma_row(
          &packed[x * 2..width * 2],
          &mut luma_out[x..width],
          width - x,
        );
      } else {
        scalar::uyvy422_to_luma_row(
          &packed[x * 2..width * 2],
          &mut luma_out[x..width],
          width - x,
        );
      }
    }
  }
}
