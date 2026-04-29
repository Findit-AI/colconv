use core::arch::x86_64::*;

use super::*;

/// AVX‑512 YUYV422 → packed RGB. Semantics match
/// [`scalar::yuyv422_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `packed.len() >= 2 * width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuyv422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 YUYV422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuyv422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 UYVY422 → packed RGB.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn uyvy422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 UYVY422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn uyvy422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 YVYU422 → packed RGB.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yvyu422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 YVYU422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yvyu422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// Generic packed YUV 4:2:2 → RGB / RGBA AVX‑512 kernel. 64 px / iter.
///
/// Per-lane `shuffle_epi8` + cross-lane `permutexvar_epi64` deinterleave:
/// load 128 packed bytes via two `_mm512_loadu_si512`, run per-lane
/// shuffle to put Y/chroma into low/high halves of each 128-bit lane,
/// then `permutexvar_epi64` with index `[0,2,4,6, 1,3,5,7]` consolidates
/// each vector into Y-then-chroma layout. A `permutex2var_epi64` merge
/// across two vectors yields 64 Y bytes in one register and 64 chroma
/// bytes in another. A second pass on chroma splits U / V.
///
/// # Safety
///
/// Caller has verified AVX‑512F + AVX‑512BW. `packed.len() >= 2 * width`.
/// `width` even. `out.len() >= bpp * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let mid128 = _mm512_set1_epi16(128);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    // Lane‑fixup permute indices, computed once per call.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    // Per-lane split mask (replicated across all 4 × 128-bit lanes):
    // gather Y bytes in low 8 lanes, chroma bytes in high 8 lanes.
    let split_mask = if Y_LSB {
      _mm512_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5,
        7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10,
        12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
      )
    } else {
      _mm512_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4,
        6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11,
        13, 15, 0, 2, 4, 6, 8, 10, 12, 14,
      )
    };

    // Cross-vector merge indices (for `permutex2var_epi64` selecting
    // qwords from concat(v0, v1) where v1 starts at index 8).
    let merge_low = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    let merge_high = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

    // Chroma split mask (same as split_mask in shape but applied to a
    // 64-byte chroma vector — replicate `[0,2,..,14, 1,3,..,15]`).
    let chroma_split = _mm512_setr_epi8(
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7,
      9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12,
      14, 1, 3, 5, 7, 9, 11, 13, 15,
    );

    let mut x = 0usize;
    while x + 64 <= width {
      // Load 128 packed bytes (64 pixels = 32 chroma pairs).
      let p0 = _mm512_loadu_si512(packed.as_ptr().add(x * 2).cast());
      let p1 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 64).cast());

      // Per-lane shuffle: each 128-bit lane → [Y_8, c_8] split.
      let p0s = _mm512_shuffle_epi8(p0, split_mask);
      let p1s = _mm512_shuffle_epi8(p1, split_mask);

      // Consolidate: pack_fixup `[0,2,4,6, 1,3,5,7]` rearranges 64-bit
      // chunks so each vector becomes [Y0..Y31, c0..c31].
      let p0p = _mm512_permutexvar_epi64(pack_fixup, p0s);
      let p1p = _mm512_permutexvar_epi64(pack_fixup, p1s);

      // Cross-vector merge: collect Y from low 256 of both, chroma
      // from high 256 of both.
      let y_vec = _mm512_permutex2var_epi64(p0p, merge_low, p1p);
      let chroma_vec = _mm512_permutex2var_epi64(p0p, merge_high, p1p);

      // Split chroma into evens / odds via per-lane shuffle + permute.
      let cs = _mm512_shuffle_epi8(chroma_vec, chroma_split);
      let cs_p = _mm512_permutexvar_epi64(pack_fixup, cs);
      // cs_p has 32 c-evens in low 256, 32 c-odds in high 256.

      // For SWAP_UV = false (YUYV / UYVY) → c_evens = U.
      // For SWAP_UV = true  (YVYU)        → c_odds  = U.
      // The yuv_420 AVX-512 kernel reads u_vec / v_vec via
      // `_mm256_loadu_si256` (32 bytes) — extract 256-bit halves.
      let u_vec_256 = if SWAP_UV {
        _mm512_extracti64x4_epi64::<1>(cs_p)
      } else {
        _mm512_castsi512_si256(cs_p)
      };
      let v_vec_256 = if SWAP_UV {
        _mm512_castsi512_si256(cs_p)
      } else {
        _mm512_extracti64x4_epi64::<1>(cs_p)
      };

      // From here, math byte-identical to yuv_420's AVX-512 kernel.
      let u_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_vec_256), mid128);
      let v_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_vec_256), mid128);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

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

/// AVX‑512 YUYV422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuyv422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

/// AVX‑512 UYVY422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn uyvy422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<false>(packed, luma_out, width);
  }
}

/// AVX‑512 YVYU422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yvyu422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn yuv422_packed_to_luma_row<const Y_LSB: bool>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let merge_low = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    let split_mask = if Y_LSB {
      _mm512_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5,
        7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10,
        12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
      )
    } else {
      _mm512_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4,
        6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11,
        13, 15, 0, 2, 4, 6, 8, 10, 12, 14,
      )
    };

    let mut x = 0usize;
    while x + 64 <= width {
      let p0 = _mm512_loadu_si512(packed.as_ptr().add(x * 2).cast());
      let p1 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 64).cast());
      let p0s = _mm512_shuffle_epi8(p0, split_mask);
      let p1s = _mm512_shuffle_epi8(p1, split_mask);
      let p0p = _mm512_permutexvar_epi64(pack_fixup, p0s);
      let p1p = _mm512_permutexvar_epi64(pack_fixup, p1s);
      let y_vec = _mm512_permutex2var_epi64(p0p, merge_low, p1p);
      _mm512_storeu_si512(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 64;
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
