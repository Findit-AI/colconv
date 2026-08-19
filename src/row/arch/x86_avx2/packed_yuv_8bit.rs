use super::*;

/// AVX2 YUYV422 → packed RGB. Semantics match
/// [`scalar::yuyv422_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `packed.len() >= 2 * width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuyv422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 YUYV422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuyv422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX2 UYVY422 → packed RGB.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyvy422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 UYVY422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyvy422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX2 YVYU422 → packed RGB.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yvyu422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 YVYU422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yvyu422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// Generic packed YUV 4:2:2 → RGB / RGBA AVX2 kernel. 32 px / iter.
///
/// Per-lane shuffle + cross-lane permute deinterleave: load 64 packed
/// bytes via two `_mm256_loadu_si256`, run per-lane `shuffle_epi8` to
/// put Y/chroma into the low/high 64 bits of each lane, then a
/// `permute4x64<0xD8>` consolidation + `permute2x128` cross-vector
/// merge yields 32 Y bytes in one register and 32 chroma bytes in
/// another. A second pass on the chroma vector splits U / V into
/// 16-byte halves, after which the math mirrors `yuv_420`.
///
/// # Safety
///
/// Caller has verified AVX2. `packed.len() >= 2 * width`. `width`
/// even. `out.len() >= bpp * width`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yuv422_packed_to_rgb_or_rgba_row<
  const Y_LSB: bool,
  const SWAP_UV: bool,
  const ALPHA: bool,
>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let mid128 = _mm256_set1_epi16(128);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    // Per-lane split mask: within each 128-bit lane, gather Y bytes
    // in the low 8 lanes (positions 0..7) and chroma bytes in the
    // high 8 lanes (positions 8..15). Mask is replicated across both
    // 128-bit halves of the 256-bit vector.
    let split_mask = if Y_LSB {
      _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // low lane
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // high lane
      )
    } else {
      _mm256_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, // low lane
        1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, // high lane
      )
    };

    // Mask to split chroma bytes (16-byte vector) into evens (low 8)
    // and odds (high 8) — replicated across both 128-bit lanes of
    // the 256-bit chroma vector.
    let chroma_split = _mm256_setr_epi8(
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // low lane
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // high lane
    );

    let mut x = 0usize;
    while x + 32 <= width {
      // Load 64 packed bytes (32 pixels = 16 chroma pairs).
      let p0 = _mm256_loadu_si256(packed.as_ptr().add(x * 2).cast());
      let p1 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 32).cast());

      // Per-lane shuffle: each 128-bit lane → [Y_8, c_8].
      let p0s = _mm256_shuffle_epi8(p0, split_mask);
      let p1s = _mm256_shuffle_epi8(p1, split_mask);

      // Consolidate within each 256-bit vector via permute4x64<0xD8>:
      // pre  = [Y_8a, c_8a, Y_8b, c_8b]   (each 64-bit chunk)
      // post = [Y_8a, Y_8b, c_8a, c_8b]   (Y in low 128, chroma in high 128)
      let p0p = _mm256_permute4x64_epi64::<0xD8>(p0s);
      let p1p = _mm256_permute4x64_epi64::<0xD8>(p1s);

      // Cross-vector merge: collect Y from low 128 of both, chroma
      // from high 128 of both.
      let y_vec = _mm256_permute2x128_si256::<0x20>(p0p, p1p);
      let chroma_vec = _mm256_permute2x128_si256::<0x31>(p0p, p1p);

      // Split chroma evens / odds: per-lane shuffle + permute4x64.
      let cs = _mm256_shuffle_epi8(chroma_vec, chroma_split);
      let cs_p = _mm256_permute4x64_epi64::<0xD8>(cs);
      // cs_p has 16 c-evens in low 128, 16 c-odds in high 128.

      // For SWAP_UV = false (YUYV / UYVY) → c_evens = U, c_odds = V.
      // For SWAP_UV = true  (YVYU)        → c_evens = V, c_odds = U.
      // The `yuv_420` AVX2 kernel reads u_vec / v_vec via
      // `_mm_loadu_si128` (16 bytes) — so we extract 128-bit halves
      // from cs_p.
      let u_vec_128 = if SWAP_UV {
        _mm256_extracti128_si256::<1>(cs_p) // c_odds
      } else {
        _mm256_castsi256_si128(cs_p) // c_evens
      };
      let v_vec_128 = if SWAP_UV {
        _mm256_castsi256_si128(cs_p)
      } else {
        _mm256_extracti128_si256::<1>(cs_p)
      };

      // From here, the math is byte-identical to yuv_420's AVX2 kernel.
      let u_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_vec_128), mid128);
      let v_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_vec_128), mid128);

      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma);

      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
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

// ---- Packed YUV 4:2:2 (8-bit) → HSV (staged via a reused RGB chunk) --
//
// The SIMD twins of the scalar `*_to_hsv_row` kernels. Rather than
// re-derive an HSV-specific register pipeline, each fills a small fixed
// reused RGB scratch (one `HSV_CHUNK`-pixel chunk at a time) using the
// EXISTING AVX2 packed-4:2:2 RGB kernel — so the chunk filler IS the
// production RGB kernel — then runs this backend's `rgb_to_hsv_row` on
// the chunk. This keeps the per-format SIMD surface tiny (only the
// chunked driver is new) and makes the result byte-identical to
// `rgb_to_hsv_row(*_to_rgb_row(...))` within this tier. The scalar tail
// of each underlying RGB kernel handles widths below the SIMD block, so
// no separate tail is needed here.
//
// This driver is LOCAL to the packed family (the shared
// `yuv_to_hsv_via_rgb_chunks` is gated on `yuv-planar`; the packed
// formats compile under `yuv-packed` alone) and shared by both packed
// files of this arch — the sibling 4:1:1 module reaches it via
// `super::packed_yuv_8bit`. `HSV_CHUNK = 64` is even AND a multiple of 4,
// so every chunk offset lands on a 4:2:2 (4-byte) AND a 4:1:1 (6-byte)
// block boundary.

/// One reused RGB chunk's worth of pixels staged before the HSV pass.
pub(super) const HSV_CHUNK: usize = 64;

/// Shared driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing RGB kernel
/// for the format, passed the chunk `offset` and length `n`), then runs
/// [`rgb_to_hsv_row`] on that chunk into the H/S/V planes. Byte-identical
/// to `rgb_to_hsv_row(*_to_rgb_row(...))` within this tier, with no
/// source-width RGB allocation. Shared by the packed 4:2:2 kernels here
/// and the 4:1:1 kernel in the sibling module.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// The SIMD feature must be available, and `fill_rgb` must uphold the
/// underlying RGB kernel's safety contract for each chunk. Each of
/// `h_out` / `s_out` / `v_out` must be `>= width`.
#[inline]
pub(super) unsafe fn packed_hsv_via_rgb_chunks(
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
    // SAFETY: SIMD verified by the wrapper's `#[target_feature]`; the
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

/// AVX2: YUYV422 (4:2:2) → planar HSV bytes (OpenCV encoding), staged
/// via the reused-RGB-chunk pattern over this backend's
/// [`yuyv422_to_rgb_row`] + [`rgb_to_hsv_row`]. Byte-identical to
/// `rgb_to_hsv_row(yuyv422_to_rgb_row(...))` within this tier.
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `width & 1 == 0`.
/// 3. `packed.len() >= 2 * width`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuyv422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the chunk filler forwards the per-chunk
  // sub-slices to this backend's YUYV422 RGB kernel under the same
  // contract. The packed byte offset for the chunk at pixel `offset` is
  // `offset * 2` (2 bytes per pixel).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuyv422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// AVX2: UYVY422 (4:2:2) → planar HSV bytes, staged via this backend's
/// [`uyvy422_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`yuyv422_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn uyvy422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; forwards to this backend's UYVY422 RGB kernel
  // under the same contract (packed byte offset = `offset * 2`).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      uyvy422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// AVX2: YVYU422 (4:2:2) → planar HSV bytes, staged via this backend's
/// [`yvyu422_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`yuyv422_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yvyu422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; forwards to this backend's YVYU422 RGB kernel
  // under the same contract (packed byte offset = `offset * 2`).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yvyu422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// AVX2 YUYV422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuyv422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

/// AVX2 UYVY422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyvy422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<false>(packed, luma_out, width);
  }
}

/// AVX2 YVYU422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yvyu422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

/// AVX2 YUYV422 → u16 luma extraction. Block size: 32 px / iter.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuyv422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<true>(packed, out, width);
  }
}

/// AVX2 UYVY422 → u16 luma extraction. Block size: 32 px / iter.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyvy422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<false>(packed, out, width);
  }
}

/// AVX2 YVYU422 → u16 luma extraction (Y positions same as YUYV).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yvyu422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<true>(packed, out, width);
  }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yuv422_packed_to_luma_u16_row<const Y_LSB: bool>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    // Per-lane shuffle: moves Y bytes to the low 8 positions of each
    // 128-bit lane. The AVX2 shuffle operates per 128-bit lane
    // independently, so each loaded 32-byte chunk yields 8 + 8 = 16 Y
    // bytes (8 per lane), with the upper 8 of each lane zeroed.
    let split_mask = if Y_LSB {
      _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14, -1,
        -1, -1, -1, -1, -1, -1, -1,
      )
    } else {
      _mm256_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5, 7, 9, 11, 13, 15, -1,
        -1, -1, -1, -1, -1, -1, -1,
      )
    };

    let mut x = 0usize;
    while x + 32 <= width {
      // Load 32 px worth of packed data (2 × 32-byte AVX2 loads; each
      // covers 16 packed pixels = 16 Y bytes split 8 / 8 across lanes).
      let p0 = _mm256_loadu_si256(packed.as_ptr().add(x * 2).cast());
      let p1 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 32).cast());
      // After shuffle: bytes [0..8) of each 128-bit lane = Y values
      // (low 8 of each lane). Bytes [8..16) of each lane are zeroed.
      let p0s = _mm256_shuffle_epi8(p0, split_mask);
      let p1s = _mm256_shuffle_epi8(p1, split_mask);
      // Extract BOTH 128-bit halves of each shuffled vector — each
      // half holds 8 valid Y bytes in its low 8 lanes. `_mm_cvtepu8_epi16`
      // widens the low 8 u8 → 8 u16 (one __m128i = 16 bytes), discarding
      // the (zeroed) high 8 bytes; this matches the Y-bytes-only contract.
      let p0_lo = _mm256_castsi256_si128(p0s); // 8 Y bytes from p0 lane 0
      let p0_hi = _mm256_extracti128_si256::<1>(p0s); // 8 Y bytes from p0 lane 1
      let p1_lo = _mm256_castsi256_si128(p1s); // 8 Y bytes from p1 lane 0
      let p1_hi = _mm256_extracti128_si256::<1>(p1s); // 8 Y bytes from p1 lane 1
      let w0_lo = _mm_cvtepu8_epi16(p0_lo); // 8 u16
      let w0_hi = _mm_cvtepu8_epi16(p0_hi); // 8 u16
      let w1_lo = _mm_cvtepu8_epi16(p1_lo); // 8 u16
      let w1_hi = _mm_cvtepu8_epi16(p1_hi); // 8 u16
      // Combine each pair into a __m256i (16 u16 = 32 bytes) for stores.
      let w_lo = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w0_lo), w0_hi);
      let w_hi = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w1_lo), w1_hi);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), w_lo);
      _mm256_storeu_si256(out.as_mut_ptr().add(x + 16).cast(), w_hi);
      x += 32;
    }
    if x < width {
      if Y_LSB {
        scalar::yuyv422_to_luma_u16_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
      } else {
        scalar::uyvy422_to_luma_u16_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
      }
    }
  }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yuv422_packed_to_luma_row<const Y_LSB: bool>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let split_mask = if Y_LSB {
      _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5,
        7, 9, 11, 13, 15,
      )
    } else {
      _mm256_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4,
        6, 8, 10, 12, 14,
      )
    };

    let mut x = 0usize;
    while x + 32 <= width {
      let p0 = _mm256_loadu_si256(packed.as_ptr().add(x * 2).cast());
      let p1 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 32).cast());
      let p0s = _mm256_shuffle_epi8(p0, split_mask);
      let p1s = _mm256_shuffle_epi8(p1, split_mask);
      let p0p = _mm256_permute4x64_epi64::<0xD8>(p0s);
      let p1p = _mm256_permute4x64_epi64::<0xD8>(p1s);
      let y_vec = _mm256_permute2x128_si256::<0x20>(p0p, p1p);
      _mm256_storeu_si256(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 32;
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
