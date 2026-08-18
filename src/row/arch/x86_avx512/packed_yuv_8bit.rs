use super::*;

/// 64-byte byte-shuffle mask for `_mm512_shuffle_epi8`: per 128-bit
/// lane, gathers Y bytes from even positions into the low 8 lanes
/// and chroma bytes from odd positions into the high 8 lanes.
/// Replicated across all 4 × 128-bit lanes of a `__m512i`.
///
/// Loaded via `_mm512_loadu_si512` because Rust's stdarch ships
/// `_mm512_setr_epi64` but not `_mm512_setr_epi8`.
#[rustfmt::skip]
static SPLIT_MASK_Y_LSB: [i8; 64] = [
  0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
  0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
  0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
  0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
];

/// Mirror of [`SPLIT_MASK_Y_LSB`] for the `Y_LSB = false` (UYVY)
/// layout — Y bytes in odd positions, chroma in even.
#[rustfmt::skip]
static SPLIT_MASK_Y_MSB: [i8; 64] = [
  1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14,
  1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14,
  1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14,
  1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14,
];

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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
  matrix: KernelMatrix,
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
    // Loaded from a static `[i8; 64]` because stdarch ships
    // `_mm512_setr_epi64` but not `_mm512_setr_epi8`.
    let split_mask = if Y_LSB {
      _mm512_loadu_si512(SPLIT_MASK_Y_LSB.as_ptr().cast())
    } else {
      _mm512_loadu_si512(SPLIT_MASK_Y_MSB.as_ptr().cast())
    };

    // Cross-vector merge indices (for `permutex2var_epi64` selecting
    // qwords from concat(v0, v1) where v1 starts at index 8).
    let merge_low = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    let merge_high = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

    // Chroma split mask — identical to the `Y_LSB = true` split mask
    // applied to a 64-byte chroma vector.
    let chroma_split = _mm512_loadu_si512(SPLIT_MASK_Y_LSB.as_ptr().cast());

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

// ---- Packed YUV 4:2:2 (8-bit) → HSV (staged via a reused RGB chunk) --
//
// The SIMD twins of the scalar `*_to_hsv_row` kernels. Rather than
// re-derive an HSV-specific register pipeline, each fills a small fixed
// reused RGB scratch (one `HSV_CHUNK`-pixel chunk at a time) using the
// EXISTING AVX-512 packed-4:2:2 RGB kernel — so the chunk filler IS the
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

/// AVX-512: YUYV422 (4:2:2) → planar HSV bytes (OpenCV encoding), staged
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
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512: UYVY422 (4:2:2) → planar HSV bytes, staged via this backend's
/// [`uyvy422_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`yuyv422_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512: YVYU422 (4:2:2) → planar HSV bytes, staged via this backend's
/// [`yvyu422_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`yuyv422_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 YUYV422 → u16 luma extraction. Block size: 32 px / iter.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuyv422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<true>(packed, out, width);
  }
}

/// AVX-512 UYVY422 → u16 luma extraction. Block size: 32 px / iter.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn uyvy422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<false>(packed, out, width);
  }
}

/// AVX-512 YVYU422 → u16 luma extraction (Y positions same as YUYV).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yvyu422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<true>(packed, out, width);
  }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn yuv422_packed_to_luma_u16_row<const Y_LSB: bool>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX-512BW availability is the caller's obligation.
  unsafe {
    // Reuse the same 512-bit gather pipeline as the u8 luma kernel:
    // shuffle each 64-byte chunk per-128-bit-lane, then permute the
    // 64-bit chunks across lanes (`pack_fixup`) and merge across the
    // two vectors (`merge_low`) so all 64 Y bytes land contiguously in
    // a single __m512i. Then widen 64 u8 → 64 u16 via two
    // `_mm512_cvtepu8_epi16` calls on the two 256-bit halves.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let merge_low = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    let split_mask = if Y_LSB {
      _mm512_loadu_si512(SPLIT_MASK_Y_LSB.as_ptr().cast())
    } else {
      _mm512_loadu_si512(SPLIT_MASK_Y_MSB.as_ptr().cast())
    };

    let mut x = 0usize;
    while x + 64 <= width {
      // Load 128 packed bytes (64 px × 2 bytes/px).
      let p0 = _mm512_loadu_si512(packed.as_ptr().add(x * 2).cast());
      let p1 = _mm512_loadu_si512(packed.as_ptr().add(x * 2 + 64).cast());
      let p0s = _mm512_shuffle_epi8(p0, split_mask);
      let p1s = _mm512_shuffle_epi8(p1, split_mask);
      let p0p = _mm512_permutexvar_epi64(pack_fixup, p0s);
      let p1p = _mm512_permutexvar_epi64(pack_fixup, p1s);
      // y_vec holds 64 contiguous Y bytes — same layout as the u8 kernel.
      let y_vec = _mm512_permutex2var_epi64(p0p, merge_low, p1p);
      // Widen 64 u8 → 64 u16: split y_vec into two 256-bit halves and
      // zero-extend each via _mm512_cvtepu8_epi16 (32 u8 → 32 u16).
      let y_lo_256 = _mm512_castsi512_si256(y_vec);
      let y_hi_256 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let w_lo = _mm512_cvtepu8_epi16(y_lo_256); // 32 u16
      let w_hi = _mm512_cvtepu8_epi16(y_hi_256); // 32 u16
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), w_lo);
      _mm512_storeu_si512(out.as_mut_ptr().add(x + 32).cast(), w_hi);
      x += 64;
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
      _mm512_loadu_si512(SPLIT_MASK_Y_LSB.as_ptr().cast())
    } else {
      _mm512_loadu_si512(SPLIT_MASK_Y_MSB.as_ptr().cast())
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
