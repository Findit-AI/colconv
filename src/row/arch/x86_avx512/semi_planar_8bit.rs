use super::*;

/// AVX‑512 NV12 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_or_nv21_to_rgb_or_rgba_row_impl`]:
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
///    Direct callers are responsible for verifying this; the
///    dispatcher in [`crate::row::nv12_to_rgb_row`] checks it.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `uv_half.len() >= width` (interleaved UV bytes, 2 per chroma pair).
/// 5. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 NV21 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_to_rgb_row`]; `vu_half` carries the same
/// number of bytes (`>= width`) but in V-then-U order per chroma
/// pair.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
      y, vu_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 NV12 → packed RGBA. Same contract as [`nv12_to_rgb_row`]
/// but writes 4 bytes per pixel via [`write_rgba_64`].
/// `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv12_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes (one extra byte per pixel for the opaque
/// alpha).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 NV21 → packed RGBA. Same contract as [`nv21_to_rgb_row`]
/// but writes 4 bytes per pixel via [`write_rgba_64`].
/// `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv21_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
      y, vu_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX‑512 NV12/NV21 kernel at 3 bpp (RGB) or 4 bpp + opaque
/// alpha (RGBA). `SWAP_UV` selects chroma byte order; `ALPHA = true`
/// writes via [`write_rgba_64`], `ALPHA = false` via [`write_rgb_64`].
/// Both const generics drive compile-time monomorphization.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (64 interleaved bytes per 64 Y pixels).
/// 5. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn nv12_or_nv21_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu_half.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation; all
  // pointer adds below are bounded by the `while x + 64 <= width`
  // condition and the caller‑promised slice lengths.
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
    let alpha_u8 = _mm512_set1_epi8(-1); // 0xFF as i8

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    // Per‑128‑bit‑lane UV deinterleave mask. Broadcast to all 4 lanes.
    // Within each 16‑byte chunk, pack even‑offset (U) bytes into the
    // low 8 lanes and odd‑offset (V) bytes into the high 8 lanes.
    let uv_lane_mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After per‑lane shuffle the 64‑bit lane layout is
    // `[U0, V0, U1, V1, U2, V2, U3, V3]`; permuting with
    // `[0, 2, 4, 6, 1, 3, 5, 7]` compacts to
    // `[U0, U1, U2, U3 | V0, V1, V2, V3]` — low 256 = U, high 256 = V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 64 Y pixels → 32 chroma pairs = 64 interleaved bytes at
      // offset `x` in the chroma row.
      let uv_vec = _mm512_loadu_si512(uv_or_vu_half.as_ptr().add(x).cast());

      // Per-lane shuffle + permute packs even-offset bytes into low
      // 256 and odd-offset bytes into high 256. For NV12 that's
      // (U, V); for NV21 the roles swap.
      let deint = _mm512_shuffle_epi8(uv_vec, uv_deint_mask);
      let uv_compact = _mm512_permutexvar_epi64(uv_collect, deint);
      let (u_vec_256, v_vec_256) = if SWAP_UV {
        (
          _mm512_extracti64x4_epi64::<1>(uv_compact),
          _mm512_castsi512_si256(uv_compact),
        )
      } else {
        (
          _mm512_castsi512_si256(uv_compact),
          _mm512_extracti64x4_epi64::<1>(uv_compact),
        )
      };

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

/// AVX-512 NV24 → packed RGB (UV-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, false>(y, uv, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 NV42 → packed RGB (VU-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, false>(y, vu, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 NV24 → packed RGBA (UV-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv24_to_rgba_row(
  y: &[u8],
  uv: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, true>(y, uv, rgba_out, width, matrix, full_range);
  }
}

/// AVX-512 NV42 → packed RGBA (VU-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv42_to_rgba_row(
  y: &[u8],
  vu: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, true>(y, vu, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX-512 NV24/NV42 kernel (4:4:4 semi-planar). 64 Y pixels /
/// 64 chroma pairs / 128 UV bytes per iteration. Unlike
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`], chroma is not subsampled — one
/// UV pair per Y pixel — so the `chroma_dup` step disappears; two
/// `chroma_i16x32` calls per channel produce 64 chroma values
/// directly.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`.
/// 3. `uv_or_vu.len() >= 2 * width`.
/// 4. `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn nv24_or_nv42_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; all
  // pointer adds below are bounded by the `while x + 64 <= width`
  // loop and the caller-promised slice lengths.
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

    // Same lane fixups as NV12 kernel — inherited verbatim.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // Per-128-bit-lane UV deinterleave mask, broadcast to all 4 lanes.
    let uv_lane_mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After per-lane shuffle the 64-bit lane layout is
    // `[U0, V0, U1, V1, U2, V2, U3, V3]`; this permute collects to
    // `[U0, U1, U2, U3 | V0, V1, V2, V3]` — low 256 = U, high 256 = V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 64 Y pixels → 128 UV bytes (two 512-bit loads).
      let uv_vec_lo = _mm512_loadu_si512(uv_or_vu.as_ptr().add(x * 2).cast());
      let uv_vec_hi = _mm512_loadu_si512(uv_or_vu.as_ptr().add(x * 2 + 64).cast());

      // Per 512-bit vec: deinterleave → low 256 = U (32 bytes),
      // high 256 = V (32 bytes). Roles swap for NV42.
      let d_lo =
        _mm512_permutexvar_epi64(uv_collect, _mm512_shuffle_epi8(uv_vec_lo, uv_deint_mask));
      let d_hi =
        _mm512_permutexvar_epi64(uv_collect, _mm512_shuffle_epi8(uv_vec_hi, uv_deint_mask));
      let (u_bytes_lo_256, v_bytes_lo_256, u_bytes_hi_256, v_bytes_hi_256) = if SWAP_UV {
        (
          _mm512_extracti64x4_epi64::<1>(d_lo),
          _mm512_castsi512_si256(d_lo),
          _mm512_extracti64x4_epi64::<1>(d_hi),
          _mm512_castsi512_si256(d_hi),
        )
      } else {
        (
          _mm512_castsi512_si256(d_lo),
          _mm512_extracti64x4_epi64::<1>(d_lo),
          _mm512_castsi512_si256(d_hi),
          _mm512_extracti64x4_epi64::<1>(d_hi),
        )
      };

      // Widen each 32-byte U/V chunk to i16x32 and subtract 128.
      let u_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_bytes_lo_256), mid128);
      let u_hi_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_bytes_hi_256), mid128);
      let v_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_bytes_lo_256), mid128);
      let v_hi_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_bytes_hi_256), mid128);

      // Split each i16x32 into two i32x16 halves.
      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      // u_d / v_d = (u * c_scale + RND) >> 15.
      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      // 64 chroma per channel (two chroma_i16x32 per channel, no
      // duplication).
      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);

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

// ---- Semi-planar 8-bit NV → HSV (staged via a reused RGB chunk) ------
//
// The SIMD twins of the scalar `nv*_to_hsv_row` kernels. Rather than
// re-derive an HSV-specific register pipeline, each fills a small fixed
// reused RGB scratch (one `HSV_CHUNK`-pixel chunk at a time) using the
// EXISTING `nv*_to_rgb_row` kernel of this backend — so the chunk filler
// IS the production RGB kernel — then runs this backend's
// `rgb_to_hsv_row` on the chunk. This keeps the per-format SIMD surface
// tiny (only the chunked driver is new) and makes the result
// byte-identical to `rgb_to_hsv_row(nv*_to_rgb_row(...))` within this
// tier. The scalar tail of each underlying RGB kernel handles widths
// below the SIMD block, so no separate tail is needed here.
//
// `HSV_CHUNK = 64` is a multiple of 2, so every chunk offset lands on a
// chroma-sample boundary for the 1→2 (4:2:0 / 4:2:2) shape and trivially
// for the 1→1 (4:4:4) shape.

/// One reused RGB chunk's worth of pixels staged before the HSV pass.
const HSV_CHUNK: usize = 64;

/// Shared driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs [`rgb_to_hsv_row`] on that chunk into the H/S/V planes. The
/// result is byte-identical to `rgb_to_hsv_row(nv*_to_rgb_row(...))`
/// within this tier, with no source-width RGB allocation.
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
unsafe fn nv_to_hsv_via_rgb_chunks(
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

/// NV12 (4:2:0, UV-ordered) → planar HSV bytes (OpenCV encoding), staged
/// via this backend's [`nv12_to_rgb_row`] + [`rgb_to_hsv_row`]. Also
/// serves NV16 (4:2:2 — identical per-row chroma shape). Byte-identical
/// to `rgb_to_hsv_row(nv12_to_rgb_row(...))` within this tier.
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nv12_to_hsv_row(
  y: &[u8],
  uv_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV16 require even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "chroma row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the chunk filler forwards the per-chunk
  // sub-slices to this backend's NV12 RGB kernel under the same
  // contract. The 4:2:0 chroma byte offset for the chunk at pixel
  // `offset` is `offset` bytes (one UV pair per two pixels).
  unsafe {
    nv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      nv12_to_rgb_row(&y[offset..], &uv_half[offset..], rgb, n, matrix, full_range);
    });
  }
}

/// NV21 (4:2:0, VU-ordered) → planar HSV bytes, staged via this
/// backend's [`nv21_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`nv12_to_hsv_row`]; `vu_half` carries the same
/// `width` chroma bytes in V-then-U order per pair.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nv21_to_hsv_row(
  y: &[u8],
  vu_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV21 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(vu_half.len() >= width, "chroma row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; forwards to this backend's NV21 RGB kernel
  // under the same contract (4:2:0 chroma byte offset = `offset`).
  unsafe {
    nv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      nv21_to_rgb_row(&y[offset..], &vu_half[offset..], rgb, n, matrix, full_range);
    });
  }
}

/// NV24 (4:4:4, UV-ordered) → planar HSV bytes, staged via this
/// backend's [`nv24_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `y.len() >= width`, `uv.len() >= 2 * width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nv24_to_hsv_row(
  y: &[u8],
  uv: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv.len() >= 2 * width, "chroma row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; forwards to this backend's NV24 RGB kernel
  // under the same contract. The 4:4:4 chroma byte offset for the chunk
  // at pixel `offset` is `offset * 2` (one UV pair per pixel).
  unsafe {
    nv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      nv24_to_rgb_row(&y[offset..], &uv[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// NV42 (4:4:4, VU-ordered) → planar HSV bytes, staged via this
/// backend's [`nv42_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`nv24_to_hsv_row`]; `vu` carries the same
/// `2 * width` chroma bytes in V-then-U order per pair.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nv42_to_hsv_row(
  y: &[u8],
  vu: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(vu.len() >= 2 * width, "chroma row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; forwards to this backend's NV42 RGB kernel
  // under the same contract (4:4:4 chroma byte offset = `offset * 2`).
  unsafe {
    nv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      nv42_to_rgb_row(&y[offset..], &vu[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}
