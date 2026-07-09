use super::*;

/// AVX‑512 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
///    The dispatcher in [`crate::row`] verifies this with
///    `is_x86_feature_detected!("avx512bw")` (runtime, std) or
///    `cfg!(target_feature = "avx512bw")` (compile‑time, no‑std).
///    AVX‑512BW implies AVX‑512F on all real CPUs. Calling this kernel
///    on a CPU without AVX‑512BW triggers an illegal‑instruction trap.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`_mm512_loadu_si512`, `_mm256_loadu_si256`,
/// `_mm_storeu_si128` inside `write_rgb_16`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 YUV 4:2:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. AVX‑512F + AVX‑512BW must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUVA 4:2:0 → packed **8-bit RGBA** with the per-pixel
/// alpha byte **sourced from `a_src`** (8-bit YUVA's alpha is already
/// `u8` — no depth conversion needed). Same numerical contract as
/// [`yuv_420_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgba_row`] plus `a_src.len() >= width`.
#[cfg(feature = "yuva")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420_to_rgba_with_alpha_src_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true, true>(
      y,
      u_half,
      v_half,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared AVX‑512 kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, [`write_rgb_64`]), [`yuv_420_to_rgba_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, [`write_rgba_64`] with constant
/// `0xFF` alpha) and [`yuv_420_to_rgba_with_alpha_src_row`]
/// (`ALPHA = true, ALPHA_SRC = true`, [`write_rgba_64`] with the
/// alpha lane loaded directly from `a_src`).
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long. When `ALPHA_SRC = true`: `a_src` must be `Some(_)`
/// and `a_src.unwrap().len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation per the
  // `# Safety` section; the dispatcher in `crate::row` checks it.
  // All pointer adds below are bounded by the `while x + 64 <= width`
  // loop condition and the caller‑promised slice lengths.
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
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = _mm512_set1_epi8(-1); // 0xFF as i8

    // Lane‑fixup permute indices, computed once per call.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec_256 = _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast());
      let v_vec_256 = _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast());

      // Widen U/V to i16x32 and subtract 128.
      let u_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_vec_256), mid128);
      let v_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_vec_256), mid128);

      // Split each i16x32 into two i32x16 halves for the Q15 multiplies.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
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

      // Per‑channel chroma → i16x32 (natural order after pack fixup).
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // Nearest‑neighbor upsample: pair‑duplicate each chroma lane into
      // two i16x32 vectors covering 64 Y lanes.
      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      // Y path: widen 64 Y to two i16x32, scale.
      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x64 per channel with the same pack fixup.
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit YUVA alpha is already u8; load 64 bytes via 512-bit
          // load.
          _mm512_loadu_si512(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u8
        };
        // 4‑way interleave → packed RGBA (256 bytes = 4 × 64).
        write_rgba_64(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        // 3‑way interleave → packed RGB (192 bytes = 4 × 48).
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

    // Scalar tail for the 0..62 leftover pixels (always even; 4:2:0
    // requires even width so x/2 and width/2 are well‑defined).
    if x < width {
      let tail_a = if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        Some(&a_src.as_ref().unwrap_unchecked()[x..width])
      } else {
        None
      };
      scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        tail_a,
        &mut out[x * bpp..width * bpp],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}
// ---- YUV 4:1:0 AVX-512 entries --------------------------------------
//
// 4:1:0: planar YUV with chroma subsampled 4:1 in **both** axes. Each
// (U, V) sample covers a 4×4 luma block; vertical 4× re-use is the
// walker's job (chroma row = `y_row / 4`). This kernel handles the
// per-row 4× horizontal upsample. Math is byte-identical to scalar.
//
// Block size: 64 Y / 16 chroma per iteration (matches the 4:2:0
// AVX-512 kernel's 64-Y throughput). Chroma upsample uses
// `_mm512_permutexvar_epi16` with 4× duplicate indices — one
// permute per channel produces the i16x32 vector covering Y[0..32]
// and another covers Y[32..64]. This is much simpler than the
// 4:2:0 kernel's pair-duplicate path because we go straight from
// per-chroma to 4×-fanned via a single lane-crossing permute.

/// AVX-512 YUV 4:1:0 → packed RGB. Semantics match
/// [`scalar::yuv_410_to_rgb_row`] byte-identically.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width % 4 == 0` (4:1:0 requires width multiple of 4).
/// 3. `y.len() >= width`, `u_quarter.len() >= width / 4`,
///    `v_quarter.len() >= width / 4`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_410_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_410_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_410_to_rgb_or_rgba_row::<false>(
      y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUV 4:1:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_410_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_410_to_rgb_row`] except `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_410_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_410_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_410_to_rgb_or_rgba_row::<true>(
      y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX-512 kernel for [`yuv_410_to_rgb_row`] (`ALPHA = false`,
/// [`write_rgb_64`]) and [`yuv_410_to_rgba_row`] (`ALPHA = true`,
/// [`write_rgba_64`] with constant `0xFF` alpha). Math is byte-
/// identical to `scalar::yuv_410_to_rgb_or_rgba_row::<ALPHA>`.
///
/// Pipeline per 64 Y pixels / 16 chroma samples:
/// 1. Load 64 Y (`_mm512_loadu_si512`) + 16 U + 16 V (each as
///    `_mm_loadu_si128` from quarter-width planes).
/// 2. Widen 16 chroma → i16x16 (256-bit XMM/YMM), subtract 128.
/// 3. Widen to i32x16 (512-bit) for Q15 multiplies.
/// 4. `u_d = (u * c_scale + RND) >> 15`, same for `v_d` (i32x16).
/// 5. Per channel: `(C_u*u_d + C_v*v_d + RND) >> 15` (i32x16),
///    saturate-pack to i16x32 with `pack_fixup` lane permute (only
///    the low 16 i16 lanes carry chroma; high 16 are duplicates we
///    discard via the permute below).
/// 6. Apply two `_mm512_permutexvar_epi16` permutes per channel:
///    one with index `[0,0,0,0, 1,1,1,1, ..., 7,7,7,7]` (covering
///    Y[0..32]) and one with index `[8,8,8,8, ..., 15,15,15,15]`
///    (covering Y[32..64]).
/// 7. Y path: widen 64 Y → two i16x32, scale.
/// 8. Saturating add Y + chroma per channel (i16x32), saturate-
///    narrow to u8x64, interleave via [`write_rgb_64`] /
///    [`write_rgba_64`].
///
/// # Safety
///
/// Same as [`yuv_410_to_rgb_row`] / [`yuv_410_to_rgba_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn yuv_410_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  debug_assert!(y.len() >= width);
  debug_assert!(u_quarter.len() >= width / 4);
  debug_assert!(v_quarter.len() >= width / 4);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation per the
  // `# Safety` section. All pointer adds below are bounded by the
  // `while x + 64 <= width` loop condition and caller-promised slice
  // lengths.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let mid128_256 = _mm256_set1_epi16(128);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    // Pack fixup for the i32x16→i16x32 saturating pack: same pattern
    // as the 4:2:0 kernel's `pack_fixup`. Used by the shared
    // `narrow_u8x64` / `scale_y` helpers.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // 4× chroma fan-out indices (i16 lanes).
    // Low half: [c0,c0,c0,c0, c1,c1,c1,c1, ..., c7,c7,c7,c7] — covers
    // Y[0..32]. High half: [c8,c8,c8,c8, ..., c15,c15,c15,c15] —
    // covers Y[32..64]. Built via `_mm512_set_epi16` (reverse order:
    // arg 31 = lane 0, arg 0 = lane 31).
    #[rustfmt::skip]
    let dup_lo_idx = _mm512_set_epi16(
      7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4,
      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0,
    );
    #[rustfmt::skip]
    let dup_hi_idx = _mm512_set_epi16(
      15, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12,
      11, 11, 11, 11, 10, 10, 10, 10,  9,  9,  9,  9,  8,  8,  8,  8,
    );

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());

      // Load 16 chroma bytes per plane (`_mm_loadu_si128`).
      let u_xmm = _mm_loadu_si128(u_quarter.as_ptr().add(x / 4).cast());
      let v_xmm = _mm_loadu_si128(v_quarter.as_ptr().add(x / 4).cast());

      // Widen 16 chroma bytes → i16x16 (YMM), subtract 128.
      let u_i16x16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_xmm), mid128_256);
      let v_i16x16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_xmm), mid128_256);

      // Widen to i32x16 (ZMM) for Q15 multiplies.
      let u_i32x16 = _mm512_cvtepi16_epi32(u_i16x16);
      let v_i32x16 = _mm512_cvtepi16_epi32(v_i16x16);

      // u_d, v_d = (u * c_scale + RND) >> 15.
      let u_d = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32x16, c_scale_v),
        rnd_v,
      ));
      let v_d = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32x16, c_scale_v),
        rnd_v,
      ));

      // Per-channel chroma contribution as i32x16.
      let r_i32 = q15_shift(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cru, u_d), _mm512_mullo_epi32(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = q15_shift(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cgu, u_d), _mm512_mullo_epi32(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = q15_shift(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cbu, u_d), _mm512_mullo_epi32(cbv, v_d)),
        rnd_v,
      ));

      // Saturate-pack i32x16 → i16x32. `_mm512_packs_epi32` is per
      // 128-bit lane; passing the same vector twice gives lane-split
      // duplicate halves. The `permutexvar_epi64` with `pack_fixup`
      // restores natural order so the low 16 i16 lanes hold
      // [c0..c15] contiguously (high 16 lanes are duplicates we
      // discard via the next permute step).
      let r_packed = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(r_i32, r_i32));
      let g_packed = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(g_i32, g_i32));
      let b_packed = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(b_i32, b_i32));

      // 4× fan-out per channel: one permute for Y[0..32], one for
      // Y[32..64]. The permute reads from the low 16 i16 lanes
      // (which hold [c0..c15]) according to the index vectors.
      let r_dup_lo = _mm512_permutexvar_epi16(dup_lo_idx, r_packed);
      let r_dup_hi = _mm512_permutexvar_epi16(dup_hi_idx, r_packed);
      let g_dup_lo = _mm512_permutexvar_epi16(dup_lo_idx, g_packed);
      let g_dup_hi = _mm512_permutexvar_epi16(dup_hi_idx, g_packed);
      let b_dup_lo = _mm512_permutexvar_epi16(dup_lo_idx, b_packed);
      let b_dup_hi = _mm512_permutexvar_epi16(dup_hi_idx, b_packed);

      // Y path: widen 64 Y → two i16x32, scale.
      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate-narrow per channel → u8x64 (with pack fixup).
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

    // Scalar tail. `width % 4 == 0` so `width - x` is also a multiple
    // of 4 (worst case 60 leftover pixels: width = 60 → no SIMD iter).
    if x < width {
      scalar::yuv_410_to_rgb_or_rgba_row::<ALPHA>(
        &y[x..width],
        &u_quarter[x / 4..width / 4],
        &v_quarter[x / 4..width / 4],
        &mut out[x * bpp..width * bpp],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<false, false>(y, u, v, None, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 YUV 4:4:4 planar → packed **RGBA** (8-bit). Same contract
/// as [`yuv_444_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_64`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_444_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true, false>(y, u, v, None, rgba_out, width, matrix, full_range);
  }
}

/// AVX-512 YUVA 4:4:4 → packed **RGBA** with source alpha. R/G/B are
/// byte-identical to [`yuv_444_to_rgb_row`]; the per-pixel alpha byte
/// is sourced from `a_src` (8-bit, no shift needed) instead of being
/// constant `0xFF`. Used by [`crate::source::Yuva444p`].
///
/// Thin wrapper over [`yuv_444_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444_to_rgba_row`] plus `a_src.len() >= width`.
#[cfg(feature = "yuva")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444_to_rgba_with_alpha_src_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true, true>(
      y,
      u,
      v,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared AVX-512 YUV 4:4:4 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: [`write_rgb_64`].
/// - `ALPHA = true, ALPHA_SRC = false`: [`write_rgba_64`] with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: [`write_rgba_64`] with the
///   alpha lane loaded from `a_src` (8-bit input — no shift needed).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
/// 4. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

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

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 4:4:4: 64 U + 64 V directly, no deinterleave.
      let u_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let v_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());

      // Widen low / high halves of U / V (32 bytes each) to i16x32.
      let u_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(u_vec)), mid128);
      let u_hi_i16 = _mm512_sub_epi16(
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(u_vec)),
        mid128,
      );
      let v_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(v_vec)), mid128);
      let v_hi_i16 = _mm512_sub_epi16(
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(v_vec)),
        mid128,
      );

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit alpha — load 64 bytes verbatim.
          _mm512_loadu_si512(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u8
        };
        write_rgba_64(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444_to_rgba_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444_to_rgba_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yuv_444_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- YUV 4:1:1 → RGB / RGBA (AVX-512BW) --------------------------------

/// AVX-512 YUV 4:1:1 planar → packed RGB. One chroma sample drives four
/// Y pixels (1→4 nearest-neighbor upsample). Processes 64 Y / 16 chroma
/// samples per iteration — matches the AVX-512 4:2:0 block size with
/// 1/4 the chroma load count.
///
/// Same Q15 arithmetic as the scalar reference; output is byte-identical.
///
/// FFmpeg-compatible widths: arbitrary `width` accepted. Chroma row
/// is `width.div_ceil(4)` samples; the SIMD body strides 64 Y pixels
/// (multiple of 4), and the trailing 1..63 Y pixels — including any
/// partial 1..3-pixel chroma group — fall through to the scalar
/// reference.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `y.len() >= width`,
///    `u_quarter.len() >= width.div_ceil(4)`,
///    `v_quarter.len() >= width.div_ceil(4)`.
/// 3. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_411_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds — see
  // [`yuv_411_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_411_to_rgb_or_rgba_row::<false>(
      y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUV 4:1:1 planar → packed **RGBA** (8-bit). Same contract as
/// [`yuv_411_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_64`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_411_to_rgb_row`] except `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_411_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds — see
  // [`yuv_411_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_411_to_rgb_or_rgba_row::<true>(
      y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX-512 YUV 4:1:1 kernel. Processes 64 Y pixels (= 16 chroma
/// samples) per iteration; the 1→4 chroma upsample is materialized
/// across two i16x32 vectors via `_mm512_permutexvar_epi16` with a
/// repeat-each-lane-4-times index pattern:
///
/// 1. Load 16 chroma bytes via `_mm_loadu_si128`, widen → i16x16
///    (`__m256i`), subtract 128, sign-extend → i32x16 (`__m512i`).
/// 2. Compute per-channel chroma → i32x16, then saturating-pack to i16
///    via `_mm512_packs_epi32(x, x)`. Apply the standard `pack_fixup`
///    permute so that c0..c15 land in the low 256 bits of `__m512i`.
/// 3. Cross-lane 1→4 fan-out via `_mm512_permutexvar_epi16` with two
///    index vectors:
///    - `dup_lo32_idx = [0×4, 1×4, 2×4, ..., 7×4]` → i16x32 covering
///      Y[0..32].
///    - `dup_hi32_idx = [8×4, 9×4, 10×4, ..., 15×4]` → i16x32 covering
///      Y[32..64].
///
/// 4:1:1 has no source-alpha variant (no `Yuva411p` exists), so the
/// const-generic surface stays 1-D (`ALPHA` only).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `y.len() >= width`,
///    `u_quarter.len() >= width.div_ceil(4)`,
///    `v_quarter.len() >= width.div_ceil(4)`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn yuv_411_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u_quarter.len() >= width.div_ceil(4));
  debug_assert!(v_quarter.len() >= width.div_ceil(4));
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation per the
  // `# Safety` section. All pointer adds below are bounded by the
  // `while x + 64 <= width` loop condition and the caller-promised
  // slice lengths.
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

    // Lane-fixup permute (used by chroma_i16x32 / scale_y / narrow_u8x64).
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // 1→4 fan-out indices for cross-lane `_mm512_permutexvar_epi16`.
    // Build [0×4, 1×4, 2×4, …, 7×4] (32 i16 lanes) for the low half
    // (covering Y[0..32]) and the analogous [8×4 … 15×4] for the high
    // half (Y[32..64]). Constructed once per call from the i16 layout
    // above; AVX-512 has no `setr_epi16`, so we build via `set_epi16`
    // listing lane 31 first.
    let dup_lo32_idx = _mm512_set_epi16(
      7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0,
      0,
    );
    let dup_hi32_idx = _mm512_set_epi16(
      15, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10,
      10, 9, 9, 9, 9, 8, 8, 8, 8,
    );

    let mut x = 0usize;
    while x + 64 <= width {
      // Load 64 Y bytes.
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());

      // Load 16 chroma bytes per 64 Y pixels (4:1:1 quarter-rate).
      let u_16 = _mm_loadu_si128(u_quarter.as_ptr().add(x / 4).cast());
      let v_16 = _mm_loadu_si128(v_quarter.as_ptr().add(x / 4).cast());

      // Widen 16 u8 → i16x16 (__m256i), subtract 128. Then sign-extend
      // to i32x16 (__m512i) for the Q15 multiplies. Only the low 256
      // bits of the i32x16 hold real chroma (16 chroma values, 16 lanes
      // — the entire i32x16); we operate at full 512-bit width. To
      // reuse `chroma_i16x32` (which expects two i32x16 inputs), we
      // pass `u_d` as both halves, but only consume the i16x16 half
      // (lanes 0..15) of its result.
      let u_i16_256 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_16), _mm512_castsi512_si256(mid128));
      let v_i16_256 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_16), _mm512_castsi512_si256(mid128));

      let u_i32 = _mm512_cvtepi16_epi32(u_i16_256);
      let v_i32 = _mm512_cvtepi16_epi32(v_i16_256);

      let u_d = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_v,
      ));
      let v_d = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_v,
      ));

      // Per-channel chroma → i16x16 in the low 256 bits of an __m512i.
      // `_mm512_packs_epi32(x, x)` produces lane-split [c0..3, c0..3,
      // c4..7, c4..7, c8..11, c8..11, c12..15, c12..15]; permute via
      // `pack_fixup = [0,2,4,6,1,3,5,7]` puts c0..c15 in the low 256
      // bits and a duplicate in the high 256 bits (which we don't use).
      let r_i32 = q15_shift(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cru, u_d), _mm512_mullo_epi32(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = q15_shift(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cgu, u_d), _mm512_mullo_epi32(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = q15_shift(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cbu, u_d), _mm512_mullo_epi32(cbv, v_d)),
        rnd_v,
      ));

      let r_c16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(r_i32, r_i32));
      let g_c16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(g_i32, g_i32));
      let b_c16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(b_i32, b_i32));

      // 1→4 cross-lane fan-out via `_mm512_permutexvar_epi16`. Each of
      // the 16 chroma values maps to 4 Y lanes (32 Y per output → 64
      // total). Result is two i16x32 vectors covering the 64 Y lanes
      // in natural order — no further fixup required.
      let r_dup_lo = _mm512_permutexvar_epi16(dup_lo32_idx, r_c16);
      let r_dup_hi = _mm512_permutexvar_epi16(dup_hi32_idx, r_c16);
      let g_dup_lo = _mm512_permutexvar_epi16(dup_lo32_idx, g_c16);
      let g_dup_hi = _mm512_permutexvar_epi16(dup_hi32_idx, g_c16);
      let b_dup_lo = _mm512_permutexvar_epi16(dup_lo32_idx, b_c16);
      let b_dup_hi = _mm512_permutexvar_epi16(dup_hi32_idx, b_c16);

      // Y path: widen 64 Y to two i16x32, scale (with the standard
      // pack fixup applied inside `scale_y`).
      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate-narrow to u8x64 per channel.
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

    // Scalar tail. The SIMD loop strides 64 Y pixels (multiple of 4),
    // so `x` is a multiple of 4 ≤ width. The remaining 0..63 Y pixels
    // and chroma samples up to `width.div_ceil(4)` (FFmpeg ceil-shift)
    // — which may include a partial 1..3-pixel final chroma group —
    // are handled by the scalar reference.
    if x < width {
      let tail_w = width - x;
      let chroma_end = width.div_ceil(4);
      let tail_u = &u_quarter[x / 4..chroma_end];
      let tail_v = &v_quarter[x / 4..chroma_end];
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA {
        scalar::yuv_411_to_rgba_row(
          &y[x..width],
          tail_u,
          tail_v,
          tail_out,
          tail_w,
          matrix,
          full_range,
        );
      } else {
        scalar::yuv_411_to_rgb_row(
          &y[x..width],
          tail_u,
          tail_v,
          tail_out,
          tail_w,
          matrix,
          full_range,
        );
      }
    }
  }
}

// ---- Planar 8-bit YUV → HSV (staged via a reused RGB chunk) ----------
//
// The SIMD twins of the scalar `yuv_*_to_hsv_row` kernels. Rather than
// re-derive an HSV-specific register pipeline, each fills a small fixed
// reused RGB scratch (one `HSV_CHUNK`-pixel chunk at a time) using the
// EXISTING `yuv_*_to_rgb_*` kernel of this backend — so the chunk filler
// IS the production RGB kernel — then runs this backend's
// `rgb_to_hsv_row` on the chunk. This keeps the per-format SIMD surface
// tiny (only the chunked driver is new) and makes the result
// byte-identical to `rgb_to_hsv_row(yuv_*_to_rgb_row(...))` within this
// tier. The scalar tail of each underlying RGB kernel handles widths
// below the SIMD block, so no separate tail is needed here.
//
// `HSV_CHUNK = 64` is a multiple of 4, so every chunk offset lands on a
// chroma-sample boundary for the 1→2 (4:2:0 / 4:2:2) and 1→4
// (4:1:0 / 4:1:1) upsampling shapes alike.

/// One reused RGB chunk's worth of pixels staged before the HSV pass.
pub(crate) const HSV_CHUNK: usize = 64;

/// Shared driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs [`rgb_to_hsv_row`] on that chunk into the H/S/V planes. The
/// result is byte-identical to `rgb_to_hsv_row(yuv_*_to_rgb_row(...))`
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
pub(crate) unsafe fn yuv_to_hsv_via_rgb_chunks(
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

/// YUV 4:2:0 planar → planar HSV bytes (OpenCV encoding), staged via
/// this backend's [`yuv_420_to_rgb_row`] + [`rgb_to_hsv_row`]. Also
/// serves 4:2:2. Byte-identical to
/// `rgb_to_hsv_row(yuv_420_to_rgb_row(...))` within this tier.
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420_to_hsv_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the chunk filler forwards the per-chunk
  // sub-slices to the 4:2:0 RGB kernel under the same contract.
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_420_to_rgb_row(
        &y[offset..],
        &u_half[offset / 2..],
        &v_half[offset / 2..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// YUV 4:4:4 planar → planar HSV bytes, staged via this backend's
/// [`yuv_444_to_rgb_row`] + [`rgb_to_hsv_row`]. Also serves 4:4:0.
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444_to_hsv_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the chunk filler forwards to the 4:4:4 RGB
  // kernel under the same contract.
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_444_to_rgb_row(
        &y[offset..],
        &u[offset..],
        &v[offset..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// YUV 4:1:0 planar → planar HSV bytes, staged via this backend's
/// [`yuv_410_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `width % 4 == 0`.
/// 3. `y.len() >= width`, `u_quarter.len() >= width / 4`,
///    `v_quarter.len() >= width / 4`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_410_to_hsv_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  debug_assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the chunk filler forwards to the 4:1:0 RGB
  // kernel under the same contract.
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_410_to_rgb_row(
        &y[offset..],
        &u_quarter[offset / 4..],
        &v_quarter[offset / 4..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// YUV 4:1:1 planar → planar HSV bytes, staged via this backend's
/// [`yuv_411_to_rgb_row`] + [`rgb_to_hsv_row`]. FFmpeg-compatible
/// arbitrary widths: `HSV_CHUNK` is a multiple of 4 so every chunk but
/// the last is chroma-aligned; the final chunk's 1..3-pixel partial
/// group is handled by the underlying RGB kernel's own partial-group
/// logic.
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `y.len() >= width`, `u_quarter.len() >= width.div_ceil(4)`,
///    `v_quarter.len() >= width.div_ceil(4)`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_411_to_hsv_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(
    u_quarter.len() >= width.div_ceil(4),
    "u_quarter row too short"
  );
  debug_assert!(
    v_quarter.len() >= width.div_ceil(4),
    "v_quarter row too short"
  );
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the chunk filler forwards to the 4:1:1 RGB
  // kernel (whose own partial-group logic covers the final chunk's
  // sub-4 tail) under the same contract.
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_411_to_rgb_row(
        &y[offset..],
        &u_quarter[offset / 4..],
        &v_quarter[offset / 4..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}
