use core::arch::x86_64::*;

use super::*;

/// AVX‑512 YUV 4:2:0 10‑bit → packed **8‑bit** RGB.
///
/// Block size 64 Y pixels / 32 chroma pairs per iteration (matching
/// the 8‑bit AVX‑512 kernel). Structural differences:
/// - Two `_mm512_loadu_si512` loads for Y (each 32 `u16` = 64 bytes);
///   one `_mm512_loadu_si512` each for U / V (32 `u16`).
/// - No u8→i16 widening — 10‑bit samples already occupy 16‑bit lanes.
/// - Chroma bias is 512 (10‑bit center).
/// - `range_params_n::<10, 8>` calibrates scales for 10→8 in one shift.
///
/// Reuses [`chroma_i16x32`], [`chroma_dup`], [`scale_y`],
/// [`narrow_u8x64`], and [`write_rgb_64`] along with the pack / dup
/// lane‑fixup indices from the 8‑bit path — post‑chroma math is
/// identical across bit depths.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::yuv_420p_n_to_rgb_row::<10>`].
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 high-bit-depth YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUVA 4:2:0 high-bit-depth → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (depth-converted via
/// `>> (BITS - 8)` to fit `u8`). Same numerical contract as
/// [`yuv_420p_n_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgba_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, true>(
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

/// Shared AVX-512 high-bit YUV 4:2:0 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_64`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_64` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_64` with the alpha
///   lane loaded from `a_src`, masked to BITS, and depth-converted
///   via `_mm512_srl_epi16` with a count of `BITS - 8`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. slices long enough + `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 5. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      // AND‑mask every load to the low 10 bits — see matching
      // comment in [`crate::row::scalar::yuv_420p_n_to_rgb_row`].
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_vec = _mm512_and_si512(
        _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );
      let v_vec = _mm512_and_si512(
        _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );

      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

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
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar.
          // `_mm512_srli_epi16::<IMM8>` requires a literal const
          // generic shift (not stable for `BITS - 8`); use
          // `_mm512_srl_epi16` with a count vector built from
          // `BITS - 8`. Reuse `narrow_u8x64` so the alpha bytes line
          // up with R/G/B in `write_rgba_64`.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x).cast()), mask_v);
          let a_hi = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x + 32).cast()), mask_v);
          let a_shr = _mm_cvtsi32_si128((BITS - 8) as i32);
          let a_lo_shifted = _mm512_srl_epi16(a_lo, a_shr);
          let a_hi_shifted = _mm512_srl_epi16(a_hi, a_shr);
          narrow_u8x64(a_lo_shifted, a_hi_shifted, pack_fixup)
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
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p_n_to_rgba_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p_n_to_rgb_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// AVX‑512 YUV 4:2:0 10‑bit → packed **10‑bit `u16`** RGB.
///
/// Block size 64 Y pixels per iteration. Mirrors
/// [`yuv420p10_to_rgb_row`]'s pre‑write math; output uses explicit
/// min/max clamp to `[0, 1023]` and 8 calls to [`write_rgb_u16_8`]
/// per block (each handles 8 pixels). A true AVX‑512 u16 interleave
/// would cut store count ~8×; left as a follow‑up optimization.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<10>`].
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 sibling of [`yuv_420p_n_to_rgba_row`] for native-depth
/// `u16` output. Alpha samples are `(1 << BITS) - 1` (opaque maximum
/// at the input bit depth).
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUVA 4:2:0 high-bit-depth → **native-depth `u16`** packed
/// RGBA with the per-pixel alpha element **sourced from `a_src`**
/// (masked to BITS, no shift) instead of being the opaque maximum
/// `(1 << BITS) - 1`.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true, true>(
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

/// Shared AVX-512 high-bit YUV 4:2:0 → native-depth `u16` kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: 8× `write_quarter` per 64-pixel block.
/// - `ALPHA = true, ALPHA_SRC = false`: 8× `write_quarter_rgba` with
///   constant alpha `(1 << BITS) - 1`.
/// - `ALPHA = true, ALPHA_SRC = true`: 8× `write_quarter_rgba` with
///   the alpha quarters extracted from `a_src` (masked to BITS).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 5. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      // AND‑mask loads to the low 10 bits so `chroma_i16x32`'s
      // `_mm512_packs_epi32` narrow stays lossless.
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_vec = _mm512_and_si512(
        _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );
      let v_vec = _mm512_and_si512(
        _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );

      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      // Eight 8‑pixel u16 writes per 64‑pixel block. For each i16x32
      // channel vector we extract four 128‑bit quarters and hand each
      // to the shared SSE4.1 u16 interleave helper.
      if ALPHA {
        // Pre-extract 8 alpha quarters (one per 8-pixel slot) when
        // `ALPHA_SRC = true`. Each quarter is a 128-bit i16x8.
        let (a0, a1, a2, a3, a4, a5, a6, a7) = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // Mask alpha loads to BITS — same hardening as Y/U/V. Native
          // bit depth, no shift; just split each 512-bit load into
          // four 128-bit quarters via `_mm512_extracti32x4_epi32`.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x).cast()), mask_v);
          let a_hi = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x + 32).cast()), mask_v);
          (
            _mm512_extracti32x4_epi32::<0>(a_lo),
            _mm512_extracti32x4_epi32::<1>(a_lo),
            _mm512_extracti32x4_epi32::<2>(a_lo),
            _mm512_extracti32x4_epi32::<3>(a_lo),
            _mm512_extracti32x4_epi32::<0>(a_hi),
            _mm512_extracti32x4_epi32::<1>(a_hi),
            _mm512_extracti32x4_epi32::<2>(a_hi),
            _mm512_extracti32x4_epi32::<3>(a_hi),
          )
        } else {
          (
            alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16,
          )
        };
        let dst = out.as_mut_ptr().add(x * 4);
        write_quarter_rgba(r_lo, g_lo, b_lo, a0, 0, dst);
        write_quarter_rgba(r_lo, g_lo, b_lo, a1, 1, dst.add(32));
        write_quarter_rgba(r_lo, g_lo, b_lo, a2, 2, dst.add(64));
        write_quarter_rgba(r_lo, g_lo, b_lo, a3, 3, dst.add(96));
        write_quarter_rgba(r_hi, g_hi, b_hi, a4, 0, dst.add(128));
        write_quarter_rgba(r_hi, g_hi, b_hi, a5, 1, dst.add(160));
        write_quarter_rgba(r_hi, g_hi, b_hi, a6, 2, dst.add(192));
        write_quarter_rgba(r_hi, g_hi, b_hi, a7, 3, dst.add(224));
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_quarter(r_lo, g_lo, b_lo, 0, dst);
        write_quarter(r_lo, g_lo, b_lo, 1, dst.add(24));
        write_quarter(r_lo, g_lo, b_lo, 2, dst.add(48));
        write_quarter(r_lo, g_lo, b_lo, 3, dst.add(72));
        write_quarter(r_hi, g_hi, b_hi, 0, dst.add(96));
        write_quarter(r_hi, g_hi, b_hi, 1, dst.add(120));
        write_quarter(r_hi, g_hi, b_hi, 2, dst.add(144));
        write_quarter(r_hi, g_hi, b_hi, 3, dst.add(168));
      }

      x += 64;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p_n_to_rgba_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}
/// AVX-512 YUV 4:4:4 planar 9/10/12/14-bit → packed **u8** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`. Block size 64 pixels.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, false, false>(
      y, u, v, rgb_out, width, matrix, full_range, None,
    );
  }
}

/// AVX-512 YUV 4:4:4 planar 9/10/12/14-bit → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p_n_to_rgb_row`].
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, false>(
      y, u, v, rgba_out, width, matrix, full_range, None,
    );
  }
}

/// AVX-512 YUVA 4:4:4 planar 9/10/12/14-bit → packed **8-bit RGBA**
/// with the per-pixel alpha byte **sourced from `a_src`**
/// (depth-converted via `>> (BITS - 8)`) instead of being constant
/// `0xFF`. Same numerical contract as [`yuv_444p_n_to_rgba_row`] for
/// R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgba_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, true>(
      y,
      u,
      v,
      rgba_out,
      width,
      matrix,
      full_range,
      Some(a_src),
    );
  }
}

/// Shared AVX-512 high-bit-depth YUV 4:4:4 kernel for
/// [`yuv_444p_n_to_rgb_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `write_rgb_64`), [`yuv_444p_n_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `write_rgba_64` with constant `0xFF` alpha) and
/// [`yuv_444p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `write_rgba_64` with the alpha lane loaded and
/// depth-converted from `a_src`).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` must be one of `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  a_src: Option<&[u16]>,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      // 64 Y + 64 U + 64 V per iter. Full-width chroma (two 512-bit
      // loads each) — no horizontal duplication, 4:4:4 is 1:1.
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_lo_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x + 32).cast()), mask_v);
      let v_lo_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x + 32).cast()), mask_v);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

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

      // Two chroma_i16x32 per channel → 64 chroma values. No dup.
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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x).cast()), mask_v);
          let a_hi = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x + 32).cast()), mask_v);
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar. AVX-512's
          // `_mm512_srli_epi16::<IMM8>` requires a literal shift, so
          // use `_mm512_srl_epi16` with a count vector built from
          // `BITS - 8`. `narrow_u8x64` applies the per-lane permute
          // fixup `pack_fixup` that R/G/B already pay for.
          let a_shr = _mm_cvtsi32_si128((BITS - 8) as i32);
          let a_lo_shifted = _mm512_srl_epi16(a_lo, a_shr);
          let a_hi_shifted = _mm512_srl_epi16(a_hi, a_shr);
          narrow_u8x64(a_lo_shifted, a_hi_shifted, pack_fixup)
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
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p_n_to_rgba_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p_n_to_rgb_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// AVX-512 YUV 4:4:4 planar 9/10/12/14-bit → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`. 64 pixels per iter.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, false>(
      y, u, v, rgb_out, width, matrix, full_range, None,
    );
  }
}

/// AVX-512 sibling of [`yuv_444p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1`.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, false>(
      y, u, v, rgba_out, width, matrix, full_range, None,
    );
  }
}

/// AVX-512 YUVA 4:4:4 planar 9/10/12/14-bit → **native-depth `u16`**
/// packed RGBA with the per-pixel alpha element **sourced from
/// `a_src`** (already at the source's native bit depth — no depth
/// conversion) instead of being the opaque maximum `(1 << BITS) - 1`.
/// Same numerical contract as [`yuv_444p_n_to_rgba_u16_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, true>(
      y,
      u,
      v,
      rgba_out,
      width,
      matrix,
      full_range,
      Some(a_src),
    );
  }
}

/// Shared AVX-512 high-bit YUV 4:4:4 → native-depth `u16` kernel for
/// [`yuv_444p_n_to_rgb_u16_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// 8× `write_quarter`), [`yuv_444p_n_to_rgba_u16_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, 8× `write_quarter_rgba` with constant alpha
/// `(1 << BITS) - 1`) and
/// [`yuv_444p_n_to_rgba_u16_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 8× `write_quarter_rgba` with the alpha quarters
/// loaded from `a_src` and masked to native bit depth — no shift since
/// both the source alpha and the u16 output element are at the same
/// native bit depth).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  a_src: Option<&[u16]>,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_lo_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x + 32).cast()), mask_v);
      let v_lo_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x + 32).cast()), mask_v);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        // Per-quarter alpha vectors, one for each of the 8
        // `write_quarter_rgba` calls below. With ALPHA_SRC = false they
        // all collapse to the constant `alpha_u16`. With ALPHA_SRC =
        // true we load 64 source-alpha u16 values (two `__m512i`
        // halves), AND-mask any over-range bits to native depth (no
        // shift — both source and output are at the same bit depth)
        // and split each half into the four 128-bit quarters consumed
        // by `write_quarter_rgba`.
        let (a_lo_q0, a_lo_q1, a_lo_q2, a_lo_q3, a_hi_q0, a_hi_q1, a_hi_q2, a_hi_q3) = if ALPHA_SRC
        {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_v = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x).cast()), mask_v);
          let a_hi_v = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x + 32).cast()), mask_v);
          (
            _mm512_extracti32x4_epi32::<0>(a_lo_v),
            _mm512_extracti32x4_epi32::<1>(a_lo_v),
            _mm512_extracti32x4_epi32::<2>(a_lo_v),
            _mm512_extracti32x4_epi32::<3>(a_lo_v),
            _mm512_extracti32x4_epi32::<0>(a_hi_v),
            _mm512_extracti32x4_epi32::<1>(a_hi_v),
            _mm512_extracti32x4_epi32::<2>(a_hi_v),
            _mm512_extracti32x4_epi32::<3>(a_hi_v),
          )
        } else {
          (
            alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16,
          )
        };
        let dst = out.as_mut_ptr().add(x * 4);
        write_quarter_rgba(r_lo, g_lo, b_lo, a_lo_q0, 0, dst);
        write_quarter_rgba(r_lo, g_lo, b_lo, a_lo_q1, 1, dst.add(32));
        write_quarter_rgba(r_lo, g_lo, b_lo, a_lo_q2, 2, dst.add(64));
        write_quarter_rgba(r_lo, g_lo, b_lo, a_lo_q3, 3, dst.add(96));
        write_quarter_rgba(r_hi, g_hi, b_hi, a_hi_q0, 0, dst.add(128));
        write_quarter_rgba(r_hi, g_hi, b_hi, a_hi_q1, 1, dst.add(160));
        write_quarter_rgba(r_hi, g_hi, b_hi, a_hi_q2, 2, dst.add(192));
        write_quarter_rgba(r_hi, g_hi, b_hi, a_hi_q3, 3, dst.add(224));
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_quarter(r_lo, g_lo, b_lo, 0, dst);
        write_quarter(r_lo, g_lo, b_lo, 1, dst.add(24));
        write_quarter(r_lo, g_lo, b_lo, 2, dst.add(48));
        write_quarter(r_lo, g_lo, b_lo, 3, dst.add(72));
        write_quarter(r_hi, g_hi, b_hi, 0, dst.add(96));
        write_quarter(r_hi, g_hi, b_hi, 1, dst.add(120));
        write_quarter(r_hi, g_hi, b_hi, 2, dst.add(144));
        write_quarter(r_hi, g_hi, b_hi, 3, dst.add(168));
      }

      x += 64;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p_n_to_rgba_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}
