use core::arch::x86_64::*;

use super::*;

/// AVX-512 YUV 4:4:4 planar **16-bit** → packed **u8** RGB. Stays on
/// the i32 Q15 pipeline — output-range scaling keeps `coeff × u_d`
/// within i32 for u8 output. 64 pixels per iter.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p16_to_rgb_row(
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
    yuv_444p16_to_rgb_or_rgba_row::<false, false>(
      y, u, v, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUV 4:4:4 planar **16-bit** → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p16_to_rgba_row(
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
    yuv_444p16_to_rgb_or_rgba_row::<true, false>(
      y, u, v, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUVA 4:4:4 16-bit → packed **8-bit RGBA** with source
/// alpha. Same R/G/B numerical contract as [`yuv_444p16_to_rgba_row`];
/// the per-pixel alpha byte is **sourced from `a_src`** (depth-converted
/// via `_mm512_srli_epi16::<8>` to fit `u8`).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgba_with_alpha_src_row(
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
    yuv_444p16_to_rgb_or_rgba_row::<true, true>(
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

/// Shared AVX-512 16-bit YUV 4:4:4 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_64`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_64` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_64` with the alpha
///   lane loaded from `a_src` and depth-converted via
///   `_mm512_srli_epi16::<8>`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
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
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let u_lo_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let u_hi_vec = _mm512_loadu_si512(u.as_ptr().add(x + 32).cast());
      let v_lo_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());
      let v_hi_vec = _mm512_loadu_si512(v.as_ptr().add(x + 32).cast());

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — `>> 8` to fit u8.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm512_srli_epi16::<8>(_mm512_loadu_si512(a_ptr.add(x).cast()));
          let a_hi = _mm512_srli_epi16::<8>(_mm512_loadu_si512(a_ptr.add(x + 32).cast()));
          narrow_u8x64(a_lo, a_hi, pack_fixup)
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
        scalar::yuv_444p16_to_rgba_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p16_to_rgba_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX-512 YUV 4:4:4 planar **16-bit** → packed **u16** RGB. Native
/// 512-bit i64-chroma kernel using `_mm512_srai_epi64`.
///
/// Block size 32 pixels per iter. Mirrors
/// [`yuv_420p16_to_rgb_u16_row`] but with full-width chroma loads
/// and no duplication step (4:4:4 is 1:1 with Y).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p16_to_rgb_u16_row(
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
    yuv_444p16_to_rgb_or_rgba_u16_row::<false, false>(
      y, u, v, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 sibling of [`yuv_444p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF`.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p16_to_rgba_u16_row(
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
    yuv_444p16_to_rgb_or_rgba_u16_row::<true, false>(
      y, u, v, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUVA 4:4:4 16-bit → packed **native-depth `u16`** RGBA with
/// source alpha. Same R/G/B numerical contract as
/// [`yuv_444p16_to_rgba_u16_row`]; the per-pixel alpha element is
/// **sourced from `a_src`** at native depth (no shift needed).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgba_u16_with_alpha_src_row(
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
    yuv_444p16_to_rgb_or_rgba_u16_row::<true, true>(
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

/// Shared AVX-512 16-bit YUV 4:4:4 → native-depth `u16` kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: writes RGB triples via
///   `write_rgb_u16_32`.
/// - `ALPHA = true, ALPHA_SRC = false`: writes RGBA quads via
///   `write_rgba_u16_32` with constant alpha `0xFFFF`.
/// - `ALPHA = true, ALPHA_SRC = true`: 4× `write_rgba_u16_8` with the
///   alpha element loaded from `a_src` (the standard `write_rgba_u16_32`
///   broadcasts a single 128-bit alpha lane, which doesn't fit the
///   per-pixel-source-alpha case).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 pixels/iter. 4:4:4: full-width chroma (32 samples each).
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let v_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

      // Widen each i16x32 → two i32x16 halves.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      // Scale in i32 — `u_centered * c_scale` fits i32 after >> 15.
      let u_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let u_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_i32_v,
      ));

      // i64 chroma: 4 calls per channel (even/odd of u_d_lo and u_d_hi).
      let u_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(u_d_lo);
      let u_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(v_d_lo);
      let v_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_e = chroma_i64x8_avx512(cru, crv, u_d_lo, v_d_lo, rnd_i64_v);
      let r_ch_lo_o = chroma_i64x8_avx512(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let r_ch_hi_e = chroma_i64x8_avx512(cru, crv, u_d_hi, v_d_hi, rnd_i64_v);
      let r_ch_hi_o = chroma_i64x8_avx512(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let g_ch_lo_e = chroma_i64x8_avx512(cgu, cgv, u_d_lo, v_d_lo, rnd_i64_v);
      let g_ch_lo_o = chroma_i64x8_avx512(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let g_ch_hi_e = chroma_i64x8_avx512(cgu, cgv, u_d_hi, v_d_hi, rnd_i64_v);
      let g_ch_hi_o = chroma_i64x8_avx512(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let b_ch_lo_e = chroma_i64x8_avx512(cbu, cbv, u_d_lo, v_d_lo, rnd_i64_v);
      let b_ch_lo_o = chroma_i64x8_avx512(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let b_ch_hi_e = chroma_i64x8_avx512(cbu, cbv, u_d_hi, v_d_hi, rnd_i64_v);
      let b_ch_hi_o = chroma_i64x8_avx512(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);

      // Reassemble to i32x16 per half.
      let r_ch_lo = reassemble_i32x16(r_ch_lo_e, r_ch_lo_o, interleave_idx);
      let r_ch_hi = reassemble_i32x16(r_ch_hi_e, r_ch_hi_o, interleave_idx);
      let g_ch_lo = reassemble_i32x16(g_ch_lo_e, g_ch_lo_o, interleave_idx);
      let g_ch_hi = reassemble_i32x16(g_ch_hi_e, g_ch_hi_o, interleave_idx);
      let b_ch_lo = reassemble_i32x16(b_ch_lo_e, b_ch_lo_o, interleave_idx);
      let b_ch_hi = reassemble_i32x16(b_ch_hi_e, b_ch_hi_o, interleave_idx);

      // Y path: widen 32 u16 → two i32x16, subtract y_off, scale in i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Add Y + chroma (no dup — 4:4:4 is 1:1), saturate to u16.
      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_ch_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_ch_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_ch_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_ch_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_ch_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_ch_hi);

      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      if ALPHA {
        if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — load 32 lanes (one
          // __m512i = 64 bytes), split into four 128-bit quarters
          // and inline the 4× write_rgba_u16_8 calls (the standard
          // `write_rgba_u16_32` helper broadcasts a single alpha
          // 128-bit lane to all 4 quarters, which doesn't fit the
          // per-pixel-source-alpha case).
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_vec = _mm512_loadu_si512(a_ptr.add(x).cast());
          let a0 = _mm512_extracti32x4_epi32::<0>(a_vec);
          let a1 = _mm512_extracti32x4_epi32::<1>(a_vec);
          let a2 = _mm512_extracti32x4_epi32::<2>(a_vec);
          let a3 = _mm512_extracti32x4_epi32::<3>(a_vec);
          let dst = out.as_mut_ptr().add(x * 4);
          write_rgba_u16_8(
            _mm512_castsi512_si128(r_u16),
            _mm512_castsi512_si128(g_u16),
            _mm512_castsi512_si128(b_u16),
            a0,
            dst,
          );
          write_rgba_u16_8(
            _mm512_extracti32x4_epi32::<1>(r_u16),
            _mm512_extracti32x4_epi32::<1>(g_u16),
            _mm512_extracti32x4_epi32::<1>(b_u16),
            a1,
            dst.add(32),
          );
          write_rgba_u16_8(
            _mm512_extracti32x4_epi32::<2>(r_u16),
            _mm512_extracti32x4_epi32::<2>(g_u16),
            _mm512_extracti32x4_epi32::<2>(b_u16),
            a2,
            dst.add(64),
          );
          write_rgba_u16_8(
            _mm512_extracti32x4_epi32::<3>(r_u16),
            _mm512_extracti32x4_epi32::<3>(g_u16),
            _mm512_extracti32x4_epi32::<3>(b_u16),
            a3,
            dst.add(96),
          );
        } else {
          write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
        }
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
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
        scalar::yuv_444p16_to_rgba_u16_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p16_to_rgba_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p16_to_rgb_row(
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
    yuv_420p16_to_rgb_or_rgba_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 16-bit YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p16_to_rgba_row(
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
    yuv_420p16_to_rgb_or_rgba_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 16-bit YUVA 4:2:0 → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (depth-converted via
/// `>> 8` to fit `u8`). 16-bit alpha is full-range u16 — no AND-mask
/// step. Same numerical contract as [`yuv_420p16_to_rgba_row`] for
/// R/G/B.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgba_with_alpha_src_row(
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
    yuv_420p16_to_rgb_or_rgba_row::<true, true>(
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

/// Shared AVX-512 16-bit YUV 4:2:0 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_64`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_64` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_64` with the alpha
///   lane loaded from `a_src` and depth-converted via
///   `_mm512_srli_epi16::<8>` (literal const shift).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
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
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let u_vec = _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast());

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — `>> 8` to fit u8.
          // `_mm512_srli_epi16::<8>` accepts a const literal shift.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm512_srli_epi16::<8>(_mm512_loadu_si512(a_ptr.add(x).cast()));
          let a_hi = _mm512_srli_epi16::<8>(_mm512_loadu_si512(a_ptr.add(x + 32).cast()));
          narrow_u8x64(a_lo, a_hi, pack_fixup)
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
        scalar::yuv_420p16_to_rgba_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p16_to_rgba_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX-512 YUV 4:2:0 16-bit → packed **16-bit** RGB. Native 512-bit
/// implementation, 32 pixels per iteration.
///
/// Uses AVX-512's native `_mm512_srai_epi64` — unlike the SSE4.1 /
/// AVX2 u16 paths which need the `srai64_15` bias trick. Processes
/// chroma and Y in i64 lanes via `_mm512_mul_epi32` (even i32 lanes
/// → i64x8 products), handling even- and odd-indexed lanes
/// separately and reassembling via `_mm512_permutex2var_epi32`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p16_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 sibling of [`yuv_420p16_to_rgba_row`] for native-depth
/// `u16` output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p16_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 16-bit YUVA 4:2:0 → **native-depth `u16`** packed RGBA
/// with the per-pixel alpha element **sourced from `a_src`**
/// (full-range u16, no mask, no shift) instead of being constant
/// `0xFFFF`.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgba_u16_with_alpha_src_row(
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
    yuv_420p16_to_rgb_or_rgba_u16_row::<true, true>(
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

/// Shared AVX-512 16-bit YUV 4:2:0 → native-depth `u16` kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_u16_32`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_u16_32` with
///   constant alpha `0xFFFF` (broadcast 128-bit lane).
/// - `ALPHA = true, ALPHA_SRC = true`: 4× `write_rgba_u16_8` with the
///   alpha quarters loaded from `a_src` (full-range u16, no shift).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; pointer
  // adds below are bounded by `while x + 32 <= width` and the caller-
  // promised slice lengths.
  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    // UV centering: subtract 32768 via wrapping i16 add of -32768.
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    // Permute indices, built once per call.
    let dup_lo_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let dup_hi_idx = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    // Interleave two i32x8 (each in the low 256 bits of a __m512i) into
    // an i32x16: [e0, o0, e1, o1, ..., e7, o7].
    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    // `_mm512_packus_epi32` per-128-bit-lane fixup (same as the 8-bit
    // AVX-512 kernels).
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 Y pixels / 16 chroma pairs per iter. The Y load is a
      // full 512-bit read (32 × u16); UV only needs 16 × u16 per
      // plane, so a 256-bit load is sufficient — a 512-bit load
      // would read past the end of `u_half` / `v_half` on the last
      // iteration where `x / 2 + 16 == u_half.len()`.
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec = _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast());

      // Center UV by subtracting 32768 (wrapping i16 sub). Using
      // `_mm256_sub_epi16` with bias16 (which carries -32768 as i16):
      // `sample - (-32768) == sample + 32768` mod 2^16, giving the
      // centered signed value.
      let u_i16 = _mm256_sub_epi16(u_vec, _mm512_castsi512_si256(bias16_v));
      let v_i16 = _mm256_sub_epi16(v_vec, _mm512_castsi512_si256(bias16_v));

      // Widen i16x16 → i32x16 (each value sign-extended).
      let u_i32 = _mm512_cvtepi16_epi32(u_i16);
      let v_i32 = _mm512_cvtepi16_epi32(v_i16);

      // Scale UV in i32: |u_centered * c_scale| peaks at
      // 32768 * ~38300 ≈ 1.26·10⁹ — fits i32. Result `u_d` range is
      // ~[-37450, 37449].
      let u_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_i32_v,
      ));

      // Chroma in i64 via `_mm512_mul_epi32` on even lanes, then
      // shuffle odd lanes to even and repeat. Each call produces 8
      // i64 products from the 8 even-indexed i32 lanes.
      let u_d_odd = _mm512_shuffle_epi32::<0xF5>(u_d); // lanes [1,1,3,3,5,5,7,7] per 128-bit
      let v_d_odd = _mm512_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x8_avx512(cru, crv, u_d, v_d, rnd_i64_v);
      let r_ch_odd = chroma_i64x8_avx512(cru, crv, u_d_odd, v_d_odd, rnd_i64_v);
      let g_ch_even = chroma_i64x8_avx512(cgu, cgv, u_d, v_d, rnd_i64_v);
      let g_ch_odd = chroma_i64x8_avx512(cgu, cgv, u_d_odd, v_d_odd, rnd_i64_v);
      let b_ch_even = chroma_i64x8_avx512(cbu, cbv, u_d, v_d, rnd_i64_v);
      let b_ch_odd = chroma_i64x8_avx512(cbu, cbv, u_d_odd, v_d_odd, rnd_i64_v);

      // Reassemble i64x8 pairs to i32x16.
      let r_ch_i32 = reassemble_i32x16(r_ch_even, r_ch_odd, interleave_idx);
      let g_ch_i32 = reassemble_i32x16(g_ch_even, g_ch_odd, interleave_idx);
      let b_ch_i32 = reassemble_i32x16(b_ch_even, b_ch_odd, interleave_idx);

      // Duplicate chroma for 2 Y pixels each: 16 chroma → 32 values.
      let r_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, r_ch_i32);
      let r_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, r_ch_i32);
      let g_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, g_ch_i32);
      let g_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, g_ch_i32);
      let b_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, b_ch_i32);
      let b_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, b_ch_i32);

      // Y path: split 32 u16 into two halves, widen to i32x16 each,
      // subtract y_off, scale in i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Add Y + chroma, pack with unsigned saturation to u16.
      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_dup_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_dup_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_dup_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_dup_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_dup_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_dup_hi);

      // `_mm512_packus_epi32` signed i32 → unsigned u16 with
      // saturation. Produces u16x32 with per-128-bit-lane split order;
      // fix via `pack_fixup` permute (same as NV12 AVX-512 u8 kernel).
      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      // Write 32 pixels via the appropriate 4× 8-pixel helper.
      if ALPHA {
        if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — load 32 lanes (one
          // __m512i = 64 bytes), split into four 128-bit quarters
          // and inline the 4× write_rgba_u16_8 calls (the standard
          // `write_rgba_u16_32` helper broadcasts a single alpha
          // 128-bit lane to all 4 quarters, which doesn't fit the
          // per-pixel-source-alpha case).
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_vec = _mm512_loadu_si512(a_ptr.add(x).cast());
          let a0 = _mm512_extracti32x4_epi32::<0>(a_vec);
          let a1 = _mm512_extracti32x4_epi32::<1>(a_vec);
          let a2 = _mm512_extracti32x4_epi32::<2>(a_vec);
          let a3 = _mm512_extracti32x4_epi32::<3>(a_vec);
          let dst = out.as_mut_ptr().add(x * 4);
          write_rgba_u16_8(
            _mm512_castsi512_si128(r_u16),
            _mm512_castsi512_si128(g_u16),
            _mm512_castsi512_si128(b_u16),
            a0,
            dst,
          );
          write_rgba_u16_8(
            _mm512_extracti32x4_epi32::<1>(r_u16),
            _mm512_extracti32x4_epi32::<1>(g_u16),
            _mm512_extracti32x4_epi32::<1>(b_u16),
            a1,
            dst.add(32),
          );
          write_rgba_u16_8(
            _mm512_extracti32x4_epi32::<2>(r_u16),
            _mm512_extracti32x4_epi32::<2>(g_u16),
            _mm512_extracti32x4_epi32::<2>(b_u16),
            a2,
            dst.add(64),
          );
          write_rgba_u16_8(
            _mm512_extracti32x4_epi32::<3>(r_u16),
            _mm512_extracti32x4_epi32::<3>(g_u16),
            _mm512_extracti32x4_epi32::<3>(b_u16),
            a3,
            dst.add(96),
          );
        } else {
          write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
        }
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
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
        scalar::yuv_420p16_to_rgba_u16_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p16_to_rgba_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}
