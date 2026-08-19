use super::*;

/// Compile-time host endianness. `true` on BE targets, `false` on LE
/// targets (always `false` on `x86_64` / `i686` in practice).
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// De-packs a host-native `__m256i` of wire u16 samples to the `BITS`-bit
/// active value: a `_mm256_srl_epi16` right shift by `16 - BITS` (count in
/// the low 64b of `shr_count`) for the high-bit-packed P-formats
/// (`LOW_PACKED = false`), or a low-mask `_mm256_and_si256(v, low_mask)`
/// for low-bit-packed NV20 (`LOW_PACKED = true`). The AVX2 twin of the
/// scalar `depack_pn`. `LOW_PACKED` is const, so the unused branch is
/// eliminated; byte-identical to the scalar de-pack per lane.
///
/// # Safety
///
/// AVX2 must be available on the current CPU.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn depack_u16x16<const LOW_PACKED: bool>(
  v: __m256i,
  shr_count: __m128i,
  low_mask: __m256i,
) -> __m256i {
  if LOW_PACKED {
    _mm256_and_si256(v, low_mask)
  } else {
    _mm256_srl_epi16(v, shr_count)
  }
}

/// Byte-swap every u16 lane of `v` in-register when the source (wire)
/// endian differs from the host's native u16 byte order.
///
/// Used after `deinterleave_uv_u16_avx2` to apply per-lane byte-swapping.
/// Gated on `BE != HOST_NATIVE_BE` so a hypothetical BE-x86 host would
/// not double-swap. When the gate folds to `false` at compile time, the
/// call compiles away entirely.
#[inline(always)]
unsafe fn byteswap_u16x16<const BE: bool>(v: __m256i) -> __m256i {
  if BE != HOST_NATIVE_BE {
    let mask = unsafe {
      core::mem::transmute::<[u8; 32], __m256i>([
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14,
      ])
    };
    unsafe { _mm256_shuffle_epi8(v, mask) }
  } else {
    v
  }
}

/// AVX2 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **8‑bit** RGB.
///
/// Block size 32 Y pixels / 16 chroma pairs per iteration. Mirrors
/// [`super::x86_avx2::yuv_420p_n_to_rgb_row`] with two structural
/// differences:
/// - Samples are shifted right by `16 - BITS` (`_mm256_srl_epi16`,
///   with a shift count computed from `BITS` once per call) instead
///   of AND‑masked.
/// - Semi‑planar UV is deinterleaved via [`deinterleave_uv_u16_avx2`]
///   (two `_mm256_shuffle_epi8` + two `_mm256_permute4x64_epi64` +
///   two `_mm256_permute2x128_si256` per 32 chroma elements).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, false, BE, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 high-bit-packed semi-planar 4:2:0 → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, true, BE, false>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX2 NV20 (low-bit-packed 4:2:2, 10-bit) → packed **8-bit** RGB.
/// Low-bit twin of [`p_n_to_rgb_row`] (`LOW_PACKED = true`, `BITS = 10`).
/// Byte-identical to [`scalar::nv20_to_rgb_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgb_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nv20_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<10, false, BE, true>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 NV20 → packed **8-bit RGBA** (`R, G, B, 0xFF`). Low-bit twin of
/// [`p_n_to_rgba_row`]. Byte-identical to [`scalar::nv20_to_rgba_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgba_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nv20_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<10, true, BE, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 P010/P012 kernel. `ALPHA = false` uses `write_rgb_32`;
/// `ALPHA = true` uses `write_rgba_32` with constant `0xFF` alpha.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    // High-bit-packed: `_mm256_srl_epi16` by `16 - BITS`; low-bit-packed
    // NV20: the `low_mask`. One path is dead per monomorphization.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let low_mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 Y = two u16x16 loads, de-packed (shift or mask). BE input is
      // byte-swapped via `load_endian_u16x16::<BE>` for Y, and via
      // `byteswap_u16x16::<BE>` after deinterleave for UV.
      let y_low_i16 = depack_u16x16::<LOW_PACKED>(
        endian::load_endian_u16x16::<BE>(y.as_ptr().add(x) as *const u8),
        shr_count,
        low_mask,
      );
      let y_high_i16 = depack_u16x16::<LOW_PACKED>(
        endian::load_endian_u16x16::<BE>(y.as_ptr().add(x + 16) as *const u8),
        shr_count,
        low_mask,
      );

      // 32 UV (16 pairs) — deinterleave + byte-swap (for BE) + de-pack.
      let (u_vec, v_vec) = deinterleave_uv_u16_avx2(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x16::<BE>(u_vec);
      let v_vec = byteswap_u16x16::<BE>(v_vec);
      let u_vec = depack_u16x16::<LOW_PACKED>(u_vec, shr_count, low_mask);
      let v_vec = depack_u16x16::<LOW_PACKED>(v_vec, shr_count, low_mask);

      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

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
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      // Route the scalar tail through the same `LOW_PACKED` de-pack.
      scalar::p_n_to_rgb_or_rgba_row::<BITS, ALPHA, BE, LOW_PACKED>(
        tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
      );
    }
  }
}

/// AVX2 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
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
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 sibling of [`p_n_to_rgba_row`] for native-depth `u16` output.
/// Alpha samples are `(1 << BITS) - 1` (opaque maximum at the input
/// bit depth). P016 has its own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE, false>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX2 NV20 → packed **native-depth `u16`** RGB (low-bit-packed,
/// `[0, 1023]`). Low-bit twin of [`p_n_to_rgb_u16_row`]. Byte-identical
/// to [`scalar::nv20_to_rgb_u16_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nv20_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<10, false, BE, true>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 NV20 → packed **native-depth `u16` RGBA** (low-bit-packed;
/// alpha `1023`). Low-bit twin of [`p_n_to_rgba_u16_row`].
/// Byte-identical to [`scalar::nv20_to_rgba_u16_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgba_u16_row`].
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nv20_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<10, true, BE, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX2 Pn → native-depth `u16` kernel. `ALPHA = false` writes
/// RGB triples via 4x `write_rgb_u16_8` per 32-pixel block;
/// `ALPHA = true` writes RGBA quads via 4x `write_rgba_u16_8` with
/// constant alpha `(1 << BITS) - 1`. P016 has its own kernel family —
/// never routed here.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let max_v = _mm256_set1_epi16(out_max);
    let zero_v = _mm256_set1_epi16(0);
    // High-bit-packed: `_mm256_srl_epi16` by `16 - BITS`; low-bit-packed
    // NV20: the `low_mask`. One path is dead per monomorphization.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let low_mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let mut x = 0usize;
    while x + 32 <= width {
      // BE input is byte-swapped via `load_endian_u16x16::<BE>` for Y,
      // and via `byteswap_u16x16::<BE>` after deinterleave for UV.
      let y_low_i16 = depack_u16x16::<LOW_PACKED>(
        endian::load_endian_u16x16::<BE>(y.as_ptr().add(x) as *const u8),
        shr_count,
        low_mask,
      );
      let y_high_i16 = depack_u16x16::<LOW_PACKED>(
        endian::load_endian_u16x16::<BE>(y.as_ptr().add(x + 16) as *const u8),
        shr_count,
        low_mask,
      );
      let (u_vec, v_vec) = deinterleave_uv_u16_avx2(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x16::<BE>(u_vec);
      let v_vec = byteswap_u16x16::<BE>(v_vec);
      let u_vec = depack_u16x16::<LOW_PACKED>(u_vec, shr_count, low_mask);
      let v_vec = depack_u16x16::<LOW_PACKED>(v_vec, shr_count, low_mask);

      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          alpha_u16,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          alpha_u16,
          dst.add(32),
        );
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          alpha_u16,
          dst.add(64),
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          alpha_u16,
          dst.add(96),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          dst.add(24),
        );
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          dst.add(48),
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          dst.add(72),
        );
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      // Route the scalar tail through the same `LOW_PACKED` de-pack.
      scalar::p_n_to_rgb_or_rgba_u16_row::<BITS, ALPHA, BE, LOW_PACKED>(
        tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
      );
    }
  }
}
/// AVX2 P016 → packed **8-bit** RGB. 32 pixels per iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<false, BE>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 P016 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<true, BE>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 P016 kernel. `ALPHA = false` uses `write_rgb_32`;
/// `ALPHA = true` uses `write_rgba_32` with constant `0xFF` alpha.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      // BE input is byte-swapped via `load_endian_u16x16::<BE>` for Y,
      // and via `byteswap_u16x16::<BE>` after deinterleave for UV.
      let y_low = endian::load_endian_u16x16::<BE>(y.as_ptr().add(x) as *const u8);
      let y_high = endian::load_endian_u16x16::<BE>(y.as_ptr().add(x + 16) as *const u8);
      // Deinterleave 32 UV pairs (64 u16) from uv_half[x..x+32].
      // Uses the shared AVX2 deinterleave helper for Pn formats.
      let (u_vec, v_vec) = deinterleave_uv_u16_avx2(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x16::<BE>(u_vec);
      let v_vec = byteswap_u16x16::<BE>(v_vec);

      let u_i16 = _mm256_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16_avx2(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_avx2(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = narrow_u8x32(r_lo, r_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let b_u8 = narrow_u8x32(b_lo, b_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX2 P016 → packed **16-bit** RGB.
/// Delegates to SSE4.1 (i64 arithmetic; no AVX2 srai_epi64).
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgb_out` is `&mut [u16]`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<false, BE>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 sibling of [`p16_to_rgba_row`] for native-depth `u16` output.
/// Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<true, BE>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples; `ALPHA = true` writes RGBA
/// quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_v = _mm256_set1_epi64x(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    // 16 pixels/iter needs 8 UV pairs = 16 u16 = 32 bytes of UV data.
    // Load as two __m128i halves so we can reuse the SSE4.1 128-bit
    // byte-shuffle mask. Each half carries 4 UV pairs; we deinterleave
    // each to [U's | V's] and then join the two U halves / two V halves.
    // When wire endian differs from host (`BE != HOST_NATIVE_BE`):
    // fold per-u16-lane byte-swap into the deinterleave mask. On a
    // hypothetical BE-x86 host this avoids the double-swap that a plain
    // `BE` gate would introduce.
    let split_mask_128 = if BE != HOST_NATIVE_BE {
      _mm_setr_epi8(1, 0, 5, 4, 9, 8, 13, 12, 3, 2, 7, 6, 11, 10, 15, 14)
    } else {
      _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15)
    };

    let mut x = 0usize;
    while x + 16 <= width {
      // BE input is byte-swapped via `load_endian_u16x16::<BE>` for Y.
      let y_vec = endian::load_endian_u16x16::<BE>(y.as_ptr().add(x) as *const u8);
      // Two 128-bit UV loads: bytes [0..16) and [16..32). `x + 8` is
      // in u16 units (8 u16 = 16 bytes) — the second load starts at
      // byte offset 16, which is UV pair index 4.
      let uv_lo_raw = _mm_loadu_si128(uv_half.as_ptr().add(x).cast());
      let uv_hi_raw = _mm_loadu_si128(uv_half.as_ptr().add(x + 8).cast());
      // Deinterleave each half: [U0,V0,U1,V1,U2,V2,U3,V3] →
      // [U0,U1,U2,U3, V0,V1,V2,V3] (low 64b = U's, high 64b = V's).
      // The shuffle mask above also swaps bytes within each u16 lane
      // when `BE = true`.
      let uv_lo_split = _mm_shuffle_epi8(uv_lo_raw, split_mask_128);
      let uv_hi_split = _mm_shuffle_epi8(uv_hi_raw, split_mask_128);
      // Combine: low 64 of each → 8 U samples; high 64 of each → 8 V.
      let u_vec_128 = _mm_unpacklo_epi64(uv_lo_split, uv_hi_split);
      let v_vec_128 = _mm_unpackhi_epi64(uv_lo_split, uv_hi_split);

      let bias16_128 = _mm256_castsi256_si128(bias16_v);
      let u_i16 = _mm_sub_epi16(u_vec_128, bias16_128);
      let v_i16 = _mm_sub_epi16(v_vec_128, bias16_128);

      let u_i32 = _mm256_cvtepi16_epi32(u_i16);
      let v_i32 = _mm256_cvtepi16_epi32(v_i16);

      let rnd32_v = _mm256_set1_epi32(1 << 14);
      let u_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_i32, c_scale_v),
        rnd32_v,
      ));

      let u_d_odd = _mm256_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm256_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x4_avx2(cru, crv, u_d, v_d, rnd_v);
      let r_ch_odd = chroma_i64x4_avx2(cru, crv, u_d_odd, v_d_odd, rnd_v);
      let g_ch_even = chroma_i64x4_avx2(cgu, cgv, u_d, v_d, rnd_v);
      let g_ch_odd = chroma_i64x4_avx2(cgu, cgv, u_d_odd, v_d_odd, rnd_v);
      let b_ch_even = chroma_i64x4_avx2(cbu, cbv, u_d, v_d, rnd_v);
      let b_ch_odd = chroma_i64x4_avx2(cbu, cbv, u_d_odd, v_d_odd, rnd_v);

      let r_ch_i32 = reassemble_i64x4_to_i32x8(r_ch_even, r_ch_odd);
      let g_ch_i32 = reassemble_i64x4_to_i32x8(g_ch_even, g_ch_odd);
      let b_ch_i32 = reassemble_i64x4_to_i32x8(b_ch_even, b_ch_odd);

      let (r_dup_lo, r_dup_hi) = chroma_dup_i32(r_ch_i32);
      let (g_dup_lo, g_dup_hi) = chroma_dup_i32(g_ch_i32);
      let (b_dup_lo, b_dup_hi) = chroma_dup_i32(b_ch_i32);

      let y_lo_u16 = _mm256_castsi256_si128(y_vec);
      let y_hi_u16 = _mm256_extracti128_si256::<1>(y_vec);
      let y_lo_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x8_i64(y_lo_i32, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_i32x8_i64(y_hi_i32, y_scale_v, rnd_v);

      let r_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, r_dup_lo),
        _mm256_add_epi32(y_hi_scaled, r_dup_hi),
      ));
      let g_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, g_dup_lo),
        _mm256_add_epi32(y_hi_scaled, g_dup_hi),
      ));
      let b_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, b_dup_lo),
        _mm256_add_epi32(y_hi_scaled, b_dup_hi),
      ));

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_u16),
          _mm256_castsi256_si128(g_u16),
          _mm256_castsi256_si128(b_u16),
          alpha_u16,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_u16),
          _mm256_extracti128_si256::<1>(g_u16),
          _mm256_extracti128_si256::<1>(b_u16),
          alpha_u16,
          dst.add(32),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_u16),
          _mm256_castsi256_si128(g_u16),
          _mm256_castsi256_si128(b_u16),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_u16),
          _mm256_extracti128_si256::<1>(g_u16),
          _mm256_extracti128_si256::<1>(b_u16),
          dst.add(24),
        );
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_u16_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_u16_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- High-bit semi-planar P-format → HSV (staged via a reused 8-bit RGB chunk) ----
//
// The AVX2 twins of the scalar `p_n_to_hsv_row` / `p16_to_hsv_row`
// kernels. Rather than re-derive an HSV-specific register pipeline, each
// fills a small fixed reused **8-bit** RGB scratch (one `HSV_CHUNK`-pixel
// chunk at a time) using the EXISTING AVX2 `p_n_to_rgb_row::<BITS, BE>` /
// `p16_to_rgb_row::<BE>` kernel of this file — so the chunk filler IS the
// production 8-bit RGB kernel — then runs the AVX2 `rgb_to_hsv_row` on
// the chunk. This makes the result byte-identical to
// `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))` (and the P016
// analogue) within the AVX2 tier — the same 8-bit RGB intermediate the
// existing high-bit HSV path uses — with no source-width RGB allocation.
// The scalar tail of each underlying RGB kernel handles widths below the
// SIMD block, so no separate tail is needed here. `HSV_CHUNK` (64) is a
// multiple of 2, so every chunk offset lands on a chroma-sample boundary
// for the 1→2 (4:2:0) interleaved-UV shape (`uv` byte offset = pixel
// offset).
//
// The chunked driver is defined locally (mirroring the semi-planar 8-bit
// `nv_to_hsv_via_rgb_chunks`) rather than reusing the planar
// `yuv_to_hsv_via_rgb_chunks`, because this file compiles under
// `yuv-semi-planar` alone while the planar driver lives behind the
// `yuv-planar` gate. Only `rgb_to_hsv_row` (ungated) is shared.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared AVX2 driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing AVX2 RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the AVX2 [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. Byte-identical to `rgb_to_hsv_row(p*_to_rgb_row(...))` within
/// the AVX2 tier, with no source-width RGB allocation.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// AVX2 must be available, and `fill_rgb` must uphold the underlying RGB
/// kernel's safety contract for each chunk. Each of `h_out` / `s_out` /
/// `v_out` must be `>= width`.
#[inline]
unsafe fn pn_hsv_via_rgb_chunks(
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
    // SAFETY: AVX2 verified by the wrapper's `#[target_feature]`; the
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

/// AVX2: high-bit-packed semi-planar 4:2:0 (P010/P012) → planar HSV
/// bytes (OpenCV encoding), staged via the reused-8-bit-RGB-chunk
/// pattern over the AVX2 [`p_n_to_rgb_row`] + [`rgb_to_hsv_row`].
/// Const-generic over `BITS ∈ {10, 12}` and `BE`. Byte-identical to
/// `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))` within the AVX2
/// tier.
///
/// # Safety
///
/// 1. The AVX2 feature must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p_n_to_hsv_row<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the AVX2 high-bit 4:2:0 RGB
  // kernel under the same contract (its own scalar tail covers small n).
  // The same `LOW_PACKED` de-pack threads through the shared RGB kernel.
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p_n_to_rgb_or_rgba_row::<BITS, false, BE, LOW_PACKED>(
        &y[offset..],
        &uv_half[offset..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// AVX2 NV20 → planar HSV bytes (OpenCV encoding), the low-bit twin of
/// [`p_n_to_hsv_row`] (`LOW_PACKED = true`, `BITS = 10`). Byte-identical
/// to `rgb_to_hsv_row(nv20_to_rgb_row::<BE>(...))` within the AVX2 tier.
///
/// # Safety
///
/// Same contract as [`p_n_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nv20_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: forwarded to the shared AVX2 HSV kernel with `LOW_PACKED = true`.
  unsafe {
    p_n_to_hsv_row::<10, BE, true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// AVX2: P016 (semi-planar 4:2:0, 16-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the AVX2 [`p16_to_rgb_row`] +
/// [`rgb_to_hsv_row`]. `BE` selects the source byte order. Byte-
/// identical to `rgb_to_hsv_row(p16_to_rgb_row::<BE>(...))` within the
/// AVX2 tier.
///
/// # Safety
///
/// Same contract as [`p_n_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p16_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards to the AVX2 P016 RGB kernel under the same contract (its own
  // scalar tail covers small n).
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p16_to_rgb_row::<BE>(&y[offset..], &uv_half[offset..], rgb, n, matrix, full_range);
    });
  }
}
