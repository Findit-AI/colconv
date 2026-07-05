use super::*;

/// Compile-time host endianness. `true` on BE targets, `false` on LE
/// targets (always `false` on `x86_64` / `i686` in practice).
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// De-packs a host-native `__m512i` of wire u16 samples to the `BITS`-bit
/// active value: a `_mm512_srl_epi16` right shift by `16 - BITS` (count in
/// the low 64b of `shr_count`) for the high-bit-packed P-formats
/// (`LOW_PACKED = false`), or a low-mask `_mm512_and_si512(v, low_mask)`
/// for low-bit-packed NV20 (`LOW_PACKED = true`). The AVX-512 twin of the
/// scalar `depack_pn`. `LOW_PACKED` is const, so the unused branch is
/// eliminated; byte-identical to the scalar de-pack per lane.
///
/// # Safety
///
/// AVX-512F + AVX-512BW must be available on the current CPU.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn depack_u16x32<const LOW_PACKED: bool>(
  v: __m512i,
  shr_count: __m128i,
  low_mask: __m512i,
) -> __m512i {
  if LOW_PACKED {
    _mm512_and_si512(v, low_mask)
  } else {
    _mm512_srl_epi16(v, shr_count)
  }
}

/// Byte-swap every u16 lane of `v` in-register when the source (wire)
/// endian differs from the host's native u16 byte order.
///
/// Used after `deinterleave_uv_u16_avx512` to apply per-lane byte-swapping.
/// Gated on `BE != HOST_NATIVE_BE` so a hypothetical BE-x86 host would
/// not double-swap. When the gate folds to `false` at compile time, the
/// call compiles away entirely.
#[inline(always)]
unsafe fn byteswap_u16x32<const BE: bool>(v: __m512i) -> __m512i {
  if BE != HOST_NATIVE_BE {
    let mask = unsafe {
      core::mem::transmute::<[u8; 64], __m512i>([
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7,
        6, 9, 8, 11, 10, 13, 12, 15, 14,
      ])
    };
    unsafe { _mm512_shuffle_epi8(v, mask) }
  } else {
    v
  }
}

/// AVX‑512 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **8‑bit** RGB.
///
/// Block size 64 Y pixels / 32 chroma pairs per iteration. Mirrors
/// [`super::x86_avx512::yuv_420p_n_to_rgb_row`] with two structural
/// differences:
/// - Samples are shifted right by `16 - BITS` (`_mm512_srl_epi16`,
///   with a shift count computed from `BITS` once per call) instead
///   of AND‑masked.
/// - Semi‑planar UV is deinterleaved via [`deinterleave_uv_u16_avx512`]
///   — per‑128‑lane shuffle + 64‑bit permute + cross‑vector
///   `_mm512_permutex2var_epi64` to produce 32‑sample U and V
///   vectors.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, false, BE, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 high-bit-packed semi-planar 4:2:0 → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, true, BE, false>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 NV20 (low-bit-packed 4:2:2, 10-bit) → packed **8-bit** RGB.
/// Low-bit twin of [`p_n_to_rgb_row`] (`LOW_PACKED = true`, `BITS = 10`).
/// Byte-identical to [`scalar::nv20_to_rgb_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgb_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv20_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<10, false, BE, true>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 NV20 → packed **8-bit RGBA** (`R, G, B, 0xFF`). Low-bit twin
/// of [`p_n_to_rgba_row`]. Byte-identical to
/// [`scalar::nv20_to_rgba_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgba_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv20_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<10, true, BE, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX-512 P010/P012 kernel. `ALPHA = false` uses
/// `write_rgb_64`; `ALPHA = true` uses `write_rgba_64` with constant
/// `0xFF` alpha.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`. 3. slices long enough +
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`. 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  matrix: ColorMatrix,
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

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    // High-bit-packed: `_mm512_srl_epi16` by `16 - BITS`; low-bit-packed
    // NV20: the `low_mask`. One path is dead per monomorphization.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let low_mask = _mm512_set1_epi16(((1u32 << BITS) - 1) as i16);
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
      // BE input is byte-swapped via `load_endian_u16x32::<BE>` for Y,
      // and via `byteswap_u16x32::<BE>` after deinterleave for UV.
      let y_low_i16 = depack_u16x32::<LOW_PACKED>(
        endian::load_endian_u16x32::<BE>(y.as_ptr().add(x) as *const u8),
        shr_count,
        low_mask,
      );
      let y_high_i16 = depack_u16x32::<LOW_PACKED>(
        endian::load_endian_u16x32::<BE>(y.as_ptr().add(x + 32) as *const u8),
        shr_count,
        low_mask,
      );
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x32::<BE>(u_vec);
      let v_vec = byteswap_u16x32::<BE>(v_vec);
      let u_vec = depack_u16x32::<LOW_PACKED>(u_vec, shr_count, low_mask);
      let v_vec = depack_u16x32::<LOW_PACKED>(v_vec, shr_count, low_mask);

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
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
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

/// AVX‑512 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
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
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 sibling of [`p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth). P016 has its own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE, false>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 NV20 → packed **native-depth `u16`** RGB (low-bit-packed,
/// `[0, 1023]`). Low-bit twin of [`p_n_to_rgb_u16_row`]. Byte-identical
/// to [`scalar::nv20_to_rgb_u16_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv20_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<10, false, BE, true>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 NV20 → packed **native-depth `u16` RGBA** (low-bit-packed;
/// alpha `1023`). Low-bit twin of [`p_n_to_rgba_u16_row`].
/// Byte-identical to [`scalar::nv20_to_rgba_u16_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgba_u16_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv20_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<10, true, BE, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX-512 Pn → native-depth `u16` kernel. `ALPHA = false`
/// writes RGB triples via 8× `write_quarter` per 64-pixel block;
/// `ALPHA = true` writes RGBA quads via 8× `write_quarter_rgba` with
/// constant alpha `(1 << BITS) - 1`. P016 has its own kernel family —
/// never routed here.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  matrix: ColorMatrix,
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

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    // High-bit-packed: `_mm512_srl_epi16` by `16 - BITS`; low-bit-packed
    // NV20: the `low_mask`. One path is dead per monomorphization.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let low_mask = _mm512_set1_epi16(((1u32 << BITS) - 1) as i16);
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
      // BE input is byte-swapped via `load_endian_u16x32::<BE>` for Y,
      // and via `byteswap_u16x32::<BE>` after deinterleave for UV.
      let y_low_i16 = depack_u16x32::<LOW_PACKED>(
        endian::load_endian_u16x32::<BE>(y.as_ptr().add(x) as *const u8),
        shr_count,
        low_mask,
      );
      let y_high_i16 = depack_u16x32::<LOW_PACKED>(
        endian::load_endian_u16x32::<BE>(y.as_ptr().add(x + 32) as *const u8),
        shr_count,
        low_mask,
      );
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x32::<BE>(u_vec);
      let v_vec = byteswap_u16x32::<BE>(v_vec);
      let u_vec = depack_u16x32::<LOW_PACKED>(u_vec, shr_count, low_mask);
      let v_vec = depack_u16x32::<LOW_PACKED>(v_vec, shr_count, low_mask);

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

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 0, dst);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 1, dst.add(32));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 2, dst.add(64));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 3, dst.add(96));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 0, dst.add(128));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 1, dst.add(160));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 2, dst.add(192));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 3, dst.add(224));
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
/// AVX-512 P016 → packed **8-bit** RGB. 64 pixels per iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p16_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<false, BE>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 P016 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p16_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<true, BE>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX-512 P016 kernel. `ALPHA = false` uses `write_rgb_64`;
/// `ALPHA = true` uses `write_rgba_64` with constant `0xFF` alpha.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
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
      // BE input is byte-swapped via `load_endian_u16x32::<BE>` for Y,
      // and via `byteswap_u16x32::<BE>` after deinterleave for UV.
      let y_low = endian::load_endian_u16x32::<BE>(y.as_ptr().add(x) as *const u8);
      let y_high = endian::load_endian_u16x32::<BE>(y.as_ptr().add(x + 32) as *const u8);
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x32::<BE>(u_vec);
      let v_vec = byteswap_u16x32::<BE>(v_vec);

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
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 64;
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

/// AVX-512 P016 → packed **16-bit** RGB.
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgb_out` is `&mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p16_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<false, BE>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 sibling of [`p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p16_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<true, BE>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX-512 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `write_rgb_u16_32`;
/// `ALPHA = true` writes RGBA quads via `write_rgba_u16_32` with
/// constant alpha `0xFFFF`. 32 pixels per iter. Shares the 16-bit
/// arithmetic structure with [`yuv_420p16_to_rgb_or_rgba_u16_row`];
/// only difference is an inline U/V deinterleave at the UV load.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; pointer
  // adds are bounded by `while x + 32 <= width` and caller-promised
  // slice lengths.
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

    let dup_lo_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let dup_hi_idx = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // Per-128-bit-lane shuffle to deinterleave u16 UV pairs within
    // each lane: `[u0,v0,u1,v1,u2,v2,u3,v3] → [u0,u1,u2,u3,v0,v1,v2,v3]`
    // as u16 = byte indices `[0,1, 4,5, 8,9, 12,13 | 2,3, 6,7, 10,11, 14,15]`.
    // When wire endian differs from host (`BE != HOST_NATIVE_BE`):
    // also swap the two bytes within each u16 lane during the
    // deinterleave shuffle. On a hypothetical BE-x86 host this avoids
    // the double-swap that a plain `BE` gate would introduce.
    let uv_lane_mask = if BE != HOST_NATIVE_BE {
      _mm_setr_epi8(1, 0, 5, 4, 9, 8, 13, 12, 3, 2, 7, 6, 11, 10, 15, 14)
    } else {
      _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15)
    };
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After the per-lane shuffle the 64-bit lane layout is
    // `[U0_3, V0_3, U4_7, V4_7, U8_11, V8_11, U12_15, V12_15]`; permute
    // to `[U0_3, U4_7, U8_11, U12_15 | V0_3, V4_7, V8_11, V12_15]` so
    // low 256 = all U, high 256 = all V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // BE input is byte-swapped via `load_endian_u16x32::<BE>` for Y.
      // The UV byte-swap is folded into the deinterleave shuffle mask.
      let y_vec = endian::load_endian_u16x32::<BE>(y.as_ptr().add(x) as *const u8);
      // 16 UV pairs = 32 u16 = one 512-bit load.
      let uv_raw = _mm512_loadu_si512(uv_half.as_ptr().add(x).cast());
      let uv_deint = _mm512_shuffle_epi8(uv_raw, uv_deint_mask);
      let uv_split = _mm512_permutexvar_epi64(uv_collect, uv_deint);
      let u_vec = _mm512_castsi512_si256(uv_split);
      let v_vec = _mm512_extracti64x4_epi64::<1>(uv_split);

      // Center UV (same wrapping trick as planar kernel).
      let u_i16 = _mm256_sub_epi16(u_vec, _mm512_castsi512_si256(bias16_v));
      let v_i16 = _mm256_sub_epi16(v_vec, _mm512_castsi512_si256(bias16_v));

      let u_i32 = _mm512_cvtepi16_epi32(u_i16);
      let v_i32 = _mm512_cvtepi16_epi32(v_i16);

      let u_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_i32_v,
      ));

      let u_d_odd = _mm512_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm512_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x8_avx512(cru, crv, u_d, v_d, rnd_i64_v);
      let r_ch_odd = chroma_i64x8_avx512(cru, crv, u_d_odd, v_d_odd, rnd_i64_v);
      let g_ch_even = chroma_i64x8_avx512(cgu, cgv, u_d, v_d, rnd_i64_v);
      let g_ch_odd = chroma_i64x8_avx512(cgu, cgv, u_d_odd, v_d_odd, rnd_i64_v);
      let b_ch_even = chroma_i64x8_avx512(cbu, cbv, u_d, v_d, rnd_i64_v);
      let b_ch_odd = chroma_i64x8_avx512(cbu, cbv, u_d_odd, v_d_odd, rnd_i64_v);

      let r_ch_i32 = reassemble_i32x16(r_ch_even, r_ch_odd, interleave_idx);
      let g_ch_i32 = reassemble_i32x16(g_ch_even, g_ch_odd, interleave_idx);
      let b_ch_i32 = reassemble_i32x16(b_ch_even, b_ch_odd, interleave_idx);

      let r_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, r_ch_i32);
      let r_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, r_ch_i32);
      let g_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, g_ch_i32);
      let g_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, g_ch_i32);
      let b_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, b_ch_i32);
      let b_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, b_ch_i32);

      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_dup_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_dup_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_dup_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_dup_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_dup_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_dup_hi);

      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      if ALPHA {
        write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
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
// The AVX‑512 twins of the scalar `p_n_to_hsv_row` / `p16_to_hsv_row`
// kernels. Rather than re-derive an HSV-specific register pipeline, each
// fills a small fixed reused **8-bit** RGB scratch (one `HSV_CHUNK`-pixel
// chunk at a time) using the EXISTING AVX‑512 `p_n_to_rgb_row::<BITS,
// BE>` / `p16_to_rgb_row::<BE>` kernel of this file — so the chunk filler
// IS the production 8-bit RGB kernel — then runs the AVX‑512
// `rgb_to_hsv_row` on the chunk. This makes the result byte-identical to
// `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))` (and the P016
// analogue) within the AVX‑512 tier — the same 8-bit RGB intermediate
// the existing high-bit HSV path uses — with no source-width RGB
// allocation. The scalar tail of each underlying RGB kernel handles
// widths below the SIMD block, so no separate tail is needed here.
// `HSV_CHUNK` (64) is a multiple of 2, so every chunk offset lands on a
// chroma-sample boundary for the 1→2 (4:2:0) interleaved-UV shape (`uv`
// byte offset = pixel offset).
//
// The chunked driver is defined locally (mirroring the semi-planar 8-bit
// `nv_to_hsv_via_rgb_chunks`) rather than reusing the planar
// `yuv_to_hsv_via_rgb_chunks`, because this file compiles under
// `yuv-semi-planar` alone while the planar driver lives behind the
// `yuv-planar` gate. Only `rgb_to_hsv_row` (ungated) is shared.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared AVX‑512 driver: walks `width` in `HSV_CHUNK`-pixel chunks,
/// fills a small reused stack RGB scratch via `fill_rgb` (the existing
/// AVX‑512 RGB kernel for the format, passed the chunk `offset` and
/// length `n`), then runs the AVX‑512 [`rgb_to_hsv_row`] on that chunk
/// into the H/S/V planes. Byte-identical to
/// `rgb_to_hsv_row(p*_to_rgb_row(...))` within the AVX‑512 tier, with no
/// source-width RGB allocation.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// AVX‑512BW must be available, and `fill_rgb` must uphold the underlying
/// RGB kernel's safety contract for each chunk. Each of `h_out` /
/// `s_out` / `v_out` must be `>= width`.
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
    // SAFETY: AVX‑512 verified by the wrapper's `#[target_feature]`; the
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

/// AVX‑512: high-bit-packed semi-planar 4:2:0 (P010/P012) → planar HSV
/// bytes (OpenCV encoding), staged via the reused-8-bit-RGB-chunk
/// pattern over the AVX‑512 [`p_n_to_rgb_row`] + [`rgb_to_hsv_row`].
/// Const-generic over `BITS ∈ {10, 12}` and `BE`. Byte-identical to
/// `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))` within the AVX‑512
/// tier.
///
/// # Safety
///
/// 1. The AVX‑512BW feature must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p_n_to_hsv_row<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the AVX‑512 high-bit 4:2:0 RGB
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

/// AVX-512 NV20 → planar HSV bytes (OpenCV encoding), the low-bit twin
/// of [`p_n_to_hsv_row`] (`LOW_PACKED = true`, `BITS = 10`).
/// Byte-identical to `rgb_to_hsv_row(nv20_to_rgb_row::<BE>(...))` within
/// the AVX-512 tier.
///
/// # Safety
///
/// Same contract as [`p_n_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nv20_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: forwarded to the shared AVX-512 HSV kernel with `LOW_PACKED = true`.
  unsafe {
    p_n_to_hsv_row::<10, BE, true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// AVX‑512: P016 (semi-planar 4:2:0, 16-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the AVX‑512 [`p16_to_rgb_row`] +
/// [`rgb_to_hsv_row`]. `BE` selects the source byte order. Byte-
/// identical to `rgb_to_hsv_row(p16_to_rgb_row::<BE>(...))` within the
/// AVX‑512 tier.
///
/// # Safety
///
/// Same contract as [`p_n_to_hsv_row`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p16_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards to the AVX‑512 P016 RGB kernel under the same contract (its
  // own scalar tail covers small n).
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p16_to_rgb_row::<BE>(&y[offset..], &uv_half[offset..], rgb, n, matrix, full_range);
    });
  }
}
