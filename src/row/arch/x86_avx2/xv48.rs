//! AVX2 kernels for XV48 packed YUV 4:4:4 16-bit family
//! (FFmpeg `AV_PIX_FMT_XV48LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), X(16)]`, each
//! holding a full 16-bit sample (no padding bits, no right-shift on
//! load — the full-depth sibling of XV36). The `X` slot is **padding** —
//! loaded but discarded. RGBA outputs force α = max.
//!
//! ## Per-iter pipeline (32 px / iter for u8, 16 px / iter for u16)
//!
//! Both paths use the same 16-pixel deinterleave helper (the post-fix
//! XV36 AVX2 cross-lane reshape, slots U=0, Y=1, V=2, X=3). The u8 path
//! runs the helper twice per main-loop iteration; the u16 path runs it
//! once.
//!
//! - u8 output: chroma centered (subtract 32768 via wrapping
//!   `-32768i16`), Q15 chroma scale via `chroma_i16x16` (i32 widening —
//!   no overflow at BITS=16/8). Y scaled via `scale_y_u16_avx2`.
//!
//! - u16 output: i64 chroma via `chroma_i64x4_avx2` to avoid i32
//!   overflow at BITS=16/16. Y scaled via `scale_y_i32x8_i64`.
//!
//! ## Tail
//!
//! `width % block_size` remaining pixels fall through to `scalar::xv48_*`.

use super::{endian, *};
use crate::{KernelMatrix, row::scalar};

// ---- Deinterleave helper (16 pixels / 64 u16 / 128 bytes) ---------------

/// Deinterleaves 16 XV48 quadruples (64 u16 = 128 bytes) from `ptr`
/// into `(u_vec, y_vec, v_vec)` — three `__m256i` vectors each holding
/// 16 `u16` samples in **natural pixel order** (lane n = u16 from pixel
/// n). Channel slot order in source: U=0, Y=1, V=2, X=3. No shift
/// (16-bit native). The X channel (slot 3) is computed but discarded.
///
/// Uses the post-fix XV36 AVX2 cross-lane reshape (no trailing
/// `_mm256_permute4x64_epi64` — see the XV36 backend).
///
/// # Safety
///
/// `ptr` must point to at least 128 readable bytes (64 `u16`
/// elements). Caller's `target_feature` must include AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_xv48_16px_avx2<const BE: bool>(
  ptr: *const u16,
) -> (__m256i, __m256i, __m256i) {
  // SAFETY: caller obligation — `ptr` has 128 bytes readable; AVX2 is
  // available.
  unsafe {
    // Load 4 × __m256i contiguously (16 pixels × 4 channels × u16 = 128 bytes).
    let raw_c0 = endian::load_endian_u16x16::<BE>(ptr as *const u8);
    let raw_c1 = endian::load_endian_u16x16::<BE>(ptr.add(16) as *const u8);
    let raw_c2 = endian::load_endian_u16x16::<BE>(ptr.add(32) as *const u8);
    let raw_c3 = endian::load_endian_u16x16::<BE>(ptr.add(48) as *const u8);

    // Reshape via cross-lane permute so each register holds strided lanes:
    //   raw0 lo=P0,P1 hi=P8,P9   raw1 lo=P2,P3 hi=P10,P11
    //   raw2 lo=P4,P5 hi=P12,P13 raw3 lo=P6,P7 hi=P14,P15
    let raw0 = _mm256_permute2x128_si256::<0x20>(raw_c0, raw_c2);
    let raw1 = _mm256_permute2x128_si256::<0x31>(raw_c0, raw_c2);
    let raw2 = _mm256_permute2x128_si256::<0x20>(raw_c1, raw_c3);
    let raw3 = _mm256_permute2x128_si256::<0x31>(raw_c1, raw_c3);

    // Level 1: unpack pairs within each lane (XV48 order: U=0, Y=1, V=2, X=3).
    let s1_lo = _mm256_unpacklo_epi16(raw0, raw1);
    let s1_hi = _mm256_unpackhi_epi16(raw0, raw1);
    let s2_lo = _mm256_unpacklo_epi16(raw2, raw3);
    let s2_hi = _mm256_unpackhi_epi16(raw2, raw3);

    // Level 2: merge lo/hi within each group.
    let s3_lo = _mm256_unpacklo_epi16(s1_lo, s1_hi);
    let s3_hi = _mm256_unpackhi_epi16(s1_lo, s1_hi);
    let s4_lo = _mm256_unpacklo_epi16(s2_lo, s2_hi);
    let s4_hi = _mm256_unpackhi_epi16(s2_lo, s2_hi);

    // Level 3: combine the two groups via per-lane unpacklo/hi_epi64.
    // The reshape already accumulated natural [0..15] order — no 4x64
    // cross-lane permute needed.
    let u_vec = _mm256_unpacklo_epi64(s3_lo, s4_lo); // U0..U15
    let y_vec = _mm256_unpackhi_epi64(s3_lo, s4_lo); // Y0..Y15
    let v_vec = _mm256_unpacklo_epi64(s3_hi, s4_hi); // V0..V15
    // x_vec = _mm256_unpackhi_epi64(s3_hi, s4_hi) — X0..X15, discarded.

    (u_vec, y_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (32 px/iter) ----------------------------------

/// AVX2 XV48 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv48_to_rgb_or_rgba_row::<ALPHA, BE>`.
///
/// Block size: 32 pixels per SIMD iteration (two 16-pixel deinterleaves).
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn xv48_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
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
    // X is padding — RGBA forces α = 0xFF.
    let alpha_u8 = _mm256_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 32 <= width {
      // --- lo half: pixels x..x+15 (one 16-pixel deinterleave) ----------
      let (u_lo_u16, y_lo_u16, v_lo_u16) =
        deinterleave_xv48_16px_avx2::<BE>(packed.as_ptr().add(x * 4));

      let u_lo_i16 = _mm256_sub_epi16(u_lo_u16, bias16_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_u16, bias16_v);

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);

      // Y: full u16 values → use scale_y_u16_avx2 (NOT scale_y).
      let y_lo_scaled = scale_y_u16_avx2(y_lo_u16, y_off_v, y_scale_v, rnd_v);

      // --- hi half: pixels x+16..x+31 (one more 16-pixel deinterleave) --
      let (u_hi_u16, y_hi_u16, v_hi_u16) =
        deinterleave_xv48_16px_avx2::<BE>(packed.as_ptr().add(x * 4 + 64));

      let u_hi_i16 = _mm256_sub_epi16(u_hi_u16, bias16_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_u16, bias16_v);

      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_hi_scaled = scale_y_u16_avx2(y_hi_u16, y_off_v, y_scale_v, rnd_v);

      // Saturating add Y + chroma per channel; narrow both halves to u8x32.
      let r_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, r_chroma_lo),
        _mm256_adds_epi16(y_hi_scaled, r_chroma_hi),
      );
      let g_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, g_chroma_lo),
        _mm256_adds_epi16(y_hi_scaled, g_chroma_hi),
      );
      let b_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, b_chroma_lo),
        _mm256_adds_epi16(y_hi_scaled, b_chroma_hi),
      );

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out_ptr);
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv48_to_rgb_or_rgba_row::<ALPHA, BE>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (16 px/iter) --------------------

/// AVX2 XV48 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x4_avx2`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>`.
///
/// Block size: 16 pixels per SIMD iteration (one 16-pixel deinterleave).
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn xv48_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16); // 0xFFFF
    let rnd_v = _mm256_set1_epi64x(RND);
    let rnd32_v = _mm256_set1_epi32(1 << 14);
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

    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 XV48 quadruples → U, Y, V as u16x16 (natural order).
      let (u_u16, y_vec, v_u16) = deinterleave_xv48_16px_avx2::<BE>(packed.as_ptr().add(x * 4));

      let u_i16 = _mm256_sub_epi16(u_u16, bias16_v);
      let v_i16 = _mm256_sub_epi16(v_u16, bias16_v);

      // Lo half (pixels 0..7): widen 8 i16 → i32x8.
      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd32_v,
      ));

      // Hi half (pixels 8..15): widen upper 8 i16 → i32x8.
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd32_v,
      ));

      // i64 chroma: even/odd i32 lanes via 0xF5 shuffle.
      let u_d_lo_odd = _mm256_shuffle_epi32::<0xF5>(u_d_lo);
      let v_d_lo_odd = _mm256_shuffle_epi32::<0xF5>(v_d_lo);
      let u_d_hi_odd = _mm256_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_hi_odd = _mm256_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_even = chroma_i64x4_avx2(cru, crv, u_d_lo, v_d_lo, rnd_v);
      let r_ch_lo_odd = chroma_i64x4_avx2(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let g_ch_lo_even = chroma_i64x4_avx2(cgu, cgv, u_d_lo, v_d_lo, rnd_v);
      let g_ch_lo_odd = chroma_i64x4_avx2(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let b_ch_lo_even = chroma_i64x4_avx2(cbu, cbv, u_d_lo, v_d_lo, rnd_v);
      let b_ch_lo_odd = chroma_i64x4_avx2(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_v);

      let r_ch_hi_even = chroma_i64x4_avx2(cru, crv, u_d_hi, v_d_hi, rnd_v);
      let r_ch_hi_odd = chroma_i64x4_avx2(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let g_ch_hi_even = chroma_i64x4_avx2(cgu, cgv, u_d_hi, v_d_hi, rnd_v);
      let g_ch_hi_odd = chroma_i64x4_avx2(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let b_ch_hi_even = chroma_i64x4_avx2(cbu, cbv, u_d_hi, v_d_hi, rnd_v);
      let b_ch_hi_odd = chroma_i64x4_avx2(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_v);

      // Reassemble each pair of i64x4 → i32x8.
      let r_ch_lo_i32 = reassemble_i64x4_to_i32x8(r_ch_lo_even, r_ch_lo_odd);
      let g_ch_lo_i32 = reassemble_i64x4_to_i32x8(g_ch_lo_even, g_ch_lo_odd);
      let b_ch_lo_i32 = reassemble_i64x4_to_i32x8(b_ch_lo_even, b_ch_lo_odd);
      let r_ch_hi_i32 = reassemble_i64x4_to_i32x8(r_ch_hi_even, r_ch_hi_odd);
      let g_ch_hi_i32 = reassemble_i64x4_to_i32x8(g_ch_hi_even, g_ch_hi_odd);
      let b_ch_hi_i32 = reassemble_i64x4_to_i32x8(b_ch_hi_even, b_ch_hi_odd);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      let y_lo_u16 = _mm256_castsi256_si128(y_vec);
      let y_hi_u16 = _mm256_extracti128_si256::<1>(y_vec);
      let y_lo_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x8_i64(y_lo_i32, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_i32x8_i64(y_hi_i32, y_scale_v, rnd_v);

      // Add Y + chroma in i32; saturate-narrow to u16 via _mm256_packus_epi32
      // + 0xD8 lane fixup.
      let r_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, r_ch_lo_i32),
        _mm256_add_epi32(y_hi_scaled, r_ch_hi_i32),
      ));
      let g_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, g_ch_lo_i32),
        _mm256_add_epi32(y_hi_scaled, g_ch_hi_i32),
      ));
      let b_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, b_ch_lo_i32),
        _mm256_add_epi32(y_hi_scaled, b_ch_hi_i32),
      ));

      // Write 16 pixels via two 8-pixel u16 helpers.
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

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (16 px/iter) -----------------------------------------------

/// AVX2 XV48 → u8 luma. Y is quadruple element 1; `>> 8` extracts the
/// high byte. Reuses the full deinterleave helper and discards U/V.
///
/// Byte-identical to `scalar::xv48_to_luma_row`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn xv48_to_luma_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_u, y_vec, _v) = deinterleave_xv48_16px_avx2::<BE>(packed.as_ptr().add(x * 4));

      let y_shr = _mm256_srli_epi16::<8>(y_vec);
      let zero = _mm256_setzero_si256();
      let y_u8 = narrow_u8x32(y_shr, zero);

      // Store low 16 bytes (the valid Y values).
      let mut tmp = [0u8; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), y_u8);
      luma_out[x..x + 16].copy_from_slice(&tmp[..16]);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_row::<BE>(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- Luma u16 (16 px/iter) ----------------------------------------------

/// AVX2 XV48 → u16 luma. Direct copy of Y samples (slot 1, no shift —
/// 16-bit native). Reuses the full deinterleave helper and discards U/V.
///
/// Byte-identical to `scalar::xv48_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn xv48_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_u, y_vec, _v) = deinterleave_xv48_16px_avx2::<BE>(packed.as_ptr().add(x * 4));
      // Direct store — Y samples are 16-bit native, in natural pixel order.
      _mm256_storeu_si256(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_u16_row::<BE>(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- XV48 → HSV (staged via a reused 8-bit RGB chunk) ----------------
//
// The SIMD twin of the scalar `xv48_to_hsv_row` kernel — fills a small
// reused 8-bit RGB scratch via the production `xv48_to_rgb_row` kernel,
// then runs `rgb_to_hsv_row`. The X slot is dropped (HSV is colour-only).

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared SIMD driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb`, then runs the SIMD
/// [`rgb_to_hsv_row`] on that chunk into the H/S/V planes.
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
unsafe fn xv48_hsv_via_rgb_chunks(
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
    // SAFETY: the SIMD feature is verified by the wrapper's
    // `#[target_feature]`; the chunk and the output sub-slices are all
    // length `n`.
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

/// SIMD: XV48 (packed 4:4:4, 16-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the reused-8-bit-RGB-chunk pattern over the
/// SIMD [`xv48_to_rgb_or_rgba_row`] + [`rgb_to_hsv_row`]. Byte-identical
/// to `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))` within
/// this SIMD tier. The padding X slot is dropped (HSV is colour-only).
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn xv48_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the SIMD XV48 RGB kernel under
  // the same contract (its own scalar tail covers small n).
  unsafe {
    xv48_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      xv48_to_rgb_or_rgba_row::<false, BE>(&packed[offset * 4..], rgb, n, matrix, full_range);
    });
  }
}
