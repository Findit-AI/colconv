//! AVX-512 (F + BW) kernels for the Tier 12 (DCP / `Xyz12`) source.
//!
//! Each kernel processes **16 pixels per SIMD iteration** in `__m512`
//! (512-bit, 16 f32 lanes) registers — twice the AVX2 throughput.
//! The matmul, integer narrow, and interleaved store are vectorized;
//! the SMPTE 428-1 inverse OETF and sRGB-shape forward OETF run
//! **scalar per lane** via the same scalar functions the reference
//! kernel uses, preserving the 0-ULP scalar↔SIMD parity contract by
//! construction.
//!
//! Pipeline:
//! 1. Six `__m128i` loads (16 px × 3 channels = 48 u16 = 96 bytes)
//!    feed two SSE-style 8-pixel deinterleaves, each producing
//!    `(X8, Y8, Z8)` u16x8. The two halves are merged with
//!    `_mm256_inserti128_si256` into `__m256i` channel vectors.
//! 2. Right-shift by 4 (`_mm256_srli_epi16::<4>`) extracts the active
//!    12-bit code from the high-bit-packed `u16` (FFmpeg
//!    `AV_PIX_FMT_XYZ12LE/BE`: code in `[15:4]`, low 4 bits zero);
//!    defensive `_mm256_and_si256` mask + `_mm512_cvtepu16_epi32`
//!    widens to i32x16 in `__m512i`; `_mm512_cvtepi32_ps` casts to
//!    f32x16 in `__m512`.
//! 3. Per-lane scalar `smpte428_inverse_oetf` produces linear XYZ.
//! 4. Vectorized 3×3 matmul uses plain `_mm512_mul_ps + _mm512_add_ps`
//!    (NOT FMA via `_mm512_fmadd_ps`) — single-rounding FMA breaks
//!    the 0-ULP parity contract on integer-narrow output paths.
//! 5. Per-lane scalar `oetf_srgb` (only for u8 / u16 / f16 outputs).
//! 6. Clamp `[0, 1]` × `scale` + `+0.5` truncate via
//!    `_mm512_cvttps_epi32` (round-half-up matches scalar's
//!    `(c * scale + 0.5) as int`); saturate to u16 via
//!    `_mm512_packus_epi32` followed by `_mm512_permutexvar_epi64`
//!    lane fixup (per-128-bit pack reorders channels).
//!
//! Width remainder (`width % 16`) is handled by the scalar reference.
//!
//! # Numerical contract
//!
//! Bit-identical scalar↔SIMD output across all integer / f16 / f32
//! paths (verified by `tests::xyz12`).

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use super::super::{
  x86_common::{write_rgb_16, write_rgba_16},
  x86_sse41::endian::load_endian_u16x8,
};
use crate::{
  KernelGamut,
  row::scalar::{
    self,
    xyz12::{oetf_srgb, smpte428_inverse_oetf},
    xyz12_constants::xyz_to_rgb_matrix,
  },
};

const PIXELS_PER_ITER: usize = 16;
const SAMPLE_MASK_U16: u16 = 0x0FFF;

// ---- Internal helpers --------------------------------------------------

/// Same SSE4.1 byte-shuffle pattern as `x86_sse41::xyz12` /
/// `x86_avx2::xyz12`. Inlined here rather than crossing module
/// boundaries because the per-arch wrappers carry different
/// `target_feature` annotations.
#[inline(always)]
unsafe fn deinterleave_xyz12_8px(
  v0: __m128i,
  v1: __m128i,
  v2: __m128i,
) -> (__m128i, __m128i, __m128i) {
  unsafe {
    let x_v0 = _mm_setr_epi8(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let x_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
    let x_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
    let x = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, x_v0), _mm_shuffle_epi8(v1, x_v1)),
      _mm_shuffle_epi8(v2, x_v2),
    );

    let y_v0 = _mm_setr_epi8(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
    let y_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
    let y = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, y_v0), _mm_shuffle_epi8(v1, y_v1)),
      _mm_shuffle_epi8(v2, y_v2),
    );

    let z_v0 = _mm_setr_epi8(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let z_v1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1);
    let z_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15);
    let z = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, z_v0), _mm_shuffle_epi8(v1, z_v1)),
      _mm_shuffle_epi8(v2, z_v2),
    );

    (x, y, z)
  }
}

/// Loads 16 pixels of packed XYZ12 via two 8-pixel SSE deinterleaves.
/// Returns three `__m256i` channel vectors (16 u16 each).
#[inline(always)]
unsafe fn load_and_deinterleave_16px<const BE: bool>(p: *const u8) -> (__m256i, __m256i, __m256i) {
  unsafe {
    let v0 = load_endian_u16x8::<BE>(p);
    let v1 = load_endian_u16x8::<BE>(p.add(16));
    let v2 = load_endian_u16x8::<BE>(p.add(32));
    let v3 = load_endian_u16x8::<BE>(p.add(48));
    let v4 = load_endian_u16x8::<BE>(p.add(64));
    let v5 = load_endian_u16x8::<BE>(p.add(80));
    let (x_lo, y_lo, z_lo) = deinterleave_xyz12_8px(v0, v1, v2);
    let (x_hi, y_hi, z_hi) = deinterleave_xyz12_8px(v3, v4, v5);
    let x_full = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(x_lo), x_hi);
    let y_full = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(y_lo), y_hi);
    let z_full = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(z_lo), z_hi);
    (x_full, y_full, z_full)
  }
}

/// Widens a u16x16 channel vector (already 12-bit-masked) to f32x16.
#[inline(always)]
unsafe fn u16x16_to_f32x16(v: __m256i) -> __m512 {
  unsafe { _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v)) }
}

/// Per-lane scalar SMPTE 428-1 inverse OETF on an `__m512` (16 lanes).
#[inline(always)]
unsafe fn smpte428_inv_oetf_scalar16(v: __m512) -> __m512 {
  unsafe {
    let mut buf = [0.0_f32; 16];
    _mm512_storeu_ps(buf.as_mut_ptr(), v);
    for slot in &mut buf {
      *slot = smpte428_inverse_oetf(*slot as u16);
    }
    _mm512_loadu_ps(buf.as_ptr())
  }
}

/// Per-lane scalar sRGB OETF on an `__m512` (16 lanes).
#[inline(always)]
unsafe fn oetf_srgb_scalar16(v: __m512) -> __m512 {
  unsafe {
    let mut buf = [0.0_f32; 16];
    _mm512_storeu_ps(buf.as_mut_ptr(), v);
    for slot in &mut buf {
      *slot = oetf_srgb(*slot);
    }
    _mm512_loadu_ps(buf.as_ptr())
  }
}

/// Vectorized 3×3 matmul on a 16-lane f32 vector. Plain mul + add
/// (NOT FMA).
#[inline(always)]
unsafe fn matmul_xyz_to_rgb_16lane(
  m: &[[f32; 3]; 3],
  x: __m512,
  y: __m512,
  z: __m512,
) -> (__m512, __m512, __m512) {
  unsafe {
    let m00 = _mm512_set1_ps(m[0][0]);
    let m01 = _mm512_set1_ps(m[0][1]);
    let m02 = _mm512_set1_ps(m[0][2]);
    let m10 = _mm512_set1_ps(m[1][0]);
    let m11 = _mm512_set1_ps(m[1][1]);
    let m12 = _mm512_set1_ps(m[1][2]);
    let m20 = _mm512_set1_ps(m[2][0]);
    let m21 = _mm512_set1_ps(m[2][1]);
    let m22 = _mm512_set1_ps(m[2][2]);
    let r = _mm512_add_ps(
      _mm512_add_ps(_mm512_mul_ps(m00, x), _mm512_mul_ps(m01, y)),
      _mm512_mul_ps(m02, z),
    );
    let g = _mm512_add_ps(
      _mm512_add_ps(_mm512_mul_ps(m10, x), _mm512_mul_ps(m11, y)),
      _mm512_mul_ps(m12, z),
    );
    let b = _mm512_add_ps(
      _mm512_add_ps(_mm512_mul_ps(m20, x), _mm512_mul_ps(m21, y)),
      _mm512_mul_ps(m22, z),
    );
    (r, g, b)
  }
}

/// Loads 16 XYZ12 pixels and produces (R, G, B) `__m512` after the
/// inverse-OETF + 3×3 matmul.
#[inline(always)]
unsafe fn load_and_matmul_16px<const BE: bool>(
  p: *const u8,
  m: &[[f32; 3]; 3],
) -> (__m512, __m512, __m512) {
  unsafe {
    let (x_u, y_u, z_u) = load_and_deinterleave_16px::<BE>(p);
    // Shift right 4 to extract active 12-bit code (high-bit-packed
    // FFmpeg `AV_PIX_FMT_XYZ12LE/BE`); mask defensively.
    let mask = _mm256_set1_epi16(SAMPLE_MASK_U16 as i16);
    let x_shr = _mm256_and_si256(_mm256_srli_epi16::<4>(x_u), mask);
    let y_shr = _mm256_and_si256(_mm256_srli_epi16::<4>(y_u), mask);
    let z_shr = _mm256_and_si256(_mm256_srli_epi16::<4>(z_u), mask);
    let x_lin = smpte428_inv_oetf_scalar16(u16x16_to_f32x16(x_shr));
    let y_lin = smpte428_inv_oetf_scalar16(u16x16_to_f32x16(y_shr));
    let z_lin = smpte428_inv_oetf_scalar16(u16x16_to_f32x16(z_shr));
    matmul_xyz_to_rgb_16lane(m, x_lin, y_lin, z_lin)
  }
}

/// Loads 16 XYZ12 pixels and produces (X, Y, Z) `__m512` linear-XYZ.
#[inline(always)]
unsafe fn load_xyz_linear_16px<const BE: bool>(p: *const u8) -> (__m512, __m512, __m512) {
  unsafe {
    let (x_u, y_u, z_u) = load_and_deinterleave_16px::<BE>(p);
    let mask = _mm256_set1_epi16(SAMPLE_MASK_U16 as i16);
    let x_shr = _mm256_and_si256(_mm256_srli_epi16::<4>(x_u), mask);
    let y_shr = _mm256_and_si256(_mm256_srli_epi16::<4>(y_u), mask);
    let z_shr = _mm256_and_si256(_mm256_srli_epi16::<4>(z_u), mask);
    let x_lin = smpte428_inv_oetf_scalar16(u16x16_to_f32x16(x_shr));
    let y_lin = smpte428_inv_oetf_scalar16(u16x16_to_f32x16(y_shr));
    let z_lin = smpte428_inv_oetf_scalar16(u16x16_to_f32x16(z_shr));
    (x_lin, y_lin, z_lin)
  }
}

/// Vectorized clamp `[0, 1]` × scale + `+0.5` truncate.
#[inline(always)]
unsafe fn clamp_scale_to_u32x16(v: __m512, zero: __m512, one: __m512, scale: __m512) -> __m512i {
  unsafe {
    let half = _mm512_set1_ps(0.5);
    let clamped = _mm512_min_ps(_mm512_max_ps(v, zero), one);
    let scaled = _mm512_add_ps(_mm512_mul_ps(clamped, scale), half);
    _mm512_cvttps_epi32(scaled)
  }
}

/// Narrow 16 u32 lanes → 16 u16 lanes via `_mm512_packus_epi32` +
/// 64-bit lane fixup. `_mm512_packus_epi32` packs per 128-bit lane,
/// so the natural u32 indices `[0..16]` end up scrambled across the
/// 4 × 128-bit halves; the `0xD8`-equivalent 64-bit permute restores
/// the natural order.
#[inline(always)]
unsafe fn narrow_u32x16_to_u16x16(v: __m512i) -> __m256i {
  unsafe {
    // Pack with self → 32 u16 lanes (duplicated halves).
    // Each 128-bit lane contains: [v0..3, v0..3] (low pair from v
    // packed twice). The natural channel order across the 4 lanes
    // is {0..3, 0..3, 4..7, 4..7, 8..11, 8..11, 12..15, 12..15}.
    let packed = _mm512_packus_epi32(v, v);
    // Fix the lane scrambling so we get [0..7] in low 256 bits,
    // [8..15] in high 256 bits (i.e., the natural order of the 16
    // input u32 lanes mapped 1:1 to 16 u16 lanes).
    //
    // Per-lane content after `_mm512_packus_epi32(v, v)` (each lane
    // is a 128-bit window over the concatenated v||v):
    //   lane0 (qwords 0,1) = u32 lanes 0,1,2,3 → u16 0..3 plus duplicate
    //   lane1 (qwords 2,3) = u32 lanes 4,5,6,7
    //   lane2 (qwords 4,5) = u32 lanes 8,9,10,11
    //   lane3 (qwords 6,7) = u32 lanes 12,13,14,15
    // The "natural" u16 order for our 16 input u32s is q0,q2,q4,q6
    // (skipping the duplicated halves). Use `_mm512_permutex_epi64`
    // to gather those four qwords into the low 256 bits.
    let permuted = _mm512_permutexvar_epi64(_mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0), packed);
    _mm512_castsi512_si256(permuted)
  }
}

// ---- Per-output kernels ------------------------------------------------

/// XYZ12 → packed u8 RGB (16 px/iter).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xyz12_to_rgb_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = _mm512_setzero_ps();
    let one_ps = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_16px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar16(r_lin);
      let g_oetf = oetf_srgb_scalar16(g_lin);
      let b_oetf = oetf_srgb_scalar16(b_lin);
      let r_u32 = clamp_scale_to_u32x16(r_oetf, zero_ps, one_ps, scale);
      let g_u32 = clamp_scale_to_u32x16(g_oetf, zero_ps, one_ps, scale);
      let b_u32 = clamp_scale_to_u32x16(b_oetf, zero_ps, one_ps, scale);
      let r_u16 = narrow_u32x16_to_u16x16(r_u32);
      let g_u16 = narrow_u32x16_to_u16x16(g_u32);
      let b_u16 = narrow_u32x16_to_u16x16(b_u32);
      // Pack u16x16 → u8x16: AVX2 `packus_epi16` is per-128-bit-lane,
      // so a `permute4x64_epi64::<0xD8>` fixup restores the natural
      // [0..15] u8 order in the low 128 bits. Same idiom as
      // `planar_gbr_high_bit::rgb48_to_rgb_row`.
      let zero256 = _mm256_setzero_si256();
      let r_u8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(
        r_u16, zero256,
      )));
      let g_u8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(
        g_u16, zero256,
      )));
      let b_u8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(
        b_u16, zero256,
      )));
      // In-register 16-pixel RGB interleave via the shared
      // `write_rgb_16` helper (48-byte store) avoids a 3× `[u16; 16]`
      // stack-temp + per-pixel scalar scatter.
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgb_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed u8 RGBA (alpha = `0xFF`).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xyz12_to_rgba_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = _mm512_setzero_ps();
    let one_ps = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_16px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar16(r_lin);
      let g_oetf = oetf_srgb_scalar16(g_lin);
      let b_oetf = oetf_srgb_scalar16(b_lin);
      let r_u32 = clamp_scale_to_u32x16(r_oetf, zero_ps, one_ps, scale);
      let g_u32 = clamp_scale_to_u32x16(g_oetf, zero_ps, one_ps, scale);
      let b_u32 = clamp_scale_to_u32x16(b_oetf, zero_ps, one_ps, scale);
      let r_u16 = narrow_u32x16_to_u16x16(r_u32);
      let g_u16 = narrow_u32x16_to_u16x16(g_u32);
      let b_u16 = narrow_u32x16_to_u16x16(b_u32);
      let zero256 = _mm256_setzero_si256();
      let r_u8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(
        r_u16, zero256,
      )));
      let g_u8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(
        g_u16, zero256,
      )));
      let b_u8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(
        b_u16, zero256,
      )));
      let a_u8 = _mm_set1_epi8(-1_i8);
      // In-register 16-pixel RGBA interleave via the shared
      // `write_rgba_16` helper (64-byte store) avoids a 3× `[u16; 16]`
      // stack-temp + per-pixel scalar scatter.
      write_rgba_16(r_u8, g_u8, b_u8, a_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgba_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgba_out[x * 4..width * 4],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed u16 RGB (full-range scaling).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xyz12_to_rgb_u16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = _mm512_setzero_ps();
    let one_ps = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_16px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar16(r_lin);
      let g_oetf = oetf_srgb_scalar16(g_lin);
      let b_oetf = oetf_srgb_scalar16(b_lin);
      let r_u32 = clamp_scale_to_u32x16(r_oetf, zero_ps, one_ps, scale);
      let g_u32 = clamp_scale_to_u32x16(g_oetf, zero_ps, one_ps, scale);
      let b_u32 = clamp_scale_to_u32x16(b_oetf, zero_ps, one_ps, scale);
      let r_u16 = narrow_u32x16_to_u16x16(r_u32);
      let g_u16 = narrow_u32x16_to_u16x16(g_u32);
      let b_u16 = narrow_u32x16_to_u16x16(b_u32);
      let mut tmp_r = [0u16; 16];
      let mut tmp_g = [0u16; 16];
      let mut tmp_b = [0u16; 16];
      _mm256_storeu_si256(tmp_r.as_mut_ptr() as *mut __m256i, r_u16);
      _mm256_storeu_si256(tmp_g.as_mut_ptr() as *mut __m256i, g_u16);
      _mm256_storeu_si256(tmp_b.as_mut_ptr() as *mut __m256i, b_u16);
      let dst = rgb_out.as_mut_ptr().add(x * 3);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 3) = tmp_r[i];
        *dst.add(i * 3 + 1) = tmp_g[i];
        *dst.add(i * 3 + 2) = tmp_b[i];
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgb_u16_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed u16 RGBA (alpha = `0xFFFF`).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xyz12_to_rgba_u16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = _mm512_setzero_ps();
    let one_ps = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_16px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar16(r_lin);
      let g_oetf = oetf_srgb_scalar16(g_lin);
      let b_oetf = oetf_srgb_scalar16(b_lin);
      let r_u32 = clamp_scale_to_u32x16(r_oetf, zero_ps, one_ps, scale);
      let g_u32 = clamp_scale_to_u32x16(g_oetf, zero_ps, one_ps, scale);
      let b_u32 = clamp_scale_to_u32x16(b_oetf, zero_ps, one_ps, scale);
      let r_u16 = narrow_u32x16_to_u16x16(r_u32);
      let g_u16 = narrow_u32x16_to_u16x16(g_u32);
      let b_u16 = narrow_u32x16_to_u16x16(b_u32);
      let mut tmp_r = [0u16; 16];
      let mut tmp_g = [0u16; 16];
      let mut tmp_b = [0u16; 16];
      _mm256_storeu_si256(tmp_r.as_mut_ptr() as *mut __m256i, r_u16);
      _mm256_storeu_si256(tmp_g.as_mut_ptr() as *mut __m256i, g_u16);
      _mm256_storeu_si256(tmp_b.as_mut_ptr() as *mut __m256i, b_u16);
      let dst = rgba_out.as_mut_ptr().add(x * 4);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 4) = tmp_r[i];
        *dst.add(i * 4 + 1) = tmp_g[i];
        *dst.add(i * 4 + 2) = tmp_b[i];
        *dst.add(i * 4 + 3) = 0xFFFF;
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgba_u16_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgba_out[x * 4..width * 4],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed linear RGB f32 (lossless: matrix only).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xyz12_to_rgb_f32_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [f32],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_16px::<BE>(p, &m);
      let mut rb = [0.0_f32; 16];
      let mut gb = [0.0_f32; 16];
      let mut bb = [0.0_f32; 16];
      _mm512_storeu_ps(rb.as_mut_ptr(), r_lin);
      _mm512_storeu_ps(gb.as_mut_ptr(), g_lin);
      _mm512_storeu_ps(bb.as_mut_ptr(), b_lin);
      let dst = rgb_out.as_mut_ptr().add(x * 3);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 3) = rb[i];
        *dst.add(i * 3 + 1) = gb[i];
        *dst.add(i * 3 + 2) = bb[i];
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgb_f32_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed linear XYZ f32 (step 1 only).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW.
/// 2. `xyz.len() >= width * 3`; `xyz_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xyz12_to_xyz_f32_row<const BE: bool>(
  xyz: &[u16],
  xyz_out: &mut [f32],
  width: usize,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(xyz_out.len() >= width * 3, "xyz_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (xv, yv, zv) = load_xyz_linear_16px::<BE>(p);
      let mut xb = [0.0_f32; 16];
      let mut yb = [0.0_f32; 16];
      let mut zb = [0.0_f32; 16];
      _mm512_storeu_ps(xb.as_mut_ptr(), xv);
      _mm512_storeu_ps(yb.as_mut_ptr(), yv);
      _mm512_storeu_ps(zb.as_mut_ptr(), zv);
      let dst = xyz_out.as_mut_ptr().add(x * 3);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 3) = xb[i];
        *dst.add(i * 3 + 1) = yb[i];
        *dst.add(i * 3 + 2) = zb[i];
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_xyz_f32_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut xyz_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// XYZ12 → packed f16 RGB.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xyz12_to_rgb_f16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [half::f16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = _mm512_setzero_ps();
    let one_ps = _mm512_set1_ps(1.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_16px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar16(r_lin);
      let g_oetf = oetf_srgb_scalar16(g_lin);
      let b_oetf = oetf_srgb_scalar16(b_lin);
      let r_clamp = _mm512_min_ps(_mm512_max_ps(r_oetf, zero_ps), one_ps);
      let g_clamp = _mm512_min_ps(_mm512_max_ps(g_oetf, zero_ps), one_ps);
      let b_clamp = _mm512_min_ps(_mm512_max_ps(b_oetf, zero_ps), one_ps);
      let mut rb = [0.0_f32; 16];
      let mut gb = [0.0_f32; 16];
      let mut bb = [0.0_f32; 16];
      _mm512_storeu_ps(rb.as_mut_ptr(), r_clamp);
      _mm512_storeu_ps(gb.as_mut_ptr(), g_clamp);
      _mm512_storeu_ps(bb.as_mut_ptr(), b_clamp);
      for i in 0..PIXELS_PER_ITER {
        let oi = (x + i) * 3;
        rgb_out[oi] = half::f16::from_f32(rb[i]);
        rgb_out[oi + 1] = half::f16::from_f32(gb[i]);
        rgb_out[oi + 2] = half::f16::from_f32(bb[i]);
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgb_f16_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed f16 RGBA (alpha = `1.0`).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xyz12_to_rgba_f16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [half::f16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  let one_f16 = half::f16::from_f32(1.0);

  unsafe {
    let zero_ps = _mm512_setzero_ps();
    let one_ps = _mm512_set1_ps(1.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_16px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar16(r_lin);
      let g_oetf = oetf_srgb_scalar16(g_lin);
      let b_oetf = oetf_srgb_scalar16(b_lin);
      let r_clamp = _mm512_min_ps(_mm512_max_ps(r_oetf, zero_ps), one_ps);
      let g_clamp = _mm512_min_ps(_mm512_max_ps(g_oetf, zero_ps), one_ps);
      let b_clamp = _mm512_min_ps(_mm512_max_ps(b_oetf, zero_ps), one_ps);
      let mut rb = [0.0_f32; 16];
      let mut gb = [0.0_f32; 16];
      let mut bb = [0.0_f32; 16];
      _mm512_storeu_ps(rb.as_mut_ptr(), r_clamp);
      _mm512_storeu_ps(gb.as_mut_ptr(), g_clamp);
      _mm512_storeu_ps(bb.as_mut_ptr(), b_clamp);
      for i in 0..PIXELS_PER_ITER {
        let oi = (x + i) * 4;
        rgba_out[oi] = half::f16::from_f32(rb[i]);
        rgba_out[oi + 1] = half::f16::from_f32(gb[i]);
        rgba_out[oi + 2] = half::f16::from_f32(bb[i]);
        rgba_out[oi + 3] = one_f16;
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgba_f16_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgba_out[x * 4..width * 4],
        width - x,
        target_gamut,
      );
    }
  }
}
