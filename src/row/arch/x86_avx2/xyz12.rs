//! AVX2 kernels for the Tier 12 (DCP / `Xyz12`) source.
//!
//! Each kernel processes **8 pixels per SIMD iteration** in `__m256`
//! (256-bit, 8 f32 lanes) registers. The matmul, integer narrow, and
//! interleaved store are vectorized; the SMPTE 428-1 inverse OETF and
//! sRGB-shape forward OETF run **scalar per lane** via the same
//! scalar functions the reference kernel uses, preserving the 0-ULP
//! scalar↔SIMD parity contract by construction.
//!
//! Pipeline:
//! 1. Three `__m128i` loads via SSE4.1's `load_endian_u16x8::<BE>`
//!    feed the 3-channel deinterleave shuffle (the same byte-shuffle
//!    pattern the SSE4.1 backend uses, since AVX2 reuses
//!    `_mm_shuffle_epi8` per 128-bit lane). The result is three
//!    `__m128i` channel vectors (8 u16 each).
//! 2. Each `(X8, Y8, Z8)` is right-shifted by 4 (`_mm_srli_epi16::<4>`)
//!    to extract the active 12-bit code from the high-bit-packed `u16`
//!    (FFmpeg `AV_PIX_FMT_XYZ12LE/BE`: code in `[15:4]`, low 4 bits
//!    zero), defensively masked (`_mm_and_si128`), widened to
//!    `__m256i` u32x8 via `_mm256_cvtepu16_epi32`, then cast to
//!    `__m256` via `_mm256_cvtepi32_ps`.
//! 3. Per-lane scalar `smpte428_inverse_oetf` produces linear XYZ as
//!    `__m256`.
//! 4. Vectorized 3×3 matmul uses plain `_mm256_mul_ps + _mm256_add_ps`
//!    (NOT FMA via `_mm256_fmadd_ps`) — single-rounding FMA breaks the
//!    0-ULP parity contract on integer-narrow output paths.
//! 5. Per-lane scalar `oetf_srgb` (only for u8 / u16 / f16 outputs).
//! 6. Clamp `[0, 1]` × `scale` + `+0.5` truncate via
//!    `_mm256_cvttps_epi32` (round-half-up matches scalar's
//!    `(c * scale + 0.5) as int`); saturate to u16 via
//!    `_mm256_packus_epi32` followed by `_mm256_permute4x64_epi64`
//!    to undo per-128-bit-lane reordering.
//!
//! Width remainder (`width % 8`) is handled by the scalar reference
//! kernel.
//!
//! # Numerical contract
//!
//! Bit-identical scalar↔SIMD output across all integer / f16 / f32
//! paths (verified by `tests::xyz12`).

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use super::super::{
  x86_common::{write_rgb_u8_8, write_rgba_u8_8},
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

const PIXELS_PER_ITER: usize = 8;
const SAMPLE_MASK_U16: u16 = 0x0FFF;

// ---- Internal helpers --------------------------------------------------

/// Deinterleaves 3 `__m128i` registers (24 u16 = 8 packed XYZ pixels)
/// into `(X8, Y8, Z8)` u16x8 channel vectors. Same shuffle pattern as
/// the SSE4.1 backend's `deinterleave_xyz12_8px` — kept inline here
/// rather than imported across module boundaries.
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

/// Widens a u16x8 channel vector (already 12-bit-masked) to f32x8.
/// `_mm256_cvtepu16_epi32` zero-extends u16 → i32 (no sign issue
/// because samples are < 4096); `_mm256_cvtepi32_ps` casts to f32.
#[inline(always)]
unsafe fn u16x8_to_f32x8(v: __m128i) -> __m256 {
  unsafe { _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v)) }
}

/// Per-lane scalar SMPTE 428-1 inverse OETF on an `__m256` (8 lanes).
#[inline(always)]
unsafe fn smpte428_inv_oetf_scalar8(v: __m256) -> __m256 {
  unsafe {
    let mut buf = [0.0_f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), v);
    for slot in &mut buf {
      *slot = smpte428_inverse_oetf(*slot as u16);
    }
    _mm256_loadu_ps(buf.as_ptr())
  }
}

/// Per-lane scalar sRGB OETF on an `__m256` (8 lanes).
#[inline(always)]
unsafe fn oetf_srgb_scalar8(v: __m256) -> __m256 {
  unsafe {
    let mut buf = [0.0_f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), v);
    for slot in &mut buf {
      *slot = oetf_srgb(*slot);
    }
    _mm256_loadu_ps(buf.as_ptr())
  }
}

/// Vectorized 3×3 matmul on an 8-lane f32 vector. Plain mul + add
/// (NOT FMA) — see `super::super::neon::xyz12::matmul_xyz_to_rgb`
/// docstring for the rounding-schedule rationale.
#[inline(always)]
unsafe fn matmul_xyz_to_rgb_8lane(
  m: &[[f32; 3]; 3],
  x: __m256,
  y: __m256,
  z: __m256,
) -> (__m256, __m256, __m256) {
  unsafe {
    let m00 = _mm256_set1_ps(m[0][0]);
    let m01 = _mm256_set1_ps(m[0][1]);
    let m02 = _mm256_set1_ps(m[0][2]);
    let m10 = _mm256_set1_ps(m[1][0]);
    let m11 = _mm256_set1_ps(m[1][1]);
    let m12 = _mm256_set1_ps(m[1][2]);
    let m20 = _mm256_set1_ps(m[2][0]);
    let m21 = _mm256_set1_ps(m[2][1]);
    let m22 = _mm256_set1_ps(m[2][2]);
    let r = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(m00, x), _mm256_mul_ps(m01, y)),
      _mm256_mul_ps(m02, z),
    );
    let g = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(m10, x), _mm256_mul_ps(m11, y)),
      _mm256_mul_ps(m12, z),
    );
    let b = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(m20, x), _mm256_mul_ps(m21, y)),
      _mm256_mul_ps(m22, z),
    );
    (r, g, b)
  }
}

/// Loads 8 XYZ12 pixels and produces (R, G, B) `__m256` after the
/// inverse-OETF + 3×3 matmul.
#[inline(always)]
unsafe fn load_and_matmul_8px<const BE: bool>(
  p: *const u8,
  m: &[[f32; 3]; 3],
) -> (__m256, __m256, __m256) {
  unsafe {
    let v0 = load_endian_u16x8::<BE>(p);
    let v1 = load_endian_u16x8::<BE>(p.add(16));
    let v2 = load_endian_u16x8::<BE>(p.add(32));
    let (x_u, y_u, z_u) = deinterleave_xyz12_8px(v0, v1, v2);
    // Shift right 4 to extract active 12-bit code (high-bit-packed
    // FFmpeg `AV_PIX_FMT_XYZ12LE/BE`); mask defensively.
    let mask = _mm_set1_epi16(SAMPLE_MASK_U16 as i16);
    let x_shr = _mm_and_si128(_mm_srli_epi16::<4>(x_u), mask);
    let y_shr = _mm_and_si128(_mm_srli_epi16::<4>(y_u), mask);
    let z_shr = _mm_and_si128(_mm_srli_epi16::<4>(z_u), mask);
    let x_lin = smpte428_inv_oetf_scalar8(u16x8_to_f32x8(x_shr));
    let y_lin = smpte428_inv_oetf_scalar8(u16x8_to_f32x8(y_shr));
    let z_lin = smpte428_inv_oetf_scalar8(u16x8_to_f32x8(z_shr));
    matmul_xyz_to_rgb_8lane(m, x_lin, y_lin, z_lin)
  }
}

/// Loads 8 XYZ12 pixels and produces (X, Y, Z) `__m256` linear-XYZ —
/// step 1 only, no matmul.
#[inline(always)]
unsafe fn load_xyz_linear_8px<const BE: bool>(p: *const u8) -> (__m256, __m256, __m256) {
  unsafe {
    let v0 = load_endian_u16x8::<BE>(p);
    let v1 = load_endian_u16x8::<BE>(p.add(16));
    let v2 = load_endian_u16x8::<BE>(p.add(32));
    let (x_u, y_u, z_u) = deinterleave_xyz12_8px(v0, v1, v2);
    let mask = _mm_set1_epi16(SAMPLE_MASK_U16 as i16);
    let x_shr = _mm_and_si128(_mm_srli_epi16::<4>(x_u), mask);
    let y_shr = _mm_and_si128(_mm_srli_epi16::<4>(y_u), mask);
    let z_shr = _mm_and_si128(_mm_srli_epi16::<4>(z_u), mask);
    let x_lin = smpte428_inv_oetf_scalar8(u16x8_to_f32x8(x_shr));
    let y_lin = smpte428_inv_oetf_scalar8(u16x8_to_f32x8(y_shr));
    let z_lin = smpte428_inv_oetf_scalar8(u16x8_to_f32x8(z_shr));
    (x_lin, y_lin, z_lin)
  }
}

/// Vectorized clamp `[0, 1]` × scale + `+0.5` truncate, returning a
/// u32x8 (`__m256i`) ready for saturating narrow.
#[inline(always)]
unsafe fn clamp_scale_to_u32x8(v: __m256, zero: __m256, one: __m256, scale: __m256) -> __m256i {
  unsafe {
    let half = _mm256_set1_ps(0.5);
    let clamped = _mm256_min_ps(_mm256_max_ps(v, zero), one);
    let scaled = _mm256_add_ps(_mm256_mul_ps(clamped, scale), half);
    _mm256_cvttps_epi32(scaled)
  }
}

/// Narrow a u32x8 → u16x8 via `_mm256_packus_epi32` + lane fixup.
///
/// `_mm256_packus_epi32` packs per-128-bit-lane (low-pair from each
/// lane → low half, high-pair → high half), so the lane-split output
/// has channels {0..3, 8..11, 4..7, 12..15} pre-permute. We pack with
/// itself and read the low 128 bits as the natural `[0..7]` channel.
#[inline(always)]
unsafe fn narrow_u32x8_to_u16x8(lo: __m256i) -> __m128i {
  unsafe {
    // Pack with self → 16 u16 lanes (duplicated). Extract the low 128.
    let packed = _mm256_packus_epi32(lo, lo);
    // Lane fixup so the natural `[0..7]` u16 order falls in lane 0.
    let permuted = _mm256_permute4x64_epi64::<0xD8>(packed);
    _mm256_castsi256_si128(permuted)
  }
}

// ---- Per-output kernels ------------------------------------------------

/// XYZ12 → packed u8 RGB (8 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero_ps = _mm256_setzero_ps();
    let one_ps = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);
    let zero_si = _mm_setzero_si128();
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_8px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar8(r_lin);
      let g_oetf = oetf_srgb_scalar8(g_lin);
      let b_oetf = oetf_srgb_scalar8(b_lin);
      let r_u32 = clamp_scale_to_u32x8(r_oetf, zero_ps, one_ps, scale);
      let g_u32 = clamp_scale_to_u32x8(g_oetf, zero_ps, one_ps, scale);
      let b_u32 = clamp_scale_to_u32x8(b_oetf, zero_ps, one_ps, scale);
      let r_u16 = narrow_u32x8_to_u16x8(r_u32);
      let g_u16 = narrow_u32x8_to_u16x8(g_u32);
      let b_u16 = narrow_u32x8_to_u16x8(b_u32);
      let r_u8 = _mm_packus_epi16(r_u16, zero_si);
      let g_u8 = _mm_packus_epi16(g_u16, zero_si);
      let b_u8 = _mm_packus_epi16(b_u16, zero_si);
      // In-register 8-pixel RGB interleave + 16-byte + 8-byte stores
      // avoid a 3× stack-temp + scalar-scatter loop.
      write_rgb_u8_8(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
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
/// 1. AVX2 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero_ps = _mm256_setzero_ps();
    let one_ps = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);
    let zero_si = _mm_setzero_si128();
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_8px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar8(r_lin);
      let g_oetf = oetf_srgb_scalar8(g_lin);
      let b_oetf = oetf_srgb_scalar8(b_lin);
      let r_u32 = clamp_scale_to_u32x8(r_oetf, zero_ps, one_ps, scale);
      let g_u32 = clamp_scale_to_u32x8(g_oetf, zero_ps, one_ps, scale);
      let b_u32 = clamp_scale_to_u32x8(b_oetf, zero_ps, one_ps, scale);
      let r_u16 = narrow_u32x8_to_u16x8(r_u32);
      let g_u16 = narrow_u32x8_to_u16x8(g_u32);
      let b_u16 = narrow_u32x8_to_u16x8(b_u32);
      let r_u8 = _mm_packus_epi16(r_u16, zero_si);
      let g_u8 = _mm_packus_epi16(g_u16, zero_si);
      let b_u8 = _mm_packus_epi16(b_u16, zero_si);
      // Alpha = 0xFF in every lane (only the low 8 bytes are read by
      // `write_rgba_u8_8`'s `unpacklo_epi8`).
      let a_u8 = _mm_set1_epi8(-1_i8);
      // In-register 8-pixel RGBA interleave via two unpack stages +
      // two 16-byte stores avoids a 3× stack-temp + scalar-scatter loop.
      write_rgba_u8_8(r_u8, g_u8, b_u8, a_u8, rgba_out.as_mut_ptr().add(x * 4));
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
/// 1. AVX2 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero_ps = _mm256_setzero_ps();
    let one_ps = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_8px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar8(r_lin);
      let g_oetf = oetf_srgb_scalar8(g_lin);
      let b_oetf = oetf_srgb_scalar8(b_lin);
      let r_u32 = clamp_scale_to_u32x8(r_oetf, zero_ps, one_ps, scale);
      let g_u32 = clamp_scale_to_u32x8(g_oetf, zero_ps, one_ps, scale);
      let b_u32 = clamp_scale_to_u32x8(b_oetf, zero_ps, one_ps, scale);
      let r_u16 = narrow_u32x8_to_u16x8(r_u32);
      let g_u16 = narrow_u32x8_to_u16x8(g_u32);
      let b_u16 = narrow_u32x8_to_u16x8(b_u32);
      let mut tmp_r = [0u16; 8];
      let mut tmp_g = [0u16; 8];
      let mut tmp_b = [0u16; 8];
      _mm_storeu_si128(tmp_r.as_mut_ptr() as *mut __m128i, r_u16);
      _mm_storeu_si128(tmp_g.as_mut_ptr() as *mut __m128i, g_u16);
      _mm_storeu_si128(tmp_b.as_mut_ptr() as *mut __m128i, b_u16);
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
/// 1. AVX2 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero_ps = _mm256_setzero_ps();
    let one_ps = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_8px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar8(r_lin);
      let g_oetf = oetf_srgb_scalar8(g_lin);
      let b_oetf = oetf_srgb_scalar8(b_lin);
      let r_u32 = clamp_scale_to_u32x8(r_oetf, zero_ps, one_ps, scale);
      let g_u32 = clamp_scale_to_u32x8(g_oetf, zero_ps, one_ps, scale);
      let b_u32 = clamp_scale_to_u32x8(b_oetf, zero_ps, one_ps, scale);
      let r_u16 = narrow_u32x8_to_u16x8(r_u32);
      let g_u16 = narrow_u32x8_to_u16x8(g_u32);
      let b_u16 = narrow_u32x8_to_u16x8(b_u32);
      let mut tmp_r = [0u16; 8];
      let mut tmp_g = [0u16; 8];
      let mut tmp_b = [0u16; 8];
      _mm_storeu_si128(tmp_r.as_mut_ptr() as *mut __m128i, r_u16);
      _mm_storeu_si128(tmp_g.as_mut_ptr() as *mut __m128i, g_u16);
      _mm_storeu_si128(tmp_b.as_mut_ptr() as *mut __m128i, b_u16);
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

/// XYZ12 → packed linear RGB f32 (lossless: matrix only, no OETF).
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
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
      let (r_lin, g_lin, b_lin) = load_and_matmul_8px::<BE>(p, &m);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      _mm256_storeu_ps(rb.as_mut_ptr(), r_lin);
      _mm256_storeu_ps(gb.as_mut_ptr(), g_lin);
      _mm256_storeu_ps(bb.as_mut_ptr(), b_lin);
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
/// 1. AVX2 must be available.
/// 2. `xyz.len() >= width * 3`; `xyz_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
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
      let (xv, yv, zv) = load_xyz_linear_8px::<BE>(p);
      let mut xb = [0.0_f32; 8];
      let mut yb = [0.0_f32; 8];
      let mut zb = [0.0_f32; 8];
      _mm256_storeu_ps(xb.as_mut_ptr(), xv);
      _mm256_storeu_ps(yb.as_mut_ptr(), yv);
      _mm256_storeu_ps(zb.as_mut_ptr(), zv);
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
/// 1. AVX2 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero_ps = _mm256_setzero_ps();
    let one_ps = _mm256_set1_ps(1.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_8px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar8(r_lin);
      let g_oetf = oetf_srgb_scalar8(g_lin);
      let b_oetf = oetf_srgb_scalar8(b_lin);
      let r_clamp = _mm256_min_ps(_mm256_max_ps(r_oetf, zero_ps), one_ps);
      let g_clamp = _mm256_min_ps(_mm256_max_ps(g_oetf, zero_ps), one_ps);
      let b_clamp = _mm256_min_ps(_mm256_max_ps(b_oetf, zero_ps), one_ps);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      _mm256_storeu_ps(rb.as_mut_ptr(), r_clamp);
      _mm256_storeu_ps(gb.as_mut_ptr(), g_clamp);
      _mm256_storeu_ps(bb.as_mut_ptr(), b_clamp);
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
/// 1. AVX2 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero_ps = _mm256_setzero_ps();
    let one_ps = _mm256_set1_ps(1.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let (r_lin, g_lin, b_lin) = load_and_matmul_8px::<BE>(p, &m);
      let r_oetf = oetf_srgb_scalar8(r_lin);
      let g_oetf = oetf_srgb_scalar8(g_lin);
      let b_oetf = oetf_srgb_scalar8(b_lin);
      let r_clamp = _mm256_min_ps(_mm256_max_ps(r_oetf, zero_ps), one_ps);
      let g_clamp = _mm256_min_ps(_mm256_max_ps(g_oetf, zero_ps), one_ps);
      let b_clamp = _mm256_min_ps(_mm256_max_ps(b_oetf, zero_ps), one_ps);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      _mm256_storeu_ps(rb.as_mut_ptr(), r_clamp);
      _mm256_storeu_ps(gb.as_mut_ptr(), g_clamp);
      _mm256_storeu_ps(bb.as_mut_ptr(), b_clamp);
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
