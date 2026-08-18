//! SSE4.1 kernels for the Tier 12 (DCP / `Xyz12`) source.
//!
//! Each kernel processes **8 pixels per SIMD iteration**. The matmul,
//! integer narrow, and interleaved store are vectorized; the SMPTE
//! 428-1 inverse OETF and sRGB-shape forward OETF run **scalar per
//! lane** via the same scalar functions the reference kernel uses,
//! preserving the 0-ULP scalar↔SIMD parity contract by construction.
//!
//! Pipeline:
//! 1. Three `load_endian_u16x8::<BE>` loads (3 × 16 B = 48 B = 8 px ×
//!    XYZ in 24 u16) feed the 3-channel deinterleave shuffle. The
//!    shuffle pattern matches the established
//!    `deinterleave_rgb48_8px` layout from the Tier 8 kernels.
//! 2. Each `(X8, Y8, Z8)` vector is right-shifted by 4
//!    (`_mm_srli_epi16::<4>`) to extract the active 12-bit code from
//!    the high-bit-packed `u16` (FFmpeg `AV_PIX_FMT_XYZ12LE/BE`:
//!    active 12 bits in `[15:4]`, low 4 bits reserved zero), then
//!    defensively masked via `_mm_and_si128` with `SAMPLE_MASK`. The
//!    result is split into low/high u32x4 halves
//!    (`_mm_unpacklo/hi_epi16`) and converted to `f32x4` via
//!    `_mm_cvtepi32_ps` (the masked u16 fits in the i32 positive
//!    range, so signed and unsigned conversions agree).
//! 3. Per-lane scalar `smpte428_inverse_oetf` produces the linear XYZ
//!    `f32x4` halves.
//! 4. Vectorized 3×3 matmul uses plain `_mm_mul_ps + _mm_add_ps`
//!    (NOT FMA) — single-rounding FMA breaks the 0-ULP parity
//!    contract on integer-narrow output paths, same as the NEON
//!    backend.
//! 5. Per-lane scalar `oetf_srgb` (only for u8 / u16 / f16 outputs).
//! 6. Clamp `[0, 1]` × `scale` + `+ 0.5` truncate via
//!    `_mm_cvttps_epi32` (round-half-up matches scalar's
//!    `(c * scale + 0.5) as int`); saturate to u16 with
//!    `_mm_packus_epi32` and to u8 with `_mm_packus_epi16`. Writes
//!    use the per-format helpers from the Tier 8 family.
//!
//! Width remainder (`width % 8`) is handled by the scalar reference
//! `scalar::xyz12::xyz12_to_*_row::<BE>`.
//!
//! # Numerical contract
//!
//! Bit-identical scalar↔SIMD output across all integer / f16 / f32
//! paths (verified by `tests::xyz12`).

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use super::{
  super::x86_common::{write_rgb_u8_8, write_rgba_u8_8},
  endian::load_endian_u16x8,
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
/// into `(X8, Y8, Z8)` u16x8 channel vectors.
///
/// Mirrors the byte-shuffle pattern of `deinterleave_rgb48_8px` from
/// the Tier 8 packed-RGB-16-bit family. Each output channel pulls
/// from all 3 input registers via `_mm_shuffle_epi8`; the `_mm_or_si128`
/// merge folds the three masked partial vectors.
#[inline(always)]
unsafe fn deinterleave_xyz12_8px(
  v0: __m128i,
  v1: __m128i,
  v2: __m128i,
) -> (__m128i, __m128i, __m128i) {
  unsafe {
    // ---- ch0 (X) -------------------------------------------------
    // From v0: u16 positions 0, 3, 6 → output positions 0, 1, 2.
    let x_v0 = _mm_setr_epi8(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let x_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
    let x_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
    let x = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, x_v0), _mm_shuffle_epi8(v1, x_v1)),
      _mm_shuffle_epi8(v2, x_v2),
    );

    // ---- ch1 (Y) -------------------------------------------------
    let y_v0 = _mm_setr_epi8(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
    let y_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
    let y = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, y_v0), _mm_shuffle_epi8(v1, y_v1)),
      _mm_shuffle_epi8(v2, y_v2),
    );

    // ---- ch2 (Z) -------------------------------------------------
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

/// Splits a u16x8 channel vector (already 12-bit-masked) into low/high
/// f32x4 halves via zero-extending unpack + signed-i32 → f32 cvt
/// (`_mm_cvtepi32_ps`). The masked value `< 4096` fits in positive
/// i32, so signed/unsigned cvt agree.
#[inline(always)]
unsafe fn u16x8_to_f32x4_pair(v: __m128i) -> (__m128, __m128) {
  unsafe {
    let zero = _mm_setzero_si128();
    let lo_u32 = _mm_unpacklo_epi16(v, zero);
    let hi_u32 = _mm_unpackhi_epi16(v, zero);
    (_mm_cvtepi32_ps(lo_u32), _mm_cvtepi32_ps(hi_u32))
  }
}

/// Per-lane scalar SMPTE 428-1 inverse OETF on an `__m128` (4 lanes).
/// Stores to a stack array, calls the scalar function 4 times, reloads.
#[inline(always)]
unsafe fn smpte428_inv_oetf_scalar4(v: __m128) -> __m128 {
  unsafe {
    let mut buf = [0.0_f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), v);
    for slot in &mut buf {
      *slot = smpte428_inverse_oetf(*slot as u16);
    }
    _mm_loadu_ps(buf.as_ptr())
  }
}

/// Per-lane scalar sRGB OETF on an `__m128` (4 lanes).
#[inline(always)]
unsafe fn oetf_srgb_scalar4(v: __m128) -> __m128 {
  unsafe {
    let mut buf = [0.0_f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), v);
    for slot in &mut buf {
      *slot = oetf_srgb(*slot);
    }
    _mm_loadu_ps(buf.as_ptr())
  }
}

/// Vectorized 3×3 matmul on a 4-lane f32 vector: `[R G B]^T = M ·
/// [X Y Z]^T`. Plain mul + add (NOT FMA) — see the NEON
/// implementation's matmul_xyz_to_rgb docstring.
#[inline(always)]
unsafe fn matmul_xyz_to_rgb_4lane(
  m: &[[f32; 3]; 3],
  x: __m128,
  y: __m128,
  z: __m128,
) -> (__m128, __m128, __m128) {
  unsafe {
    let m00 = _mm_set1_ps(m[0][0]);
    let m01 = _mm_set1_ps(m[0][1]);
    let m02 = _mm_set1_ps(m[0][2]);
    let m10 = _mm_set1_ps(m[1][0]);
    let m11 = _mm_set1_ps(m[1][1]);
    let m12 = _mm_set1_ps(m[1][2]);
    let m20 = _mm_set1_ps(m[2][0]);
    let m21 = _mm_set1_ps(m[2][1]);
    let m22 = _mm_set1_ps(m[2][2]);
    let r = _mm_add_ps(
      _mm_add_ps(_mm_mul_ps(m00, x), _mm_mul_ps(m01, y)),
      _mm_mul_ps(m02, z),
    );
    let g = _mm_add_ps(
      _mm_add_ps(_mm_mul_ps(m10, x), _mm_mul_ps(m11, y)),
      _mm_mul_ps(m12, z),
    );
    let b = _mm_add_ps(
      _mm_add_ps(_mm_mul_ps(m20, x), _mm_mul_ps(m21, y)),
      _mm_mul_ps(m22, z),
    );
    (r, g, b)
  }
}

/// Loads 8 XYZ12 pixels and produces the linear RGB f32 lanes after
/// the inverse-OETF + 3×3 matmul.
///
/// Returns six `__m128` vectors: low/high halves of R, G, B (in that
/// order).
#[inline(always)]
unsafe fn load_and_matmul_8px<const BE: bool>(
  p: *const u8,
  m: &[[f32; 3]; 3],
) -> ((__m128, __m128), (__m128, __m128), (__m128, __m128)) {
  unsafe {
    let v0 = load_endian_u16x8::<BE>(p);
    let v1 = load_endian_u16x8::<BE>(p.add(16));
    let v2 = load_endian_u16x8::<BE>(p.add(32));
    let (x_u, y_u, z_u) = deinterleave_xyz12_8px(v0, v1, v2);
    // Shift right 4 to extract the active 12-bit code from the
    // high-bit-packed u16 (FFmpeg `AV_PIX_FMT_XYZ12LE/BE`: code in
    // `[15:4]`, low 4 bits zero). Defensive `& SAMPLE_MASK` is a
    // no-op for spec-compliant input but tolerates a producer that
    // sets bits above 15 (impossible in u16 — included for symmetry
    // with the scalar's `(raw >> 4) & SAMPLE_MASK` decode).
    let mask = _mm_set1_epi16(SAMPLE_MASK_U16 as i16);
    let x_shr = _mm_and_si128(_mm_srli_epi16::<4>(x_u), mask);
    let y_shr = _mm_and_si128(_mm_srli_epi16::<4>(y_u), mask);
    let z_shr = _mm_and_si128(_mm_srli_epi16::<4>(z_u), mask);
    let (x_lo, x_hi) = u16x8_to_f32x4_pair(x_shr);
    let (y_lo, y_hi) = u16x8_to_f32x4_pair(y_shr);
    let (z_lo, z_hi) = u16x8_to_f32x4_pair(z_shr);
    let x_lo = smpte428_inv_oetf_scalar4(x_lo);
    let x_hi = smpte428_inv_oetf_scalar4(x_hi);
    let y_lo = smpte428_inv_oetf_scalar4(y_lo);
    let y_hi = smpte428_inv_oetf_scalar4(y_hi);
    let z_lo = smpte428_inv_oetf_scalar4(z_lo);
    let z_hi = smpte428_inv_oetf_scalar4(z_hi);
    let (r_lo, g_lo, b_lo) = matmul_xyz_to_rgb_4lane(m, x_lo, y_lo, z_lo);
    let (r_hi, g_hi, b_hi) = matmul_xyz_to_rgb_4lane(m, x_hi, y_hi, z_hi);
    ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi))
  }
}

/// Loads 8 XYZ12 pixels and produces 6 `__m128` linear-XYZ f32 vectors
/// (low/high halves of X, Y, Z) — step 1 only, no matmul.
#[inline(always)]
unsafe fn load_xyz_linear_8px<const BE: bool>(
  p: *const u8,
) -> ((__m128, __m128), (__m128, __m128), (__m128, __m128)) {
  unsafe {
    let v0 = load_endian_u16x8::<BE>(p);
    let v1 = load_endian_u16x8::<BE>(p.add(16));
    let v2 = load_endian_u16x8::<BE>(p.add(32));
    let (x_u, y_u, z_u) = deinterleave_xyz12_8px(v0, v1, v2);
    let mask = _mm_set1_epi16(SAMPLE_MASK_U16 as i16);
    let x_shr = _mm_and_si128(_mm_srli_epi16::<4>(x_u), mask);
    let y_shr = _mm_and_si128(_mm_srli_epi16::<4>(y_u), mask);
    let z_shr = _mm_and_si128(_mm_srli_epi16::<4>(z_u), mask);
    let (x_lo, x_hi) = u16x8_to_f32x4_pair(x_shr);
    let (y_lo, y_hi) = u16x8_to_f32x4_pair(y_shr);
    let (z_lo, z_hi) = u16x8_to_f32x4_pair(z_shr);
    (
      (
        smpte428_inv_oetf_scalar4(x_lo),
        smpte428_inv_oetf_scalar4(x_hi),
      ),
      (
        smpte428_inv_oetf_scalar4(y_lo),
        smpte428_inv_oetf_scalar4(y_hi),
      ),
      (
        smpte428_inv_oetf_scalar4(z_lo),
        smpte428_inv_oetf_scalar4(z_hi),
      ),
    )
  }
}

/// Vectorized clamp `[0, 1]` × scale + `+0.5` truncate, returning a
/// u32x4 ready to feed the saturating `_mm_packus_epi32`.
///
/// Matches the scalar `(c.clamp(0, 1) * scale + 0.5) as int` exactly:
/// clamp via `_mm_min_ps + _mm_max_ps`, scale, add 0.5, truncate via
/// `_mm_cvttps_epi32` (the f32 has been pre-clamped to ≤ scale + 0.5,
/// well below `i32::MAX`).
#[inline(always)]
unsafe fn clamp_scale_to_u32x4(v: __m128, zero: __m128, one: __m128, scale: __m128) -> __m128i {
  unsafe {
    let half = _mm_set1_ps(0.5);
    let clamped = _mm_min_ps(_mm_max_ps(v, zero), one);
    let scaled = _mm_add_ps(_mm_mul_ps(clamped, scale), half);
    _mm_cvttps_epi32(scaled)
  }
}

/// Combine two u32x4 vectors into one u16x8 with saturating narrow.
/// The values are pre-clamped, so saturation is a no-op on valid input.
#[inline(always)]
unsafe fn pack_u32x4_pair_to_u16x8(lo: __m128i, hi: __m128i) -> __m128i {
  unsafe { _mm_packus_epi32(lo, hi) }
}

// ---- Per-output kernels ------------------------------------------------

/// XYZ12 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let zero_ps = _mm_setzero_ps();
    let one_ps = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);
    let zero_si = _mm_setzero_si128();

    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo_i = clamp_scale_to_u32x4(r_lo, zero_ps, one_ps, scale);
      let r_hi_i = clamp_scale_to_u32x4(r_hi, zero_ps, one_ps, scale);
      let g_lo_i = clamp_scale_to_u32x4(g_lo, zero_ps, one_ps, scale);
      let g_hi_i = clamp_scale_to_u32x4(g_hi, zero_ps, one_ps, scale);
      let b_lo_i = clamp_scale_to_u32x4(b_lo, zero_ps, one_ps, scale);
      let b_hi_i = clamp_scale_to_u32x4(b_hi, zero_ps, one_ps, scale);
      let r_u16 = pack_u32x4_pair_to_u16x8(r_lo_i, r_hi_i);
      let g_u16 = pack_u32x4_pair_to_u16x8(g_lo_i, g_hi_i);
      let b_u16 = pack_u32x4_pair_to_u16x8(b_lo_i, b_hi_i);
      // Narrow each u16x8 to u8x8 (low 8 bytes) via packus; merge with zero.
      let r_u8 = _mm_packus_epi16(r_u16, zero_si);
      let g_u8 = _mm_packus_epi16(g_u16, zero_si);
      let b_u8 = _mm_packus_epi16(b_u16, zero_si);
      // In-register 8-pixel RGB interleave via the shared `write_rgb_u8_8`
      // helper avoids a 3× stack-temp + per-pixel scalar scatter.
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
/// 1. SSE4.1 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let zero_ps = _mm_setzero_ps();
    let one_ps = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);
    let zero_si = _mm_setzero_si128();

    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo_i = clamp_scale_to_u32x4(r_lo, zero_ps, one_ps, scale);
      let r_hi_i = clamp_scale_to_u32x4(r_hi, zero_ps, one_ps, scale);
      let g_lo_i = clamp_scale_to_u32x4(g_lo, zero_ps, one_ps, scale);
      let g_hi_i = clamp_scale_to_u32x4(g_hi, zero_ps, one_ps, scale);
      let b_lo_i = clamp_scale_to_u32x4(b_lo, zero_ps, one_ps, scale);
      let b_hi_i = clamp_scale_to_u32x4(b_hi, zero_ps, one_ps, scale);
      let r_u16 = pack_u32x4_pair_to_u16x8(r_lo_i, r_hi_i);
      let g_u16 = pack_u32x4_pair_to_u16x8(g_lo_i, g_hi_i);
      let b_u16 = pack_u32x4_pair_to_u16x8(b_lo_i, b_hi_i);
      let r_u8 = _mm_packus_epi16(r_u16, zero_si);
      let g_u8 = _mm_packus_epi16(g_u16, zero_si);
      let b_u8 = _mm_packus_epi16(b_u16, zero_si);
      let a_u8 = _mm_set1_epi8(-1_i8);
      // In-register 8-pixel RGBA interleave via the shared `write_rgba_u8_8`
      // helper avoids a 3× stack-temp + per-pixel scalar scatter.
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

/// XYZ12 → packed u16 RGB (full-range scaling, ×65535).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let zero_ps = _mm_setzero_ps();
    let one_ps = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo_i = clamp_scale_to_u32x4(r_lo, zero_ps, one_ps, scale);
      let r_hi_i = clamp_scale_to_u32x4(r_hi, zero_ps, one_ps, scale);
      let g_lo_i = clamp_scale_to_u32x4(g_lo, zero_ps, one_ps, scale);
      let g_hi_i = clamp_scale_to_u32x4(g_hi, zero_ps, one_ps, scale);
      let b_lo_i = clamp_scale_to_u32x4(b_lo, zero_ps, one_ps, scale);
      let b_hi_i = clamp_scale_to_u32x4(b_hi, zero_ps, one_ps, scale);
      let r_u16 = pack_u32x4_pair_to_u16x8(r_lo_i, r_hi_i);
      let g_u16 = pack_u32x4_pair_to_u16x8(g_lo_i, g_hi_i);
      let b_u16 = pack_u32x4_pair_to_u16x8(b_lo_i, b_hi_i);
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
/// 1. SSE4.1 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let zero_ps = _mm_setzero_ps();
    let one_ps = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo_i = clamp_scale_to_u32x4(r_lo, zero_ps, one_ps, scale);
      let r_hi_i = clamp_scale_to_u32x4(r_hi, zero_ps, one_ps, scale);
      let g_lo_i = clamp_scale_to_u32x4(g_lo, zero_ps, one_ps, scale);
      let g_hi_i = clamp_scale_to_u32x4(g_hi, zero_ps, one_ps, scale);
      let b_lo_i = clamp_scale_to_u32x4(b_lo, zero_ps, one_ps, scale);
      let b_hi_i = clamp_scale_to_u32x4(b_hi, zero_ps, one_ps, scale);
      let r_u16 = pack_u32x4_pair_to_u16x8(r_lo_i, r_hi_i);
      let g_u16 = pack_u32x4_pair_to_u16x8(g_lo_i, g_hi_i);
      let b_u16 = pack_u32x4_pair_to_u16x8(b_lo_i, b_hi_i);
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

/// XYZ12 → packed linear RGB f32 (lossless: matrix only, no OETF, no
/// clamp).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      _mm_storeu_ps(rb.as_mut_ptr(), r_lo);
      _mm_storeu_ps(rb.as_mut_ptr().add(4), r_hi);
      _mm_storeu_ps(gb.as_mut_ptr(), g_lo);
      _mm_storeu_ps(gb.as_mut_ptr().add(4), g_hi);
      _mm_storeu_ps(bb.as_mut_ptr(), b_lo);
      _mm_storeu_ps(bb.as_mut_ptr().add(4), b_hi);
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
/// 1. SSE4.1 must be available.
/// 2. `xyz.len() >= width * 3`; `xyz_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)) = load_xyz_linear_8px::<BE>(p);
      let mut xb = [0.0_f32; 8];
      let mut yb = [0.0_f32; 8];
      let mut zb = [0.0_f32; 8];
      _mm_storeu_ps(xb.as_mut_ptr(), x_lo);
      _mm_storeu_ps(xb.as_mut_ptr().add(4), x_hi);
      _mm_storeu_ps(yb.as_mut_ptr(), y_lo);
      _mm_storeu_ps(yb.as_mut_ptr().add(4), y_hi);
      _mm_storeu_ps(zb.as_mut_ptr(), z_lo);
      _mm_storeu_ps(zb.as_mut_ptr().add(4), z_hi);
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

/// XYZ12 → packed f16 RGB (gamma-encoded, clamped to `[0, 1]`).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let zero_ps = _mm_setzero_ps();
    let one_ps = _mm_set1_ps(1.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo = _mm_min_ps(_mm_max_ps(r_lo, zero_ps), one_ps);
      let r_hi = _mm_min_ps(_mm_max_ps(r_hi, zero_ps), one_ps);
      let g_lo = _mm_min_ps(_mm_max_ps(g_lo, zero_ps), one_ps);
      let g_hi = _mm_min_ps(_mm_max_ps(g_hi, zero_ps), one_ps);
      let b_lo = _mm_min_ps(_mm_max_ps(b_lo, zero_ps), one_ps);
      let b_hi = _mm_min_ps(_mm_max_ps(b_hi, zero_ps), one_ps);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      _mm_storeu_ps(rb.as_mut_ptr(), r_lo);
      _mm_storeu_ps(rb.as_mut_ptr().add(4), r_hi);
      _mm_storeu_ps(gb.as_mut_ptr(), g_lo);
      _mm_storeu_ps(gb.as_mut_ptr().add(4), g_hi);
      _mm_storeu_ps(bb.as_mut_ptr(), b_lo);
      _mm_storeu_ps(bb.as_mut_ptr().add(4), b_hi);
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
/// 1. SSE4.1 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let zero_ps = _mm_setzero_ps();
    let one_ps = _mm_set1_ps(1.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo = _mm_min_ps(_mm_max_ps(r_lo, zero_ps), one_ps);
      let r_hi = _mm_min_ps(_mm_max_ps(r_hi, zero_ps), one_ps);
      let g_lo = _mm_min_ps(_mm_max_ps(g_lo, zero_ps), one_ps);
      let g_hi = _mm_min_ps(_mm_max_ps(g_hi, zero_ps), one_ps);
      let b_lo = _mm_min_ps(_mm_max_ps(b_lo, zero_ps), one_ps);
      let b_hi = _mm_min_ps(_mm_max_ps(b_hi, zero_ps), one_ps);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      _mm_storeu_ps(rb.as_mut_ptr(), r_lo);
      _mm_storeu_ps(rb.as_mut_ptr().add(4), r_hi);
      _mm_storeu_ps(gb.as_mut_ptr(), g_lo);
      _mm_storeu_ps(gb.as_mut_ptr().add(4), g_hi);
      _mm_storeu_ps(bb.as_mut_ptr(), b_lo);
      _mm_storeu_ps(bb.as_mut_ptr().add(4), b_hi);
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
