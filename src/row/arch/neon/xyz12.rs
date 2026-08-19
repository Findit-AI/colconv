//! NEON kernels for the Tier 12 (DCP / `Xyz12`) source.
//!
//! Each kernel processes **4 pixels per SIMD iteration** in `float32x4_t`
//! lanes. Pipeline steps:
//!
//! 1. `vld3_u16` deinterleaves 12 packed u16 (`X, Y, Z` x 4 pixels) into
//!    three `uint16x4_t` channel vectors. `vld3_u16` materializes each
//!    lane in **host-native** u16 byte order, so the SIMD body only
//!    runs when `BE == HOST_NATIVE_BE`; cross-endian inputs fall back
//!    to the scalar reference (which applies `from_le` / `from_be`).
//!    `vld3_u16` is used in the 4-lane form rather than the 8-lane
//!    `vld3q_u16` so the matmul + scalar-OETF round-trip stays in
//!    `float32x4_t` lanes.
//! 2. Each channel vector is right-shifted by 4 (`vshr_n_u16::<4>`) to
//!    extract the active 12-bit code per the FFmpeg
//!    `AV_PIX_FMT_XYZ12LE/BE` convention (active bits in `[15:4]`,
//!    low 4 bits zero), then defensively masked with `vand_u16` and
//!    `SAMPLE_MASK`. Samples are zero-widened to `u32x4` and converted
//!    to `f32x4` via `vcvtq_f32_u32`.
//! 3. **SMPTE 428-1 §8 inverse OETF** runs scalar per lane via
//!    `smpte428_inverse_oetf` — `f32::powf(c, 2.6)` is subject to the
//!    same f32 hardware floor that ruled out polynomial vectorization
//!    of the sRGB OETF, so a scalar fall-through preserves the 0-ULP
//!    parity contract between scalar and SIMD.
//! 4. **3x3 matmul** to one of three target gamuts is fully vectorized
//!    via `vmulq_f32 + vaddq_f32` chains. We deliberately avoid
//!    `vfmaq_f32` (FMA) — single-rounding semantics would diverge by
//!    up to 0.5 ULP from the scalar's mul-then-add schedule, breaking
//!    the 0-ULP parity contract on integer-narrow output paths.
//! 5. **sRGB-shape OETF** (only for u8 / u16 / f16 outputs) runs scalar
//!    per lane via `oetf_srgb` (192-segment polynomial) — same 0-ULP
//!    parity contract.
//! 6. **Clamp + scale + integer narrow + interleave** are vectorized:
//!    `vminq_f32` / `vmaxq_f32` for `[0, 1]` clamp, then `+ 0.5` followed
//!    by `vcvtq_u32_f32` (truncating cast) — i.e. **round-half-up**, so
//!    the cast matches the scalar reference's `(c x max + 0.5) as T`
//!    (NOT `vcvtnq_u32_f32` round-half-to-even, which would diverge by
//!    1 ULP at exact `*.5` ties). `vqmovn_u32` / `vqmovn_u16` perform
//!    the saturating narrow; `vst3_u8` / `vst3_u16` / `vst4_u8` /
//!    `vst4_u16` perform the interleaved store.
//!
//! Width remainder (`width % 4`) is handled by the scalar reference
//! kernel (`scalar::xyz12::xyz12_to_*_row::<BE>`).
//!
//! # Numerical contract
//!
//! Every f32 computation either matches the scalar reference bit-exact
//! (vectorized matmul uses plain `vmulq_f32 + vaddq_f32`, replicating
//! the scalar's mul-then-add rounding schedule lane-for-lane) or *is*
//! the scalar reference (per-lane OETF calls reuse the scalar
//! `smpte428_inverse_oetf` / `oetf_srgb` directly). The narrow + clamp
//! mirrors the scalar's `(c x max + 0.5)` round-half-up + saturating
//! integer cast.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use super::miri_compat::*;
use crate::{
  KernelGamut,
  row::scalar::{
    self,
    xyz12::{oetf_srgb, smpte428_inverse_oetf},
    xyz12_constants::xyz_to_rgb_matrix,
  },
};

const LANES: usize = 4;
const SAMPLE_MASK_U16: u16 = 0x0FFF;

/// Compile-time host endianness. `vld3_u16` deinterleaves using
/// **host-native** u16 reads, so the SIMD body is only correct when
/// the encoded byte order matches the host. Truth table:
///
/// | wire `BE` | host       | `BE == HOST_NATIVE_BE` | path   | correct via    |
/// |-----------|------------|------------------------|--------|----------------|
/// | false     | LE         | true                   | SIMD   | host-native LE |
/// | false     | BE         | false                  | scalar | `from_le`      |
/// | true      | LE         | false                  | scalar | `from_be`      |
/// | true      | BE         | true                   | SIMD   | host-native BE |
///
/// The previous `if BE { swap }` gate was target-endian-dependent: on
/// BE hosts the LE-encoded input arrived already host-native and was
/// never swapped, while BE-encoded input was swapped a second time.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// ---- Internal helpers --------------------------------------------------

/// Loads 4 packed XYZ12 pixels (12 u16 = `X, Y, Z` x 4) deinterleaved
/// into `(X4, Y4, Z4)` u16x4 channel vectors.
///
/// Caller obligation: `BE == HOST_NATIVE_BE`. `vld3_u16` materializes
/// each lane in host-native u16 order, so the input bytes must already
/// match the host endianness — cross-endian decode is handled by the
/// scalar fallback in each per-output kernel.
#[inline(always)]
unsafe fn load_xyz4(p: *const u16) -> (uint16x4_t, uint16x4_t, uint16x4_t) {
  unsafe {
    let triple = vld3_u16(p);
    (triple.0, triple.1, triple.2)
  }
}

/// Right-shifts each lane of `v` by 4 to recover the active 12-bit
/// XYZ12 code from the high-bit-packed `u16` (FFmpeg
/// `AV_PIX_FMT_XYZ12LE/BE`: active 12 bits in `[15:4]`, low 4 bits
/// reserved zero), then defensively masks with `SAMPLE_MASK` (no-op
/// for spec-compliant input), zero-widens to u32x4, and casts to
/// `f32x4`. Mirrors the scalar `(x_u16 >> 4) & SAMPLE_MASK` decode.
#[inline(always)]
unsafe fn shift_mask_widen_cvt(v: uint16x4_t) -> float32x4_t {
  unsafe {
    let mask = vdup_n_u16(SAMPLE_MASK_U16);
    let shifted = vshr_n_u16::<4>(v);
    let masked = vand_u16(shifted, mask);
    let widened = vmovl_u16(masked); // u32x4
    vcvtq_f32_u32(widened)
  }
}

/// Per-lane scalar SMPTE 428-1 inverse OETF on a `float32x4_t`. Stores
/// to a stack array, calls `smpte428_inverse_oetf` 4 times, reloads.
///
/// Input is the **raw u12 sample value** as f32 (0..=4095) — the
/// scalar function masks again internally and applies `(x/4095)^2.6 /
/// 0.91653`.
#[inline(always)]
unsafe fn smpte428_inv_oetf_scalar4(v: float32x4_t) -> float32x4_t {
  unsafe {
    let mut buf = [0.0_f32; LANES];
    vst1q_f32(buf.as_mut_ptr(), v);
    for slot in &mut buf {
      *slot = smpte428_inverse_oetf(*slot as u16);
    }
    vld1q_f32(buf.as_ptr())
  }
}

/// Per-lane scalar sRGB OETF on a `float32x4_t`. Same gather/scatter
/// pattern as the inverse-OETF helper.
#[inline(always)]
unsafe fn oetf_srgb_scalar4(v: float32x4_t) -> float32x4_t {
  unsafe {
    let mut buf = [0.0_f32; LANES];
    vst1q_f32(buf.as_mut_ptr(), v);
    for slot in &mut buf {
      *slot = oetf_srgb(*slot);
    }
    vld1q_f32(buf.as_ptr())
  }
}

/// Vectorized 3x3 matmul: `[R G B]^T = M · [X Y Z]^T`.
///
/// Uses plain `vmulq_f32 + vaddq_f32` (NOT FMA) so the f32 rounding
/// schedule is identical to the scalar reference's
/// `(m[i][0]*x + m[i][1]*y) + m[i][2]*z`. FMA's single-rounding
/// changes the f32 result by up to 0.5 ULP, which after the per-lane
/// scalar OETF + integer narrow can flip output integers near
/// boundaries. The 0-ULP scalar↔SIMD parity contract demands matching
/// the scalar's mul-then-add evaluation exactly.
#[inline(always)]
unsafe fn matmul_xyz_to_rgb(
  m: &[[f32; 3]; 3],
  x: float32x4_t,
  y: float32x4_t,
  z: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let m00 = vdupq_n_f32(m[0][0]);
    let m01 = vdupq_n_f32(m[0][1]);
    let m02 = vdupq_n_f32(m[0][2]);
    let m10 = vdupq_n_f32(m[1][0]);
    let m11 = vdupq_n_f32(m[1][1]);
    let m12 = vdupq_n_f32(m[1][2]);
    let m20 = vdupq_n_f32(m[2][0]);
    let m21 = vdupq_n_f32(m[2][1]);
    let m22 = vdupq_n_f32(m[2][2]);

    let r = vaddq_f32(
      vaddq_f32(vmulq_f32(m00, x), vmulq_f32(m01, y)),
      vmulq_f32(m02, z),
    );
    let g = vaddq_f32(
      vaddq_f32(vmulq_f32(m10, x), vmulq_f32(m11, y)),
      vmulq_f32(m12, z),
    );
    let b = vaddq_f32(
      vaddq_f32(vmulq_f32(m20, x), vmulq_f32(m21, y)),
      vmulq_f32(m22, z),
    );
    (r, g, b)
  }
}

/// Loads 4 XYZ12 pixels and produces 3 `float32x4_t` vectors of
/// linear RGB after the inverse-OETF + matmul.
///
/// Caller obligation: `BE == HOST_NATIVE_BE` — see `load_xyz4`. The
/// `BE` parameter is preserved on the signature so the call site
/// reads symmetrically with the scalar fallback's `BE` route.
#[inline(always)]
unsafe fn load_and_matmul<const BE: bool>(
  p: *const u16,
  m: &[[f32; 3]; 3],
) -> (float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let _ = BE; // BE == HOST_NATIVE_BE precondition; no runtime branch.
    let (x_u, y_u, z_u) = load_xyz4(p);
    // Shift right 4 to extract the active 12-bit code, then run scalar
    // inverse-OETF per lane to get linear XYZ.
    let x_lin = smpte428_inv_oetf_scalar4(shift_mask_widen_cvt(x_u));
    let y_lin = smpte428_inv_oetf_scalar4(shift_mask_widen_cvt(y_u));
    let z_lin = smpte428_inv_oetf_scalar4(shift_mask_widen_cvt(z_u));
    matmul_xyz_to_rgb(m, x_lin, y_lin, z_lin)
  }
}

/// Loads 4 XYZ12 pixels and produces 3 `float32x4_t` vectors of
/// linear XYZ (step 1 only; no matmul).
///
/// Caller obligation: `BE == HOST_NATIVE_BE`.
#[inline(always)]
unsafe fn load_xyz_linear<const BE: bool>(
  p: *const u16,
) -> (float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let _ = BE; // BE == HOST_NATIVE_BE precondition.
    let (x_u, y_u, z_u) = load_xyz4(p);
    (
      smpte428_inv_oetf_scalar4(shift_mask_widen_cvt(x_u)),
      smpte428_inv_oetf_scalar4(shift_mask_widen_cvt(y_u)),
      smpte428_inv_oetf_scalar4(shift_mask_widen_cvt(z_u)),
    )
  }
}

/// Vectorized clamp `[0, 1]` x `scale` followed by **round-half-up**
/// cast to u32 then saturating narrow to u16.
///
/// The scalar reference is `((c.clamp(0,1) * scale) + 0.5) as int`,
/// i.e. round-half-up. We mirror it with `+ 0.5` then `vcvtq_u32_f32`
/// (truncating cast) — bit-exact across all inputs. (NEON's
/// `vcvtnq_u32_f32` is round-half-to-even, which would diverge by
/// 1 ULP at exact `*.5` ties; we deliberately avoid it.)
/// Re-tested by the SIMD-vs-scalar parity tests below.
#[inline(always)]
unsafe fn clamp_scale_to_u16x4(v: float32x4_t, scale: float32x4_t) -> uint16x4_t {
  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let clamped = vminq_f32_compat(vmaxq_f32_compat(v, zero), one);
    // Add 0.5 then truncate (round-half-up) to match scalar's
    // `(c * scale + 0.5) as u8/u16`.
    let half = vdupq_n_f32(0.5);
    let scaled = vaddq_f32(vmulq_f32(clamped, scale), half);
    let as_u32 = vcvtq_u32_f32(scaled); // truncation
    vqmovn_u32_compat(as_u32)
  }
}

// ---- Per-output kernels ------------------------------------------------

/// XYZ12 → packed u8 RGB. 4 pixels per SIMD iteration; tail handed to
/// the scalar reference.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xyz12_to_rgb_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  let scale = vdupq_n_f32(255.0);

  unsafe {
    // SIMD body runs only when wire byte order matches the host
    // (`BE == HOST_NATIVE_BE`); otherwise scalar handles the full row
    // via `from_le` / `from_be`.
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + LANES <= width {
        let p = xyz.as_ptr().add(x * 3);
        let (r_lin, g_lin, b_lin) = load_and_matmul::<BE>(p, &m);
        // Forward sRGB OETF: per-lane scalar.
        let r_oetf = oetf_srgb_scalar4(r_lin);
        let g_oetf = oetf_srgb_scalar4(g_lin);
        let b_oetf = oetf_srgb_scalar4(b_lin);
        // Narrow each f32x4 → u8x4 (via u16x4 saturating narrow then
        // vqmovn_u16). vqmovn_u16 takes u16x8 so we duplicate.
        let r_u16 = clamp_scale_to_u16x4(r_oetf, scale);
        let g_u16 = clamp_scale_to_u16x4(g_oetf, scale);
        let b_u16 = clamp_scale_to_u16x4(b_oetf, scale);
        let r_u8 = vqmovn_u16_compat(vcombine_u16(r_u16, r_u16));
        let g_u8 = vqmovn_u16_compat(vcombine_u16(g_u16, g_u16));
        let b_u8 = vqmovn_u16_compat(vcombine_u16(b_u16, b_u16));
        // vst3_u8 writes 24 bytes interleaved; we need 12 bytes (4 pixels).
        // vst3_u8 takes u8x8x3 — use the low half of each combined vector
        // and write only the first 12 bytes via a stack staging array.
        let mut tmp = [0u8; 24];
        vst3_u8(tmp.as_mut_ptr(), uint8x8x3_t(r_u8, g_u8, b_u8));
        rgb_out
          .get_unchecked_mut(x * 3..x * 3 + 12)
          .copy_from_slice(&tmp[..12]);
        x += LANES;
      }
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
/// 1. NEON must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xyz12_to_rgba_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  let scale = vdupq_n_f32(255.0);
  let alpha = vdup_n_u8(0xFF);

  unsafe {
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + LANES <= width {
        let p = xyz.as_ptr().add(x * 3);
        let (r_lin, g_lin, b_lin) = load_and_matmul::<BE>(p, &m);
        let r_oetf = oetf_srgb_scalar4(r_lin);
        let g_oetf = oetf_srgb_scalar4(g_lin);
        let b_oetf = oetf_srgb_scalar4(b_lin);
        let r_u16 = clamp_scale_to_u16x4(r_oetf, scale);
        let g_u16 = clamp_scale_to_u16x4(g_oetf, scale);
        let b_u16 = clamp_scale_to_u16x4(b_oetf, scale);
        let r_u8 = vqmovn_u16_compat(vcombine_u16(r_u16, r_u16));
        let g_u8 = vqmovn_u16_compat(vcombine_u16(g_u16, g_u16));
        let b_u8 = vqmovn_u16_compat(vcombine_u16(b_u16, b_u16));
        let mut tmp = [0u8; 32];
        vst4_u8(tmp.as_mut_ptr(), uint8x8x4_t(r_u8, g_u8, b_u8, alpha));
        rgba_out
          .get_unchecked_mut(x * 4..x * 4 + 16)
          .copy_from_slice(&tmp[..16]);
        x += LANES;
      }
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

/// XYZ12 → packed u16 RGB (full-range scaling, x65535).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xyz12_to_rgb_u16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  let scale = vdupq_n_f32(65535.0);

  unsafe {
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + LANES <= width {
        let p = xyz.as_ptr().add(x * 3);
        let (r_lin, g_lin, b_lin) = load_and_matmul::<BE>(p, &m);
        let r_oetf = oetf_srgb_scalar4(r_lin);
        let g_oetf = oetf_srgb_scalar4(g_lin);
        let b_oetf = oetf_srgb_scalar4(b_lin);
        let r_u16 = clamp_scale_to_u16x4(r_oetf, scale);
        let g_u16 = clamp_scale_to_u16x4(g_oetf, scale);
        let b_u16 = clamp_scale_to_u16x4(b_oetf, scale);
        // vst3_u16 writes 12 u16 elements (24 bytes) — exactly 4 pixels.
        vst3_u16(
          rgb_out.as_mut_ptr().add(x * 3),
          uint16x4x3_t(r_u16, g_u16, b_u16),
        );
        x += LANES;
      }
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
/// 1. NEON must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xyz12_to_rgba_u16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  let scale = vdupq_n_f32(65535.0);
  let alpha = vdup_n_u16(0xFFFF);

  unsafe {
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + LANES <= width {
        let p = xyz.as_ptr().add(x * 3);
        let (r_lin, g_lin, b_lin) = load_and_matmul::<BE>(p, &m);
        let r_oetf = oetf_srgb_scalar4(r_lin);
        let g_oetf = oetf_srgb_scalar4(g_lin);
        let b_oetf = oetf_srgb_scalar4(b_lin);
        let r_u16 = clamp_scale_to_u16x4(r_oetf, scale);
        let g_u16 = clamp_scale_to_u16x4(g_oetf, scale);
        let b_u16 = clamp_scale_to_u16x4(b_oetf, scale);
        vst4_u16(
          rgba_out.as_mut_ptr().add(x * 4),
          uint16x4x4_t(r_u16, g_u16, b_u16, alpha),
        );
        x += LANES;
      }
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

/// XYZ12 → packed linear RGB f32. Lossless: matrix only, no OETF, no
/// clamp.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
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
    if BE == HOST_NATIVE_BE {
      while x + LANES <= width {
        let p = xyz.as_ptr().add(x * 3);
        let (r_lin, g_lin, b_lin) = load_and_matmul::<BE>(p, &m);
        vst3q_f32(
          rgb_out.as_mut_ptr().add(x * 3),
          float32x4x3_t(r_lin, g_lin, b_lin),
        );
        x += LANES;
      }
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
/// 1. NEON must be available.
/// 2. `xyz.len() >= width * 3`; `xyz_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xyz12_to_xyz_f32_row<const BE: bool>(
  xyz: &[u16],
  xyz_out: &mut [f32],
  width: usize,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(xyz_out.len() >= width * 3, "xyz_out row too short");

  unsafe {
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + LANES <= width {
        let p = xyz.as_ptr().add(x * 3);
        let (xv, yv, zv) = load_xyz_linear::<BE>(p);
        vst3q_f32(xyz_out.as_mut_ptr().add(x * 3), float32x4x3_t(xv, yv, zv));
        x += LANES;
      }
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
/// f16 narrow runs scalar per lane via `half::f16::from_f32` for
/// portability — NEON-fp16 (`+fp16` feature) would let us use
/// `vcvt_f16_f32` directly, but the scalar narrow is a few cycles per
/// pixel and matches the IEEE-754 RNE semantics of the scalar
/// reference exactly.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
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
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + LANES <= width {
        let p = xyz.as_ptr().add(x * 3);
        let (r_lin, g_lin, b_lin) = load_and_matmul::<BE>(p, &m);
        let r_oetf = oetf_srgb_scalar4(r_lin);
        let g_oetf = oetf_srgb_scalar4(g_lin);
        let b_oetf = oetf_srgb_scalar4(b_lin);
        // Clamp [0, 1] then narrow to f16 per lane.
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let r_clamp = vminq_f32_compat(vmaxq_f32_compat(r_oetf, zero), one);
        let g_clamp = vminq_f32_compat(vmaxq_f32_compat(g_oetf, zero), one);
        let b_clamp = vminq_f32_compat(vmaxq_f32_compat(b_oetf, zero), one);
        let mut rb = [0.0_f32; LANES];
        let mut gb = [0.0_f32; LANES];
        let mut bb = [0.0_f32; LANES];
        vst1q_f32(rb.as_mut_ptr(), r_clamp);
        vst1q_f32(gb.as_mut_ptr(), g_clamp);
        vst1q_f32(bb.as_mut_ptr(), b_clamp);
        for i in 0..LANES {
          let oi = (x + i) * 3;
          rgb_out[oi] = half::f16::from_f32(rb[i]);
          rgb_out[oi + 1] = half::f16::from_f32(gb[i]);
          rgb_out[oi + 2] = half::f16::from_f32(bb[i]);
        }
        x += LANES;
      }
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
/// 1. NEON must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
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
    let mut x = 0usize;
    if BE == HOST_NATIVE_BE {
      while x + LANES <= width {
        let p = xyz.as_ptr().add(x * 3);
        let (r_lin, g_lin, b_lin) = load_and_matmul::<BE>(p, &m);
        let r_oetf = oetf_srgb_scalar4(r_lin);
        let g_oetf = oetf_srgb_scalar4(g_lin);
        let b_oetf = oetf_srgb_scalar4(b_lin);
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let r_clamp = vminq_f32_compat(vmaxq_f32_compat(r_oetf, zero), one);
        let g_clamp = vminq_f32_compat(vmaxq_f32_compat(g_oetf, zero), one);
        let b_clamp = vminq_f32_compat(vmaxq_f32_compat(b_oetf, zero), one);
        let mut rb = [0.0_f32; LANES];
        let mut gb = [0.0_f32; LANES];
        let mut bb = [0.0_f32; LANES];
        vst1q_f32(rb.as_mut_ptr(), r_clamp);
        vst1q_f32(gb.as_mut_ptr(), g_clamp);
        vst1q_f32(bb.as_mut_ptr(), b_clamp);
        for i in 0..LANES {
          let oi = (x + i) * 4;
          rgba_out[oi] = half::f16::from_f32(rb[i]);
          rgba_out[oi + 1] = half::f16::from_f32(gb[i]);
          rgba_out[oi + 2] = half::f16::from_f32(bb[i]);
          rgba_out[oi + 3] = one_f16;
        }
        x += LANES;
      }
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
