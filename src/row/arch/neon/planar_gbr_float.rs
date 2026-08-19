//! NEON SIMD kernels for planar GBR float sources (Tier 10).
//!
//! f32 path: 4 pixels / iteration via `float32x4_t` (one G, B, R [+ A] lane
//! each). f16 widening / narrowing gated on the aarch64 `fp16` feature;
//! scalar fallback used when that feature is absent at runtime.
//!
//! # Rounding (f32 → u8 / u16)
//!
//! `+ 0.5` then `vcvtq_u32_f32` (truncation toward zero). This is the
//! round-half-up contract shared with the scalar kernels — MXCSR-independent
//! and consistent with the Rgbf32 path.
//!
//! # Rounding (f32 → f16)
//!
//! `vcvt_f16_f32` — IEEE-754 round-to-nearest-even, which matches
//! `half::f16::from_f32`.
//!
//! # f16 lossless interleave
//!
//! Treat f16 lanes as opaque `u16` for `vst3_u16` / `vst4_u16` — no
//! arithmetic, so no `fp16` feature gate needed.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use super::miri_compat::*;
use crate::{
  KernelMatrix,
  row::{
    arch::neon::endian::{load_endian_u16x4, load_endian_u32x4},
    scalar::{planar_gbr_f16 as scalar_f16, planar_gbr_float as scalar},
  },
};

/// `BE` value that makes the downstream `scalar::gbrpf32_to_*` kernels treat
/// their `f32` scratch input as **host-native** (no `from_be` / `from_le`
/// byte-swap). After we widen f16 → f32 via
/// [`scalar_f16::widen_f16_be_to_host_f32`] (which normalizes the source
/// f16 bits per the source `BE` and produces host-native f32), the resulting
/// scratch must be routed via `HOST_NATIVE_BE` so the downstream kernel's
/// `from_le` / `from_be` loaders no-op the swap. Without this routing the
/// SIMD scalar tail double-byte-swaps on `BE`-source-on-LE-host (and
/// symmetrically `LE`-source-on-BE-host).
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// ---- shared helpers ---------------------------------------------------------

/// Clamp a `float32x4_t` to `[0.0, 1.0]`.
#[inline(always)]
unsafe fn clamp01(v: float32x4_t, zero: float32x4_t, one: float32x4_t) -> float32x4_t {
  unsafe { vminq_f32_compat(vmaxq_f32_compat(v, zero), one) }
}

/// Load 4 f32 values with optional BE byte-swap via `load_endian_u32x4`.
/// This is the endian-aware replacement for `vld1q_f32(ptr.add(x))`.
#[inline(always)]
unsafe fn load_f32x4<const BE: bool>(ptr: *const f32, x: usize) -> float32x4_t {
  unsafe {
    let u = load_endian_u32x4::<BE>(ptr.add(x).cast::<u8>());
    vreinterpretq_f32_u32(u)
  }
}

/// Scale, add 0.5, truncate → `uint32x4_t` (round-half-up).
#[inline(always)]
unsafe fn scale_round_u32(v: float32x4_t, scale: float32x4_t, half: float32x4_t) -> uint32x4_t {
  unsafe { vcvtq_u32_f32(vaddq_f32(vmulq_f32(v, scale), half)) }
}

/// Narrow `uint32x4_t` → `uint16x4_t` → `uint8x8_t` (first 4 bytes valid).
#[inline(always)]
unsafe fn narrow_to_u8(v: uint32x4_t) -> uint8x8_t {
  unsafe { vqmovn_u16_compat(vcombine_u16(vqmovn_u32_compat(v), vdup_n_u16(0))) }
}

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

/// NEON: planar Gbrpf32 → packed `R, G, B` bytes.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf32_to_rgb_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(load_f32x4::<BE>(g.as_ptr(), x), zero, one);
      let bv = clamp01(load_f32x4::<BE>(b.as_ptr(), x), zero, one);
      let rv = clamp01(load_f32x4::<BE>(r.as_ptr(), x), zero, one);
      let gi = narrow_to_u8(scale_round_u32(gv, scale, half));
      let bi = narrow_to_u8(scale_round_u32(bv, scale, half));
      let ri = narrow_to_u8(scale_round_u32(rv, scale, half));
      // vst3_u8 writes 24 bytes; only 12 are valid (4 pixels x 3 channels).
      let mut tmp = [0u8; 24];
      vst3_u8(tmp.as_mut_ptr(), uint8x8x3_t(ri, gi, bi));
      out
        .get_unchecked_mut(x * 3..x * 3 + 12)
        .copy_from_slice(&tmp[..12]);
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → u8 RGBA (opaque α) ------------------------------------------

/// NEON: planar Gbrpf32 → packed `R, G, B, A` bytes (α = 0xFF).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);
    let alpha = vdup_n_u8(0xFF);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(load_f32x4::<BE>(g.as_ptr(), x), zero, one);
      let bv = clamp01(load_f32x4::<BE>(b.as_ptr(), x), zero, one);
      let rv = clamp01(load_f32x4::<BE>(r.as_ptr(), x), zero, one);
      let gi = narrow_to_u8(scale_round_u32(gv, scale, half));
      let bi = narrow_to_u8(scale_round_u32(bv, scale, half));
      let ri = narrow_to_u8(scale_round_u32(rv, scale, half));
      // vst4_u8 writes 32 bytes; 16 are valid (4 pixels x 4 channels).
      let mut tmp = [0u8; 32];
      vst4_u8(tmp.as_mut_ptr(), uint8x8x4_t(ri, gi, bi, alpha));
      out
        .get_unchecked_mut(x * 4..x * 4 + 16)
        .copy_from_slice(&tmp[..16]);
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 4..], width - x);
    }
  }
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

/// NEON: planar Gbrpf32 → packed `R, G, B` u16.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf32_to_rgb_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(65535.0);
    let half = vdupq_n_f32(0.5);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(load_f32x4::<BE>(g.as_ptr(), x), zero, one);
      let bv = clamp01(load_f32x4::<BE>(b.as_ptr(), x), zero, one);
      let rv = clamp01(load_f32x4::<BE>(r.as_ptr(), x), zero, one);
      let gu = vqmovn_u32_compat(scale_round_u32(gv, scale, half));
      let bu = vqmovn_u32_compat(scale_round_u32(bv, scale, half));
      let ru = vqmovn_u32_compat(scale_round_u32(rv, scale, half));
      // vst3_u16 writes 24 bytes; 24 are valid (4 pixels x 3 x 2 bytes).
      vst3_u16(out.as_mut_ptr().add(x * 3), uint16x4x3_t(ru, gu, bu));
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_u16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → u16 RGBA (opaque α) -----------------------------------------

/// NEON: planar Gbrpf32 → packed `R, G, B, A` u16 (α = 0xFFFF).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(65535.0);
    let half = vdupq_n_f32(0.5);
    let alpha = vdup_n_u16(0xFFFF);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(load_f32x4::<BE>(g.as_ptr(), x), zero, one);
      let bv = clamp01(load_f32x4::<BE>(b.as_ptr(), x), zero, one);
      let rv = clamp01(load_f32x4::<BE>(r.as_ptr(), x), zero, one);
      let gu = vqmovn_u32_compat(scale_round_u32(gv, scale, half));
      let bu = vqmovn_u32_compat(scale_round_u32(bv, scale, half));
      let ru = vqmovn_u32_compat(scale_round_u32(rv, scale, half));
      // vst4_u16 writes 32 bytes; 32 are valid (4 pixels x 4 x 2 bytes).
      vst4_u16(out.as_mut_ptr().add(x * 4), uint16x4x4_t(ru, gu, bu, alpha));
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_u16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

/// NEON: planar Gbrpf32 → packed `R, G, B` f32 (lossless interleave).
///
/// Uses `vst3q_f32` to interleave 4-lane f32 vectors in a single instruction.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf32_to_rgb_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = load_f32x4::<BE>(g.as_ptr(), x);
      let bv = load_f32x4::<BE>(b.as_ptr(), x);
      let rv = load_f32x4::<BE>(r.as_ptr(), x);
      vst3q_f32(out.as_mut_ptr().add(x * 3), float32x4x3_t(rv, gv, bv));
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_f32_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) --------------------------------

/// NEON: planar Gbrpf32 → packed `R, G, B, A` f32 (lossless, α = 1.0).
///
/// Uses `vst4q_f32` to interleave with a constant 1.0 alpha lane.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let one_v = vdupq_n_f32(1.0);
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = load_f32x4::<BE>(g.as_ptr(), x);
      let bv = load_f32x4::<BE>(b.as_ptr(), x);
      let rv = load_f32x4::<BE>(r.as_ptr(), x);
      vst4q_f32(
        out.as_mut_ptr().add(x * 4),
        float32x4x4_t(rv, gv, bv, one_v),
      );
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_f32_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf32 → f16 RGB (fused narrow, fp16-gated) ---------------------------

/// NEON: planar Gbrpf32 → packed `R, G, B` f16 (fused narrow + interleave).
///
/// Requires the `fp16` feature (NEON float16 arithmetic). Falls back to scalar
/// when `fp16` is absent at runtime — the dispatcher handles this.
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn gbrpf32_to_rgb_f16_row_fp16<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = load_f32x4::<BE>(g.as_ptr(), x);
      let bv = load_f32x4::<BE>(b.as_ptr(), x);
      let rv = load_f32x4::<BE>(r.as_ptr(), x);
      // IEEE-754 RNE narrow via vcvt_f16_f32.
      let gh = vcvt_f16_f32(gv);
      let bh = vcvt_f16_f32(bv);
      let rh = vcvt_f16_f32(rv);
      // Reinterpret f16x4 as u16x4 for vst3_u16 interleave.
      vst3_u16(
        out.as_mut_ptr().add(x * 3).cast::<u16>(),
        uint16x4x3_t(
          vreinterpret_u16_f16(rh),
          vreinterpret_u16_f16(gh),
          vreinterpret_u16_f16(bh),
        ),
      );
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_f16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → f16 RGBA (fused narrow, fp16-gated) -------------------------

/// NEON: planar Gbrpf32 → packed `R, G, B, A` f16 (fused narrow, α = f16(1.0)).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn gbrpf32_to_rgba_f16_row_fp16<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    // f16(1.0) = 0x3C00
    let one_h = vreinterpret_u16_f16(vcvt_f16_f32(vdupq_n_f32(1.0)));
    let alpha = vdup_n_u16(0x3C00u16);
    let _ = one_h; // computed above; use constant for clarity
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = load_f32x4::<BE>(g.as_ptr(), x);
      let bv = load_f32x4::<BE>(b.as_ptr(), x);
      let rv = load_f32x4::<BE>(r.as_ptr(), x);
      let gh = vreinterpret_u16_f16(vcvt_f16_f32(gv));
      let bh = vreinterpret_u16_f16(vcvt_f16_f32(bv));
      let rh = vreinterpret_u16_f16(vcvt_f16_f32(rv));
      vst4_u16(
        out.as_mut_ptr().add(x * 4).cast::<u16>(),
        uint16x4x4_t(rh, gh, bh, alpha),
      );
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf32 → u8 luma (staged via RGB scratch) ----------------------------

/// NEON: planar Gbrpf32 → u8 luma (staged via NEON RGB kernel + luma kernel).
///
/// Converts in 64-pixel chunks: Gbrpf32 → u8 RGB scratch via NEON, then
/// `rgb_to_luma_row` (also NEON-dispatched).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gbrpf32_to_luma_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width, "out row too short");
  // Stage via scalar (which already handles chunking + rgb_to_luma).
  // NEON for luma is the rgb_to_luma step; the RGB staging kernel here
  // provides the NEON acceleration for the f32 → u8 conversion.
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf32_to_rgb_row::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_luma_row(
      &scratch[..n * 3],
      &mut out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → u16 luma (staged via RGB scratch) ---------------------------

/// NEON: planar Gbrpf32 → u16 luma (staged via NEON RGB kernel).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gbrpf32_to_luma_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width, "out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf32_to_rgb_row::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_luma_u16_row(
      &scratch[..n * 3],
      &mut out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → HSV (staged via RGB scratch) --------------------------------

/// NEON: planar Gbrpf32 → planar HSV bytes (staged via NEON RGB kernel).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf32_to_hsv_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf32_to_rgb_row::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_hsv_row(
      &scratch[..n * 3],
      &mut h_out[offset..offset + n],
      &mut s_out[offset..offset + n],
      &mut v_out[offset..offset + n],
      n,
    );
    offset += n;
  }
}

// ---- Gbrapf32 → u8 RGBA (source α) -----------------------------------------

/// NEON: planar Gbrapf32 → packed `R, G, B, A` bytes (source α plane).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrapf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(load_f32x4::<BE>(g.as_ptr(), x), zero, one);
      let bv = clamp01(load_f32x4::<BE>(b.as_ptr(), x), zero, one);
      let rv = clamp01(load_f32x4::<BE>(r.as_ptr(), x), zero, one);
      let av = clamp01(load_f32x4::<BE>(a.as_ptr(), x), zero, one);
      let gi = narrow_to_u8(scale_round_u32(gv, scale, half));
      let bi = narrow_to_u8(scale_round_u32(bv, scale, half));
      let ri = narrow_to_u8(scale_round_u32(rv, scale, half));
      let ai = narrow_to_u8(scale_round_u32(av, scale, half));
      let mut tmp = [0u8; 32];
      vst4_u8(tmp.as_mut_ptr(), uint8x8x4_t(ri, gi, bi, ai));
      out
        .get_unchecked_mut(x * 4..x * 4 + 16)
        .copy_from_slice(&tmp[..16]);
      x += 4;
    }
    if x < width {
      scalar::gbrapf32_to_rgba_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

/// NEON: planar Gbrapf32 → packed `R, G, B, A` u16 (source α plane).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrapf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(65535.0);
    let half = vdupq_n_f32(0.5);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(load_f32x4::<BE>(g.as_ptr(), x), zero, one);
      let bv = clamp01(load_f32x4::<BE>(b.as_ptr(), x), zero, one);
      let rv = clamp01(load_f32x4::<BE>(r.as_ptr(), x), zero, one);
      let av = clamp01(load_f32x4::<BE>(a.as_ptr(), x), zero, one);
      let gu = vqmovn_u32_compat(scale_round_u32(gv, scale, half));
      let bu = vqmovn_u32_compat(scale_round_u32(bv, scale, half));
      let ru = vqmovn_u32_compat(scale_round_u32(rv, scale, half));
      let au = vqmovn_u32_compat(scale_round_u32(av, scale, half));
      vst4_u16(out.as_mut_ptr().add(x * 4), uint16x4x4_t(ru, gu, bu, au));
      x += 4;
    }
    if x < width {
      scalar::gbrapf32_to_rgba_u16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless source α) --------------------------------

/// NEON: planar Gbrapf32 → packed `R, G, B, A` f32 (lossless, source α).
///
/// Uses `vst4q_f32` for 4-channel interleave in a single instruction.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrapf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = load_f32x4::<BE>(g.as_ptr(), x);
      let bv = load_f32x4::<BE>(b.as_ptr(), x);
      let rv = load_f32x4::<BE>(r.as_ptr(), x);
      let av = load_f32x4::<BE>(a.as_ptr(), x);
      vst4q_f32(out.as_mut_ptr().add(x * 4), float32x4x4_t(rv, gv, bv, av));
      x += 4;
    }
    if x < width {
      scalar::gbrapf32_to_rgba_f32_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrapf32 → f16 RGBA (fused narrow, fp16-gated) ------------------------

/// NEON: planar Gbrapf32 → packed `R, G, B, A` f16 (fused narrow, source α).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn gbrapf32_to_rgba_f16_row_fp16<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = load_f32x4::<BE>(g.as_ptr(), x);
      let bv = load_f32x4::<BE>(b.as_ptr(), x);
      let rv = load_f32x4::<BE>(r.as_ptr(), x);
      let av = load_f32x4::<BE>(a.as_ptr(), x);
      let gh = vreinterpret_u16_f16(vcvt_f16_f32(gv));
      let bh = vreinterpret_u16_f16(vcvt_f16_f32(bv));
      let rh = vreinterpret_u16_f16(vcvt_f16_f32(rv));
      let ah = vreinterpret_u16_f16(vcvt_f16_f32(av));
      vst4_u16(
        out.as_mut_ptr().add(x * 4).cast::<u16>(),
        uint16x4x4_t(rh, gh, bh, ah),
      );
      x += 4;
    }
    if x < width {
      scalar::gbrapf32_to_rgba_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf16 → u8 RGB (fp16-gated widening) --------------------------------

/// NEON: planar Gbrpf16 → packed `R, G, B` bytes (widen f16→f32, then convert).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn gbrpf16_to_rgb_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(255.0);
    let half_v = vdupq_n_f32(0.5);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gi = narrow_to_u8(scale_round_u32(gc, scale, half_v));
      let bi = narrow_to_u8(scale_round_u32(bc, scale, half_v));
      let ri = narrow_to_u8(scale_round_u32(rc, scale, half_v));
      let mut tmp = [0u8; 24];
      vst3_u8(tmp.as_mut_ptr(), uint8x8x3_t(ri, gi, bi));
      out
        .get_unchecked_mut(x * 3..x * 3 + 12)
        .copy_from_slice(&tmp[..12]);
      x += 4;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 3..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → u8 RGBA (fp16-gated widening) --------------------------------

/// NEON: planar Gbrpf16 → packed `R, G, B, A` bytes (widen f16→f32, α = 0xFF).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn gbrpf16_to_rgba_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(255.0);
    let half_v = vdupq_n_f32(0.5);
    let alpha = vdup_n_u8(0xFF);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gi = narrow_to_u8(scale_round_u32(gc, scale, half_v));
      let bi = narrow_to_u8(scale_round_u32(bc, scale, half_v));
      let ri = narrow_to_u8(scale_round_u32(rc, scale, half_v));
      let mut tmp = [0u8; 32];
      vst4_u8(tmp.as_mut_ptr(), uint8x8x4_t(ri, gi, bi, alpha));
      out
        .get_unchecked_mut(x * 4..x * 4 + 16)
        .copy_from_slice(&tmp[..16]);
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → u16 RGB (fp16-gated widening) --------------------------------

/// NEON: planar Gbrpf16 → packed `R, G, B` u16 (widen f16→f32, then convert).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_rgb_u16_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(65535.0);
    let half_v = vdupq_n_f32(0.5);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gu = vqmovn_u32_compat(scale_round_u32(gc, scale, half_v));
      let bu = vqmovn_u32_compat(scale_round_u32(bc, scale, half_v));
      let ru = vqmovn_u32_compat(scale_round_u32(rc, scale, half_v));
      vst3_u16(out.as_mut_ptr().add(x * 3), uint16x4x3_t(ru, gu, bu));
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 3..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → u16 RGBA (fp16-gated widening) --------------------------------

/// NEON: planar Gbrpf16 → packed `R, G, B, A` u16 (widen f16→f32, α = 0xFFFF).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_rgba_u16_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(65535.0);
    let half_v = vdupq_n_f32(0.5);
    let alpha = vdup_n_u16(0xFFFF);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gu = vqmovn_u32_compat(scale_round_u32(gc, scale, half_v));
      let bu = vqmovn_u32_compat(scale_round_u32(bc, scale, half_v));
      let ru = vqmovn_u32_compat(scale_round_u32(rc, scale, half_v));
      vst4_u16(out.as_mut_ptr().add(x * 4), uint16x4x4_t(ru, gu, bu, alpha));
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → f32 RGB (fp16-gated widening, lossless) ----------------------

/// NEON: planar Gbrpf16 → packed `R, G, B` f32 (lossless widen + interleave).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_rgb_f32_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      vst3q_f32(out.as_mut_ptr().add(x * 3), float32x4x3_t(rv, gv, bv));
      x += 4;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 3..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → f32 RGBA (fp16-gated widening, lossless, α = 1.0) -----------

/// NEON: planar Gbrpf16 → packed `R, G, B, A` f32 (lossless widen, α = 1.0).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_rgba_f32_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let one_v = vdupq_n_f32(1.0);
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      vst4q_f32(
        out.as_mut_ptr().add(x * 4),
        float32x4x4_t(rv, gv, bv, one_v),
      );
      x += 4;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → f16 RGB (lossless, opaque u16 interleave, no fp16 needed) ---

/// NEON: planar Gbrpf16 → packed `R, G, B` f16 (lossless — treat f16 as u16).
///
/// No `fp16` feature needed: f16 planes are bit-copied as opaque u16 lanes.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf16_to_rgb_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gu = load_endian_u16x4::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = load_endian_u16x4::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = load_endian_u16x4::<BE>(r.as_ptr().add(x).cast::<u8>());
      vst3_u16(
        out.as_mut_ptr().add(x * 3).cast::<u16>(),
        uint16x4x3_t(ru, gu, bu),
      );
      x += 4;
    }
    if x < width {
      scalar_f16::gbrpf16_to_rgb_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 3..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf16 → f16 RGBA (lossless, opaque u16, no fp16 needed) -------------

/// NEON: planar Gbrpf16 → packed `R, G, B, A` f16 (lossless, α = f16(1.0)).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrpf16_to_rgba_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    // f16(1.0) = 0x3C00
    let alpha = vdup_n_u16(0x3C00u16);
    let mut x = 0usize;
    while x + 4 <= width {
      let gu = load_endian_u16x4::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = load_endian_u16x4::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = load_endian_u16x4::<BE>(r.as_ptr().add(x).cast::<u8>());
      vst4_u16(
        out.as_mut_ptr().add(x * 4).cast::<u16>(),
        uint16x4x4_t(ru, gu, bu, alpha),
      );
      x += 4;
    }
    if x < width {
      scalar_f16::gbrpf16_to_rgba_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf16 → u8 luma (fp16-gated) ----------------------------------------

/// NEON: planar Gbrpf16 → u8 luma (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_luma_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width, "out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf16_to_rgb_row_fp16::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_luma_row(
      &scratch[..n * 3],
      &mut out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf16 → u16 luma (fp16-gated) ----------------------------------------

/// NEON: planar Gbrpf16 → u16 luma (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_luma_u16_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width, "out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf16_to_rgb_row_fp16::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_luma_u16_row(
      &scratch[..n * 3],
      &mut out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf16 → HSV (fp16-gated) --------------------------------------------

/// NEON: planar Gbrpf16 → planar HSV bytes (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_hsv_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf16_to_rgb_row_fp16::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_hsv_row(
      &scratch[..n * 3],
      &mut h_out[offset..offset + n],
      &mut s_out[offset..offset + n],
      &mut v_out[offset..offset + n],
      n,
    );
    offset += n;
  }
}

// ---- Gbrapf16 → u8 RGBA (fp16-gated widening) --------------------------------

/// NEON: planar Gbrapf16 → packed `R, G, B, A` bytes (widen f16→f32, source α).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrapf16_to_rgba_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(255.0);
    let half_v = vdupq_n_f32(0.5);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      let av = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      )));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let ac = clamp01(av, zero, one);
      let gi = narrow_to_u8(scale_round_u32(gc, scale, half_v));
      let bi = narrow_to_u8(scale_round_u32(bc, scale, half_v));
      let ri = narrow_to_u8(scale_round_u32(rc, scale, half_v));
      let ai = narrow_to_u8(scale_round_u32(ac, scale, half_v));
      let mut tmp = [0u8; 32];
      vst4_u8(tmp.as_mut_ptr(), uint8x8x4_t(ri, gi, bi, ai));
      out
        .get_unchecked_mut(x * 4..x * 4 + 16)
        .copy_from_slice(&tmp[..16]);
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      let mut af = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(a, x, &mut af, tail);
      scalar::gbrapf32_to_rgba_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &af[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrapf16 → u16 RGBA (fp16-gated widening) ------------------------------

/// NEON: planar Gbrapf16 → packed `R, G, B, A` u16 (widen f16→f32, source α).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrapf16_to_rgba_u16_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(65535.0);
    let half_v = vdupq_n_f32(0.5);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      let av = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      )));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let ac = clamp01(av, zero, one);
      let gu = vqmovn_u32_compat(scale_round_u32(gc, scale, half_v));
      let bu = vqmovn_u32_compat(scale_round_u32(bc, scale, half_v));
      let ru = vqmovn_u32_compat(scale_round_u32(rc, scale, half_v));
      let au = vqmovn_u32_compat(scale_round_u32(ac, scale, half_v));
      vst4_u16(out.as_mut_ptr().add(x * 4), uint16x4x4_t(ru, gu, bu, au));
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      let mut af = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(a, x, &mut af, tail);
      scalar::gbrapf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &af[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrapf16 → f32 RGBA (fp16-gated widening, lossless) --------------------

/// NEON: planar Gbrapf16 → packed `R, G, B, A` f32 (lossless widen, source α).
///
/// # Safety
///
/// 1. NEON + fp16 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrapf16_to_rgba_f32_row_fp16<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      )));
      let bv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      )));
      let rv = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      )));
      let av = vcvt_f32_f16(vreinterpret_f16_u16(load_endian_u16x4::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      )));
      vst4q_f32(out.as_mut_ptr().add(x * 4), float32x4x4_t(rv, gv, bv, av));
      x += 4;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      let mut af = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(a, x, &mut af, tail);
      scalar::gbrapf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &af[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrapf16 → f16 RGBA (lossless, opaque u16, no fp16 needed) -------------

/// NEON: planar Gbrapf16 → packed `R, G, B, A` f16 (lossless, source α).
///
/// No `fp16` feature needed — f16 planes treated as opaque u16.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbrapf16_to_rgba_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gu = load_endian_u16x4::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = load_endian_u16x4::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = load_endian_u16x4::<BE>(r.as_ptr().add(x).cast::<u8>());
      let au = load_endian_u16x4::<BE>(a.as_ptr().add(x).cast::<u8>());
      vst4_u16(
        out.as_mut_ptr().add(x * 4).cast::<u16>(),
        uint16x4x4_t(ru, gu, bu, au),
      );
      x += 4;
    }
    if x < width {
      scalar_f16::gbrapf16_to_rgba_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}
