//! Scalar reference implementations of the row primitives.
//!
//! Always compiled. SIMD backends live in [`super::arch`] and dispatch
//! to these as their tail fallback. Per-call dispatch in
//! [`super`]`::{yuv_420_to_rgb_row, rgb_to_hsv_row}` picks the best
//! backend at the module boundary.

use crate::ColorMatrix;

// Per-conversion-family submodules. Each holds a self-contained
// cluster of scalar reference kernels; `mod.rs` retains only the
// cross-cutting helpers (`clamp_u8`, `q15_*`, `bits_mask`,
// `Coefficients`, …) that every family pulls in.
mod bayer;
mod hsv;
mod packed_rgb;
mod rgb_expand;
mod semi_planar_8bit;
mod subsampled_high_bit_pn;
mod yuv_planar_16bit;
mod yuv_planar_8bit;
mod yuv_planar_high_bit;

pub(crate) use bayer::*;
pub(crate) use hsv::*;
pub(crate) use packed_rgb::*;
pub(crate) use rgb_expand::*;
pub(crate) use semi_planar_8bit::*;
pub(crate) use subsampled_high_bit_pn::*;
pub(crate) use yuv_planar_8bit::*;
pub(crate) use yuv_planar_16bit::*;
pub(crate) use yuv_planar_high_bit::*;

// ---- Shared scalar helpers (used across all conversion families) -------

#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn clamp_u8(v: i32) -> u8 {
  v.clamp(0, 255) as u8
}

/// `(sample * scale_q15 + RND) >> 15`. With input masked to BITS,
/// the `sample * scale` product cannot overflow i32 for any
/// reasonable `OUT_BITS ≤ 16`, so plain arithmetic is sufficient.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn q15_scale(sample: i32, scale_q15: i32) -> i32 {
  (sample * scale_q15 + (1 << 14)) >> 15
}

/// `(c_u * u_d + c_v * v_d + RND) >> 15`. Chroma sum max ≈ 10⁹ for
/// 14‑bit masked input, well within i32.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn q15_chroma(c_u: i32, u_d: i32, c_v: i32, v_d: i32) -> i32 {
  (c_u * u_d + c_v * v_d + (1 << 14)) >> 15
}

/// `(c_u * u_d + c_v * v_d + RND) >> 15` computed in i64. Chroma sum
/// max ≈ 4.3·10⁹ at 16-bit limited range — above i32 but well within
/// i64. Result after the shift is bounded by ~130 000 so the final
/// `as i32` narrow is lossless.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn q15_chroma64(c_u: i32, u_d: i32, c_v: i32, v_d: i32) -> i32 {
  let sum = (c_u as i64) * (u_d as i64) + (c_v as i64) * (v_d as i64);
  ((sum + (1 << 14)) >> 15) as i32
}

/// `(sample * scale_q15 + RND) >> 15` computed in i64. For 16-bit
/// samples at limited-range 16 → u16 scaling, `sample * y_scale` can
/// reach ~2.35·10⁹ — just over i32::MAX — when unclamped `u16` input
/// exceeds the nominal limited-range Y max. Result after the shift
/// is bounded by ~65 536 so the final `as i32` narrow is lossless.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn q15_scale64(sample: i32, scale_q15: i32) -> i32 {
  (((sample as i64) * (scale_q15 as i64) + (1 << 14)) >> 15) as i32
}

/// Compile‑time sample mask for `BITS`: `(1 << BITS) - 1` as `u16`.
/// Returns `0x03FF` for 10‑bit, `0x0FFF` for 12‑bit, `0x3FFF` for
/// 14‑bit. SIMD backends splat this into a vector constant and AND
/// every load against it.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn bits_mask<const BITS: u32>() -> u16 {
  ((1u32 << BITS) - 1) as u16
}

/// Chroma bias for input bit depth `BITS` — `128 << (BITS - 8)`.
/// 128 for 8‑bit, 512 for 10‑bit, 2048 for 12‑bit, 8192 for 14‑bit.
/// Exposed at module visibility so SIMD backends can reuse it.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn chroma_bias<const BITS: u32>() -> i32 {
  128i32 << (BITS - 8)
}

/// Range‑scaling params `(y_off, y_scale_q15, c_scale_q15)` for the
/// high‑bit‑depth kernel family.
///
/// `BITS` is the input bit depth (10 / 12 / 14); `OUT_BITS` is the
/// target output range (8 for u8‑packed RGB, equal to `BITS` for
/// native‑depth `u16` output).
///
/// The scales are chosen so that after `((sample - y_off) * scale + RND) >> 15`
/// the result lies in `[0, (1 << OUT_BITS) - 1]` without further
/// downshifting. This keeps the fast path a single Q15 multiply for
/// both output widths.
///
/// - Full range: luma and chroma both use the same scale, mapping
///   `[0, in_max]` to `[0, out_max]`. Same shape as 8‑bit's
///   `(0, 1<<15, 1<<15)` for `BITS == OUT_BITS`.
/// - Limited range: luma maps `[16·k, 235·k]` to `[0, out_max]`,
///   chroma maps `[16·k, 240·k]` to `[0, out_max]`, where
///   `k = 1 << (BITS - 8)`. Matches FFmpeg's `AVCOL_RANGE_MPEG`
///   semantics.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn range_params_n<const BITS: u32, const OUT_BITS: u32>(
  full_range: bool,
) -> (i32, i32, i32) {
  let in_max: i64 = (1i64 << BITS) - 1;
  let out_max: i64 = (1i64 << OUT_BITS) - 1;
  if full_range {
    // `scale = round((out_max << 15) / in_max)`. For `BITS == OUT_BITS`
    // the quotient is exactly `1 << 15` (no rounding needed); for
    // 10‑bit→8‑bit it's `(255 << 15) / 1023 ≈ 8167.5`, which rounds to 8168.
    let scale = ((out_max << 15) + in_max / 2) / in_max;
    (0, scale as i32, scale as i32)
  } else {
    let y_off = 16i32 << (BITS - 8);
    let y_range: i64 = 219i64 << (BITS - 8);
    let c_range: i64 = 224i64 << (BITS - 8);
    let y_scale = ((out_max << 15) + y_range / 2) / y_range;
    let c_scale = ((out_max << 15) + c_range / 2) / c_range;
    (y_off, y_scale as i32, c_scale as i32)
  }
}

/// Range-scaling params: `(y_off, y_scale_q15, c_scale_q15)`.
///
/// Full range: no offset, unit scales (Q15 = 2^15).
///
/// Limited range: map Y from `[16, 235]` to `[0, 255]` via
/// `y_scaled = (y - 16) * (255 / 219)`; map chroma from `[16, 240]`
/// to `[0, 255]` via `c_scaled = (c - 128) * (255 / 224)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn range_params(full_range: bool) -> (i32, i32, i32) {
  if full_range {
    (0, 1 << 15, 1 << 15)
  } else {
    //  255 / 219 ≈ 1.164383; * 2^15 ≈ 38142.
    //  255 / 224 ≈ 1.138393; * 2^15 ≈ 37306.
    (16, 38142, 37306)
  }
}

/// Q15 YUV → RGB coefficients for a given matrix.
///
/// Full generalized 3×3 matrix:
/// - `R = Y + r_u·u_d + r_v·v_d`
/// - `G = Y + g_u·u_d + g_v·v_d`
/// - `B = Y + b_u·u_d + b_v·v_d`
///
/// where `u_d = U - 128`, `v_d = V - 128`. Standard matrices
/// (BT.601, BT.709, BT.2020-NCL, SMPTE 240M, FCC) have sparse layout
/// with `r_u = b_v = 0`; YCgCo uses all six entries.
pub(super) struct Coefficients {
  r_u: i32,
  r_v: i32,
  g_u: i32,
  g_v: i32,
  b_u: i32,
  b_v: i32,
}

impl Coefficients {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn for_matrix(m: ColorMatrix) -> Self {
    match m {
      // BT.601: r_v=1.402, g_u=-0.344136, g_v=-0.714136, b_u=1.772.
      ColorMatrix::Bt601 | ColorMatrix::Fcc => Self {
        r_u: 0,
        r_v: 45941,
        g_u: -11277,
        g_v: -23401,
        b_u: 58065,
        b_v: 0,
      },
      // BT.709: r_v=1.5748, g_u=-0.1873, g_v=-0.4681, b_u=1.8556.
      ColorMatrix::Bt709 => Self {
        r_u: 0,
        r_v: 51606,
        g_u: -6136,
        g_v: -15339,
        b_u: 60808,
        b_v: 0,
      },
      // BT.2020-NCL: r_v=1.4746, g_u=-0.164553, g_v=-0.571353, b_u=1.8814.
      ColorMatrix::Bt2020Ncl => Self {
        r_u: 0,
        r_v: 48325,
        g_u: -5391,
        g_v: -18722,
        b_u: 61653,
        b_v: 0,
      },
      // SMPTE 240M: r_v=1.576, g_u=-0.2253, g_v=-0.4767, b_u=1.826.
      ColorMatrix::Smpte240m => Self {
        r_u: 0,
        r_v: 51642,
        g_u: -7383,
        g_v: -15620,
        b_u: 59834,
        b_v: 0,
      },
      // YCgCo per H.273 MatrixCoefficients = 8.
      //   U plane → Cg, V plane → Co (biased by 128 each).
      //   R = Y - (Cg - 128) + (Co - 128) = Y - u_d + v_d
      //   G = Y + (Cg - 128)              = Y + u_d
      //   B = Y - (Cg - 128) - (Co - 128) = Y - u_d - v_d
      // Each coefficient is ±1.0 → ±32768 in Q15.
      ColorMatrix::YCgCo => Self {
        r_u: -32768,
        r_v: 32768,
        g_u: 32768,
        g_v: 0,
        b_u: -32768,
        b_v: -32768,
      },
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn r_u(&self) -> i32 {
    self.r_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn r_v(&self) -> i32 {
    self.r_v
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn g_u(&self) -> i32 {
    self.g_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn g_v(&self) -> i32 {
    self.g_v
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn b_u(&self) -> i32 {
    self.b_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn b_v(&self) -> i32 {
    self.b_v
  }
}

// ---- BGR ↔ RGB byte swap ------------------------------------------------

/// Swaps the outer two channels of each packed RGB / BGR triple
/// (byte 0 ↔ byte 2), leaving the middle byte (G) untouched.
///
/// This is the shared implementation behind both `bgr_to_rgb_row` and
/// `rgb_to_bgr_row` — the transformation is a self‑inverse.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");
  for x in 0..width {
    let i = x * 3;
    output[i] = input[i + 2];
    output[i + 1] = input[i + 1];
    output[i + 2] = input[i];
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
