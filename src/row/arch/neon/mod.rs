//! aarch64 NEON backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_aarch64_feature_detected!("neon")` returns true (runtime,
//! std‑gated) or `cfg!(target_feature = "neon")` evaluates true
//! (compile‑time, no‑std). The kernel itself carries
//! `#[target_feature(enable = "neon")]` so its intrinsics execute in
//! an explicitly NEON‑enabled context rather than one merely inherited
//! from the aarch64 target's default feature set.
//!
//! # Numerical contract
//!
//! The kernel uses i32 widening multiplies and the same
//! `(prod + (1 << 14)) >> 15` Q15 rounding as
//! [`crate::row::scalar::yuv_420_to_rgb_row`], so output is
//! **byte‑identical** to the scalar reference for every input. This is
//! asserted by the equivalence tests below.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`vld1q_u8`) + 8 U (`vld1_u8`) + 8 V (`vld1_u8`).
//! 2. Widen U/V to i16, subtract 128 → `u_i16`, `v_i16`.
//! 3. Widen to i32 and apply `c_scale` (Q15) → `u_d`, `v_d` (i32x4 x 2).
//! 4. Per channel C ∈ {R, G, B}:
//!    `C_chroma = (C_u * u_d + C_v * v_d + RND) >> 15` in i32,
//!    narrow‑saturate to i16x8 (8 lanes = 8 chroma pairs).
//! 5. Duplicate each chroma lane into its Y‑pair slot with
//!    `vzip1q_s16` / `vzip2q_s16` → 16 i16 chroma lanes matching the
//!    16 Y lanes (nearest‑neighbor upsample in registers, no memory
//!    traffic).
//! 6. Y path: `(Y - y_off) * y_scale + RND >> 15` in i32, narrow to i16.
//! 7. Saturating add Y + chroma per channel → i16x16.
//! 8. Saturate‑narrow to u8x16 and interleave with `vst3q_u8`.

// Used by the shared helpers below (`clamp_u16_max`, `q15_shift`,
// `chroma_*`, `scale_y_*`, `bswap_*`) when at least one of the
// YUV-class or `rgb` source families is enabled. Under feature subsets
// where no NEON kernel needs these intrinsics (e.g. bayer-only,
// gbr-only, gray-only, …), this import would otherwise be flagged
// unused. Submodules reach these intrinsics through `use super::*`.
#[cfg(any(
  feature = "rgb",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

#[allow(unused_imports)]
pub(super) use crate::{ColorMatrix, row::scalar};

// `cfg(miri)` fallbacks for the specialized NEON intrinsics Miri cannot
// execute (horizontal reduce, float→int convert, saturating narrow).
// Declared unconditionally because the always-compiled `hsv` kernel
// uses them; every NEON kernel reaches the helpers through this
// re-export. Production (`not(miri)`) is byte-identical to the raw
// intrinsics.
pub(crate) mod miri_compat;
// Re-exported for the `use super::*` kernels; feature subsets whose
// kernels reach the helpers through an explicit `use super::miri_compat::*`
// (or use none of these intrinsics) leave this glob unconsumed.
#[allow(unused_imports)]
pub(super) use miri_compat::*;

// Consumers: source families with a source-α channel (`gbr` Gbrap,
// `yuv-444-packed` AYUV64, `yuva` planar α).
#[cfg(any(feature = "gbr", feature = "yuv-444-packed", feature = "yuva"))]
pub(crate) mod alpha_extract;
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  )
))]
pub(crate) mod area_reduce;
#[cfg(feature = "yuv-444-packed")]
mod ayuv64;
// Centered (#302 phase-0.5) 2:1 horizontal chroma-upsample reconstruct
// kernels (u8 planar, u16 planar, interleaved-UV P-format). The centered
// reconstruct is reachable only when `yuv-planar` is present — packed /
// semi-solo builds stay co-sited and never reconstruct — so the module is
// gated `yuv-planar`; the per-kernel gates below add `yuv-semi-planar` for
// the P-format kernel and carve out the u16 / P-format kernels elsewhere.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
pub(crate) mod chroma_upsample;
pub(crate) mod endian;
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  )
))]
pub(crate) mod filter_reduce;
#[cfg(feature = "gray")]
mod gray;
mod hsv;
#[cfg(feature = "rgb-legacy")]
pub(crate) mod legacy_rgb;
#[cfg(feature = "mono")]
pub(crate) mod mono1bit;
#[cfg(feature = "rgb")]
mod packed_rgb;
#[cfg(feature = "rgb")]
mod packed_rgb_16bit;
#[cfg(feature = "rgb")]
mod packed_rgb_32bit;
#[cfg(feature = "rgb-float")]
mod packed_rgb_float;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_4_1_1;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_8bit;
#[cfg(feature = "mono")]
pub(crate) mod pal8;
#[cfg(feature = "gbr")]
mod planar_gbr;
#[cfg(feature = "gbr")]
mod planar_gbr_32bit;
#[cfg(feature = "gbr")]
mod planar_gbr_float;
#[cfg(feature = "gbr")]
mod planar_gbr_high_bit;
#[cfg(feature = "gbr")]
mod planar_gbr_msb;
#[cfg(feature = "yuv-semi-planar")]
mod semi_planar_8bit;
// The Pn 4:2:0 (P010/P012/P016) NEON kernels are consumed by
// `dispatch::yuv420::p{010,012,016}` — semi-planar P-formats. A single
// `yuv-semi-planar` gate keeps them reachable in a `yuv-semi-planar`-solo
// build (their scalar tails are widened to the same union).
#[cfg(feature = "yuv-semi-planar")]
mod subsampled_high_bit_pn_4_2_0;
// The Pn 4:4:4 NEON kernels are reachable from `dispatch::pn`
// (yuv-semi-planar only) as well as from `dispatch::yuv444::p{410,412,416}`
// (yuv-planar + yuv-semi-planar), so a single `yuv-semi-planar` gate
// suffices.
#[cfg(feature = "yuv-semi-planar")]
mod subsampled_high_bit_pn_4_4_4;
#[cfg(feature = "v210")]
mod v210;
#[cfg(feature = "yuv-444-packed")]
mod v30x;
#[cfg(feature = "yuv-444-packed")]
mod v410;
#[cfg(feature = "yuv-444-packed")]
mod vuya;
#[cfg(feature = "yuv-444-packed")]
mod xv36;
#[cfg(feature = "yuv-444-packed")]
mod xv48;
#[cfg(all(feature = "xyz", any(feature = "std", feature = "alloc")))]
pub(crate) mod xyz12;
#[cfg(feature = "y2xx")]
mod y216;
#[cfg(feature = "y2xx")]
mod y2xx;
// See `dispatch::mod.rs` for the consumer list.
#[cfg(any(
  feature = "gray",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
pub(crate) mod y_plane_to_luma_u16;
// The NEON variant of this file only hosts the `yuv_{420,444}p16_to_*`
// kernels (planar). The semi-planar `p16_to_*` kernels live in
// `subsampled_high_bit_pn_4_2_0` / `subsampled_high_bit_pn_4_4_4`. So
// a single `yuv-planar` gate suffices here — unlike the scalar variant.
#[cfg(feature = "yuv-planar")]
mod yuv_planar_16bit;
#[cfg(feature = "yuv-planar")]
mod yuv_planar_8bit;
#[cfg(feature = "yuv-planar")]
mod yuv_planar_high_bit;

#[cfg(any(feature = "gbr", feature = "yuv-444-packed", feature = "yuva"))]
pub(crate) use alpha_extract::*;
#[cfg(feature = "yuv-444-packed")]
pub(crate) use ayuv64::*;
#[cfg(feature = "gray")]
pub(crate) use gray::*;
pub(crate) use hsv::*;
#[cfg(feature = "rgb")]
pub(crate) use packed_rgb::*;
#[cfg(feature = "rgb")]
#[allow(unused_imports)]
pub(crate) use packed_rgb_16bit::*;
#[cfg(feature = "rgb")]
#[allow(unused_imports)]
pub(crate) use packed_rgb_32bit::*;
#[cfg(feature = "rgb-float")]
pub(crate) use packed_rgb_float::*;
#[cfg(feature = "yuv-packed")]
pub(crate) use packed_yuv_4_1_1::*;
#[cfg(feature = "yuv-packed")]
pub(crate) use packed_yuv_8bit::*;
#[cfg(feature = "gbr")]
pub(crate) use planar_gbr::*;
#[cfg(feature = "gbr")]
pub(crate) use planar_gbr_32bit::*;
#[cfg(feature = "gbr")]
pub(crate) use planar_gbr_float::*;
#[cfg(feature = "gbr")]
pub(crate) use planar_gbr_high_bit::*;
#[cfg(feature = "gbr")]
pub(crate) use planar_gbr_msb::*;
#[cfg(feature = "yuv-semi-planar")]
pub(crate) use semi_planar_8bit::*;
#[cfg(feature = "yuv-semi-planar")]
pub(crate) use subsampled_high_bit_pn_4_2_0::*;
#[cfg(feature = "yuv-semi-planar")]
pub(crate) use subsampled_high_bit_pn_4_4_4::*;
#[cfg(feature = "yuv-444-packed")]
pub(crate) use v30x::*;
#[cfg(feature = "v210")]
pub(crate) use v210::*;
#[cfg(feature = "yuv-444-packed")]
pub(crate) use v410::*;
#[cfg(feature = "yuv-444-packed")]
pub(crate) use vuya::*;
#[cfg(feature = "yuv-444-packed")]
pub(crate) use xv36::*;
#[cfg(feature = "yuv-444-packed")]
pub(crate) use xv48::*;
#[cfg(any(
  feature = "gray",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
pub(crate) use y_plane_to_luma_u16::*;
#[cfg(feature = "y2xx")]
pub(crate) use y2xx::*;
#[cfg(feature = "y2xx")]
pub(crate) use y216::*;
#[cfg(feature = "yuv-planar")]
pub(crate) use yuv_planar_8bit::*;
#[cfg(feature = "yuv-planar")]
pub(crate) use yuv_planar_16bit::*;
#[cfg(feature = "yuv-planar")]
pub(crate) use yuv_planar_high_bit::*;

// ---- Shared helpers (used across submodules) -------------------------

/// Clamps an i16x8 vector to `[0, max]` and reinterprets to u16x8.
/// Used by native-depth u16 output paths (10/12/14 bit) to avoid
/// `vqmovun_s16`'s u8 saturation. Reachable only from native-depth
/// YUV kernel families.
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[inline(always)]
pub(super) fn clamp_u16_max(v: int16x8_t, zero_v: int16x8_t, max_v: int16x8_t) -> uint16x8_t {
  unsafe { vreinterpretq_u16_s16(vminq_s16_compat(vmaxq_s16_compat(v, zero_v), max_v)) }
}

// The helpers below wrap NEON register‑only intrinsics (shifts, adds,
// multiplies, narrowing conversions, lane movers). None of them touch
// memory or take pointers, so there is no safety invariant to hoist to
// the caller — the functions themselves are safe. The `unsafe { ... }`
// blocks inside are only required because `core::arch::aarch64`
// intrinsics are marked `unsafe fn` in the standard library.
//
// `#[inline(always)]` guarantees these are inlined into the NEON‑
// enabled caller (`yuv_420_to_rgb_row` has
// `#[target_feature(enable = "neon")]`), so the intrinsics execute in
// a context where NEON is explicitly enabled — not just implicitly
// via the aarch64 target's default feature set.

/// `>>_a 15` shift (arithmetic, sign‑extending). Used by every NEON
/// YUV-class kernel after the Q15 multiply.
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[inline(always)]
pub(super) fn q15_shift(v: int32x4_t) -> int32x4_t {
  unsafe { vshrq_n_s32::<15>(v) }
}

/// Build an i16x8 channel chroma vector from the 8 paired i32 chroma
/// samples. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`.
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[inline(always)]
pub(super) fn chroma_i16x8(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d_lo: int32x4_t,
  v_d_lo: int32x4_t,
  u_d_hi: int32x4_t,
  v_d_hi: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_lo), vmulq_s32(cv, v_d_lo)),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_hi), vmulq_s32(cv, v_d_hi)),
      rnd,
    ));
    vcombine_s16(vqmovn_s32_compat(lo), vqmovn_s32_compat(hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` returned as i16x8 (8 Y pixels).
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[inline(always)]
pub(super) fn scale_y(
  y_i16: int16x8_t,
  y_off_v: int16x8_t,
  y_scale_v: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let shifted = vsubq_s16(y_i16, y_off_v);
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_low_s16(shifted)), y_scale_v),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_high_s16(shifted)), y_scale_v),
      rnd,
    ));
    vcombine_s16(vqmovn_s32_compat(lo), vqmovn_s32_compat(hi))
  }
}

// ===== 16-bit helpers =====================================================

/// Scale 8 u16 Y pixels to i16x8 for the 16-bit u8-output path.
///
/// Unsigned-widens via `vmovl_u16`, subtracts `y_off` in i32, multiplies
/// by `y_scale` (small for u8 output — no i32 overflow), Q15-shifts, and
/// narrows to i16x8 with `vqmovn_s32`.
#[cfg(any(
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[inline(always)]
pub(super) fn scale_y_u16_to_i16(
  y_vec: uint16x8_t,
  y_off_v: int32x4_t,
  y_scale_v: int32x4_t,
  rnd_v: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec)));
    let lo_s = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vsubq_s32(lo, y_off_v), y_scale_v),
      rnd_v,
    ));
    let hi_s = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vsubq_s32(hi, y_off_v), y_scale_v),
      rnd_v,
    ));
    vcombine_s16(vqmovn_s32_compat(lo_s), vqmovn_s32_compat(hi_s))
  }
}

/// `(cu*u_d + cv*v_d + RND) >> 15` in i64 for 4 chroma values → i32x4.
///
/// Used by the 16-bit u16-output path where `coeff * u_d` exceeds i32.
/// `vmull_s32` widens each 32x32 product to 64 bits, avoiding overflow.
#[cfg(any(
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[inline(always)]
pub(super) fn chroma_i64x4(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d: int32x4_t,
  v_d: int32x4_t,
  rnd64: int64x2_t,
) -> int32x4_t {
  unsafe {
    let sum_lo = vshrq_n_s64::<15>(vaddq_s64(
      vaddq_s64(
        vmull_s32(vget_low_s32(cu), vget_low_s32(u_d)),
        vmull_s32(vget_low_s32(cv), vget_low_s32(v_d)),
      ),
      rnd64,
    ));
    let sum_hi = vshrq_n_s64::<15>(vaddq_s64(
      vaddq_s64(
        vmull_s32(vget_high_s32(cu), vget_high_s32(u_d)),
        vmull_s32(vget_high_s32(cv), vget_high_s32(v_d)),
      ),
      rnd64,
    ));
    vcombine_s32(vmovn_s64(sum_lo), vmovn_s64(sum_hi))
  }
}

/// Scale 4 u16 Y pixels via i64 widening for the 16-bit u16-output path.
///
/// `(y - y_off) * y_scale` can reach ~2.35x10⁹ at 16-bit limited range,
/// overflowing i32. `vmull_s32` widens to i64 before the Q15 shift.
/// Input `y_u32` is already unsigned-widened and reinterpreted as i32.
#[cfg(any(
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[inline(always)]
pub(super) fn scale_y_u16_i64(
  y_i32: int32x4_t,
  y_off_v: int32x4_t,
  y_scale_d: int32x2_t,
  rnd64: int64x2_t,
) -> int32x4_t {
  unsafe {
    let sub = vsubq_s32(y_i32, y_off_v);
    let lo = vshrq_n_s64::<15>(vaddq_s64(vmull_s32(vget_low_s32(sub), y_scale_d), rnd64));
    let hi = vshrq_n_s64::<15>(vaddq_s64(vmull_s32(vget_high_s32(sub), y_scale_d), rnd64));
    vcombine_s32(vmovn_s64(lo), vmovn_s64(hi))
  }
}

// ---- BE helpers ----------------------------------------------------------

/// Compile-time host endianness. `true` on BE targets (e.g. `s390x`,
/// `powerpc`-BE), `false` on LE targets (e.g. `aarch64-apple-darwin`,
/// `x86_64`).
///
/// Used by the conditional byte-swap helpers below to decide whether a raw
/// NEON load already matches the wire endian. Without this, the helpers
/// would only correctly handle two of the four `host x wire` quadrants.
#[cfg(any(
  feature = "rgb",
  feature = "yuv-444-packed",
  feature = "yuv-semi-planar"
))]
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Conditionally byte-swap 8 u16 lanes in a NEON register so that the
/// returned value is in **host-native** byte order, regardless of the
/// host endianness.
///
/// The gate is `BE != HOST_NATIVE_BE`:
///
/// | wire `BE` | host       | gate    | action            |
/// |-----------|------------|---------|-------------------|
/// | `false`   | LE         | `false` | no swap (LE→LE)   |
/// | `false`   | BE         | `true`  | swap (LE→BE)      |
/// | `true`    | LE         | `true`  | swap (BE→LE)      |
/// | `true`    | BE         | `false` | no swap (BE→BE)   |
///
/// The unused branch is eliminated by the compiler — `BE` and
/// `HOST_NATIVE_BE` are both compile-time constants, so the gate folds.
///
/// Used by the packed YUV 4:4:4 kernels (XV36, AYUV64) after `vld4q_u16`
/// to correct samples loaded from a wire-encoded buffer.
///
/// Mirrors the same gate shape as the dispatcher routing fix and the
/// Ya16 SIMD gate — only swapping on `BE = true` (instead of on
/// `BE != HOST_NATIVE_BE`) double-swaps on BE hosts.
#[cfg(any(
  feature = "rgb",
  feature = "yuv-444-packed",
  feature = "yuv-semi-planar"
))]
#[inline(always)]
pub(super) unsafe fn bswap_u16x8_if_be<const BE: bool>(v: uint16x8_t) -> uint16x8_t {
  if BE != HOST_NATIVE_BE {
    unsafe { vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v))) }
  } else {
    v
  }
}

/// Conditionally byte-swap 4 u32 lanes in a NEON register so that the
/// returned value is in **host-native** byte order, regardless of the
/// host endianness.
///
/// Same `BE != HOST_NATIVE_BE` gate as [`bswap_u16x8_if_be`] — see that
/// helper for the truth table.
///
/// Used by the V410 kernel after `vld1q_u32` and by the packed 32-bit RGB
/// (Rgb96 / Rgba128) kernels after `vld3q_u32` / `vld4q_u32` to correct u32
/// words loaded from a wire-encoded buffer.
#[cfg(any(feature = "rgb", feature = "yuv-444-packed"))]
#[inline(always)]
pub(super) unsafe fn bswap_u32x4_if_be<const BE: bool>(v: uint32x4_t) -> uint32x4_t {
  if BE != HOST_NATIVE_BE {
    unsafe { vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(v))) }
  } else {
    v
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
