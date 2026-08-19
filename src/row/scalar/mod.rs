//! Scalar reference implementations of the row primitives.
//!
//! Always compiled. SIMD backends live in [`super::arch`] and dispatch
//! to these as their tail fallback. Per-call dispatch in
//! [`super`]`::{yuv_420_to_rgb_row, rgb_to_hsv_row}` picks the best
//! backend at the module boundary.
//!
//! # Rounding convention
//!
//! The crate uses two distinct rounding strategies — choose based on
//! whether the operation is *precision-critical* or *bookkeeping*:
//!
//! - **Q15 chroma + Y arithmetic (final RGB output)**: round-to-nearest,
//!   implemented as `(value + (1 << 14)) >> 15` (or via the `q15_shift`
//!   helper). Maximum error: ±0.5 LSB symmetric. Used in every YUV→RGB
//!   pixel computation across all formats × backends.
//!
//! - **Narrow→wider depth conversions** (e.g., 16-bit luma → 8-bit
//!   luma via `Y_u16 >> 8`, or 10-bit packed → 8-bit RGB via `>> 2`):
//!   plain truncation, no rounding bias. Maximum error: -0.5 to 0 LSB
//!   (uniformly downward bias). Used in every `*_to_luma_row` (u8
//!   variant) for high-bit-depth sources, and in the `X2RGB10`/`X2BGR10`
//!   → u8 RGB conversion at the last narrow step.
//!
//! The asymmetry is intentional: precision-critical arithmetic earns
//! the rounding bias's symmetric error bound; depth-conversion is
//! bookkeeping where consistent downward-truncation matches FFmpeg's
//! `swscale` behavior and preserves "no-clip-into-overflow" guarantees.
//! Cross-format consistency on this distinction is verified by the
//! per-arch SIMD-vs-scalar parity tests.

use crate::KernelMatrix;
#[cfg(feature = "yuv-planar")]
use crate::Primaries;

// Per-conversion-family submodules. Each holds a self-contained
// cluster of scalar reference kernels; `mod.rs` retains only the
// cross-cutting helpers (`clamp_u8`, `q15_*`, `bits_mask`,
// `Coefficients`, …) that every family pulls in.
// Consumers: source families with a source-α channel (`gbr` Gbrap,
// `gray` Ya8 / Ya16, `rgb` 16-bit RGBA at_3, `yuv-444-packed`
// AYUV64 / VUYA, `yuva` planar α).
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "rgb",
  feature = "yuv-444-packed",
  feature = "yuva",
))]
pub(crate) mod alpha_extract;
// Consumer: the fused-downscale engine (`crate::resample`), compiled
// under `any(std, alloc)`.
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
// Consumer: the separable filter resampler (`crate::resample::filter`),
// the signed twin of `area_reduce`; same 14-feature engine cascade.
#[cfg(feature = "yuv-444-packed")]
mod ayuv64;
#[cfg(feature = "bayer")]
mod bayer;
// Constant-luminance `YcCbcCrc` (BT.2020 CL, H.273 MatrixCoefficients = 13)
// non-affine decode. Gated like `ictcp`: the transcendental inverse-OETF /
// OETF need `std` or `alloc` (libm); the representative wiring targets the
// `yuv-planar` 4:4:4 family (#303).
#[cfg(all(feature = "yuv-planar", any(feature = "std", feature = "alloc")))]
pub(crate) mod cl;
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
pub(crate) mod gray;
#[cfg(feature = "gray")]
pub(crate) mod grayf16;
#[cfg(feature = "gray")]
pub(crate) mod grayf32;
mod hsv;
// ICtCp (BT.2100, H.273 MatrixCoefficients = 14) non-affine decode. Gated
// like `xyz12`: the transcendental EOTF/OETF need `std` or `alloc` (libm);
// the representative wiring targets the `yuv-planar` 4:4:4 family (#303).
#[cfg(all(feature = "yuv-planar", any(feature = "std", feature = "alloc")))]
pub(crate) mod ictcp;
// IPT-C2 (H.273 MatrixCoefficients = 15, the Dolby Vision Profile 5 base
// colour space) non-affine decode. Gated like `ictcp`: the transcendental PQ
// EOTF/OETF need `std` or `alloc` (libm); the representative wiring targets the
// `yuv-planar` 4:4:4 family (#303).
#[cfg(all(feature = "yuv-planar", any(feature = "std", feature = "alloc")))]
pub(crate) mod iptc2;
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
pub(crate) mod planar_gbr_32bit;
#[cfg(feature = "gbr")]
pub(crate) mod planar_gbr_f16;
#[cfg(feature = "gbr")]
pub(crate) mod planar_gbr_float;
#[cfg(feature = "gbr")]
pub(crate) mod planar_gbr_high_bit;
#[cfg(feature = "gbr")]
pub(crate) mod planar_gbr_msb;
mod rgb_expand;
#[cfg(feature = "yuv-semi-planar")]
mod semi_planar_8bit;
// SMPTE ST 2085 (H.273 MatrixCoefficients = 11, "Y'D'zD'x", the PQ-only
// non-affine X'Y'Z' colour-difference model) non-affine decode. Gated like
// `iptc2`: the transcendental PQ EOTF/OETF need `std` or `alloc` (libm); the
// representative wiring targets the `yuv-planar` 4:4:4 family (#303).
#[cfg(all(feature = "yuv-planar", any(feature = "std", feature = "alloc")))]
pub(crate) mod smpte2085;
// `subsampled_high_bit_pn` provides the scalar reference kernels for
// both the 4:2:0 (P010 / P012 / P016) and 4:4:4 (P410 / P412 / P416)
// families. All are semi-planar P-formats, so a single
// `yuv-semi-planar` module gate keeps the whole file reachable; the
// per-fn `any(yuv-planar, yuv-semi-planar)` gates on the 4:2:0 helpers
// also let them serve the `yuv-planar` planar oracles that compare
// against them.
#[cfg(feature = "yuv-semi-planar")]
mod subsampled_high_bit_pn;
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
#[cfg(all(feature = "xyz", any(feature = "std", feature = "alloc")))]
pub(crate) mod xyz12_constants;
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
#[cfg(feature = "gray")]
pub(crate) mod ya16;
#[cfg(feature = "gray")]
pub(crate) mod ya8;
#[cfg(feature = "gray")]
pub(crate) mod yaf16;
#[cfg(feature = "gray")]
pub(crate) mod yaf32;
// yuv_planar_16bit also contains the P016 semi-planar 4:2:0 / P216
// semi-planar 4:2:2 / P416 semi-planar 4:4:4 16-bit kernels (`p16_to_rgb*_row`),
// so compile whenever either `yuv-planar` or `yuv-semi-planar` is enabled.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod yuv_planar_16bit;
#[cfg(feature = "yuv-planar")]
mod yuv_planar_8bit;
#[cfg(feature = "yuv-planar")]
mod yuv_planar_high_bit;

// alpha_extract functions are imported directly by dispatch::alpha_extract
// via `crate::row::scalar::alpha_extract as scalar` (the module path).
// This glob re-exports into `crate::row::scalar::*` for Task 8+ callers;
// suppress unused-imports until then.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "rgb",
  feature = "yuv-444-packed",
  feature = "yuva",
))]
#[allow(unused_imports)]
pub(crate) use alpha_extract::*;
#[cfg(feature = "yuv-444-packed")]
pub(crate) use ayuv64::*;
#[cfg(feature = "bayer")]
pub(crate) use bayer::*;
// legacy_rgb functions are consumed by the dispatcher via `use crate::row::{..., scalar};`
// and called as `scalar::legacy_rgb::...`.
// This glob re-exports them into the scalar namespace for direct callers (SIMD tails, tests).
#[cfg(feature = "rgb-legacy")]
#[allow(unused_imports)]
pub(crate) use legacy_rgb::*;
// gray functions are consumed by dispatch::gray via `crate::row::scalar::gray as scalar`.
// This glob re-exports them into the scalar namespace for direct callers (SIMD tails, tests).
#[cfg(feature = "gray")]
#[allow(unused_imports)]
pub(crate) use gray::*;
#[cfg(feature = "gray")]
#[allow(unused_imports)]
pub(crate) use grayf16::*;
#[cfg(feature = "gray")]
#[allow(unused_imports)]
pub(crate) use grayf32::*;
pub(crate) use hsv::*;
// mono1bit functions are consumed by dispatch via the module path.
#[cfg(feature = "mono")]
#[allow(unused_imports)]
pub(crate) use mono1bit::*;
#[cfg(feature = "rgb")]
pub(crate) use packed_rgb::*;
#[cfg(feature = "rgb")]
pub(crate) use packed_rgb_16bit::*;
#[cfg(feature = "rgb")]
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
#[allow(unused_imports)]
pub(crate) use planar_gbr_f16::*;
#[cfg(feature = "gbr")]
#[allow(unused_imports)]
pub(crate) use planar_gbr_float::*;
#[cfg(feature = "gbr")]
pub(crate) use planar_gbr_high_bit::*;
#[cfg(feature = "gbr")]
pub(crate) use planar_gbr_msb::*;
// Same consumer set as the `rgb_expand` helpers themselves: every source
// family that fans an RGB row out to an RGBA row via Strategy A
// (Bayer is RGB-only, mono / rgb-float / rgb-legacy / xyz never go
// through the fan-out, so they're excluded).
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "gbr",
    feature = "gray",
    feature = "rgb",
    feature = "v210",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ),
))]
pub(crate) use rgb_expand::*;
#[cfg(feature = "yuv-semi-planar")]
pub(crate) use semi_planar_8bit::*;
#[cfg(feature = "yuv-semi-planar")]
pub(crate) use subsampled_high_bit_pn::*;
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
// `xyz12` and `xyz12_constants` are crate-internal modules; consumers (dispatcher
// + SIMD tails) reach in via `crate::row::scalar::xyz12::xyz12_to_rgb_row::<BE>`
// rather than a glob re-export, so the constants table and helpers stay
// addressable without polluting the scalar namespace.
#[cfg(feature = "y2xx")]
pub(crate) use y2xx::*;
#[cfg(feature = "y2xx")]
pub(crate) use y216::*;
#[cfg(feature = "gray")]
#[allow(unused_imports)]
pub(crate) use ya8::*;
#[cfg(feature = "gray")]
#[allow(unused_imports)]
pub(crate) use ya16::*;
#[cfg(feature = "gray")]
#[allow(unused_imports)]
pub(crate) use yaf16::*;
#[cfg(feature = "gray")]
#[allow(unused_imports)]
pub(crate) use yaf32::*;
#[cfg(feature = "yuv-planar")]
pub(crate) use yuv_planar_8bit::*;
// The file is compiled whenever either family is on. Its public items
// are gated per-family: `yuv_{420,444}p16_to_*` need `yuv-planar`,
// while the P016 semi-planar `p16_to_*` kernels need `yuv-semi-planar`.
// Re-export under the same union so a `yuv-semi-planar`-solo build
// still reaches the `p16_to_*` glob (the planar items simply don't
// exist there, so the glob carries only what is compiled).
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
pub(crate) use yuv_planar_16bit::*;
#[cfg(feature = "yuv-planar")]
pub(crate) use yuv_planar_high_bit::*;

// ---- Shared scalar helpers (used across all conversion families) -------

/// Reads one `u16` from the byte address `ptr` in the endianness
/// indicated by `BE`. `BE = false` → little-endian (native v210/Y2xx
/// on-wire format); `BE = true` → big-endian. The unused branch is
/// eliminated by the compiler when the caller is monomorphized.
///
/// **Target-endian aware** — this matches the SIMD `load_endian_u16x*`
/// helpers' semantics: `u16::from_be_bytes` / `u16::from_le_bytes`
/// each emit a `bswap` only when the source byte order differs from
/// the host CPU's native order. On a BE host the `BE = true` branch
/// is a plain load (no swap) and the `BE = false` branch swaps; on
/// an LE host the polarity reverses. This is the strict-superset-of-
/// bugs alternative to a naive `if BE { x.swap_bytes() }` pattern,
/// which would corrupt rows on s390x / other BE hosts.
///
/// # Safety
///
/// `ptr` must point to at least 2 readable bytes.
#[cfg(feature = "y2xx")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) unsafe fn load_endian_u16<const BE: bool>(ptr: *const u8) -> u16 {
  let bytes = unsafe { [*ptr, *ptr.add(1)] };
  if BE {
    u16::from_be_bytes(bytes)
  } else {
    u16::from_le_bytes(bytes)
  }
}

/// Reads one `u32` from the byte address `ptr` in the endianness
/// indicated by `BE`. `BE = false` → little-endian; `BE = true` →
/// big-endian. The unused branch is eliminated by the compiler when
/// the caller is monomorphized.
///
/// **Target-endian aware** — `u32::from_be_bytes` / `u32::from_le_bytes`
/// each emit a `bswap` only when the source byte order differs from
/// the host CPU's native order, matching the SIMD `load_endian_u32x*`
/// helpers. See [`load_endian_u16`] for the full target-endian
/// contract.
///
/// # Safety
///
/// `ptr` must point to at least 4 readable bytes.
#[cfg(feature = "v210")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) unsafe fn load_endian_u32<const BE: bool>(ptr: *const u8) -> u32 {
  let bytes = unsafe { [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)] };
  if BE {
    u32::from_be_bytes(bytes)
  } else {
    u32::from_le_bytes(bytes)
  }
}

#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn clamp_u8(v: i32) -> u8 {
  v.clamp(0, 255) as u8
}

/// Normalize a `u16` sample (just read host-native from memory) to the
/// host-native interpretation of the source byte order indicated by `BE`.
/// `BE = false` → little-endian source; `BE = true` → big-endian source.
/// The `if BE` branch is dead-code-eliminated per monomorphization, so
/// the matching-endian path is a zero-overhead no-op.
///
/// **Target-endian aware** — matches the SIMD `load_endian_u16x*::<BE>`
/// helpers' semantics: `u16::from_be` / `u16::from_le` each emit a
/// `bswap` only when the source byte order differs from the host CPU's
/// native order. On a BE host the `BE = true` branch is a plain pass-
/// through (no swap) and the `BE = false` branch swaps; on an LE host
/// the polarity reverses. This is the strict-superset-of-bugs
/// alternative to a naive `if BE { v.swap_bytes() }` pattern, which
/// would corrupt rows on s390x / other BE hosts.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar", feature = "yuva",))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn load_u16<const BE: bool>(v: u16) -> u16 {
  if BE { u16::from_be(v) } else { u16::from_le(v) }
}

/// `(sample * scale_q15 + RND) >> 15`. With input masked to BITS,
/// the `sample * scale` product cannot overflow i32 for any
/// reasonable `OUT_BITS ≤ 16`, so plain arithmetic is sufficient.
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn q15_scale(sample: i32, scale_q15: i32) -> i32 {
  (sample * scale_q15 + (1 << 14)) >> 15
}

/// `(c_u * u_d + c_v * v_d + RND) >> 15`. Chroma sum max ≈ 10⁹ for
/// 14‑bit masked input, well within i32.
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn q15_chroma(c_u: i32, u_d: i32, c_v: i32, v_d: i32) -> i32 {
  (c_u * u_d + c_v * v_d + (1 << 14)) >> 15
}

/// `(c_u * u_d + c_v * v_d + RND) >> 15` computed in i64. Chroma sum
/// max ≈ 4.3·10⁹ at 16-bit limited range — above i32 but well within
/// i64. Result after the shift is bounded by ~130 000 so the final
/// `as i32` narrow is lossless.
#[cfg(any(
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
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
#[cfg(any(
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn q15_scale64(sample: i32, scale_q15: i32) -> i32 {
  (((sample as i64) * (scale_q15 as i64) + (1 << 14)) >> 15) as i32
}

/// Compile‑time sample mask for `BITS`: `(1 << BITS) - 1` as `u16`.
/// Returns `0x03FF` for 10‑bit, `0x0FFF` for 12‑bit, `0x3FFF` for
/// 14‑bit. SIMD backends splat this into a vector constant and AND
/// every load against it.
#[cfg(any(feature = "gray", feature = "yuv-planar", feature = "yuva"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn bits_mask<const BITS: u32>() -> u16 {
  ((1u32 << BITS) - 1) as u16
}

/// Chroma bias for input bit depth `BITS` — `128 << (BITS - 8)`.
/// 128 for 8‑bit, 512 for 10‑bit, 2048 for 12‑bit, 8192 for 14‑bit.
/// Exposed at module visibility so SIMD backends can reuse it.
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
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
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
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
#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
pub(crate) struct Coefficients {
  r_u: i32,
  r_v: i32,
  g_u: i32,
  g_v: i32,
  b_u: i32,
  b_v: i32,
}

#[cfg(any(
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
impl Coefficients {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn for_matrix(m: KernelMatrix) -> Self {
    match m {
      // BT.601: r_v=1.402, g_u=-0.344136, g_v=-0.714136, b_u=1.772.
      // SMPTE 170M (BT.601 525-line) and BT.470 System BG (BT.601
      // 625-line) carry the identical Kr=0.299/Kb=0.114 coefficients, so
      // they share this arm exactly; FCC (Kr=0.30/Kb=0.11) is a close
      // numerical approximation grouped here historically.
      KernelMatrix::Bt601 | KernelMatrix::Fcc | KernelMatrix::Smpte170M | KernelMatrix::Bt470Bg => {
        Self {
          r_u: 0,
          r_v: 45941,
          g_u: -11277,
          g_v: -23401,
          b_u: 58065,
          b_v: 0,
        }
      }
      // BT.709: r_v=1.5748, g_u=-0.1873, g_v=-0.4681, b_u=1.8556.
      KernelMatrix::Bt709 => Self {
        r_u: 0,
        r_v: 51606,
        g_u: -6136,
        g_v: -15339,
        b_u: 60808,
        b_v: 0,
      },
      // BT.2020-NCL: r_v=1.4746, g_u=-0.164553, g_v=-0.571353, b_u=1.8814.
      KernelMatrix::Bt2020Ncl => Self {
        r_u: 0,
        r_v: 48325,
        g_u: -5391,
        g_v: -18722,
        b_u: 61653,
        b_v: 0,
      },
      // SMPTE 240M: r_v=1.576, g_u=-0.2253, g_v=-0.4767, b_u=1.826.
      // Coefficients are taken from the SMPTE 240M-1999 published rounded
      // table values, NOT re-derived from KR/KB. Re-derivation from
      // KR=0.212, KB=0.087, KG=0.701 yields g_u ≈ -0.2266 (Q15 ≈ -7423),
      // which differs by ~0.13% (~43 LSB pre-Q15-shift). This is well
      // within rounding tolerance and matches the standard's published
      // text — do not "fix" to the analytic value without coordinating
      // with downstream pipelines that also use the published table.
      KernelMatrix::Smpte240m => Self {
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
      KernelMatrix::YCgCo => Self {
        r_u: -32768,
        r_v: 32768,
        g_u: 32768,
        g_v: 0,
        b_u: -32768,
        b_v: -32768,
      },
      // The two matrices whose coefficients this primaries-blind entry point
      // cannot derive, spelled out rather than left to a wildcard so the
      // BT.709 result is a decision on the record and not a fallback:
      //
      // - `Unspecified` is `Matrix`'s own default and reaches the kernels
      //   routinely. mediaframe documents the resolution as the consumer's,
      //   and pixon's is BT.709 unconditionally (FFmpeg infers BT.709 for
      //   `height >= 720` and BT.601 below; a row kernel has no height).
      // - `ChromaDerivedNcl` is *derived* from the signalled primaries, which
      //   this entry point is not given — see
      //   [`for_matrix_with_primaries`](Self::for_matrix_with_primaries),
      //   which resolves it properly and lands back here only when the
      //   primaries carry no chromaticities.
      //
      // There is no `_` arm: `KernelMatrix` is a closed vocabulary, so this
      // match is exhaustiveness-checked and a new coefficient set upstream
      // becomes a compile error here rather than a silent BT.709 picture.
      KernelMatrix::Unspecified | KernelMatrix::ChromaDerivedNcl => Self {
        r_u: 0,
        r_v: 51606,
        g_u: -6136,
        g_v: -15339,
        b_u: 60808,
        b_v: 0,
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

// ---- Chromaticity-derived (H.273 MatrixCoefficients = 12) ----------------

/// Chromaticity-derived **non-constant-luminance** luma weights
/// `(Kr, Kg, Kb)` for a set of colour [`Primaries`], per ITU-T H.273
/// `MatrixCoefficients = 12` ([`KernelMatrix::ChromaDerivedNcl`]).
///
/// The weights are the **Y row** of the RGB→XYZ matrix built from the
/// primaries' CIE 1931 `xy` chromaticities and reference white point — the
/// standard primary-scaling construction (ITU-R BT.709-6 §3 / BT.2020-2;
/// identical math to `examples/derive_xyz_matrices.rs`). Solving
/// `M · S = W_xyz` (with the unscaled primary `XYZ` vectors as the columns
/// of `M` and the white point's luminance normalised to 1) yields
/// `S = (Kr, Kg, Kb)`, which therefore sums to 1.
///
/// Returns [`None`] for primaries that carry no chromaticities / white point
/// ([`Primaries::Unspecified`] / [`Primaries::Unknown`]), or in the singular
/// case (unreachable for any real, non-degenerate primary set).
///
/// All arithmetic is `f64` using only `+ - * /` — no transcendental
/// functions — so it stays `no_std` / `libm`-free. It is evaluated once per
/// frame at coefficient-resolution time, never per pixel.
#[cfg(feature = "yuv-planar")]
pub(crate) fn chroma_derived_luma_weights(primaries: &Primaries) -> Option<(f64, f64, f64)> {
  let rgb = primaries.chromaticities()?;
  let white = primaries.white_point()?;
  // ST 2086 raw units → CIE `xy` (raw / 50000.0), then the unscaled primary
  // `XYZ` at unit luminance: `(x/y, 1, (1 - x - y) / y)`.
  let xyz = |raw_x: u32, raw_y: u32| {
    let x = raw_x as f64 / 50_000.0;
    let y = raw_y as f64 / 50_000.0;
    [x / y, 1.0, (1.0 - x - y) / y]
  };
  let r = xyz(rgb[0].x(), rgb[0].y());
  let g = xyz(rgb[1].x(), rgb[1].y());
  let b = xyz(rgb[2].x(), rgb[2].y());
  let w = xyz(white.x(), white.y());

  // `M` has the unscaled primary `XYZ` vectors as columns; solve
  // `M · S = W_xyz` for `S = (Kr, Kg, Kb)` via the cofactor inverse.
  let m = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
  let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  let det_abs = if det < 0.0 { -det } else { det };
  if det_abs < 1e-30 {
    return None;
  }
  let inv_det = 1.0 / det;
  // Rows of `M^-1` (adjugate / det), then `S = M^-1 · W_xyz`.
  let i00 = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
  let i01 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
  let i02 = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
  let i10 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
  let i11 = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
  let i12 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;
  let i20 = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
  let i21 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
  let i22 = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
  let kr = i00 * w[0] + i01 * w[1] + i02 * w[2];
  let kg = i10 * w[0] + i11 * w[1] + i12 * w[2];
  let kb = i20 * w[0] + i21 * w[1] + i22 * w[2];
  Some((kr, kg, kb))
}

#[cfg(feature = "yuv-planar")]
impl Coefficients {
  /// Q15 round-half-away-from-zero of a real coefficient `c` (i.e.
  /// `round(c · 32768)`). Avoids `f64::round` so the derivation needs no
  /// `libm` under `no_std`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn q15(c: f64) -> i32 {
    if c < 0.0 {
      (c * 32768.0 - 0.5) as i32
    } else {
      (c * 32768.0 + 0.5) as i32
    }
  }

  /// Builds the YCbCr→RGB affine [`Coefficients`] from non-constant-
  /// luminance luma weights `(kr, kg, kb)` using the standard BT.601 /
  /// BT.709-form relations:
  ///
  /// ```text
  ///   r_u = 0          r_v = 2·(1 - kr)
  ///   g_u = -2·(1 - kb)·kb / kg     g_v = -2·(1 - kr)·kr / kg
  ///   b_u = 2·(1 - kb)             b_v = 0
  /// ```
  ///
  /// Each entry is rounded to Q15. `kg` is non-zero for every real primary
  /// set, so the division is well-defined.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn from_luma_weights(kr: f64, kg: f64, kb: f64) -> Self {
    Self {
      r_u: 0,
      r_v: Self::q15(2.0 * (1.0 - kr)),
      g_u: Self::q15(-2.0 * (1.0 - kb) * kb / kg),
      g_v: Self::q15(-2.0 * (1.0 - kr) * kr / kg),
      b_u: Self::q15(2.0 * (1.0 - kb)),
      b_v: 0,
    }
  }

  /// [`for_matrix`](Self::for_matrix), extended to resolve
  /// [`KernelMatrix::ChromaDerivedNcl`] (ITU-T H.273 `MatrixCoefficients =
  /// 12`) from the signalled colour [`Primaries`].
  ///
  /// `ChromaDerivedNcl` is **affine**: its `Kr` / `Kb` are *derived* from
  /// the primaries' chromaticities ([`chroma_derived_luma_weights`]) and
  /// then drive the same standard YCbCr formula as the hard-coded matrices
  /// (via [`from_luma_weights`](Self::from_luma_weights)). Every other
  /// matrix ignores `primaries` and resolves byte-identically to
  /// [`for_matrix`](Self::for_matrix).
  ///
  /// `ChromaDerivedNcl` with primaries that carry no chromaticities
  /// (`Unspecified` / `Unknown`) falls back to the BT.709 coefficients
  /// [`for_matrix`](Self::for_matrix) already returns for that matrix,
  /// preserving the prior behaviour.
  ///
  /// Cost is one 3×3 solve per call — resolved per row/frame, never per
  /// pixel — so it is negligible against the conversion itself.
  ///
  /// # SMPTE ST 428-1 interpretation (#310)
  ///
  /// This function always derives from the **FFmpeg-tabulated** chromaticities
  /// `Primaries::chromaticities` reports, including for `Primaries::SmpteSt428`
  /// (FFmpeg's D-Cinema RGB values). The alternative CIE-XYZ reading of
  /// ST 428-1 — where the data *is* CIE XYZ and a YCbCr `ChromaDerivedNcl`
  /// matrix is meaningless — is enforced *upstream* by the sink's
  /// `St428Interpretation::CieXyz` guard, which rejects that combination before
  /// this derivation is reached; it is intentionally not re-checked here. A
  /// full XYZ↔RGB matrix derivation (tabulated → real matrix vs CIE-XYZ →
  /// identity) is future work with no consumer today.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn for_matrix_with_primaries(matrix: KernelMatrix, primaries: &Primaries) -> Self {
    match matrix {
      KernelMatrix::ChromaDerivedNcl => match chroma_derived_luma_weights(primaries) {
        Some((kr, kg, kb)) => Self::from_luma_weights(kr, kg, kb),
        None => Self::for_matrix(KernelMatrix::Bt709),
      },
      other => Self::for_matrix(other),
    }
  }
}

// ---- BGR ↔ RGB byte swap ------------------------------------------------

/// Swaps the outer two channels of each packed RGB / BGR triple
/// (byte 0 ↔ byte 2), leaving the middle byte (G) untouched.
///
/// This is the shared implementation behind both `bgr_to_rgb_row` and
/// `rgb_to_bgr_row` — the transformation is a self‑inverse.
#[cfg(feature = "rgb")]
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
