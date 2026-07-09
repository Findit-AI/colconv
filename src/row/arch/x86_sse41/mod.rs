//! x86_64 SSE4.1 backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher as a fallback when AVX2 is
//! not available. SSE4.1 is a wide baseline on x86 (Penryn and newer,
//! ~2008), so this covers essentially all x86 hardware still in
//! production use that lacks AVX2.
//!
//! The kernel carries `#[target_feature(enable = "sse4.1")]` so its
//! intrinsics execute in an explicitly feature‑enabled context. The
//! shared [`super::x86_common::write_rgb_16`] helper uses SSSE3
//! (`_mm_shuffle_epi8`), which is a subset of SSE4.1 and thus
//! available here.
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON and AVX2 backends.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`_mm_loadu_si128`) + 8 U + 8 V (low 8 bytes of each
//!    via `_mm_loadl_epi64`).
//! 2. Widen U, V to i16x8 (`_mm_cvtepu8_epi16`), subtract 128.
//! 3. Split each i16x8 into two i32x4 halves and apply `c_scale`.
//! 4. Per channel C ∈ {R, G, B}: `(C_u*u_d + C_v*v_d + RND) >> 15` in
//!    i32, narrow‑saturate to i16x8.
//! 5. Nearest‑neighbor chroma upsample: `_mm_unpacklo_epi16` /
//!    `_mm_unpackhi_epi16` duplicate each of 8 chroma lanes into its
//!    pair slot → two i16x8 vectors covering 16 Y lanes. No lane‑
//!    crossing fixups are needed at 128 bits.
//! 6. Y path: widen low/high 8 Y to i16x8, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel.
//! 8. Saturate‑narrow to u8x16 per channel, then interleave via
//!    `super::x86_common::write_rgb_16`.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

#[allow(unused_imports)]
#[cfg(feature = "rgb")]
pub(super) use crate::row::arch::x86_common::{
  abgr_to_rgb_16_pixels, abgr_to_rgba_4_pixels, argb_to_rgb_16_pixels, argb_to_rgba_4_pixels,
  bgra_to_rgb_16_pixels, bgrx_to_rgba_4_pixels, drop_alpha_16_pixels, rgbx_to_rgba_4_pixels,
  swap_rb_16_pixels, swap_rb_alpha_4_pixels, x2bgr10_to_rgb_16_pixels, x2bgr10_to_rgb_u16_8_pixels,
  x2bgr10_to_rgba_16_pixels, x2rgb10_to_rgb_16_pixels, x2rgb10_to_rgb_u16_8_pixels,
  x2rgb10_to_rgba_16_pixels, xbgr_to_rgba_4_pixels, xrgb_to_rgba_4_pixels,
};
#[allow(unused_imports)]
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "rgb-legacy",
  feature = "mono",
  feature = "rgb",
  feature = "yuv-packed",
  feature = "gbr",
  feature = "yuv-semi-planar",
  feature = "yuv-planar",
  feature = "y2xx",
  feature = "xyz",
))]
pub(super) use crate::row::arch::x86_common::{write_rgb_16, write_rgba_16};
#[allow(unused_imports)]
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "rgb-legacy",
  feature = "mono",
  feature = "rgb",
  feature = "gbr",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "y2xx",
))]
pub(super) use crate::row::arch::x86_common::{write_rgb_u16_8, write_rgba_u16_8};
#[allow(unused_imports)]
pub(super) use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{deinterleave_rgb_16, rgb_to_hsv_16_pixels, rgb_to_luma_16_pixels},
    scalar,
  },
};

#[cfg(any(feature = "gbr", feature = "yuv-444-packed", feature = "yuva"))]
mod alpha_extract;
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
// kernels; see the NEON twin for the design and the `yuv-planar` module
// gate; per-kernel gates carve out the rest.
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
// Both the 4:2:0 (P010/P012/P016) and 4:4:4 (P410/P412/P416) kernels
// are semi-planar P-formats consumed by `dispatch::yuv420::p{010,012,016}`
// and `dispatch::pn` respectively, so a single `yuv-semi-planar` gate
// keeps each reachable in a `yuv-semi-planar`-solo build.
#[cfg(feature = "yuv-semi-planar")]
mod subsampled_high_bit_pn_4_2_0;
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
#[cfg(any(
  feature = "gray",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
mod y_plane_to_luma_u16;
// Semi-planar `p16_to_*` lives in `subsampled_high_bit_pn_*`; this file
// only hosts the planar `yuv_{420,444}p16_to_*` kernels.
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
#[cfg(feature = "rgb-legacy")]
#[allow(unused_imports)] // dispatcher wired in Task 5
pub(crate) use legacy_rgb::*;
#[cfg(feature = "mono")]
pub(crate) use mono1bit::*;
#[cfg(feature = "rgb")]
pub(crate) use packed_rgb::*;
#[cfg(feature = "rgb")]
#[allow(unused_imports)] // dispatcher wired in later task
pub(crate) use packed_rgb_16bit::*;
#[cfg(feature = "rgb")]
#[allow(unused_imports)] // dispatcher wired in later task
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
#[allow(unused_imports)] // dispatcher wired in Task 8 (MixedSinker)
pub(crate) use planar_gbr_float::*;
#[cfg(feature = "gbr")]
#[allow(unused_imports)] // dispatcher wired in Task 9
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

/// Clamps an i16x8 vector to `[0, max]` for native-depth u16 output
/// paths (10/12/14 bit). `_mm_packus_epi16` would clip to u8, so we
/// use explicit min/max with a caller-provided `max`.
#[cfg(any(
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "v210",
  feature = "yuv-444-packed",
  feature = "y2xx",
))]
#[inline(always)]
pub(super) fn clamp_u16_max(v: __m128i, zero_v: __m128i, max_v: __m128i) -> __m128i {
  unsafe { _mm_min_epi16(_mm_max_epi16(v, zero_v), max_v) }
}

/// Deinterleaves 16 `u16` elements at `ptr` (`[U0, V0, U1, V1, …,
/// U7, V7]`) into `(u_vec, v_vec)` where each vector holds 8 packed
/// `u16` samples.
///
/// Each of the two 128‑bit loads is byte‑shuffled via
/// `_mm_shuffle_epi8` so that U samples land in the low 64 bits and
/// V samples in the high 64. Then `_mm_unpacklo_epi64` /
/// `_mm_unpackhi_epi64` combine the two halves into full u16×8
/// vectors. 2 loads + 2 shuffles + 2 unpacks = 6 ops.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes (16 `u16`
/// elements). Caller's `target_feature` must include SSSE3 (via
/// SSE4.1 or a superset).
#[cfg(feature = "yuv-semi-planar")]
#[inline(always)]
pub(super) unsafe fn deinterleave_uv_u16(ptr: *const u16) -> (__m128i, __m128i) {
  unsafe {
    // Per‑chunk mask: pack even u16s (U's) into low 8 bytes, odd u16s
    // (V's) into high 8 bytes.
    let split_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    let chunk0 = _mm_loadu_si128(ptr.cast());
    let chunk1 = _mm_loadu_si128(ptr.add(8).cast());
    let s0 = _mm_shuffle_epi8(chunk0, split_mask);
    let s1 = _mm_shuffle_epi8(chunk1, split_mask);
    let u_vec = _mm_unpacklo_epi64(s0, s1);
    let v_vec = _mm_unpackhi_epi64(s0, s1);
    (u_vec, v_vec)
  }
}

// ---- helpers (inlined into the target_feature‑enabled caller) ----------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-semi-planar",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-planar",
))]
#[inline(always)]
pub(super) fn q15_shift(v: __m128i) -> __m128i {
  unsafe { _mm_srai_epi32::<15>(v) }
}

/// Computes one i16x8 chroma channel vector from the 4 × i32x4 chroma
/// inputs. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, then saturating‑packs
/// to i16x8. No lane fixup needed at 128 bits.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-semi-planar",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-planar",
))]
#[inline(always)]
pub(super) fn chroma_i16x8(
  cu: __m128i,
  cv: __m128i,
  u_d_lo: __m128i,
  v_d_lo: __m128i,
  u_d_hi: __m128i,
  v_d_hi: __m128i,
  rnd: __m128i,
) -> __m128i {
  unsafe {
    let lo = _mm_srai_epi32::<15>(_mm_add_epi32(
      _mm_add_epi32(_mm_mullo_epi32(cu, u_d_lo), _mm_mullo_epi32(cv, v_d_lo)),
      rnd,
    ));
    let hi = _mm_srai_epi32::<15>(_mm_add_epi32(
      _mm_add_epi32(_mm_mullo_epi32(cu, u_d_hi), _mm_mullo_epi32(cv, v_d_hi)),
      rnd,
    ));
    _mm_packs_epi32(lo, hi)
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x8 vector,
/// returned as i16x8.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-semi-planar",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-planar",
))]
#[inline(always)]
pub(super) fn scale_y(
  y_i16: __m128i,
  y_off_v: __m128i,
  y_scale_v: __m128i,
  rnd: __m128i,
) -> __m128i {
  unsafe {
    let shifted = _mm_sub_epi16(y_i16, y_off_v);
    let lo_i32 = _mm_cvtepi16_epi32(shifted);
    let hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(shifted));
    let lo_scaled = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(lo_i32, y_scale_v), rnd));
    let hi_scaled = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(hi_i32, y_scale_v), rnd));
    _mm_packs_epi32(lo_scaled, hi_scaled)
  }
}

// ===== 16-bit YUV → RGB helpers =========================================

/// `(Y_u16 - y_off) * y_scale + RND >> 15` for full u16 Y samples
/// (unsigned widening via `_mm_cvtepu16_epi32`). Returns i16x8.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "y2xx",
))]
#[inline(always)]
pub(super) fn scale_y_u16(
  y_u16: __m128i,
  y_off_v: __m128i,
  y_scale_v: __m128i,
  rnd_v: __m128i,
) -> __m128i {
  unsafe {
    let y_lo_i32 = _mm_sub_epi32(_mm_cvtepu16_epi32(y_u16), y_off_v);
    let y_hi_u16 = _mm_srli_si128::<8>(y_u16);
    let y_hi_i32 = _mm_sub_epi32(_mm_cvtepu16_epi32(y_hi_u16), y_off_v);
    let lo = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(y_lo_i32, y_scale_v), rnd_v));
    let hi = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(y_hi_i32, y_scale_v), rnd_v));
    _mm_packs_epi32(lo, hi)
  }
}

/// `srai64_15(x) = srli64_15(x + 2^32) - 2^17` — arithmetic right-shift
/// by 15 for i64x2. Mathematically valid for `x >= -2^32` (i.e.
/// `x + 2^32 >= 0` so the unsigned shift matches the signed one).
/// No `_mm_srai_epi64` in SSE4.1, so AVX2/AVX-512 u16 paths delegate
/// to the SSE4.1 kernel that uses this helper.
///
/// Callers: both u16 callers stay strictly inside this domain.
/// - **Chroma sum** `c_u * u_d + c_v * v_d + RND` reaches at most
///   `|c|_max * |u_d|_max ≈ 61655 * 37449 ≈ 2.31·10⁹` across all
///   supported matrices at 16-bit limited range (Bt2020Ncl b_u is
///   the tightest case). `|x| ≤ 2.31·10⁹ < 2^32`.
/// - **Y scale** `(y - y_off) * y_scale + RND` reaches at most
///   `61439 * ~38290 ≈ 2.35·10⁹` at 16-bit limited range. Still
///   `|x| < 2^32`.
///
/// The scalar comment's pessimistic `~4.3·10⁹` upper bound
/// overcounts by summing `|c_u|+|c_v|` against the same worst-case
/// chroma; in practice only one of the two is near the peak per
/// output channel.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-semi-planar",
  feature = "y2xx",
  feature = "yuv-planar",
))]
#[inline(always)]
pub(super) fn srai64_15(x: __m128i) -> __m128i {
  unsafe {
    // Bias x up by 2^32 so the unsigned shift is correct, then undo the
    // extra 2^17 (= 2^32 >> 15) introduced by the bias.
    let biased = _mm_add_epi64(x, _mm_set1_epi64x(1i64 << 32));
    let shifted = _mm_srli_epi64::<15>(biased);
    _mm_sub_epi64(shifted, _mm_set1_epi64x(1i64 << 17))
  }
}

/// Computes one i64x2 chroma channel from 2 × i64 (u_d, v_d) inputs.
/// Returns i64x2 with [`srai64_15`]-shifted results.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-semi-planar",
  feature = "y2xx",
  feature = "yuv-planar",
))]
#[inline(always)]
pub(super) fn chroma_i64x2(
  cu: __m128i,
  cv: __m128i,
  u_d: __m128i,
  v_d: __m128i,
  rnd_v: __m128i,
) -> __m128i {
  unsafe {
    srai64_15(_mm_add_epi64(
      _mm_add_epi64(_mm_mul_epi32(cu, u_d), _mm_mul_epi32(cv, v_d)),
      rnd_v,
    ))
  }
}

/// `(y_minus_off * y_scale + RND) >> 15` in i64 via `_mm_mul_epi32` (even
/// lanes). Caller must supply an i32x4 that is already `Y - y_off`.
/// Returns i64x2 for the two even-indexed lanes.
#[cfg(any(
  feature = "yuv-444-packed",
  feature = "yuv-semi-planar",
  feature = "y2xx",
  feature = "yuv-planar",
))]
#[inline(always)]
pub(super) fn scale_y16_i64(y_minus_off: __m128i, y_scale_v: __m128i, rnd_v: __m128i) -> __m128i {
  unsafe { srai64_15(_mm_add_epi64(_mm_mul_epi32(y_minus_off, y_scale_v), rnd_v)) }
}

#[cfg(all(test, feature = "std"))]
mod tests;
