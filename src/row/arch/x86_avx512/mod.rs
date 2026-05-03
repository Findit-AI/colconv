//! x86_64 AVX‑512 backend (F + BW) for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_x86_feature_detected!("avx512bw")` returns true (runtime,
//! std‑gated) or `cfg!(target_feature = "avx512bw")` evaluates true
//! (compile‑time, no‑std). The kernel carries
//! `#[target_feature(enable = "avx512f,avx512bw")]` so its intrinsics
//! execute in an explicitly feature‑enabled context.
//!
//! Requires AVX‑512F (foundation) and AVX‑512BW (byte/word integer
//! ops). All real AVX‑512 CPUs have both — Intel Skylake‑X / Cascade
//! Lake / Ice Lake / Sapphire Rapids Xeons, AMD Zen 4+ (Genoa,
//! Ryzen 7000+).
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON / SSE4.1 / AVX2 backends.
//!
//! # Pipeline (per 64 Y pixels / 32 chroma samples)
//!
//! 1. Load 64 Y (`_mm512_loadu_si512`) + 32 U + 32 V (`_mm256_loadu_si256`).
//! 2. Widen U, V to i16x32 (`_mm512_cvtepu8_epi16`), subtract 128.
//! 3. Split each i16x32 into two i32x16 halves and apply `c_scale`.
//! 4. Per channel C ∈ {R, G, B}: `(C_u*u_d + C_v*v_d + RND) >> 15` in
//!    i32, narrow‑saturate to i16x32.
//! 5. Nearest‑neighbor chroma upsample: duplicate each of the 32 chroma
//!    lanes into its pair slot → two i16x32 vectors covering 64 Y lanes.
//! 6. Y path: widen 64 Y to two i16x32 vectors, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel.
//! 8. Saturate‑narrow to u8x64 per channel, then interleave as packed
//!    RGB via four calls to the shared [`super::x86_common::write_rgb_16`]
//!    (192 output bytes = 4 × 48).
//!
//! # AVX‑512 lane‑crossing fixups
//!
//! AVX‑512 registers act as four 128‑bit lanes for most of the ops we
//! use. `_mm512_packs_epi32`, `_mm512_packus_epi16`, and
//! `_mm512_unpack{lo,hi}_epi16` all operate per 128‑bit lane,
//! producing lane‑split results.
//!
//! - **Pack fixup** (shared by `packs_epi32` → i16x32 and
//!   `packus_epi16` → u8x64): after either pack, 64‑bit lane order is
//!   `[lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]`. Permute via
//!   `_mm512_permutexvar_epi64` with index `[0, 2, 4, 6, 1, 3, 5, 7]`
//!   restores natural `[lo0..3 contiguous, hi0..3 contiguous]`.
//! - **Chroma‑dup fixup**: `unpacklo`/`unpackhi` each produce per‑lane
//!   duplicated pairs but the halves for a given Y block are split
//!   across lanes. `_mm512_permutex2var_epi64` with indices
//!   `[0,1,8,9,2,3,10,11]` and `[4,5,12,13,6,7,14,15]` rebuilds the
//!   two 32‑Y‑block‑aligned vectors from unpacklo + unpackhi.

use core::arch::x86_64::*;

#[allow(unused_imports)]
pub(super) use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{
      abgr_to_rgb_16_pixels, abgr_to_rgba_4_pixels, argb_to_rgb_16_pixels, argb_to_rgba_4_pixels,
      bgra_to_rgb_16_pixels, bgrx_to_rgba_4_pixels, drop_alpha_16_pixels, rgb_to_hsv_16_pixels,
      rgbx_to_rgba_4_pixels, swap_rb_16_pixels, swap_rb_alpha_4_pixels, write_rgb_16,
      write_rgb_u16_8, write_rgba_16, write_rgba_u16_8, x2bgr10_to_rgb_16_pixels,
      x2bgr10_to_rgb_u16_8_pixels, x2bgr10_to_rgba_16_pixels, x2rgb10_to_rgb_16_pixels,
      x2rgb10_to_rgb_u16_8_pixels, x2rgb10_to_rgba_16_pixels, xbgr_to_rgba_4_pixels,
      xrgb_to_rgba_4_pixels,
    },
    scalar,
  },
};

mod ayuv64;
mod hsv;
mod packed_rgb;
mod packed_yuv_8bit;
mod semi_planar_8bit;
mod subsampled_high_bit_pn_4_2_0;
mod subsampled_high_bit_pn_4_4_4;
mod v210;
mod v30x;
mod v410;
mod vuya;
mod xv36;
mod y216;
mod y2xx;
mod yuv_planar_16bit;
mod yuv_planar_8bit;
mod yuv_planar_high_bit;

pub(crate) use ayuv64::*;
pub(crate) use hsv::*;
pub(crate) use packed_rgb::*;
pub(crate) use packed_yuv_8bit::*;
pub(crate) use semi_planar_8bit::*;
pub(crate) use subsampled_high_bit_pn_4_2_0::*;
pub(crate) use subsampled_high_bit_pn_4_4_4::*;
pub(crate) use v30x::*;
pub(crate) use v210::*;
pub(crate) use v410::*;
pub(crate) use vuya::*;
pub(crate) use xv36::*;
pub(crate) use y2xx::*;
pub(crate) use y216::*;
pub(crate) use yuv_planar_8bit::*;
pub(crate) use yuv_planar_16bit::*;
pub(crate) use yuv_planar_high_bit::*;

// ---- Shared helpers (used across submodules) -------------------------

/// Clamps an `i16x32` vector to `[0, max]` via AVX‑512
/// `_mm512_min_epi16` / `_mm512_max_epi16`. Used by native-depth
/// u16 output paths (10/12/14 bit).
#[inline(always)]
pub(super) fn clamp_u16_max_x32(v: __m512i, zero_v: __m512i, max_v: __m512i) -> __m512i {
  unsafe { _mm512_min_epi16(_mm512_max_epi16(v, zero_v), max_v) }
}

/// Writes one 8‑pixel u16 RGB chunk using a 128‑bit quarter of each
/// `i16x32` channel vector. `idx` ∈ `{0,1,2,3}` selects which of the
/// four 128‑bit lanes to extract via `_mm512_extracti32x4_epi32`.
///
/// # Safety
///
/// Same as [`write_rgb_u16_8`] — `ptr` must point to at least 48
/// writable bytes (24 `u16`). Caller's `target_feature` must include
/// AVX‑512F + AVX‑512BW (so `_mm512_extracti32x4_epi32` is available)
/// and SSSE3 (for the underlying `_mm_shuffle_epi8` inside
/// `write_rgb_u16_8`).
#[inline(always)]
pub(super) unsafe fn write_quarter(r: __m512i, g: __m512i, b: __m512i, idx: u8, ptr: *mut u16) {
  // SAFETY: caller holds the AVX‑512F + SSSE3 target‑feature context.
  // Constant generic arg `IDX` picks one of four 128‑bit lanes; `idx`
  // is bounded to 0..=3 by call sites.
  unsafe {
    let (rq, gq, bq) = match idx {
      0 => (
        _mm512_extracti32x4_epi32::<0>(r),
        _mm512_extracti32x4_epi32::<0>(g),
        _mm512_extracti32x4_epi32::<0>(b),
      ),
      1 => (
        _mm512_extracti32x4_epi32::<1>(r),
        _mm512_extracti32x4_epi32::<1>(g),
        _mm512_extracti32x4_epi32::<1>(b),
      ),
      2 => (
        _mm512_extracti32x4_epi32::<2>(r),
        _mm512_extracti32x4_epi32::<2>(g),
        _mm512_extracti32x4_epi32::<2>(b),
      ),
      _ => (
        _mm512_extracti32x4_epi32::<3>(r),
        _mm512_extracti32x4_epi32::<3>(g),
        _mm512_extracti32x4_epi32::<3>(b),
      ),
    };
    write_rgb_u16_8(rq, gq, bq, ptr);
  }
}

/// RGBA sibling of [`write_quarter`]. Extracts one 128‑bit quarter of
/// each `i16x32` channel vector and hands it (plus a splatted alpha)
/// to [`write_rgba_u16_8`].
///
/// # Safety
///
/// Same as [`write_rgba_u16_8`] — `ptr` must point to at least 64
/// writable bytes (32 `u16`). Caller's `target_feature` must include
/// AVX‑512F + AVX‑512BW (so `_mm512_extracti32x4_epi32` is available)
/// and SSE2 (for the underlying unpack/store inside
/// `write_rgba_u16_8`).
#[inline(always)]
pub(super) unsafe fn write_quarter_rgba(
  r: __m512i,
  g: __m512i,
  b: __m512i,
  a: __m128i,
  idx: u8,
  ptr: *mut u16,
) {
  unsafe {
    let (rq, gq, bq) = match idx {
      0 => (
        _mm512_extracti32x4_epi32::<0>(r),
        _mm512_extracti32x4_epi32::<0>(g),
        _mm512_extracti32x4_epi32::<0>(b),
      ),
      1 => (
        _mm512_extracti32x4_epi32::<1>(r),
        _mm512_extracti32x4_epi32::<1>(g),
        _mm512_extracti32x4_epi32::<1>(b),
      ),
      2 => (
        _mm512_extracti32x4_epi32::<2>(r),
        _mm512_extracti32x4_epi32::<2>(g),
        _mm512_extracti32x4_epi32::<2>(b),
      ),
      _ => (
        _mm512_extracti32x4_epi32::<3>(r),
        _mm512_extracti32x4_epi32::<3>(g),
        _mm512_extracti32x4_epi32::<3>(b),
      ),
    };
    write_rgba_u16_8(rq, gq, bq, a, ptr);
  }
}

/// Deinterleaves 64 `u16` elements at `ptr` into `(u_vec, v_vec)` —
/// two AVX‑512 vectors each holding 32 packed `u16` samples.
///
/// Per‑128‑bit‑lane `_mm512_shuffle_epi8` packs even u16s (U's) into
/// each lane's low 64 bits, odd u16s (V's) into the high 64. Then
/// `_mm512_permutexvar_epi64` with the existing `pack_fixup` index
/// `[0, 2, 4, 6, 1, 3, 5, 7]` rearranges the 64‑bit chunks so each
/// vector becomes `[U0..U15 | V0..V15]`. Finally
/// `_mm512_permutex2var_epi64` combines the two vectors into the
/// full 32‑sample U and V vectors.
///
/// # Safety
///
/// `ptr` must point to at least 128 readable bytes (64 `u16`
/// elements). Caller's `target_feature` must include AVX‑512F +
/// AVX‑512BW.
#[inline(always)]
pub(super) unsafe fn deinterleave_uv_u16_avx512(ptr: *const u16) -> (__m512i, __m512i) {
  unsafe {
    // Per‑128‑lane mask (same byte pattern replicated across the 4
    // lanes of a `__m512i`).
    let split_mask = _mm512_broadcast_i32x4(_mm_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
    ));
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    // Cross-vector 2x8 permute indices:
    //   u_vec = low 256 of each vec → chunks [0..3 of a, 0..3 of b]
    //   v_vec = high 256 of each vec → chunks [4..7 of a, 4..7 of b]
    let u_perm = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    let v_perm = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

    let uv0 = _mm512_loadu_si512(ptr.cast());
    let uv1 = _mm512_loadu_si512(ptr.add(32).cast());

    let s0 = _mm512_shuffle_epi8(uv0, split_mask);
    let s1 = _mm512_shuffle_epi8(uv1, split_mask);

    // After per-lane shuffle + per-vector 64-bit permute, each vector
    // is `[U0..U15 | V0..V15]` (low 256 = U's, high 256 = V's).
    let s0_p = _mm512_permutexvar_epi64(pack_fixup, s0);
    let s1_p = _mm512_permutexvar_epi64(pack_fixup, s1);

    let u_vec = _mm512_permutex2var_epi64(s0_p, u_perm, s1_p);
    let v_vec = _mm512_permutex2var_epi64(s0_p, v_perm, s1_p);
    (u_vec, v_vec)
  }
}

// ---- helpers (inlined into the target_feature‑enabled caller) ----------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
pub(super) fn q15_shift(v: __m512i) -> __m512i {
  unsafe { _mm512_srai_epi32::<15>(v) }
}

/// Computes one i16x32 chroma channel vector from the four i32x16
/// chroma inputs (lo/hi halves of `u_d` and `v_d`). Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, saturating‑packs to
/// i16x32, then applies `pack_fixup` to restore natural element order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn chroma_i16x32(
  cu: __m512i,
  cv: __m512i,
  u_d_lo: __m512i,
  v_d_lo: __m512i,
  u_d_hi: __m512i,
  v_d_hi: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_add_epi32(
        _mm512_mullo_epi32(cu, u_d_lo),
        _mm512_mullo_epi32(cv, v_d_lo),
      ),
      rnd,
    ));
    let hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_add_epi32(
        _mm512_mullo_epi32(cu, u_d_hi),
        _mm512_mullo_epi32(cv, v_d_hi),
      ),
      rnd,
    ));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo, hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x32 vector,
/// returned as i16x32 (with pack fixup applied).
#[inline(always)]
pub(super) fn scale_y(
  y_i16: __m512i,
  y_off_v: __m512i,
  y_scale_v: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let shifted = _mm512_sub_epi16(y_i16, y_off_v);
    let lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(shifted));
    let hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(shifted));
    let lo_scaled =
      _mm512_srai_epi32::<15>(_mm512_add_epi32(_mm512_mullo_epi32(lo_i32, y_scale_v), rnd));
    let hi_scaled =
      _mm512_srai_epi32::<15>(_mm512_add_epi32(_mm512_mullo_epi32(hi_i32, y_scale_v), rnd));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo_scaled, hi_scaled))
  }
}

/// Duplicates each of 32 chroma lanes into its adjacent pair slot,
/// splitting across two i16x32 vectors covering 64 Y lanes.
#[inline(always)]
pub(super) fn chroma_dup(
  chroma: __m512i,
  dup_lo_idx: __m512i,
  dup_hi_idx: __m512i,
) -> (__m512i, __m512i) {
  unsafe {
    let a = _mm512_unpacklo_epi16(chroma, chroma);
    let b = _mm512_unpackhi_epi16(chroma, chroma);
    let lo32 = _mm512_permutex2var_epi64(a, dup_lo_idx, b);
    let hi32 = _mm512_permutex2var_epi64(a, dup_hi_idx, b);
    (lo32, hi32)
  }
}

/// Saturating‑narrows two i16x32 vectors into one u8x64 with natural
/// element order.
#[inline(always)]
pub(super) fn narrow_u8x64(lo: __m512i, hi: __m512i, pack_fixup: __m512i) -> __m512i {
  unsafe { _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi16(lo, hi)) }
}

/// Writes 64 pixels of packed RGB (192 bytes) by splitting the u8x64
/// channel vectors into four 128‑bit halves and calling the shared
/// [`write_rgb_16`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 192 writable bytes.
#[inline(always)]
pub(super) unsafe fn write_rgb_64(r: __m512i, g: __m512i, b: __m512i, ptr: *mut u8) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);

    write_rgb_16(r0, g0, b0, ptr);
    write_rgb_16(r1, g1, b1, ptr.add(48));
    write_rgb_16(r2, g2, b2, ptr.add(96));
    write_rgb_16(r3, g3, b3, ptr.add(144));
  }
}

/// Writes 64 pixels of packed RGBA (256 bytes) by splitting the
/// u8x64 channel vectors into four 128‑bit quarters and calling the
/// shared [`write_rgba_16`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 256 writable bytes.
#[inline(always)]
pub(super) unsafe fn write_rgba_64(r: __m512i, g: __m512i, b: __m512i, a: __m512i, ptr: *mut u8) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);
    let a0: __m128i = _mm512_castsi512_si128(a);
    let a1: __m128i = _mm512_extracti32x4_epi32::<1>(a);
    let a2: __m128i = _mm512_extracti32x4_epi32::<2>(a);
    let a3: __m128i = _mm512_extracti32x4_epi32::<3>(a);

    write_rgba_16(r0, g0, b0, a0, ptr);
    write_rgba_16(r1, g1, b1, a1, ptr.add(64));
    write_rgba_16(r2, g2, b2, a2, ptr.add(128));
    write_rgba_16(r3, g3, b3, a3, ptr.add(192));
  }
}

// ===== 16-bit u16-output helpers ========================================

/// `(c_u * u_d + c_v * v_d + RND) >> 15` in i64 via two
/// `_mm512_mul_epi32` (even i32 lanes → i64x8 products) plus native
/// `_mm512_srai_epi64`. Result is i64x8 with each lane's low 32 bits
/// holding the i32-range output.
#[inline(always)]
pub(super) fn chroma_i64x8_avx512(
  cu: __m512i,
  cv: __m512i,
  u_d_even: __m512i,
  v_d_even: __m512i,
  rnd_i64: __m512i,
) -> __m512i {
  unsafe {
    _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_add_epi64(
        _mm512_mul_epi32(cu, u_d_even),
        _mm512_mul_epi32(cv, v_d_even),
      ),
      rnd_i64,
    ))
  }
}

/// Combines two i64x8 results (one from even-indexed, one from
/// odd-indexed i32 lanes of the pre-multiply source) into an
/// interleaved i32x16 `[even_0, odd_0, even_1, odd_1, ..., even_7,
/// odd_7]` — i.e. the natural sequential order of 16 scalar results.
///
/// Each i64 lane's low 32 bits contain the result (high 32 are sign);
/// `_mm512_cvtepi64_epi32` truncates to i32x8 per vector, then
/// `_mm512_permutex2var_epi32` interleaves them.
#[inline(always)]
pub(super) fn reassemble_i32x16(
  even_i64: __m512i,
  odd_i64: __m512i,
  interleave_idx: __m512i,
) -> __m512i {
  unsafe {
    let even_i32 = _mm512_cvtepi64_epi32(even_i64); // __m256i i32x8
    let odd_i32 = _mm512_cvtepi64_epi32(odd_i64);
    _mm512_permutex2var_epi32(
      _mm512_castsi256_si512(even_i32),
      interleave_idx,
      _mm512_castsi256_si512(odd_i32),
    )
  }
}

/// `(y_minus_off * y_scale + RND) >> 15` computed in i64 for all 16
/// lanes of an i32x16 Y stream, returning an i32x16 result. Needs
/// i64 because limited-range 16→u16 `(Y - y_off) * y_scale` can
/// reach ~2.35·10⁹ (> i32::MAX). Splits the input into even and
/// odd-indexed i32 lanes, multiplies each set via
/// `_mm512_mul_epi32`, shifts in i64, and reassembles to i32x16.
#[inline(always)]
pub(super) fn scale_y_i32x16_i64(
  y_minus_off: __m512i,
  y_scale_v: __m512i,
  rnd_i64: __m512i,
  interleave_idx: __m512i,
) -> __m512i {
  unsafe {
    let even = _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_mul_epi32(y_scale_v, y_minus_off),
      rnd_i64,
    ));
    let odd = _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_mul_epi32(y_scale_v, _mm512_shuffle_epi32::<0xF5>(y_minus_off)),
      rnd_i64,
    ));
    reassemble_i32x16(even, odd, interleave_idx)
  }
}

/// Writes 32 pixels of packed RGB-u16 (192 u16 = 384 bytes) by
/// splitting each u16x32 channel vector into four 128-bit halves and
/// calling the shared [`write_rgb_u16_8`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 384 writable bytes.
#[inline(always)]
pub(super) unsafe fn write_rgb_u16_32(r: __m512i, g: __m512i, b: __m512i, ptr: *mut u16) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);

    // Each `write_rgb_u16_8` writes 8 pixels × 3 × u16 = 48 bytes =
    // 24 u16 elements. Four calls → 96 u16 = 32 pixels.
    write_rgb_u16_8(r0, g0, b0, ptr);
    write_rgb_u16_8(r1, g1, b1, ptr.add(24));
    write_rgb_u16_8(r2, g2, b2, ptr.add(48));
    write_rgb_u16_8(r3, g3, b3, ptr.add(72));
  }
}

/// Writes 32 pixels of packed RGBA-u16 (128 u16 = 256 bytes) by
/// splitting each u16x32 channel vector into four 128-bit halves and
/// calling the shared [`write_rgba_u16_8`] helper four times. Alpha
/// is supplied as a single i16x8 vector splatted into all 32 alpha
/// lanes.
///
/// # Safety
///
/// `ptr` must point to at least 256 writable bytes.
#[inline(always)]
pub(super) unsafe fn write_rgba_u16_32(
  r: __m512i,
  g: __m512i,
  b: __m512i,
  a: __m128i,
  ptr: *mut u16,
) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);

    // Each `write_rgba_u16_8` writes 8 pixels × 4 × u16 = 64 bytes =
    // 32 u16 elements. Four calls → 128 u16 = 32 pixels.
    write_rgba_u16_8(r0, g0, b0, a, ptr);
    write_rgba_u16_8(r1, g1, b1, a, ptr.add(32));
    write_rgba_u16_8(r2, g2, b2, a, ptr.add(64));
    write_rgba_u16_8(r3, g3, b3, a, ptr.add(96));
  }
}

// ===== 16-bit YUV → RGB ==================================================

/// `(Y_u16x32 - y_off) * y_scale + RND >> 15` for full u16 Y samples.
/// Unsigned widening via `_mm512_cvtepu16_epi32`. Returns i16x32.
#[inline(always)]
pub(super) fn scale_y_u16_avx512(
  y_u16x32: __m512i,
  y_off_v: __m512i,
  y_scale_v: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let y_lo_i32 = _mm512_sub_epi32(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(y_u16x32)),
      y_off_v,
    );
    let y_hi_i32 = _mm512_sub_epi32(
      _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(y_u16x32)),
      y_off_v,
    );
    let lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_mullo_epi32(y_lo_i32, y_scale_v),
      rnd,
    ));
    let hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_mullo_epi32(y_hi_i32, y_scale_v),
      rnd,
    ));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo, hi))
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
