//! x86_64 AVX2 backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_x86_feature_detected!("avx2")` returns true (runtime, std‑gated)
//! or `cfg!(target_feature = "avx2")` evaluates true (compile‑time,
//! no‑std). The kernel itself carries `#[target_feature(enable = "avx2")]`
//! so its intrinsics execute in an explicitly AVX2‑enabled context.
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON backend.
//!
//! # Pipeline (per 32 Y pixels / 16 chroma samples)
//!
//! 1. Load 32 Y (`_mm256_loadu_si256`) + 16 U (`_mm_loadu_si128`) +
//!    16 V (`_mm_loadu_si128`).
//! 2. Widen U, V to i16x16, subtract 128.
//! 3. Split each i16x16 into two i32x8 halves and apply `c_scale`.
//! 4. Per channel C ∈ {R, G, B}: compute `(C_u*u_d + C_v*v_d + RND) >> 15`
//!    in i32, narrow‑saturate to i16x16.
//! 5. Nearest‑neighbor chroma upsample: duplicate each of the 16 chroma
//!    lanes into its pair slot → two i16x16 vectors covering 32 Y
//!    lanes.
//! 6. Y path: widen 32 Y to two i16x16 vectors, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel.
//! 8. Saturate‑narrow to u8x32 per channel, then interleave as packed
//!    RGB via two halves of `_mm_shuffle_epi8` 3‑way interleave.
//!
//! # AVX2 lane‑crossing fixups
//!
//! Several AVX2 ops (`packs_epi32`, `packus_epi16`, `unpack*_epi16`,
//! `permute2x128_si256`) operate per 128‑bit lane, producing
//! lane‑split results. Each such op is immediately followed by the
//! correct permute (`permute4x64_epi64::<0xD8>` for pack results,
//! `permute2x128_si256` for unpack‑and‑split) to restore natural
//! element order. Every fixup is called out inline.

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

mod hsv;
mod packed_rgb;
mod packed_yuv_8bit;
mod semi_planar_8bit;
mod subsampled_high_bit_pn_4_2_0;
mod subsampled_high_bit_pn_4_4_4;
mod yuv_planar_16bit;
mod yuv_planar_8bit;
mod yuv_planar_high_bit;

pub(crate) use hsv::*;
pub(crate) use packed_rgb::*;
pub(crate) use packed_yuv_8bit::*;
pub(crate) use semi_planar_8bit::*;
pub(crate) use subsampled_high_bit_pn_4_2_0::*;
pub(crate) use subsampled_high_bit_pn_4_4_4::*;
pub(crate) use yuv_planar_8bit::*;
pub(crate) use yuv_planar_16bit::*;
pub(crate) use yuv_planar_high_bit::*;

// ---- Shared helpers (used across submodules) -------------------------

/// Clamps an `i16x16` vector to `[0, max]` via AVX2 `_mm256_min_epi16`
/// / `_mm256_max_epi16`. Used by native-depth u16 output paths
/// (10/12/14 bit) where `_mm256_packus_epi16` would incorrectly
/// clip to u8.
#[inline(always)]
pub(super) fn clamp_u16_max_x16(v: __m256i, zero_v: __m256i, max_v: __m256i) -> __m256i {
  unsafe { _mm256_min_epi16(_mm256_max_epi16(v, zero_v), max_v) }
}

/// Deinterleaves 32 `u16` elements at `ptr` (`[U0, V0, U1, V1, …,
/// U15, V15]`) into `(u_vec, v_vec)` — two AVX2 vectors each holding
/// 16 packed `u16` samples.
///
/// Uses per‑lane `_mm256_shuffle_epi8` to pack each 128‑bit lane's
/// U/V samples into the low/high 64 bits, then
/// `_mm256_permute4x64_epi64::<0xD8>` to move the two U halves
/// together (low 128) and the two V halves together (high 128) within
/// each source vector, and finally `_mm256_permute2x128_si256` to
/// combine the four U halves and the four V halves across the two
/// vectors. 2 loads + 2 shuffles + 2 per-vector permutes + 2 cross-
/// vector permutes = 8 ops.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (32 `u16`
/// elements). Caller's `target_feature` must include AVX2.
#[inline(always)]
pub(super) unsafe fn deinterleave_uv_u16_avx2(ptr: *const u16) -> (__m256i, __m256i) {
  unsafe {
    // Per‑lane byte mask: within each 128‑bit lane, pack even u16s
    // (U's) into low 8 bytes, odd u16s (V's) into high 8 bytes.
    let split_mask = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // low lane
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // high lane
    );

    let uv0 = _mm256_loadu_si256(ptr.cast());
    let uv1 = _mm256_loadu_si256(ptr.add(16).cast());

    // After per‑lane shuffle: each vector is
    // `[U_lane0_lo, V_lane0_lo, U_lane1_lo, V_lane1_lo]` in 64‑bit
    // chunks.
    let s0 = _mm256_shuffle_epi8(uv0, split_mask);
    let s1 = _mm256_shuffle_epi8(uv1, split_mask);

    // Permute 4×64 within each vector to get [U0..U7, V0..V7] and
    // [U8..U15, V8..V15]. Mask 0xD8 = (3,1,2,0) → picks 64-bit
    // chunks 0, 2, 1, 3 from the source, rearranging
    // [A, B, C, D] → [A, C, B, D].
    let s0_p = _mm256_permute4x64_epi64::<0xD8>(s0);
    let s1_p = _mm256_permute4x64_epi64::<0xD8>(s1);

    // Cross-vector permute: low 128 of s0_p + low 128 of s1_p → U's;
    // high 128 of s0_p + high 128 of s1_p → V's.
    let u_vec = _mm256_permute2x128_si256::<0x20>(s0_p, s1_p);
    let v_vec = _mm256_permute2x128_si256::<0x31>(s0_p, s1_p);
    (u_vec, v_vec)
  }
}

// ---- helpers (all `#[inline(always)]` so the `#[target_feature]`
// context from the caller flows through) --------------------------------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
pub(super) fn q15_shift(v: __m256i) -> __m256i {
  unsafe { _mm256_srai_epi32::<15>(v) }
}

/// Computes one i16x16 chroma channel vector from the 4 × i32x8 chroma
/// inputs (lo/hi splits of u_d and v_d). Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, then saturating‑packs
/// to i16x16 and **fixes the lane order** with
/// `permute4x64_epi64::<0xD8>` so the result is in natural
/// `[0..16)` element order rather than the per‑lane‑split form
/// `_mm256_packs_epi32` produces.
#[inline(always)]
pub(super) fn chroma_i16x16(
  cu: __m256i,
  cv: __m256i,
  u_d_lo: __m256i,
  v_d_lo: __m256i,
  u_d_hi: __m256i,
  v_d_hi: __m256i,
  rnd: __m256i,
) -> __m256i {
  unsafe {
    let lo = _mm256_srai_epi32::<15>(_mm256_add_epi32(
      _mm256_add_epi32(
        _mm256_mullo_epi32(cu, u_d_lo),
        _mm256_mullo_epi32(cv, v_d_lo),
      ),
      rnd,
    ));
    let hi = _mm256_srai_epi32::<15>(_mm256_add_epi32(
      _mm256_add_epi32(
        _mm256_mullo_epi32(cu, u_d_hi),
        _mm256_mullo_epi32(cv, v_d_hi),
      ),
      rnd,
    ));
    // `packs_epi32` produces lane‑split [lo0..3, hi0..3, lo4..7, hi4..7];
    // 0xD8 = 0b11_01_10_00 reorders 64‑bit lanes to [0, 2, 1, 3] giving
    // natural [lo0..7, hi0..7].
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo, hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x16 vector,
/// returned as i16x16. The Q15 multiply uses i32 widening identical to
/// scalar, then the result is saturating‑packed back to i16 (result is
/// in [0, 255] range so no saturation occurs in practice).
#[inline(always)]
pub(super) fn scale_y(
  y_i16: __m256i,
  y_off_v: __m256i,
  y_scale_v: __m256i,
  rnd: __m256i,
) -> __m256i {
  unsafe {
    let shifted = _mm256_sub_epi16(y_i16, y_off_v);
    // Widen to two i32x8 halves.
    let lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(shifted));
    let hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(shifted));
    let lo_scaled =
      _mm256_srai_epi32::<15>(_mm256_add_epi32(_mm256_mullo_epi32(lo_i32, y_scale_v), rnd));
    let hi_scaled =
      _mm256_srai_epi32::<15>(_mm256_add_epi32(_mm256_mullo_epi32(hi_i32, y_scale_v), rnd));
    // Narrow + lane fixup (same pattern as `chroma_i16x16`).
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo_scaled, hi_scaled))
  }
}

/// Duplicates each of the 16 chroma lanes in `chroma` into its adjacent
/// pair slot, splitting the result across two i16x16 vectors that
/// cover 32 Y lanes:
///
/// - Return.0 (for Y[0..16]): `[c0,c0, c1,c1, ..., c7,c7]`.
/// - Return.1 (for Y[16..32]): `[c8,c8, c9,c9, ..., c15,c15]`.
///
/// `_mm256_unpack*_epi16` are per‑128‑bit‑lane, so they produce
/// interleaved‑but‑lane‑split outputs; `_mm256_permute2x128_si256`
/// with selectors 0x20 / 0x31 selects the matching halves from each
/// unpack to restore the per‑Y‑block order above.
#[inline(always)]
pub(super) fn chroma_dup(chroma: __m256i) -> (__m256i, __m256i) {
  unsafe {
    // unpacklo per‑lane: [c0,c0,c1,c1,c2,c2,c3,c3, c8,c8,c9,c9,c10,c10,c11,c11]
    // unpackhi per‑lane: [c4,c4,c5,c5,c6,c6,c7,c7, c12,c12,c13,c13,c14,c14,c15,c15]
    let a = _mm256_unpacklo_epi16(chroma, chroma);
    let b = _mm256_unpackhi_epi16(chroma, chroma);
    // 0x20 = take 128‑bit lane 0 from a, lane 0 from b
    //      → [c0..3 dup, c4..7 dup] = pair‑expanded c0..c7.
    // 0x31 = take lane 1 from a, lane 1 from b
    //      → [c8..11 dup, c12..15 dup] = pair‑expanded c8..c15.
    let lo16 = _mm256_permute2x128_si256::<0x20>(a, b);
    let hi16 = _mm256_permute2x128_si256::<0x31>(a, b);
    (lo16, hi16)
  }
}

/// Saturating‑narrows two i16x16 vectors into one u8x32 with natural
/// element order. `_mm256_packus_epi16` is per‑lane and produces
/// lane‑split u8x32; `permute4x64_epi64::<0xD8>` fixes it.
#[inline(always)]
pub(super) fn narrow_u8x32(lo: __m256i, hi: __m256i) -> __m256i {
  unsafe { _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(lo, hi)) }
}

/// Writes 32 pixels of packed RGB (96 bytes) by interleaving three
/// u8x32 B/G/R channel vectors. Processed as two 16‑pixel halves via
/// the shared [`write_rgb_16`](super::x86_common::write_rgb_16) helper.
///
/// # Safety
///
/// `ptr` must point to at least 96 writable bytes.
#[inline(always)]
pub(super) unsafe fn write_rgb_32(r: __m256i, g: __m256i, b: __m256i, ptr: *mut u8) {
  unsafe {
    let r_lo = _mm256_castsi256_si128(r);
    let r_hi = _mm256_extracti128_si256::<1>(r);
    let g_lo = _mm256_castsi256_si128(g);
    let g_hi = _mm256_extracti128_si256::<1>(g);
    let b_lo = _mm256_castsi256_si128(b);
    let b_hi = _mm256_extracti128_si256::<1>(b);

    write_rgb_16(r_lo, g_lo, b_lo, ptr);
    write_rgb_16(r_hi, g_hi, b_hi, ptr.add(48));
  }
}

/// Writes 32 pixels of packed RGBA (128 bytes) by interleaving four
/// u8x32 R/G/B/A channel vectors. Processed as two 16‑pixel halves
/// via the shared
/// [`write_rgba_16`](super::x86_common::write_rgba_16) helper.
///
/// # Safety
///
/// `ptr` must point to at least 128 writable bytes.
#[inline(always)]
pub(super) unsafe fn write_rgba_32(r: __m256i, g: __m256i, b: __m256i, a: __m256i, ptr: *mut u8) {
  unsafe {
    let r_lo = _mm256_castsi256_si128(r);
    let r_hi = _mm256_extracti128_si256::<1>(r);
    let g_lo = _mm256_castsi256_si128(g);
    let g_hi = _mm256_extracti128_si256::<1>(g);
    let b_lo = _mm256_castsi256_si128(b);
    let b_hi = _mm256_extracti128_si256::<1>(b);
    let a_lo = _mm256_castsi256_si128(a);
    let a_hi = _mm256_extracti128_si256::<1>(a);

    write_rgba_16(r_lo, g_lo, b_lo, a_lo, ptr);
    write_rgba_16(r_hi, g_hi, b_hi, a_hi, ptr.add(64));
  }
}

// ===== 16-bit YUV → RGB ==================================================

/// `(Y_u16x16 - y_off) * y_scale + RND >> 15` for full u16 Y samples.
/// Unsigned widening via `_mm256_cvtepu16_epi32`. Returns i16x16.
#[inline(always)]
pub(super) fn scale_y_u16_avx2(
  y_u16x16: __m256i,
  y_off_v: __m256i,
  y_scale_v: __m256i,
  rnd_v: __m256i,
) -> __m256i {
  unsafe {
    let y_lo_i32 = _mm256_sub_epi32(
      _mm256_cvtepu16_epi32(_mm256_castsi256_si128(y_u16x16)),
      y_off_v,
    );
    let y_hi_i32 = _mm256_sub_epi32(
      _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(y_u16x16)),
      y_off_v,
    );
    let lo = _mm256_srai_epi32::<15>(_mm256_add_epi32(
      _mm256_mullo_epi32(y_lo_i32, y_scale_v),
      rnd_v,
    ));
    let hi = _mm256_srai_epi32::<15>(_mm256_add_epi32(
      _mm256_mullo_epi32(y_hi_i32, y_scale_v),
      rnd_v,
    ));
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo, hi))
  }
}

/// Arithmetic right shift by 15 on an `i64x4` vector via the
/// "bias trick" — AVX2 lacks `_mm256_srai_epi64` until AVX-512VL, so
/// we add `2^32` before an unsigned shift and subtract the resulting
/// `2^17` offset. Valid for `|x| < 2^32`, which easily holds for our
/// chroma and Y products at 16-bit limited range.
#[inline(always)]
pub(super) fn srai64_15_x4(x: __m256i) -> __m256i {
  unsafe {
    let biased = _mm256_add_epi64(x, _mm256_set1_epi64x(1i64 << 32));
    let shifted = _mm256_srli_epi64::<15>(biased);
    _mm256_sub_epi64(shifted, _mm256_set1_epi64x(1i64 << 17))
  }
}

/// Computes one i64x4 chroma channel from 4 × i64 (u_d, v_d) inputs
/// using `_mm256_mul_epi32` (even-indexed i32 lanes → 4 i64 products
/// per call). Returns i64x4 with [`srai64_15_x4`]-shifted results in
/// the low 32 bits of each i64 lane.
#[inline(always)]
pub(super) fn chroma_i64x4_avx2(
  cu: __m256i,
  cv: __m256i,
  u_d: __m256i,
  v_d: __m256i,
  rnd_v: __m256i,
) -> __m256i {
  unsafe {
    srai64_15_x4(_mm256_add_epi64(
      _mm256_add_epi64(_mm256_mul_epi32(cu, u_d), _mm256_mul_epi32(cv, v_d)),
      rnd_v,
    ))
  }
}

/// Combines two i64x4 results (from even + odd i32 lanes of the
/// pre-multiply source) into an interleaved i32x8 = `[even.low32[0],
/// odd.low32[0], even.low32[1], odd.low32[1], ..., even.low32[3],
/// odd.low32[3]]`. Same shape as the SSE4.1
/// `_mm_unpacklo_epi64(_mm_unpacklo_epi32(even, odd),
/// _mm_unpackhi_epi32(even, odd))` pattern, lifted to 256 bits.
#[inline(always)]
pub(super) fn reassemble_i64x4_to_i32x8(even: __m256i, odd: __m256i) -> __m256i {
  unsafe {
    _mm256_unpacklo_epi64(
      _mm256_unpacklo_epi32(even, odd),
      _mm256_unpackhi_epi32(even, odd),
    )
  }
}

/// `(y_minus_off * y_scale + RND) >> 15` computed in i64 for all 8
/// lanes of an i32x8 Y stream, returning an i32x8 result. Needs i64
/// because limited-range 16→u16 `(Y - y_off) * y_scale` can reach
/// ~2.35·10⁹ (> i32::MAX).
#[inline(always)]
pub(super) fn scale_y_i32x8_i64(
  y_minus_off: __m256i,
  y_scale_v: __m256i,
  rnd_v: __m256i,
) -> __m256i {
  unsafe {
    let even = srai64_15_x4(_mm256_add_epi64(
      _mm256_mul_epi32(y_minus_off, y_scale_v),
      rnd_v,
    ));
    let odd = srai64_15_x4(_mm256_add_epi64(
      _mm256_mul_epi32(_mm256_shuffle_epi32::<0xF5>(y_minus_off), y_scale_v),
      rnd_v,
    ));
    reassemble_i64x4_to_i32x8(even, odd)
  }
}

/// Duplicates an i32x8 chroma vector into two i32x8 `"2-Y-per-chroma"`
/// vectors covering 16 pixels:
/// - Return.0 (for Y[0..8]): `[c0,c0, c1,c1, c2,c2, c3,c3]`
/// - Return.1 (for Y[8..16]): `[c4,c4, c5,c5, c6,c6, c7,c7]`
///
/// Mirrors the i16 `chroma_dup` helper's lane-cross restoration
/// pattern (`_mm256_permute2x128_si256::<0x20>` / `<0x31>`).
#[inline(always)]
pub(super) fn chroma_dup_i32(chroma: __m256i) -> (__m256i, __m256i) {
  unsafe {
    let a = _mm256_unpacklo_epi32(chroma, chroma);
    let b = _mm256_unpackhi_epi32(chroma, chroma);
    let lo = _mm256_permute2x128_si256::<0x20>(a, b);
    let hi = _mm256_permute2x128_si256::<0x31>(a, b);
    (lo, hi)
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
