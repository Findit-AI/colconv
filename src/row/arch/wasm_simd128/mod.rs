//! WebAssembly simd128 backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher when
//! `cfg!(target_feature = "simd128")` evaluates true at compile time.
//! WASM does **not** support runtime CPU feature detection — a WASM
//! module either contains SIMD opcodes (which require runtime support
//! at instantiation) or it doesn't. So the gate is always
//! compile‑time, regardless of `feature = "std"`.
//!
//! The kernel carries `#[target_feature(enable = "simd128")]` so its
//! intrinsics are accessible to the function body even when simd128 is
//! not enabled for the whole crate.
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON / SSE4.1 / AVX2 / AVX‑512 backends.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`v128_load`) + 8 U + 8 V (`u16x8_load_extend_u8x8`,
//!    which loads 8 u8 and zero‑extends to 8 u16 in one op).
//! 2. Subtract 128 from U, V (as i16x8) to get `u_i16`, `v_i16`.
//! 3. Split each i16x8 into two i32x4 halves via
//!    `i32x4_extend_{low,high}_i16x8` and apply `c_scale`.
//! 4. Per channel: `(C_u*u_d + C_v*v_d + RND) >> 15` in i32,
//!    saturating‑narrow to i16x8 via `i16x8_narrow_i32x4`.
//! 5. Nearest‑neighbor chroma upsample with two `i8x16_shuffle`
//!    invocations (compile‑time byte indices duplicate each 16‑bit
//!    chroma lane into its pair slot).
//! 6. Y path: widen low / high 8 Y to i16x8, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel (`i16x8_add_sat`).
//! 8. Saturate‑narrow to u8x16 per channel (`u8x16_narrow_i16x8`),
//!    interleave as packed RGB via three `u8x16_swizzle` calls.

use core::arch::wasm32::*;

#[allow(unused_imports)]
pub(super) use crate::{ColorMatrix, row::scalar};

mod hsv;
mod packed_rgb;
mod packed_yuv_8bit;
mod semi_planar_8bit;
mod subsampled_high_bit_pn_4_2_0;
mod subsampled_high_bit_pn_4_4_4;
mod v210;
mod y2xx;
mod yuv_planar_16bit;
mod yuv_planar_8bit;
mod yuv_planar_high_bit;

pub(crate) use hsv::*;
pub(crate) use packed_rgb::*;
pub(crate) use packed_yuv_8bit::*;
pub(crate) use semi_planar_8bit::*;
pub(crate) use subsampled_high_bit_pn_4_2_0::*;
pub(crate) use subsampled_high_bit_pn_4_4_4::*;
pub(crate) use v210::*;
pub(crate) use y2xx::*;
pub(crate) use yuv_planar_8bit::*;
pub(crate) use yuv_planar_16bit::*;
pub(crate) use yuv_planar_high_bit::*;

// ---- Shared helpers (used across submodules) -------------------------

/// Clamps an i16x8 vector to `[0, max]`. Used by native-depth u16
/// output paths (10/12/14 bit).
#[inline(always)]
pub(super) fn clamp_u16_max_wasm(v: v128, zero_v: v128, max_v: v128) -> v128 {
  i16x8_min(i16x8_max(v, zero_v), max_v)
}

/// Writes 8 pixels of packed `u16` RGB (24 `u16` = 48 bytes) using
/// the SSSE3‑style 3‑way interleave pattern adapted to 16‑bit lanes.
/// Mirrors [`crate::row::arch::x86_common::write_rgb_u16_8`] — each
/// output u16 is two adjacent bytes sourced from one of the three
/// channel vectors via `u8x16_swizzle` with a compile‑time byte
/// mask (0xFF / negative zeros the lane, matching `_mm_shuffle_epi8`
/// semantics).
///
/// # Safety
///
/// `ptr` must point to at least 48 writable bytes (24 `u16`). Caller
/// must have simd128 enabled at compile time.
#[inline(always)]
pub(super) unsafe fn write_rgb_u16_8(r: v128, g: v128, b: v128, ptr: *mut u16) {
  unsafe {
    // Block 0 = [R0 G0 B0 R1 G1 B1 R2 G2]. Masks identical in shape
    // to x86_common::write_rgb_u16_8 — each output u16 pulls two
    // adjacent bytes from one channel.
    let r0 = i8x16(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
    let g0 = i8x16(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
    let b0 = i8x16(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(r, r0), u8x16_swizzle(g, g0)),
      u8x16_swizzle(b, b0),
    );

    // Block 1 = [B2 R3 G3 B3 R4 G4 B4 R5].
    let r1 = i8x16(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
    let g1 = i8x16(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
    let b1 = i8x16(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(r, r1), u8x16_swizzle(g, g1)),
      u8x16_swizzle(b, b1),
    );

    // Block 2 = [G5 B5 R6 G6 B6 R7 G7 B7].
    let r2 = i8x16(
      -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1,
    );
    let g2 = i8x16(
      10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1,
    );
    let b2 = i8x16(
      -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15,
    );
    let out2 = v128_or(
      v128_or(u8x16_swizzle(r, r2), u8x16_swizzle(g, g2)),
      u8x16_swizzle(b, b2),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(8).cast(), out1);
    v128_store(ptr.add(16).cast(), out2);
  }
}

/// Interleaves 8 R/G/B/A `u16` samples into packed RGBA quads (32
/// `u16` = 64 bytes). Two `i16x8_shuffle` stages: first interleave
/// R+G and B+A into pairs, then combine pair-vectors into RGBA quads.
///
/// # Safety
///
/// `ptr` must point to at least 64 writable bytes. Caller must have
/// `simd128` enabled at compile time.
#[inline(always)]
pub(super) unsafe fn write_rgba_u16_8(r: v128, g: v128, b: v128, a: v128, ptr: *mut u16) {
  unsafe {
    // Stage 1: interleave R+G and B+A pairwise.
    // rg_lo = [R0, G0, R1, G1, R2, G2, R3, G3]
    // rg_hi = [R4, G4, R5, G5, R6, G6, R7, G7]
    // ba_lo = [B0, A0, B1, A1, B2, A2, B3, A3]
    // ba_hi = [B4, A4, B5, A5, B6, A6, B7, A7]
    let rg_lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r, g);
    let rg_hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r, g);
    let ba_lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(b, a);
    let ba_hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(b, a);

    // Stage 2: combine RG pairs with BA pairs to produce RGBA quads.
    // q0 = [R0, G0, B0, A0, R1, G1, B1, A1]
    // q1 = [R2, G2, B2, A2, R3, G3, B3, A3]
    // q2 = [R4, G4, B4, A4, R5, G5, B5, A5]
    // q3 = [R6, G6, B6, A6, R7, G7, B7, A7]
    let q0 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(rg_lo, ba_lo);
    let q1 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(rg_lo, ba_lo);
    let q2 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(rg_hi, ba_hi);
    let q3 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(rg_hi, ba_hi);

    v128_store(ptr.cast(), q0);
    v128_store(ptr.add(8).cast(), q1);
    v128_store(ptr.add(16).cast(), q2);
    v128_store(ptr.add(24).cast(), q3);
  }
}

/// Deinterleaves 16 `u16` elements at `ptr` into `(u_vec, v_vec)` —
/// two 128‑bit vectors each holding 8 `u16` samples. Wasm's
/// `u8x16_swizzle` is semantically equivalent to SSSE3
/// `_mm_shuffle_epi8` (indices ≥ 16 zero the lane), so the same
/// split‑mask pattern applies. `i8x16_shuffle` is used for the
/// cross‑vector 64‑bit recombine.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes (16 `u16`
/// elements). Caller must have simd128 enabled at compile time.
#[inline(always)]
pub(super) unsafe fn deinterleave_uv_u16_wasm(ptr: *const u16) -> (v128, v128) {
  unsafe {
    // Pack evens (U's) into low 8 bytes, odds (V's) into high 8 bytes.
    let split_mask = i8x16(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);

    let chunk0 = v128_load(ptr.cast());
    let chunk1 = v128_load(ptr.add(8).cast());

    let s0 = u8x16_swizzle(chunk0, split_mask);
    let s1 = u8x16_swizzle(chunk1, split_mask);

    // u_vec = low 8 bytes of s0 + low 8 bytes of s1.
    // v_vec = high 8 bytes of s0 + high 8 bytes of s1.
    let u_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(s0, s1);
    let v_vec =
      i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(s0, s1);
    (u_vec, v_vec)
  }
}

// ---- helpers -----------------------------------------------------------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
pub(super) fn q15_shift(v: v128) -> v128 {
  i32x4_shr(v, 15)
}

/// Computes one i16x8 chroma channel vector from the 4 × i32x4 chroma
/// inputs. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, then
/// saturating‑packs to i16x8. No lane fixup needed at 128 bits.
#[inline(always)]
pub(super) fn chroma_i16x8(
  cu: v128,
  cv: v128,
  u_d_lo: v128,
  v_d_lo: v128,
  u_d_hi: v128,
  v_d_hi: v128,
  rnd: v128,
) -> v128 {
  let lo = i32x4_shr(
    i32x4_add(i32x4_add(i32x4_mul(cu, u_d_lo), i32x4_mul(cv, v_d_lo)), rnd),
    15,
  );
  let hi = i32x4_shr(
    i32x4_add(i32x4_add(i32x4_mul(cu, u_d_hi), i32x4_mul(cv, v_d_hi)), rnd),
    15,
  );
  i16x8_narrow_i32x4(lo, hi)
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x8 vector,
/// returned as i16x8.
#[inline(always)]
pub(super) fn scale_y(y_i16: v128, y_off_v: v128, y_scale_v: v128, rnd: v128) -> v128 {
  let shifted = i16x8_sub(y_i16, y_off_v);
  let lo_i32 = i32x4_extend_low_i16x8(shifted);
  let hi_i32 = i32x4_extend_high_i16x8(shifted);
  let lo_scaled = i32x4_shr(i32x4_add(i32x4_mul(lo_i32, y_scale_v), rnd), 15);
  let hi_scaled = i32x4_shr(i32x4_add(i32x4_mul(hi_i32, y_scale_v), rnd), 15);
  i16x8_narrow_i32x4(lo_scaled, hi_scaled)
}

/// Widens the low 8 bytes of a u8x16 to i16x8 (zero‑extended since
/// Y ∈ [0, 255] fits in non‑negative i16).
#[inline(always)]
pub(super) fn u8_low_to_i16x8(v: v128) -> v128 {
  // i8x16_shuffle picks bytes pairwise: for each output i16 lane i,
  // take byte i of the source as the low byte and pad with a zero
  // byte from the all‑zero operand.
  i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(v, i16x8_splat(0))
}

/// Widens the high 8 bytes of a u8x16 to i16x8 (zero‑extended).
#[inline(always)]
pub(super) fn u8_high_to_i16x8(v: v128) -> v128 {
  i8x16_shuffle::<8, 16, 9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15, 23>(v, i16x8_splat(0))
}

/// Duplicates the low 4 × i16 lanes of `chroma` into 8 lanes
/// `[c0,c0, c1,c1, c2,c2, c3,c3]` — nearest‑neighbor upsample for the
/// low 8 Y lanes of a 16‑pixel block.
#[inline(always)]
pub(super) fn dup_lo(chroma: v128) -> v128 {
  i8x16_shuffle::<0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7>(chroma, chroma)
}

/// Duplicates the high 4 × i16 lanes of `chroma` into 8 lanes
/// `[c4,c4, c5,c5, c6,c6, c7,c7]` — upsample for the high 8 Y lanes.
#[inline(always)]
pub(super) fn dup_hi(chroma: v128) -> v128 {
  i8x16_shuffle::<8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15>(chroma, chroma)
}

/// Writes 16 pixels of packed RGB (48 bytes) from three u8x16 channel
/// vectors, using the SSSE3‑style 3‑way interleave pattern. `u8x16_swizzle`
/// treats indices ≥ 16 as "zero the lane" — same semantics as
/// `_mm_shuffle_epi8`, so the same shuffle masks apply.
///
/// # Safety
///
/// `ptr` must point to at least 48 writable bytes.
#[inline(always)]
pub(super) unsafe fn write_rgb_16(r: v128, g: v128, b: v128, ptr: *mut u8) {
  unsafe {
    // Block 0 (bytes 0..16): [R0,G0,B0, R1,G1,B1, ..., R5].
    // `-1` as i8 is 0xFF ≥ 16 → zeroes that output lane.
    let r0 = i8x16(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g0 = i8x16(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let b0 = i8x16(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(r, r0), u8x16_swizzle(g, g0)),
      u8x16_swizzle(b, b0),
    );

    // Block 1 (bytes 16..32): [G5,B5, R6,G6,B6, ..., G10].
    let r1 = i8x16(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g1 = i8x16(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let b1 = i8x16(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(r, r1), u8x16_swizzle(g, g1)),
      u8x16_swizzle(b, b1),
    );

    // Block 2 (bytes 32..48): [B10, R11,G11,B11, ..., B15].
    let r2 = i8x16(
      -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
    );
    let g2 = i8x16(
      -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
    );
    let b2 = i8x16(
      10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
    );
    let out2 = v128_or(
      v128_or(u8x16_swizzle(r, r2), u8x16_swizzle(g, g2)),
      u8x16_swizzle(b, b2),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(16).cast(), out1);
    v128_store(ptr.add(32).cast(), out2);
  }
}

/// Writes 16 pixels of packed RGBA (64 bytes) from four u8x16 channel
/// vectors. Mirror of [`write_rgb_16`] for the 4-channel output path.
///
/// The 4-byte stride aligns cleanly with the 16-byte register width:
/// each output block holds exactly 4 RGBA quads (16 bytes), with R,
/// G, B, A interleaved at positions `(0, 1, 2, 3)`, `(4, 5, 6, 7)`,
/// etc. `u8x16_swizzle` indices ≥ 16 zero the lane.
///
/// # Safety
///
/// `ptr` must point to at least 64 writable bytes.
#[inline(always)]
pub(super) unsafe fn write_rgba_16(r: v128, g: v128, b: v128, a: v128, ptr: *mut u8) {
  unsafe {
    // Block 0 (bytes 0..16): pixels 0..3, source bytes 0..3.
    let r0 = i8x16(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);
    let g0 = i8x16(-1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1);
    let b0 = i8x16(-1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1);
    let a0 = i8x16(-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(r, r0), u8x16_swizzle(g, g0)),
      v128_or(u8x16_swizzle(b, b0), u8x16_swizzle(a, a0)),
    );

    // Block 1 (bytes 16..32): pixels 4..7, source bytes 4..7.
    let r1 = i8x16(4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1, -1);
    let g1 = i8x16(-1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1);
    let b1 = i8x16(-1, -1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1);
    let a1 = i8x16(-1, -1, -1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(r, r1), u8x16_swizzle(g, g1)),
      v128_or(u8x16_swizzle(b, b1), u8x16_swizzle(a, a1)),
    );

    // Block 2 (bytes 32..48): pixels 8..11, source bytes 8..11.
    let r2 = i8x16(8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1, -1);
    let g2 = i8x16(-1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1);
    let b2 = i8x16(-1, -1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1);
    let a2 = i8x16(-1, -1, -1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11);
    let out2 = v128_or(
      v128_or(u8x16_swizzle(r, r2), u8x16_swizzle(g, g2)),
      v128_or(u8x16_swizzle(b, b2), u8x16_swizzle(a, a2)),
    );

    // Block 3 (bytes 48..64): pixels 12..15, source bytes 12..15.
    let r3 = i8x16(
      12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1,
    );
    let g3 = i8x16(
      -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1,
    );
    let b3 = i8x16(
      -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1,
    );
    let a3 = i8x16(
      -1, -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15,
    );
    let out3 = v128_or(
      v128_or(u8x16_swizzle(r, r3), u8x16_swizzle(g, g3)),
      v128_or(u8x16_swizzle(b, b3), u8x16_swizzle(a, a3)),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(16).cast(), out1);
    v128_store(ptr.add(32).cast(), out2);
    v128_store(ptr.add(48).cast(), out3);
  }
}

// ===== 16-bit YUV → RGB ==================================================

/// `(Y_u16x8 - y_off) * y_scale + RND >> 15` for full u16 Y samples.
/// Unsigned widening via `u32x4_extend_{low,high}_u16x8`. Returns i16x8.
#[inline(always)]
pub(super) fn scale_y_u16_wasm(y_u16: v128, y_off32_v: v128, y_scale_v: v128, rnd_v: v128) -> v128 {
  // y_off32_v = i32x4_splat(y_off)
  let lo_u32 = u32x4_extend_low_u16x8(y_u16);
  let hi_u32 = u32x4_extend_high_u16x8(y_u16);
  let lo_i32 = i32x4_sub(lo_u32, y_off32_v);
  let hi_i32 = i32x4_sub(hi_u32, y_off32_v);
  let lo = q15_shift(i32x4_add(i32x4_mul(lo_i32, y_scale_v), rnd_v));
  let hi = q15_shift(i32x4_add(i32x4_mul(hi_i32, y_scale_v), rnd_v));
  i16x8_narrow_i32x4(lo, hi)
}

/// Computes 2 × i64 chroma products with the Q15 shift using native
/// wasm simd128 `i64x2_mul` + `i64x2_shr` (signed arithmetic right
/// shift). All inputs are i64x2; `cu_i64` / `cv_i64` are the
/// coefficient broadcast widened once per row via
/// `i64x2_extend_low_i32x4`.
#[inline(always)]
pub(super) fn chroma_i64x2_wasm(
  cu_i64: v128,
  cv_i64: v128,
  u_d_i64: v128,
  v_d_i64: v128,
  rnd_i64: v128,
) -> v128 {
  let sum = i64x2_add(
    i64x2_add(i64x2_mul(cu_i64, u_d_i64), i64x2_mul(cv_i64, v_d_i64)),
    rnd_i64,
  );
  i64x2_shr(sum, 15)
}

/// Combines two i64x2 vectors into an i32x4 of their low 32 bits.
/// Valid when each i64 fits in i32 (true for our Q15-shifted chroma
/// and Y-scale results).
///
/// Uses `i8x16_shuffle` since wasm simd128 does not provide a direct
/// i64x2 → i32x2 narrow primitive.
#[inline(always)]
pub(super) fn combine_i64x2_pair_to_i32x4(lo: v128, hi: v128) -> v128 {
  // Byte indices: low 4 bytes of each i64 lane.
  i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(lo, hi)
}

/// Duplicates each i32 lane of `chroma` into a pair for the 4:2:0 u16
/// output pipeline: `[c0, c1, c2, c3]` →
/// Return.0 = `[c0, c0, c1, c1]`, Return.1 = `[c2, c2, c3, c3]`.
#[inline(always)]
pub(super) fn chroma_dup_i32x4_u16(chroma: v128) -> (v128, v128) {
  let lo = i8x16_shuffle::<0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7>(chroma, chroma);
  let hi =
    i8x16_shuffle::<8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15>(chroma, chroma);
  (lo, hi)
}

/// `(y_minus_off * y_scale + RND) >> 15` computed in i64 for all 4
/// lanes of an i32x4 Y stream, returning i32x4.
#[inline(always)]
pub(super) fn scale_y_i32x4_i64_wasm(y_minus_off: v128, y_scale_i64: v128, rnd_i64: v128) -> v128 {
  let lo = i64x2_shr(
    i64x2_add(
      i64x2_mul(y_scale_i64, i64x2_extend_low_i32x4(y_minus_off)),
      rnd_i64,
    ),
    15,
  );
  let hi = i64x2_shr(
    i64x2_add(
      i64x2_mul(y_scale_i64, i64x2_extend_high_i32x4(y_minus_off)),
      rnd_i64,
    ),
    15,
  );
  combine_i64x2_pair_to_i32x4(lo, hi)
}

/// WASM simd128 YUV 4:2:0 16-bit → packed **8-bit** RGB. 16 pixels per iteration.
/// UV centering via wrapping 0x8000 trick; unsigned Y widening.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.

#[cfg(all(test, feature = "std"))]
mod tests;
