//! wasm-simd128 kernels for 32-bit packed RGB / RGBA sources (Rgb96 / Rgba128).
//!
//! ## Format layouts
//!
//! | Format  | Elements per pixel | Channel order in memory |
//! |---------|--------------------|------------------------|
//! | Rgb96   | 3 u32              | R, G, B                |
//! | Rgba128 | 4 u32              | R, G, B, A             |
//!
//! ## Per-format SIMD strategy (8 pixels per SIMD iteration via v128)
//!
//! Each 4-pixel sub-group is loaded as three (Rgb96) / four (Rgba128) `v128`
//! registers (4 u32 lanes each), byte-swapped per u32 lane when `BE = true`,
//! and deinterleaved into per-channel `u32x4` lane vectors with `u8x16_swizzle`
//! (the same byte-gather masks as the SSE4.1 sibling). Two sub-groups feed the
//! shared [`super::write_rgb_16`] / [`super::write_rgba_16`] /
//! [`super::write_rgb_u16_8`] / [`super::write_rgba_u16_8`] writers.
//!
//! ## Depth conversion
//!
//! - **u32 → u8:**  `u32x4_shr(v, 24)` + `u16x8_narrow_i32x4` + `u8x16_narrow_i16x8`
//!   (net `>> 24`, matching scalar `(v >> 24) as u8`).
//! - **u32 → u16:** `u32x4_shr(v, 16)` + `u16x8_narrow_i32x4` (`>> 16`, matching
//!   scalar `(v >> 16) as u16`). Values fit their target width after the shift,
//!   so the saturating narrows never clamp.
//!
//! ## Scalar tail
//!
//! All kernels handle `width % 8` remaining pixels via the scalar reference.
// Kernels are wired into the dispatcher in the dispatch-wiring step; suppress
// dead_code until then.
#![allow(dead_code)]

use core::arch::wasm32::*;

use super::*;

// ---- endian byte-swap helper ------------------------------------------------

/// Compile-time host endianness. `true` on BE targets, `false` on LE.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Conditionally byte-swap every u32 lane in `v` into host-native order. The
/// gate is `BE != HOST_NATIVE_BE`; wasm32 is always little-endian, so the swap
/// fires exactly when `BE = true`.
#[inline(always)]
unsafe fn byteswap32_if_be<const BE: bool>(v: v128) -> v128 {
  if BE != HOST_NATIVE_BE {
    u8x16_swizzle(
      v,
      i8x16(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12),
    )
  } else {
    v
  }
}

/// Deinterleave 4 pixels of stride-3 u32 (Rgb96) into `(R, G, B)` `u32x4`
/// channel lane vectors. See the SSE4.1 sibling for the mask derivation.
///
/// # Safety
///
/// Caller must hold the `simd128` target_feature.
#[inline(always)]
unsafe fn deinterleave_rgb96_4px(v0: v128, v1: v128, v2: v128) -> (v128, v128, v128) {
  let r_v0 = i8x16(0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
  let r_v1 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
  let r_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
  let r = v128_or(
    v128_or(u8x16_swizzle(v0, r_v0), u8x16_swizzle(v1, r_v1)),
    u8x16_swizzle(v2, r_v2),
  );

  let g_v0 = i8x16(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  let g_v1 = i8x16(-1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1);
  let g_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
  let g = v128_or(
    v128_or(u8x16_swizzle(v0, g_v0), u8x16_swizzle(v1, g_v1)),
    u8x16_swizzle(v2, g_v2),
  );

  let b_v0 = i8x16(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  let b_v1 = i8x16(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
  let b_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15);
  let b = v128_or(
    v128_or(u8x16_swizzle(v0, b_v0), u8x16_swizzle(v1, b_v1)),
    u8x16_swizzle(v2, b_v2),
  );

  (r, g, b)
}

/// Loads, byte-swaps, and deinterleaves 4 pixels of Rgb96 into `u32x4` planes.
///
/// # Safety
///
/// `ptr` must point to at least 12 readable u32; `simd128` must be held.
#[inline(always)]
unsafe fn load_deint_rgb96_4px<const BE: bool>(ptr: *const u32) -> (v128, v128, v128) {
  unsafe {
    let v0 = byteswap32_if_be::<BE>(v128_load(ptr.cast()));
    let v1 = byteswap32_if_be::<BE>(v128_load(ptr.add(4).cast()));
    let v2 = byteswap32_if_be::<BE>(v128_load(ptr.add(8).cast()));
    deinterleave_rgb96_4px(v0, v1, v2)
  }
}

/// Narrows two `u32x4` planes (`>> 16` applied) into a `u16x8` channel vector.
#[inline(always)]
unsafe fn narrow_pair_u32_to_u16x8(lo: v128, hi: v128) -> v128 {
  u16x8_narrow_i32x4(u32x4_shr(lo, 16), u32x4_shr(hi, 16))
}

/// Narrows two `u32x4` planes (`>> 24` applied) into a `u8x16` channel vector
/// whose low 8 bytes carry the 8 valid samples.
#[inline(always)]
unsafe fn narrow_pair_u32_to_u8x8(lo: v128, hi: v128) -> v128 {
  let u16 = u16x8_narrow_i32x4(u32x4_shr(lo, 24), u32x4_shr(hi, 24));
  u8x16_narrow_i16x8(u16, u16x8_splat(0))
}

// Rgb96 (R, G, B — 3 u32 elements per pixel).

/// wasm Rgb96 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. The `simd128` target_feature must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgb96_to_rgb_row<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgb96.as_ptr().add(x * 3);
      let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(p);
      let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(p.add(12));
      let r = narrow_pair_u32_to_u8x8(r0, r1);
      let g = narrow_pair_u32_to_u8x8(g0, g1);
      let b = narrow_pair_u32_to_u8x8(b0, b1);
      // write_rgb_16 writes 16 px (48 bytes); only first 8 px (24 bytes) valid.
      let mut tmp = [0u8; 48];
      write_rgb_16(r, g, b, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgb_row::<BE>(&rgb96[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm Rgb96 → packed u8 RGBA. 8 pixels per SIMD iteration. Alpha forced to 0xFF.
///
/// # Safety
///
/// 1. The `simd128` target_feature must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgb96_to_rgba_row<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgb96.as_ptr().add(x * 3);
      let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(p);
      let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(p.add(12));
      let r = narrow_pair_u32_to_u8x8(r0, r1);
      let g = narrow_pair_u32_to_u8x8(g0, g1);
      let b = narrow_pair_u32_to_u8x8(b0, b1);
      let mut tmp = [0u8; 64];
      write_rgba_16(r, g, b, opaque, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgba_row::<BE>(&rgb96[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// wasm Rgb96 → native-depth u16 RGB. 8 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. The `simd128` target_feature must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgb96_to_rgb_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgb96.as_ptr().add(x * 3);
      let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(p);
      let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(p.add(12));
      let r = narrow_pair_u32_to_u16x8(r0, r1);
      let g = narrow_pair_u32_to_u16x8(g0, g1);
      let b = narrow_pair_u32_to_u16x8(b0, b1);
      write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgb_u16_row::<BE>(&rgb96[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm Rgb96 → native-depth u16 RGBA. 8 pixels per SIMD iteration. Alpha 0xFFFF.
///
/// # Safety
///
/// 1. The `simd128` target_feature must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgb96_to_rgba_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgb96.as_ptr().add(x * 3);
      let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(p);
      let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(p.add(12));
      let r = narrow_pair_u32_to_u16x8(r0, r1);
      let g = narrow_pair_u32_to_u16x8(g0, g1);
      let b = narrow_pair_u32_to_u16x8(b0, b1);
      write_rgba_u16_8(r, g, b, opaque, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgba_u16_row::<BE>(&rgb96[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// ---- Rgba128 deinterleave (4 u32 per pixel, 4 loads per 4 px) ---------------

/// Deinterleave 4 pixels of stride-4 u32 (Rgba128) into `(R, G, B, A)` `u32x4`
/// lane vectors via the compile-time `i32x4_shuffle` 4x4 transpose ladder
/// (the wasm analogue of the SSE `unpacklo/hi` ladder).
///
/// # Safety
///
/// Caller must hold the `simd128` target_feature.
#[inline(always)]
unsafe fn deinterleave_rgba128_4px(
  v0: v128,
  v1: v128,
  v2: v128,
  v3: v128,
) -> (v128, v128, v128, v128) {
  // v_n = [Rn, Gn, Bn, An].
  let t0 = i32x4_shuffle::<0, 4, 1, 5>(v0, v1); // [R0, R1, G0, G1]
  let t1 = i32x4_shuffle::<2, 6, 3, 7>(v0, v1); // [B0, B1, A0, A1]
  let t2 = i32x4_shuffle::<0, 4, 1, 5>(v2, v3); // [R2, R3, G2, G3]
  let t3 = i32x4_shuffle::<2, 6, 3, 7>(v2, v3); // [B2, B3, A2, A3]
  let r = i32x4_shuffle::<0, 1, 4, 5>(t0, t2); // [R0, R1, R2, R3]
  let g = i32x4_shuffle::<2, 3, 6, 7>(t0, t2); // [G0, G1, G2, G3]
  let b = i32x4_shuffle::<0, 1, 4, 5>(t1, t3); // [B0, B1, B2, B3]
  let a = i32x4_shuffle::<2, 3, 6, 7>(t1, t3); // [A0, A1, A2, A3]
  (r, g, b, a)
}

/// Loads, byte-swaps, and deinterleaves 4 pixels of Rgba128 into `u32x4` planes.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable u32; `simd128` must be held.
#[inline(always)]
unsafe fn load_deint_rgba128_4px<const BE: bool>(ptr: *const u32) -> (v128, v128, v128, v128) {
  unsafe {
    let v0 = byteswap32_if_be::<BE>(v128_load(ptr.cast()));
    let v1 = byteswap32_if_be::<BE>(v128_load(ptr.add(4).cast()));
    let v2 = byteswap32_if_be::<BE>(v128_load(ptr.add(8).cast()));
    let v3 = byteswap32_if_be::<BE>(v128_load(ptr.add(12).cast()));
    deinterleave_rgba128_4px(v0, v1, v2, v3)
  }
}

// Rgba128 (R, G, B, A — 4 u32 elements per pixel).

/// wasm Rgba128 → packed u8 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// # Safety
///
/// 1. The `simd128` target_feature must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgba128_to_rgb_row<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgba128.as_ptr().add(x * 4);
      let (r0, g0, b0, _) = load_deint_rgba128_4px::<BE>(p);
      let (r1, g1, b1, _) = load_deint_rgba128_4px::<BE>(p.add(16));
      let r = narrow_pair_u32_to_u8x8(r0, r1);
      let g = narrow_pair_u32_to_u8x8(g0, g1);
      let b = narrow_pair_u32_to_u8x8(b0, b1);
      let mut tmp = [0u8; 48];
      write_rgb_16(r, g, b, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgb_row::<BE>(&rgba128[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm Rgba128 → packed u8 RGBA. 8 pixels per SIMD iteration. Alpha passed through.
///
/// # Safety
///
/// 1. The `simd128` target_feature must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgba128_to_rgba_row<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgba128.as_ptr().add(x * 4);
      let (r0, g0, b0, a0) = load_deint_rgba128_4px::<BE>(p);
      let (r1, g1, b1, a1) = load_deint_rgba128_4px::<BE>(p.add(16));
      let r = narrow_pair_u32_to_u8x8(r0, r1);
      let g = narrow_pair_u32_to_u8x8(g0, g1);
      let b = narrow_pair_u32_to_u8x8(b0, b1);
      let a = narrow_pair_u32_to_u8x8(a0, a1);
      let mut tmp = [0u8; 64];
      write_rgba_16(r, g, b, a, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgba_row::<BE>(&rgba128[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// wasm Rgba128 → native-depth u16 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// # Safety
///
/// 1. The `simd128` target_feature must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgba128_to_rgb_u16_row<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgba128.as_ptr().add(x * 4);
      let (r0, g0, b0, _) = load_deint_rgba128_4px::<BE>(p);
      let (r1, g1, b1, _) = load_deint_rgba128_4px::<BE>(p.add(16));
      let r = narrow_pair_u32_to_u16x8(r0, r1);
      let g = narrow_pair_u32_to_u16x8(g0, g1);
      let b = narrow_pair_u32_to_u16x8(b0, b1);
      write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgb_u16_row::<BE>(&rgba128[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm Rgba128 → native-depth u16 RGBA. 8 pixels per SIMD iteration. Alpha passed through.
///
/// # Safety
///
/// 1. The `simd128` target_feature must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgba128_to_rgba_u16_row<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgba128.as_ptr().add(x * 4);
      let (r0, g0, b0, a0) = load_deint_rgba128_4px::<BE>(p);
      let (r1, g1, b1, a1) = load_deint_rgba128_4px::<BE>(p.add(16));
      let r = narrow_pair_u32_to_u16x8(r0, r1);
      let g = narrow_pair_u32_to_u16x8(g0, g1);
      let b = narrow_pair_u32_to_u16x8(b0, b1);
      let a = narrow_pair_u32_to_u16x8(a0, a1);
      write_rgba_u16_8(r, g, b, a, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgba_u16_row::<BE>(&rgba128[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}
