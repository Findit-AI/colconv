use core::arch::wasm32::*;

use super::*;

// ===== BGR ↔ RGB byte swap ==============================================

/// WASM simd128 BGR ↔ RGB byte swap. 16 pixels per iteration via the
/// same 7‑shuffle + 4‑OR pattern as the x86 / NEON backends.
/// `u8x16_swizzle` matches `_mm_shuffle_epi8` semantics (indices ≥ 16
/// zero the output lane), so the mask values translate directly.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  unsafe {
    // Precomputed byte‑shuffle masks. See the x86_common::swap_rb_16_pixels
    // comments for the derivation — identical pattern at 128‑bit width.
    let m00 = i8x16(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, -1);
    let m01 = i8x16(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    );
    let m10 = i8x16(
      -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m11 = i8x16(0, -1, 4, 3, 2, 7, 6, 5, 10, 9, 8, 13, 12, 11, -1, 15);
    let m12 = i8x16(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1,
    );
    let m20 = i8x16(
      14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m21 = i8x16(-1, 3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10, 15, 14, 13);

    let mut x = 0usize;
    while x + 16 <= width {
      let in0 = v128_load(input.as_ptr().add(x * 3).cast());
      let in1 = v128_load(input.as_ptr().add(x * 3 + 16).cast());
      let in2 = v128_load(input.as_ptr().add(x * 3 + 32).cast());

      let out0 = v128_or(u8x16_swizzle(in0, m00), u8x16_swizzle(in1, m01));
      let out1 = v128_or(
        v128_or(u8x16_swizzle(in0, m10), u8x16_swizzle(in1, m11)),
        u8x16_swizzle(in2, m12),
      );
      let out2 = v128_or(u8x16_swizzle(in1, m20), u8x16_swizzle(in2, m21));

      v128_store(output.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(output.as_mut_ptr().add(x * 3 + 16).cast(), out1);
      v128_store(output.as_mut_ptr().add(x * 3 + 32).cast(), out2);

      x += 16;
    }
    if x < width {
      scalar::bgr_rgb_swap_row(
        &input[x * 3..width * 3],
        &mut output[x * 3..width * 3],
        width - x,
      );
    }
  }
}

// ===== Packed-RGBA shuffles (Ship 9b) ====================================

/// WASM simd128 RGBA→RGB drop-alpha. 16 pixels per iteration via the
/// same 6-shuffle + 3-OR pattern as the x86 backends. Mask values are
/// identical because `u8x16_swizzle` matches `_mm_shuffle_epi8`
/// semantics (indices ≥ 16 zero the output lane).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `rgba.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgba` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgba_to_rgb_row(rgba: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgba.len() >= width * 4, "rgba row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let m00 = i8x16(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let m01 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 4);
    let m11 = i8x16(5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let m12 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 4, 5, 6, 8, 9);
    let m22 = i8x16(
      10, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m23 = i8x16(-1, -1, -1, -1, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14);

    let mut x = 0usize;
    while x + 16 <= width {
      let in0 = v128_load(rgba.as_ptr().add(x * 4).cast());
      let in1 = v128_load(rgba.as_ptr().add(x * 4 + 16).cast());
      let in2 = v128_load(rgba.as_ptr().add(x * 4 + 32).cast());
      let in3 = v128_load(rgba.as_ptr().add(x * 4 + 48).cast());

      let out0 = v128_or(u8x16_swizzle(in0, m00), u8x16_swizzle(in1, m01));
      let out1 = v128_or(u8x16_swizzle(in1, m11), u8x16_swizzle(in2, m12));
      let out2 = v128_or(u8x16_swizzle(in2, m22), u8x16_swizzle(in3, m23));

      v128_store(rgb_out.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 16).cast(), out1);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 32).cast(), out2);

      x += 16;
    }
    if x < width {
      scalar::rgba_to_rgb_row(
        &rgba[x * 4..width * 4],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// WASM simd128 BGRA→RGBA R↔B swap with alpha pass-through. 16 pixels
/// per iteration via four `u8x16_swizzle` calls (one per 16-byte
/// vector, four pixels each). Within each 4-byte pixel, byte 0 ↔
/// byte 2; alpha at byte 3 is unchanged.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `bgra.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `bgra` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgra_to_rgba_row(bgra: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask = i8x16(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = bgra.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let v0 = v128_load(base_in.cast());
      let v1 = v128_load(base_in.add(16).cast());
      let v2 = v128_load(base_in.add(32).cast());
      let v3 = v128_load(base_in.add(48).cast());
      v128_store(base_out.cast(), u8x16_swizzle(v0, mask));
      v128_store(base_out.add(16).cast(), u8x16_swizzle(v1, mask));
      v128_store(base_out.add(32).cast(), u8x16_swizzle(v2, mask));
      v128_store(base_out.add(48).cast(), u8x16_swizzle(v3, mask));
      x += 16;
    }
    if x < width {
      scalar::bgra_to_rgba_row(
        &bgra[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// WASM simd128 BGRA→RGB combined R↔B swap and alpha drop. 16 pixels
/// per iteration via the same compaction shape as
/// [`rgba_to_rgb_row`], with each pixel triple read from the input as
/// `(byte+2, byte+1, byte+0)` to flip channel order while dropping
/// alpha at `byte+3`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `bgra.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `bgra` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgra_to_rgb_row(bgra: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let m00 = i8x16(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
    let m01 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0, 6);
    let m11 = i8x16(5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1, -1, -1, -1, -1);
    let m12 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0, 6, 5, 4, 10, 9);
    let m22 = i8x16(
      8, 14, 13, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m23 = i8x16(-1, -1, -1, -1, 2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12);

    let mut x = 0usize;
    while x + 16 <= width {
      let in0 = v128_load(bgra.as_ptr().add(x * 4).cast());
      let in1 = v128_load(bgra.as_ptr().add(x * 4 + 16).cast());
      let in2 = v128_load(bgra.as_ptr().add(x * 4 + 32).cast());
      let in3 = v128_load(bgra.as_ptr().add(x * 4 + 48).cast());

      let out0 = v128_or(u8x16_swizzle(in0, m00), u8x16_swizzle(in1, m01));
      let out1 = v128_or(u8x16_swizzle(in1, m11), u8x16_swizzle(in2, m12));
      let out2 = v128_or(u8x16_swizzle(in2, m22), u8x16_swizzle(in3, m23));

      v128_store(rgb_out.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 16).cast(), out1);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 32).cast(), out2);

      x += 16;
    }
    if x < width {
      scalar::bgra_to_rgb_row(
        &bgra[x * 4..width * 4],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

// ===== Leading-alpha shuffles (Ship 9c) ==================================

/// WASM simd128 ARGB→RGB drop-leading-alpha. 16 pixels per iteration
/// using the same compaction shape as [`rgba_to_rgb_row`] but with
/// pixel triple offsets `(+1, +2, +3)`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `argb.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `argb` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn argb_to_rgb_row(argb: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let m00 = i8x16(1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, -1, -1, -1, -1);
    let m01 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 5);
    let m11 = i8x16(6, 7, 9, 10, 11, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let m12 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 5, 6, 7, 9, 10);
    let m22 = i8x16(
      11, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m23 = i8x16(-1, -1, -1, -1, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15);

    let mut x = 0usize;
    while x + 16 <= width {
      let in0 = v128_load(argb.as_ptr().add(x * 4).cast());
      let in1 = v128_load(argb.as_ptr().add(x * 4 + 16).cast());
      let in2 = v128_load(argb.as_ptr().add(x * 4 + 32).cast());
      let in3 = v128_load(argb.as_ptr().add(x * 4 + 48).cast());

      let out0 = v128_or(u8x16_swizzle(in0, m00), u8x16_swizzle(in1, m01));
      let out1 = v128_or(u8x16_swizzle(in1, m11), u8x16_swizzle(in2, m12));
      let out2 = v128_or(u8x16_swizzle(in2, m22), u8x16_swizzle(in3, m23));

      v128_store(rgb_out.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 16).cast(), out1);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 32).cast(), out2);

      x += 16;
    }
    if x < width {
      scalar::argb_to_rgb_row(
        &argb[x * 4..width * 4],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// WASM simd128 ABGR→RGB combined drop-leading-alpha + R↔B swap.
/// Per-pixel input offsets are read in reverse order `(+3, +2, +1)`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `abgr.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `abgr` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn abgr_to_rgb_row(abgr: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let m00 = i8x16(3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13, -1, -1, -1, -1);
    let m01 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 2, 1, 7);
    let m11 = i8x16(6, 5, 11, 10, 9, 15, 14, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let m12 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 3, 2, 1, 7, 6, 5, 11, 10);
    let m22 = i8x16(
      9, 15, 14, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m23 = i8x16(-1, -1, -1, -1, 3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13);

    let mut x = 0usize;
    while x + 16 <= width {
      let in0 = v128_load(abgr.as_ptr().add(x * 4).cast());
      let in1 = v128_load(abgr.as_ptr().add(x * 4 + 16).cast());
      let in2 = v128_load(abgr.as_ptr().add(x * 4 + 32).cast());
      let in3 = v128_load(abgr.as_ptr().add(x * 4 + 48).cast());

      let out0 = v128_or(u8x16_swizzle(in0, m00), u8x16_swizzle(in1, m01));
      let out1 = v128_or(u8x16_swizzle(in1, m11), u8x16_swizzle(in2, m12));
      let out2 = v128_or(u8x16_swizzle(in2, m22), u8x16_swizzle(in3, m23));

      v128_store(rgb_out.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 16).cast(), out1);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 32).cast(), out2);

      x += 16;
    }
    if x < width {
      scalar::abgr_to_rgb_row(
        &abgr[x * 4..width * 4],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// WASM simd128 ARGB→RGBA leading-alpha rotation. 16 pixels per
/// iteration via four `u8x16_swizzle` calls.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `argb.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `argb` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn argb_to_rgba_row(argb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask = i8x16(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = argb.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let v0 = v128_load(base_in.cast());
      let v1 = v128_load(base_in.add(16).cast());
      let v2 = v128_load(base_in.add(32).cast());
      let v3 = v128_load(base_in.add(48).cast());
      v128_store(base_out.cast(), u8x16_swizzle(v0, mask));
      v128_store(base_out.add(16).cast(), u8x16_swizzle(v1, mask));
      v128_store(base_out.add(32).cast(), u8x16_swizzle(v2, mask));
      v128_store(base_out.add(48).cast(), u8x16_swizzle(v3, mask));
      x += 16;
    }
    if x < width {
      scalar::argb_to_rgba_row(
        &argb[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// WASM simd128 ABGR→RGBA full byte reverse. 16 pixels per iteration
/// via four `u8x16_swizzle` calls.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `abgr.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `abgr` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn abgr_to_rgba_row(abgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask = i8x16(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = abgr.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let v0 = v128_load(base_in.cast());
      let v1 = v128_load(base_in.add(16).cast());
      let v2 = v128_load(base_in.add(32).cast());
      let v3 = v128_load(base_in.add(48).cast());
      v128_store(base_out.cast(), u8x16_swizzle(v0, mask));
      v128_store(base_out.add(16).cast(), u8x16_swizzle(v1, mask));
      v128_store(base_out.add(32).cast(), u8x16_swizzle(v2, mask));
      v128_store(base_out.add(48).cast(), u8x16_swizzle(v3, mask));
      x += 16;
    }
    if x < width {
      scalar::abgr_to_rgba_row(
        &abgr[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ===== Padding-byte to RGBA shuffles (Ship 9d) ===========================

/// WASM simd128 XRGB→RGBA. 16 pixels per iteration via four
/// `u8x16_swizzle` + `v128_or` calls. Each pixel: bytes 1,2,3 → R,G,B
/// output positions; alpha lane forced to `0xFF` via OR.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `xrgb.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `xrgb` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xrgb_to_rgba_row(xrgb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xrgb.len() >= width * 4, "xrgb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask = i8x16(1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, -1, 13, 14, 15, -1);
    // Sentinel `-1` after `as i8` becomes `0xFF` — exactly what we
    // need in the alpha lanes.
    let alpha = i8x16(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = xrgb.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let v0 = v128_load(base_in.cast());
      let v1 = v128_load(base_in.add(16).cast());
      let v2 = v128_load(base_in.add(32).cast());
      let v3 = v128_load(base_in.add(48).cast());
      v128_store(base_out.cast(), v128_or(u8x16_swizzle(v0, mask), alpha));
      v128_store(
        base_out.add(16).cast(),
        v128_or(u8x16_swizzle(v1, mask), alpha),
      );
      v128_store(
        base_out.add(32).cast(),
        v128_or(u8x16_swizzle(v2, mask), alpha),
      );
      v128_store(
        base_out.add(48).cast(),
        v128_or(u8x16_swizzle(v3, mask), alpha),
      );
      x += 16;
    }
    if x < width {
      scalar::xrgb_to_rgba_row(
        &xrgb[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// WASM simd128 RGBX→RGBA. 16 pixels per iteration.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `rgbx.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `rgbx` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbx_to_rgba_row(rgbx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgbx.len() >= width * 4, "rgbx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask = i8x16(0, 1, 2, -1, 4, 5, 6, -1, 8, 9, 10, -1, 12, 13, 14, -1);
    let alpha = i8x16(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = rgbx.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let v0 = v128_load(base_in.cast());
      let v1 = v128_load(base_in.add(16).cast());
      let v2 = v128_load(base_in.add(32).cast());
      let v3 = v128_load(base_in.add(48).cast());
      v128_store(base_out.cast(), v128_or(u8x16_swizzle(v0, mask), alpha));
      v128_store(
        base_out.add(16).cast(),
        v128_or(u8x16_swizzle(v1, mask), alpha),
      );
      v128_store(
        base_out.add(32).cast(),
        v128_or(u8x16_swizzle(v2, mask), alpha),
      );
      v128_store(
        base_out.add(48).cast(),
        v128_or(u8x16_swizzle(v3, mask), alpha),
      );
      x += 16;
    }
    if x < width {
      scalar::rgbx_to_rgba_row(
        &rgbx[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// WASM simd128 XBGR→RGBA. 16 pixels per iteration.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `xbgr.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `xbgr` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xbgr_to_rgba_row(xbgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xbgr.len() >= width * 4, "xbgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask = i8x16(3, 2, 1, -1, 7, 6, 5, -1, 11, 10, 9, -1, 15, 14, 13, -1);
    let alpha = i8x16(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = xbgr.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let v0 = v128_load(base_in.cast());
      let v1 = v128_load(base_in.add(16).cast());
      let v2 = v128_load(base_in.add(32).cast());
      let v3 = v128_load(base_in.add(48).cast());
      v128_store(base_out.cast(), v128_or(u8x16_swizzle(v0, mask), alpha));
      v128_store(
        base_out.add(16).cast(),
        v128_or(u8x16_swizzle(v1, mask), alpha),
      );
      v128_store(
        base_out.add(32).cast(),
        v128_or(u8x16_swizzle(v2, mask), alpha),
      );
      v128_store(
        base_out.add(48).cast(),
        v128_or(u8x16_swizzle(v3, mask), alpha),
      );
      x += 16;
    }
    if x < width {
      scalar::xbgr_to_rgba_row(
        &xbgr[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// WASM simd128 BGRX→RGBA. 16 pixels per iteration.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `bgrx.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `bgrx` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgrx_to_rgba_row(bgrx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgrx.len() >= width * 4, "bgrx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask = i8x16(2, 1, 0, -1, 6, 5, 4, -1, 10, 9, 8, -1, 14, 13, 12, -1);
    let alpha = i8x16(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = bgrx.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let v0 = v128_load(base_in.cast());
      let v1 = v128_load(base_in.add(16).cast());
      let v2 = v128_load(base_in.add(32).cast());
      let v3 = v128_load(base_in.add(48).cast());
      v128_store(base_out.cast(), v128_or(u8x16_swizzle(v0, mask), alpha));
      v128_store(
        base_out.add(16).cast(),
        v128_or(u8x16_swizzle(v1, mask), alpha),
      );
      v128_store(
        base_out.add(32).cast(),
        v128_or(u8x16_swizzle(v2, mask), alpha),
      );
      v128_store(
        base_out.add(48).cast(),
        v128_or(u8x16_swizzle(v3, mask), alpha),
      );
      x += 16;
    }
    if x < width {
      scalar::bgrx_to_rgba_row(
        &bgrx[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ===== 10-bit packed RGB shuffles (Ship 9e) ==============================

/// WASM simd128 X2RGB10→RGB. 8 pixels per iteration: load 2 `u32x4`
/// vectors, extract R/G/B as `u16x8`, then narrow to `u8` via a
/// final `u8x16_narrow_i16x8` step that pairs two iterations'
/// channel halves. We process 16 pixels at a time so we can call
/// the existing `write_rgb_16` pattern. This routine inlines the
/// narrowing.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `x2rgb10.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `x2rgb10` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn x2rgb10_to_rgb_row(x2rgb10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mask_3ff = u32x4_splat(0x3FF);
    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = v128_load(x2rgb10.as_ptr().add(x * 4).cast());
      let p1 = v128_load(x2rgb10.as_ptr().add(x * 4 + 16).cast());
      let p2 = v128_load(x2rgb10.as_ptr().add(x * 4 + 32).cast());
      let p3 = v128_load(x2rgb10.as_ptr().add(x * 4 + 48).cast());

      // Extract 10-bit channels as u32x4 (low 10 bits set per lane).
      // X2RGB10: R at >>20, G at >>10, B at >>0.
      let r0 = v128_and(u32x4_shr(p0, 20), mask_3ff);
      let r1 = v128_and(u32x4_shr(p1, 20), mask_3ff);
      let r2 = v128_and(u32x4_shr(p2, 20), mask_3ff);
      let r3 = v128_and(u32x4_shr(p3, 20), mask_3ff);
      let g0 = v128_and(u32x4_shr(p0, 10), mask_3ff);
      let g1 = v128_and(u32x4_shr(p1, 10), mask_3ff);
      let g2 = v128_and(u32x4_shr(p2, 10), mask_3ff);
      let g3 = v128_and(u32x4_shr(p3, 10), mask_3ff);
      let b0 = v128_and(p0, mask_3ff);
      let b1 = v128_and(p1, mask_3ff);
      let b2 = v128_and(p2, mask_3ff);
      let b3 = v128_and(p3, mask_3ff);

      // Down-shift 10-bit → 8-bit.
      let r0_u8 = u32x4_shr(r0, 2);
      let r1_u8 = u32x4_shr(r1, 2);
      let r2_u8 = u32x4_shr(r2, 2);
      let r3_u8 = u32x4_shr(r3, 2);
      let g0_u8 = u32x4_shr(g0, 2);
      let g1_u8 = u32x4_shr(g1, 2);
      let g2_u8 = u32x4_shr(g2, 2);
      let g3_u8 = u32x4_shr(g3, 2);
      let b0_u8 = u32x4_shr(b0, 2);
      let b1_u8 = u32x4_shr(b1, 2);
      let b2_u8 = u32x4_shr(b2, 2);
      let b3_u8 = u32x4_shr(b3, 2);

      // u32x4 → u16x8 (saturating narrow).
      let r_lo = u16x8_narrow_i32x4(r0_u8, r1_u8);
      let r_hi = u16x8_narrow_i32x4(r2_u8, r3_u8);
      let g_lo = u16x8_narrow_i32x4(g0_u8, g1_u8);
      let g_hi = u16x8_narrow_i32x4(g2_u8, g3_u8);
      let b_lo = u16x8_narrow_i32x4(b0_u8, b1_u8);
      let b_hi = u16x8_narrow_i32x4(b2_u8, b3_u8);

      // u16x8 → u8x16.
      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);

      // Interleave (R, G, B) into 48 packed bytes via the same
      // 9-shuffle pattern used by the YUV→RGB kernels.
      let r_mask0 = i8x16(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
      let g_mask0 = i8x16(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
      let b_mask0 = i8x16(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
      let out0 = v128_or(
        v128_or(u8x16_swizzle(r_u8, r_mask0), u8x16_swizzle(g_u8, g_mask0)),
        u8x16_swizzle(b_u8, b_mask0),
      );
      let r_mask1 = i8x16(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
      let g_mask1 = i8x16(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
      let b_mask1 = i8x16(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
      let out1 = v128_or(
        v128_or(u8x16_swizzle(r_u8, r_mask1), u8x16_swizzle(g_u8, g_mask1)),
        u8x16_swizzle(b_u8, b_mask1),
      );
      let r_mask2 = i8x16(
        -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
      );
      let g_mask2 = i8x16(
        -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
      );
      let b_mask2 = i8x16(
        10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
      );
      let out2 = v128_or(
        v128_or(u8x16_swizzle(r_u8, r_mask2), u8x16_swizzle(g_u8, g_mask2)),
        u8x16_swizzle(b_u8, b_mask2),
      );

      v128_store(rgb_out.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 16).cast(), out1);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 32).cast(), out2);

      x += 16;
    }
    if x < width {
      scalar::x2rgb10_to_rgb_row(
        &x2rgb10[x * 4..width * 4],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// WASM simd128 X2RGB10→RGBA. 16 pixels per iteration; alpha forced
/// to `0xFF`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn x2rgb10_to_rgba_row(x2rgb10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask_3ff = u32x4_splat(0x3FF);
    let alpha_const = u32x4_splat(0xFF00_0000);
    let mut x = 0usize;
    while x + 4 <= width {
      let pix = v128_load(x2rgb10.as_ptr().add(x * 4).cast());

      // Extract 10-bit channels into u32 lanes, down-shift to u8.
      let r = v128_and(u32x4_shr(pix, 20), mask_3ff);
      let g = v128_and(u32x4_shr(pix, 10), mask_3ff);
      let b = v128_and(pix, mask_3ff);
      let r = u32x4_shr(r, 2);
      let g = u32x4_shr(g, 2);
      let b = u32x4_shr(b, 2);

      // Pack (R, G, B, 0xFF) bytes per pixel.
      // Each channel value is in low byte of its u32 lane.
      // Shuffle to byte positions: R→[0,4,8,12], G→[1,5,9,13], B→[2,6,10,14], A→[3,7,11,15].
      let r_mask = i8x16(0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);
      let g_mask = i8x16(-1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1);
      let b_mask = i8x16(-1, -1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1);
      let out = v128_or(
        v128_or(
          v128_or(u8x16_swizzle(r, r_mask), u8x16_swizzle(g, g_mask)),
          u8x16_swizzle(b, b_mask),
        ),
        alpha_const,
      );

      v128_store(rgba_out.as_mut_ptr().add(x * 4).cast(), out);
      x += 4;
    }
    if x < width {
      scalar::x2rgb10_to_rgba_row(
        &x2rgb10[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// WASM simd128 X2RGB10→u16 RGB native. 8 pixels per iteration.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn x2rgb10_to_rgb_u16_row(x2rgb10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mask_3ff = u32x4_splat(0x3FF);
    let mut x = 0usize;
    while x + 8 <= width {
      let p0 = v128_load(x2rgb10.as_ptr().add(x * 4).cast());
      let p1 = v128_load(x2rgb10.as_ptr().add(x * 4 + 16).cast());

      let r0 = v128_and(u32x4_shr(p0, 20), mask_3ff);
      let r1 = v128_and(u32x4_shr(p1, 20), mask_3ff);
      let g0 = v128_and(u32x4_shr(p0, 10), mask_3ff);
      let g1 = v128_and(u32x4_shr(p1, 10), mask_3ff);
      let b0 = v128_and(p0, mask_3ff);
      let b1 = v128_and(p1, mask_3ff);

      let r = u16x8_narrow_i32x4(r0, r1);
      let g = u16x8_narrow_i32x4(g0, g1);
      let b = u16x8_narrow_i32x4(b0, b1);

      // Interleave (R, G, B) u16x8 into 24 u16 elements.
      // Element granularity is u16 (2 bytes); shuffle masks below
      // index by byte. For u16-per-element interleave, byte mask
      // pulls 2 consecutive bytes per element.
      let r_mask0 = i8x16(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
      let g_mask0 = i8x16(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
      let b_mask0 = i8x16(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
      let out0 = v128_or(
        v128_or(u8x16_swizzle(r, r_mask0), u8x16_swizzle(g, g_mask0)),
        u8x16_swizzle(b, b_mask0),
      );
      // Block 1 (output u16s 8..15 = [B2, R3, G3, B3, R4, G4, B4, R5]).
      // Each u16 takes 2 bytes; the channel vectors hold element `i` at
      // byte indices `(2*i, 2*i+1)`.
      let r_mask1 = i8x16(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
      let g_mask1 = i8x16(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
      let b_mask1 = i8x16(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
      let out1 = v128_or(
        v128_or(u8x16_swizzle(r, r_mask1), u8x16_swizzle(g, g_mask1)),
        u8x16_swizzle(b, b_mask1),
      );
      // Block 2 (output u16s 16..23 = [G5, B5, R6, G6, B6, R7, G7, B7]).
      let r_mask2 = i8x16(
        -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1,
      );
      let g_mask2 = i8x16(
        10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1,
      );
      let b_mask2 = i8x16(
        -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15,
      );
      let out2 = v128_or(
        v128_or(u8x16_swizzle(r, r_mask2), u8x16_swizzle(g, g_mask2)),
        u8x16_swizzle(b, b_mask2),
      );

      v128_store(rgb_out.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 8).cast(), out1);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 16).cast(), out2);

      x += 8;
    }
    if x < width {
      scalar::x2rgb10_to_rgb_u16_row(
        &x2rgb10[x * 4..width * 4],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// WASM simd128 X2BGR10→RGB. Mirrors [`x2rgb10_to_rgb_row`] but
/// extracts R from low bits and B from high bits.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn x2bgr10_to_rgb_row(x2bgr10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mask_3ff = u32x4_splat(0x3FF);
    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = v128_load(x2bgr10.as_ptr().add(x * 4).cast());
      let p1 = v128_load(x2bgr10.as_ptr().add(x * 4 + 16).cast());
      let p2 = v128_load(x2bgr10.as_ptr().add(x * 4 + 32).cast());
      let p3 = v128_load(x2bgr10.as_ptr().add(x * 4 + 48).cast());

      // X2BGR10: R at low 10, G at >>10, B at >>20.
      let r0 = u32x4_shr(v128_and(p0, mask_3ff), 2);
      let r1 = u32x4_shr(v128_and(p1, mask_3ff), 2);
      let r2 = u32x4_shr(v128_and(p2, mask_3ff), 2);
      let r3 = u32x4_shr(v128_and(p3, mask_3ff), 2);
      let g0 = u32x4_shr(v128_and(u32x4_shr(p0, 10), mask_3ff), 2);
      let g1 = u32x4_shr(v128_and(u32x4_shr(p1, 10), mask_3ff), 2);
      let g2 = u32x4_shr(v128_and(u32x4_shr(p2, 10), mask_3ff), 2);
      let g3 = u32x4_shr(v128_and(u32x4_shr(p3, 10), mask_3ff), 2);
      let b0 = u32x4_shr(v128_and(u32x4_shr(p0, 20), mask_3ff), 2);
      let b1 = u32x4_shr(v128_and(u32x4_shr(p1, 20), mask_3ff), 2);
      let b2 = u32x4_shr(v128_and(u32x4_shr(p2, 20), mask_3ff), 2);
      let b3 = u32x4_shr(v128_and(u32x4_shr(p3, 20), mask_3ff), 2);

      let r_lo = u16x8_narrow_i32x4(r0, r1);
      let r_hi = u16x8_narrow_i32x4(r2, r3);
      let g_lo = u16x8_narrow_i32x4(g0, g1);
      let g_hi = u16x8_narrow_i32x4(g2, g3);
      let b_lo = u16x8_narrow_i32x4(b0, b1);
      let b_hi = u16x8_narrow_i32x4(b2, b3);

      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);

      let r_mask0 = i8x16(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
      let g_mask0 = i8x16(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
      let b_mask0 = i8x16(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
      let out0 = v128_or(
        v128_or(u8x16_swizzle(r_u8, r_mask0), u8x16_swizzle(g_u8, g_mask0)),
        u8x16_swizzle(b_u8, b_mask0),
      );
      let r_mask1 = i8x16(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
      let g_mask1 = i8x16(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
      let b_mask1 = i8x16(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
      let out1 = v128_or(
        v128_or(u8x16_swizzle(r_u8, r_mask1), u8x16_swizzle(g_u8, g_mask1)),
        u8x16_swizzle(b_u8, b_mask1),
      );
      let r_mask2 = i8x16(
        -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
      );
      let g_mask2 = i8x16(
        -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
      );
      let b_mask2 = i8x16(
        10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
      );
      let out2 = v128_or(
        v128_or(u8x16_swizzle(r_u8, r_mask2), u8x16_swizzle(g_u8, g_mask2)),
        u8x16_swizzle(b_u8, b_mask2),
      );

      v128_store(rgb_out.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 16).cast(), out1);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 32).cast(), out2);

      x += 16;
    }
    if x < width {
      scalar::x2bgr10_to_rgb_row(
        &x2bgr10[x * 4..width * 4],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// WASM simd128 X2BGR10→RGBA. 4 pixels per iteration (single vector
/// holds 4 RGBA pixels).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn x2bgr10_to_rgba_row(x2bgr10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mask_3ff = u32x4_splat(0x3FF);
    let alpha_const = u32x4_splat(0xFF00_0000);
    let mut x = 0usize;
    while x + 4 <= width {
      let pix = v128_load(x2bgr10.as_ptr().add(x * 4).cast());

      // X2BGR10 channel positions: R at low, G mid, B high.
      let r = u32x4_shr(v128_and(pix, mask_3ff), 2);
      let g = u32x4_shr(v128_and(u32x4_shr(pix, 10), mask_3ff), 2);
      let b = u32x4_shr(v128_and(u32x4_shr(pix, 20), mask_3ff), 2);

      let r_mask = i8x16(0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);
      let g_mask = i8x16(-1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1);
      let b_mask = i8x16(-1, -1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1);
      let out = v128_or(
        v128_or(
          v128_or(u8x16_swizzle(r, r_mask), u8x16_swizzle(g, g_mask)),
          u8x16_swizzle(b, b_mask),
        ),
        alpha_const,
      );

      v128_store(rgba_out.as_mut_ptr().add(x * 4).cast(), out);
      x += 4;
    }
    if x < width {
      scalar::x2bgr10_to_rgba_row(
        &x2bgr10[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// WASM simd128 X2BGR10→u16 RGB native. 8 pixels per iteration.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn x2bgr10_to_rgb_u16_row(x2bgr10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mask_3ff = u32x4_splat(0x3FF);
    let mut x = 0usize;
    while x + 8 <= width {
      let p0 = v128_load(x2bgr10.as_ptr().add(x * 4).cast());
      let p1 = v128_load(x2bgr10.as_ptr().add(x * 4 + 16).cast());

      let r0 = v128_and(p0, mask_3ff);
      let r1 = v128_and(p1, mask_3ff);
      let g0 = v128_and(u32x4_shr(p0, 10), mask_3ff);
      let g1 = v128_and(u32x4_shr(p1, 10), mask_3ff);
      let b0 = v128_and(u32x4_shr(p0, 20), mask_3ff);
      let b1 = v128_and(u32x4_shr(p1, 20), mask_3ff);

      let r = u16x8_narrow_i32x4(r0, r1);
      let g = u16x8_narrow_i32x4(g0, g1);
      let b = u16x8_narrow_i32x4(b0, b1);

      let r_mask0 = i8x16(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
      let g_mask0 = i8x16(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
      let b_mask0 = i8x16(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
      let out0 = v128_or(
        v128_or(u8x16_swizzle(r, r_mask0), u8x16_swizzle(g, g_mask0)),
        u8x16_swizzle(b, b_mask0),
      );
      // Block 1 (output u16s 8..15 = [B2, R3, G3, B3, R4, G4, B4, R5]).
      // Each u16 takes 2 bytes; the channel vectors hold element `i` at
      // byte indices `(2*i, 2*i+1)`.
      let r_mask1 = i8x16(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
      let g_mask1 = i8x16(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
      let b_mask1 = i8x16(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
      let out1 = v128_or(
        v128_or(u8x16_swizzle(r, r_mask1), u8x16_swizzle(g, g_mask1)),
        u8x16_swizzle(b, b_mask1),
      );
      // Block 2 (output u16s 16..23 = [G5, B5, R6, G6, B6, R7, G7, B7]).
      let r_mask2 = i8x16(
        -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1,
      );
      let g_mask2 = i8x16(
        10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1,
      );
      let b_mask2 = i8x16(
        -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15,
      );
      let out2 = v128_or(
        v128_or(u8x16_swizzle(r, r_mask2), u8x16_swizzle(g, g_mask2)),
        u8x16_swizzle(b, b_mask2),
      );

      v128_store(rgb_out.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 8).cast(), out1);
      v128_store(rgb_out.as_mut_ptr().add(x * 3 + 16).cast(), out2);

      x += 8;
    }
    if x < width {
      scalar::x2bgr10_to_rgb_u16_row(
        &x2bgr10[x * 4..width * 4],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}
