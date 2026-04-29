use core::arch::aarch64::*;

use crate::{ColorMatrix, row::scalar};

use super::*;

// ===== BGR ↔ RGB byte swap ==============================================

/// Swaps the outer two channels of each packed 3‑byte triple. Drives
/// both `bgr_to_rgb_row` and `rgb_to_bgr_row` since the transformation
/// is self‑inverse.
///
/// NEON makes this almost free: `vld3q_u8` deinterleaves 16 pixels into
/// three channel vectors `(ch0, ch1, ch2)`, and `vst3q_u8` re‑interleaves
/// them — passing the deinterleaved vectors back in reversed order
/// `(ch2, ch1, ch0)` swaps the outer channels in a single store.
///
/// # Safety
///
/// 1. NEON must be available (same obligation as the other NEON kernels).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section. All pointer adds are bounded by the
  // `while x + 16 <= width` condition and the caller‑promised
  // slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let triple = vld3q_u8(input.as_ptr().add(x * 3));
      let swapped = uint8x16x3_t(triple.2, triple.1, triple.0);
      vst3q_u8(output.as_mut_ptr().add(x * 3), swapped);
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

/// Drops the alpha byte from packed `R, G, B, A` input, producing
/// packed `R, G, B` output. NEON makes this nearly free: `vld4q_u8`
/// deinterleaves 16 RGBA pixels into four u8x16 channel vectors
/// `(R, G, B, A)`, and `vst3q_u8` re-interleaves three of them as
/// RGB triples — the alpha vector is simply discarded.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation, same as the other
///    NEON kernels).
/// 2. `rgba.len() >= 4 * width`.
/// 3. `rgb_out.len() >= 3 * width`.
/// 4. `rgba` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgba_to_rgb_row(rgba: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgba.len() >= width * 4, "rgba row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // SAFETY: NEON availability is the caller's obligation. All pointer
  // adds are bounded by the `while x + 16 <= width` condition and the
  // caller-promised slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(rgba.as_ptr().add(x * 4));
      let triple = uint8x16x3_t(quad.0, quad.1, quad.2);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), triple);
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

/// Swaps R↔B in packed `B, G, R, A` input, producing packed
/// `R, G, B, A` (alpha lane preserved). `vld4q_u8` deinterleaves
/// 16 BGRA pixels into channel vectors `(B, G, R, A)`; `vst4q_u8`
/// interleaves them back as `(R, G, B, A)` simply by reordering
/// the channel-vector tuple.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgra.len() >= 4 * width`.
/// 3. `rgba_out.len() >= 4 * width`.
/// 4. `bgra` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgra_to_rgba_row(bgra: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(bgra.as_ptr().add(x * 4));
      let swapped = uint8x16x4_t(quad.2, quad.1, quad.0, quad.3);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), swapped);
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

/// Swaps R↔B and drops alpha from packed `B, G, R, A` input,
/// producing packed `R, G, B`. Combines [`rgba_to_rgb_row`]'s
/// alpha drop with [`bgra_to_rgba_row`]'s channel swap in one pass:
/// `vld4q_u8` → reorder vectors → `vst3q_u8` (alpha discarded).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgra.len() >= 4 * width`.
/// 3. `rgb_out.len() >= 3 * width`.
/// 4. `bgra` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgra_to_rgb_row(bgra: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(bgra.as_ptr().add(x * 4));
      let triple = uint8x16x3_t(quad.2, quad.1, quad.0);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), triple);
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

/// Drops the leading alpha byte from packed `A, R, G, B` input,
/// producing packed `R, G, B`. `vld4q_u8` deinterleaves 16 ARGB
/// pixels into channel vectors `(A, R, G, B)`; `vst3q_u8` interleaves
/// channels 1..3 (R, G, B) — the alpha vector is discarded.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `argb.len() >= 4 * width`.
/// 3. `rgb_out.len() >= 3 * width`.
/// 4. `argb` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn argb_to_rgb_row(argb: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(argb.as_ptr().add(x * 4));
      let triple = uint8x16x3_t(quad.1, quad.2, quad.3);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), triple);
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

/// Swaps R↔B and drops leading alpha from packed `A, B, G, R`
/// input, producing packed `R, G, B`. `vld4q_u8` deinterleaves into
/// `(A, B, G, R)` channel vectors; `vst3q_u8` interleaves
/// `(channel 3, channel 2, channel 1)` = `(R, G, B)`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `abgr.len() >= 4 * width`.
/// 3. `rgb_out.len() >= 3 * width`.
/// 4. `abgr` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn abgr_to_rgb_row(abgr: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(abgr.as_ptr().add(x * 4));
      let triple = uint8x16x3_t(quad.3, quad.2, quad.1);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), triple);
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

/// Rotates leading alpha to trailing position in packed `A, R, G, B`
/// input, producing packed `R, G, B, A`. `vld4q_u8` deinterleaves
/// into `(A, R, G, B)` channel vectors; `vst4q_u8` interleaves
/// `(R, G, B, A)` = `(channel 1, 2, 3, 0)`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `argb.len() >= 4 * width`.
/// 3. `rgba_out.len() >= 4 * width`.
/// 4. `argb` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn argb_to_rgba_row(argb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(argb.as_ptr().add(x * 4));
      let rotated = uint8x16x4_t(quad.1, quad.2, quad.3, quad.0);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), rotated);
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

/// Reverses byte order in packed `A, B, G, R` input, producing
/// packed `R, G, B, A`. `vld4q_u8` deinterleaves into `(A, B, G, R)`
/// channel vectors; `vst4q_u8` interleaves them in reverse
/// `(R, G, B, A)` = `(channel 3, 2, 1, 0)`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `abgr.len() >= 4 * width`.
/// 3. `rgba_out.len() >= 4 * width`.
/// 4. `abgr` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn abgr_to_rgba_row(abgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(abgr.as_ptr().add(x * 4));
      let reversed = uint8x16x4_t(quad.3, quad.2, quad.1, quad.0);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), reversed);
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

/// Drops the leading padding byte from packed `X, R, G, B` input,
/// producing packed `R, G, B, A` with `A = 0xFF`. `vld4q_u8`
/// deinterleaves into channel vectors `(X, R, G, B)`; we reinterleave
/// `(R, G, B, splat(0xFF))` via `vst4q_u8`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `xrgb.len() >= 4 * width`.
/// 3. `rgba_out.len() >= 4 * width`.
/// 4. `xrgb` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xrgb_to_rgba_row(xrgb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xrgb.len() >= width * 4, "xrgb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdupq_n_u8(0xFF);
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(xrgb.as_ptr().add(x * 4));
      let out = uint8x16x4_t(quad.1, quad.2, quad.3, alpha);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), out);
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

/// Drops the trailing padding byte from packed `R, G, B, X` input,
/// producing packed `R, G, B, A` with `A = 0xFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgbx.len() >= 4 * width`.
/// 3. `rgba_out.len() >= 4 * width`.
/// 4. `rgbx` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgbx_to_rgba_row(rgbx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgbx.len() >= width * 4, "rgbx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdupq_n_u8(0xFF);
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(rgbx.as_ptr().add(x * 4));
      let out = uint8x16x4_t(quad.0, quad.1, quad.2, alpha);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), out);
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

/// Reverses RGB and drops leading padding from packed `X, B, G, R`
/// input, producing packed `R, G, B, A` with `A = 0xFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `xbgr.len() >= 4 * width`.
/// 3. `rgba_out.len() >= 4 * width`.
/// 4. `xbgr` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xbgr_to_rgba_row(xbgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xbgr.len() >= width * 4, "xbgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdupq_n_u8(0xFF);
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(xbgr.as_ptr().add(x * 4));
      let out = uint8x16x4_t(quad.3, quad.2, quad.1, alpha);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), out);
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

/// Reverses RGB and drops trailing padding from packed `B, G, R, X`
/// input, producing packed `R, G, B, A` with `A = 0xFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgrx.len() >= 4 * width`.
/// 3. `rgba_out.len() >= 4 * width`.
/// 4. `bgrx` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgrx_to_rgba_row(bgrx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgrx.len() >= width * 4, "bgrx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdupq_n_u8(0xFF);
    let mut x = 0usize;
    while x + 16 <= width {
      let quad = vld4q_u8(bgrx.as_ptr().add(x * 4));
      let out = uint8x16x4_t(quad.2, quad.1, quad.0, alpha);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), out);
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

/// LE-explicit `u32x4` load from a packed-X2*10 byte stream.
///
/// `vld1q_u32` interprets the 16 source bytes as `u32` lanes in
/// host-endian order. The X2RGB10 / X2BGR10 source contract is
/// documented as **little-endian** (matching the scalar's
/// `u32::from_le_bytes`), so on big-endian aarch64 (rare —
/// `aarch64_be-*` custom targets) the host-endian load would put
/// the bytes in reversed positions within each lane, corrupting
/// every subsequent shift-and-mask. The `vrev32q_u8` on the
/// big-endian branch byte-reverses each `u32` lane back to the
/// LE byte ordering. On every standard aarch64 target
/// (LE) the `cfg!` evaluates to `false` at compile time and the
/// load reduces to a plain `vld1q_u32`.
#[inline(always)]
unsafe fn x2_load_le_u32x4(ptr: *const u8) -> uint32x4_t {
  unsafe {
    let raw = vld1q_u32(ptr as *const u32);
    if cfg!(target_endian = "big") {
      vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(raw)))
    } else {
      raw
    }
  }
}

/// Extracts a 10-bit channel as a `u8` (top 8 bits) from each of 4
/// `u32` pixels in a `uint32x4_t`. Returns 4 `u16` lanes packed in a
/// `uint16x4_t`. The dropped low 2 bits of each 10-bit value match
/// the scalar `>> 2` truncation.
#[inline(always)]
unsafe fn x2_extract_10bit_u8_lane(pix: uint32x4_t, shift: i32) -> uint16x4_t {
  unsafe {
    // Shift down then narrow + saturate. Values are bounded to
    // `[0, 255]` so saturation never triggers.
    let shifted = match shift {
      22 => vshrq_n_u32(pix, 22),
      12 => vshrq_n_u32(pix, 12),
      2 => vshrq_n_u32(pix, 2),
      _ => unreachable!(),
    };
    let mask = vdupq_n_u32(0xFF);
    vqmovn_u32(vandq_u32(shifted, mask))
  }
}

/// Extracts a 10-bit channel into the low 10 bits of a `u16`. 4
/// pixels in → 4 `u16` lanes out.
#[inline(always)]
unsafe fn x2_extract_10bit_u16_lane(pix: uint32x4_t, shift: i32) -> uint16x4_t {
  unsafe {
    let shifted = match shift {
      20 => vshrq_n_u32(pix, 20),
      10 => vshrq_n_u32(pix, 10),
      0 => pix,
      _ => unreachable!(),
    };
    let mask = vdupq_n_u32(0x3FF);
    vqmovn_u32(vandq_u32(shifted, mask))
  }
}

/// NEON X2RGB10→RGB. 16 pixels per iteration: load 4 `u32x4`
/// vectors, extract R/G/B channels, narrow to `u8x16` per channel,
/// `vst3q_u8`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `x2rgb10.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `x2rgb10` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn x2rgb10_to_rgb_row(x2rgb10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4));
      let p1 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4 + 16));
      let p2 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4 + 32));
      let p3 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4 + 48));

      // X2RGB10: R at >>22, G at >>12, B at >>2 (top 8 of 10-bit).
      let r_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 22),
        x2_extract_10bit_u8_lane(p1, 22),
      );
      let r_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 22),
        x2_extract_10bit_u8_lane(p3, 22),
      );
      let g_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 12),
        x2_extract_10bit_u8_lane(p1, 12),
      );
      let g_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 12),
        x2_extract_10bit_u8_lane(p3, 12),
      );
      let b_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 2),
        x2_extract_10bit_u8_lane(p1, 2),
      );
      let b_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 2),
        x2_extract_10bit_u8_lane(p3, 2),
      );

      let r = vcombine_u8(vqmovn_u16(r_lo), vqmovn_u16(r_hi));
      let g = vcombine_u8(vqmovn_u16(g_lo), vqmovn_u16(g_hi));
      let b = vcombine_u8(vqmovn_u16(b_lo), vqmovn_u16(b_hi));

      let rgb = uint8x16x3_t(r, g, b);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), rgb);

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

/// NEON X2RGB10→RGBA. 16 pixels per iteration; alpha forced to
/// `0xFF` via `vdupq_n_u8`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `x2rgb10.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `x2rgb10` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn x2rgb10_to_rgba_row(x2rgb10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdupq_n_u8(0xFF);
    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4));
      let p1 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4 + 16));
      let p2 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4 + 32));
      let p3 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4 + 48));

      let r_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 22),
        x2_extract_10bit_u8_lane(p1, 22),
      );
      let r_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 22),
        x2_extract_10bit_u8_lane(p3, 22),
      );
      let g_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 12),
        x2_extract_10bit_u8_lane(p1, 12),
      );
      let g_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 12),
        x2_extract_10bit_u8_lane(p3, 12),
      );
      let b_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 2),
        x2_extract_10bit_u8_lane(p1, 2),
      );
      let b_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 2),
        x2_extract_10bit_u8_lane(p3, 2),
      );

      let r = vcombine_u8(vqmovn_u16(r_lo), vqmovn_u16(r_hi));
      let g = vcombine_u8(vqmovn_u16(g_lo), vqmovn_u16(g_hi));
      let b = vcombine_u8(vqmovn_u16(b_lo), vqmovn_u16(b_hi));

      let rgba = uint8x16x4_t(r, g, b, alpha);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), rgba);

      x += 16;
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

/// NEON X2RGB10→u16 RGB native (10-bit, low-bit aligned). 8 pixels
/// per iteration.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `x2rgb10.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `x2rgb10` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn x2rgb10_to_rgb_u16_row(x2rgb10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p0 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4));
      let p1 = x2_load_le_u32x4(x2rgb10.as_ptr().add(x * 4 + 16));

      // Channel low bit positions: R at 20, G at 10, B at 0.
      let r = vcombine_u16(
        x2_extract_10bit_u16_lane(p0, 20),
        x2_extract_10bit_u16_lane(p1, 20),
      );
      let g = vcombine_u16(
        x2_extract_10bit_u16_lane(p0, 10),
        x2_extract_10bit_u16_lane(p1, 10),
      );
      let b = vcombine_u16(
        x2_extract_10bit_u16_lane(p0, 0),
        x2_extract_10bit_u16_lane(p1, 0),
      );

      let rgb = uint16x8x3_t(r, g, b);
      vst3q_u16(rgb_out.as_mut_ptr().add(x * 3), rgb);

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

/// NEON X2BGR10→RGB. 16 pixels per iteration; same shape as
/// [`x2rgb10_to_rgb_row`] but channel shifts swapped (R at >>2,
/// B at >>22).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn x2bgr10_to_rgb_row(x2bgr10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4));
      let p1 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4 + 16));
      let p2 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4 + 32));
      let p3 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4 + 48));

      let r_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 2),
        x2_extract_10bit_u8_lane(p1, 2),
      );
      let r_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 2),
        x2_extract_10bit_u8_lane(p3, 2),
      );
      let g_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 12),
        x2_extract_10bit_u8_lane(p1, 12),
      );
      let g_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 12),
        x2_extract_10bit_u8_lane(p3, 12),
      );
      let b_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 22),
        x2_extract_10bit_u8_lane(p1, 22),
      );
      let b_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 22),
        x2_extract_10bit_u8_lane(p3, 22),
      );

      let r = vcombine_u8(vqmovn_u16(r_lo), vqmovn_u16(r_hi));
      let g = vcombine_u8(vqmovn_u16(g_lo), vqmovn_u16(g_hi));
      let b = vcombine_u8(vqmovn_u16(b_lo), vqmovn_u16(b_hi));

      let rgb = uint8x16x3_t(r, g, b);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), rgb);

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

/// NEON X2BGR10→RGBA. 16 pixels per iteration.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn x2bgr10_to_rgba_row(x2bgr10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdupq_n_u8(0xFF);
    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4));
      let p1 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4 + 16));
      let p2 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4 + 32));
      let p3 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4 + 48));

      let r_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 2),
        x2_extract_10bit_u8_lane(p1, 2),
      );
      let r_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 2),
        x2_extract_10bit_u8_lane(p3, 2),
      );
      let g_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 12),
        x2_extract_10bit_u8_lane(p1, 12),
      );
      let g_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 12),
        x2_extract_10bit_u8_lane(p3, 12),
      );
      let b_lo = vcombine_u16(
        x2_extract_10bit_u8_lane(p0, 22),
        x2_extract_10bit_u8_lane(p1, 22),
      );
      let b_hi = vcombine_u16(
        x2_extract_10bit_u8_lane(p2, 22),
        x2_extract_10bit_u8_lane(p3, 22),
      );

      let r = vcombine_u8(vqmovn_u16(r_lo), vqmovn_u16(r_hi));
      let g = vcombine_u8(vqmovn_u16(g_lo), vqmovn_u16(g_hi));
      let b = vcombine_u8(vqmovn_u16(b_lo), vqmovn_u16(b_hi));

      let rgba = uint8x16x4_t(r, g, b, alpha);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), rgba);

      x += 16;
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

/// NEON X2BGR10→u16 RGB native. 8 pixels per iteration.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn x2bgr10_to_rgb_u16_row(x2bgr10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p0 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4));
      let p1 = x2_load_le_u32x4(x2bgr10.as_ptr().add(x * 4 + 16));

      // X2BGR10: R at low 10 bits, G at 10..19, B at 20..29.
      let r = vcombine_u16(
        x2_extract_10bit_u16_lane(p0, 0),
        x2_extract_10bit_u16_lane(p1, 0),
      );
      let g = vcombine_u16(
        x2_extract_10bit_u16_lane(p0, 10),
        x2_extract_10bit_u16_lane(p1, 10),
      );
      let b = vcombine_u16(
        x2_extract_10bit_u16_lane(p0, 20),
        x2_extract_10bit_u16_lane(p1, 20),
      );

      let rgb = uint16x8x3_t(r, g, b);
      vst3q_u16(rgb_out.as_mut_ptr().add(x * 3), rgb);

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
