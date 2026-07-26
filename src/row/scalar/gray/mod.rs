//! Scalar gray → {RGB, RGBA, HSV, luma, luma_u16} kernels.
//!
//! # Gray8 (u8 source)
//!
//! `gray8_to_rgb_row`  — broadcast Y to each R/G/B channel.
//! `gray8_to_rgba_row` — broadcast Y + force α = 0xFF.
//! `gray8_to_hsv_row`  — H=0, S=0, V=Y (S=0 convention: H is fixed to 0).
//! `gray8_to_luma_u16_row` — zero-extend Y to u16 (`out[x] = y[x] as u16`).
//!   (luma u8 identity copy is handled by the sinker directly via
//!   `copy_from_slice`, no dedicated kernel needed.)
//!
//! # GrayN / Gray16 (u16 source)
//!
//! `gray_n_to_rgb_row<BITS>`       — mask → downshift to u8, broadcast.
//! `gray_n_to_rgba_row<BITS>`      — same + α = 0xFF.
//! `gray_n_to_rgb_u16_row<BITS>`   — mask, keep native depth, broadcast.
//! `gray_n_to_rgba_u16_row<BITS>`  — same + α = full-range max for BITS.
//! `gray_n_to_luma_row<BITS>`      — mask → downshift to u8.
//! `gray_n_to_luma_u16_row<BITS>`  — mask → identity (samples already u16).
//! `gray_n_to_hsv_row<BITS>`       — mask → downshift to u8 → H=0 S=0 V=Y8.
//!
//! `gray16_to_rgb_row`    — `>> 8` to u8, broadcast.
//! `gray16_to_rgba_row`   — same + α = 0xFF.
//! `gray16_to_rgb_u16_row`  — native u16, broadcast.
//! `gray16_to_rgba_u16_row` — native u16 + α = 0xFFFF.
//! `gray16_to_luma_row`   — `>> 8` to u8.
//! `gray16_to_luma_u16_row` — identity copy to u16.
//! `gray16_to_hsv_row`    — `>> 8` to u8 → H=0 S=0 V=Y8.
//!
//! # HSV S=0 convention
//!
//! When S=0 (which is always for gray sources — delta = 0), H is set to 0.
//! This matches OpenCV `cv2.COLOR_GRAY2HSV` behavior.
//!
//! # `full_range` parameter (RGB/RGBA/HSV outputs only)
//!
//! - `full_range = true`: raw Y is used directly (broadcast / downshift).
//!   This is the standard full-range path.
//! - `full_range = false`: limited-range Y is rescaled to full range before
//!   broadcast. For 8-bit limited range: black=16, white=235. The rescaled
//!   value is `clamp_u8(((y - black) * 255 + range/2) / range)`.
//!
//! **Luma outputs** (`with_luma`, `with_luma_u16`) always pass Y through
//! without rescaling — the caller is explicitly requesting the source luma
//! plane as-is, regardless of `full_range`.

use super::bits_mask;

// ---- helpers ----------------------------------------------------------------

/// Broadcasts a `u8` gray value to packed RGB (3 bytes: R=G=B=y).
#[inline(always)]
fn broadcast_u8_to_rgb(y: u8, out: &mut [u8], x: usize) {
  let i = x * 3;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
}

/// Broadcasts a `u8` gray value to packed RGBA (4 bytes: R=G=B=y, A=0xFF).
#[inline(always)]
fn broadcast_u8_to_rgba(y: u8, out: &mut [u8], x: usize) {
  let i = x * 4;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
  out[i + 3] = 0xFF;
}

/// Broadcasts a `u16` gray value to packed u16 RGB (3 u16: R=G=B=y).
#[inline(always)]
fn broadcast_u16_to_rgb(y: u16, out: &mut [u16], x: usize) {
  let i = x * 3;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
}

/// Broadcasts a `u16` gray value to packed u16 RGBA (4 u16: R=G=B=y, A=alpha).
#[inline(always)]
fn broadcast_u16_to_rgba(y: u16, alpha: u16, out: &mut [u16], x: usize) {
  let i = x * 4;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
  out[i + 3] = alpha;
}

/// Rescales a limited-range 8-bit luma value to full-range u8.
///
/// Limited-range 8-bit: black=16, white=235 (range=219).
/// Formula: `clamp_u8(((y - 16) * 255 + 109) / 219)` (109 = 219/2, rounding).
#[inline(always)]
fn limited_to_full_u8(y: u8) -> u8 {
  let y = y as i32;
  let rescaled = (y - 16) * 255 + 109; // 109 = 219/2 for rounding
  let result = rescaled / 219;
  result.clamp(0, 255) as u8
}

/// Rescales a limited-range N-bit luma value to full-range u8.
///
/// Limited-range: black = 16 << (BITS-8), range = 219 << (BITS-8).
#[inline(always)]
fn limited_n_to_full_u8<const BITS: u32>(y: u16) -> u8 {
  let black = 16i32 << (BITS - 8);
  let range = 219i32 << (BITS - 8);
  let y = y as i32;
  let rescaled = (y - black) * 255 + range / 2;
  let result = rescaled / range;
  result.clamp(0, 255) as u8
}

/// Rescales a limited-range N-bit luma value to full-range u16 (native depth).
///
/// Limited-range: black = 16 << (BITS-8), range = 219 << (BITS-8).
/// Output is clamped to [0, max_native] where max_native = (1 << BITS) - 1.
///
/// Math runs in `i64` to keep the `(y - black) * max_native` product safe at
/// `BITS = 16`: limited-range white `60160` minus black `4096` times
/// `max_native = 65535` is `~3.67 x 10^9`, which overflows `i32`. Lower bit
/// depths fit in `i32` but using `i64` uniformly keeps one signature and
/// avoids per-BITS branches.
#[inline(always)]
fn limited_n_to_full_u16<const BITS: u32>(y: u16) -> u16 {
  let black = 16i64 << (BITS - 8);
  let range = 219i64 << (BITS - 8);
  let max_native = ((1u64 << BITS) - 1) as i64;
  let y = y as i64;
  let rescaled = (y - black) * max_native + range / 2;
  let result = rescaled / range;
  result.clamp(0, max_native) as u16
}

/// Rescales a limited-range 16-bit luma value to full-range u8.
///
/// Limited-range 16-bit: black = 16 << 8 = 4096, range = 219 << 8 = 56064.
#[inline(always)]
fn limited_16_to_full_u8(y: u16) -> u8 {
  limited_n_to_full_u8::<16>(y)
}

/// Rescales a limited-range 16-bit luma value to full-range u16.
///
/// Limited-range 16-bit: black = 4096, range = 56064, max_native = 65535.
#[inline(always)]
fn limited_16_to_full_u16(y: u16) -> u16 {
  limited_n_to_full_u16::<16>(y)
}

/// Rescales a limited-range 32-bit luma value to full-range u8.
///
/// Limited-range 32-bit: black = `16 << 24`, range = `219 << 24`. The
/// `(y - black) * 255` product reaches `~9.4 x 10^11` (well past `i32`), so
/// the math runs in `i64`.
#[inline(always)]
fn limited_32_to_full_u8(y: u32) -> u8 {
  let black = 16i64 << 24;
  let range = 219i64 << 24;
  let y = y as i64;
  let rescaled = (y - black) * 255 + range / 2;
  let result = rescaled / range;
  result.clamp(0, 255) as u8
}

/// Rescales a limited-range 32-bit luma value to full-range u16 (native
/// output depth — `Gray32`'s widest broadcast is `u16`).
///
/// Limited-range 32-bit: black = `16 << 24`, range = `219 << 24`,
/// `max_native = 65535`. The `(y - black) * max_native` product reaches
/// `~2.4 x 10^14`, far past `i32`, so the math runs in `i64`.
#[inline(always)]
fn limited_32_to_full_u16(y: u32) -> u16 {
  let black = 16i64 << 24;
  let range = 219i64 << 24;
  let max_native = 65535i64;
  let y = y as i64;
  let rescaled = (y - black) * max_native + range / 2;
  let result = rescaled / range;
  result.clamp(0, max_native) as u16
}

// ---- Gray8 ------------------------------------------------------------------

/// Broadcasts each `u8` gray sample to packed RGB (`R = G = B = Y`).
///
/// When `full_range = false`, limited-range Y (black=16, white=235) is
/// rescaled to full-range [0, 255] before broadcast.
/// When `full_range = true`, raw Y is used directly.
///
/// Luma outputs always pass Y through without rescaling.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_rgb_row(y_plane: &[u8], out: &mut [u8], width: usize, full_range: bool) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    let y_out = if full_range { y } else { limited_to_full_u8(y) };
    broadcast_u8_to_rgb(y_out, out, x);
  }
}

/// Broadcasts each `u8` gray sample to packed RGBA (`R = G = B = Y`, `A = 0xFF`).
///
/// When `full_range = false`, limited-range Y is rescaled to full range.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_rgba_row(y_plane: &[u8], out: &mut [u8], width: usize, full_range: bool) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    let y_out = if full_range { y } else { limited_to_full_u8(y) };
    broadcast_u8_to_rgba(y_out, out, x);
  }
}

/// Gray8 → HSV row. Convention: S=0, H=0, V=Y (or rescaled Y if limited range).
///
/// Gray sources are achromatic (saturation = 0). When S=0, H is
/// undefined in the continuous HSV model; this crate fixes H=0 to
/// match OpenCV's `cv2.COLOR_GRAY2HSV` convention and avoid
/// non-deterministic hue output.
///
/// When `full_range = false`, the V channel uses the rescaled luma value.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_hsv_row(
  y_plane: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = if full_range { y } else { limited_to_full_u8(y) };
  }
}

// ---- GrayN (u16 low-bit-packed, BITS in {9,10,12,14}) ----------------------

/// GrayN → packed RGB u8. Masks to BITS bits, downshifts `BITS - 8` to u8,
/// broadcasts.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 255]
/// before broadcast. Luma outputs always pass Y through without rescaling.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    let y8 = if full_range {
      (masked >> shift) as u8
    } else {
      limited_n_to_full_u8::<BITS>(masked)
    };
    broadcast_u8_to_rgb(y8, out, x);
  }
}

/// GrayN → packed RGBA u8. Masks to BITS bits, downshifts to u8, broadcasts,
/// α = 0xFF.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 255].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    let y8 = if full_range {
      (masked >> shift) as u8
    } else {
      limited_n_to_full_u8::<BITS>(masked)
    };
    broadcast_u8_to_rgba(y8, out, x);
  }
}

/// GrayN → packed u16 RGB. Masks to BITS bits, broadcasts at native depth.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to full native range
/// [0, (1<<BITS)-1] before broadcast.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  let mask = bits_mask::<BITS>();
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    let y_out = if full_range {
      masked
    } else {
      limited_n_to_full_u16::<BITS>(masked)
    };
    broadcast_u16_to_rgb(y_out, out, x);
  }
}

/// GrayN → packed u16 RGBA. Masks to BITS bits, broadcasts, α = `(1 << BITS) - 1`.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to full native range.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  let mask = bits_mask::<BITS>();
  let alpha = mask; // full-range max for BITS
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    let y_out = if full_range {
      masked
    } else {
      limited_n_to_full_u16::<BITS>(masked)
    };
    broadcast_u16_to_rgba(y_out, alpha, out, x);
  }
}

/// GrayN → luma u8. Masks to BITS bits, downshifts `BITS - 8`.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_luma_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (out_byte, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    *out_byte = ((raw & mask) >> shift) as u8;
  }
}

/// GrayN → luma u16. Masks to BITS bits, identity copy.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_luma_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  let mask = bits_mask::<BITS>();
  for (out_el, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    *out_el = raw & mask;
  }
}

/// GrayN → HSV u8. Masks to BITS bits, downshifts to u8, H=0 S=0 V=Y8.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, the V channel uses the rescaled luma value.
/// See [`gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_hsv_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = if full_range {
      (masked >> shift) as u8
    } else {
      limited_n_to_full_u8::<BITS>(masked)
    };
  }
}

// ---- Gray16 (u16, all 16 bits active) ----------------------------------------

/// Gray16 → packed RGB u8. Downshifts `>> 8` to u8, broadcasts.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y (black=4096, white=56064+4096)
/// is rescaled to [0, 255] before broadcast.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgb_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let y8 = if full_range {
      (raw >> 8) as u8
    } else {
      limited_16_to_full_u8(raw)
    };
    broadcast_u8_to_rgb(y8, out, x);
  }
}

/// Gray16 → packed RGBA u8. Downshifts `>> 8`, broadcasts, α = 0xFF.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 255].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgba_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let y8 = if full_range {
      (raw >> 8) as u8
    } else {
      limited_16_to_full_u8(raw)
    };
    broadcast_u8_to_rgba(y8, out, x);
  }
}

/// Gray16 → packed u16 RGB. Identity broadcast, native 16-bit depth.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 65535].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgb_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let y_out = if full_range {
      raw
    } else {
      limited_16_to_full_u16(raw)
    };
    broadcast_u16_to_rgb(y_out, out, x);
  }
}

/// Gray16 → packed u16 RGBA. Identity broadcast, α = 0xFFFF.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 65535].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgba_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let y_out = if full_range {
      raw
    } else {
      limited_16_to_full_u16(raw)
    };
    broadcast_u16_to_rgba(y_out, 0xFFFF, out, x);
  }
}

/// Gray16 → luma u8. Downshifts `>> 8`.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_luma_row<const BE: bool>(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for (out_byte, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    *out_byte = (raw >> 8) as u8;
  }
}

/// Gray16 → luma u16. Identity copy (or byte-swap copy for BE).
///
/// When `BE = true`, each u16 sample is byte-swapped before output.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_luma_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for (o, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    *o = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
  }
}

/// Gray16 → HSV u8. `>> 8` to u8, H=0 S=0 V=Y8.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, the V channel uses the rescaled luma value.
/// See [`gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_hsv_row<const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = if full_range {
      (raw >> 8) as u8
    } else {
      limited_16_to_full_u8(raw)
    };
  }
}

// ---- Gray32 (u32, all 32 bits active) ---------------------------------------
//
// Full-bit integer twin of Gray16, widened u16 → u32. The widest output
// broadcast pixon emits is u16, so the depth narrows are `>> 24` (u8) and
// `>> 16` (native u16); luma_u16 / native carry the `>> 16` sample. The
// limited-range rescale operates on the raw u32 sample (black = 16 << 24,
// range = 219 << 24) in i64.

/// Gray32 → packed RGB u8. Downshifts `>> 24` to u8, broadcasts.
///
/// When `BE = true`, each u32 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y (black = `16 << 24`) is
/// rescaled to [0, 255] before broadcast.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_rgb_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    let y8 = if full_range {
      (raw >> 24) as u8
    } else {
      limited_32_to_full_u8(raw)
    };
    broadcast_u8_to_rgb(y8, out, x);
  }
}

/// Gray32 → packed RGBA u8. Downshifts `>> 24`, broadcasts, α = 0xFF.
///
/// When `BE = true`, each u32 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 255].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_rgba_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    let y8 = if full_range {
      (raw >> 24) as u8
    } else {
      limited_32_to_full_u8(raw)
    };
    broadcast_u8_to_rgba(y8, out, x);
  }
}

/// Gray32 → packed u16 RGB. Downshifts `>> 16` to native u16, broadcasts.
///
/// When `BE = true`, each u32 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 65535].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_rgb_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    let y_out = if full_range {
      (raw >> 16) as u16
    } else {
      limited_32_to_full_u16(raw)
    };
    broadcast_u16_to_rgb(y_out, out, x);
  }
}

/// Gray32 → packed u16 RGBA. Downshifts `>> 16`, broadcasts, α = 0xFFFF.
///
/// When `BE = true`, each u32 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 65535].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_rgba_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    let y_out = if full_range {
      (raw >> 16) as u16
    } else {
      limited_32_to_full_u16(raw)
    };
    broadcast_u16_to_rgba(y_out, 0xFFFF, out, x);
  }
}

/// Gray32 → luma u8. Downshifts `>> 24`.
///
/// When `BE = true`, each u32 sample is byte-swapped before processing.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_luma_row<const BE: bool>(y_plane: &[u32], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for (out_byte, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    let raw = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    *out_byte = (raw >> 24) as u8;
  }
}

/// Gray32 → luma u16. Downshifts `>> 16` to native u16.
///
/// When `BE = true`, each u32 sample is byte-swapped before processing.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_luma_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for (o, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    let raw = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    *o = (raw >> 16) as u16;
  }
}

/// Gray32 → host-native `u32` value (the wire `BE` swap only, **no depth
/// narrow**). Stages the source plane for the native-`u32` resample tier so
/// binning runs at full `u32` precision and each output narrows only after
/// binning (0-ULP, closes issue #289).
///
/// When `BE = true`, each wire `u32` is byte-swapped to the host value;
/// `BE = false` reads it host-native (swapping only on a big-endian host).
/// The emitted samples derive through the `gray32_to_*::<HOST_NATIVE_BE>`
/// kernels, whose swap is then the identity recovery for an already-native
/// value. Scalar-only (the `u32` resample tier ships no SIMD).
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_native_u32_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u32],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for (o, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    *o = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
  }
}

/// Gray32 → HSV u8. `>> 24` to u8, H=0 S=0 V=Y8.
///
/// When `BE = true`, each u32 sample is byte-swapped before processing.
/// When `full_range = false`, the V channel uses the rescaled luma value.
/// See [`gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_hsv_row<const BE: bool>(
  y_plane: &[u32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u32::from_be(raw)
    } else {
      u32::from_le(raw)
    };
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = if full_range {
      (raw >> 24) as u8
    } else {
      limited_32_to_full_u8(raw)
    };
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
