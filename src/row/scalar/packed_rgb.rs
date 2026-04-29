// ---- Tier 6 packed-RGBA helpers (Ship 9b) ------------------------------
//
// Compact byte-rearrangement kernels behind the [`Rgba`] / [`Bgra`]
// source-side sinker family. This file provides the scalar reference /
// fallback implementations; SIMD dispatch and the per-arch backends
// (NEON / SSE4.1 / AVX2 / AVX-512 / wasm-simd128) live in
// `row::dispatch::rgb_ops` and `row::arch::*`.

/// Drops the alpha byte from packed `R, G, B, A` input, producing
/// packed `R, G, B` output (`4 * width` → `3 * width` bytes).
///
/// # Panics
///
/// Panics (any build profile) if `rgba.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba_to_rgb_row(rgba: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgba.len() >= width * 4, "rgba row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = rgba[src];
    rgb_out[dst + 1] = rgba[src + 1];
    rgb_out[dst + 2] = rgba[src + 2];
  }
}

/// Swaps R↔B in packed `B, G, R, A` input, producing packed
/// `R, G, B, A` (alpha lane preserved). The transformation is
/// self‑inverse, so the same routine can be used for
/// `BGRA → RGBA` and `RGBA → BGRA`.
///
/// # Panics
///
/// Panics (any build profile) if `bgra.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra_to_rgba_row(bgra: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = bgra[i + 2];
    rgba_out[i + 1] = bgra[i + 1];
    rgba_out[i + 2] = bgra[i];
    rgba_out[i + 3] = bgra[i + 3];
  }
}

/// Swaps R↔B and drops alpha from packed `B, G, R, A` input,
/// producing packed `R, G, B` (`4 * width` → `3 * width` bytes).
/// Used by [`Bgra`](crate::yuv::Bgra) sinker's RGB / luma / HSV
/// paths — stages a single RGB scratch row that all three reuse.
///
/// # Panics
///
/// Panics (any build profile) if `bgra.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra_to_rgb_row(bgra: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = bgra[src + 2];
    rgb_out[dst + 1] = bgra[src + 1];
    rgb_out[dst + 2] = bgra[src];
  }
}

// ---- Tier 6 leading-alpha helpers (Ship 9c) ----------------------------
//
// Compact byte-rearrangement kernels behind the [`Argb`] / [`Abgr`]
// source-side sinker family. Like Ship 9b, this file provides the
// scalar reference / fallback implementations; SIMD dispatch and the
// per-arch backends live in `row::dispatch::rgb_ops` and
// `row::arch::*`.
//
// `Argb` and `Abgr` differ from `Rgba` / `Bgra` only in alpha
// position — the 4th byte for the trailing-alpha pair, the **0th**
// byte for the leading-alpha pair. The inner three RGB bytes still
// follow each format's marker name.

/// Drops the leading alpha byte from packed `A, R, G, B` input,
/// producing packed `R, G, B` output (`4 * width` → `3 * width`
/// bytes).
///
/// # Panics
///
/// Panics (any build profile) if `argb.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn argb_to_rgb_row(argb: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = argb[src + 1];
    rgb_out[dst + 1] = argb[src + 2];
    rgb_out[dst + 2] = argb[src + 3];
  }
}

/// Swaps R↔B and drops the leading alpha byte from packed
/// `A, B, G, R` input, producing packed `R, G, B`. Used by
/// [`Abgr`](crate::yuv::Abgr) sinker's RGB / luma / HSV paths —
/// stages a single RGB scratch row that all three reuse.
///
/// # Panics
///
/// Panics (any build profile) if `abgr.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn abgr_to_rgb_row(abgr: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = abgr[src + 3];
    rgb_out[dst + 1] = abgr[src + 2];
    rgb_out[dst + 2] = abgr[src + 1];
  }
}

/// Rotates the alpha byte from leading position to trailing position
/// in packed `A, R, G, B` input, producing packed `R, G, B, A`.
/// Alpha pass-through — the rotation is the only mutation per pixel.
///
/// # Panics
///
/// Panics (any build profile) if `argb.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn argb_to_rgba_row(argb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = argb[i + 1];
    rgba_out[i + 1] = argb[i + 2];
    rgba_out[i + 2] = argb[i + 3];
    rgba_out[i + 3] = argb[i];
  }
}

/// Reverses byte order in packed `A, B, G, R` input, producing
/// packed `R, G, B, A`. Combines the leading-alpha → trailing-alpha
/// rotation with an R↔B swap. Self-inverse: the same routine
/// converts `RGBA → ABGR`.
///
/// # Panics
///
/// Panics (any build profile) if `abgr.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn abgr_to_rgba_row(abgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = abgr[i + 3];
    rgba_out[i + 1] = abgr[i + 2];
    rgba_out[i + 2] = abgr[i + 1];
    rgba_out[i + 3] = abgr[i];
  }
}

// ---- Tier 6 padding-byte helpers (Ship 9d) -----------------------------
//
// Compact byte-rearrangement kernels behind the [`Xrgb`] / [`Rgbx`] /
// [`Xbgr`] / [`Bgrx`] source-side sinker family. The 4th byte position
// (leading or trailing) is **ignored padding** — its value is undefined,
// so the `*_to_rgba` kernels force alpha to `0xFF` rather than passing
// through.
//
// `*_to_rgb` paths reuse the Ship 9b/9c kernels (`argb_to_rgb_row`,
// `rgba_to_rgb_row`, `abgr_to_rgb_row`, `bgra_to_rgb_row`) — at the byte
// level, "drop alpha" and "drop padding" are identical operations
// because both ignore the same byte position.
//
// As with Ships 9b/9c, this file holds the scalar reference / fallback
// implementations; SIMD dispatch lives in `row::dispatch::rgb_ops`
// and the per-arch backends in `row::arch::*`.

/// Drops the leading padding byte from packed `X, R, G, B` input,
/// producing packed `R, G, B, A` with `A = 0xFF` (the source has no
/// real alpha — `X` is undefined padding).
///
/// # Panics
///
/// Panics (any build profile) if `xrgb.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xrgb_to_rgba_row(xrgb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xrgb.len() >= width * 4, "xrgb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = xrgb[i + 1];
    rgba_out[i + 1] = xrgb[i + 2];
    rgba_out[i + 2] = xrgb[i + 3];
    rgba_out[i + 3] = 0xFF;
  }
}

/// Drops the trailing padding byte from packed `R, G, B, X` input,
/// producing packed `R, G, B, A` with `A = 0xFF`.
///
/// # Panics
///
/// Panics (any build profile) if `rgbx.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbx_to_rgba_row(rgbx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgbx.len() >= width * 4, "rgbx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = rgbx[i];
    rgba_out[i + 1] = rgbx[i + 1];
    rgba_out[i + 2] = rgbx[i + 2];
    rgba_out[i + 3] = 0xFF;
  }
}

/// Reverses the inner three bytes and drops the leading padding byte
/// from packed `X, B, G, R` input, producing packed `R, G, B, A` with
/// `A = 0xFF`.
///
/// # Panics
///
/// Panics (any build profile) if `xbgr.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xbgr_to_rgba_row(xbgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xbgr.len() >= width * 4, "xbgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = xbgr[i + 3];
    rgba_out[i + 1] = xbgr[i + 2];
    rgba_out[i + 2] = xbgr[i + 1];
    rgba_out[i + 3] = 0xFF;
  }
}

/// Reverses the inner three bytes and drops the trailing padding
/// byte from packed `B, G, R, X` input, producing packed `R, G, B, A`
/// with `A = 0xFF`.
///
/// # Panics
///
/// Panics (any build profile) if `bgrx.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgrx_to_rgba_row(bgrx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgrx.len() >= width * 4, "bgrx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = bgrx[i + 2];
    rgba_out[i + 1] = bgrx[i + 1];
    rgba_out[i + 2] = bgrx[i];
    rgba_out[i + 3] = 0xFF;
  }
}

// ---- Tier 6 10-bit packed-RGB helpers (Ship 9e) ------------------------
//
// Bit-extraction kernels behind the [`X2Rgb10`] / [`X2Bgr10`]
// source-side sinker family. Each input pixel is a 32-bit
// little-endian word with 2 bits of unused padding at the top and
// three 10-bit channels packed below.
//
// FFmpeg layout (LE, MSB→LSB):
// - `X2RGB10`: `2X | 10R | 10G | 10B` — read as `u32`, R is at
//   bits 20–29, G at 10–19, B at 0–9.
// - `X2BGR10`: `2X | 10B | 10G | 10R` — channel positions swapped;
//   R is at bits 0–9, B at 20–29.
//
// Output flavours per source:
// - **u8 RGB** — each 10-bit channel down-shifted by 2 (drop low
//   bits) into a packed `R, G, B` triple. The dropped bits are not
//   rounded; the existing 8-bit output path uses truncation
//   throughout.
// - **u8 RGBA** — same channel handling + alpha set to `0xFF` (the
//   2-bit field is padding, not real alpha).
// - **u16 native RGB** — channel value preserved at full 10-bit
//   precision; emitted as `u16` in the **low 10 bits**, matching the
//   convention used by the rest of the high-bit-depth crate (e.g.
//   `yuv_420p_n_to_rgb_u16_row::<10>`). Max value `1023`.
//
// SIMD dispatch and per-arch backends live in `row::dispatch::rgb_ops`
// and `row::arch::*`.

/// Drops the 2-bit padding and down-shifts each 10-bit channel to
/// 8 bits, producing packed `R, G, B` from packed `X2RGB10` LE
/// input.
///
/// # Panics
///
/// Panics (any build profile) if `x2rgb10.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn x2rgb10_to_rgb_row(x2rgb10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 4;
    let pix = u32::from_le_bytes([x2rgb10[i], x2rgb10[i + 1], x2rgb10[i + 2], x2rgb10[i + 3]]);
    let r10 = (pix >> 20) & 0x3FF;
    let g10 = (pix >> 10) & 0x3FF;
    let b10 = pix & 0x3FF;
    let dst = x * 3;
    rgb_out[dst] = (r10 >> 2) as u8;
    rgb_out[dst + 1] = (g10 >> 2) as u8;
    rgb_out[dst + 2] = (b10 >> 2) as u8;
  }
}

/// Drops the 2-bit padding, down-shifts each 10-bit channel to
/// 8 bits, and forces alpha to `0xFF`. Output: packed `R, G, B, A`.
///
/// # Panics
///
/// Panics (any build profile) if `x2rgb10.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn x2rgb10_to_rgba_row(x2rgb10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    let pix = u32::from_le_bytes([x2rgb10[i], x2rgb10[i + 1], x2rgb10[i + 2], x2rgb10[i + 3]]);
    let r10 = (pix >> 20) & 0x3FF;
    let g10 = (pix >> 10) & 0x3FF;
    let b10 = pix & 0x3FF;
    rgba_out[i] = (r10 >> 2) as u8;
    rgba_out[i + 1] = (g10 >> 2) as u8;
    rgba_out[i + 2] = (b10 >> 2) as u8;
    rgba_out[i + 3] = 0xFF;
  }
}

/// Extracts each 10-bit channel into native-depth `u16` (low-bit
/// aligned, max value `1023`), producing packed `R, G, B` `u16`
/// elements from packed `X2RGB10` LE input.
///
/// # Panics
///
/// Panics (any build profile) if `x2rgb10.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn x2rgb10_to_rgb_u16_row(x2rgb10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 4;
    let pix = u32::from_le_bytes([x2rgb10[i], x2rgb10[i + 1], x2rgb10[i + 2], x2rgb10[i + 3]]);
    let dst = x * 3;
    rgb_out[dst] = ((pix >> 20) & 0x3FF) as u16;
    rgb_out[dst + 1] = ((pix >> 10) & 0x3FF) as u16;
    rgb_out[dst + 2] = (pix & 0x3FF) as u16;
  }
}

/// `X2BGR10` LE counterpart of [`x2rgb10_to_rgb_row`]. Channel
/// positions are swapped: R is at bits 0–9 and B at 20–29 of the
/// `u32`. Output is still `R, G, B`.
///
/// # Panics
///
/// Panics (any build profile) if `x2bgr10.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn x2bgr10_to_rgb_row(x2bgr10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 4;
    let pix = u32::from_le_bytes([x2bgr10[i], x2bgr10[i + 1], x2bgr10[i + 2], x2bgr10[i + 3]]);
    let r10 = pix & 0x3FF;
    let g10 = (pix >> 10) & 0x3FF;
    let b10 = (pix >> 20) & 0x3FF;
    let dst = x * 3;
    rgb_out[dst] = (r10 >> 2) as u8;
    rgb_out[dst + 1] = (g10 >> 2) as u8;
    rgb_out[dst + 2] = (b10 >> 2) as u8;
  }
}

/// `X2BGR10` LE counterpart of [`x2rgb10_to_rgba_row`].
///
/// # Panics
///
/// Panics (any build profile) if `x2bgr10.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn x2bgr10_to_rgba_row(x2bgr10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    let pix = u32::from_le_bytes([x2bgr10[i], x2bgr10[i + 1], x2bgr10[i + 2], x2bgr10[i + 3]]);
    let r10 = pix & 0x3FF;
    let g10 = (pix >> 10) & 0x3FF;
    let b10 = (pix >> 20) & 0x3FF;
    rgba_out[i] = (r10 >> 2) as u8;
    rgba_out[i + 1] = (g10 >> 2) as u8;
    rgba_out[i + 2] = (b10 >> 2) as u8;
    rgba_out[i + 3] = 0xFF;
  }
}

/// `X2BGR10` LE counterpart of [`x2rgb10_to_rgb_u16_row`].
///
/// # Panics
///
/// Panics (any build profile) if `x2bgr10.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn x2bgr10_to_rgb_u16_row(x2bgr10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 4;
    let pix = u32::from_le_bytes([x2bgr10[i], x2bgr10[i + 1], x2bgr10[i + 2], x2bgr10[i + 3]]);
    let dst = x * 3;
    rgb_out[dst] = (pix & 0x3FF) as u16;
    rgb_out[dst + 1] = ((pix >> 10) & 0x3FF) as u16;
    rgb_out[dst + 2] = ((pix >> 20) & 0x3FF) as u16;
  }
}
