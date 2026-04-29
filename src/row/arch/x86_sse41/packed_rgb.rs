use core::arch::x86_64::*;

use super::*;

// ===== BGR ↔ RGB byte swap ==============================================

/// SSE4.1 BGR ↔ RGB byte swap. 16 pixels per iteration via the shared
/// [`super::x86_common::swap_rb_16_pixels`] helper (SSSE3 `_mm_shuffle_epi8`
/// underneath). Drives both conversion directions since the swap is
/// self‑inverse.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  // SAFETY: SSE4.1 is available per caller obligation; SSSE3 (required
  // by `swap_rb_16_pixels`) is a subset. All pointer adds are bounded
  // by the `while x + 16 <= width` condition.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      swap_rb_16_pixels(input.as_ptr().add(x * 3), output.as_mut_ptr().add(x * 3));
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

/// SSE4.1 RGBA→RGB drop-alpha. 16 pixels per iteration via the shared
/// [`super::x86_common::drop_alpha_16_pixels`] helper.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `rgba.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgba` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgba_to_rgb_row(rgba: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgba.len() >= width * 4, "rgba row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      drop_alpha_16_pixels(rgba.as_ptr().add(x * 4), rgb_out.as_mut_ptr().add(x * 3));
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

/// SSE4.1 BGRA→RGBA R↔B swap with alpha pass-through. 16 pixels per
/// iteration via four `_mm_shuffle_epi8` calls (one per 16-byte
/// vector, four pixels each).
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `bgra.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `bgra` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgra_to_rgba_row(bgra: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = bgra.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      swap_rb_alpha_4_pixels(base_in, base_out);
      swap_rb_alpha_4_pixels(base_in.add(16), base_out.add(16));
      swap_rb_alpha_4_pixels(base_in.add(32), base_out.add(32));
      swap_rb_alpha_4_pixels(base_in.add(48), base_out.add(48));
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

/// SSE4.1 BGRA→RGB combined R↔B swap and alpha drop. 16 pixels per
/// iteration via the shared
/// [`super::x86_common::bgra_to_rgb_16_pixels`] helper.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `bgra.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `bgra` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgra_to_rgb_row(bgra: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      bgra_to_rgb_16_pixels(bgra.as_ptr().add(x * 4), rgb_out.as_mut_ptr().add(x * 3));
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

/// SSE4.1 ARGB→RGB drop-leading-alpha. 16 pixels per iteration via
/// the shared [`super::x86_common::argb_to_rgb_16_pixels`] helper.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `argb.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `argb` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn argb_to_rgb_row(argb: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      argb_to_rgb_16_pixels(argb.as_ptr().add(x * 4), rgb_out.as_mut_ptr().add(x * 3));
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

/// SSE4.1 ABGR→RGB combined drop-leading-alpha + R↔B swap. 16 pixels
/// per iteration via the shared
/// [`super::x86_common::abgr_to_rgb_16_pixels`] helper.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `abgr.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `abgr` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn abgr_to_rgb_row(abgr: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      abgr_to_rgb_16_pixels(abgr.as_ptr().add(x * 4), rgb_out.as_mut_ptr().add(x * 3));
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

/// SSE4.1 ARGB→RGBA leading-alpha rotation. 16 pixels per iteration
/// via four `_mm_shuffle_epi8` calls (one per 16-byte vector, four
/// pixels each).
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `argb.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `argb` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn argb_to_rgba_row(argb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = argb.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      argb_to_rgba_4_pixels(base_in, base_out);
      argb_to_rgba_4_pixels(base_in.add(16), base_out.add(16));
      argb_to_rgba_4_pixels(base_in.add(32), base_out.add(32));
      argb_to_rgba_4_pixels(base_in.add(48), base_out.add(48));
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

/// SSE4.1 ABGR→RGBA full byte reverse. 16 pixels per iteration via
/// four `_mm_shuffle_epi8` calls.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `abgr.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `abgr` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn abgr_to_rgba_row(abgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = abgr.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      abgr_to_rgba_4_pixels(base_in, base_out);
      abgr_to_rgba_4_pixels(base_in.add(16), base_out.add(16));
      abgr_to_rgba_4_pixels(base_in.add(32), base_out.add(32));
      abgr_to_rgba_4_pixels(base_in.add(48), base_out.add(48));
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

/// SSE4.1 XRGB→RGBA drop-leading-padding + force `A = 0xFF`. 16
/// pixels per iteration via four
/// [`super::x86_common::xrgb_to_rgba_4_pixels`] calls.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `xrgb.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `xrgb` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xrgb_to_rgba_row(xrgb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xrgb.len() >= width * 4, "xrgb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = xrgb.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      xrgb_to_rgba_4_pixels(base_in, base_out);
      xrgb_to_rgba_4_pixels(base_in.add(16), base_out.add(16));
      xrgb_to_rgba_4_pixels(base_in.add(32), base_out.add(32));
      xrgb_to_rgba_4_pixels(base_in.add(48), base_out.add(48));
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

/// SSE4.1 RGBX→RGBA drop-trailing-padding + force `A = 0xFF`. 16
/// pixels per iteration.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgbx.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `rgbx` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbx_to_rgba_row(rgbx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgbx.len() >= width * 4, "rgbx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = rgbx.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      rgbx_to_rgba_4_pixels(base_in, base_out);
      rgbx_to_rgba_4_pixels(base_in.add(16), base_out.add(16));
      rgbx_to_rgba_4_pixels(base_in.add(32), base_out.add(32));
      rgbx_to_rgba_4_pixels(base_in.add(48), base_out.add(48));
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

/// SSE4.1 XBGR→RGBA reverse-and-drop-leading-padding + force
/// `A = 0xFF`. 16 pixels per iteration.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `xbgr.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `xbgr` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xbgr_to_rgba_row(xbgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xbgr.len() >= width * 4, "xbgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = xbgr.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      xbgr_to_rgba_4_pixels(base_in, base_out);
      xbgr_to_rgba_4_pixels(base_in.add(16), base_out.add(16));
      xbgr_to_rgba_4_pixels(base_in.add(32), base_out.add(32));
      xbgr_to_rgba_4_pixels(base_in.add(48), base_out.add(48));
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

/// SSE4.1 BGRX→RGBA reverse-and-drop-trailing-padding + force
/// `A = 0xFF`. 16 pixels per iteration.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgrx.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `bgrx` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgrx_to_rgba_row(bgrx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgrx.len() >= width * 4, "bgrx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let base_in = bgrx.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      bgrx_to_rgba_4_pixels(base_in, base_out);
      bgrx_to_rgba_4_pixels(base_in.add(16), base_out.add(16));
      bgrx_to_rgba_4_pixels(base_in.add(32), base_out.add(32));
      bgrx_to_rgba_4_pixels(base_in.add(48), base_out.add(48));
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

/// SSE4.1 X2RGB10→RGB. 16 pixels per iteration via the shared
/// [`super::x86_common::x2rgb10_to_rgb_16_pixels`] helper.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `x2rgb10.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `x2rgb10` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn x2rgb10_to_rgb_row(x2rgb10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      x2rgb10_to_rgb_16_pixels(x2rgb10.as_ptr().add(x * 4), rgb_out.as_mut_ptr().add(x * 3));
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

/// SSE4.1 X2RGB10→RGBA. 16 pixels per iteration.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn x2rgb10_to_rgba_row(x2rgb10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      x2rgb10_to_rgba_16_pixels(
        x2rgb10.as_ptr().add(x * 4),
        rgba_out.as_mut_ptr().add(x * 4),
      );
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

/// SSE4.1 X2RGB10→u16 RGB native (10-bit values low-bit aligned in
/// `u16`, max value `1023`). 8 pixels per iteration.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn x2rgb10_to_rgb_u16_row(x2rgb10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      x2rgb10_to_rgb_u16_8_pixels(
        x2rgb10.as_ptr().add(x * 4),
        rgb_out.as_mut_ptr().add(x * 3).cast::<u8>(),
      );
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

/// SSE4.1 X2BGR10→RGB. 16 pixels per iteration.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn x2bgr10_to_rgb_row(x2bgr10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      x2bgr10_to_rgb_16_pixels(x2bgr10.as_ptr().add(x * 4), rgb_out.as_mut_ptr().add(x * 3));
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

/// SSE4.1 X2BGR10→RGBA. 16 pixels per iteration.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn x2bgr10_to_rgba_row(x2bgr10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      x2bgr10_to_rgba_16_pixels(
        x2bgr10.as_ptr().add(x * 4),
        rgba_out.as_mut_ptr().add(x * 4),
      );
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

/// SSE4.1 X2BGR10→u16 RGB native. 8 pixels per iteration.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn x2bgr10_to_rgb_u16_row(x2bgr10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      x2bgr10_to_rgb_u16_8_pixels(
        x2bgr10.as_ptr().add(x * 4),
        rgb_out.as_mut_ptr().add(x * 3).cast::<u8>(),
      );
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
