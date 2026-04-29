use super::*;

// ===== BGR ↔ RGB byte swap ==============================================

/// AVX‑512 BGR ↔ RGB byte swap. 64 pixels per iteration via four calls
/// to [`super::x86_common::swap_rb_16_pixels`]. The helper uses SSSE3
/// `_mm_shuffle_epi8`, which AVX‑512BW (a superset) allows.
///
/// # Safety
///
/// 1. AVX‑512BW must be available (dispatcher obligation).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = input.as_ptr().add(x * 3);
      let base_out = output.as_mut_ptr().add(x * 3);
      swap_rb_16_pixels(base_in, base_out);
      swap_rb_16_pixels(base_in.add(48), base_out.add(48));
      swap_rb_16_pixels(base_in.add(96), base_out.add(96));
      swap_rb_16_pixels(base_in.add(144), base_out.add(144));
      x += 64;
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

/// AVX-512 RGBA→RGB drop-alpha. 64 pixels per iteration via four
/// calls to [`super::x86_common::drop_alpha_16_pixels`]. Memory-
/// bandwidth bound — wider 512-bit lanes don't change practical
/// throughput.
///
/// # Safety
///
/// 1. AVX-512BW must be available (dispatcher obligation).
/// 2. `rgba.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgba` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgba_to_rgb_row(rgba: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgba.len() >= width * 4, "rgba row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = rgba.as_ptr().add(x * 4);
      let base_out = rgb_out.as_mut_ptr().add(x * 3);
      drop_alpha_16_pixels(base_in, base_out);
      drop_alpha_16_pixels(base_in.add(64), base_out.add(48));
      drop_alpha_16_pixels(base_in.add(128), base_out.add(96));
      drop_alpha_16_pixels(base_in.add(192), base_out.add(144));
      x += 64;
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

/// AVX-512 BGRA→RGBA R↔B swap with alpha pass-through. 64 pixels per
/// iteration via 16 calls to
/// [`super::x86_common::swap_rb_alpha_4_pixels`].
///
/// # Safety
///
/// 1. AVX-512BW must be available.
/// 2. `bgra.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `bgra` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgra_to_rgba_row(bgra: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = bgra.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      // 16 × 16-byte chunks = 256 bytes = 64 pixels per iteration.
      let mut off = 0usize;
      while off < 256 {
        swap_rb_alpha_4_pixels(base_in.add(off), base_out.add(off));
        off += 16;
      }
      x += 64;
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

/// AVX-512 BGRA→RGB combined R↔B swap and alpha drop. 64 pixels per
/// iteration via four calls to
/// [`super::x86_common::bgra_to_rgb_16_pixels`].
///
/// # Safety
///
/// 1. AVX-512BW must be available.
/// 2. `bgra.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `bgra` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgra_to_rgb_row(bgra: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgra.len() >= width * 4, "bgra row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = bgra.as_ptr().add(x * 4);
      let base_out = rgb_out.as_mut_ptr().add(x * 3);
      bgra_to_rgb_16_pixels(base_in, base_out);
      bgra_to_rgb_16_pixels(base_in.add(64), base_out.add(48));
      bgra_to_rgb_16_pixels(base_in.add(128), base_out.add(96));
      bgra_to_rgb_16_pixels(base_in.add(192), base_out.add(144));
      x += 64;
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

/// AVX-512 ARGB→RGB drop-leading-alpha. 64 pixels per iteration via
/// four calls to [`super::x86_common::argb_to_rgb_16_pixels`].
///
/// # Safety
///
/// 1. AVX-512BW must be available (dispatcher obligation).
/// 2. `argb.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `argb` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn argb_to_rgb_row(argb: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = argb.as_ptr().add(x * 4);
      let base_out = rgb_out.as_mut_ptr().add(x * 3);
      argb_to_rgb_16_pixels(base_in, base_out);
      argb_to_rgb_16_pixels(base_in.add(64), base_out.add(48));
      argb_to_rgb_16_pixels(base_in.add(128), base_out.add(96));
      argb_to_rgb_16_pixels(base_in.add(192), base_out.add(144));
      x += 64;
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

/// AVX-512 ABGR→RGB combined drop-leading-alpha + R↔B swap. 64
/// pixels per iteration via four calls to
/// [`super::x86_common::abgr_to_rgb_16_pixels`].
///
/// # Safety
///
/// 1. AVX-512BW must be available.
/// 2. `abgr.len() >= 4 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `abgr` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn abgr_to_rgb_row(abgr: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = abgr.as_ptr().add(x * 4);
      let base_out = rgb_out.as_mut_ptr().add(x * 3);
      abgr_to_rgb_16_pixels(base_in, base_out);
      abgr_to_rgb_16_pixels(base_in.add(64), base_out.add(48));
      abgr_to_rgb_16_pixels(base_in.add(128), base_out.add(96));
      abgr_to_rgb_16_pixels(base_in.add(192), base_out.add(144));
      x += 64;
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

/// AVX-512 ARGB→RGBA leading-alpha rotation. 64 pixels per iteration
/// via 16 calls to [`super::x86_common::argb_to_rgba_4_pixels`].
///
/// # Safety
///
/// 1. AVX-512BW must be available.
/// 2. `argb.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `argb` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn argb_to_rgba_row(argb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(argb.len() >= width * 4, "argb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = argb.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let mut off = 0usize;
      while off < 256 {
        argb_to_rgba_4_pixels(base_in.add(off), base_out.add(off));
        off += 16;
      }
      x += 64;
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

/// AVX-512 ABGR→RGBA full byte reverse. 64 pixels per iteration via
/// 16 calls to [`super::x86_common::abgr_to_rgba_4_pixels`].
///
/// # Safety
///
/// 1. AVX-512BW must be available.
/// 2. `abgr.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `abgr` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn abgr_to_rgba_row(abgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(abgr.len() >= width * 4, "abgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = abgr.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let mut off = 0usize;
      while off < 256 {
        abgr_to_rgba_4_pixels(base_in.add(off), base_out.add(off));
        off += 16;
      }
      x += 64;
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

/// AVX-512 XRGB→RGBA. 64 pixels per iteration via 16 calls to
/// [`super::x86_common::xrgb_to_rgba_4_pixels`].
///
/// # Safety
///
/// 1. AVX-512BW must be available (dispatcher obligation).
/// 2. `xrgb.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `xrgb` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xrgb_to_rgba_row(xrgb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xrgb.len() >= width * 4, "xrgb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = xrgb.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let mut off = 0usize;
      while off < 256 {
        xrgb_to_rgba_4_pixels(base_in.add(off), base_out.add(off));
        off += 16;
      }
      x += 64;
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

/// AVX-512 RGBX→RGBA. 64 pixels per iteration.
///
/// # Safety
///
/// 1. AVX-512BW must be available.
/// 2. `rgbx.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `rgbx` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgbx_to_rgba_row(rgbx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgbx.len() >= width * 4, "rgbx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = rgbx.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let mut off = 0usize;
      while off < 256 {
        rgbx_to_rgba_4_pixels(base_in.add(off), base_out.add(off));
        off += 16;
      }
      x += 64;
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

/// AVX-512 XBGR→RGBA. 64 pixels per iteration.
///
/// # Safety
///
/// 1. AVX-512BW must be available.
/// 2. `xbgr.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `xbgr` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xbgr_to_rgba_row(xbgr: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(xbgr.len() >= width * 4, "xbgr row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = xbgr.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let mut off = 0usize;
      while off < 256 {
        xbgr_to_rgba_4_pixels(base_in.add(off), base_out.add(off));
        off += 16;
      }
      x += 64;
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

/// AVX-512 BGRX→RGBA. 64 pixels per iteration.
///
/// # Safety
///
/// 1. AVX-512BW must be available.
/// 2. `bgrx.len() >= 4 * width`; `rgba_out.len() >= 4 * width`.
/// 3. `bgrx` / `rgba_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgrx_to_rgba_row(bgrx: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgrx.len() >= width * 4, "bgrx row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = bgrx.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      let mut off = 0usize;
      while off < 256 {
        bgrx_to_rgba_4_pixels(base_in.add(off), base_out.add(off));
        off += 16;
      }
      x += 64;
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

/// AVX-512 X2RGB10→RGB. 64 pixels per iteration via four calls to
/// [`super::x86_common::x2rgb10_to_rgb_16_pixels`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn x2rgb10_to_rgb_row(x2rgb10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = x2rgb10.as_ptr().add(x * 4);
      let base_out = rgb_out.as_mut_ptr().add(x * 3);
      x2rgb10_to_rgb_16_pixels(base_in, base_out);
      x2rgb10_to_rgb_16_pixels(base_in.add(64), base_out.add(48));
      x2rgb10_to_rgb_16_pixels(base_in.add(128), base_out.add(96));
      x2rgb10_to_rgb_16_pixels(base_in.add(192), base_out.add(144));
      x += 64;
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

/// AVX-512 X2RGB10→RGBA. 64 pixels per iteration.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn x2rgb10_to_rgba_row(x2rgb10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = x2rgb10.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      x2rgb10_to_rgba_16_pixels(base_in, base_out);
      x2rgb10_to_rgba_16_pixels(base_in.add(64), base_out.add(64));
      x2rgb10_to_rgba_16_pixels(base_in.add(128), base_out.add(128));
      x2rgb10_to_rgba_16_pixels(base_in.add(192), base_out.add(192));
      x += 64;
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

/// AVX-512 X2RGB10→u16 RGB native. 32 pixels per iteration.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn x2rgb10_to_rgb_u16_row(x2rgb10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2rgb10.len() >= width * 4, "x2rgb10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let base_in = x2rgb10.as_ptr().add(x * 4);
      let base_out = rgb_out.as_mut_ptr().add(x * 3).cast::<u8>();
      x2rgb10_to_rgb_u16_8_pixels(base_in, base_out);
      x2rgb10_to_rgb_u16_8_pixels(base_in.add(32), base_out.add(48));
      x2rgb10_to_rgb_u16_8_pixels(base_in.add(64), base_out.add(96));
      x2rgb10_to_rgb_u16_8_pixels(base_in.add(96), base_out.add(144));
      x += 32;
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

/// AVX-512 X2BGR10→RGB. 64 pixels per iteration.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn x2bgr10_to_rgb_row(x2bgr10: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = x2bgr10.as_ptr().add(x * 4);
      let base_out = rgb_out.as_mut_ptr().add(x * 3);
      x2bgr10_to_rgb_16_pixels(base_in, base_out);
      x2bgr10_to_rgb_16_pixels(base_in.add(64), base_out.add(48));
      x2bgr10_to_rgb_16_pixels(base_in.add(128), base_out.add(96));
      x2bgr10_to_rgb_16_pixels(base_in.add(192), base_out.add(144));
      x += 64;
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

/// AVX-512 X2BGR10→RGBA. 64 pixels per iteration.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn x2bgr10_to_rgba_row(x2bgr10: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = x2bgr10.as_ptr().add(x * 4);
      let base_out = rgba_out.as_mut_ptr().add(x * 4);
      x2bgr10_to_rgba_16_pixels(base_in, base_out);
      x2bgr10_to_rgba_16_pixels(base_in.add(64), base_out.add(64));
      x2bgr10_to_rgba_16_pixels(base_in.add(128), base_out.add(128));
      x2bgr10_to_rgba_16_pixels(base_in.add(192), base_out.add(192));
      x += 64;
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

/// AVX-512 X2BGR10→u16 RGB native. 32 pixels per iteration.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn x2bgr10_to_rgb_u16_row(x2bgr10: &[u8], rgb_out: &mut [u16], width: usize) {
  debug_assert!(x2bgr10.len() >= width * 4, "x2bgr10 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let base_in = x2bgr10.as_ptr().add(x * 4);
      let base_out = rgb_out.as_mut_ptr().add(x * 3).cast::<u8>();
      x2bgr10_to_rgb_u16_8_pixels(base_in, base_out);
      x2bgr10_to_rgb_u16_8_pixels(base_in.add(32), base_out.add(48));
      x2bgr10_to_rgb_u16_8_pixels(base_in.add(64), base_out.add(96));
      x2bgr10_to_rgb_u16_8_pixels(base_in.add(96), base_out.add(144));
      x += 32;
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
