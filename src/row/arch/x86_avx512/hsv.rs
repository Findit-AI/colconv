use super::*;

// ===== RGB → HSV =========================================================

/// AVX‑512 RGB → planar HSV. 64 pixels per iteration via four calls to
/// the shared [`super::x86_common::rgb_to_hsv_16_pixels`] helper
/// (SSE4.1‑level compute under AVX‑512 target_feature). Matches the
/// scalar reference within ±1 LSB — the shared helper uses `_mm_rcp_ps`
/// + one Newton‑Raphson step instead of true division (see `x86_common.rs`).
///
/// # Safety
///
/// 1. AVX‑512BW must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`; each output plane `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = rgb.as_ptr().add(x * 3);
      let base_h = h_out.as_mut_ptr().add(x);
      let base_s = s_out.as_mut_ptr().add(x);
      let base_v = v_out.as_mut_ptr().add(x);
      rgb_to_hsv_16_pixels(base_in, base_h, base_s, base_v);
      rgb_to_hsv_16_pixels(
        base_in.add(48),
        base_h.add(16),
        base_s.add(16),
        base_v.add(16),
      );
      rgb_to_hsv_16_pixels(
        base_in.add(96),
        base_h.add(32),
        base_s.add(32),
        base_v.add(32),
      );
      rgb_to_hsv_16_pixels(
        base_in.add(144),
        base_h.add(48),
        base_s.add(48),
        base_v.add(48),
      );
      x += 64;
    }
    if x < width {
      scalar::rgb_to_hsv_row(
        &rgb[x * 3..width * 3],
        &mut h_out[x..width],
        &mut s_out[x..width],
        &mut v_out[x..width],
        width - x,
      );
    }
  }
}
