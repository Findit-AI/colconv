use super::*;

// ===== RGB → HSV =========================================================

/// SSE4.1 RGB → planar HSV (OpenCV 8‑bit encoding). 16 pixels per
/// iteration via the shared [`super::x86_common::rgb_to_hsv_16_pixels`]
/// helper.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`.
/// 3. `h_out.len() >= width`, `s_out.len() >= width`, `v_out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    while x + 16 <= width {
      rgb_to_hsv_16_pixels(
        rgb.as_ptr().add(x * 3),
        h_out.as_mut_ptr().add(x),
        s_out.as_mut_ptr().add(x),
        v_out.as_mut_ptr().add(x),
      );
      x += 16;
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
