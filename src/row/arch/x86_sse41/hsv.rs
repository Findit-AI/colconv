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

// ===== RGB → luma (Y') ===================================================

/// SSE4.1 RGB → planar luma (Y'). Byte‑identical to
/// [`scalar::rgb_to_luma_row`]. 16 pixels per iteration via the shared
/// [`rgb_to_luma_16_pixels`] helper. Coefficients are hoisted once
/// outside the loop into `__m128i` broadcasts so `matrix` selection
/// only costs a few setup ops.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb_to_luma_row(
  rgb: &[u8],
  luma_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(rgb.len() >= width * 3);
  debug_assert!(luma_out.len() >= width);

  let (k_r, k_g, k_b) = scalar::luma_coefficients_q15(matrix);
  // SAFETY: SSE4.1 verified at the dispatcher; loop guard `x + 16 <=
  // width` keeps the 48‑byte read and 16‑byte write inside the
  // caller‑promised slice lengths.
  unsafe {
    let kr_v = _mm_set1_epi32(k_r);
    let kg_v = _mm_set1_epi32(k_g);
    let kb_v = _mm_set1_epi32(k_b);
    let rnd_v = _mm_set1_epi32(1 << 14);

    let mut x = 0usize;
    while x + 16 <= width {
      rgb_to_luma_16_pixels(
        rgb.as_ptr().add(x * 3),
        luma_out.as_mut_ptr().add(x),
        kr_v,
        kg_v,
        kb_v,
        rnd_v,
        full_range,
      );
      x += 16;
    }
    if x < width {
      scalar::rgb_to_luma_row(
        &rgb[x * 3..width * 3],
        &mut luma_out[x..width],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}
