use super::*;

// ===== RGB → HSV =========================================================

/// AVX2 RGB → planar HSV. 32 pixels per iteration via two calls to the
/// shared [`super::x86_common::rgb_to_hsv_16_pixels`] helper (SSE4.1
/// level compute, memory‑bandwidth‑bound — wider f32 registers would
/// help if we restructured, but the current structure already wins
/// versus scalar).
///
/// # Safety
///
/// 1. AVX2 must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`; each output plane `>= width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 32 <= width {
      rgb_to_hsv_16_pixels(
        rgb.as_ptr().add(x * 3),
        h_out.as_mut_ptr().add(x),
        s_out.as_mut_ptr().add(x),
        v_out.as_mut_ptr().add(x),
      );
      rgb_to_hsv_16_pixels(
        rgb.as_ptr().add(x * 3 + 48),
        h_out.as_mut_ptr().add(x + 16),
        s_out.as_mut_ptr().add(x + 16),
        v_out.as_mut_ptr().add(x + 16),
      );
      x += 32;
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

/// AVX2 RGB → planar luma (Y'). 32 pixels per iteration via two calls
/// to the shared [`super::x86_common::rgb_to_luma_16_pixels`] helper
/// (SSE4.1‑level compute under AVX2 target_feature). Coefficients
/// hoisted once into i32x4 broadcasts so per‑matrix selection is
/// negligible.
///
/// # Safety
///
/// 1. AVX2 must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`; `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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
  // SAFETY: AVX2 verified at the dispatcher; loop guard `x + 32 <=
  // width` keeps both 48‑byte reads and 16‑byte writes inside the
  // caller‑promised slice lengths.
  unsafe {
    let kr_v = _mm_set1_epi32(k_r);
    let kg_v = _mm_set1_epi32(k_g);
    let kb_v = _mm_set1_epi32(k_b);
    let rnd_v = _mm_set1_epi32(1 << 14);

    let mut x = 0usize;
    while x + 32 <= width {
      rgb_to_luma_16_pixels(
        rgb.as_ptr().add(x * 3),
        luma_out.as_mut_ptr().add(x),
        kr_v,
        kg_v,
        kb_v,
        rnd_v,
        full_range,
      );
      rgb_to_luma_16_pixels(
        rgb.as_ptr().add(x * 3 + 48),
        luma_out.as_mut_ptr().add(x + 16),
        kr_v,
        kg_v,
        kb_v,
        rnd_v,
        full_range,
      );
      x += 32;
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
