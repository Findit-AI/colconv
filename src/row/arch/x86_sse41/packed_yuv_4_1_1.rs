//! SSE4.1 kernels for the Tier 5.25 packed YUV 4:1:1 source (UYYVYY411).
//!
//! Per‑block layout (6 bytes / 4 pixels): `[U, Y0, Y1, V, Y2, Y3]`.
//! Each (U, V) chroma pair is shared by 4 adjacent luma samples
//! (1 → 4 horizontal chroma fan‑out).
//!
//! ## Per‑iter pipeline (16 px / 24 input bytes)
//!
//! 1. Two overlapping `_mm_loadu_si128` loads at offsets 0 and 8 cover
//!    the full 24‑byte / 4‑block window. Loop bound `x + 16 <= width`
//!    plus the `packed.len() >= width * 3 / 2` contract guarantee
//!    24 readable bytes.
//! 2. Three `_mm_shuffle_epi8` calls extract:
//!    - `y_vec` (16 Y bytes — Y0..Y3 of each of the 4 blocks),
//!    - `uv_vec` (4 U + 4 V bytes, OR‑merged across the two halves).
//! 3. Widen U / V to i32x4, run Q15 chroma math (same `chroma_i16x8` as
//!    `packed_yuv_8bit`) producing i16x8 chroma vectors with 4 valid
//!    lanes (one per chroma sample).
//! 4. Fan each of the 4 chroma i16 values to 4 adjacent lanes (1 → 4
//!    upsample) via `_mm_shuffle_epi8`, yielding two i16x8 chroma
//!    vectors covering the 16 Y pixels.
//! 5. Standard `scale_y` + saturating add + `_mm_packus_epi16` →
//!    `write_rgb_16` / `write_rgba_16`.
//! 6. Scalar tail for `width % 16 != 0`.

use super::*;

/// SSE4.1 UYYVYY411 → packed RGB. Semantics match
/// [`scalar::uyyvyy411_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 3 == 0` (4:1:1 chroma group).
/// 3. `packed.len() >= width * 3 / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn uyyvyy411_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 UYYVYY411 → packed RGBA (alpha = 0xFF).
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_rgb_row`] with `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn uyyvyy411_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// Generic UYYVYY411 → RGB / RGBA SSE4.1 kernel. 16 px / iter.
///
/// # Safety
///
/// Caller has verified SSE4.1. `packed.len() >= width * 3 / 2`. `width`
/// is a multiple of 4. `out.len() >= bpp * width` where bpp = 3 for
/// `ALPHA = false`, 4 for `ALPHA = true`.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn uyyvyy411_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm_set1_epi8(-1);

    // Y mask for p0 (byte offsets 0..15 of input): extract 8 Y bytes
    // (Y00, Y01, Y02, Y03, Y10, Y11, Y12, Y13) into low 8 lanes.
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    // Y mask for p1 (byte offsets 8..23 of input → indexed 0..15 in p1):
    // extract 8 Y bytes (Y20, Y21, Y22, Y23, Y30, Y31, Y32, Y33) into
    // low 8 lanes. Block 2 starts at input byte 12 = p1 byte 4.
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    // UV mask for p0: U0 → lane 0, U1 → lane 1, U2 → lane 2 (input byte
    // 12 = p0 byte 12); V0 → lane 4, V1 → lane 5, V2 → lane 6 (input
    // byte 15 = p0 byte 15). Lanes 3, 7 zeroed (filled by p1 via OR).
    let uv_mask_p0 = _mm_setr_epi8(0, 6, 12, -1, 3, 9, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // UV mask for p1: U3 → lane 3 (input byte 18 = p1 byte 10); V3 →
    // lane 7 (input byte 21 = p1 byte 13). Other lanes zeroed.
    let uv_mask_p1 = _mm_setr_epi8(
      -1, -1, -1, 10, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    );

    // Chroma fan‑out masks: each i16 chroma value (2 bytes) replicated
    // into 4 adjacent i16 lanes. dup_lo covers chroma[0..2] (Y[0..8]
    // pixels); dup_hi covers chroma[2..4] (Y[8..16] pixels).
    let dup_lo_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3);
    let dup_hi_mask = _mm_setr_epi8(4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7);

    let mut x = 0usize;
    while x + 16 <= width {
      let block = (x / 4) * 6; // input byte offset of first block in this iter
      // Two overlapping 16-byte loads cover the 24-byte / 4-block window.
      let p0 = _mm_loadu_si128(packed.as_ptr().add(block).cast());
      let p1 = _mm_loadu_si128(packed.as_ptr().add(block + 8).cast());

      // 16 Y bytes: low 8 from p0, high 8 from p1.
      let y_p0 = _mm_shuffle_epi8(p0, y_mask_p0);
      let y_p1 = _mm_shuffle_epi8(p1, y_mask_p1);
      let y_vec = _mm_unpacklo_epi64(y_p0, y_p1);

      // 4 U + 4 V bytes packed into low 8 lanes of one register.
      let uv_p0 = _mm_shuffle_epi8(p0, uv_mask_p0);
      let uv_p1 = _mm_shuffle_epi8(p1, uv_mask_p1);
      let uv = _mm_or_si128(uv_p0, uv_p1);
      // u_packed has U0..U3 in low 4 lanes (other lanes don't‑care);
      // v_packed has V0..V3 in low 4 lanes.
      let u_packed = uv;
      let v_packed = _mm_srli_si128::<4>(uv);

      // Widen 4 chroma bytes → i32x4, subtract 128, scale.
      let u_i32 = _mm_sub_epi32(_mm_cvtepu8_epi32(u_packed), _mm_set1_epi32(128));
      let v_i32 = _mm_sub_epi32(_mm_cvtepu8_epi32(v_packed), _mm_set1_epi32(128));
      let u_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_i32, c_scale_v), rnd_v));
      let v_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_i32, c_scale_v), rnd_v));

      // Per channel: (cu * u_d + cv * v_d + RND) >> 15 → i16 in 4 lanes.
      // The high 4 lanes of the i32x4 are don't‑care; we pack them via
      // `_mm_packs_epi32(x, x)` to get i16x8 with chroma[0..4] in low
      // 4 i16 lanes.
      let r_lo = _mm_srai_epi32::<15>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(cru, u_d), _mm_mullo_epi32(crv, v_d)),
        rnd_v,
      ));
      let g_lo = _mm_srai_epi32::<15>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(cgu, u_d), _mm_mullo_epi32(cgv, v_d)),
        rnd_v,
      ));
      let b_lo = _mm_srai_epi32::<15>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(cbu, u_d), _mm_mullo_epi32(cbv, v_d)),
        rnd_v,
      ));
      // Pack to i16x8: low 4 i16 lanes hold the 4 chroma values; high
      // 4 are duplicates we don't read.
      let r_chroma = _mm_packs_epi32(r_lo, r_lo);
      let g_chroma = _mm_packs_epi32(g_lo, g_lo);
      let b_chroma = _mm_packs_epi32(b_lo, b_lo);

      // 1 → 4 chroma fan‑out: each chroma i16 replicated into 4
      // adjacent i16 lanes. dup_lo holds 2 chroma replicated 4× →
      // 8 lanes covering Y[0..8]; dup_hi covers Y[8..16].
      let r_dup_lo = _mm_shuffle_epi8(r_chroma, dup_lo_mask);
      let r_dup_hi = _mm_shuffle_epi8(r_chroma, dup_hi_mask);
      let g_dup_lo = _mm_shuffle_epi8(g_chroma, dup_lo_mask);
      let g_dup_hi = _mm_shuffle_epi8(g_chroma, dup_hi_mask);
      let b_dup_lo = _mm_shuffle_epi8(b_chroma, dup_lo_mask);
      let b_dup_hi = _mm_shuffle_epi8(b_chroma, dup_hi_mask);

      // Y path identical to packed_yuv_8bit.
      let y_low_i16 = _mm_cvtepu8_epi16(y_vec);
      let y_high_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo16 = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi16 = _mm_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo16 = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi16 = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo16 = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi16 = _mm_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = _mm_packus_epi16(b_lo16, b_hi16);
      let g_u8 = _mm_packus_epi16(g_lo16, g_hi16);
      let r_u8 = _mm_packus_epi16(r_lo16, r_hi16);

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail for `width % 16 != 0` (width is already a multiple
    // of 4 so the tail is also a multiple of 4 pixels).
    if x < width {
      let tail_block = (x / 4) * 6;
      let tail_packed = &packed[tail_block..(width / 4) * 6];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::uyyvyy411_to_rgba_row(tail_packed, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::uyyvyy411_to_rgb_row(tail_packed, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- Packed YUV 4:1:1 (8-bit) → HSV (staged via a reused RGB chunk) --
//
// The SIMD twin of the scalar `uyyvyy411_to_hsv_row` kernel. Reuses the
// LOCAL packed-family driver `packed_hsv_via_rgb_chunks` (defined in the
// sibling `packed_yuv_8bit` module) to fill a small reused RGB scratch
// via the EXISTING SSE4.1 `uyyvyy411_to_rgb_row` kernel, then runs this
// backend's `rgb_to_hsv_row` on the chunk. Byte-identical to
// `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` within this tier with no
// source-width RGB allocation. `HSV_CHUNK` is a multiple of 4, so every
// chunk offset lands on a 6-byte / 4-pixel block boundary.

/// SSE4.1: UYYVYY411 (4:1:1) → planar HSV bytes (OpenCV encoding),
/// staged via the reused-RGB-chunk pattern over this backend's
/// [`uyyvyy411_to_rgb_row`] + `rgb_to_hsv_row`. Byte-identical to
/// `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` within this tier.
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn uyyvyy411_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the shared chunk driver forwards the per-chunk
  // sub-slices to this backend's UYYVYY411 RGB kernel under the same
  // contract. The packed byte offset for the chunk at pixel `offset` (a
  // multiple of 4) is `offset * 3 / 2` (6 bytes per 4-pixel block).
  unsafe {
    super::packed_yuv_8bit::packed_hsv_via_rgb_chunks(
      h_out,
      s_out,
      v_out,
      width,
      |offset, n, rgb| {
        uyyvyy411_to_rgb_row(&packed[offset * 3 / 2..], rgb, n, matrix, full_range);
      },
    );
  }
}

/// SSE4.1 UYYVYY411 → 8-bit luma extraction. Y bytes live at offsets
/// 1, 2, 4, 5 of each 6-byte block. 16 px / iter.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`, `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn uyyvyy411_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      let block = (x / 4) * 6;
      let p0 = _mm_loadu_si128(packed.as_ptr().add(block).cast());
      let p1 = _mm_loadu_si128(packed.as_ptr().add(block + 8).cast());
      let y_p0 = _mm_shuffle_epi8(p0, y_mask_p0);
      let y_p1 = _mm_shuffle_epi8(p1, y_mask_p1);
      let y_vec = _mm_unpacklo_epi64(y_p0, y_p1);
      _mm_storeu_si128(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 16;
    }
    if x < width {
      let tail_block = (x / 4) * 6;
      scalar::uyyvyy411_to_luma_row(
        &packed[tail_block..(width / 4) * 6],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

/// SSE4.1 UYYVYY411 → u16 luma extraction (zero-extended Y bytes). 16
/// px / iter.
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_luma_row`] with `out.len() >= width`
/// `u16` elements.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn uyyvyy411_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(out.len() >= width);

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      let block = (x / 4) * 6;
      let p0 = _mm_loadu_si128(packed.as_ptr().add(block).cast());
      let p1 = _mm_loadu_si128(packed.as_ptr().add(block + 8).cast());
      let y_p0 = _mm_shuffle_epi8(p0, y_mask_p0);
      let y_p1 = _mm_shuffle_epi8(p1, y_mask_p1);
      // Zero-extend low 8 of each shuffle result to u16x8 and store.
      let lo = _mm_cvtepu8_epi16(y_p0);
      let hi = _mm_cvtepu8_epi16(y_p1);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), lo);
      _mm_storeu_si128(out.as_mut_ptr().add(x + 8).cast(), hi);
      x += 16;
    }
    if x < width {
      let tail_block = (x / 4) * 6;
      scalar::uyyvyy411_to_luma_u16_row(
        &packed[tail_block..(width / 4) * 6],
        &mut out[x..width],
        width - x,
      );
    }
  }
}
