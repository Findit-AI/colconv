//! AVX2 kernels for the Tier 5.25 packed YUV 4:1:1 source (UYYVYY411).
//!
//! Per‑block layout (6 bytes / 4 pixels): `[U, Y0, Y1, V, Y2, Y3]`.
//! Each (U, V) chroma pair is shared by 4 adjacent luma samples
//! (1 → 4 horizontal chroma fan‑out).
//!
//! ## Per‑iter pipeline (32 px / 48 input bytes)
//!
//! 1. Four 16‑byte loads at offsets 0, 8, 24, 32 (relative to the
//!    iter's first block). The `+0` / `+8` pair covers blocks 0..3
//!    (16 px); the `+24` / `+32` pair covers blocks 4..7 (next 16 px).
//!    Loop bound `x + 32 <= width` plus the `packed.len() >= width *
//!    3 / 2` contract guarantee 48 readable bytes.
//! 2. Per 16‑px window, two `_mm_shuffle_epi8` calls extract 8 Y bytes
//!    from each load; concatenate via `_mm_unpacklo_epi64` for 16 Y
//!    bytes. Combined into a 32‑byte AVX2 vector.
//! 3. Per 16‑px window, two `_mm_shuffle_epi8` + OR extract 4 U + 4 V
//!    bytes. Combined across the two 16‑px windows for 8 U + 8 V.
//! 4. Widen 8 U / 8 V → i32x8 each, run Q15 chroma math producing 8
//!    i16 chroma values per channel.
//! 5. Fan each of 8 chroma i16 to 4 adjacent lanes (1 → 4 upsample) via
//!    `_mm256_permutevar8x32_epi32` + `_mm256_shuffle_epi8`, yielding
//!    two i16x16 chroma vectors covering the 32 Y pixels.
//! 6. Standard `scale_y` + saturating add + `narrow_u8x32` →
//!    `write_rgb_32` / `write_rgba_32`.
//! 7. Scalar tail for `width % 32 != 0`.

use super::*;

/// AVX2 UYYVYY411 → packed RGB. Semantics match
/// [`scalar::uyyvyy411_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 3 == 0` (4:1:1 chroma group).
/// 3. `packed.len() >= width * 3 / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyyvyy411_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 UYYVYY411 → packed RGBA (alpha = 0xFF).
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_rgb_row`] with `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyyvyy411_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// Generic UYYVYY411 → RGB / RGBA AVX2 kernel. 32 px / iter.
///
/// # Safety
///
/// Caller has verified AVX2. `packed.len() >= width * 3 / 2`. `width`
/// is a multiple of 4. `out.len() >= bpp * width`.
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    // 16-byte SSE-style masks for the per-16-pixel deinterleave step.
    // These mirror the SSE4.1 4:1:1 kernel's masks 1:1.
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let uv_mask_p0 = _mm_setr_epi8(0, 6, 12, -1, 3, 9, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let uv_mask_p1 = _mm_setr_epi8(
      -1, -1, -1, 10, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    );

    // 1 → 4 chroma fan-out mask. Each 128-bit AVX2 lane (16 bytes,
    // 8 i16 lanes) takes 2 chroma i16 from its low 4 bytes and fans
    // each to 4 adjacent i16 lanes:
    //   bytes 0..1 (chroma a) → i16 lanes 0..3
    //   bytes 2..3 (chroma b) → i16 lanes 4..7
    // Mask is replicated across both 128-bit lanes.
    let dup_mask = _mm256_setr_epi8(
      0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, // lane 0
      0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, // lane 1
    );

    // Cross-lane permute indices (i32 chunks) that arrange 2 chroma
    // values into the low 4 bytes of each 128-bit AVX2 lane:
    //
    // `chroma_low` covers Y[0..16]:
    //   lane 0 chunk 0 = source chunk 0 = [r0, r1]
    //   lane 1 chunk 0 = source chunk 1 = [r2, r3]
    //   (other chunks within each lane don't matter — `dup_mask` only
    //   reads bytes 0..3 per lane.)
    //
    // `chroma_high` covers Y[16..32]:
    //   lane 0 chunk 0 = source chunk 2 = [r4, r5]
    //   lane 1 chunk 0 = source chunk 3 = [r6, r7]
    //
    // Source chroma layout (after the `permute4x64<0xD8>(packs)`
    // fixup below) is `r[0..8]` packed into 8 i16 lanes of the low
    // 128 bits, with the high 128 bits an exact duplicate. So
    // selecting source i32 chunks 0..3 or 4..7 yields the same data.
    let perm_low = _mm256_setr_epi32(0, -1, -1, -1, 1, -1, -1, -1);
    let perm_high = _mm256_setr_epi32(2, -1, -1, -1, 3, -1, -1, -1);

    let mut x = 0usize;
    while x + 32 <= width {
      let block = (x / 4) * 6;
      // Two 16-px / 24-byte windows: blocks 0..3 (offset 0) and blocks
      // 4..7 (offset 24). Each window uses 2 overlapping 16-byte loads
      // at offsets 0 / 8 within the window.
      let p0_a = _mm_loadu_si128(packed.as_ptr().add(block).cast());
      let p0_b = _mm_loadu_si128(packed.as_ptr().add(block + 8).cast());
      let p1_a = _mm_loadu_si128(packed.as_ptr().add(block + 24).cast());
      let p1_b = _mm_loadu_si128(packed.as_ptr().add(block + 32).cast());

      // 16 Y bytes per window: shuffle each load to extract 8 Y bytes
      // into low 8 lanes, then `unpacklo_epi64` concatenates.
      let y_w0 = _mm_unpacklo_epi64(
        _mm_shuffle_epi8(p0_a, y_mask_p0),
        _mm_shuffle_epi8(p0_b, y_mask_p1),
      );
      let y_w1 = _mm_unpacklo_epi64(
        _mm_shuffle_epi8(p1_a, y_mask_p0),
        _mm_shuffle_epi8(p1_b, y_mask_p1),
      );
      // Combine both 16-px windows into one 32-byte AVX2 vector.
      let y_vec = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(y_w0), y_w1);

      // 4 U + 4 V bytes per window. After OR, low 8 bytes hold the
      // 4 U + 4 V samples (U[0..4] in bytes 0..3, V[0..4] in bytes 4..7
      // for window 0; similarly for window 1).
      let uv_w0 = _mm_or_si128(
        _mm_shuffle_epi8(p0_a, uv_mask_p0),
        _mm_shuffle_epi8(p0_b, uv_mask_p1),
      );
      let uv_w1 = _mm_or_si128(
        _mm_shuffle_epi8(p1_a, uv_mask_p0),
        _mm_shuffle_epi8(p1_b, uv_mask_p1),
      );
      // Pack 8 U bytes (4 from each window) into low 8 bytes of one
      // 128-bit register; same for V. `_mm_unpacklo_epi32` interleaves
      // the low 4 bytes of each operand so the result is
      // [U0..U3 from w0, U0..U3 from w1] = [U0..U3, U4..U7].
      let u_packed = _mm_unpacklo_epi32(uv_w0, uv_w1);
      let v_packed = _mm_unpacklo_epi32(_mm_srli_si128::<4>(uv_w0), _mm_srli_si128::<4>(uv_w1));

      // Widen 8 U / 8 V bytes → i32x8 each, subtract 128, scale by
      // c_scale.
      let u_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(u_packed), _mm256_set1_epi32(128));
      let v_i32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(v_packed), _mm256_set1_epi32(128));
      let u_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_i32, c_scale_v),
        rnd_v,
      ));
      let v_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_i32, c_scale_v),
        rnd_v,
      ));

      // (cu * u_d + cv * v_d + RND) >> 15 in i32x8, then pack to i16.
      // `packs_epi32(x, x)` produces per-lane interleaved output;
      // `permute4x64<0xD8>` restores natural order so r[0..8] sit
      // contiguously in the low 128 bits (high 128 bits = duplicate).
      let r_i32 = _mm256_srai_epi32::<15>(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cru, u_d), _mm256_mullo_epi32(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = _mm256_srai_epi32::<15>(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cgu, u_d), _mm256_mullo_epi32(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = _mm256_srai_epi32::<15>(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cbu, u_d), _mm256_mullo_epi32(cbv, v_d)),
        rnd_v,
      ));
      let r_chroma = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(r_i32, r_i32));
      let g_chroma = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(g_i32, g_i32));
      let b_chroma = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(b_i32, b_i32));

      // Fan-out each chroma channel to 32 Y pixels.
      let r_for_lo = _mm256_permutevar8x32_epi32(r_chroma, perm_low);
      let g_for_lo = _mm256_permutevar8x32_epi32(g_chroma, perm_low);
      let b_for_lo = _mm256_permutevar8x32_epi32(b_chroma, perm_low);
      let r_for_hi = _mm256_permutevar8x32_epi32(r_chroma, perm_high);
      let g_for_hi = _mm256_permutevar8x32_epi32(g_chroma, perm_high);
      let b_for_hi = _mm256_permutevar8x32_epi32(b_chroma, perm_high);
      let r_dup_lo = _mm256_shuffle_epi8(r_for_lo, dup_mask);
      let g_dup_lo = _mm256_shuffle_epi8(g_for_lo, dup_mask);
      let b_dup_lo = _mm256_shuffle_epi8(b_for_lo, dup_mask);
      let r_dup_hi = _mm256_shuffle_epi8(r_for_hi, dup_mask);
      let g_dup_hi = _mm256_shuffle_epi8(g_for_hi, dup_mask);
      let b_dup_hi = _mm256_shuffle_epi8(b_for_hi, dup_mask);

      // Y path identical to packed_yuv_8bit.
      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail.
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
// via the EXISTING AVX2 `uyyvyy411_to_rgb_row` kernel, then runs this
// backend's `rgb_to_hsv_row` on the chunk. Byte-identical to
// `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` within this tier with no
// source-width RGB allocation. `HSV_CHUNK` is a multiple of 4, so every
// chunk offset lands on a 6-byte / 4-pixel block boundary.

/// AVX2: UYYVYY411 (4:1:1) → planar HSV bytes (OpenCV encoding),
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
#[target_feature(enable = "avx2")]
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

/// AVX2 UYYVYY411 → 8-bit luma extraction. 32 px / iter.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`, `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyyvyy411_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 32 <= width {
      let block = (x / 4) * 6;
      let p0_a = _mm_loadu_si128(packed.as_ptr().add(block).cast());
      let p0_b = _mm_loadu_si128(packed.as_ptr().add(block + 8).cast());
      let p1_a = _mm_loadu_si128(packed.as_ptr().add(block + 24).cast());
      let p1_b = _mm_loadu_si128(packed.as_ptr().add(block + 32).cast());

      let y_w0 = _mm_unpacklo_epi64(
        _mm_shuffle_epi8(p0_a, y_mask_p0),
        _mm_shuffle_epi8(p0_b, y_mask_p1),
      );
      let y_w1 = _mm_unpacklo_epi64(
        _mm_shuffle_epi8(p1_a, y_mask_p0),
        _mm_shuffle_epi8(p1_b, y_mask_p1),
      );
      let y_vec = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(y_w0), y_w1);
      _mm256_storeu_si256(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 32;
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

/// AVX2 UYYVYY411 → u16 luma extraction (zero-extended Y bytes). 32 px /
/// iter.
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_luma_row`] with `out.len() >= width`
/// `u16` elements.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn uyyvyy411_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 32 <= width {
      let block = (x / 4) * 6;
      let p0_a = _mm_loadu_si128(packed.as_ptr().add(block).cast());
      let p0_b = _mm_loadu_si128(packed.as_ptr().add(block + 8).cast());
      let p1_a = _mm_loadu_si128(packed.as_ptr().add(block + 24).cast());
      let p1_b = _mm_loadu_si128(packed.as_ptr().add(block + 32).cast());

      let y_w0 = _mm_unpacklo_epi64(
        _mm_shuffle_epi8(p0_a, y_mask_p0),
        _mm_shuffle_epi8(p0_b, y_mask_p1),
      );
      let y_w1 = _mm_unpacklo_epi64(
        _mm_shuffle_epi8(p1_a, y_mask_p0),
        _mm_shuffle_epi8(p1_b, y_mask_p1),
      );
      // Zero-extend each 16-byte Y vector to 16 u16 = 32 bytes via
      // `_mm256_cvtepu8_epi16`.
      let w_lo = _mm256_cvtepu8_epi16(y_w0);
      let w_hi = _mm256_cvtepu8_epi16(y_w1);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), w_lo);
      _mm256_storeu_si256(out.as_mut_ptr().add(x + 16).cast(), w_hi);
      x += 32;
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
