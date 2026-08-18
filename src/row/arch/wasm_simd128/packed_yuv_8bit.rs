use core::arch::wasm32::*;

use super::*;

/// wasm simd128 YUYV422 → packed RGB. Semantics match
/// [`scalar::yuyv422_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `packed.len() >= 2 * width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuyv422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: simd128 is compile-time enabled; caller obligation per docs.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 YUYV422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuyv422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 UYVY422 → packed RGB.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyvy422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 UYVY422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyvy422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 YVYU422 → packed RGB.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yvyu422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 YVYU422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yvyu422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// Generic packed YUV 4:2:2 → RGB / RGBA wasm simd128 kernel.
/// 16 px / iter; deinterleaves bytes via two `u8x16_swizzle` per
/// vector then combines across two vectors via `i8x16_shuffle`.
///
/// # Safety
///
/// `simd128` enabled at compile time. `packed.len() >= 2 * width`.
/// `width` even. `out.len() >= bpp * width`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn yuv422_packed_to_rgb_or_rgba_row<
  const Y_LSB: bool,
  const SWAP_UV: bool,
  const ALPHA: bool,
>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let mid128 = i16x8_splat(128);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let alpha_u8 = u8x16_splat(0xFF);

    // Per-block split mask for `u8x16_swizzle`: rearrange a 16-byte
    // chunk (4 blocks × 4 bytes) so Y bytes land in low 8 lanes and
    // chroma bytes in high 8 lanes.
    let split_mask = if Y_LSB {
      u8x16(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)
    } else {
      u8x16(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14)
    };
    // Chroma split mask: same shape applied to a 16-byte chroma
    // vector to put evens in low 8 lanes, odds in high 8 lanes.
    let chroma_split_mask = u8x16(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = v128_load(packed.as_ptr().add(x * 2).cast());
      let p1 = v128_load(packed.as_ptr().add(x * 2 + 16).cast());

      // Per-vector split: low 8 bytes = Y of that 16-byte half,
      // high 8 bytes = chroma of that half.
      let p0s = u8x16_swizzle(p0, split_mask);
      let p1s = u8x16_swizzle(p1, split_mask);

      // Combine across two vectors:
      // y_vec = low 8 from p0s + low 8 from p1s (= 16 Y bytes total).
      // chroma_vec = high 8 from p0s + high 8 from p1s.
      let y_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(p0s, p1s);
      let chroma_vec =
        i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(p0s, p1s);

      // Split chroma evens / odds: low 8 = c_evens, high 8 = c_odds.
      let cs = u8x16_swizzle(chroma_vec, chroma_split_mask);
      // u_vec / v_vec extracted as zero-extended u16x8 vectors. The
      // existing yuv_420 wasm reads u_half via `u16x8_load_extend_u8x8`
      // (memory load). Here we widen from in-register bytes via
      // `u16x8_extend_low_u8x16` (low 8 bytes → u16x8) or `_high_`.
      let u_i16_zero = if SWAP_UV {
        u16x8_extend_high_u8x16(cs) // c_odds = U
      } else {
        u16x8_extend_low_u8x16(cs) // c_evens = U
      };
      let v_i16_zero = if SWAP_UV {
        u16x8_extend_low_u8x16(cs)
      } else {
        u16x8_extend_high_u8x16(cs)
      };

      // From here, math byte-identical to yuv_420's wasm kernel.
      let u_i16 = i16x8_sub(u_i16_zero, mid128);
      let v_i16 = i16x8_sub(v_i16_zero, mid128);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = dup_lo(r_chroma);
      let r_dup_hi = dup_hi(r_chroma);
      let g_dup_lo = dup_lo(g_chroma);
      let g_dup_hi = dup_hi(g_chroma);
      let b_dup_lo = dup_lo(b_chroma);
      let b_dup_hi = dup_hi(b_chroma);

      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = i16x8_add_sat(y_scaled_lo, b_dup_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_dup_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_dup_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_dup_hi);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_dup_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_dup_hi);

      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        if Y_LSB && !SWAP_UV {
          scalar::yuyv422_to_rgba_row(tail_packed, tail_out, tail_w, matrix, full_range);
        } else if !Y_LSB && !SWAP_UV {
          scalar::uyvy422_to_rgba_row(tail_packed, tail_out, tail_w, matrix, full_range);
        } else {
          scalar::yvyu422_to_rgba_row(tail_packed, tail_out, tail_w, matrix, full_range);
        }
      } else if Y_LSB && !SWAP_UV {
        scalar::yuyv422_to_rgb_row(tail_packed, tail_out, tail_w, matrix, full_range);
      } else if !Y_LSB && !SWAP_UV {
        scalar::uyvy422_to_rgb_row(tail_packed, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yvyu422_to_rgb_row(tail_packed, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- Packed YUV 4:2:2 (8-bit) → HSV (staged via a reused RGB chunk) --
//
// The SIMD twins of the scalar `*_to_hsv_row` kernels. Rather than
// re-derive an HSV-specific register pipeline, each fills a small fixed
// reused RGB scratch (one `HSV_CHUNK`-pixel chunk at a time) using the
// EXISTING wasm simd128 packed-4:2:2 RGB kernel — so the chunk filler IS the
// production RGB kernel — then runs this backend's `rgb_to_hsv_row` on
// the chunk. This keeps the per-format SIMD surface tiny (only the
// chunked driver is new) and makes the result byte-identical to
// `rgb_to_hsv_row(*_to_rgb_row(...))` within this tier. The scalar tail
// of each underlying RGB kernel handles widths below the SIMD block, so
// no separate tail is needed here.
//
// This driver is LOCAL to the packed family (the shared
// `yuv_to_hsv_via_rgb_chunks` is gated on `yuv-planar`; the packed
// formats compile under `yuv-packed` alone) and shared by both packed
// files of this arch — the sibling 4:1:1 module reaches it via
// `super::packed_yuv_8bit`. `HSV_CHUNK = 64` is even AND a multiple of 4,
// so every chunk offset lands on a 4:2:2 (4-byte) AND a 4:1:1 (6-byte)
// block boundary.

/// One reused RGB chunk's worth of pixels staged before the HSV pass.
pub(super) const HSV_CHUNK: usize = 64;

/// Shared driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing RGB kernel
/// for the format, passed the chunk `offset` and length `n`), then runs
/// [`rgb_to_hsv_row`] on that chunk into the H/S/V planes. Byte-identical
/// to `rgb_to_hsv_row(*_to_rgb_row(...))` within this tier, with no
/// source-width RGB allocation. Shared by the packed 4:2:2 kernels here
/// and the 4:1:1 kernel in the sibling module.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// The SIMD feature must be available, and `fill_rgb` must uphold the
/// underlying RGB kernel's safety contract for each chunk. Each of
/// `h_out` / `s_out` / `v_out` must be `>= width`.
#[inline]
pub(super) unsafe fn packed_hsv_via_rgb_chunks(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  mut fill_rgb: impl FnMut(usize, usize, &mut [u8]),
) {
  let mut scratch = [0u8; HSV_CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(HSV_CHUNK);
    fill_rgb(offset, n, &mut scratch[..n * 3]);
    // SAFETY: SIMD verified by the wrapper's `#[target_feature]`; the
    // chunk and the output sub-slices are all length `n`.
    unsafe {
      rgb_to_hsv_row(
        &scratch[..n * 3],
        &mut h_out[offset..offset + n],
        &mut s_out[offset..offset + n],
        &mut v_out[offset..offset + n],
        n,
      );
    }
    offset += n;
  }
}

/// wasm simd128: YUYV422 (4:2:2) → planar HSV bytes (OpenCV encoding), staged
/// via the reused-RGB-chunk pattern over this backend's
/// [`yuyv422_to_rgb_row`] + [`rgb_to_hsv_row`]. Byte-identical to
/// `rgb_to_hsv_row(yuyv422_to_rgb_row(...))` within this tier.
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `width & 1 == 0`.
/// 3. `packed.len() >= 2 * width`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuyv422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the chunk filler forwards the per-chunk
  // sub-slices to this backend's YUYV422 RGB kernel under the same
  // contract. The packed byte offset for the chunk at pixel `offset` is
  // `offset * 2` (2 bytes per pixel).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuyv422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// wasm simd128: UYVY422 (4:2:2) → planar HSV bytes, staged via this backend's
/// [`uyvy422_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`yuyv422_to_hsv_row`].
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn uyvy422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; forwards to this backend's UYVY422 RGB kernel
  // under the same contract (packed byte offset = `offset * 2`).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      uyvy422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// wasm simd128: YVYU422 (4:2:2) → planar HSV bytes, staged via this backend's
/// [`yvyu422_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`yuyv422_to_hsv_row`].
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yvyu422_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; forwards to this backend's YVYU422 RGB kernel
  // under the same contract (packed byte offset = `offset * 2`).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yvyu422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// wasm simd128 YUYV422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuyv422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

/// wasm simd128 UYVY422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyvy422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_luma_row::<false>(packed, luma_out, width);
  }
}

/// wasm simd128 YVYU422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yvyu422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

/// wasm simd128 YUYV422 → u16 luma extraction. Block size: 16 px / iter.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuyv422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_luma_u16_row::<true>(packed, out, width);
  }
}

/// wasm simd128 UYVY422 → u16 luma extraction. Block size: 16 px / iter.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyvy422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_luma_u16_row::<false>(packed, out, width);
  }
}

/// wasm simd128 YVYU422 → u16 luma extraction (Y positions same as YUYV).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yvyu422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    yuv422_packed_to_luma_u16_row::<true>(packed, out, width);
  }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn yuv422_packed_to_luma_u16_row<const Y_LSB: bool>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    let split_mask = if Y_LSB {
      u8x16(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)
    } else {
      u8x16(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14)
    };

    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = v128_load(packed.as_ptr().add(x * 2).cast());
      let p1 = v128_load(packed.as_ptr().add(x * 2 + 16).cast());
      let p0s = u8x16_swizzle(p0, split_mask);
      let p1s = u8x16_swizzle(p1, split_mask);
      // Combine low 8 bytes from p0s and p1s into one 16-byte vector
      // containing the 16 packed Y bytes, then zero-extend low/high halves.
      let y_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(p0s, p1s);
      let low = u16x8_extend_low_u8x16(y_vec);
      let high = u16x8_extend_high_u8x16(y_vec);
      v128_store(out.as_mut_ptr().add(x).cast(), low);
      v128_store(out.as_mut_ptr().add(x + 8).cast(), high);
      x += 16;
    }
    if x < width {
      if Y_LSB {
        scalar::yuyv422_to_luma_u16_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
      } else {
        scalar::uyvy422_to_luma_u16_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
      }
    }
  }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn yuv422_packed_to_luma_row<const Y_LSB: bool>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    let split_mask = if Y_LSB {
      u8x16(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)
    } else {
      u8x16(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14)
    };

    let mut x = 0usize;
    while x + 16 <= width {
      let p0 = v128_load(packed.as_ptr().add(x * 2).cast());
      let p1 = v128_load(packed.as_ptr().add(x * 2 + 16).cast());
      let p0s = u8x16_swizzle(p0, split_mask);
      let p1s = u8x16_swizzle(p1, split_mask);
      let y_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(p0s, p1s);
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 16;
    }
    if x < width {
      if Y_LSB {
        scalar::yuyv422_to_luma_row(
          &packed[x * 2..width * 2],
          &mut luma_out[x..width],
          width - x,
        );
      } else {
        scalar::uyvy422_to_luma_row(
          &packed[x * 2..width * 2],
          &mut luma_out[x..width],
          width - x,
        );
      }
    }
  }
}
