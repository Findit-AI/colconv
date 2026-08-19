//! wasm simd128 kernels for the Tier 5.25 packed YUV 4:1:1 source
//! (UYYVYY411).
//!
//! Per‑block layout (6 bytes / 4 pixels): `[U, Y0, Y1, V, Y2, Y3]`.
//! Each (U, V) chroma pair is shared by 4 adjacent luma samples
//! (1 → 4 horizontal chroma fan‑out).
//!
//! ## Per‑iter pipeline (16 px / 24 input bytes)
//!
//! 1. Two overlapping `v128_load` reads at offsets 0 and 8 cover the
//!    24‑byte / 4‑block window. Loop bound `x + 16 <= width` plus the
//!    `packed.len() >= width * 3 / 2` contract guarantee 24 readable
//!    bytes.
//! 2. `u8x16_swizzle` extracts Y / UV bytes via per‑vector masks
//!    (mirroring the SSE4.1 strategy).
//! 3. Standard Q15 chroma math producing 4 i16 chroma values per
//!    channel; fan each to 4 adjacent lanes (1 → 4 upsample) via
//!    `u8x16_swizzle` with i16‑replication tables.
//! 4. Standard `scale_y` + saturating add + `u8x16_narrow_i16x8` →
//!    `write_rgb_16` / `write_rgba_16`.
//! 5. Scalar tail for `width % 16 != 0`.

use core::arch::wasm32::*;

use super::*;

/// wasm simd128 UYYVYY411 → packed RGB. Semantics match
/// [`scalar::uyyvyy411_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 3 == 0` (4:1:1 chroma group).
/// 3. `packed.len() >= width * 3 / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyyvyy411_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: simd128 is compile-time enabled; caller obligation per docs.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// wasm simd128 UYYVYY411 → packed RGBA (alpha = 0xFF).
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_rgb_row`] with `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyyvyy411_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// Generic UYYVYY411 → RGB / RGBA wasm simd128 kernel. 16 px / iter.
///
/// # Safety
///
/// `simd128` enabled at compile time. `packed.len() >= width * 3 / 2`.
/// `width` is a multiple of 4. `out.len() >= bpp * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let alpha_u8 = u8x16_splat(0xFF);

    // Per-vector swizzle masks mirroring the SSE4.1 4:1:1 kernel.
    let y_mask_p0 = u8x16(
      1, 2, 4, 5, 7, 8, 10, 11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    );
    let y_mask_p1 = u8x16(
      5, 6, 8, 9, 11, 12, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    );
    let uv_mask_p0 = u8x16(
      0, 6, 12, 0xFF, 3, 9, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    );
    let uv_mask_p1 = u8x16(
      0xFF, 0xFF, 0xFF, 10, 0xFF, 0xFF, 0xFF, 13, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    );

    // 1 → 4 chroma fan-out: each i16 chroma value (2 bytes) replicated
    // into 4 adjacent i16 lanes.
    //   dup_lo_mask: chroma i16 lanes 0, 1 → output i16 lanes 0..3, 4..7
    //   dup_hi_mask: chroma i16 lanes 2, 3 → output i16 lanes 0..3, 4..7
    let dup_lo_mask = u8x16(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3);
    let dup_hi_mask = u8x16(4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7);

    let mut x = 0usize;
    while x + 16 <= width {
      let block = (x / 4) * 6;
      let p0 = v128_load(packed.as_ptr().add(block).cast());
      let p1 = v128_load(packed.as_ptr().add(block + 8).cast());

      // 16 Y bytes: 8 from p0 + 8 from p1, combined via i8x16_shuffle.
      let y_p0 = u8x16_swizzle(p0, y_mask_p0);
      let y_p1 = u8x16_swizzle(p1, y_mask_p1);
      let y_vec =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_p0, y_p1);

      // 4 U + 4 V bytes packed into low 8 lanes via OR.
      let uv_p0 = u8x16_swizzle(p0, uv_mask_p0);
      let uv_p1 = u8x16_swizzle(p1, uv_mask_p1);
      let uv = v128_or(uv_p0, uv_p1);
      // Extract u_packed (low 4 bytes) and v_packed (next 4 bytes) as
      // u8 vectors. Use `u16x8_extend_low_u8x16` after isolating low 4
      // bytes via swizzle / extend.
      let u_packed8 = u16x8_extend_low_u8x16(uv); // low 8 u8 → 8 u16
      // The first 4 i16 lanes of u_packed8 hold U[0..4]; lanes 4..7 hold
      // V[0..4]. We want u_i32 = i32x4 of U[0..4] and v_i32 = i32x4 of
      // V[0..4]. Splitting the low 4 i16 lanes vs high 4:
      let u_i32 = i32x4_extend_low_i16x8(u_packed8); // U[0..4] as i32x4
      let v_i32 = i32x4_extend_high_i16x8(u_packed8); // V[0..4] as i32x4
      let u_i32 = i32x4_sub(u_i32, i32x4_splat(128));
      let v_i32 = i32x4_sub(v_i32, i32x4_splat(128));
      let u_d = q15_shift(i32x4_add(i32x4_mul(u_i32, c_scale_v), rnd_v));
      let v_d = q15_shift(i32x4_add(i32x4_mul(v_i32, c_scale_v), rnd_v));

      // Per-channel chroma in i32x4 → narrow-saturate to i16.
      let r_i32 = i32x4_shr(
        i32x4_add(i32x4_add(i32x4_mul(cru, u_d), i32x4_mul(crv, v_d)), rnd_v),
        15,
      );
      let g_i32 = i32x4_shr(
        i32x4_add(i32x4_add(i32x4_mul(cgu, u_d), i32x4_mul(cgv, v_d)), rnd_v),
        15,
      );
      let b_i32 = i32x4_shr(
        i32x4_add(i32x4_add(i32x4_mul(cbu, u_d), i32x4_mul(cbv, v_d)), rnd_v),
        15,
      );
      // Pack i32x4 → i16x8 with chroma values in low 4 i16 lanes (high
      // 4 are duplicates).
      let r_chroma = i16x8_narrow_i32x4(r_i32, r_i32);
      let g_chroma = i16x8_narrow_i32x4(g_i32, g_i32);
      let b_chroma = i16x8_narrow_i32x4(b_i32, b_i32);

      // Fan-out via swizzle.
      let r_dup_lo = u8x16_swizzle(r_chroma, dup_lo_mask);
      let r_dup_hi = u8x16_swizzle(r_chroma, dup_hi_mask);
      let g_dup_lo = u8x16_swizzle(g_chroma, dup_lo_mask);
      let g_dup_hi = u8x16_swizzle(g_chroma, dup_hi_mask);
      let b_dup_lo = u8x16_swizzle(b_chroma, dup_lo_mask);
      let b_dup_hi = u8x16_swizzle(b_chroma, dup_hi_mask);

      // Y path identical to packed_yuv_8bit.
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
// via the EXISTING wasm simd128 `uyyvyy411_to_rgb_row` kernel, then runs this
// backend's `rgb_to_hsv_row` on the chunk. Byte-identical to
// `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` within this tier with no
// source-width RGB allocation. `HSV_CHUNK` is a multiple of 4, so every
// chunk offset lands on a 6-byte / 4-pixel block boundary.

/// wasm simd128: UYYVYY411 (4:1:1) → planar HSV bytes (OpenCV encoding),
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
#[target_feature(enable = "simd128")]
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

/// wasm simd128 UYYVYY411 → 8-bit luma extraction. 16 px / iter.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`, `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyyvyy411_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    let y_mask_p0 = u8x16(
      1, 2, 4, 5, 7, 8, 10, 11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    );
    let y_mask_p1 = u8x16(
      5, 6, 8, 9, 11, 12, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    );

    let mut x = 0usize;
    while x + 16 <= width {
      let block = (x / 4) * 6;
      let p0 = v128_load(packed.as_ptr().add(block).cast());
      let p1 = v128_load(packed.as_ptr().add(block + 8).cast());
      let y_p0 = u8x16_swizzle(p0, y_mask_p0);
      let y_p1 = u8x16_swizzle(p1, y_mask_p1);
      let y_vec =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_p0, y_p1);
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_vec);
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

/// wasm simd128 UYYVYY411 → u16 luma extraction (zero-extended Y bytes).
/// 16 px / iter.
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_luma_row`] with `out.len() >= width`
/// `u16` elements.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyyvyy411_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(out.len() >= width);

  // SAFETY: simd128 is compile-time enabled.
  unsafe {
    let y_mask_p0 = u8x16(
      1, 2, 4, 5, 7, 8, 10, 11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    );
    let y_mask_p1 = u8x16(
      5, 6, 8, 9, 11, 12, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    );

    let mut x = 0usize;
    while x + 16 <= width {
      let block = (x / 4) * 6;
      let p0 = v128_load(packed.as_ptr().add(block).cast());
      let p1 = v128_load(packed.as_ptr().add(block + 8).cast());
      let y_p0 = u8x16_swizzle(p0, y_mask_p0);
      let y_p1 = u8x16_swizzle(p1, y_mask_p1);
      // Each shuffle has 8 valid Y bytes in low 8 lanes; zero-extend
      // to u16x8 and store.
      let lo = u16x8_extend_low_u8x16(y_p0);
      let hi = u16x8_extend_low_u8x16(y_p1);
      v128_store(out.as_mut_ptr().add(x).cast(), lo);
      v128_store(out.as_mut_ptr().add(x + 8).cast(), hi);
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
