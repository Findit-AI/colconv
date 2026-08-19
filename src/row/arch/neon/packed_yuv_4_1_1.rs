//! NEON kernels for the Tier 5.25 packed YUV 4:1:1 source (UYYVYY411).
//!
//! Per‑block layout (6 bytes / 4 pixels): `[U, Y0, Y1, V, Y2, Y3]`.
//! Each (U, V) chroma pair is shared by 4 adjacent luma samples
//! (1 → 4 horizontal chroma fan‑out).
//!
//! ## Per‑iter pipeline (32 px / 48 input bytes)
//!
//! 1. `vld3q_u8` reads 48 bytes and deinterleaves into three 16‑byte
//!    lanes covering 8 6‑byte blocks:
//!    - lane 0: `[U0, V0, U1, V1, …, U7, V7]` — UV pairs interleaved
//!    - lane 1: `[Y0_0, Y0_2, Y1_0, Y1_2, …, Y7_0, Y7_2]` — Y at
//!      block‑offsets 1, 4
//!    - lane 2: `[Y0_1, Y0_3, Y1_1, Y1_3, …, Y7_1, Y7_3]` — Y at
//!      block‑offsets 2, 5
//! 2. `vzip1q_u8(l1, l2)` produces Y[0..16] in natural order
//!    (Y0_0, Y0_1, Y0_2, Y0_3, Y1_0, …, Y3_3); `vzip2q_u8` produces
//!    Y[16..32].
//! 3. `vuzp_u8(low(l0), high(l0))` splits the 16-byte UV-interleaved
//!    lane into 8 U and 8 V bytes.
//! 4. Standard Q15 chroma math: widen 8 chroma → i32x4 halves, scale,
//!    apply matrix → i16x8 chroma per channel.
//! 5. Fan 8 chroma values to 32 lanes (1 → 4 upsample) via two
//!    `vqtbl1q_u8` calls per channel, yielding two i16x8 chroma
//!    vectors covering the 32 Y pixels (split as low‑16 / high‑16).
//! 6. Standard `scale_y` + saturating add + `vqmovun_s16` →
//!    `vst3q_u8` / `vst4q_u8` interleaved store.
//! 7. Scalar tail for `width % 32 != 0` (multiple of 4).

use crate::{KernelMatrix, row::scalar};

use super::*;

/// NEON UYYVYY411 → packed RGB. Semantics match
/// [`scalar::uyyvyy411_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 3 == 0` (4:1:1 chroma group).
/// 3. `packed.len() >= width * 3 / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyyvyy411_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// NEON UYYVYY411 → packed RGBA (alpha = 0xFF).
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_rgb_row`] with `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyyvyy411_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// Generic UYYVYY411 → RGB / RGBA NEON kernel. 32 px / iter.
///
/// # Safety
///
/// Caller has verified NEON. `packed.len() >= width * 3 / 2`. `width`
/// is a multiple of 4. `out.len() >= bpp * width`.
#[inline]
#[target_feature(enable = "neon")]
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

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let mid128 = vdupq_n_s16(128);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    // Fan‑out tables: each input chroma byte (or i16 lane) replicated
    // 4x into adjacent output lanes via `vqtbl1q_u8`. The tables index
    // into the chroma vector (16 bytes = 8 i16 lanes); each output i16
    // (2 bytes) reads from a fixed 2-byte source.
    //
    // For the low 16 chroma output lanes (covering Y[0..16]):
    //   output i16[0..3]   ← chroma i16[0]
    //   output i16[4..7]   ← chroma i16[1]
    //   output i16[8..11]  ← chroma i16[2]
    //   output i16[12..15] ← chroma i16[3]
    // Each i16 = 2 bytes; output byte 2k reads source byte 2*(k/4),
    // output byte 2k+1 reads source byte 2*(k/4)+1.
    let dup_lo_tbl = vld1q_u8([0u8, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3].as_ptr());
    let dup_hi_tbl = vld1q_u8([4u8, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7].as_ptr());
    // For Y[16..32] chroma output: read i16 lanes 4..7 of source (4x).
    let dup_lo_tbl2 = vld1q_u8([8u8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11].as_ptr());
    let dup_hi_tbl2 = vld1q_u8(
      [
        12u8, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15,
      ]
      .as_ptr(),
    );

    let mut x = 0usize;
    while x + 32 <= width {
      let block = (x / 4) * 6;
      let v3 = vld3q_u8(packed.as_ptr().add(block));
      // v3.0 = UV interleaved (16 bytes); v3.1 = Y at block-offsets 1, 4;
      // v3.2 = Y at block-offsets 2, 5.
      let l0 = v3.0;
      let l1 = v3.1;
      let l2 = v3.2;

      // Y[0..16]  = vzip1(l1, l2) = [l1[0], l2[0], l1[1], l2[1], ...]
      // Y[16..32] = vzip2(l1, l2)
      let y_lo16 = vzip1q_u8(l1, l2);
      let y_hi16 = vzip2q_u8(l1, l2);

      // UV split: 16 UV-interleaved bytes → 8 U + 8 V.
      // `vuzp_u8(a, b)` → (.0 = even bytes of [a||b], .1 = odd bytes).
      // For l0 = [U0, V0, U1, V1, ..., U7, V7]:
      //   take low half (8 bytes) and high half (8 bytes) and unzip.
      let uv_pair = vuzp_u8(vget_low_u8(l0), vget_high_u8(l0));
      let u_vec = uv_pair.0; // 8 U bytes (u8x8)
      let v_vec = uv_pair.1; // 8 V bytes (u8x8)

      // Widen 8 U / 8 V → i16x8 each, subtract 128.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_vec)), mid128);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_vec)), mid128);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // Per-channel chroma i16x8 (8 chroma values, one per chroma pair).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // 1 → 4 chroma fan‑out via byte‑level table lookup. Each
      // `vqtbl1q_u8` indexes into the 16‑byte chroma vector and emits
      // 16 bytes = 8 i16 lanes.
      let r_dup_lo = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(r_chroma), dup_lo_tbl));
      let r_dup_mid = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(r_chroma), dup_hi_tbl));
      let r_dup_3 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(r_chroma), dup_lo_tbl2));
      let r_dup_4 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(r_chroma), dup_hi_tbl2));
      let g_dup_lo = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(g_chroma), dup_lo_tbl));
      let g_dup_mid = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(g_chroma), dup_hi_tbl));
      let g_dup_3 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(g_chroma), dup_lo_tbl2));
      let g_dup_4 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(g_chroma), dup_hi_tbl2));
      let b_dup_lo = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(b_chroma), dup_lo_tbl));
      let b_dup_mid = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(b_chroma), dup_hi_tbl));
      let b_dup_3 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(b_chroma), dup_lo_tbl2));
      let b_dup_4 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(b_chroma), dup_hi_tbl2));

      // Y path identical to packed_yuv_8bit. Need 4 x i16x8 vectors of
      // scaled Y to cover the 32 Y pixels.
      let y_lo_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_lo16)));
      let y_lo_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_lo16)));
      let y_hi_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_hi16)));
      let y_hi_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_hi16)));
      let ys0 = scale_y(y_lo_lo, y_off_v, y_scale_v, rnd_v);
      let ys1 = scale_y(y_lo_hi, y_off_v, y_scale_v, rnd_v);
      let ys2 = scale_y(y_hi_lo, y_off_v, y_scale_v, rnd_v);
      let ys3 = scale_y(y_hi_hi, y_off_v, y_scale_v, rnd_v);

      let r0 = vqaddq_s16(ys0, r_dup_lo);
      let r1 = vqaddq_s16(ys1, r_dup_mid);
      let r2 = vqaddq_s16(ys2, r_dup_3);
      let r3 = vqaddq_s16(ys3, r_dup_4);
      let g0 = vqaddq_s16(ys0, g_dup_lo);
      let g1 = vqaddq_s16(ys1, g_dup_mid);
      let g2 = vqaddq_s16(ys2, g_dup_3);
      let g3 = vqaddq_s16(ys3, g_dup_4);
      let b0 = vqaddq_s16(ys0, b_dup_lo);
      let b1 = vqaddq_s16(ys1, b_dup_mid);
      let b2 = vqaddq_s16(ys2, b_dup_3);
      let b3 = vqaddq_s16(ys3, b_dup_4);

      let r_lo16 = vcombine_u8(vqmovun_s16_compat(r0), vqmovun_s16_compat(r1));
      let r_hi16 = vcombine_u8(vqmovun_s16_compat(r2), vqmovun_s16_compat(r3));
      let g_lo16 = vcombine_u8(vqmovun_s16_compat(g0), vqmovun_s16_compat(g1));
      let g_hi16 = vcombine_u8(vqmovun_s16_compat(g2), vqmovun_s16_compat(g3));
      let b_lo16 = vcombine_u8(vqmovun_s16_compat(b0), vqmovun_s16_compat(b1));
      let b_hi16 = vcombine_u8(vqmovun_s16_compat(b2), vqmovun_s16_compat(b3));

      if ALPHA {
        let rgba_lo = uint8x16x4_t(r_lo16, g_lo16, b_lo16, alpha_u8);
        let rgba_hi = uint8x16x4_t(r_hi16, g_hi16, b_hi16, alpha_u8);
        vst4q_u8(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u8(out.as_mut_ptr().add(x * 4 + 64), rgba_hi);
      } else {
        let rgb_lo = uint8x16x3_t(r_lo16, g_lo16, b_lo16);
        let rgb_hi = uint8x16x3_t(r_hi16, g_hi16, b_hi16);
        vst3q_u8(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u8(out.as_mut_ptr().add(x * 3 + 48), rgb_hi);
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
// via the EXISTING NEON `uyyvyy411_to_rgb_row` kernel, then runs the NEON
// `rgb_to_hsv_row` on the chunk. Byte-identical to
// `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` within the NEON tier with no
// source-width RGB allocation. `HSV_CHUNK` is a multiple of 4, so every
// chunk offset lands on a 6-byte / 4-pixel block boundary.

/// NEON: UYYVYY411 (4:1:1) → planar HSV bytes (OpenCV encoding), staged
/// via the reused-RGB-chunk pattern over the NEON
/// [`uyyvyy411_to_rgb_row`] then the NEON `rgb_to_hsv_row`. Byte-identical
/// to `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` within the NEON tier.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
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

  // SAFETY: NEON verified; the shared chunk driver forwards the per-chunk
  // sub-slices to the NEON UYYVYY411 RGB kernel under the same contract.
  // The packed byte offset for the chunk at pixel `offset` (a multiple of
  // 4) is `offset * 3 / 2` (6 bytes per 4-pixel block).
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

/// NEON UYYVYY411 → 8-bit luma extraction. 32 px / iter.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`, `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyyvyy411_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let block = (x / 4) * 6;
      let v3 = vld3q_u8(packed.as_ptr().add(block));
      let l1 = v3.1;
      let l2 = v3.2;
      let y_lo16 = vzip1q_u8(l1, l2);
      let y_hi16 = vzip2q_u8(l1, l2);
      vst1q_u8(luma_out.as_mut_ptr().add(x), y_lo16);
      vst1q_u8(luma_out.as_mut_ptr().add(x + 16), y_hi16);
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

/// NEON UYYVYY411 → u16 luma extraction (zero-extended Y bytes). 32 px /
/// iter.
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_luma_row`] with `out.len() >= width`
/// `u16` elements.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyyvyy411_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(out.len() >= width);

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let block = (x / 4) * 6;
      let v3 = vld3q_u8(packed.as_ptr().add(block));
      let l1 = v3.1;
      let l2 = v3.2;
      let y_lo16 = vzip1q_u8(l1, l2);
      let y_hi16 = vzip2q_u8(l1, l2);
      // Widen each 16 u8 → 16 u16 = 32 bytes via 2x `vmovl_u8`.
      let w0 = vmovl_u8(vget_low_u8(y_lo16));
      let w1 = vmovl_u8(vget_high_u8(y_lo16));
      let w2 = vmovl_u8(vget_low_u8(y_hi16));
      let w3 = vmovl_u8(vget_high_u8(y_hi16));
      vst1q_u16(out.as_mut_ptr().add(x), w0);
      vst1q_u16(out.as_mut_ptr().add(x + 8), w1);
      vst1q_u16(out.as_mut_ptr().add(x + 16), w2);
      vst1q_u16(out.as_mut_ptr().add(x + 24), w3);
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
