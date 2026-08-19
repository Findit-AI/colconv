use crate::{KernelMatrix, row::scalar};

use super::*;

/// NEON YUYV422 → packed RGB. Semantics match
/// [`scalar::yuyv422_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= 2 * width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuyv422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON YUYV422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuyv422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON UYVY422 → packed RGB.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyvy422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON UYVY422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyvy422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<false, false, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON YVYU422 → packed RGB.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yvyu422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, false>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON YVYU422 → packed RGBA (alpha = 0xFF).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yvyu422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_rgb_or_rgba_row::<true, true, true>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// Generic packed YUV 4:2:2 → RGB / RGBA NEON kernel.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. The pipeline
/// mirrors `yuv_420_to_rgb_or_rgba_row` byte‑for‑byte after the inline
/// deinterleave: load 32 packed bytes, split Y from chroma via
/// `vld2q_u8`, split chroma into U / V via `vuzp_u8`, then run the
/// same Q15 chroma / Y / channel pipeline.
///
/// The two const generics select **which lanes** of the deinterleaved
/// pair are Y vs chroma (`Y_LSB`) and U vs V (`SWAP_UV`):
///
/// | Format | `Y_LSB` | `SWAP_UV` | Block bytes |
/// |---|---|---|---|
/// | YUYV422 | true  | false | `[Y0, U0, Y1, V0]` |
/// | UYVY422 | false | false | `[U0, Y0, V0, Y1]` |
/// | YVYU422 | true  | true  | `[Y0, V0, Y1, U0]` |
///
/// `(Y_LSB=false, SWAP_UV=true)` would be VYUY422 (not in FFmpeg),
/// never instantiated.
///
/// # Safety
///
/// Caller has verified NEON. `packed.len() >= 2 * width`. `width` is
/// even. `out.len() >= bpp * width` where bpp = 3 for ALPHA=false,
/// 4 for ALPHA=true.
#[inline]
#[target_feature(enable = "neon")]
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

  // SAFETY: NEON availability is the caller's obligation. Pointer
  // arithmetic below is bounded by the loop condition and the
  // caller-promised slice lengths.
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

    let mut x = 0usize;
    while x + 16 <= width {
      // vld2q_u8 reads 32 bytes and returns even-indexed bytes in `.0`
      // (16 bytes) and odd-indexed bytes in `.1`. Within a 4-byte block
      // (`[b0, b1, b2, b3]`), bytes 0 / 2 land in `.0` and bytes 1 / 3
      // land in `.1`. So:
      //   Y_LSB = true  → Y is bytes 0, 2 → y_vec = pair.0; chroma = pair.1
      //   Y_LSB = false → Y is bytes 1, 3 → y_vec = pair.1; chroma = pair.0
      let pair = vld2q_u8(packed.as_ptr().add(x * 2));
      let y_vec = if Y_LSB { pair.0 } else { pair.1 };
      let chroma_vec = if Y_LSB { pair.1 } else { pair.0 };

      // chroma_vec has 16 bytes — alternating U/V (or V/U) — covering
      // 8 chroma pairs. Split into two 8-byte halves so we can use
      // `vuzp1_u8` / `vuzp2_u8` to separate even-position chroma bytes
      // from odd-position ones.
      let chroma_lo = vget_low_u8(chroma_vec);
      let chroma_hi = vget_high_u8(chroma_vec);
      // c_evens = chroma bytes at even positions (the first byte of
      // each U/V or V/U pair); c_odds = the second byte of each pair.
      let c_evens = vuzp1_u8(chroma_lo, chroma_hi);
      let c_odds = vuzp2_u8(chroma_lo, chroma_hi);
      // Map to U / V:
      //   SWAP_UV = false (YUYV / UYVY) → c_evens = U, c_odds = V
      //   SWAP_UV = true  (YVYU)        → c_evens = V, c_odds = U
      let u_vec = if SWAP_UV { c_odds } else { c_evens };
      let v_vec = if SWAP_UV { c_evens } else { c_odds };

      // From here, the math is byte-identical to yuv_420's NEON kernel.
      // Widen Y halves to i16x8.
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      // Widen U, V to i16x8 and subtract 128.
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

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest-neighbor upsample: each chroma sample serves 2 Y pixels.
      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );

      if ALPHA {
        let rgba = uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8);
        vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
      } else {
        let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
        vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
      }

      x += 16;
    }

    // Scalar tail.
    if x < width {
      // Scalar tail dispatch — pick the right scalar entry based on
      // const generics. ALPHA=false → *_to_rgb_row; ALPHA=true → *_to_rgba_row.
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
// EXISTING NEON packed-4:2:2 RGB kernel — so the chunk filler IS the
// production RGB kernel — then runs the NEON `rgb_to_hsv_row` on the
// chunk. This keeps the per-format SIMD surface tiny (only the chunked
// driver is new) and makes the result byte-identical to
// `rgb_to_hsv_row(*_to_rgb_row(...))` within the NEON tier. The scalar
// tail of each underlying RGB kernel handles widths below the SIMD
// block, so no separate tail is needed here.
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

/// Shared NEON driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing NEON RGB
/// kernel for the format, passed the chunk `offset` and length `n`), then
/// runs the NEON [`rgb_to_hsv_row`] on that chunk into the H/S/V planes.
/// Byte-identical to `rgb_to_hsv_row(*_to_rgb_row(...))` within the NEON
/// tier, with no source-width RGB allocation. Shared by the packed 4:2:2
/// kernels here and the 4:1:1 kernel in the sibling module.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// NEON must be available, and `fill_rgb` must uphold the underlying RGB
/// kernel's safety contract for each chunk. Each of `h_out` / `s_out` /
/// `v_out` must be `>= width`.
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
    // SAFETY: NEON verified by the wrapper's `#[target_feature]`; the
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

/// NEON: YUYV422 (4:2:2) → planar HSV bytes (OpenCV encoding), staged via
/// the reused-RGB-chunk pattern over the NEON [`yuyv422_to_rgb_row`] +
/// [`rgb_to_hsv_row`]. Byte-identical to
/// `rgb_to_hsv_row(yuyv422_to_rgb_row(...))` within the NEON tier.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `packed.len() >= 2 * width`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
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

  // SAFETY: NEON verified; the chunk filler forwards the per-chunk
  // sub-slices to the NEON YUYV422 RGB kernel under the same contract.
  // The packed byte offset for the chunk at pixel `offset` is
  // `offset * 2` (2 bytes per pixel).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuyv422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// NEON: UYVY422 (4:2:2) → planar HSV bytes, staged via the NEON
/// [`uyvy422_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`yuyv422_to_hsv_row`].
#[inline]
#[target_feature(enable = "neon")]
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

  // SAFETY: NEON verified; forwards to the NEON UYVY422 RGB kernel under
  // the same contract (packed byte offset = `offset * 2`).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      uyvy422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// NEON: YVYU422 (4:2:2) → planar HSV bytes, staged via the NEON
/// [`yvyu422_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`yuyv422_to_hsv_row`].
#[inline]
#[target_feature(enable = "neon")]
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

  // SAFETY: NEON verified; forwards to the NEON YVYU422 RGB kernel under
  // the same contract (packed byte offset = `offset * 2`).
  unsafe {
    packed_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yvyu422_to_rgb_row(&packed[offset * 2..], rgb, n, matrix, full_range);
    });
  }
}

/// NEON YUYV422 → 8-bit luma extraction. Reads packed bytes via
/// `vld2q_u8` and stores the Y vector directly.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuyv422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

/// NEON UYVY422 → 8-bit luma extraction.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyvy422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<false>(packed, luma_out, width);
  }
}

/// NEON YVYU422 → 8-bit luma extraction (Y positions same as YUYV).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yvyu422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_row::<true>(packed, luma_out, width);
  }
}

/// NEON YUYV422 → u16 luma extraction (zero-extends Y bytes via `vmovl_u8`).
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuyv422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<true>(packed, out, width);
  }
}

/// NEON UYVY422 → u16 luma extraction.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyvy422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<false>(packed, out, width);
  }
}

/// NEON YVYU422 → u16 luma extraction (Y positions same as YUYV).
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yvyu422_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    yuv422_packed_to_luma_u16_row::<true>(packed, out, width);
  }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn yuv422_packed_to_luma_u16_row<const Y_LSB: bool>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let pair = vld2q_u8(packed.as_ptr().add(x * 2));
      let y_vec = if Y_LSB { pair.0 } else { pair.1 };
      let y_lo = vmovl_u8(vget_low_u8(y_vec));
      let y_hi = vmovl_u8(vget_high_u8(y_vec));
      vst1q_u16(out.as_mut_ptr().add(x), y_lo);
      vst1q_u16(out.as_mut_ptr().add(x + 8), y_hi);
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
#[target_feature(enable = "neon")]
unsafe fn yuv422_packed_to_luma_row<const Y_LSB: bool>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let pair = vld2q_u8(packed.as_ptr().add(x * 2));
      let y_vec = if Y_LSB { pair.0 } else { pair.1 };
      vst1q_u8(luma_out.as_mut_ptr().add(x), y_vec);
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
