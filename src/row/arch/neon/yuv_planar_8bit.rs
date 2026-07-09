use crate::{ColorMatrix, row::scalar};

use super::*;

/// NEON YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **NEON must be available on the current CPU.** The dispatcher
///    in [`crate::row`] verifies this with
///    `is_aarch64_feature_detected!("neon")` (runtime) or
///    `cfg!(target_feature = "neon")` (compile‑time, no‑std). If you
///    call this kernel directly, you are responsible for the check —
///    executing NEON instructions on a CPU without NEON traps.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`vld1q_u8`, `vld1_u8`, `vst3q_u8`).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked NEON availability + slice bounds — see
  // [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON YUV 4:2:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked NEON availability + slice bounds — see
  // [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON YUVA 4:2:0 → packed **RGBA** (8-bit) with the per-pixel
/// alpha byte **sourced from `a_src`** (8-bit YUVA's alpha is already
/// `u8`, so no depth-conversion is needed) instead of being constant
/// `0xFF`. R / G / B math is byte-identical to [`yuv_420_to_rgba_row`].
///
/// Thin wrapper over [`yuv_420_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgba_row`] plus `a_src.len() >= width`.
#[cfg(feature = "yuva")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420_to_rgba_with_alpha_src_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true, true>(
      y,
      u_half,
      v_half,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared NEON kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, `vst3q_u8`), [`yuv_420_to_rgba_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, `vst4q_u8` with constant
/// `0xFF` alpha) and [`yuv_420_to_rgba_with_alpha_src_row`]
/// (`ALPHA = true, ALPHA_SRC = true`, `vst4q_u8` with the alpha lane
/// loaded from `a_src` — 8-bit YUVA alpha is already `u8` so no
/// depth conversion is needed). Math is byte-identical to
/// `scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`; only
/// the store intrinsic and alpha lane differ. `const` generics
/// monomorphize per call site, so the `if ALPHA` / `if ALPHA_SRC`
/// branches are eliminated.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long. When `ALPHA_SRC = true`: `a_src` must be `Some(_)`
/// and `a_src.unwrap().len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section above; the dispatcher in `crate::row` checks
  // it. All pointer adds below are bounded by the
  // `while x + 16 <= width` loop condition and the caller‑promised
  // slice lengths checked above.
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
    // Constant opaque-alpha vector for the RGBA path. Materializing
    // it outside the loop costs one `vdupq_n_u8` regardless of
    // ALPHA; the compiler DCE's it when ALPHA = false.
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      let u_vec = vld1_u8(u_half.as_ptr().add(x / 2));
      let v_vec = vld1_u8(v_half.as_ptr().add(x / 2));

      // Widen Y halves to i16x8 (unsigned → signed, Y ≤ 255 fits).
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      // Widen U, V to i16x8 and subtract 128.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_vec)), mid128);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_vec)), mid128);

      // Split to i32x4 halves so the Q15 multiplies don't overflow.
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      // u_d = (u * c_scale + RND) >> 15, bit‑exact to scalar.
      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // Per‑channel chroma contribution, narrow to i16 for later adds.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: duplicate each of the 8 chroma
      // lanes into an adjacent pair to cover 16 Y lanes. vzip1q takes
      // lanes 0..3 from both operands interleaved → [c0,c0,c1,c1,...];
      // vzip2q does the same for lanes 4..7.
      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      // Y path → i16x8 (two vectors covering 16 pixels).
      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // B, G, R = saturating_add(Y, chroma); saturate‑narrow to u8.
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
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit YUVA alpha is already u8 — load 16 bytes directly,
          // no mask + depth-convert step needed (high-bit kernels do).
          vld1q_u8(a_src.as_ref().unwrap_unchecked().as_ptr().add(x))
        } else {
          alpha_u8
        };
        // vst4q_u8 writes 64 bytes as interleaved R, G, B, A
        // quadruplets — native AArch64 4-channel store.
        let rgba = uint8x16x4_t(r_u8, g_u8, b_u8, a_u8);
        vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
      } else {
        // vst3q_u8 writes 48 bytes as interleaved R, G, B triples.
        let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
        vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
      }

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels (always even, 4:2:0
    // requires even width so x/2 and width/2 are well‑defined).
    if x < width {
      let tail_a = if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        Some(&a_src.as_ref().unwrap_unchecked()[x..width])
      } else {
        None
      };
      scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        tail_a,
        &mut out[x * bpp..width * bpp],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}
// ---- YUV 4:1:0 NEON entries -----------------------------------------
//
// 4:1:0 is a Tier 1 P3 legacy format (Cinepak / Sorenson). Per-row
// throughput on this format does not justify a hand-rolled NEON
// pipeline: each 4-pixel group shares one (U, V) sample so the chroma
// arithmetic happens 1/4 as often as in 4:2:0, and modern decoders
// almost never produce it. The NEON entry points below compute via a
// real NEON loop over 16 Y pixels at a time — loading 4 chroma
// samples and broadcasting each across 4 Y lanes via i16 lane
// duplication. Math is byte-identical to scalar by construction (same
// Q15 sequence, same saturation primitives).

/// NEON YUV 4:1:0 → packed RGB. Semantics match
/// [`scalar::yuv_410_to_rgb_row`] byte-identically.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width % 4 == 0` (4:1:0 requires width multiple of 4).
/// 3. `y.len() >= width`, `u_quarter.len() >= width / 4`,
///    `v_quarter.len() >= width / 4`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_410_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked NEON availability + slice bounds — see
  // [`yuv_410_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_410_to_rgb_or_rgba_row::<false>(
      y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON YUV 4:1:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_410_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_410_to_rgb_row`] except `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_410_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked NEON availability + slice bounds — see
  // [`yuv_410_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_410_to_rgb_or_rgba_row::<true>(
      y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared NEON kernel for [`yuv_410_to_rgb_row`] (`ALPHA = false`,
/// `vst3q_u8`) and [`yuv_410_to_rgba_row`] (`ALPHA = true`, `vst4q_u8`
/// with constant `0xFF` alpha). Math is byte-identical to
/// `scalar::yuv_410_to_rgb_or_rgba_row::<ALPHA>` — same Q15 sequence
/// and saturating-narrow primitives as the 4:2:0 NEON kernel; only
/// the chroma-fanout shape differs (4x horizontal duplication).
///
/// Pipeline per 16 Y pixels:
/// 1. Load 16 Y, 4 U, 4 V (the chroma planes are quarter-width).
/// 2. Widen U/V to i16x4, subtract 128, widen to i32x4.
/// 3. `u_d = (u * c_scale + RND) >> 15`, same for v_d (i32x4).
/// 4. Per channel C ∈ {R, G, B}:
///    `C_chroma = (C_u * u_d + C_v * v_d + RND) >> 15` (i32x4),
///    narrow-saturate to i16x4.
/// 5. Duplicate each of the 4 chroma lanes 4x to fill an i16x8 pair
///    of vectors covering 16 Y lanes — `vzip1` / `vzip2` chained gives
///    `[c0,c0,c0,c0,c1,c1,c1,c1]` and `[c2,c2,c2,c2,c3,c3,c3,c3]`.
/// 6. Y path → i16x8 pair via `scale_y`.
/// 7. Saturating add Y + chroma per channel, narrow to u8x16,
///    interleave with `vst3q_u8` / `vst4q_u8`.
///
/// # Safety
///
/// Same as [`yuv_410_to_rgb_row`] / [`yuv_410_to_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn yuv_410_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  debug_assert!(y.len() >= width);
  debug_assert!(u_quarter.len() >= width / 4);
  debug_assert!(v_quarter.len() >= width / 4);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section above; the dispatcher in `crate::row` checks
  // it. All pointer adds below are bounded by the
  // `while x + 16 <= width` loop condition and the caller-promised
  // slice lengths checked above.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = vld1q_u8(y.as_ptr().add(x));

      // Load 4 chroma bytes per plane via four `vld1_lane_u8` byte
      // loads. Each `vld1_lane_u8` writes the byte at `ptr + i` into
      // u8x8 lane i, so the resulting lane order is
      // `[c0, c1, c2, c3, _, _, _, _]` regardless of host endianness.
      // The earlier `(*const u32).read_unaligned() + vdup_n_u32`
      // sequence was native-endian dependent — on big-endian aarch64
      // it would reorder the chroma bytes, putting U/V samples on the
      // wrong horizontal pixel groups.
      //
      // We initialise the u8x8 to zero and write only the four low
      // lanes; the upper 4 are duplicated by `vmovl_u8` then sliced
      // off via `vget_low_s16`, so their values do not matter.
      //
      // SAFETY: the outer `while x + 16 <= width` bound and the
      // caller-guaranteed `u_quarter.len() >= width / 4` precondition
      // give `x / 4 + 4 <= u_quarter.len()` (and likewise for V), so
      // each of the four byte reads is in-bounds.
      let u_chroma_ptr = u_quarter.as_ptr().add(x / 4);
      let v_chroma_ptr = v_quarter.as_ptr().add(x / 4);
      let zero_u8x8 = vdup_n_u8(0);
      let u_u8x8 = vld1_lane_u8::<3>(
        u_chroma_ptr.add(3),
        vld1_lane_u8::<2>(
          u_chroma_ptr.add(2),
          vld1_lane_u8::<1>(
            u_chroma_ptr.add(1),
            vld1_lane_u8::<0>(u_chroma_ptr, zero_u8x8),
          ),
        ),
      );
      let v_u8x8 = vld1_lane_u8::<3>(
        v_chroma_ptr.add(3),
        vld1_lane_u8::<2>(
          v_chroma_ptr.add(2),
          vld1_lane_u8::<1>(
            v_chroma_ptr.add(1),
            vld1_lane_u8::<0>(v_chroma_ptr, zero_u8x8),
          ),
        ),
      );

      // Widen 4 chroma samples to i16x8 (the upper 4 lanes are zeros
      // from the `vdup_n_u8(0)` initializer but we discard them via
      // `vget_low_s16` after widening).
      let u_i16x8 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_u8x8)), vdupq_n_s16(128));
      let v_i16x8 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_u8x8)), vdupq_n_s16(128));
      // Take the low 4 lanes (the meaningful chroma samples).
      let u_i16 = vget_low_s16(u_i16x8); // i16x4
      let v_i16 = vget_low_s16(v_i16x8);

      // Widen to i32x4 for the Q15 multiplies.
      let u_i32 = vmovl_s16(u_i16);
      let v_i32 = vmovl_s16(v_i16);

      // u_d = (u * c_scale + RND) >> 15
      let u_d = q15_shift(vaddq_s32(vmulq_s32(u_i32, c_scale_v), rnd_v));
      let v_d = q15_shift(vaddq_s32(vmulq_s32(v_i32, c_scale_v), rnd_v));

      // Per-channel chroma contribution, 4 i32 lanes → i16x4.
      let r_i32 = vshrq_n_s32::<15>(vaddq_s32(
        vaddq_s32(vmulq_s32(cru, u_d), vmulq_s32(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = vshrq_n_s32::<15>(vaddq_s32(
        vaddq_s32(vmulq_s32(cgu, u_d), vmulq_s32(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = vshrq_n_s32::<15>(vaddq_s32(
        vaddq_s32(vmulq_s32(cbu, u_d), vmulq_s32(cbv, v_d)),
        rnd_v,
      ));
      // Saturate-narrow each i32x4 to i16x4.
      let r_i16x4 = vqmovn_s32_compat(r_i32);
      let g_i16x4 = vqmovn_s32_compat(g_i32);
      let b_i16x4 = vqmovn_s32_compat(b_i32);

      // Duplicate each chroma lane 4x to cover 16 Y lanes:
      //   chroma  = [c0, c1, c2, c3]                       (i16x4)
      //   dup_lo  = [c0, c0, c0, c0, c1, c1, c1, c1]       (covers Y lanes 0..7)
      //   dup_hi  = [c2, c2, c2, c2, c3, c3, c3, c3]       (covers Y lanes 8..15)
      //
      // Lane-fanout sequence (two zip layers for 4x duplication, mirroring
      // the 4:2:0 kernel which only needs one zip for 2x duplication):
      //   1. `vcombine_s16` two copies of the i16x4 → i16x8 = [c0..c3, c0..c3].
      //   2. `vzip1q_s16(x, x)` interleaves lanes [0,0,1,1,2,2,3,3] →
      //      `pair = [c0,c0,c1,c1, c2,c2,c3,c3]`.
      //   3. A second zip pass on `pair` fans each adjacent pair into a quartet:
      //      `vzip1q(pair,pair) = [c0,c0,c0,c0, c1,c1,c1,c1]` → dup_lo (Y0..7),
      //      `vzip2q(pair,pair) = [c2,c2,c2,c2, c3,c3,c3,c3]` → dup_hi (Y8..15).
      let r_i16x8 = vcombine_s16(r_i16x4, r_i16x4); // [c0..c3, c0..c3]
      let g_i16x8 = vcombine_s16(g_i16x4, g_i16x4);
      let b_i16x8 = vcombine_s16(b_i16x4, b_i16x4);

      // First zip pass (step 2 above): pair = [c0,c0,c1,c1, c2,c2,c3,c3].
      let r_pair = vzip1q_s16(r_i16x8, r_i16x8);
      let g_pair = vzip1q_s16(g_i16x8, g_i16x8);
      let b_pair = vzip1q_s16(b_i16x8, b_i16x8);

      // Second zip pass (step 3 above): pairs → quartets, split lo/hi.
      let r_dup_lo = vzip1q_s16(r_pair, r_pair);
      let r_dup_hi = vzip2q_s16(r_pair, r_pair);
      let g_dup_lo = vzip1q_s16(g_pair, g_pair);
      let g_dup_hi = vzip2q_s16(g_pair, g_pair);
      let b_dup_lo = vzip1q_s16(b_pair, b_pair);
      let b_dup_hi = vzip2q_s16(b_pair, b_pair);

      // Y path: widen 16 u8 → two i16x8 vectors, scale.
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));
      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // Saturating add per channel, then saturate-narrow to u8.
      let r_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_dup_hi)),
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

    // Scalar tail. `width` is a multiple of 4 by precondition, so the
    // tail width `width - x` is also a multiple of 4 (the only widths
    // that exit the SIMD loop early have `width < 16`, in which case
    // `x = 0` and the tail handles 4, 8, or 12 pixels).
    if x < width {
      scalar::yuv_410_to_rgb_or_rgba_row::<ALPHA>(
        &y[x..width],
        &u_quarter[x / 4..width / 4],
        &v_quarter[x / 4..width / 4],
        &mut out[x * bpp..width * bpp],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// NEON YUV 4:4:4 planar → packed RGB. Same arithmetic as
/// [`nv24_to_rgb_row`] — one UV pair per Y pixel, no chroma
/// duplication — but U and V come from separate planes instead of
/// an interleaved UV stream.
///
/// # Safety
///
/// Same contract as [`yuv_444_to_rgb_or_rgba_row`] with
/// `ALPHA = false` (so `out.len() >= width * 3` specializes to
/// `rgb_out.len() >= 3 * width`):
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `rgb_out.len() >= 3 * width`.
///
/// No width parity constraint (4:4:4).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked NEON availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<false, false>(y, u, v, None, rgb_out, width, matrix, full_range);
  }
}

/// NEON YUV 4:4:4 planar → packed **RGBA** (8-bit). Same contract
/// as [`yuv_444_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8` (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_444_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked NEON availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true, false>(y, u, v, None, rgba_out, width, matrix, full_range);
  }
}

/// NEON YUVA 4:4:4 → packed **RGBA** with source alpha. R/G/B are
/// byte-identical to [`yuv_444_to_rgb_row`]; the per-pixel alpha byte
/// is sourced from `a_src` (8-bit, no shift needed) instead of being
/// constant `0xFF`. Used by [`crate::source::Yuva444p`].
///
/// Thin wrapper over [`yuv_444_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444_to_rgba_row`] plus `a_src.len() >= width`.
#[cfg(feature = "yuva")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444_to_rgba_with_alpha_src_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true, true>(
      y,
      u,
      v,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared NEON YUV 4:4:4 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `vst3q_u8`.
/// - `ALPHA = true, ALPHA_SRC = false`: `vst4q_u8` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `vst4q_u8` with the alpha
///   lane loaded from `a_src` (8-bit input — no shift needed).
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
/// 4. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
///
/// No width parity constraint (4:4:4).
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; pointer
  // adds are bounded by `while x + 16 <= width` and caller-promised
  // slice lengths.
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
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      // 4:4:4: load 16 U + 16 V directly from separate planes (no
      // deinterleave step needed).
      let u_vec = vld1q_u8(u.as_ptr().add(x));
      let v_vec = vld1q_u8(v.as_ptr().add(x));

      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_vec))), mid128);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u_vec))), mid128);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_vec))), mid128);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_vec))), mid128);

      let u_lo_a = vmovl_s16(vget_low_s16(u_lo_i16));
      let u_lo_b = vmovl_s16(vget_high_s16(u_lo_i16));
      let u_hi_a = vmovl_s16(vget_low_s16(u_hi_i16));
      let u_hi_b = vmovl_s16(vget_high_s16(u_hi_i16));
      let v_lo_a = vmovl_s16(vget_low_s16(v_lo_i16));
      let v_lo_b = vmovl_s16(vget_high_s16(v_lo_i16));
      let v_hi_a = vmovl_s16(vget_low_s16(v_hi_i16));
      let v_hi_b = vmovl_s16(vget_high_s16(v_hi_i16));

      let u_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
      );

      if ALPHA {
        let a_v = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit alpha — load 16 bytes verbatim.
          vld1q_u8(a_src.as_ref().unwrap_unchecked().as_ptr().add(x))
        } else {
          alpha_u8
        };
        let rgba = uint8x16x4_t(r_u8, g_u8, b_u8, a_v);
        vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
      } else {
        let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
        vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444_to_rgba_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444_to_rgba_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yuv_444_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- YUV 4:1:1 → RGB / RGBA (NEON) -------------------------------------

/// NEON YUV 4:1:1 planar → packed RGB. One chroma sample drives four
/// Y pixels (1→4 nearest-neighbor upsample in registers).
///
/// Same Q15 arithmetic as the scalar reference; output is byte-
/// identical. The SIMD body processes 16 Y / 4 chroma samples per
/// iteration; the trailing 1..15 Y pixels (including any partial
/// 1..3-pixel chroma group when `width % 4 != 0`) fall through to
/// the scalar reference, which handles FFmpeg's `div_ceil(4)`
/// chroma semantics.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`,
///    `u_quarter.len() >= width.div_ceil(4)`,
///    `v_quarter.len() >= width.div_ceil(4)`.
/// 3. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_411_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked NEON availability + slice bounds — see
  // [`yuv_411_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_411_to_rgb_or_rgba_row::<false>(
      y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON YUV 4:1:1 planar → packed **RGBA** (8-bit). Same contract as
/// [`yuv_411_to_rgb_row`] but writes 4 bytes per pixel via `vst4q_u8`
/// (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_411_to_rgb_row`] except `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_411_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked NEON availability + slice bounds — see
  // [`yuv_411_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_411_to_rgb_or_rgba_row::<true>(
      y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared NEON YUV 4:1:1 kernel. Processes 16 Y pixels (= 4 chroma
/// samples) per iteration; the 1→4 chroma upsample is materialized
/// in registers via paired `vzip1q_s16` / `vzip2q_s16` cascades:
/// 4 chroma lanes → 8 (each duplicated once) → 16 (each duplicated
/// three more times) matches the 16 Y lanes.
///
/// 4:1:1 has no source-alpha variant (no `Yuva411p` exists), so the
/// const-generic surface stays 1-D (`ALPHA` only).
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`,
///    `u_quarter.len() >= width.div_ceil(4)`,
///    `v_quarter.len() >= width.div_ceil(4)`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn yuv_411_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u_quarter.len() >= width.div_ceil(4));
  debug_assert!(v_quarter.len() >= width.div_ceil(4));
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section above; the dispatcher in `crate::row` checks
  // it. All pointer adds below are bounded by the
  // `while x + 16 <= width` loop condition and the caller-promised
  // slice lengths checked above.
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
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      // Load 4 chroma bytes per plane via four `vld1_lane_u8` byte
      // loads. Each `vld1_lane_u8` writes the byte at `ptr + i` into
      // u8x8 lane i, so the resulting lane order is
      // `[c0, c1, c2, c3, _, _, _, _]` regardless of host endianness.
      // The earlier `(*const u32).read_unaligned() + vcreate_u8`
      // sequence was native-endian dependent — on big-endian aarch64
      // it would reorder the chroma bytes, putting U/V samples on the
      // wrong horizontal pixel groups. The 4:1:0 NEON kernel above
      // uses the same byte-lane cascade for the same reason.
      //
      // We initialise the u8x8 to zero and write only the four low
      // lanes; the upper 4 lanes stay zero and are sliced off via
      // `vget_low_s16` after `vmovl_u8`, so their values do not
      // matter.
      //
      // SAFETY: the outer `while x + 16 <= width` bound and the
      // caller-guaranteed `u_quarter.len() >= width / 4` precondition
      // give `x / 4 + 4 <= u_quarter.len()` (and likewise for V), so
      // each of the four byte reads is in-bounds.
      let u_chroma_ptr = u_quarter.as_ptr().add(x / 4);
      let v_chroma_ptr = v_quarter.as_ptr().add(x / 4);
      let zero_u8x8 = vdup_n_u8(0);
      let u_u8x8 = vld1_lane_u8::<3>(
        u_chroma_ptr.add(3),
        vld1_lane_u8::<2>(
          u_chroma_ptr.add(2),
          vld1_lane_u8::<1>(
            u_chroma_ptr.add(1),
            vld1_lane_u8::<0>(u_chroma_ptr, zero_u8x8),
          ),
        ),
      );
      let v_u8x8 = vld1_lane_u8::<3>(
        v_chroma_ptr.add(3),
        vld1_lane_u8::<2>(
          v_chroma_ptr.add(2),
          vld1_lane_u8::<1>(
            v_chroma_ptr.add(1),
            vld1_lane_u8::<0>(v_chroma_ptr, zero_u8x8),
          ),
        ),
      );

      // Widen 4 chroma samples to i16x8. Lanes 0..3 carry the four
      // chroma samples; lanes 4..7 stay zero from the `vdup_n_u8(0)`
      // initializer and are discarded via `vget_low_s16` below.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_u8x8)), mid128);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_u8x8)), mid128);

      // Promote the low 4 lanes to i32x4 — these carry the meaningful
      // chroma samples that feed the 1→4 fanout cascade below.
      let u_i32 = vmovl_s16(vget_low_s16(u_i16));
      let v_i32 = vmovl_s16(vget_low_s16(v_i16));

      // u_d / v_d in i32x4 (4 chroma values).
      let u_d = q15_shift(vaddq_s32(vmulq_s32(u_i32, c_scale_v), rnd_v));
      let v_d = q15_shift(vaddq_s32(vmulq_s32(v_i32, c_scale_v), rnd_v));

      // Per-channel chroma contribution as i32x4 (4 lanes, one per
      // chroma sample). Narrow to i16x4 stuffed into the low half of
      // an i16x8.
      let r_i32 = q15_shift(vaddq_s32(
        vaddq_s32(vmulq_s32(cru, u_d), vmulq_s32(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = q15_shift(vaddq_s32(
        vaddq_s32(vmulq_s32(cgu, u_d), vmulq_s32(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = q15_shift(vaddq_s32(
        vaddq_s32(vmulq_s32(cbu, u_d), vmulq_s32(cbv, v_d)),
        rnd_v,
      ));

      // Narrow to i16x4 in the low half of an i16x8 (the high half is
      // unused). `vqmovn_s32` saturates i32 → i16, then `vcombine_s16`
      // pairs it with arbitrary garbage in the high half — we never
      // touch those lanes.
      let r_low4 = vqmovn_s32_compat(r_i32);
      let g_low4 = vqmovn_s32_compat(g_i32);
      let b_low4 = vqmovn_s32_compat(b_i32);

      // 1→4 nearest-neighbor upsample. Stage 1: duplicate each of the
      // 4 lanes once via `vzip1_s16(x, x)` over an i16x4 — produces
      // the i16x4 [c0,c0,c1,c1] (low half of the doubled vector). Use
      // `vzip2_s16(x, x)` for the high half [c2,c2,c3,c3].
      //
      // Stage 2: combine those two i16x4 halves back into one i16x8,
      // giving [c0,c0,c1,c1,c2,c2,c3,c3]. Then run `vzip1q_s16(s,s)`
      // and `vzip2q_s16(s,s)` over that 8-lane vector to land at
      // [c0x4,c1x4] (low) and [c2x4,c3x4] (high), matching the 16 Y
      // lanes.
      let r_dup8 = vcombine_s16(vzip1_s16(r_low4, r_low4), vzip2_s16(r_low4, r_low4));
      let g_dup8 = vcombine_s16(vzip1_s16(g_low4, g_low4), vzip2_s16(g_low4, g_low4));
      let b_dup8 = vcombine_s16(vzip1_s16(b_low4, b_low4), vzip2_s16(b_low4, b_low4));
      // r_dup8 = [c0,c0,c1,c1,c2,c2,c3,c3].

      let r_x2 = vzip1q_s16(r_dup8, r_dup8);
      let g_x2 = vzip1q_s16(g_dup8, g_dup8);
      let b_x2 = vzip1q_s16(b_dup8, b_dup8);
      // r_x2 holds [c0,c0,c0,c0,c1,c1,c1,c1] (`vzip1q(s,s)` takes
      // lanes 0..3 of both operands interleaved → c0,c0,c0,c0,c1,c1,c1,c1).
      let r_x2_hi = vzip2q_s16(r_dup8, r_dup8);
      let g_x2_hi = vzip2q_s16(g_dup8, g_dup8);
      let b_x2_hi = vzip2q_s16(b_dup8, b_dup8);
      // r_x2_hi holds [c2,c2,c2,c2,c3,c3,c3,c3].

      // Y path → i16x8 (two vectors covering 16 pixels).
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));
      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_x2)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_x2_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_x2)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_x2_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_x2)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_x2_hi)),
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

    // Scalar tail. The SIMD loop strides 16 Y pixels (always a
    // multiple of 4), so `x` is a multiple of 4 ≤ width. The tail
    // covers the remaining 0..15 Y pixels and the corresponding
    // chroma samples up to `width.div_ceil(4)` (FFmpeg ceil-shift),
    // which may include a partial 1..3-pixel final chroma group when
    // `width % 4 != 0`. The scalar reference handles both the
    // aligned-tail and partial-chroma cases.
    if x < width {
      let tail_w = width - x;
      let chroma_end = width.div_ceil(4);
      let tail_u = &u_quarter[x / 4..chroma_end];
      let tail_v = &v_quarter[x / 4..chroma_end];
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA {
        scalar::yuv_411_to_rgba_row(
          &y[x..width],
          tail_u,
          tail_v,
          tail_out,
          tail_w,
          matrix,
          full_range,
        );
      } else {
        scalar::yuv_411_to_rgb_row(
          &y[x..width],
          tail_u,
          tail_v,
          tail_out,
          tail_w,
          matrix,
          full_range,
        );
      }
    }
  }
}

// ---- Planar 8-bit YUV → HSV (staged via a reused RGB chunk) ----------
//
// The SIMD twins of the scalar `yuv_*_to_hsv_row` kernels. Rather than
// re-derive an HSV-specific register pipeline, each fills a small fixed
// reused RGB scratch (one `CHUNK`-pixel chunk at a time) using the
// EXISTING NEON `yuv_*_to_rgb_*` kernel — so the chunk filler IS the
// production RGB kernel — then runs the NEON `rgb_to_hsv_row` on the
// chunk. This keeps the per-format SIMD surface tiny (only the chunked
// driver is new) and makes the result byte-identical to
// `rgb_to_hsv_row(yuv_*_to_rgb_row(...))` within the NEON tier. The
// scalar tail of each underlying RGB kernel handles widths below the
// SIMD block, so no separate tail is needed here.
//
// `CHUNK = 64` is a multiple of 4, so every chunk offset lands on a
// chroma-sample boundary for the 1→2 (4:2:0 / 4:2:2) and 1→4
// (4:1:0 / 4:1:1) upsampling shapes alike.

/// One reused RGB chunk's worth of pixels staged before the HSV pass.
/// A multiple of 4 so every chunk offset lands on a chroma boundary for
/// both the 1→2 (4:2:0 / 4:2:2) and 1→4 (4:1:0 / 4:1:1) shapes.
pub(crate) const HSV_CHUNK: usize = 64;

/// Shared NEON driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills
/// a small reused stack RGB scratch via `fill_rgb` (the existing NEON
/// RGB kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the NEON [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. The result is byte-identical to
/// `rgb_to_hsv_row(yuv_*_to_rgb_row(...))` within the NEON tier, with no
/// source-width RGB allocation.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`. The width
/// alignment / bounds obligations are the caller's (checked by the
/// format wrappers' `debug_assert!`s).
///
/// # Safety
///
/// NEON must be available, and `fill_rgb` must uphold the underlying RGB
/// kernel's safety contract for each chunk. Each of `h_out` / `s_out` /
/// `v_out` must be `>= width`.
#[inline]
pub(crate) unsafe fn yuv_to_hsv_via_rgb_chunks(
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

/// NEON: YUV 4:2:0 planar → planar HSV bytes (OpenCV encoding), staged
/// via the reused-RGB-chunk pattern over the NEON
/// [`yuv_420_to_rgb_row`] + [`rgb_to_hsv_row`]. Also serves 4:2:2
/// (identical per-row chroma shape). Byte-identical to
/// `rgb_to_hsv_row(yuv_420_to_rgb_row(...))` within the NEON tier.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420_to_hsv_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: NEON verified; the chunk filler forwards the per-chunk
  // sub-slices to the NEON 4:2:0 RGB kernel under the same contract.
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_420_to_rgb_row(
        &y[offset..],
        &u_half[offset / 2..],
        &v_half[offset / 2..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// NEON: YUV 4:4:4 planar → planar HSV bytes, staged via the NEON
/// [`yuv_444_to_rgb_row`] + [`rgb_to_hsv_row`]. Also serves 4:4:0.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444_to_hsv_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: NEON verified; the chunk filler forwards to the NEON 4:4:4
  // RGB kernel under the same contract.
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_444_to_rgb_row(
        &y[offset..],
        &u[offset..],
        &v[offset..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// NEON: YUV 4:1:0 planar → planar HSV bytes, staged via the NEON
/// [`yuv_410_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width % 4 == 0`.
/// 3. `y.len() >= width`, `u_quarter.len() >= width / 4`,
///    `v_quarter.len() >= width / 4`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_410_to_hsv_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  debug_assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: NEON verified; the chunk filler forwards to the NEON 4:1:0
  // RGB kernel under the same contract.
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_410_to_rgb_row(
        &y[offset..],
        &u_quarter[offset / 4..],
        &v_quarter[offset / 4..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// NEON: YUV 4:1:1 planar → planar HSV bytes, staged via the NEON
/// [`yuv_411_to_rgb_row`] + [`rgb_to_hsv_row`]. FFmpeg-compatible
/// arbitrary widths: `HSV_CHUNK` is a multiple of 4 so every chunk but
/// the last is chroma-aligned; the final chunk's 1..3-pixel partial
/// group is handled by the underlying RGB kernel's own partial-group
/// logic.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `y.len() >= width`, `u_quarter.len() >= width.div_ceil(4)`,
///    `v_quarter.len() >= width.div_ceil(4)`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_411_to_hsv_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(
    u_quarter.len() >= width.div_ceil(4),
    "u_quarter row too short"
  );
  debug_assert!(
    v_quarter.len() >= width.div_ceil(4),
    "v_quarter row too short"
  );
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: NEON verified; the chunk filler forwards to the NEON 4:1:1
  // RGB kernel (whose own partial-group logic covers the final chunk's
  // sub-4 tail) under the same contract.
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_411_to_rgb_row(
        &y[offset..],
        &u_quarter[offset / 4..],
        &v_quarter[offset / 4..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}
