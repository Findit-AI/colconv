//! aarch64 NEON backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_aarch64_feature_detected!("neon")` returns true (runtime,
//! std‑gated) or `cfg!(target_feature = "neon")` evaluates true
//! (compile‑time, no‑std). The kernel itself carries
//! `#[target_feature(enable = "neon")]` so its intrinsics execute in
//! an explicitly NEON‑enabled context rather than one merely inherited
//! from the aarch64 target's default feature set.
//!
//! # Numerical contract
//!
//! The kernel uses i32 widening multiplies and the same
//! `(prod + (1 << 14)) >> 15` Q15 rounding as
//! [`crate::row::scalar::yuv_420_to_rgb_row`], so output is
//! **byte‑identical** to the scalar reference for every input. This is
//! asserted by the equivalence tests below.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`vld1q_u8`) + 8 U (`vld1_u8`) + 8 V (`vld1_u8`).
//! 2. Widen U/V to i16, subtract 128 → `u_i16`, `v_i16`.
//! 3. Widen to i32 and apply `c_scale` (Q15) → `u_d`, `v_d` (i32x4 × 2).
//! 4. Per channel C ∈ {R, G, B}:
//!    `C_chroma = (C_u * u_d + C_v * v_d + RND) >> 15` in i32,
//!    narrow‑saturate to i16x8 (8 lanes = 8 chroma pairs).
//! 5. Duplicate each chroma lane into its Y‑pair slot with
//!    `vzip1q_s16` / `vzip2q_s16` → 16 i16 chroma lanes matching the
//!    16 Y lanes (nearest‑neighbor upsample in registers, no memory
//!    traffic).
//! 6. Y path: `(Y - y_off) * y_scale + RND >> 15` in i32, narrow to i16.
//! 7. Saturating add Y + chroma per channel → i16x16.
//! 8. Saturate‑narrow to u8x16 and interleave with `vst3q_u8`.

use core::arch::aarch64::*;

use crate::{ColorMatrix, row::scalar};

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
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
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
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
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

/// NEON high‑bit‑depth YUV 4:2:0 (`BITS` ∈ {10, 12, 14}) → packed
/// **8‑bit** RGB.
///
/// Block size is 16 Y pixels / 8 chroma pairs per iteration. The
/// pipeline mirrors [`yuv_420_to_rgb_row`] byte‑for‑byte; the only
/// structural differences are:
/// - Loads are `vld1q_u16` (8 lanes of `u16`) instead of `vld1q_u8`
///   (16 lanes of `u8`), so each Y iteration needs two Y loads to
///   cover 16 pixels — there's no widening step because the samples
///   already live in 16‑bit lanes.
/// - Chroma bias is `128 << (BITS - 8)` (512 for 10‑bit, 2048 for
///   12‑bit, 8192 for 14‑bit) rather than 128.
/// - Range‑scaling params come from [`scalar::range_params_n`] with
///   the matching `BITS` const, so `y_scale` / `c_scale` map the
///   source depth to 8‑bit output in a single Q15 shift.
/// - Each load is AND‑masked to the low `BITS` bits so out‑of‑range
///   samples (e.g. high‑bit‑packed data mistakenly handed to the
///   low‑packed kernel) produce deterministic, backend‑consistent
///   output.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::yuv_420p_n_to_rgb_row::<BITS>`] across
/// all supported bit depths.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
/// 4. `BITS` must be one of `{9, 10, 12, 14}` — the Q15 pipeline
///    overflows i32 at 16 bits; see [`scalar::range_params_n`].
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON high-bit-depth YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Same numerical contract as [`yuv_420p_n_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the constant
/// alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON YUVA 4:2:0 high-bit-depth → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (depth-converted via
/// `>> (BITS - 8)` to fit `u8`) instead of being constant `0xFF`.
/// Same numerical contract as [`yuv_420p_n_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgba_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, true>(
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

/// Shared NEON high-bit-depth YUV 4:2:0 kernel for
/// [`yuv_420p_n_to_rgb_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `vst3q_u8`), [`yuv_420p_n_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `vst4q_u8` with constant `0xFF` alpha) and
/// [`yuv_420p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `vst4q_u8` with the alpha lane loaded from
/// `a_src`, masked to BITS, and depth-converted via the variable
/// shift `>> (BITS - 8)`).
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 5. `BITS` must be one of `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 16 <= width` loop condition and the caller‑
  // promised slice lengths checked above.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let mask_v = vdupq_n_u16(scalar::bits_mask::<BITS>());
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // Two Y loads cover 16 lanes; one U load + one V load cover 8
      // chroma each. Each load is AND‑masked to the low BITS bits so
      // out‑of‑range samples (e.g. high‑bit‑packed data handed to
      // the low‑packed kernel) can never push an intermediate past
      // i16 range. For valid input the AND is a no‑op.
      let y_vec_lo = vandq_u16(vld1q_u16(y.as_ptr().add(x)), mask_v);
      let y_vec_hi = vandq_u16(vld1q_u16(y.as_ptr().add(x + 8)), mask_v);
      let u_vec = vandq_u16(vld1q_u16(u_half.as_ptr().add(x / 2)), mask_v);
      let v_vec = vandq_u16(vld1q_u16(v_half.as_ptr().add(x / 2)), mask_v);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      // c - 512 for 10‑bit chroma, fits i16 since c ≤ 1023.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      // Widen to i32x4 halves so the Q15 multiplies don't overflow.
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

      // Duplicate the 8 chroma lanes into 16‑lane pairs — identical
      // nearest‑neighbor upsample strategy as the 8‑bit kernel.
      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // u8 output: saturate‑narrow i16 → u8 clamps to [0, 255].
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_u16 = vandq_u16(vld1q_u16(a_ptr.add(x)), mask_v);
          let a_hi_u16 = vandq_u16(vld1q_u16(a_ptr.add(x + 8)), mask_v);
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar. NEON's
          // `vshrq_n_u16` requires a literal const generic shift, but
          // `BITS - 8` is not a stable const expression on a const
          // generic — `vshlq_u16` with a negative count vector
          // performs the same logical right shift dynamically.
          let a_shr = vdupq_n_s16(-((BITS - 8) as i16));
          let a_lo_shifted = vshlq_u16(a_lo_u16, a_shr);
          let a_hi_shifted = vshlq_u16(a_hi_u16, a_shr);
          vcombine_u8(vqmovn_u16(a_lo_shifted), vqmovn_u16(a_hi_shifted))
        } else {
          alpha_u8
        };
        let rgba = uint8x16x4_t(r_u8, g_u8, b_u8, a_u8);
        vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
      } else {
        let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
        vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels (always even per 4:2:0).
    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p_n_to_rgba_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p_n_to_rgb_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON high‑bit‑depth YUV 4:2:0 (`BITS` ∈ {10, 12, 14}) → packed
/// **native‑depth `u16`** RGB.
///
/// Block size is 16 Y pixels / 8 chroma pairs per iteration. Shares
/// all pre‑write math with [`yuv_420p_n_to_rgb_row`]; the only
/// difference is the final clamp + write:
/// - Y‑path scale is calibrated for `OUT_BITS = BITS` rather than 8,
///   so `y_scaled` lives in `[0, (1 << BITS) - 1]`.
/// - The `y_scaled + chroma` sum is clamped to `[0, (1 << BITS) - 1]`
///   with `vmaxq_s16(vminq_s16(_, max), 0)` — a simple saturate‑
///   narrow doesn't suffice because the sum can overshoot the
///   `BITS`-bit max without saturating at i16 bounds.
/// - Writes use two `vst3q_u16` calls per iteration — each handles 8
///   pixels × 3 channels = 24 `u16` elements, so two cover 16 pixels.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<BITS>`] across
/// supported `BITS` values.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
/// 4. `BITS` must be one of `{10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, false>(
      y, u_half, v_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON sibling of [`yuv_420p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth) — matches `scalar::yuv_420p_n_to_rgba_u16_row`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_u16_row`], plus
/// `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true>(
      y, u_half, v_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared NEON high-bit YUV 4:2:0 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `vst3q_u16`; `ALPHA = true`
/// writes RGBA quads via `vst4q_u16` with constant alpha
/// `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: NEON availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 16 <= width` loop condition and the caller‑
  // promised slice lengths.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let mask_v = vdupq_n_u16(scalar::bits_mask::<BITS>());
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      // AND‑mask each load to the low BITS bits so intermediates
      // stay within the i16 range the Q15 narrow steps expect — see
      // matching comment in [`yuv_420p_n_to_rgb_row`].
      let y_vec_lo = vandq_u16(vld1q_u16(y.as_ptr().add(x)), mask_v);
      let y_vec_hi = vandq_u16(vld1q_u16(y.as_ptr().add(x + 8)), mask_v);
      let u_vec = vandq_u16(vld1q_u16(u_half.as_ptr().add(x / 2)), mask_v);
      let v_vec = vandq_u16(vld1q_u16(v_half.as_ptr().add(x / 2)), mask_v);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

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

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // Native‑depth output: add Y + chroma in i16, then clamp to
      // [0, (1 << BITS) - 1] explicitly. `vqaddq_s16` saturates at
      // i16 bounds (irrelevant here: |sum| stays well inside i16
      // for BITS ≤ 14), so the subsequent max/min clamps to the
      // native bit depth.
      let r_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      if ALPHA {
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, alpha_u16);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, alpha_u16);
        vst4q_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u16(out.as_mut_ptr().add(x * 4 + 32), rgba_hi);
      } else {
        // Two interleaved u16 writes — each `vst3q_u16` covers 8 pixels.
        let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
        let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
        vst3q_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u16(out.as_mut_ptr().add(x * 3 + 24), rgb_hi);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_420p_n_to_rgba_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON YUV 4:4:4 planar high-bit-depth → **u8** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`. Same structure as
/// [`yuv_420p_n_to_rgb_row`] but with full-width U/V (no chroma
/// duplication) and no width parity constraint.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// 1. **NEON must be available.** 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, false, false>(
      y, u, v, rgb_out, width, matrix, full_range, None,
    );
  }
}

/// NEON YUV 4:4:4 planar high-bit-depth → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p_n_to_rgb_row`]; the only differences are the per-pixel
/// stride (4 vs 3) and the constant alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, false>(
      y, u, v, rgba_out, width, matrix, full_range, None,
    );
  }
}

/// NEON YUVA 4:4:4 planar high-bit-depth → packed **8-bit RGBA** with
/// the per-pixel alpha byte **sourced from `a_src`** (depth-converted
/// via `>> (BITS - 8)` to fit `u8`) instead of being constant `0xFF`.
/// Same numerical contract as [`yuv_444p_n_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgba_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, true>(
      y,
      u,
      v,
      rgba_out,
      width,
      matrix,
      full_range,
      Some(a_src),
    );
  }
}

/// Shared NEON high-bit-depth YUV 4:4:4 kernel for
/// [`yuv_444p_n_to_rgb_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `vst3q_u8`), [`yuv_444p_n_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `vst4q_u8` with constant `0xFF` alpha vector)
/// and [`yuv_444p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `vst4q_u8` with the alpha lane loaded and
/// depth-converted from `a_src`).
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` must be one of `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  a_src: Option<&[u16]>,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let mask_v = vdupq_n_u16(scalar::bits_mask::<BITS>());
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y + 16 U + 16 V per iter, loaded as two u16x8 halves each.
      let y_vec_lo = vandq_u16(vld1q_u16(y.as_ptr().add(x)), mask_v);
      let y_vec_hi = vandq_u16(vld1q_u16(y.as_ptr().add(x + 8)), mask_v);
      let u_lo_u16 = vandq_u16(vld1q_u16(u.as_ptr().add(x)), mask_v);
      let u_hi_u16 = vandq_u16(vld1q_u16(u.as_ptr().add(x + 8)), mask_v);
      let v_lo_u16 = vandq_u16(vld1q_u16(v.as_ptr().add(x)), mask_v);
      let v_hi_u16 = vandq_u16(vld1q_u16(v.as_ptr().add(x + 8)), mask_v);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(u_lo_u16), bias_v);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(u_hi_u16), bias_v);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(v_lo_u16), bias_v);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(v_hi_u16), bias_v);

      // Widen each i16x8 → two i32x4 halves. Chroma is 1:1 with Y,
      // so we compute 8 chroma per Y-half directly.
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
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
      );

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_u16 = vandq_u16(vld1q_u16(a_ptr.add(x)), mask_v);
          let a_hi_u16 = vandq_u16(vld1q_u16(a_ptr.add(x + 8)), mask_v);
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar. NEON's
          // `vshrq_n_u16` requires a literal const generic shift, but
          // `BITS - 8` is not a stable const expression on a const
          // generic — `vshlq_u16` with a negative count vector
          // performs the same logical right shift dynamically.
          let a_shr = vdupq_n_s16(-((BITS - 8) as i16));
          let a_lo_shifted = vshlq_u16(a_lo_u16, a_shr);
          let a_hi_shifted = vshlq_u16(a_hi_u16, a_shr);
          vcombine_u8(vqmovn_u16(a_lo_shifted), vqmovn_u16(a_hi_shifted))
        } else {
          alpha_u8
        };
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, a_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p_n_to_rgba_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p_n_to_rgb_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON YUV 4:4:4 planar high-bit-depth → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, false>(
      y, u, v, rgb_out, width, matrix, full_range, None,
    );
  }
}

/// NEON sibling of [`yuv_444p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth) — matches `scalar::yuv_444p_n_to_rgba_u16_row`.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_u16_row`], plus
/// `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, false>(
      y, u, v, rgba_out, width, matrix, full_range, None,
    );
  }
}

/// NEON YUVA 4:4:4 planar high-bit-depth → **native-depth `u16`**
/// packed RGBA with the per-pixel alpha element **sourced from
/// `a_src`** (already at the source's native bit depth — no depth
/// conversion) instead of being the opaque maximum `(1 << BITS) - 1`.
/// Same numerical contract as [`yuv_444p_n_to_rgba_u16_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, true>(
      y,
      u,
      v,
      rgba_out,
      width,
      matrix,
      full_range,
      Some(a_src),
    );
  }
}

/// Shared NEON high-bit YUV 4:4:4 → native-depth `u16` kernel for
/// [`yuv_444p_n_to_rgb_u16_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `vst3q_u16`), [`yuv_444p_n_to_rgba_u16_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `vst4q_u16` with constant alpha
/// `(1 << BITS) - 1`) and [`yuv_444p_n_to_rgba_u16_with_alpha_src_row`]
/// (`ALPHA = true, ALPHA_SRC = true`, `vst4q_u16` with the alpha lane
/// loaded from `a_src` and masked to native bit depth — no shift since
/// both the source alpha and the u16 output element are at the same
/// native bit depth).
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  a_src: Option<&[u16]>,
) {
  // Compile-time guard — `out_max = ((1 << BITS) - 1) as i16` below
  // silently wraps to -1 at BITS=16, corrupting the u16 clamp. The
  // dedicated 16-bit u16-output path is `yuv_444p16_to_rgb_u16_row`.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let mask_v = vdupq_n_u16(scalar::bits_mask::<BITS>());
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vandq_u16(vld1q_u16(y.as_ptr().add(x)), mask_v);
      let y_vec_hi = vandq_u16(vld1q_u16(y.as_ptr().add(x + 8)), mask_v);
      let u_lo_u16 = vandq_u16(vld1q_u16(u.as_ptr().add(x)), mask_v);
      let u_hi_u16 = vandq_u16(vld1q_u16(u.as_ptr().add(x + 8)), mask_v);
      let v_lo_u16 = vandq_u16(vld1q_u16(v.as_ptr().add(x)), mask_v);
      let v_hi_u16 = vandq_u16(vld1q_u16(v.as_ptr().add(x + 8)), mask_v);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(u_lo_u16), bias_v);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(u_hi_u16), bias_v);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(v_lo_u16), bias_v);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(v_hi_u16), bias_v);

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

      let r_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let (a_lo_v, a_hi_v) = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // No depth conversion — both source alpha and u16 output are
          // at the same native bit depth (BITS), so just mask off any
          // over-range bits to match the scalar reference.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let lo = vandq_u16(vld1q_u16(a_ptr.add(x)), mask_v);
          let hi = vandq_u16(vld1q_u16(a_ptr.add(x + 8)), mask_v);
          (lo, hi)
        } else {
          (alpha_u16, alpha_u16)
        };
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, a_lo_v);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, a_hi_v);
        vst4q_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u16(out.as_mut_ptr().add(x * 4 + 32), rgba_hi);
      } else {
        let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
        let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
        vst3q_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u16(out.as_mut_ptr().add(x * 3 + 24), rgb_hi);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p_n_to_rgba_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// Clamps an i16x8 vector to `[0, max]` and reinterprets to u16x8.
/// Used by native-depth u16 output paths (10/12/14 bit) to avoid
/// `vqmovun_s16`'s u8 saturation.
#[inline(always)]
fn clamp_u16_max(v: int16x8_t, zero_v: int16x8_t, max_v: int16x8_t) -> uint16x8_t {
  unsafe { vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(v, zero_v), max_v)) }
}

/// NEON high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}: P010, P012)
/// → packed **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Differences
/// from [`yuv_420p_n_to_rgb_row`]:
/// - UV is semi‑planar interleaved (`U0, V0, U1, V1, …`), split in
///   one shot via `vld2q_u16` (returns separate U and V vectors).
/// - Each `u16` load is **right‑shifted by `16 - BITS`** — 6 for
///   P010, 4 for P012 — extracting the `BITS` active bits from the
///   high bits of each `u16` and clearing the low bits. The shift
///   runs via `vshlq_u16` with a negative loop‑invariant count so a
///   single kernel serves all supported bit depths.
///
/// After the shift, the rest of the pipeline is identical to the
/// low‑bit‑packed planar path — same `chroma_i16x8` / `scale_y` /
/// `chroma_dup` / `vst3q_u8` write, with `range_params_n::<BITS, 8>`
/// scaling.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] across all
/// supported `BITS` values.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
/// 4. `BITS` must be one of `{10, 12}`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON high-bit-packed semi-planar 4:2:0 → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as [`p_n_to_rgb_row`];
/// 4 bpp store with constant alpha.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON kernel for [`p_n_to_rgb_row`] (`ALPHA = false`,
/// `vst3q_u8`) and [`p_n_to_rgba_row`] (`ALPHA = true`, `vst4q_u8`
/// with constant `0xFF` alpha vector).
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // P016 (BITS=16) routes through `p16_to_rgb_or_rgba_row` (i64 chroma);
  // attempting `::<16, _>` here would silently overflow on high chroma.
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    // `vshlq_u16` performs right shift when the count is negative.
    // Count = -(16 - BITS) extracts the `BITS` active high bits.
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y pixels in two u16x8 loads, right-shifted by 16-BITS to
      // extract the active bits from the high-bit packing.
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);

      // Semi‑planar UV: `vld2q_u16` loads 16 interleaved `u16` elements
      // and returns (evens, odds) = (U, V) in one shot.
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = vshlq_u16(uv_pair.0, shr_count);
      let v_vec = vshlq_u16(uv_pair.1, shr_count);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

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

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
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

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_to_rgba_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_to_rgb_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **native‑depth `u16`** RGB (low‑bit‑packed output,
/// `yuv420p10le` / `yuv420p12le` convention — not P010/P012).
///
/// Same structure as [`super::neon::p_n_to_rgb_row`] up to the
/// chroma compute; the only differences are:
/// - `range_params_n::<BITS, BITS>` → larger scales targeting the
///   native‑depth output range.
/// - Clamp is explicit min/max to `[0, (1 << BITS) - 1]` via
///   [`clamp_u10`](crate::row::arch::neon::clamp_u10) — the helper
///   name is historical; the actual max is derived from `BITS` at
///   the call site (1023 for P010, 4095 for P012).
/// - Writes use two `vst3q_u16` calls per 16‑pixel block.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_u16_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p_n_to_rgba_row`] for native-depth `u16` output.
/// Alpha samples are `(1 << BITS) - 1` (opaque maximum at the input
/// bit depth) — matches `scalar::p_n_to_rgba_u16_row`. P016 has its
/// own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`], plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON Pn → native-depth `u16` kernel. `ALPHA = false` writes
/// RGB triples via `vst3q_u16`; `ALPHA = true` writes RGBA quads via
/// `vst4q_u16` with constant alpha `(1 << BITS) - 1`. P016 has its
/// own kernel family — never routed here.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = vshlq_u16(uv_pair.0, shr_count);
      let v_vec = vshlq_u16(uv_pair.1, shr_count);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

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

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      if ALPHA {
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, alpha_u16);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, alpha_u16);
        vst4q_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u16(out.as_mut_ptr().add(x * 4 + 32), rgba_hi);
      } else {
        let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
        let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
        vst3q_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u16(out.as_mut_ptr().add(x * 3 + 24), rgb_hi);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_to_rgba_u16_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_to_rgb_u16_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON NV12 → packed RGB (UV-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_or_nv21_to_rgb_or_rgba_row_impl`]:
///
/// 1. **NEON must be available on the current CPU.** Direct callers
///    are responsible for verifying this; the dispatcher in
///    [`crate::row::nv12_to_rgb_row`] checks it.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `uv_half.len() >= width` (interleaved UV bytes, 2 per chroma pair).
/// 5. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON NV21 → packed RGB (VU-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_to_rgb_row`]; `vu_half` carries the same
/// number of bytes (`>= width`) but in V-then-U order per chroma
/// pair.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
      y, vu_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON NV12 → packed RGBA (R, G, B, `0xFF` per pixel). Same
/// contract as [`nv12_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8`. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv12_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes (one extra byte per pixel for the opaque
/// alpha).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON NV21 → packed RGBA (R, G, B, `0xFF` per pixel). Same
/// contract as [`nv21_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8`. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv21_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
      y, vu_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared NEON NV12/NV21 kernel at 3 bpp (RGB) or 4 bpp + opaque
/// alpha (RGBA). `SWAP_UV = false` selects NV12 (even byte = U, odd =
/// V); `SWAP_UV = true` selects NV21 (even = V, odd = U). `ALPHA =
/// true` writes via `vst4q_u8` with constant `0xFF` alpha; `ALPHA =
/// false` writes via `vst3q_u8`. Both const generics drive
/// compile-time monomorphization — branches are eliminated and each
/// of the four wrappers produces byte‑identical output to the scalar
/// reference.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.** The dispatcher
///    verifies this; direct callers are responsible.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (2 × (width / 2) interleaved bytes).
/// 5. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
///
/// Bounds are `debug_assert`-checked; release builds trust the caller
/// because the kernel uses unchecked pointer arithmetic (`vld1q_u8`,
/// `vld2_u8`, `vst3q_u8` / `vst4q_u8`).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn nv12_or_nv21_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu_half.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 16 <= width` loop
  // condition and the caller‑promised slice lengths above.
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
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      // 16 Y pixels → 8 chroma pairs. `vld2_u8` loads 16 interleaved
      // bytes and splits into (even-offset bytes, odd-offset bytes).
      // For NV12: even=U, odd=V. For NV21: even=V, odd=U, so we
      // swap which lane becomes `u_vec`. The `const SWAP_UV` makes
      // this a compile-time choice — no runtime branch.
      let uv_pair = vld2_u8(uv_or_vu_half.as_ptr().add(x));
      let (u_vec, v_vec) = if SWAP_UV {
        (uv_pair.1, uv_pair.0)
      } else {
        (uv_pair.0, uv_pair.1)
      };

      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

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

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
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

    // Scalar tail for the 0..14 leftover pixels. Dispatch to the
    // matching scalar kernel based on SWAP_UV × ALPHA.
    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu_half[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      match (SWAP_UV, ALPHA) {
        (false, false) => {
          scalar::nv12_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, false) => {
          scalar::nv21_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (false, true) => {
          scalar::nv12_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, true) => {
          scalar::nv21_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
      }
    }
  }
}

/// NEON NV24 → packed RGB (UV-ordered, 4:4:4). Thin wrapper over
/// [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] with
/// `ALPHA = false` (so `out.len() >= width * 3` specializes to
/// `rgb_out.len() >= 3 * width`):
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`.
/// 3. `uv.len() >= 2 * width`.
/// 4. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, false>(y, uv, rgb_out, width, matrix, full_range);
  }
}

/// NEON NV42 → packed RGB (VU-ordered, 4:4:4). Thin wrapper over
/// [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv24_to_rgb_row`]; `vu` carries the same
/// `2 * width` bytes but in V-then-U order per chroma pair.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, false>(y, vu, rgb_out, width, matrix, full_range);
  }
}

/// NEON NV24 → packed RGBA (R, G, B, `0xFF` per pixel). Same
/// contract as [`nv24_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8`. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv24_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv24_to_rgba_row(
  y: &[u8],
  uv: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, true>(y, uv, rgba_out, width, matrix, full_range);
  }
}

/// NEON NV42 → packed RGBA (R, G, B, `0xFF` per pixel). Same
/// contract as [`nv42_to_rgb_row`] but writes 4 bytes per pixel via
/// `vst4q_u8`. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv42_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv42_to_rgba_row(
  y: &[u8],
  vu: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, true>(y, vu, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON NV24/NV42 kernel (4:4:4 semi-planar) at 3 bpp (RGB)
/// or 4 bpp + opaque alpha (RGBA). Unlike
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`], chroma is not
/// subsampled — one UV pair per Y pixel, so the chroma-duplication
/// step (`vzip*`) disappears: compute 16 chroma values per 16 Y
/// pixels directly.
///
/// `SWAP_UV = false` selects NV24 (even byte = U, odd = V);
/// `SWAP_UV = true` selects NV42 (even = V, odd = U). `ALPHA = true`
/// writes via `vst4q_u8` with constant `0xFF` alpha; `ALPHA = false`
/// writes via `vst3q_u8`. Both const generics drive compile-time
/// monomorphization.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`.
/// 3. `uv_or_vu.len() >= 2 * width` (one UV pair per Y pixel).
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
///
/// No width parity constraint (4:4:4).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn nv24_or_nv42_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 16 <= width` loop and
  // the caller-promised slice lengths.
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
      // 16 Y pixels → 16 chroma pairs = 32 bytes. `vld2q_u8`
      // deinterleaves 32 bytes into (even-offset, odd-offset) — 16
      // bytes each.
      let uv_pair = vld2q_u8(uv_or_vu.as_ptr().add(x * 2));
      let (u_vec, v_vec) = if SWAP_UV {
        (uv_pair.1, uv_pair.0)
      } else {
        (uv_pair.0, uv_pair.1)
      };

      // Widen Y, U, V halves to i16x8.
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_vec))), mid128);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u_vec))), mid128);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_vec))), mid128);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_vec))), mid128);

      // Widen each i16x8 to two i32x4 halves for the Q15 multiply.
      let u_lo_a = vmovl_s16(vget_low_s16(u_lo_i16));
      let u_lo_b = vmovl_s16(vget_high_s16(u_lo_i16));
      let u_hi_a = vmovl_s16(vget_low_s16(u_hi_i16));
      let u_hi_b = vmovl_s16(vget_high_s16(u_hi_i16));
      let v_lo_a = vmovl_s16(vget_low_s16(v_lo_i16));
      let v_lo_b = vmovl_s16(vget_high_s16(v_lo_i16));
      let v_hi_a = vmovl_s16(vget_low_s16(v_hi_i16));
      let v_hi_b = vmovl_s16(vget_high_s16(v_hi_i16));

      // u_d / v_d = (u * c_scale + RND) >> 15 — i32x4 lanes.
      let u_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(v_hi_b, c_scale_v), rnd_v));

      // Compute chroma per channel — 8 results covering 8 Y pixels
      // per half (no duplication, since UV is 1:1 with Y).
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
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

    // Scalar tail for 0..15 leftover pixels.
    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu[x * 2..width * 2];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      match (SWAP_UV, ALPHA) {
        (false, false) => {
          scalar::nv24_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, false) => {
          scalar::nv42_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (false, true) => {
          scalar::nv24_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, true) => {
          scalar::nv42_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
      }
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
    yuv_444_to_rgb_or_rgba_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
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
    yuv_444_to_rgb_or_rgba_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON YUV 4:4:4 kernel for [`yuv_444_to_rgb_row`]
/// (`ALPHA = false`, `vst3q_u8`) and [`yuv_444_to_rgba_row`]
/// (`ALPHA = true`, `vst4q_u8` with constant `0xFF` alpha). Math is
/// byte-identical to `scalar::yuv_444_to_rgb_or_rgba_row::<ALPHA>`;
/// only the per-block store intrinsic differs.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
///
/// No width parity constraint (4:4:4).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
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
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
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

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA {
        scalar::yuv_444_to_rgba_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yuv_444_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// The helpers below wrap NEON register‑only intrinsics (shifts, adds,
// multiplies, narrowing conversions, lane movers). None of them touch
// memory or take pointers, so there is no safety invariant to hoist to
// the caller — the functions themselves are safe. The `unsafe { ... }`
// blocks inside are only required because `core::arch::aarch64`
// intrinsics are marked `unsafe fn` in the standard library.
//
// `#[inline(always)]` guarantees these are inlined into the NEON‑
// enabled caller (`yuv_420_to_rgb_row` has
// `#[target_feature(enable = "neon")]`), so the intrinsics execute in
// a context where NEON is explicitly enabled — not just implicitly
// via the aarch64 target's default feature set.

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: int32x4_t) -> int32x4_t {
  unsafe { vshrq_n_s32::<15>(v) }
}

/// Build an i16x8 channel chroma vector from the 8 paired i32 chroma
/// samples. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`.
#[inline(always)]
fn chroma_i16x8(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d_lo: int32x4_t,
  v_d_lo: int32x4_t,
  u_d_hi: int32x4_t,
  v_d_hi: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_lo), vmulq_s32(cv, v_d_lo)),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_hi), vmulq_s32(cv, v_d_hi)),
      rnd,
    ));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` returned as i16x8 (8 Y pixels).
#[inline(always)]
fn scale_y(
  y_i16: int16x8_t,
  y_off_v: int16x8_t,
  y_scale_v: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let shifted = vsubq_s16(y_i16, y_off_v);
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_low_s16(shifted)), y_scale_v),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_high_s16(shifted)), y_scale_v),
      rnd,
    ));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
  }
}

// ===== 16-bit YUV → RGB ==================================================
//
// At 16-bit, two precision issues arise compared to the 9/10/12/14-bit generic:
//
// 1. The chroma bias (32768) and full-range u16 values (0..65535) do not fit
//    in i16, so all bias-subtractions happen in i32 after unsigned widening
//    (`vmovl_u16` → `vreinterpretq_s32_u32`).
//
// 2. For u16 output: `c_scale ≈ 37445` (limited range), so `coeff * u_d`
//    reaches ~2.17×10⁹ > i32 max; `y_scale ≈ 38304`, so `(y−y_off)*y_scale`
//    reaches ~2.35×10⁹ > i32 max. Both Y and chroma are widened to i64 via
//    `vmull_s32` and shifted back with `vshrq_n_s64::<15>`.
//
// For u8 output: `c_scale ≈ 127`, so i32 is sufficient throughout.

/// NEON 16-bit planar YUV 4:2:0 → packed 8-bit RGB.
///
/// Byte-identical to [`scalar::yuv_420p16_to_rgb_row`].
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON 16-bit YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON 16-bit YUVA 4:2:0 → packed **8-bit RGBA** with the per-pixel
/// alpha byte **sourced from `a_src`** (depth-converted via `>> 8` to
/// fit `u8`). 16-bit alpha is full-range u16 — no AND-mask step.
/// Same numerical contract as [`yuv_420p16_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgba_with_alpha_src_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_row::<true, true>(
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

/// Shared NEON 16-bit YUV 4:2:0 kernel for [`yuv_420p16_to_rgb_row`]
/// (`ALPHA = false, ALPHA_SRC = false`, `vst3q_u8`),
/// [`yuv_420p16_to_rgba_row`] (`ALPHA = true, ALPHA_SRC = false`,
/// `vst4q_u8` with constant `0xFF` alpha) and
/// [`yuv_420p16_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `vst4q_u8` with the alpha lane loaded from
/// `a_src` and depth-converted via `vshrq_n_u16::<8>`).
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>(); // = 32768
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let u_vec = vld1q_u16(u_half.as_ptr().add(x / 2));
      let v_vec = vld1q_u16(v_half.as_ptr().add(x / 2));

      // Unsigned-widen U/V to i32, subtract bias (32768 — does not fit i16).
      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // i32 chroma is enough for u8 output (c_scale ≈ 127 keeps u_d small).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y_u16_to_i16(y_vec_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_vec_hi, y_off_v, y_scale_v, rnd_v);

      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — no mask, just `>> 8` to
          // fit u8. `vshrq_n_u16` takes a const literal shift; 8 is
          // a literal here so the intrinsic is well-formed.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_u16 = vshrq_n_u16::<8>(vld1q_u16(a_ptr.add(x)));
          let a_hi_u16 = vshrq_n_u16::<8>(vld1q_u16(a_ptr.add(x + 8)));
          vcombine_u8(vqmovn_u16(a_lo_u16), vqmovn_u16(a_hi_u16))
        } else {
          alpha_u8
        };
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, a_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p16_to_rgba_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p16_to_rgba_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON 16-bit planar YUV 4:2:0 → packed native-depth u16 RGB.
///
/// Both Y scaling and chroma multiply run in i64 (via `vmull_s32` +
/// `vshrq_n_s64::<15>`) to avoid i32 overflow at 16-bit limited-range scales.
/// Byte-identical to [`scalar::yuv_420p16_to_rgb_u16_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<false>(
      y, u_half, v_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON sibling of [`yuv_420p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha is `0xFFFF` — matches `scalar::yuv_420p16_to_rgba_u16_row`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<true>(
      y, u_half, v_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared NEON 16-bit YUV 4:2:0 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `vst3q_u16`; `ALPHA = true`
/// writes RGBA quads via `vst4q_u16` with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = vdupq_n_u16(0xFFFF);
    let rnd_v = vdupq_n_s32(RND);
    let rnd64 = vdupq_n_s64(RND as i64);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_d = vdup_n_s32(y_scale); // int32x2_t for vmull_s32
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let u_vec = vld1q_u16(u_half.as_ptr().add(x / 2));
      let v_vec = vld1q_u16(v_half.as_ptr().add(x / 2));

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // i64 chroma: coeff * u_d can reach ~2.17×10⁹ at 16-bit scales.
      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      // Duplicate each chroma value into the slot for its 2 Y pixels.
      let r_cd_lo0 = vzip1q_s32(r_ch_lo, r_ch_lo);
      let r_cd_lo1 = vzip2q_s32(r_ch_lo, r_ch_lo);
      let r_cd_hi0 = vzip1q_s32(r_ch_hi, r_ch_hi);
      let r_cd_hi1 = vzip2q_s32(r_ch_hi, r_ch_hi);
      let g_cd_lo0 = vzip1q_s32(g_ch_lo, g_ch_lo);
      let g_cd_lo1 = vzip2q_s32(g_ch_lo, g_ch_lo);
      let g_cd_hi0 = vzip1q_s32(g_ch_hi, g_ch_hi);
      let g_cd_hi1 = vzip2q_s32(g_ch_hi, g_ch_hi);
      let b_cd_lo0 = vzip1q_s32(b_ch_lo, b_ch_lo);
      let b_cd_lo1 = vzip2q_s32(b_ch_lo, b_ch_lo);
      let b_cd_hi0 = vzip1q_s32(b_ch_hi, b_ch_hi);
      let b_cd_hi1 = vzip2q_s32(b_ch_hi, b_ch_hi);

      // i64 Y: (y - y_off) * y_scale can reach ~2.35×10⁹ at limited range.
      let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_lo)));
      let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_lo)));
      let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_hi)));
      let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_hi)));
      let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
      let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
      let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
      let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

      // Add Y + chroma; vqmovun_s32 saturates i32→u16 (clamps to [0, 65535]).
      let r_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, r_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, r_cd_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, r_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, r_cd_hi1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, g_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, g_cd_lo1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, g_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, g_cd_hi1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, b_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, b_cd_lo1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, b_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, b_cd_hi1)),
      );

      if ALPHA {
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_lo_u16, g_lo_u16, b_lo_u16, alpha_u16),
        );
        vst4q_u16(
          out.as_mut_ptr().add(x * 4 + 32),
          uint16x8x4_t(r_hi_u16, g_hi_u16, b_hi_u16, alpha_u16),
        );
      } else {
        vst3q_u16(
          out.as_mut_ptr().add(x * 3),
          uint16x8x3_t(r_lo_u16, g_lo_u16, b_lo_u16),
        );
        vst3q_u16(
          out.as_mut_ptr().add(x * 3 + 24),
          uint16x8x3_t(r_hi_u16, g_hi_u16, b_hi_u16),
        );
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_420p16_to_rgba_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON YUV 4:4:4 planar **16-bit** → packed **8-bit** RGB. Same i32
/// chroma pipeline as 10/12/14 (u8 output clamps `c_scale` down);
/// 1:1 chroma per Y pixel, no width parity.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but with full `u16` samples.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// NEON YUV 4:4:4 planar **16-bit** → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p16_to_rgb_row`].
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON 16-bit YUV 4:4:4 kernel for [`yuv_444p16_to_rgb_row`]
/// (`ALPHA = false`, `vst3q_u8`) and [`yuv_444p16_to_rgba_row`]
/// (`ALPHA = true`, `vst4q_u8` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let u_vec_lo = vld1q_u16(u.as_ptr().add(x));
      let u_vec_hi = vld1q_u16(u.as_ptr().add(x + 8));
      let v_vec_lo = vld1q_u16(v.as_ptr().add(x));
      let v_vec_hi = vld1q_u16(v.as_ptr().add(x + 8));

      // Unsigned-widen + subtract 32768 in i32 (doesn't fit i16).
      let u_lo_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec_lo))),
        bias_v,
      );
      let u_lo_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec_lo))),
        bias_v,
      );
      let u_hi_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec_hi))),
        bias_v,
      );
      let u_hi_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec_hi))),
        bias_v,
      );
      let v_lo_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec_lo))),
        bias_v,
      );
      let v_lo_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec_lo))),
        bias_v,
      );
      let v_hi_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec_hi))),
        bias_v,
      );
      let v_hi_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec_hi))),
        bias_v,
      );

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

      let y_scaled_lo = scale_y_u16_to_i16(y_vec_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_vec_hi, y_off_v, y_scale_v, rnd_v);

      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );

      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_444p16_to_rgba_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON YUV 4:4:4 planar **16-bit** → packed **native-depth u16** RGB.
/// i64 chroma + i64 Y (same widening as `yuv_420p16_to_rgb_u16_row`);
/// full-width U/V (no chroma duplication step).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_u16_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`yuv_444p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF` (opaque maximum at u16 range) —
/// matches `scalar::yuv_444p16_to_rgba_u16_row`.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_u16_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON 16-bit YUV 4:4:4 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `vst3q_u16`; `ALPHA = true`
/// writes RGBA quads via `vst4q_u16` with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let rnd64 = vdupq_n_s64(RND as i64);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_d = vdup_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(0xFFFF);

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 Y + 8 U + 8 V per iter — tighter block than 16 Y because
      // i64 chroma narrows throughput; matches the yuv_420p16 u16
      // kernel's cadence.
      let y_vec = vld1q_u16(y.as_ptr().add(x));
      let u_vec = vld1q_u16(u.as_ptr().add(x));
      let v_vec = vld1q_u16(v.as_ptr().add(x));

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // i64 chroma matches `yuv_420p16_to_rgb_u16_row`. 8 chroma
      // values computed as two `chroma_i64x4` calls.
      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      // i64 Y: 8 values as two i32x4 halves, scaled via i64 helper.
      let y_lo_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec)));
      let y_hi_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec)));
      let ys_lo = scale_y_u16_i64(y_lo_i32, y_off_v, y_scale_d, rnd64);
      let ys_hi = scale_y_u16_i64(y_hi_i32, y_off_v, y_scale_d, rnd64);

      let r_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, r_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, r_ch_hi)),
      );
      let g_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, g_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, g_ch_hi)),
      );
      let b_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, b_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, b_ch_hi)),
      );

      if ALPHA {
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_u16, g_u16, b_u16, alpha_u16),
        );
      } else {
        vst3q_u16(
          out.as_mut_ptr().add(x * 3),
          uint16x8x3_t(r_u16, g_u16, b_u16),
        );
      }
      x += 8;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_444p16_to_rgba_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON P016 (semi-planar 16-bit) → packed 8-bit RGB.
///
/// UV is interleaved (`U0, V0, U1, V1, …`), split via `vld2q_u16`.
/// Byte-identical to [`scalar::p16_to_rgb_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON P016 (semi-planar 4:2:0, full 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgba_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON P016 kernel for [`p16_to_rgb_row`] (`ALPHA = false`,
/// `vst3q_u8`) and [`p16_to_rgba_row`] (`ALPHA = true`, `vst4q_u8`
/// with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = uv_pair.0;
      let v_vec = uv_pair.1;

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y_u16_to_i16(y_vec_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_vec_hi, y_off_v, y_scale_v, rnd_v);

      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );

      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON P016 (semi-planar 16-bit) → packed native-depth u16 RGB.
///
/// Byte-identical to [`scalar::p16_to_rgb_u16_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p16_to_rgba_row`] for native-depth `u16` output.
/// Alpha is `0xFFFF` — matches `scalar::p16_to_rgba_u16_row`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgba_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `vst3q_u16`; `ALPHA = true`
/// writes RGBA quads via `vst4q_u16` with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = vdupq_n_u16(0xFFFF);
    let rnd_v = vdupq_n_s32(RND);
    let rnd64 = vdupq_n_s64(RND as i64);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_d = vdup_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = uv_pair.0;
      let v_vec = uv_pair.1;

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      let r_cd_lo0 = vzip1q_s32(r_ch_lo, r_ch_lo);
      let r_cd_lo1 = vzip2q_s32(r_ch_lo, r_ch_lo);
      let r_cd_hi0 = vzip1q_s32(r_ch_hi, r_ch_hi);
      let r_cd_hi1 = vzip2q_s32(r_ch_hi, r_ch_hi);
      let g_cd_lo0 = vzip1q_s32(g_ch_lo, g_ch_lo);
      let g_cd_lo1 = vzip2q_s32(g_ch_lo, g_ch_lo);
      let g_cd_hi0 = vzip1q_s32(g_ch_hi, g_ch_hi);
      let g_cd_hi1 = vzip2q_s32(g_ch_hi, g_ch_hi);
      let b_cd_lo0 = vzip1q_s32(b_ch_lo, b_ch_lo);
      let b_cd_lo1 = vzip2q_s32(b_ch_lo, b_ch_lo);
      let b_cd_hi0 = vzip1q_s32(b_ch_hi, b_ch_hi);
      let b_cd_hi1 = vzip2q_s32(b_ch_hi, b_ch_hi);

      let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_lo)));
      let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_lo)));
      let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_hi)));
      let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_hi)));
      let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
      let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
      let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
      let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

      let r_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, r_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, r_cd_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, r_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, r_cd_hi1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, g_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, g_cd_lo1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, g_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, g_cd_hi1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, b_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, b_cd_lo1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, b_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, b_cd_hi1)),
      );

      if ALPHA {
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_lo_u16, g_lo_u16, b_lo_u16, alpha_u16),
        );
        vst4q_u16(
          out.as_mut_ptr().add(x * 4 + 32),
          uint16x8x4_t(r_hi_u16, g_hi_u16, b_hi_u16, alpha_u16),
        );
      } else {
        vst3q_u16(
          out.as_mut_ptr().add(x * 3),
          uint16x8x3_t(r_lo_u16, g_lo_u16, b_lo_u16),
        );
        vst3q_u16(
          out.as_mut_ptr().add(x * 3 + 24),
          uint16x8x3_t(r_hi_u16, g_hi_u16, b_hi_u16),
        );
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ===== Pn 4:4:4 (semi-planar high-bit-packed) → RGB =======================
//
// NEON kernels for `p_n_444_to_rgb_*<BITS>` (BITS ∈ {10, 12}, Q15 i32
// pipeline) and `p_n_444_16_to_rgb_*` (BITS = 16, parallel i64-chroma
// for u16 output). The inner math mirrors `yuv_444p_n_to_rgb_row`
// (chroma is 1:1 with Y — no horizontal duplication) but the chroma
// load uses `vld2q_u16` to deinterleave the full-width UV plane in
// register, like `nv24_to_rgb_row`. Each iteration consumes 16 Y
// pixels and 32 UV `u16` elements (= 16 interleaved U/V pairs).

/// NEON Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **u8** RGB.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// NEON Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgba_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON Pn 4:4:4 high-bit-packed kernel for
/// [`p_n_444_to_rgb_row`] (`ALPHA = false`, `vst3q_u8`) and
/// [`p_n_444_to_rgba_row`] (`ALPHA = true`, `vst4q_u8` with constant
/// `0xFF` alpha).
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    // `vshlq_u16(_, vdupq_n_s16(-(16 - BITS)))` is a logical right shift
    // by `(16 - BITS)` (NEON variable shift treats negative count as right
    // shift). Same pattern as `p_n_to_rgb_row<BITS>` for the high-bit
    // packing extraction.
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y pixels in two u16x8 loads; high-bit-extracted via the
      // logical right shift.
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);

      // 32 UV elements = 16 interleaved (U, V) pairs. Two `vld2q_u16`
      // calls deinterleave them into two pairs of (U, V) u16x8 vectors.
      let uv_pair_lo = vld2q_u16(uv_full.as_ptr().add(x * 2));
      let uv_pair_hi = vld2q_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_u16 = vshlq_u16(uv_pair_lo.0, shr_count);
      let v_lo_u16 = vshlq_u16(uv_pair_lo.1, shr_count);
      let u_hi_u16 = vshlq_u16(uv_pair_hi.0, shr_count);
      let v_hi_u16 = vshlq_u16(uv_pair_hi.1, shr_count);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(u_lo_u16), bias_v);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(u_hi_u16), bias_v);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(v_lo_u16), bias_v);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(v_hi_u16), bias_v);

      // Widen each i16x8 → two i32x4 halves. 1:1 chroma per Y, no
      // duplication.
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
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
      );

      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_to_rgba_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_to_rgb_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB. Output is low-bit-packed.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p_n_444_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth) — matches `scalar::p_n_444_to_rgba_u16_row`.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON Pn 4:4:4 high-bit-packed → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `vst3q_u16`; `ALPHA = true`
/// writes RGBA quads via `vst4q_u16` with constant alpha
/// `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let zero_v = vdupq_n_s16(0);
    let max_v = vdupq_n_s16(out_max);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);

      let uv_pair_lo = vld2q_u16(uv_full.as_ptr().add(x * 2));
      let uv_pair_hi = vld2q_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_u16 = vshlq_u16(uv_pair_lo.0, shr_count);
      let v_lo_u16 = vshlq_u16(uv_pair_lo.1, shr_count);
      let u_hi_u16 = vshlq_u16(uv_pair_hi.0, shr_count);
      let v_hi_u16 = vshlq_u16(uv_pair_hi.1, shr_count);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(u_lo_u16), bias_v);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(u_hi_u16), bias_v);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(v_lo_u16), bias_v);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(v_hi_u16), bias_v);

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

      // Clamp [0, out_max] in i16, then reinterpret as u16 (all values
      // are non-negative after clamp).
      let r_lo = clamp_u16_max(vaddq_s16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vaddq_s16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vaddq_s16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vaddq_s16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vaddq_s16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vaddq_s16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, alpha_u16);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, alpha_u16);
        vst4q_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u16(out.as_mut_ptr().add(x * 4 + 32), rgba_hi);
      } else {
        let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
        let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
        vst3q_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u16(out.as_mut_ptr().add(x * 3 + 24), rgb_hi);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_to_rgba_u16_row::<BITS>(
          tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::p_n_444_to_rgb_u16_row::<BITS>(
          tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB.
/// Y and chroma both stay on i32 (output-range scaling keeps `coeff
/// × u_d` within i32 for u8 output). Mirror `yuv_444p16_to_rgb_row`
/// with full-width interleaved UV via `vld2q_u16`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// NEON P416 (semi-planar 4:4:4, 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`p_n_444_16_to_rgb_row`].
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgba_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON P416 (semi-planar 4:4:4, 16-bit) kernel for
/// [`p_n_444_16_to_rgb_row`] (`ALPHA = false`, `vst3q_u8`) and
/// [`p_n_444_16_to_rgba_row`] (`ALPHA = true`, `vst4q_u8` with constant
/// `0xFF` alpha).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));

      // 16 chroma pairs per iter — two `vld2q_u16` calls deinterleave
      // 32 UV `u16` elements into two pairs of (U, V) u16x8 vectors.
      let uv_pair_lo = vld2q_u16(uv_full.as_ptr().add(x * 2));
      let uv_pair_hi = vld2q_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_vec_lo = uv_pair_lo.0;
      let v_vec_lo = uv_pair_lo.1;
      let u_vec_hi = uv_pair_hi.0;
      let v_vec_hi = uv_pair_hi.1;

      // Unsigned-widen + bias subtract in i32 (16-bit chroma can't fit
      // i16 after subtracting 32768).
      let u_lo_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec_lo))),
        bias_v,
      );
      let u_lo_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec_lo))),
        bias_v,
      );
      let u_hi_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec_hi))),
        bias_v,
      );
      let u_hi_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec_hi))),
        bias_v,
      );
      let v_lo_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec_lo))),
        bias_v,
      );
      let v_lo_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec_lo))),
        bias_v,
      );
      let v_hi_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec_hi))),
        bias_v,
      );
      let v_hi_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec_hi))),
        bias_v,
      );

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

      let y_scaled_lo = scale_y_u16_to_i16(y_vec_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_vec_hi, y_off_v, y_scale_v, rnd_v);

      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );

      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_16_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_16_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON P416 (semi-planar 4:4:4, 16-bit) → packed **native-depth u16**
/// RGB. i64 chroma + i64 Y (chroma matrix multiply-add overflows i32
/// at u16 output for the BT.2020 blue coefficient). 8 pixels per iter.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p_n_444_16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF` (opaque maximum at u16 range) —
/// matches `scalar::p_n_444_16_to_rgba_u16_row`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgba_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON P416 (semi-planar 4:4:4, 16-bit) → native-depth `u16`
/// kernel. `ALPHA = false` writes RGB triples via `vst3q_u16`;
/// `ALPHA = true` writes RGBA quads via `vst4q_u16` with constant alpha
/// `0xFFFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let rnd64 = vdupq_n_s64(RND as i64);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_d = vdup_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(0xFFFF);

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 Y + 8 chroma pairs per iter — tighter block because i64
      // chroma narrows throughput; matches `yuv_444p16_to_rgb_u16_row`.
      let y_vec = vld1q_u16(y.as_ptr().add(x));
      let uv_pair = vld2q_u16(uv_full.as_ptr().add(x * 2));
      let u_vec = uv_pair.0;
      let v_vec = uv_pair.1;

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // i64 chroma — 8 chroma values via two `chroma_i64x4` calls.
      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      let y_lo_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec)));
      let y_hi_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec)));
      let ys_lo = scale_y_u16_i64(y_lo_i32, y_off_v, y_scale_d, rnd64);
      let ys_hi = scale_y_u16_i64(y_hi_i32, y_off_v, y_scale_d, rnd64);

      let r_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, r_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, r_ch_hi)),
      );
      let g_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, g_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, g_ch_hi)),
      );
      let b_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, b_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, b_ch_hi)),
      );

      if ALPHA {
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_u16, g_u16, b_u16, alpha_u16),
        );
      } else {
        vst3q_u16(
          out.as_mut_ptr().add(x * 3),
          uint16x8x3_t(r_u16, g_u16, b_u16),
        );
      }
      x += 8;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_16_to_rgba_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_16_to_rgb_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ===== 16-bit helpers =====================================================

/// Scale 8 u16 Y pixels to i16x8 for the 16-bit u8-output path.
///
/// Unsigned-widens via `vmovl_u16`, subtracts `y_off` in i32, multiplies
/// by `y_scale` (small for u8 output — no i32 overflow), Q15-shifts, and
/// narrows to i16x8 with `vqmovn_s32`.
#[inline(always)]
fn scale_y_u16_to_i16(
  y_vec: uint16x8_t,
  y_off_v: int32x4_t,
  y_scale_v: int32x4_t,
  rnd_v: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec)));
    let lo_s = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vsubq_s32(lo, y_off_v), y_scale_v),
      rnd_v,
    ));
    let hi_s = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vsubq_s32(hi, y_off_v), y_scale_v),
      rnd_v,
    ));
    vcombine_s16(vqmovn_s32(lo_s), vqmovn_s32(hi_s))
  }
}

/// `(cu*u_d + cv*v_d + RND) >> 15` in i64 for 4 chroma values → i32x4.
///
/// Used by the 16-bit u16-output path where `coeff * u_d` exceeds i32.
/// `vmull_s32` widens each 32×32 product to 64 bits, avoiding overflow.
#[inline(always)]
fn chroma_i64x4(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d: int32x4_t,
  v_d: int32x4_t,
  rnd64: int64x2_t,
) -> int32x4_t {
  unsafe {
    let sum_lo = vshrq_n_s64::<15>(vaddq_s64(
      vaddq_s64(
        vmull_s32(vget_low_s32(cu), vget_low_s32(u_d)),
        vmull_s32(vget_low_s32(cv), vget_low_s32(v_d)),
      ),
      rnd64,
    ));
    let sum_hi = vshrq_n_s64::<15>(vaddq_s64(
      vaddq_s64(
        vmull_s32(vget_high_s32(cu), vget_high_s32(u_d)),
        vmull_s32(vget_high_s32(cv), vget_high_s32(v_d)),
      ),
      rnd64,
    ));
    vcombine_s32(vmovn_s64(sum_lo), vmovn_s64(sum_hi))
  }
}

/// Scale 4 u16 Y pixels via i64 widening for the 16-bit u16-output path.
///
/// `(y - y_off) * y_scale` can reach ~2.35×10⁹ at 16-bit limited range,
/// overflowing i32. `vmull_s32` widens to i64 before the Q15 shift.
/// Input `y_u32` is already unsigned-widened and reinterpreted as i32.
#[inline(always)]
fn scale_y_u16_i64(
  y_i32: int32x4_t,
  y_off_v: int32x4_t,
  y_scale_d: int32x2_t,
  rnd64: int64x2_t,
) -> int32x4_t {
  unsafe {
    let sub = vsubq_s32(y_i32, y_off_v);
    let lo = vshrq_n_s64::<15>(vaddq_s64(vmull_s32(vget_low_s32(sub), y_scale_d), rnd64));
    let hi = vshrq_n_s64::<15>(vaddq_s64(vmull_s32(vget_high_s32(sub), y_scale_d), rnd64));
    vcombine_s32(vmovn_s64(lo), vmovn_s64(hi))
  }
}

// ===== RGB → HSV =========================================================

/// NEON RGB → planar HSV. Semantics match
/// [`scalar::rgb_to_hsv_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **NEON must be available on the current CPU** (same obligation
///    as `yuv_420_to_rgb_row`; the dispatcher checks this via
///    `is_aarch64_feature_detected!("neon")`).
/// 2. `rgb.len() >= 3 * width`.
/// 3. `h_out.len() >= width`.
/// 4. `s_out.len() >= width`.
/// 5. `v_out.len() >= width`.
///
/// Bounds are verified by `debug_assert` in debug builds. The kernel
/// relies on unchecked pointer arithmetic (`vld3q_u8`, `vst1q_u8`).
///
/// # Numerical contract
///
/// Bit‑identical to the scalar reference. Every scalar op has the
/// same SIMD counterpart in the same order: `vmaxq_f32` / `vminq_f32`
/// mirror `f32::max` / `f32::min`; `vdivq_f32` is true f32 division
/// (not reciprocal estimate); branch cascade uses `vbslq_f32` in the
/// same `delta == 0 → v == r → v == g → v == b` priority.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(h_out.len() >= width, "H row too short");
  debug_assert!(s_out.len() >= width, "S row too short");
  debug_assert!(v_out.len() >= width, "V row too short");

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section. All pointer adds below are bounded by the
  // `while x + 16 <= width` loop condition and the caller‑promised
  // slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 RGB pixels → three u8x16 channel vectors.
      let rgb_vec = vld3q_u8(rgb.as_ptr().add(x * 3));
      let r_u8 = rgb_vec.0;
      let g_u8 = rgb_vec.1;
      let b_u8 = rgb_vec.2;

      // Widen each u8x16 to four f32x4 (16 values split into four
      // 4‑pixel groups) for the f32 HSV math.
      let (b0, b1, b2, b3) = u8x16_to_f32x4_quad(b_u8);
      let (g0, g1, g2, g3) = u8x16_to_f32x4_quad(g_u8);
      let (r0, r1, r2, r3) = u8x16_to_f32x4_quad(r_u8);

      // HSV per 4‑pixel group. Each returns (h_quant, s_quant, v_quant)
      // as f32x4 values already in [0, 179] / [0, 255] / [0, 255].
      let (h0, s0, v0) = hsv_group(b0, g0, r0);
      let (h1, s1, v1) = hsv_group(b1, g1, r1);
      let (h2, s2, v2) = hsv_group(b2, g2, r2);
      let (h3, s3, v3) = hsv_group(b3, g3, r3);

      // Truncate f32 → u8 via u32 intermediate, matching scalar `as u8`
      // (which saturates then truncates; values are pre‑clamped so the
      // narrow is safe).
      let h_u8 = f32x4_quad_to_u8x16(h0, h1, h2, h3);
      let s_u8 = f32x4_quad_to_u8x16(s0, s1, s2, s3);
      let v_u8 = f32x4_quad_to_u8x16(v0, v1, v2, v3);

      vst1q_u8(h_out.as_mut_ptr().add(x), h_u8);
      vst1q_u8(s_out.as_mut_ptr().add(x), s_u8);
      vst1q_u8(v_out.as_mut_ptr().add(x), v_u8);

      x += 16;
    }

    // Scalar tail for the 0..15 leftover pixels.
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

/// Widens a u8x16 to four f32x4 groups (covering lanes 0..3, 4..7,
/// 8..11, 12..15 respectively). Lanes are zero‑extended at each
/// widening step, so f32 values land exactly in `[0.0, 255.0]`.
#[inline(always)]
fn u8x16_to_f32x4_quad(v: uint8x16_t) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let u16_lo = vmovl_u8(vget_low_u8(v)); // u16x8 = lanes 0..7
    let u16_hi = vmovl_u8(vget_high_u8(v)); // u16x8 = lanes 8..15
    let u32_0 = vmovl_u16(vget_low_u16(u16_lo)); // lanes 0..3
    let u32_1 = vmovl_u16(vget_high_u16(u16_lo)); // lanes 4..7
    let u32_2 = vmovl_u16(vget_low_u16(u16_hi)); // lanes 8..11
    let u32_3 = vmovl_u16(vget_high_u16(u16_hi)); // lanes 12..15
    (
      vcvtq_f32_u32(u32_0),
      vcvtq_f32_u32(u32_1),
      vcvtq_f32_u32(u32_2),
      vcvtq_f32_u32(u32_3),
    )
  }
}

/// Computes HSV for 4 pixels. Mirrors the scalar `rgb_to_hsv_pixel`
/// op‑for‑op. Returns `(h_quant, s_quant, v_quant)` — each already
/// clamped to the scalar's output range (`h ≤ 179`, `s ≤ 255`,
/// `v ≤ 255`), still as f32 awaiting u8 conversion in the caller.
#[inline(always)]
fn hsv_group(
  b: float32x4_t,
  g: float32x4_t,
  r: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let zero = vdupq_n_f32(0.0);
    let half = vdupq_n_f32(0.5);
    let sixty = vdupq_n_f32(60.0);
    let one_twenty = vdupq_n_f32(120.0);
    let two_forty = vdupq_n_f32(240.0);
    let three_sixty = vdupq_n_f32(360.0);
    let one_seventy_nine = vdupq_n_f32(179.0);
    let two_fifty_five = vdupq_n_f32(255.0);

    // V = max(b, g, r); min = min(b, g, r); delta = V - min.
    // vmaxq_f32 / vminq_f32 are NaN‑tolerant, matching f32::max / f32::min.
    let v = vmaxq_f32(vmaxq_f32(b, g), r);
    let min_bgr = vminq_f32(vminq_f32(b, g), r);
    let delta = vsubq_f32(v, min_bgr);

    // S = if v == 0 { 0 } else { 255 * delta / v }.
    let mask_v_nonzero = vmvnq_u32(vceqq_f32(v, zero));
    let s_nonzero = vdivq_f32(vmulq_f32(two_fifty_five, delta), v);
    let s = vbslq_f32(mask_v_nonzero, s_nonzero, zero);

    // Hue — compute all three candidate formulas then select.
    let mask_delta_zero = vceqq_f32(delta, zero);
    let mask_v_is_r = vceqq_f32(v, r);
    let mask_v_is_g = vceqq_f32(v, g);

    // Branch 1 (v == r): 60 * (g - b) / delta, wrap negatives by +360.
    let h_r = {
      let raw = vdivq_f32(vmulq_f32(sixty, vsubq_f32(g, b)), delta);
      let mask_neg = vcltq_f32(raw, zero);
      vbslq_f32(mask_neg, vaddq_f32(raw, three_sixty), raw)
    };
    // Branch 2 (v == g): 60 * (b - r) / delta + 120.
    let h_g = vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(b, r)), delta),
      one_twenty,
    );
    // Branch 3 (v == b, implicit): 60 * (r - g) / delta + 240.
    let h_b = vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(r, g)), delta),
      two_forty,
    );

    // Cascade: if delta == 0 → 0; else if v == r → h_r; else if v == g
    // → h_g; else → h_b. Same priority order as the scalar.
    let hue_g_or_b = vbslq_f32(mask_v_is_g, h_g, h_b);
    let hue_nonzero_delta = vbslq_f32(mask_v_is_r, h_r, hue_g_or_b);
    let hue = vbslq_f32(mask_delta_zero, zero, hue_nonzero_delta);

    // Quantize to the scalar's output ranges. Scalar:
    //   h_quant = (hue * 0.5 + 0.5).clamp(0, 179)
    //   s_quant = (s + 0.5).clamp(0, 255)
    //   v_quant = (v + 0.5).clamp(0, 255)
    // clamp → vminq(vmaxq(v, lo), hi). Inputs are all finite so NaN
    // handling is irrelevant here.
    let h_quant = vminq_f32(
      vmaxq_f32(vaddq_f32(vmulq_f32(hue, half), half), zero),
      one_seventy_nine,
    );
    let s_quant = vminq_f32(vmaxq_f32(vaddq_f32(s, half), zero), two_fifty_five);
    let v_quant = vminq_f32(vmaxq_f32(vaddq_f32(v, half), zero), two_fifty_five);

    (h_quant, s_quant, v_quant)
  }
}

/// Converts four f32x4 vectors (16 values in [0, 255]) to one u8x16.
/// Truncates f32 → u32 via `vcvtq_u32_f32` (matches scalar `as u8`
/// which saturates‑then‑truncates; values are pre‑clamped so the
/// narrowing steps below are exact).
#[inline(always)]
fn f32x4_quad_to_u8x16(
  a: float32x4_t,
  b: float32x4_t,
  c: float32x4_t,
  d: float32x4_t,
) -> uint8x16_t {
  unsafe {
    let a_u32 = vcvtq_u32_f32(a);
    let b_u32 = vcvtq_u32_f32(b);
    let c_u32 = vcvtq_u32_f32(c);
    let d_u32 = vcvtq_u32_f32(d);
    let ab_u16 = vcombine_u16(vmovn_u32(a_u32), vmovn_u32(b_u32));
    let cd_u16 = vcombine_u16(vmovn_u32(c_u32), vmovn_u32(d_u32));
    vcombine_u8(vmovn_u16(ab_u16), vmovn_u16(cd_u16))
  }
}

// ===== BGR ↔ RGB byte swap ==============================================

/// Swaps the outer two channels of each packed 3‑byte triple. Drives
/// both `bgr_to_rgb_row` and `rgb_to_bgr_row` since the transformation
/// is self‑inverse.
///
/// NEON makes this almost free: `vld3q_u8` deinterleaves 16 pixels into
/// three channel vectors `(ch0, ch1, ch2)`, and `vst3q_u8` re‑interleaves
/// them — passing the deinterleaved vectors back in reversed order
/// `(ch2, ch1, ch0)` swaps the outer channels in a single store.
///
/// # Safety
///
/// 1. NEON must be available (same obligation as the other NEON kernels).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section. All pointer adds are bounded by the
  // `while x + 16 <= width` condition and the caller‑promised
  // slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let triple = vld3q_u8(input.as_ptr().add(x * 3));
      let swapped = uint8x16x3_t(triple.2, triple.1, triple.0);
      vst3q_u8(output.as_mut_ptr().add(x * 3), swapped);
      x += 16;
    }
    if x < width {
      scalar::bgr_rgb_swap_row(
        &input[x * 3..width * 3],
        &mut output[x * 3..width * 3],
        width - x,
      );
    }
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
