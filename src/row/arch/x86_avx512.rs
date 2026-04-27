//! x86_64 AVX‑512 backend (F + BW) for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_x86_feature_detected!("avx512bw")` returns true (runtime,
//! std‑gated) or `cfg!(target_feature = "avx512bw")` evaluates true
//! (compile‑time, no‑std). The kernel carries
//! `#[target_feature(enable = "avx512f,avx512bw")]` so its intrinsics
//! execute in an explicitly feature‑enabled context.
//!
//! Requires AVX‑512F (foundation) and AVX‑512BW (byte/word integer
//! ops). All real AVX‑512 CPUs have both — Intel Skylake‑X / Cascade
//! Lake / Ice Lake / Sapphire Rapids Xeons, AMD Zen 4+ (Genoa,
//! Ryzen 7000+).
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON / SSE4.1 / AVX2 backends.
//!
//! # Pipeline (per 64 Y pixels / 32 chroma samples)
//!
//! 1. Load 64 Y (`_mm512_loadu_si512`) + 32 U + 32 V (`_mm256_loadu_si256`).
//! 2. Widen U, V to i16x32 (`_mm512_cvtepu8_epi16`), subtract 128.
//! 3. Split each i16x32 into two i32x16 halves and apply `c_scale`.
//! 4. Per channel C ∈ {R, G, B}: `(C_u*u_d + C_v*v_d + RND) >> 15` in
//!    i32, narrow‑saturate to i16x32.
//! 5. Nearest‑neighbor chroma upsample: duplicate each of the 32 chroma
//!    lanes into its pair slot → two i16x32 vectors covering 64 Y lanes.
//! 6. Y path: widen 64 Y to two i16x32 vectors, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel.
//! 8. Saturate‑narrow to u8x64 per channel, then interleave as packed
//!    RGB via four calls to the shared [`super::x86_common::write_rgb_16`]
//!    (192 output bytes = 4 × 48).
//!
//! # AVX‑512 lane‑crossing fixups
//!
//! AVX‑512 registers act as four 128‑bit lanes for most of the ops we
//! use. `_mm512_packs_epi32`, `_mm512_packus_epi16`, and
//! `_mm512_unpack{lo,hi}_epi16` all operate per 128‑bit lane,
//! producing lane‑split results.
//!
//! - **Pack fixup** (shared by `packs_epi32` → i16x32 and
//!   `packus_epi16` → u8x64): after either pack, 64‑bit lane order is
//!   `[lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]`. Permute via
//!   `_mm512_permutexvar_epi64` with index `[0, 2, 4, 6, 1, 3, 5, 7]`
//!   restores natural `[lo0..3 contiguous, hi0..3 contiguous]`.
//! - **Chroma‑dup fixup**: `unpacklo`/`unpackhi` each produce per‑lane
//!   duplicated pairs but the halves for a given Y block are split
//!   across lanes. `_mm512_permutex2var_epi64` with indices
//!   `[0,1,8,9,2,3,10,11]` and `[4,5,12,13,6,7,14,15]` rebuilds the
//!   two 32‑Y‑block‑aligned vectors from unpacklo + unpackhi.

use core::arch::x86_64::*;

use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{
      rgb_to_hsv_16_pixels, swap_rb_16_pixels, write_rgb_16, write_rgb_u16_8, write_rgba_16,
      write_rgba_u16_8,
    },
    scalar,
  },
};

/// AVX‑512 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
///    The dispatcher in [`crate::row`] verifies this with
///    `is_x86_feature_detected!("avx512bw")` (runtime, std) or
///    `cfg!(target_feature = "avx512bw")` (compile‑time, no‑std).
///    AVX‑512BW implies AVX‑512F on all real CPUs. Calling this kernel
///    on a CPU without AVX‑512BW triggers an illegal‑instruction trap.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`_mm512_loadu_si512`, `_mm256_loadu_si256`,
/// `_mm_storeu_si128` inside `write_rgb_16`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 YUV 4:2:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. AVX‑512F + AVX‑512BW must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX-512 YUVA 4:2:0 → packed **8-bit RGBA** with the per-pixel
/// alpha byte **sourced from `a_src`** (8-bit YUVA's alpha is already
/// `u8` — no depth conversion needed). Same numerical contract as
/// [`yuv_420_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX‑512 kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, [`write_rgb_64`]), [`yuv_420_to_rgba_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, [`write_rgba_64`] with constant
/// `0xFF` alpha) and [`yuv_420_to_rgba_with_alpha_src_row`]
/// (`ALPHA = true, ALPHA_SRC = true`, [`write_rgba_64`] with the
/// alpha lane loaded directly from `a_src`).
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long. When `ALPHA_SRC = true`: `a_src` must be `Some(_)`
/// and `a_src.unwrap().len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // Source alpha requires RGBA output.
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

  // SAFETY: AVX‑512BW availability is the caller's obligation per the
  // `# Safety` section; the dispatcher in `crate::row` checks it.
  // All pointer adds below are bounded by the `while x + 64 <= width`
  // loop condition and the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let mid128 = _mm512_set1_epi16(128);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = _mm512_set1_epi8(-1); // 0xFF as i8

    // Lane‑fixup permute indices, computed once per call.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec_256 = _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast());
      let v_vec_256 = _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast());

      // Widen U/V to i16x32 and subtract 128.
      let u_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_vec_256), mid128);
      let v_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_vec_256), mid128);

      // Split each i16x32 into two i32x16 halves for the Q15 multiplies.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      // Per‑channel chroma → i16x32 (natural order after pack fixup).
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // Nearest‑neighbor upsample: pair‑duplicate each chroma lane into
      // two i16x32 vectors covering 64 Y lanes.
      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      // Y path: widen 64 Y to two i16x32, scale.
      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x64 per channel with the same pack fixup.
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit YUVA alpha is already u8; load 64 bytes via 512-bit
          // load.
          _mm512_loadu_si512(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u8
        };
        // 4‑way interleave → packed RGBA (256 bytes = 4 × 64).
        write_rgba_64(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        // 3‑way interleave → packed RGB (192 bytes = 4 × 48).
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

    // Scalar tail for the 0..62 leftover pixels (always even; 4:2:0
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

/// AVX‑512 YUV 4:2:0 10‑bit → packed **8‑bit** RGB.
///
/// Block size 64 Y pixels / 32 chroma pairs per iteration (matching
/// the 8‑bit AVX‑512 kernel). Structural differences:
/// - Two `_mm512_loadu_si512` loads for Y (each 32 `u16` = 64 bytes);
///   one `_mm512_loadu_si512` each for U / V (32 `u16`).
/// - No u8→i16 widening — 10‑bit samples already occupy 16‑bit lanes.
/// - Chroma bias is 512 (10‑bit center).
/// - `range_params_n::<10, 8>` calibrates scales for 10→8 in one shift.
///
/// Reuses [`chroma_i16x32`], [`chroma_dup`], [`scale_y`],
/// [`narrow_u8x64`], and [`write_rgb_64`] along with the pack / dup
/// lane‑fixup indices from the 8‑bit path — post‑chroma math is
/// identical across bit depths.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::yuv_420p_n_to_rgb_row::<10>`].
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 high-bit-depth YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 YUVA 4:2:0 high-bit-depth → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (depth-converted via
/// `>> (BITS - 8)` to fit `u8`). Same numerical contract as
/// [`yuv_420p_n_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 high-bit YUV 4:2:0 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_64`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_64` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_64` with the alpha
///   lane loaded from `a_src`, masked to BITS, and depth-converted
///   via `_mm512_srl_epi16` with a count of `BITS - 8`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. slices long enough + `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 5. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      // AND‑mask every load to the low 10 bits — see matching
      // comment in [`crate::row::scalar::yuv_420p_n_to_rgb_row`].
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_vec = _mm512_and_si512(
        _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );
      let v_vec = _mm512_and_si512(
        _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );

      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar.
          // `_mm512_srli_epi16::<IMM8>` requires a literal const
          // generic shift (not stable for `BITS - 8`); use
          // `_mm512_srl_epi16` with a count vector built from
          // `BITS - 8`. Reuse `narrow_u8x64` so the alpha bytes line
          // up with R/G/B in `write_rgba_64`.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x).cast()), mask_v);
          let a_hi = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x + 32).cast()), mask_v);
          let a_shr = _mm_cvtsi32_si128((BITS - 8) as i32);
          let a_lo_shifted = _mm512_srl_epi16(a_lo, a_shr);
          let a_hi_shifted = _mm512_srl_epi16(a_hi, a_shr);
          narrow_u8x64(a_lo_shifted, a_hi_shifted, pack_fixup)
        } else {
          alpha_u8
        };
        write_rgba_64(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
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

/// AVX‑512 YUV 4:2:0 10‑bit → packed **10‑bit `u16`** RGB.
///
/// Block size 64 Y pixels per iteration. Mirrors
/// [`yuv420p10_to_rgb_row`]'s pre‑write math; output uses explicit
/// min/max clamp to `[0, 1023]` and 8 calls to [`write_rgb_u16_8`]
/// per block (each handles 8 pixels). A true AVX‑512 u16 interleave
/// would cut store count ~8×; left as a follow‑up optimization.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<10>`].
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 sibling of [`yuv_420p_n_to_rgba_row`] for native-depth
/// `u16` output. Alpha samples are `(1 << BITS) - 1` (opaque maximum
/// at the input bit depth).
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 high-bit YUV 4:2:0 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via 8× `write_quarter` per
/// 64-pixel block; `ALPHA = true` writes RGBA quads via 8×
/// `write_quarter_rgba` with constant alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      // AND‑mask loads to the low 10 bits so `chroma_i16x32`'s
      // `_mm512_packs_epi32` narrow stays lossless.
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_vec = _mm512_and_si512(
        _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );
      let v_vec = _mm512_and_si512(
        _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );

      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      // Eight 8‑pixel u16 writes per 64‑pixel block. For each i16x32
      // channel vector we extract four 128‑bit quarters and hand each
      // to the shared SSE4.1 u16 interleave helper.
      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 0, dst);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 1, dst.add(32));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 2, dst.add(64));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 3, dst.add(96));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 0, dst.add(128));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 1, dst.add(160));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 2, dst.add(192));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 3, dst.add(224));
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_quarter(r_lo, g_lo, b_lo, 0, dst);
        write_quarter(r_lo, g_lo, b_lo, 1, dst.add(24));
        write_quarter(r_lo, g_lo, b_lo, 2, dst.add(48));
        write_quarter(r_lo, g_lo, b_lo, 3, dst.add(72));
        write_quarter(r_hi, g_hi, b_hi, 0, dst.add(96));
        write_quarter(r_hi, g_hi, b_hi, 1, dst.add(120));
        write_quarter(r_hi, g_hi, b_hi, 2, dst.add(144));
        write_quarter(r_hi, g_hi, b_hi, 3, dst.add(168));
      }

      x += 64;
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

/// Clamps an `i16x32` vector to `[0, max]` via AVX‑512
/// `_mm512_min_epi16` / `_mm512_max_epi16`. Used by native-depth
/// u16 output paths (10/12/14 bit).
#[inline(always)]
fn clamp_u16_max_x32(v: __m512i, zero_v: __m512i, max_v: __m512i) -> __m512i {
  unsafe { _mm512_min_epi16(_mm512_max_epi16(v, zero_v), max_v) }
}

/// Writes one 8‑pixel u16 RGB chunk using a 128‑bit quarter of each
/// `i16x32` channel vector. `idx` ∈ `{0,1,2,3}` selects which of the
/// four 128‑bit lanes to extract via `_mm512_extracti32x4_epi32`.
///
/// # Safety
///
/// Same as [`write_rgb_u16_8`] — `ptr` must point to at least 48
/// writable bytes (24 `u16`). Caller's `target_feature` must include
/// AVX‑512F + AVX‑512BW (so `_mm512_extracti32x4_epi32` is available)
/// and SSSE3 (for the underlying `_mm_shuffle_epi8` inside
/// `write_rgb_u16_8`).
#[inline(always)]
unsafe fn write_quarter(r: __m512i, g: __m512i, b: __m512i, idx: u8, ptr: *mut u16) {
  // SAFETY: caller holds the AVX‑512F + SSSE3 target‑feature context.
  // Constant generic arg `IDX` picks one of four 128‑bit lanes; `idx`
  // is bounded to 0..=3 by call sites.
  unsafe {
    let (rq, gq, bq) = match idx {
      0 => (
        _mm512_extracti32x4_epi32::<0>(r),
        _mm512_extracti32x4_epi32::<0>(g),
        _mm512_extracti32x4_epi32::<0>(b),
      ),
      1 => (
        _mm512_extracti32x4_epi32::<1>(r),
        _mm512_extracti32x4_epi32::<1>(g),
        _mm512_extracti32x4_epi32::<1>(b),
      ),
      2 => (
        _mm512_extracti32x4_epi32::<2>(r),
        _mm512_extracti32x4_epi32::<2>(g),
        _mm512_extracti32x4_epi32::<2>(b),
      ),
      _ => (
        _mm512_extracti32x4_epi32::<3>(r),
        _mm512_extracti32x4_epi32::<3>(g),
        _mm512_extracti32x4_epi32::<3>(b),
      ),
    };
    write_rgb_u16_8(rq, gq, bq, ptr);
  }
}

/// RGBA sibling of [`write_quarter`]. Extracts one 128‑bit quarter of
/// each `i16x32` channel vector and hands it (plus a splatted alpha)
/// to [`write_rgba_u16_8`].
///
/// # Safety
///
/// Same as [`write_rgba_u16_8`] — `ptr` must point to at least 64
/// writable bytes (32 `u16`). Caller's `target_feature` must include
/// AVX‑512F + AVX‑512BW (so `_mm512_extracti32x4_epi32` is available)
/// and SSE2 (for the underlying unpack/store inside
/// `write_rgba_u16_8`).
#[inline(always)]
unsafe fn write_quarter_rgba(
  r: __m512i,
  g: __m512i,
  b: __m512i,
  a: __m128i,
  idx: u8,
  ptr: *mut u16,
) {
  unsafe {
    let (rq, gq, bq) = match idx {
      0 => (
        _mm512_extracti32x4_epi32::<0>(r),
        _mm512_extracti32x4_epi32::<0>(g),
        _mm512_extracti32x4_epi32::<0>(b),
      ),
      1 => (
        _mm512_extracti32x4_epi32::<1>(r),
        _mm512_extracti32x4_epi32::<1>(g),
        _mm512_extracti32x4_epi32::<1>(b),
      ),
      2 => (
        _mm512_extracti32x4_epi32::<2>(r),
        _mm512_extracti32x4_epi32::<2>(g),
        _mm512_extracti32x4_epi32::<2>(b),
      ),
      _ => (
        _mm512_extracti32x4_epi32::<3>(r),
        _mm512_extracti32x4_epi32::<3>(g),
        _mm512_extracti32x4_epi32::<3>(b),
      ),
    };
    write_rgba_u16_8(rq, gq, bq, a, ptr);
  }
}

/// AVX-512 YUV 4:4:4 planar 9/10/12/14-bit → packed **u8** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`. Block size 64 pixels.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 YUV 4:4:4 planar 9/10/12/14-bit → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p_n_to_rgb_row`].
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 YUVA 4:4:4 planar 9/10/12/14-bit → packed **8-bit RGBA**
/// with the per-pixel alpha byte **sourced from `a_src`**
/// (depth-converted via `>> (BITS - 8)`) instead of being constant
/// `0xFF`. Same numerical contract as [`yuv_444p_n_to_rgba_row`] for
/// R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 high-bit-depth YUV 4:4:4 kernel for
/// [`yuv_444p_n_to_rgb_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `write_rgb_64`), [`yuv_444p_n_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `write_rgba_64` with constant `0xFF` alpha) and
/// [`yuv_444p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `write_rgba_64` with the alpha lane loaded and
/// depth-converted from `a_src`).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` must be one of `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      // 64 Y + 64 U + 64 V per iter. Full-width chroma (two 512-bit
      // loads each) — no horizontal duplication, 4:4:4 is 1:1.
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_lo_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x + 32).cast()), mask_v);
      let v_lo_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x + 32).cast()), mask_v);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      // Two chroma_i16x32 per channel → 64 chroma values. No dup.
      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x).cast()), mask_v);
          let a_hi = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x + 32).cast()), mask_v);
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar. AVX-512's
          // `_mm512_srli_epi16::<IMM8>` requires a literal shift, so
          // use `_mm512_srl_epi16` with a count vector built from
          // `BITS - 8`. `narrow_u8x64` applies the per-lane permute
          // fixup `pack_fixup` that R/G/B already pay for.
          let a_shr = _mm_cvtsi32_si128((BITS - 8) as i32);
          let a_lo_shifted = _mm512_srl_epi16(a_lo, a_shr);
          let a_hi_shifted = _mm512_srl_epi16(a_hi, a_shr);
          narrow_u8x64(a_lo_shifted, a_hi_shifted, pack_fixup)
        } else {
          alpha_u8
        };
        write_rgba_64(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
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

/// AVX-512 YUV 4:4:4 planar 9/10/12/14-bit → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`. 64 pixels per iter.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 sibling of [`yuv_444p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1`.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 YUVA 4:4:4 planar 9/10/12/14-bit → **native-depth `u16`**
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
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 high-bit YUV 4:4:4 → native-depth `u16` kernel for
/// [`yuv_444p_n_to_rgb_u16_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// 8× `write_quarter`), [`yuv_444p_n_to_rgba_u16_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, 8× `write_quarter_rgba` with constant alpha
/// `(1 << BITS) - 1`) and
/// [`yuv_444p_n_to_rgba_u16_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 8× `write_quarter_rgba` with the alpha quarters
/// loaded from `a_src` and masked to native bit depth — no shift since
/// both the source alpha and the u16 output element are at the same
/// native bit depth).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_lo_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x + 32).cast()), mask_v);
      let v_lo_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x + 32).cast()), mask_v);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        // Per-quarter alpha vectors, one for each of the 8
        // `write_quarter_rgba` calls below. With ALPHA_SRC = false they
        // all collapse to the constant `alpha_u16`. With ALPHA_SRC =
        // true we load 64 source-alpha u16 values (two `__m512i`
        // halves), AND-mask any over-range bits to native depth (no
        // shift — both source and output are at the same bit depth)
        // and split each half into the four 128-bit quarters consumed
        // by `write_quarter_rgba`.
        let (a_lo_q0, a_lo_q1, a_lo_q2, a_lo_q3, a_hi_q0, a_hi_q1, a_hi_q2, a_hi_q3) = if ALPHA_SRC
        {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_v = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x).cast()), mask_v);
          let a_hi_v = _mm512_and_si512(_mm512_loadu_si512(a_ptr.add(x + 32).cast()), mask_v);
          (
            _mm512_extracti32x4_epi32::<0>(a_lo_v),
            _mm512_extracti32x4_epi32::<1>(a_lo_v),
            _mm512_extracti32x4_epi32::<2>(a_lo_v),
            _mm512_extracti32x4_epi32::<3>(a_lo_v),
            _mm512_extracti32x4_epi32::<0>(a_hi_v),
            _mm512_extracti32x4_epi32::<1>(a_hi_v),
            _mm512_extracti32x4_epi32::<2>(a_hi_v),
            _mm512_extracti32x4_epi32::<3>(a_hi_v),
          )
        } else {
          (
            alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16, alpha_u16,
          )
        };
        let dst = out.as_mut_ptr().add(x * 4);
        write_quarter_rgba(r_lo, g_lo, b_lo, a_lo_q0, 0, dst);
        write_quarter_rgba(r_lo, g_lo, b_lo, a_lo_q1, 1, dst.add(32));
        write_quarter_rgba(r_lo, g_lo, b_lo, a_lo_q2, 2, dst.add(64));
        write_quarter_rgba(r_lo, g_lo, b_lo, a_lo_q3, 3, dst.add(96));
        write_quarter_rgba(r_hi, g_hi, b_hi, a_hi_q0, 0, dst.add(128));
        write_quarter_rgba(r_hi, g_hi, b_hi, a_hi_q1, 1, dst.add(160));
        write_quarter_rgba(r_hi, g_hi, b_hi, a_hi_q2, 2, dst.add(192));
        write_quarter_rgba(r_hi, g_hi, b_hi, a_hi_q3, 3, dst.add(224));
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_quarter(r_lo, g_lo, b_lo, 0, dst);
        write_quarter(r_lo, g_lo, b_lo, 1, dst.add(24));
        write_quarter(r_lo, g_lo, b_lo, 2, dst.add(48));
        write_quarter(r_lo, g_lo, b_lo, 3, dst.add(72));
        write_quarter(r_hi, g_hi, b_hi, 0, dst.add(96));
        write_quarter(r_hi, g_hi, b_hi, 1, dst.add(120));
        write_quarter(r_hi, g_hi, b_hi, 2, dst.add(144));
        write_quarter(r_hi, g_hi, b_hi, 3, dst.add(168));
      }

      x += 64;
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

/// AVX-512 YUV 4:4:4 planar **16-bit** → packed **u8** RGB. Stays on
/// the i32 Q15 pipeline — output-range scaling keeps `coeff × u_d`
/// within i32 for u8 output. 64 pixels per iter.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 YUV 4:4:4 planar **16-bit** → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 16-bit YUV 4:4:4 kernel for [`yuv_444p16_to_rgb_row`]
/// (`ALPHA = false`, `write_rgb_64`) and [`yuv_444p16_to_rgba_row`]
/// (`ALPHA = true`, `write_rgba_64` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let u_lo_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let u_hi_vec = _mm512_loadu_si512(u.as_ptr().add(x + 32).cast());
      let v_lo_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());
      let v_hi_vec = _mm512_loadu_si512(v.as_ptr().add(x + 32).cast());

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias16_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 64;
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

/// AVX-512 YUV 4:4:4 planar **16-bit** → packed **u16** RGB. Native
/// 512-bit i64-chroma kernel using `_mm512_srai_epi64`.
///
/// Block size 32 pixels per iter. Mirrors
/// [`yuv_420p16_to_rgb_u16_row`] but with full-width chroma loads
/// and no duplication step (4:4:4 is 1:1 with Y).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 sibling of [`yuv_444p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF`.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 16-bit YUV 4:4:4 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `write_rgb_u16_32`;
/// `ALPHA = true` writes RGBA quads via `write_rgba_u16_32` with
/// constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 pixels/iter. 4:4:4: full-width chroma (32 samples each).
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let v_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

      // Widen each i16x32 → two i32x16 halves.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      // Scale in i32 — `u_centered * c_scale` fits i32 after >> 15.
      let u_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let u_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_i32_v,
      ));

      // i64 chroma: 4 calls per channel (even/odd of u_d_lo and u_d_hi).
      let u_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(u_d_lo);
      let u_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(v_d_lo);
      let v_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_e = chroma_i64x8_avx512(cru, crv, u_d_lo, v_d_lo, rnd_i64_v);
      let r_ch_lo_o = chroma_i64x8_avx512(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let r_ch_hi_e = chroma_i64x8_avx512(cru, crv, u_d_hi, v_d_hi, rnd_i64_v);
      let r_ch_hi_o = chroma_i64x8_avx512(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let g_ch_lo_e = chroma_i64x8_avx512(cgu, cgv, u_d_lo, v_d_lo, rnd_i64_v);
      let g_ch_lo_o = chroma_i64x8_avx512(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let g_ch_hi_e = chroma_i64x8_avx512(cgu, cgv, u_d_hi, v_d_hi, rnd_i64_v);
      let g_ch_hi_o = chroma_i64x8_avx512(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let b_ch_lo_e = chroma_i64x8_avx512(cbu, cbv, u_d_lo, v_d_lo, rnd_i64_v);
      let b_ch_lo_o = chroma_i64x8_avx512(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let b_ch_hi_e = chroma_i64x8_avx512(cbu, cbv, u_d_hi, v_d_hi, rnd_i64_v);
      let b_ch_hi_o = chroma_i64x8_avx512(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);

      // Reassemble to i32x16 per half.
      let r_ch_lo = reassemble_i32x16(r_ch_lo_e, r_ch_lo_o, interleave_idx);
      let r_ch_hi = reassemble_i32x16(r_ch_hi_e, r_ch_hi_o, interleave_idx);
      let g_ch_lo = reassemble_i32x16(g_ch_lo_e, g_ch_lo_o, interleave_idx);
      let g_ch_hi = reassemble_i32x16(g_ch_hi_e, g_ch_hi_o, interleave_idx);
      let b_ch_lo = reassemble_i32x16(b_ch_lo_e, b_ch_lo_o, interleave_idx);
      let b_ch_hi = reassemble_i32x16(b_ch_hi_e, b_ch_hi_o, interleave_idx);

      // Y path: widen 32 u16 → two i32x16, subtract y_off, scale in i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Add Y + chroma (no dup — 4:4:4 is 1:1), saturate to u16.
      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_ch_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_ch_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_ch_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_ch_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_ch_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_ch_hi);

      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      if ALPHA {
        write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
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

/// AVX‑512 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **8‑bit** RGB.
///
/// Block size 64 Y pixels / 32 chroma pairs per iteration. Mirrors
/// [`super::x86_avx512::yuv_420p_n_to_rgb_row`] with two structural
/// differences:
/// - Samples are shifted right by `16 - BITS` (`_mm512_srl_epi16`,
///   with a shift count computed from `BITS` once per call) instead
///   of AND‑masked.
/// - Semi‑planar UV is deinterleaved via [`deinterleave_uv_u16_avx512`]
///   — per‑128‑lane shuffle + 64‑bit permute + cross‑vector
///   `_mm512_permutex2var_epi64` to produce 32‑sample U and V
///   vectors.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 high-bit-packed semi-planar 4:2:0 → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 P010/P012 kernel. `ALPHA = false` uses
/// `write_rgb_64`; `ALPHA = true` uses `write_rgba_64` with constant
/// `0xFF` alpha.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`. 3. slices long enough +
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`. 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), shr_count);
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));
      let u_vec = _mm512_srl_epi16(u_vec, shr_count);
      let v_vec = _mm512_srl_epi16(v_vec, shr_count);

      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
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

/// AVX‑512 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **native‑depth `u16`** RGB (low‑bit‑packed output, `yuv420pNle`
/// convention).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_u16_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 sibling of [`p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth). P016 has its own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 Pn → native-depth `u16` kernel. `ALPHA = false`
/// writes RGB triples via 8× `write_quarter` per 64-pixel block;
/// `ALPHA = true` writes RGBA quads via 8× `write_quarter_rgba` with
/// constant alpha `(1 << BITS) - 1`. P016 has its own kernel family —
/// never routed here.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), shr_count);
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));
      let u_vec = _mm512_srl_epi16(u_vec, shr_count);
      let v_vec = _mm512_srl_epi16(v_vec, shr_count);

      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 0, dst);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 1, dst.add(32));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 2, dst.add(64));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 3, dst.add(96));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 0, dst.add(128));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 1, dst.add(160));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 2, dst.add(192));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 3, dst.add(224));
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_quarter(r_lo, g_lo, b_lo, 0, dst);
        write_quarter(r_lo, g_lo, b_lo, 1, dst.add(24));
        write_quarter(r_lo, g_lo, b_lo, 2, dst.add(48));
        write_quarter(r_lo, g_lo, b_lo, 3, dst.add(72));
        write_quarter(r_hi, g_hi, b_hi, 0, dst.add(96));
        write_quarter(r_hi, g_hi, b_hi, 1, dst.add(120));
        write_quarter(r_hi, g_hi, b_hi, 2, dst.add(144));
        write_quarter(r_hi, g_hi, b_hi, 3, dst.add(168));
      }

      x += 64;
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

/// Deinterleaves 64 `u16` elements at `ptr` into `(u_vec, v_vec)` —
/// two AVX‑512 vectors each holding 32 packed `u16` samples.
///
/// Per‑128‑bit‑lane `_mm512_shuffle_epi8` packs even u16s (U's) into
/// each lane's low 64 bits, odd u16s (V's) into the high 64. Then
/// `_mm512_permutexvar_epi64` with the existing `pack_fixup` index
/// `[0, 2, 4, 6, 1, 3, 5, 7]` rearranges the 64‑bit chunks so each
/// vector becomes `[U0..U15 | V0..V15]`. Finally
/// `_mm512_permutex2var_epi64` combines the two vectors into the
/// full 32‑sample U and V vectors.
///
/// # Safety
///
/// `ptr` must point to at least 128 readable bytes (64 `u16`
/// elements). Caller's `target_feature` must include AVX‑512F +
/// AVX‑512BW.
#[inline(always)]
unsafe fn deinterleave_uv_u16_avx512(ptr: *const u16) -> (__m512i, __m512i) {
  unsafe {
    // Per‑128‑lane mask (same byte pattern replicated across the 4
    // lanes of a `__m512i`).
    let split_mask = _mm512_broadcast_i32x4(_mm_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
    ));
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    // Cross-vector 2x8 permute indices:
    //   u_vec = low 256 of each vec → chunks [0..3 of a, 0..3 of b]
    //   v_vec = high 256 of each vec → chunks [4..7 of a, 4..7 of b]
    let u_perm = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    let v_perm = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

    let uv0 = _mm512_loadu_si512(ptr.cast());
    let uv1 = _mm512_loadu_si512(ptr.add(32).cast());

    let s0 = _mm512_shuffle_epi8(uv0, split_mask);
    let s1 = _mm512_shuffle_epi8(uv1, split_mask);

    // After per-lane shuffle + per-vector 64-bit permute, each vector
    // is `[U0..U15 | V0..V15]` (low 256 = U's, high 256 = V's).
    let s0_p = _mm512_permutexvar_epi64(pack_fixup, s0);
    let s1_p = _mm512_permutexvar_epi64(pack_fixup, s1);

    let u_vec = _mm512_permutex2var_epi64(s0_p, u_perm, s1_p);
    let v_vec = _mm512_permutex2var_epi64(s0_p, v_perm, s1_p);
    (u_vec, v_vec)
  }
}

/// AVX‑512 NV12 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_or_nv21_to_rgb_or_rgba_row_impl`]:
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
///    Direct callers are responsible for verifying this; the
///    dispatcher in [`crate::row::nv12_to_rgb_row`] checks it.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `uv_half.len() >= width` (interleaved UV bytes, 2 per chroma pair).
/// 5. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 NV21 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_to_rgb_row`]; `vu_half` carries the same
/// number of bytes (`>= width`) but in V-then-U order per chroma
/// pair.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
      y, vu_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 NV12 → packed RGBA. Same contract as [`nv12_to_rgb_row`]
/// but writes 4 bytes per pixel via [`write_rgba_64`].
/// `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv12_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes (one extra byte per pixel for the opaque
/// alpha).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX‑512 NV21 → packed RGBA. Same contract as [`nv21_to_rgb_row`]
/// but writes 4 bytes per pixel via [`write_rgba_64`].
/// `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv21_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
      y, vu_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX‑512 NV12/NV21 kernel at 3 bpp (RGB) or 4 bpp + opaque
/// alpha (RGBA). `SWAP_UV` selects chroma byte order; `ALPHA = true`
/// writes via [`write_rgba_64`], `ALPHA = false` via [`write_rgb_64`].
/// Both const generics drive compile-time monomorphization.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (64 interleaved bytes per 64 Y pixels).
/// 5. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: AVX‑512BW availability is the caller's obligation; all
  // pointer adds below are bounded by the `while x + 64 <= width`
  // condition and the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let mid128 = _mm512_set1_epi16(128);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1); // 0xFF as i8

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    // Per‑128‑bit‑lane UV deinterleave mask. Broadcast to all 4 lanes.
    // Within each 16‑byte chunk, pack even‑offset (U) bytes into the
    // low 8 lanes and odd‑offset (V) bytes into the high 8 lanes.
    let uv_lane_mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After per‑lane shuffle the 64‑bit lane layout is
    // `[U0, V0, U1, V1, U2, V2, U3, V3]`; permuting with
    // `[0, 2, 4, 6, 1, 3, 5, 7]` compacts to
    // `[U0, U1, U2, U3 | V0, V1, V2, V3]` — low 256 = U, high 256 = V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 64 Y pixels → 32 chroma pairs = 64 interleaved bytes at
      // offset `x` in the chroma row.
      let uv_vec = _mm512_loadu_si512(uv_or_vu_half.as_ptr().add(x).cast());

      // Per-lane shuffle + permute packs even-offset bytes into low
      // 256 and odd-offset bytes into high 256. For NV12 that's
      // (U, V); for NV21 the roles swap.
      let deint = _mm512_shuffle_epi8(uv_vec, uv_deint_mask);
      let uv_compact = _mm512_permutexvar_epi64(uv_collect, deint);
      let (u_vec_256, v_vec_256) = if SWAP_UV {
        (
          _mm512_extracti64x4_epi64::<1>(uv_compact),
          _mm512_castsi512_si256(uv_compact),
        )
      } else {
        (
          _mm512_castsi512_si256(uv_compact),
          _mm512_extracti64x4_epi64::<1>(uv_compact),
        )
      };

      let u_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_vec_256), mid128);
      let v_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_vec_256), mid128);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

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

/// AVX-512 NV24 → packed RGB (UV-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 NV42 → packed RGB (VU-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 NV24 → packed RGBA (UV-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 NV42 → packed RGBA (VU-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 NV24/NV42 kernel (4:4:4 semi-planar). 64 Y pixels /
/// 64 chroma pairs / 128 UV bytes per iteration. Unlike
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`], chroma is not subsampled — one
/// UV pair per Y pixel — so the `chroma_dup` step disappears; two
/// `chroma_i16x32` calls per channel produce 64 chroma values
/// directly.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`.
/// 3. `uv_or_vu.len() >= 2 * width`.
/// 4. `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn nv24_or_nv42_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; all
  // pointer adds below are bounded by the `while x + 64 <= width`
  // loop and the caller-promised slice lengths.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let mid128 = _mm512_set1_epi16(128);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    // Same lane fixups as NV12 kernel — inherited verbatim.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // Per-128-bit-lane UV deinterleave mask, broadcast to all 4 lanes.
    let uv_lane_mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After per-lane shuffle the 64-bit lane layout is
    // `[U0, V0, U1, V1, U2, V2, U3, V3]`; this permute collects to
    // `[U0, U1, U2, U3 | V0, V1, V2, V3]` — low 256 = U, high 256 = V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 64 Y pixels → 128 UV bytes (two 512-bit loads).
      let uv_vec_lo = _mm512_loadu_si512(uv_or_vu.as_ptr().add(x * 2).cast());
      let uv_vec_hi = _mm512_loadu_si512(uv_or_vu.as_ptr().add(x * 2 + 64).cast());

      // Per 512-bit vec: deinterleave → low 256 = U (32 bytes),
      // high 256 = V (32 bytes). Roles swap for NV42.
      let d_lo =
        _mm512_permutexvar_epi64(uv_collect, _mm512_shuffle_epi8(uv_vec_lo, uv_deint_mask));
      let d_hi =
        _mm512_permutexvar_epi64(uv_collect, _mm512_shuffle_epi8(uv_vec_hi, uv_deint_mask));
      let (u_bytes_lo_256, v_bytes_lo_256, u_bytes_hi_256, v_bytes_hi_256) = if SWAP_UV {
        (
          _mm512_extracti64x4_epi64::<1>(d_lo),
          _mm512_castsi512_si256(d_lo),
          _mm512_extracti64x4_epi64::<1>(d_hi),
          _mm512_castsi512_si256(d_hi),
        )
      } else {
        (
          _mm512_castsi512_si256(d_lo),
          _mm512_extracti64x4_epi64::<1>(d_lo),
          _mm512_castsi512_si256(d_hi),
          _mm512_extracti64x4_epi64::<1>(d_hi),
        )
      };

      // Widen each 32-byte U/V chunk to i16x32 and subtract 128.
      let u_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_bytes_lo_256), mid128);
      let u_hi_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_bytes_hi_256), mid128);
      let v_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_bytes_lo_256), mid128);
      let v_hi_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_bytes_hi_256), mid128);

      // Split each i16x32 into two i32x16 halves.
      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      // u_d / v_d = (u * c_scale + RND) >> 15.
      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      // 64 chroma per channel (two chroma_i16x32 per channel, no
      // duplication).
      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
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

/// AVX-512 YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 YUV 4:4:4 planar → packed **RGBA** (8-bit). Same contract
/// as [`yuv_444_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_64`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_444_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX-512 YUV 4:4:4 kernel for [`yuv_444_to_rgb_row`]
/// (`ALPHA = false`, [`write_rgb_64`]) and [`yuv_444_to_rgba_row`]
/// (`ALPHA = true`, [`write_rgba_64`] with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let mid128 = _mm512_set1_epi16(128);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1); // 0xFF as i8

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 4:4:4: 64 U + 64 V directly, no deinterleave.
      let u_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let v_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());

      // Widen low / high halves of U / V (32 bytes each) to i16x32.
      let u_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(u_vec)), mid128);
      let u_hi_i16 = _mm512_sub_epi16(
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(u_vec)),
        mid128,
      );
      let v_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(v_vec)), mid128);
      let v_hi_i16 = _mm512_sub_epi16(
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(v_vec)),
        mid128,
      );

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
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

// ---- helpers (inlined into the target_feature‑enabled caller) ----------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: __m512i) -> __m512i {
  unsafe { _mm512_srai_epi32::<15>(v) }
}

/// Computes one i16x32 chroma channel vector from the four i32x16
/// chroma inputs (lo/hi halves of `u_d` and `v_d`). Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, saturating‑packs to
/// i16x32, then applies `pack_fixup` to restore natural element order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn chroma_i16x32(
  cu: __m512i,
  cv: __m512i,
  u_d_lo: __m512i,
  v_d_lo: __m512i,
  u_d_hi: __m512i,
  v_d_hi: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_add_epi32(
        _mm512_mullo_epi32(cu, u_d_lo),
        _mm512_mullo_epi32(cv, v_d_lo),
      ),
      rnd,
    ));
    let hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_add_epi32(
        _mm512_mullo_epi32(cu, u_d_hi),
        _mm512_mullo_epi32(cv, v_d_hi),
      ),
      rnd,
    ));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo, hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x32 vector,
/// returned as i16x32 (with pack fixup applied).
#[inline(always)]
fn scale_y(
  y_i16: __m512i,
  y_off_v: __m512i,
  y_scale_v: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let shifted = _mm512_sub_epi16(y_i16, y_off_v);
    let lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(shifted));
    let hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(shifted));
    let lo_scaled =
      _mm512_srai_epi32::<15>(_mm512_add_epi32(_mm512_mullo_epi32(lo_i32, y_scale_v), rnd));
    let hi_scaled =
      _mm512_srai_epi32::<15>(_mm512_add_epi32(_mm512_mullo_epi32(hi_i32, y_scale_v), rnd));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo_scaled, hi_scaled))
  }
}

/// Duplicates each of 32 chroma lanes into its adjacent pair slot,
/// splitting across two i16x32 vectors covering 64 Y lanes.
#[inline(always)]
fn chroma_dup(chroma: __m512i, dup_lo_idx: __m512i, dup_hi_idx: __m512i) -> (__m512i, __m512i) {
  unsafe {
    let a = _mm512_unpacklo_epi16(chroma, chroma);
    let b = _mm512_unpackhi_epi16(chroma, chroma);
    let lo32 = _mm512_permutex2var_epi64(a, dup_lo_idx, b);
    let hi32 = _mm512_permutex2var_epi64(a, dup_hi_idx, b);
    (lo32, hi32)
  }
}

/// Saturating‑narrows two i16x32 vectors into one u8x64 with natural
/// element order.
#[inline(always)]
fn narrow_u8x64(lo: __m512i, hi: __m512i, pack_fixup: __m512i) -> __m512i {
  unsafe { _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi16(lo, hi)) }
}

/// Writes 64 pixels of packed RGB (192 bytes) by splitting the u8x64
/// channel vectors into four 128‑bit halves and calling the shared
/// [`write_rgb_16`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 192 writable bytes.
#[inline(always)]
unsafe fn write_rgb_64(r: __m512i, g: __m512i, b: __m512i, ptr: *mut u8) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);

    write_rgb_16(r0, g0, b0, ptr);
    write_rgb_16(r1, g1, b1, ptr.add(48));
    write_rgb_16(r2, g2, b2, ptr.add(96));
    write_rgb_16(r3, g3, b3, ptr.add(144));
  }
}

/// Writes 64 pixels of packed RGBA (256 bytes) by splitting the
/// u8x64 channel vectors into four 128‑bit quarters and calling the
/// shared [`write_rgba_16`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 256 writable bytes.
#[inline(always)]
unsafe fn write_rgba_64(r: __m512i, g: __m512i, b: __m512i, a: __m512i, ptr: *mut u8) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);
    let a0: __m128i = _mm512_castsi512_si128(a);
    let a1: __m128i = _mm512_extracti32x4_epi32::<1>(a);
    let a2: __m128i = _mm512_extracti32x4_epi32::<2>(a);
    let a3: __m128i = _mm512_extracti32x4_epi32::<3>(a);

    write_rgba_16(r0, g0, b0, a0, ptr);
    write_rgba_16(r1, g1, b1, a1, ptr.add(64));
    write_rgba_16(r2, g2, b2, a2, ptr.add(128));
    write_rgba_16(r3, g3, b3, a3, ptr.add(192));
  }
}

// ===== 16-bit u16-output helpers ========================================

/// `(c_u * u_d + c_v * v_d + RND) >> 15` in i64 via two
/// `_mm512_mul_epi32` (even i32 lanes → i64x8 products) plus native
/// `_mm512_srai_epi64`. Result is i64x8 with each lane's low 32 bits
/// holding the i32-range output.
#[inline(always)]
fn chroma_i64x8_avx512(
  cu: __m512i,
  cv: __m512i,
  u_d_even: __m512i,
  v_d_even: __m512i,
  rnd_i64: __m512i,
) -> __m512i {
  unsafe {
    _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_add_epi64(
        _mm512_mul_epi32(cu, u_d_even),
        _mm512_mul_epi32(cv, v_d_even),
      ),
      rnd_i64,
    ))
  }
}

/// Combines two i64x8 results (one from even-indexed, one from
/// odd-indexed i32 lanes of the pre-multiply source) into an
/// interleaved i32x16 `[even_0, odd_0, even_1, odd_1, ..., even_7,
/// odd_7]` — i.e. the natural sequential order of 16 scalar results.
///
/// Each i64 lane's low 32 bits contain the result (high 32 are sign);
/// `_mm512_cvtepi64_epi32` truncates to i32x8 per vector, then
/// `_mm512_permutex2var_epi32` interleaves them.
#[inline(always)]
fn reassemble_i32x16(even_i64: __m512i, odd_i64: __m512i, interleave_idx: __m512i) -> __m512i {
  unsafe {
    let even_i32 = _mm512_cvtepi64_epi32(even_i64); // __m256i i32x8
    let odd_i32 = _mm512_cvtepi64_epi32(odd_i64);
    _mm512_permutex2var_epi32(
      _mm512_castsi256_si512(even_i32),
      interleave_idx,
      _mm512_castsi256_si512(odd_i32),
    )
  }
}

/// `(y_minus_off * y_scale + RND) >> 15` computed in i64 for all 16
/// lanes of an i32x16 Y stream, returning an i32x16 result. Needs
/// i64 because limited-range 16→u16 `(Y - y_off) * y_scale` can
/// reach ~2.35·10⁹ (> i32::MAX). Splits the input into even and
/// odd-indexed i32 lanes, multiplies each set via
/// `_mm512_mul_epi32`, shifts in i64, and reassembles to i32x16.
#[inline(always)]
fn scale_y_i32x16_i64(
  y_minus_off: __m512i,
  y_scale_v: __m512i,
  rnd_i64: __m512i,
  interleave_idx: __m512i,
) -> __m512i {
  unsafe {
    let even = _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_mul_epi32(y_scale_v, y_minus_off),
      rnd_i64,
    ));
    let odd = _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_mul_epi32(y_scale_v, _mm512_shuffle_epi32::<0xF5>(y_minus_off)),
      rnd_i64,
    ));
    reassemble_i32x16(even, odd, interleave_idx)
  }
}

/// Writes 32 pixels of packed RGB-u16 (192 u16 = 384 bytes) by
/// splitting each u16x32 channel vector into four 128-bit halves and
/// calling the shared [`write_rgb_u16_8`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 384 writable bytes.
#[inline(always)]
unsafe fn write_rgb_u16_32(r: __m512i, g: __m512i, b: __m512i, ptr: *mut u16) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);

    // Each `write_rgb_u16_8` writes 8 pixels × 3 × u16 = 48 bytes =
    // 24 u16 elements. Four calls → 96 u16 = 32 pixels.
    write_rgb_u16_8(r0, g0, b0, ptr);
    write_rgb_u16_8(r1, g1, b1, ptr.add(24));
    write_rgb_u16_8(r2, g2, b2, ptr.add(48));
    write_rgb_u16_8(r3, g3, b3, ptr.add(72));
  }
}

/// Writes 32 pixels of packed RGBA-u16 (128 u16 = 256 bytes) by
/// splitting each u16x32 channel vector into four 128-bit halves and
/// calling the shared [`write_rgba_u16_8`] helper four times. Alpha
/// is supplied as a single i16x8 vector splatted into all 32 alpha
/// lanes.
///
/// # Safety
///
/// `ptr` must point to at least 256 writable bytes.
#[inline(always)]
unsafe fn write_rgba_u16_32(r: __m512i, g: __m512i, b: __m512i, a: __m128i, ptr: *mut u16) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);

    // Each `write_rgba_u16_8` writes 8 pixels × 4 × u16 = 64 bytes =
    // 32 u16 elements. Four calls → 128 u16 = 32 pixels.
    write_rgba_u16_8(r0, g0, b0, a, ptr);
    write_rgba_u16_8(r1, g1, b1, a, ptr.add(32));
    write_rgba_u16_8(r2, g2, b2, a, ptr.add(64));
    write_rgba_u16_8(r3, g3, b3, a, ptr.add(96));
  }
}

// ===== 16-bit YUV → RGB ==================================================

/// `(Y_u16x32 - y_off) * y_scale + RND >> 15` for full u16 Y samples.
/// Unsigned widening via `_mm512_cvtepu16_epi32`. Returns i16x32.
#[inline(always)]
fn scale_y_u16_avx512(
  y_u16x32: __m512i,
  y_off_v: __m512i,
  y_scale_v: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let y_lo_i32 = _mm512_sub_epi32(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(y_u16x32)),
      y_off_v,
    );
    let y_hi_i32 = _mm512_sub_epi32(
      _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(y_u16x32)),
      y_off_v,
    );
    let lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_mullo_epi32(y_lo_i32, y_scale_v),
      rnd,
    ));
    let hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_mullo_epi32(y_hi_i32, y_scale_v),
      rnd,
    ));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo, hi))
  }
}

/// AVX-512 YUV 4:2:0 16-bit → packed **8-bit** RGB. 64 pixels per iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 16-bit YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 16-bit YUVA 4:2:0 → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (depth-converted via
/// `>> 8` to fit `u8`). 16-bit alpha is full-range u16 — no AND-mask
/// step. Same numerical contract as [`yuv_420p16_to_rgba_row`] for
/// R/G/B.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 16-bit YUV 4:2:0 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_64`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_64` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_64` with the alpha
///   lane loaded from `a_src` and depth-converted via
///   `_mm512_srli_epi16::<8>` (literal const shift).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let u_vec = _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast());

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — `>> 8` to fit u8.
          // `_mm512_srli_epi16::<8>` accepts a const literal shift.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm512_srli_epi16::<8>(_mm512_loadu_si512(a_ptr.add(x).cast()));
          let a_hi = _mm512_srli_epi16::<8>(_mm512_loadu_si512(a_ptr.add(x + 32).cast()));
          narrow_u8x64(a_lo, a_hi, pack_fixup)
        } else {
          alpha_u8
        };
        write_rgba_64(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 64;
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

/// AVX-512 YUV 4:2:0 16-bit → packed **16-bit** RGB. Native 512-bit
/// implementation, 32 pixels per iteration.
///
/// Uses AVX-512's native `_mm512_srai_epi64` — unlike the SSE4.1 /
/// AVX2 u16 paths which need the `srai64_15` bias trick. Processes
/// chroma and Y in i64 lanes via `_mm512_mul_epi32` (even i32 lanes
/// → i64x8 products), handling even- and odd-indexed lanes
/// separately and reassembling via `_mm512_permutex2var_epi32`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 sibling of [`yuv_420p16_to_rgba_row`] for native-depth
/// `u16` output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 16-bit YUV 4:2:0 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `write_rgb_u16_32`;
/// `ALPHA = true` writes RGBA quads via `write_rgba_u16_32` with
/// constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; pointer
  // adds below are bounded by `while x + 32 <= width` and the caller-
  // promised slice lengths.
  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    // UV centering: subtract 32768 via wrapping i16 add of -32768.
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    // Permute indices, built once per call.
    let dup_lo_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let dup_hi_idx = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    // Interleave two i32x8 (each in the low 256 bits of a __m512i) into
    // an i32x16: [e0, o0, e1, o1, ..., e7, o7].
    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    // `_mm512_packus_epi32` per-128-bit-lane fixup (same as the 8-bit
    // AVX-512 kernels).
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 Y pixels / 16 chroma pairs per iter. The Y load is a
      // full 512-bit read (32 × u16); UV only needs 16 × u16 per
      // plane, so a 256-bit load is sufficient — a 512-bit load
      // would read past the end of `u_half` / `v_half` on the last
      // iteration where `x / 2 + 16 == u_half.len()`.
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec = _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast());

      // Center UV by subtracting 32768 (wrapping i16 sub). Using
      // `_mm256_sub_epi16` with bias16 (which carries -32768 as i16):
      // `sample - (-32768) == sample + 32768` mod 2^16, giving the
      // centered signed value.
      let u_i16 = _mm256_sub_epi16(u_vec, _mm512_castsi512_si256(bias16_v));
      let v_i16 = _mm256_sub_epi16(v_vec, _mm512_castsi512_si256(bias16_v));

      // Widen i16x16 → i32x16 (each value sign-extended).
      let u_i32 = _mm512_cvtepi16_epi32(u_i16);
      let v_i32 = _mm512_cvtepi16_epi32(v_i16);

      // Scale UV in i32: |u_centered * c_scale| peaks at
      // 32768 * ~38300 ≈ 1.26·10⁹ — fits i32. Result `u_d` range is
      // ~[-37450, 37449].
      let u_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_i32_v,
      ));

      // Chroma in i64 via `_mm512_mul_epi32` on even lanes, then
      // shuffle odd lanes to even and repeat. Each call produces 8
      // i64 products from the 8 even-indexed i32 lanes.
      let u_d_odd = _mm512_shuffle_epi32::<0xF5>(u_d); // lanes [1,1,3,3,5,5,7,7] per 128-bit
      let v_d_odd = _mm512_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x8_avx512(cru, crv, u_d, v_d, rnd_i64_v);
      let r_ch_odd = chroma_i64x8_avx512(cru, crv, u_d_odd, v_d_odd, rnd_i64_v);
      let g_ch_even = chroma_i64x8_avx512(cgu, cgv, u_d, v_d, rnd_i64_v);
      let g_ch_odd = chroma_i64x8_avx512(cgu, cgv, u_d_odd, v_d_odd, rnd_i64_v);
      let b_ch_even = chroma_i64x8_avx512(cbu, cbv, u_d, v_d, rnd_i64_v);
      let b_ch_odd = chroma_i64x8_avx512(cbu, cbv, u_d_odd, v_d_odd, rnd_i64_v);

      // Reassemble i64x8 pairs to i32x16.
      let r_ch_i32 = reassemble_i32x16(r_ch_even, r_ch_odd, interleave_idx);
      let g_ch_i32 = reassemble_i32x16(g_ch_even, g_ch_odd, interleave_idx);
      let b_ch_i32 = reassemble_i32x16(b_ch_even, b_ch_odd, interleave_idx);

      // Duplicate chroma for 2 Y pixels each: 16 chroma → 32 values.
      let r_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, r_ch_i32);
      let r_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, r_ch_i32);
      let g_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, g_ch_i32);
      let g_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, g_ch_i32);
      let b_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, b_ch_i32);
      let b_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, b_ch_i32);

      // Y path: split 32 u16 into two halves, widen to i32x16 each,
      // subtract y_off, scale in i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Add Y + chroma, pack with unsigned saturation to u16.
      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_dup_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_dup_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_dup_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_dup_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_dup_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_dup_hi);

      // `_mm512_packus_epi32` signed i32 → unsigned u16 with
      // saturation. Produces u16x32 with per-128-bit-lane split order;
      // fix via `pack_fixup` permute (same as NV12 AVX-512 u8 kernel).
      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      // Write 32 pixels via the appropriate 4× 8-pixel helper.
      if ALPHA {
        write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
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

/// AVX-512 P016 → packed **8-bit** RGB. 64 pixels per iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 P016 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 P016 kernel. `ALPHA = false` uses `write_rgb_64`;
/// `ALPHA = true` uses `write_rgba_64` with constant `0xFF` alpha.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 64;
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

/// AVX-512 P016 → packed **16-bit** RGB.
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgb_out` is `&mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 sibling of [`p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Shared AVX-512 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `write_rgb_u16_32`;
/// `ALPHA = true` writes RGBA quads via `write_rgba_u16_32` with
/// constant alpha `0xFFFF`. 32 pixels per iter. Shares the 16-bit
/// arithmetic structure with [`yuv_420p16_to_rgb_or_rgba_u16_row`];
/// only difference is an inline U/V deinterleave at the UV load.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; pointer
  // adds are bounded by `while x + 32 <= width` and caller-promised
  // slice lengths.
  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let dup_lo_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let dup_hi_idx = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // Per-128-bit-lane shuffle to deinterleave u16 UV pairs within
    // each lane: `[u0,v0,u1,v1,u2,v2,u3,v3] → [u0,u1,u2,u3,v0,v1,v2,v3]`
    // as u16 = byte indices `[0,1, 4,5, 8,9, 12,13 | 2,3, 6,7, 10,11, 14,15]`.
    let uv_lane_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After the per-lane shuffle the 64-bit lane layout is
    // `[U0_3, V0_3, U4_7, V4_7, U8_11, V8_11, U12_15, V12_15]`; permute
    // to `[U0_3, U4_7, U8_11, U12_15 | V0_3, V4_7, V8_11, V12_15]` so
    // low 256 = all U, high 256 = all V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 16 UV pairs = 32 u16 = one 512-bit load.
      let uv_raw = _mm512_loadu_si512(uv_half.as_ptr().add(x).cast());
      let uv_deint = _mm512_shuffle_epi8(uv_raw, uv_deint_mask);
      let uv_split = _mm512_permutexvar_epi64(uv_collect, uv_deint);
      let u_vec = _mm512_castsi512_si256(uv_split);
      let v_vec = _mm512_extracti64x4_epi64::<1>(uv_split);

      // Center UV (same wrapping trick as planar kernel).
      let u_i16 = _mm256_sub_epi16(u_vec, _mm512_castsi512_si256(bias16_v));
      let v_i16 = _mm256_sub_epi16(v_vec, _mm512_castsi512_si256(bias16_v));

      let u_i32 = _mm512_cvtepi16_epi32(u_i16);
      let v_i32 = _mm512_cvtepi16_epi32(v_i16);

      let u_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_i32_v,
      ));

      let u_d_odd = _mm512_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm512_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x8_avx512(cru, crv, u_d, v_d, rnd_i64_v);
      let r_ch_odd = chroma_i64x8_avx512(cru, crv, u_d_odd, v_d_odd, rnd_i64_v);
      let g_ch_even = chroma_i64x8_avx512(cgu, cgv, u_d, v_d, rnd_i64_v);
      let g_ch_odd = chroma_i64x8_avx512(cgu, cgv, u_d_odd, v_d_odd, rnd_i64_v);
      let b_ch_even = chroma_i64x8_avx512(cbu, cbv, u_d, v_d, rnd_i64_v);
      let b_ch_odd = chroma_i64x8_avx512(cbu, cbv, u_d_odd, v_d_odd, rnd_i64_v);

      let r_ch_i32 = reassemble_i32x16(r_ch_even, r_ch_odd, interleave_idx);
      let g_ch_i32 = reassemble_i32x16(g_ch_even, g_ch_odd, interleave_idx);
      let b_ch_i32 = reassemble_i32x16(b_ch_even, b_ch_odd, interleave_idx);

      let r_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, r_ch_i32);
      let r_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, r_ch_i32);
      let g_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, g_ch_i32);
      let g_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, g_ch_i32);
      let b_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, b_ch_i32);
      let b_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, b_ch_i32);

      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_dup_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_dup_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_dup_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_dup_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_dup_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_dup_hi);

      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      if ALPHA {
        write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
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
// Native AVX-512 4:4:4 Pn kernels — combine `yuv_444p_n_to_rgb_row`'s
// 1:1 chroma compute (no duplication, two `chroma_i16x32` per channel)
// with `p_n_to_rgb_row`'s `deinterleave_uv_u16_avx512` pattern. 64 Y
// pixels per iter for the i32 paths; 32 for the i64 u16-output path
// (matches `yuv_444p16_to_rgb_u16_row` cadence). The 16-bit u16 output
// uses native `_mm512_srai_epi64` via `chroma_i64x8_avx512` —
// no bias trick.

/// AVX-512 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **u8** RGB.
/// 64 pixels per iter via 512-bit vectors; 128 UV elements (= 64 pairs)
/// deinterleaved per iter via two `deinterleave_uv_u16_avx512` calls.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// AVX-512 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// Shared AVX-512 Pn 4:4:4 high-bit-packed kernel for
/// [`p_n_444_to_rgb_row`] (`ALPHA = false`, `write_rgb_64`) and
/// [`p_n_444_to_rgba_row`] (`ALPHA = true`, `write_rgba_64` with
/// constant `0xFF` alpha).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), shr_count);

      // 128 UV elements (= 64 pairs) per iter — two deinterleave calls.
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2 + 64));
      let u_lo_vec = _mm512_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm512_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm512_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm512_srl_epi16(v_hi_vec, shr_count);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
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

/// AVX-512 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB. 64 pixels per iter.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// AVX-512 sibling of [`p_n_444_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1`.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// Shared AVX-512 Pn 4:4:4 high-bit-packed → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via 8× `write_quarter`;
/// `ALPHA = true` writes RGBA quads via 8× `write_quarter_rgba` with
/// constant alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), shr_count);

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2 + 64));
      let u_lo_vec = _mm512_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm512_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm512_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm512_srl_epi16(v_hi_vec, shr_count);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 0, dst);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 1, dst.add(32));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 2, dst.add(64));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 3, dst.add(96));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 0, dst.add(128));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 1, dst.add(160));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 2, dst.add(192));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 3, dst.add(224));
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_quarter(r_lo, g_lo, b_lo, 0, dst);
        write_quarter(r_lo, g_lo, b_lo, 1, dst.add(24));
        write_quarter(r_lo, g_lo, b_lo, 2, dst.add(48));
        write_quarter(r_lo, g_lo, b_lo, 3, dst.add(72));
        write_quarter(r_hi, g_hi, b_hi, 0, dst.add(96));
        write_quarter(r_hi, g_hi, b_hi, 1, dst.add(120));
        write_quarter(r_hi, g_hi, b_hi, 2, dst.add(144));
        write_quarter(r_hi, g_hi, b_hi, 3, dst.add(168));
      }

      x += 64;
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

/// AVX-512 P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB.
/// 64 pixels per iter; Y stays on i32.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// AVX-512 P416 (semi-planar 4:4:4, 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// Shared AVX-512 P416 kernel for [`p_n_444_16_to_rgb_row`]
/// (`ALPHA = false`, `write_rgb_64`) and [`p_n_444_16_to_rgba_row`]
/// (`ALPHA = true`, `write_rgba_64` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2 + 64));

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias16_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 64;
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

/// AVX-512 P416 → packed **native-depth `u16`** RGB. 32 pixels per
/// iter (i64 narrows). Native `_mm512_srai_epi64` via
/// `chroma_i64x8_avx512` + `scale_y_i32x16_i64` — no bias trick.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// AVX-512 sibling of [`p_n_444_16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// Shared AVX-512 P416 (semi-planar 4:4:4, 16-bit) → native-depth `u16`
/// kernel. `ALPHA = false` writes RGB triples; `ALPHA = true` writes
/// RGBA quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 pixels per iter — one deinterleave (64 UV elements = 32 pairs).
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2));

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let u_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_i32_v,
      ));

      // i64 chroma via even/odd splits — uses native
      // `_mm512_srai_epi64` inside `chroma_i64x8_avx512`. No bias trick.
      let u_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(u_d_lo);
      let u_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(v_d_lo);
      let v_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_e = chroma_i64x8_avx512(cru, crv, u_d_lo, v_d_lo, rnd_i64_v);
      let r_ch_lo_o = chroma_i64x8_avx512(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let r_ch_hi_e = chroma_i64x8_avx512(cru, crv, u_d_hi, v_d_hi, rnd_i64_v);
      let r_ch_hi_o = chroma_i64x8_avx512(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let g_ch_lo_e = chroma_i64x8_avx512(cgu, cgv, u_d_lo, v_d_lo, rnd_i64_v);
      let g_ch_lo_o = chroma_i64x8_avx512(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let g_ch_hi_e = chroma_i64x8_avx512(cgu, cgv, u_d_hi, v_d_hi, rnd_i64_v);
      let g_ch_hi_o = chroma_i64x8_avx512(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let b_ch_lo_e = chroma_i64x8_avx512(cbu, cbv, u_d_lo, v_d_lo, rnd_i64_v);
      let b_ch_lo_o = chroma_i64x8_avx512(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let b_ch_hi_e = chroma_i64x8_avx512(cbu, cbv, u_d_hi, v_d_hi, rnd_i64_v);
      let b_ch_hi_o = chroma_i64x8_avx512(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);

      let r_ch_lo = reassemble_i32x16(r_ch_lo_e, r_ch_lo_o, interleave_idx);
      let r_ch_hi = reassemble_i32x16(r_ch_hi_e, r_ch_hi_o, interleave_idx);
      let g_ch_lo = reassemble_i32x16(g_ch_lo_e, g_ch_lo_o, interleave_idx);
      let g_ch_hi = reassemble_i32x16(g_ch_hi_e, g_ch_hi_o, interleave_idx);
      let b_ch_lo = reassemble_i32x16(b_ch_lo_e, b_ch_lo_o, interleave_idx);
      let b_ch_hi = reassemble_i32x16(b_ch_hi_e, b_ch_hi_o, interleave_idx);

      // Y: widen 32 u16 → two i32x16 halves, subtract y_off, scale i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_ch_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_ch_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_ch_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_ch_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_ch_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_ch_hi);

      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      if ALPHA {
        write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }
      x += 32;
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

// ===== BGR ↔ RGB byte swap ==============================================

/// AVX‑512 BGR ↔ RGB byte swap. 64 pixels per iteration via four calls
/// to [`super::x86_common::swap_rb_16_pixels`]. The helper uses SSSE3
/// `_mm_shuffle_epi8`, which AVX‑512BW (a superset) allows.
///
/// # Safety
///
/// 1. AVX‑512BW must be available (dispatcher obligation).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = input.as_ptr().add(x * 3);
      let base_out = output.as_mut_ptr().add(x * 3);
      swap_rb_16_pixels(base_in, base_out);
      swap_rb_16_pixels(base_in.add(48), base_out.add(48));
      swap_rb_16_pixels(base_in.add(96), base_out.add(96));
      swap_rb_16_pixels(base_in.add(144), base_out.add(144));
      x += 64;
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

// ===== RGB → HSV =========================================================

/// AVX‑512 RGB → planar HSV. 64 pixels per iteration via four calls to
/// the shared [`super::x86_common::rgb_to_hsv_16_pixels`] helper
/// (SSE4.1‑level compute under AVX‑512 target_feature). Matches the
/// scalar reference within ±1 LSB — the shared helper uses `_mm_rcp_ps`
/// + one Newton‑Raphson step instead of true division (see `x86_common.rs`).
///
/// # Safety
///
/// 1. AVX‑512BW must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`; each output plane `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 64 <= width {
      let base_in = rgb.as_ptr().add(x * 3);
      let base_h = h_out.as_mut_ptr().add(x);
      let base_s = s_out.as_mut_ptr().add(x);
      let base_v = v_out.as_mut_ptr().add(x);
      rgb_to_hsv_16_pixels(base_in, base_h, base_s, base_v);
      rgb_to_hsv_16_pixels(
        base_in.add(48),
        base_h.add(16),
        base_s.add(16),
        base_v.add(16),
      );
      rgb_to_hsv_16_pixels(
        base_in.add(96),
        base_h.add(32),
        base_s.add(32),
        base_v.add(32),
      );
      rgb_to_hsv_16_pixels(
        base_in.add(144),
        base_h.add(48),
        base_s.add(48),
        base_v.add(48),
      );
      x += 64;
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

#[cfg(all(test, feature = "std"))]
mod tests;
