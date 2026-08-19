//! NEON kernels for VUYA / VUYX packed YUV 4:4:4 8-bit family.
//!
//! ## Layout
//!
//! Four `u8` elements per pixel: `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)`.
//! VUYA carries a real alpha channel in byte 3. VUYX treats byte 3 as
//! padding and forces output α to `0xFF`.
//!
//! ## Per-iter pipeline (16 px / iter)
//!
//! `vld4q_u8` loads 16 quadruples (64 bytes) in one call, returning a
//! `uint8x16x4_t` where `.0 = V`, `.1 = U`, `.2 = Y`, `.3 = A`.
//! No shift is needed — samples are natively 8-bit.
//!
//! Each channel is split into low (lanes 0-7) and high (lanes 8-15)
//! halves, zero-extended to `int16x8_t`, and run through the shared
//! Q15 chroma + Y pipeline. The two halves are then narrowed to `u8`
//! and combined into `uint8x16_t` for interleaved store via `vst3q_u8`
//! (RGB) or `vst4q_u8` (RGBA).
//!
//! ## Tail
//!
//! `width % 16` remaining pixels fall through to `scalar::vuya_to_rgb_or_rgba_row`.

use super::*;
use crate::{KernelMatrix, row::scalar};

// ---- shared kernel template ---------------------------------------------

/// Pick the channel vector at byte offset `OFF` from a `vld4q_u8` result
/// (`q.0..q.3` are the four de-interleaved byte positions). A const index
/// keeps the selection a compile-time choice with no runtime branch.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pick4<const OFF: usize>(q: &uint8x16x4_t) -> uint8x16_t {
  match OFF {
    0 => q.0,
    1 => q.1,
    2 => q.2,
    _ => q.3,
  }
}

/// Shared per-block Q15 YUV→RGB compute for the NEON packed 4:4:4 family.
/// Takes the 16-lane V / U / Y byte vectors (already de-interleaved, in
/// natural pixel order) plus the broadcast range / coefficient vectors,
/// and returns the 16-lane R / G / B u8 vectors. Layout-independent — the
/// 4-byte (`vld4q_u8`) and 3-byte (`vld3q_u8`) loaders both feed this, so
/// the colour math stays bit-identical across the channel re-orderings and
/// the no-alpha 3-byte sibling.
///
/// # Safety
///
/// NEON must be available (caller's `#[target_feature]`).
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vuya_compute_rgb16(
  v_raw: uint8x16_t,
  u_raw: uint8x16_t,
  y_raw: uint8x16_t,
  y_off_v: int16x8_t,
  y_scale_v: int32x4_t,
  c_scale_v: int32x4_t,
  bias_v: int16x8_t,
  rnd_v: int32x4_t,
  cru: int32x4_t,
  crv: int32x4_t,
  cgu: int32x4_t,
  cgv: int32x4_t,
  cbu: int32x4_t,
  cbv: int32x4_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
  unsafe {
    // Zero-extend V/U/Y halves to i16x8 (8 lanes each).
    let v_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_raw)));
    let v_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_raw)));
    let u_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_raw)));
    let u_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u_raw)));
    let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_raw)));
    let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_raw)));

    // Subtract chroma bias (128 for 8-bit).
    let u_sub_lo = vsubq_s16(u_lo, bias_v);
    let u_sub_hi = vsubq_s16(u_hi, bias_v);
    let v_sub_lo = vsubq_s16(v_lo, bias_v);
    let v_sub_hi = vsubq_s16(v_hi, bias_v);

    // Widen to i32x4 lo/hi for Q15 chroma-scale multiply (low half).
    let u_lo_lo_i32 = vmovl_s16(vget_low_s16(u_sub_lo));
    let u_lo_hi_i32 = vmovl_s16(vget_high_s16(u_sub_lo));
    let v_lo_lo_i32 = vmovl_s16(vget_low_s16(v_sub_lo));
    let v_lo_hi_i32 = vmovl_s16(vget_high_s16(v_sub_lo));

    let u_d_lo_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_lo_i32, c_scale_v), rnd_v));
    let u_d_lo_hi = q15_shift(vaddq_s32(vmulq_s32(u_lo_hi_i32, c_scale_v), rnd_v));
    let v_d_lo_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_lo_i32, c_scale_v), rnd_v));
    let v_d_lo_hi = q15_shift(vaddq_s32(vmulq_s32(v_lo_hi_i32, c_scale_v), rnd_v));

    // Chroma for low 8 lanes.
    let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
    let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
    let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);

    // Widen to i32x4 lo/hi for Q15 chroma-scale multiply (high half).
    let u_hi_lo_i32 = vmovl_s16(vget_low_s16(u_sub_hi));
    let u_hi_hi_i32 = vmovl_s16(vget_high_s16(u_sub_hi));
    let v_hi_lo_i32 = vmovl_s16(vget_low_s16(v_sub_hi));
    let v_hi_hi_i32 = vmovl_s16(vget_high_s16(v_sub_hi));

    let u_d_hi_lo = q15_shift(vaddq_s32(vmulq_s32(u_hi_lo_i32, c_scale_v), rnd_v));
    let u_d_hi_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_hi_i32, c_scale_v), rnd_v));
    let v_d_hi_lo = q15_shift(vaddq_s32(vmulq_s32(v_hi_lo_i32, c_scale_v), rnd_v));
    let v_d_hi_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_hi_i32, c_scale_v), rnd_v));

    // Chroma for high 8 lanes.
    let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
    let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
    let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);

    // Y: scale both halves.
    let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
    let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

    // Saturate-add Y + chroma per channel, narrow both halves to u8,
    // then combine into a uint8x16_t.
    let r_u8 = vcombine_u8(
      vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
      vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
    );
    let g_u8 = vcombine_u8(
      vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
      vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
    );
    let b_u8 = vcombine_u8(
      vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
      vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
    );
    (r_u8, g_u8, b_u8)
  }
}

/// NEON packed 8-bit 4:4:4 → packed u8 RGB or RGBA, parameterized by the
/// per-pixel byte offsets of V / U / Y / A so the single kernel serves
/// every channel re-ordering of the 4-byte family (VUYA / VUYX with
/// `V=0,U=1,Y=2,A=3`; AYUV with `A=0,Y=1,U=2,V=3`; UYVA with
/// `U=0,Y=1,V=2,A=3`).
///
/// Byte-identical to
/// `scalar::packed444_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC, V_OFF, U_OFF, Y_OFF, A_OFF>`.
///
/// The three valid `(ALPHA, ALPHA_SRC)` monomorphizations are:
/// - `<false, false>` — RGB (drops α)
/// - `<true, true>`  — RGBA, source α pass-through
/// - `<true, false>` — RGBA, force α = `0xFF`
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn packed444_to_rgb_or_rgba_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
  const A_OFF: usize,
>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  let bias = scalar::chroma_bias::<8>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 quadruples (16 x 4 x u8 = 64 bytes). vld4q_u8
      // deinterleaves the four byte positions into .0..3; `pick4` selects
      // V/U/Y/A per the format's offsets.
      let q = vld4q_u8(packed.as_ptr().add(x * 4));
      let v_raw = pick4::<V_OFF>(&q); // uint8x16_t — 16 V bytes
      let u_raw = pick4::<U_OFF>(&q); // uint8x16_t — 16 U bytes
      let y_raw = pick4::<Y_OFF>(&q); // uint8x16_t — 16 Y bytes
      let a_raw = pick4::<A_OFF>(&q); // uint8x16_t — 16 A bytes (padding for VUYX)

      let (r_u8, g_u8, b_u8) = vuya_compute_rgb16(
        v_raw, u_raw, y_raw, y_off_v, y_scale_v, c_scale_v, bias_v, rnd_v, cru, crv, cgu, cgv, cbu,
        cbv,
      );

      // Store 16 pixels.
      let off = x * bpp;
      if ALPHA {
        let a_vec: uint8x16_t = if ALPHA_SRC {
          a_raw // source alpha pass-through (VUYA)
        } else {
          vdupq_n_u8(0xFFu8) // opaque (VUYX)
        };
        vst4q_u8(
          out.as_mut_ptr().add(off),
          uint8x16x4_t(r_u8, g_u8, b_u8, a_vec),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(off), uint8x16x3_t(r_u8, g_u8, b_u8));
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    let processed = x;
    if processed < width {
      let tail_packed = &packed[processed * 4..];
      let tail_out = &mut out[processed * bpp..];
      scalar::packed444_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC, V_OFF, U_OFF, Y_OFF, A_OFF>(
        tail_packed,
        tail_out,
        width - processed,
        matrix,
        full_range,
      );
    }
  }
}

/// VUYA / VUYX channel order (`V=0,U=1,Y=2,A=3`) over the offset-generic
/// [`packed444_to_rgb_or_rgba_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC, 0, 1, 2, 3>(
      packed, out, width, matrix, full_range,
    );
  }
}

// ---- thin wrappers -------------------------------------------------------

/// NEON VUYA / VUYX → packed **RGB** (3 bpp). Alpha byte in source is
/// discarded — RGB output has no alpha channel.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vuya_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// NEON VUYA → packed **RGBA** (4 bpp). Source A byte is passed through
/// verbatim.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// NEON VUYX → packed **RGBA** (4 bpp). Source A byte is padding;
/// output α is forced to `0xFF` (opaque).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuyx_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- luma extraction ----------------------------------------------------

/// NEON packed 8-bit 4:4:4 → u8 luma, parameterized by the Y byte offset
/// (`2` for VUYA / VUYX, `1` for AYUV / UYVA). `vld4q_u8` deinterleaves
/// the four byte positions; `pick4::<Y_OFF>` delivers Y.
///
/// Byte-identical to `scalar::packed444_to_luma_row::<Y_OFF>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn packed444_to_luma_row<const Y_OFF: usize>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let q = vld4q_u8(packed.as_ptr().add(x * 4));
      vst1q_u8(luma_out.as_mut_ptr().add(x), pick4::<Y_OFF>(&q));
      x += 16;
    }
    // Scalar tail.
    if x < width {
      scalar::packed444_to_luma_row::<Y_OFF>(&packed[x * 4..], &mut luma_out[x..], width - x);
    }
  }
}

/// VUYA / VUYX u8 luma (Y at offset 2) over [`packed444_to_luma_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<2>(packed, luma_out, width);
  }
}

/// NEON packed 8-bit 4:4:4 → u16 luma (zero-extended Y bytes),
/// parameterized by the Y byte offset. `vld4q_u8` deinterleaves; the Y
/// channel is widened to u16 via `vmovl_u8`. 16 pixels per iter.
///
/// Byte-identical to `scalar::packed444_to_luma_u16_row::<Y_OFF>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn packed444_to_luma_u16_row<const Y_OFF: usize>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(out.len() >= width, "out too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let q = vld4q_u8(packed.as_ptr().add(x * 4));
      let y = pick4::<Y_OFF>(&q);
      let y_lo = vmovl_u8(vget_low_u8(y)); // lanes 0-7 → u16x8
      let y_hi = vmovl_u8(vget_high_u8(y)); // lanes 8-15 → u16x8
      vst1q_u16(out.as_mut_ptr().add(x), y_lo);
      vst1q_u16(out.as_mut_ptr().add(x + 8), y_hi);
      x += 16;
    }
    // Scalar tail.
    if x < width {
      scalar::packed444_to_luma_u16_row::<Y_OFF>(&packed[x * 4..], &mut out[x..], width - x);
    }
  }
}

/// VUYA u16 luma (Y at offset 2) over [`packed444_to_luma_u16_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<2>(packed, out, width);
  }
}

/// NEON VUYX → u16 luma (zero-extended Y bytes). Byte-identical to
/// [`vuya_to_luma_u16_row`] — Y is at byte offset 2 of each quadruple
/// regardless of α semantics; the X byte is discarded.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuyx_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    vuya_to_luma_u16_row(packed, out, width);
  }
}

// ---- VUYA / VUYX → HSV (staged via a reused 8-bit RGB chunk) -----------
//
// The NEON twin of the scalar `vuya_to_hsv_row` kernel. Rather than
// re-derive an HSV-specific register pipeline, it fills a small fixed
// reused 8-bit RGB scratch (one `HSV_CHUNK`-pixel chunk at a time) using
// the EXISTING NEON `vuya_to_rgb_row` kernel of this file — so the chunk
// filler IS the production RGB kernel — then runs the NEON
// `rgb_to_hsv_row` on the chunk. This makes the result byte-identical to
// `rgb_to_hsv_row(vuya_to_rgb_row(...))` within the NEON tier, with no
// source-width RGB allocation. The scalar tail of the underlying RGB
// kernel handles widths below the SIMD block. The α byte (slot 3) is
// dropped by the RGB kernel — HSV is colour-only — so a single kernel
// serves both VUYA (real α) and VUYX (padding); `vuyx_to_hsv_row` is a
// thin re-export.
//
// The chunked driver is defined locally (mirroring the semi-planar
// high-bit `pn_hsv_via_rgb_chunks`) and gated `yuv-444-packed` with the
// rest of this file. Only `rgb_to_hsv_row` (ungated) is shared.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared NEON driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing NEON RGB
/// kernel, passed the chunk `offset` and length `n`), then runs the NEON
/// [`rgb_to_hsv_row`] on that chunk into the H/S/V planes. Byte-identical
/// to `rgb_to_hsv_row(vuya_to_rgb_row(...))` within the NEON tier, with
/// no source-width RGB allocation.
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
unsafe fn vuya_hsv_via_rgb_chunks(
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

/// NEON: VUYA / VUYX (packed 4:4:4, 8-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the reused-8-bit-RGB-chunk pattern over the
/// NEON [`vuya_to_rgb_row`] + [`rgb_to_hsv_row`]. Byte-identical to
/// `rgb_to_hsv_row(vuya_to_rgb_row(...))` within the NEON tier. The α
/// byte is dropped (HSV is colour-only).
///
/// # Safety
///
/// 1. The NEON feature must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn packed444_to_hsv_row<
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
>(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the NEON RGB kernel (offset-generic,
  // 3-bpp / no-alpha) under the same contract (its own scalar tail covers
  // small n). `A_OFF` is irrelevant in the RGB path; reuse `Y_OFF` as a
  // dummy 4th offset to satisfy the kernel signature.
  unsafe {
    vuya_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      packed444_to_rgb_or_rgba_row::<false, false, V_OFF, U_OFF, Y_OFF, Y_OFF>(
        &packed[offset * 4..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// VUYA / VUYX HSV (`V=0,U=1,Y=2`) over [`packed444_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<0, 1, 2>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

// ---- AYUV (A=0, Y=1, U=2, V=3) NEON wrappers --------------------------

/// NEON AYUV → packed **RGB** (3 bpp).
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<false, false, 3, 2, 1, 0>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON AYUV → packed **RGBA** (4 bpp). Source A byte (offset 0) passes
/// through verbatim.
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<true, true, 3, 2, 1, 0>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON AYUV → planar HSV bytes.
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<3, 2, 1>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// NEON AYUV → u8 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<1>(packed, luma_out, width);
  }
}

/// NEON AYUV → u16 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<1>(packed, out, width);
  }
}

// ---- UYVA (U=0, Y=1, V=2, A=3) NEON wrappers --------------------------

/// NEON UYVA → packed **RGB** (3 bpp).
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyva_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<false, false, 2, 0, 1, 3>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON UYVA → packed **RGBA** (4 bpp). Source A byte (offset 3) passes
/// through verbatim.
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyva_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<true, true, 2, 0, 1, 3>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON UYVA → planar HSV bytes.
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyva_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<2, 0, 1>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// NEON UYVA → u8 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyva_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<1>(packed, luma_out, width);
  }
}

/// NEON UYVA → u16 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyva_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<1>(packed, out, width);
  }
}

// ---- VYU444 (V=0, Y=1, U=2; 3 bytes per pixel, no alpha) NEON ----------
//
// VYU444 packs three bytes per pixel (`V ‖ Y ‖ U`, 24bpp). `vld3q_u8`
// deinterleaves 16 triples into `.0 = V`, `.1 = Y`, `.2 = U`, feeding the
// shared [`vuya_compute_rgb16`] colour math. RGBA output forces α = `0xFF`
// (no source alpha). The scalar tail handles `width % 16`.

/// NEON VYU444 → packed u8 RGB (`ALPHA = false`) or RGBA (`ALPHA = true`,
/// α forced `0xFF`). Byte-identical to
/// `scalar::vyu444_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 3`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vyu444_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  let bias = scalar::chroma_bias::<8>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // vld3q_u8 deinterleaves 16 triples: .0 = V, .1 = Y, .2 = U.
      let q = vld3q_u8(packed.as_ptr().add(x * 3));
      let v_raw = q.0;
      let y_raw = q.1;
      let u_raw = q.2;

      let (r_u8, g_u8, b_u8) = vuya_compute_rgb16(
        v_raw, u_raw, y_raw, y_off_v, y_scale_v, c_scale_v, bias_v, rnd_v, cru, crv, cgu, cgv, cbu,
        cbv,
      );

      let off = x * bpp;
      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(off),
          uint8x16x4_t(r_u8, g_u8, b_u8, vdupq_n_u8(0xFFu8)),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(off), uint8x16x3_t(r_u8, g_u8, b_u8));
      }

      x += 16;
    }

    let processed = x;
    if processed < width {
      let tail_packed = &packed[processed * 3..];
      let tail_out = &mut out[processed * bpp..];
      scalar::vyu444_to_rgb_or_rgba_row::<ALPHA>(
        tail_packed,
        tail_out,
        width - processed,
        matrix,
        full_range,
      );
    }
  }
}

/// NEON VYU444 → packed RGB (3 bpp).
///
/// # Safety
///
/// Same contract as [`vyu444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vyu444_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vyu444_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// NEON VYU444 → packed RGBA (4 bpp, α forced `0xFF`).
///
/// # Safety
///
/// Same contract as [`vyu444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vyu444_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vyu444_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// NEON VYU444 → u8 luma (Y at offset 1, 3-byte stride). `vld3q_u8`'s
/// channel 1 delivers Y for all 16 pixels.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 3`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vyu444_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let q = vld3q_u8(packed.as_ptr().add(x * 3));
      vst1q_u8(luma_out.as_mut_ptr().add(x), q.1);
      x += 16;
    }
    if x < width {
      scalar::vyu444_to_luma_row(&packed[x * 3..], &mut luma_out[x..], width - x);
    }
  }
}

/// NEON VYU444 → u16 luma (zero-extended Y at offset 1, 3-byte stride).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 3`.
/// 3. `out.len() >= width`.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vyu444_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  debug_assert!(out.len() >= width, "out too short");
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let q = vld3q_u8(packed.as_ptr().add(x * 3));
      let y_lo = vmovl_u8(vget_low_u8(q.1));
      let y_hi = vmovl_u8(vget_high_u8(q.1));
      vst1q_u16(out.as_mut_ptr().add(x), y_lo);
      vst1q_u16(out.as_mut_ptr().add(x + 8), y_hi);
      x += 16;
    }
    if x < width {
      scalar::vyu444_to_luma_u16_row(&packed[x * 3..], &mut out[x..], width - x);
    }
  }
}

/// NEON VYU444 → planar HSV bytes, staged via the reused-RGB-chunk pattern
/// over the NEON [`vyu444_to_rgb_row`] + [`rgb_to_hsv_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 3`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vyu444_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");
  unsafe {
    vuya_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      vyu444_to_rgb_row(&packed[offset * 3..], rgb, n, matrix, full_range);
    });
  }
}
