//! NEON centered (#302 phase-0.5) 2:1 horizontal chroma-upsample
//! reconstruct kernels — the SIMD twins of the scalar
//! [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h)
//! family (u8 planar, u16 planar, and the interleaved-UV semi-planar
//! P-format). Each output pair is the fixed-weight neighbour blend
//!
//! ```text
//!   even col 2j   → (c[j-1] + 3·c[j] + 2) >> 2   (c[-1]   clamps to c[0])
//!   odd  col 2j+1 → (3·c[j] + c[j+1] + 2) >> 2   (c[half] clamps to c[half-1])
//! ```
//!
//! computed with offset loads (`c[j-1]`, `c[j]`, `c[j+1]`) and an
//! interleaved store (`vst2` for the planar even/odd stream, `vst4` for
//! the P-format `U,V,U,V` stream). The two boundary columns (`j = 0` and
//! `j = half-1`) — the only samples whose neighbours clamp — reuse the
//! shared scalar per-sample reference so the edges stay byte-identical to
//! the scalar kernel; the vector loop covers the strict interior where
//! every neighbour is a real, in-bounds sample. Bit-identical to the
//! scalar reference for every input (asserted by the equivalence tests
//! below).

#![cfg_attr(
  not(all(
    any(feature = "std", feature = "alloc"),
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  )),
  allow(dead_code)
)]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use crate::row::scalar;

/// Conditionally byte-swaps eight `u16` lanes so the returned vector is in
/// host-native order. `swap` is `big_endian != host_is_big_endian`,
/// computed once by the caller.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "neon")]
fn maybe_bswap_u16x8(v: uint16x8_t, swap: bool) -> uint16x8_t {
  if swap {
    vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v)))
  } else {
    v
  }
}

/// Blends eight interior chroma samples in `u32` lanes and narrows back to
/// `u16`: returns `(even, odd)` where `even = (left + 3·mid + 2) >> 2` and
/// `odd = (3·mid + right + 2) >> 2`. The inputs are logical (masked /
/// de-packed) `u16` samples; the widen-to-`u32` avoids the `16 · 65535`
/// overflow the `u16`-domain accumulator would hit at `BITS = 16`, and the
/// narrow is exact because each result is `≤ (1 << BITS) - 1 ≤ 65535`.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "neon")]
fn blend_u16x8(left: uint16x8_t, mid: uint16x8_t, right: uint16x8_t) -> (uint16x8_t, uint16x8_t) {
  let two = vdupq_n_u32(2);
  let three = vdupq_n_u32(3);
  let mid_lo = vmovl_u16(vget_low_u16(mid));
  let mid_hi = vmovl_high_u16(mid);
  let left_lo = vmovl_u16(vget_low_u16(left));
  let left_hi = vmovl_high_u16(left);
  let right_lo = vmovl_u16(vget_low_u16(right));
  let right_hi = vmovl_high_u16(right);
  let tm_lo = vmulq_u32(mid_lo, three);
  let tm_hi = vmulq_u32(mid_hi, three);
  let even_lo = vshrq_n_u32::<2>(vaddq_u32(vaddq_u32(left_lo, tm_lo), two));
  let even_hi = vshrq_n_u32::<2>(vaddq_u32(vaddq_u32(left_hi, tm_hi), two));
  let odd_lo = vshrq_n_u32::<2>(vaddq_u32(vaddq_u32(tm_lo, right_lo), two));
  let odd_hi = vshrq_n_u32::<2>(vaddq_u32(vaddq_u32(tm_hi, right_hi), two));
  let even = vcombine_u16(vmovn_u32(even_lo), vmovn_u32(even_hi));
  let odd = vcombine_u16(vmovn_u32(odd_lo), vmovn_u32(odd_hi));
  (even, odd)
}

/// NEON u8 centered 2:1 horizontal chroma upsample — the SIMD twin of
/// [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h).
///
/// Block size: 16 chroma samples / iter (→ 32 output columns via `vst2q_u8`).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `c_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_2to1_center_h_row(
  c_half: &[u8],
  c_full: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0, "2:1 horizontal chroma requires even width");
  debug_assert!(c_half.len() >= width / 2, "c_half row too short");
  debug_assert!(c_full.len() >= width, "c_full row too short");

  let half = width / 2;
  if half == 0 {
    return;
  }
  // Left-edge column (clamps `c[-1]` to `c[0]`).
  scalar::chroma_upsample_2to1_center_h_pair(c_half, c_full, 0, half);
  if half == 1 {
    return;
  }

  let mut j = 1usize;
  // SAFETY: `j + 16 < half` keeps every offset load inside `c_half[0..half]`
  // and every 32-byte store inside `c_full[0..width]`; the interior samples
  // read here have real (non-clamped) neighbours, matching the scalar body.
  unsafe {
    let two = vdupq_n_u16(2);
    let three = vdupq_n_u16(3);
    while j + 16 < half {
      let mid = vld1q_u8(c_half.as_ptr().add(j));
      let left = vld1q_u8(c_half.as_ptr().add(j - 1));
      let right = vld1q_u8(c_half.as_ptr().add(j + 1));
      let mid_lo = vmovl_u8(vget_low_u8(mid));
      let mid_hi = vmovl_u8(vget_high_u8(mid));
      let left_lo = vmovl_u8(vget_low_u8(left));
      let left_hi = vmovl_u8(vget_high_u8(left));
      let right_lo = vmovl_u8(vget_low_u8(right));
      let right_hi = vmovl_u8(vget_high_u8(right));
      let tm_lo = vmulq_u16(mid_lo, three);
      let tm_hi = vmulq_u16(mid_hi, three);
      let even_lo = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(left_lo, tm_lo), two));
      let even_hi = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(left_hi, tm_hi), two));
      let odd_lo = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(tm_lo, right_lo), two));
      let odd_hi = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(tm_hi, right_hi), two));
      let even = vcombine_u8(vmovn_u16(even_lo), vmovn_u16(even_hi));
      let odd = vcombine_u8(vmovn_u16(odd_lo), vmovn_u16(odd_hi));
      vst2q_u8(c_full.as_mut_ptr().add(2 * j), uint8x16x2_t(even, odd));
      j += 16;
    }
  }

  // Remaining interior samples plus the right-edge column (`c[half]` clamps
  // to `c[half-1]`).
  while j < half {
    scalar::chroma_upsample_2to1_center_h_pair(c_half, c_full, j, half);
    j += 1;
  }
}

/// NEON u16 centered 2:1 horizontal chroma upsample — the SIMD twin of
/// [`chroma_upsample_2to1_center_h_u16`](crate::row::scalar::chroma_upsample_2to1_center_h_u16).
///
/// Block size: 8 chroma samples / iter (→ 16 output columns via `vst2q_u16`).
/// Samples are normalized wire → host-native, masked to the low `BITS`,
/// blended, and re-encoded to the same wire order — bit-identical to the
/// scalar reference per tier.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `c_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_2to1_center_h_u16_row<const BITS: u32>(
  c_half: &[u16],
  c_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "2:1 horizontal chroma requires even width");
  debug_assert!(c_half.len() >= width / 2, "c_half row too short");
  debug_assert!(c_full.len() >= width, "c_full row too short");

  let half = width / 2;
  if half == 0 {
    return;
  }
  scalar::chroma_upsample_2to1_center_h_u16_pair::<BITS>(c_half, c_full, 0, half, big_endian);
  if half == 1 {
    return;
  }

  let swap = big_endian != cfg!(target_endian = "big");
  let mut j = 1usize;
  // SAFETY: `j + 8 < half` keeps every offset load inside `c_half[0..half]`
  // and every 16-`u16` store inside `c_full[0..width]`.
  unsafe {
    let mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    while j + 8 < half {
      let mid = vandq_u16(
        maybe_bswap_u16x8(vld1q_u16(c_half.as_ptr().add(j)), swap),
        mask,
      );
      let left = vandq_u16(
        maybe_bswap_u16x8(vld1q_u16(c_half.as_ptr().add(j - 1)), swap),
        mask,
      );
      let right = vandq_u16(
        maybe_bswap_u16x8(vld1q_u16(c_half.as_ptr().add(j + 1)), swap),
        mask,
      );
      let (even, odd) = blend_u16x8(left, mid, right);
      let even = maybe_bswap_u16x8(even, swap);
      let odd = maybe_bswap_u16x8(odd, swap);
      vst2q_u16(c_full.as_mut_ptr().add(2 * j), uint16x8x2_t(even, odd));
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_u16_pair::<BITS>(c_half, c_full, j, half, big_endian);
    j += 1;
  }
}

/// De-packs eight wire `u16` samples to logical values: byte-swaps to
/// host-native (when `swap`), then shifts the active bits down
/// (`>> (16 - BITS)`, high-bit-packed) or masks the low `BITS`
/// (`& ((1 << BITS) - 1)`, low-packed NV20). `LOW_PACKED` is const, so the
/// branch folds; the shift arrives as `neg_shift = vdupq_n_s16(-(16-BITS))`
/// (a `vshlq_u16` register right-shift, sidestepping the const-generic
/// shift-amount limitation).
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "neon")]
fn depack_u16x8<const LOW_PACKED: bool>(
  v: uint16x8_t,
  swap: bool,
  mask: uint16x8_t,
  neg_shift: int16x8_t,
) -> uint16x8_t {
  let host = maybe_bswap_u16x8(v, swap);
  if LOW_PACKED {
    vandq_u16(host, mask)
  } else {
    vshlq_u16(host, neg_shift)
  }
}

/// Re-packs eight logical `u16` samples back to wire order — the inverse of
/// [`depack_u16x8`]: re-aligns to MSB (`<< (16 - BITS)`, high-bit-packed) or
/// keeps the value in the low `BITS` (low-packed), then byte-swaps back to
/// wire order (when `swap`). `pos_shift = vdupq_n_s16(16 - BITS)`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "neon")]
fn repack_u16x8<const LOW_PACKED: bool>(
  v: uint16x8_t,
  swap: bool,
  pos_shift: int16x8_t,
) -> uint16x8_t {
  let packed = if LOW_PACKED {
    v
  } else {
    vshlq_u16(v, pos_shift)
  };
  maybe_bswap_u16x8(packed, swap)
}

/// NEON semi-planar P-format centered 2:1 horizontal chroma upsample — the
/// SIMD twin of
/// [`chroma_upsample_2to1_center_h_p0xx`](crate::row::scalar::chroma_upsample_2to1_center_h_p0xx).
///
/// Block size: 8 chroma samples / iter. `vld2q_u16` de-interleaves the
/// half-row `U`/`V`, each channel is de-packed (`>> (16 - BITS)` for the
/// high-bit-packed P-formats, `& ((1 << BITS) - 1)` for low-packed NV20),
/// blended, re-packed, and `vst4q_u16` re-interleaves the `U,V,U,V` output
/// — bit-identical to the scalar reference per tier.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `uv_half.len() >= width`; `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_2to1_center_h_p0xx_row<
  const BITS: u32,
  const LOW_PACKED: bool,
>(
  uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(uv_half.len() >= width, "uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let half = width / 2;
  if half == 0 {
    return;
  }
  scalar::chroma_upsample_2to1_center_h_p0xx_pair::<BITS, LOW_PACKED>(
    uv_half, uv_full, 0, half, big_endian,
  );
  if half == 1 {
    return;
  }

  let swap = big_endian != cfg!(target_endian = "big");
  let shift = (16 - BITS) as i16;
  let mut j = 1usize;
  // SAFETY: `j + 8 < half` keeps every `vld2q_u16` inside `uv_half[0..width]`
  // and every `vst4q_u16` inside `uv_full[0..2*width]`.
  unsafe {
    let mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let neg_shift = vdupq_n_s16(-shift);
    let pos_shift = vdupq_n_s16(shift);
    while j + 8 < half {
      let l = vld2q_u16(uv_half.as_ptr().add(2 * (j - 1)));
      let m = vld2q_u16(uv_half.as_ptr().add(2 * j));
      let r = vld2q_u16(uv_half.as_ptr().add(2 * (j + 1)));
      let (u_even, u_odd) = blend_u16x8(
        depack_u16x8::<LOW_PACKED>(l.0, swap, mask, neg_shift),
        depack_u16x8::<LOW_PACKED>(m.0, swap, mask, neg_shift),
        depack_u16x8::<LOW_PACKED>(r.0, swap, mask, neg_shift),
      );
      let (v_even, v_odd) = blend_u16x8(
        depack_u16x8::<LOW_PACKED>(l.1, swap, mask, neg_shift),
        depack_u16x8::<LOW_PACKED>(m.1, swap, mask, neg_shift),
        depack_u16x8::<LOW_PACKED>(r.1, swap, mask, neg_shift),
      );
      vst4q_u16(
        uv_full.as_mut_ptr().add(4 * j),
        uint16x8x4_t(
          repack_u16x8::<LOW_PACKED>(u_even, swap, pos_shift),
          repack_u16x8::<LOW_PACKED>(v_even, swap, pos_shift),
          repack_u16x8::<LOW_PACKED>(u_odd, swap, pos_shift),
          repack_u16x8::<LOW_PACKED>(v_odd, swap, pos_shift),
        ),
      );
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_p0xx_pair::<BITS, LOW_PACKED>(
      uv_half, uv_full, j, half, big_endian,
    );
    j += 1;
  }
}

/// NEON u8 bottom-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottom_even_h`](crate::row::scalar::chroma_upsample_420_bottom_even_h).
/// Each interior offset's `e = (prev + cur + 1) >> 1` is one `vrhaddq_u8`
/// (rounding halving add), fed into the same centered `1/4`–`3/4` blend as the
/// horizontal sibling. Boundary columns reuse the shared scalar pair.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns via `vst2q_u8`).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `prev_half.len() >= width / 2`; `cur_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_420_bottom_even_h_row(
  prev_half: &[u8],
  cur_half: &[u8],
  c_full: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(prev_half.len() >= width / 2, "prev_half row too short");
  debug_assert!(cur_half.len() >= width / 2, "cur_half row too short");
  debug_assert!(c_full.len() >= width, "c_full row too short");

  let half = width / 2;
  if half == 0 {
    return;
  }
  scalar::chroma_upsample_420_bottom_even_h_pair(prev_half, cur_half, c_full, 0, half);
  if half == 1 {
    return;
  }

  let mut j = 1usize;
  // SAFETY: `j + 16 < half` keeps every offset load inside `prev_half` /
  // `cur_half[0..half]` and every 32-byte store inside `c_full[0..width]`; the
  // interior samples read here have real (non-clamped) neighbours.
  unsafe {
    let two = vdupq_n_u16(2);
    let three = vdupq_n_u16(3);
    while j + 16 < half {
      let e_left = vrhaddq_u8(
        vld1q_u8(prev_half.as_ptr().add(j - 1)),
        vld1q_u8(cur_half.as_ptr().add(j - 1)),
      );
      let e_mid = vrhaddq_u8(
        vld1q_u8(prev_half.as_ptr().add(j)),
        vld1q_u8(cur_half.as_ptr().add(j)),
      );
      let e_right = vrhaddq_u8(
        vld1q_u8(prev_half.as_ptr().add(j + 1)),
        vld1q_u8(cur_half.as_ptr().add(j + 1)),
      );
      let mid_lo = vmovl_u8(vget_low_u8(e_mid));
      let mid_hi = vmovl_u8(vget_high_u8(e_mid));
      let left_lo = vmovl_u8(vget_low_u8(e_left));
      let left_hi = vmovl_u8(vget_high_u8(e_left));
      let right_lo = vmovl_u8(vget_low_u8(e_right));
      let right_hi = vmovl_u8(vget_high_u8(e_right));
      let tm_lo = vmulq_u16(mid_lo, three);
      let tm_hi = vmulq_u16(mid_hi, three);
      let even_lo = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(left_lo, tm_lo), two));
      let even_hi = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(left_hi, tm_hi), two));
      let odd_lo = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(tm_lo, right_lo), two));
      let odd_hi = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(tm_hi, right_hi), two));
      let even = vcombine_u8(vmovn_u16(even_lo), vmovn_u16(even_hi));
      let odd = vcombine_u8(vmovn_u16(odd_lo), vmovn_u16(odd_hi));
      vst2q_u8(c_full.as_mut_ptr().add(2 * j), uint8x16x2_t(even, odd));
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottom_even_h_pair(prev_half, cur_half, c_full, j, half);
    j += 1;
  }
}

/// NEON u8 bottom-left-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottomleft_even_h`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h).
/// The co-sited (`h = 0`) horizontal phase is a plain 2× replicate with no
/// neighbour, so each column is independent: `e = vrhaddq_u8(prev, cur)` then a
/// self-interleaved `vst2q_u8(e, e)`.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `prev_half.len() >= width / 2`; `cur_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_420_bottomleft_even_h_row(
  prev_half: &[u8],
  cur_half: &[u8],
  c_full: &mut [u8],
  width: usize,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(prev_half.len() >= width / 2, "prev_half row too short");
  debug_assert!(cur_half.len() >= width / 2, "cur_half row too short");
  debug_assert!(c_full.len() >= width, "c_full row too short");

  let half = width / 2;
  let mut j = 0usize;
  // SAFETY: `j + 16 <= half` keeps every load inside `prev_half` /
  // `cur_half[0..half]` and every 32-byte store inside `c_full[0..width]`.
  unsafe {
    while j + 16 <= half {
      let e = vrhaddq_u8(
        vld1q_u8(prev_half.as_ptr().add(j)),
        vld1q_u8(cur_half.as_ptr().add(j)),
      );
      vst2q_u8(c_full.as_mut_ptr().add(2 * j), uint8x16x2_t(e, e));
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottomleft_even_h_pair(prev_half, cur_half, c_full, j);
    j += 1;
  }
}

/// NEON u16 bottom-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottom_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottom_even_h_u16).
/// Each offset's `e` is `vrhaddq_u16` of the two masked, host-normalized rows,
/// fed into the same centered blend as the horizontal sibling; the output is
/// re-encoded to wire order — bit-identical to the scalar reference per tier.
///
/// Block size: 8 chroma samples / iter (→ 16 output columns via `vst2q_u16`).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `prev_half.len() >= width / 2`; `cur_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_420_bottom_even_h_u16_row<const BITS: u32>(
  prev_half: &[u16],
  cur_half: &[u16],
  c_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(prev_half.len() >= width / 2, "prev_half row too short");
  debug_assert!(cur_half.len() >= width / 2, "cur_half row too short");
  debug_assert!(c_full.len() >= width, "c_full row too short");

  let half = width / 2;
  if half == 0 {
    return;
  }
  scalar::chroma_upsample_420_bottom_even_h_u16_pair::<BITS>(
    prev_half, cur_half, c_full, 0, half, big_endian,
  );
  if half == 1 {
    return;
  }

  let swap = big_endian != cfg!(target_endian = "big");
  let mut j = 1usize;
  // SAFETY: `j + 8 < half` keeps every offset load inside the half rows and
  // every 16-`u16` store inside `c_full[0..width]`.
  unsafe {
    let mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    while j + 8 < half {
      let e_left = vrhaddq_u16(
        vandq_u16(
          maybe_bswap_u16x8(vld1q_u16(prev_half.as_ptr().add(j - 1)), swap),
          mask,
        ),
        vandq_u16(
          maybe_bswap_u16x8(vld1q_u16(cur_half.as_ptr().add(j - 1)), swap),
          mask,
        ),
      );
      let e_mid = vrhaddq_u16(
        vandq_u16(
          maybe_bswap_u16x8(vld1q_u16(prev_half.as_ptr().add(j)), swap),
          mask,
        ),
        vandq_u16(
          maybe_bswap_u16x8(vld1q_u16(cur_half.as_ptr().add(j)), swap),
          mask,
        ),
      );
      let e_right = vrhaddq_u16(
        vandq_u16(
          maybe_bswap_u16x8(vld1q_u16(prev_half.as_ptr().add(j + 1)), swap),
          mask,
        ),
        vandq_u16(
          maybe_bswap_u16x8(vld1q_u16(cur_half.as_ptr().add(j + 1)), swap),
          mask,
        ),
      );
      let (even, odd) = blend_u16x8(e_left, e_mid, e_right);
      let even = maybe_bswap_u16x8(even, swap);
      let odd = maybe_bswap_u16x8(odd, swap);
      vst2q_u16(c_full.as_mut_ptr().add(2 * j), uint16x8x2_t(even, odd));
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottom_even_h_u16_pair::<BITS>(
      prev_half, cur_half, c_full, j, half, big_endian,
    );
    j += 1;
  }
}

/// NEON u16 bottom-left-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottomleft_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16).
/// Per-column `e = vrhaddq_u16` of the masked, host-normalized rows, re-encoded
/// to wire order and replicated across the column pair (`vst2q_u16(e, e)`).
///
/// Block size: 8 chroma samples / iter (→ 16 output columns).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `prev_half.len() >= width / 2`; `cur_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_420_bottomleft_even_h_u16_row<const BITS: u32>(
  prev_half: &[u16],
  cur_half: &[u16],
  c_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(prev_half.len() >= width / 2, "prev_half row too short");
  debug_assert!(cur_half.len() >= width / 2, "cur_half row too short");
  debug_assert!(c_full.len() >= width, "c_full row too short");

  let half = width / 2;
  let swap = big_endian != cfg!(target_endian = "big");
  let mut j = 0usize;
  // SAFETY: `j + 8 <= half` keeps every load inside the half rows and every
  // 16-`u16` store inside `c_full[0..width]`.
  unsafe {
    let mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    while j + 8 <= half {
      let e = vrhaddq_u16(
        vandq_u16(
          maybe_bswap_u16x8(vld1q_u16(prev_half.as_ptr().add(j)), swap),
          mask,
        ),
        vandq_u16(
          maybe_bswap_u16x8(vld1q_u16(cur_half.as_ptr().add(j)), swap),
          mask,
        ),
      );
      let e = maybe_bswap_u16x8(e, swap);
      vst2q_u16(c_full.as_mut_ptr().add(2 * j), uint16x8x2_t(e, e));
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottomleft_even_h_u16_pair::<BITS>(
      prev_half, cur_half, c_full, j, big_endian,
    );
    j += 1;
  }
}

/// NEON semi-planar P-format bottom-sited even-row 4:2:0 chroma upsample — the
/// SIMD twin of
/// [`chroma_upsample_420_bottom_even_h_p0xx`](crate::row::scalar::chroma_upsample_420_bottom_even_h_p0xx).
/// High-bit-packed (P010/P012/P016) only; `vld2q_u16` de-interleaves `U`/`V`,
/// each channel's `e = vrhaddq_u16` of the de-packed prev / cur rows feeds the
/// centered blend, and `vst4q_u16` re-interleaves the re-packed `U,V,U,V`.
///
/// Block size: 8 chroma samples / iter.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `prev_uv_half.len() >= width`; `cur_uv_half.len() >= width`;
/// `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_420_bottom_even_h_p0xx_row<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(prev_uv_half.len() >= width, "prev_uv_half row too short");
  debug_assert!(cur_uv_half.len() >= width, "cur_uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let half = width / 2;
  if half == 0 {
    return;
  }
  scalar::chroma_upsample_420_bottom_even_h_p0xx_pair::<BITS>(
    prev_uv_half,
    cur_uv_half,
    uv_full,
    0,
    half,
    big_endian,
  );
  if half == 1 {
    return;
  }

  let swap = big_endian != cfg!(target_endian = "big");
  let shift = (16 - BITS) as i16;
  let mut j = 1usize;
  // SAFETY: `j + 8 < half` keeps every `vld2q_u16` inside the half rows and
  // every `vst4q_u16` inside `uv_full[0..2*width]`.
  unsafe {
    let mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let neg_shift = vdupq_n_s16(-shift);
    let pos_shift = vdupq_n_s16(shift);
    while j + 8 < half {
      let pl = vld2q_u16(prev_uv_half.as_ptr().add(2 * (j - 1)));
      let cl = vld2q_u16(cur_uv_half.as_ptr().add(2 * (j - 1)));
      let pm = vld2q_u16(prev_uv_half.as_ptr().add(2 * j));
      let cm = vld2q_u16(cur_uv_half.as_ptr().add(2 * j));
      let pr = vld2q_u16(prev_uv_half.as_ptr().add(2 * (j + 1)));
      let cr = vld2q_u16(cur_uv_half.as_ptr().add(2 * (j + 1)));
      let (u_even, u_odd) = blend_u16x8(
        vrhaddq_u16(
          depack_u16x8::<false>(pl.0, swap, mask, neg_shift),
          depack_u16x8::<false>(cl.0, swap, mask, neg_shift),
        ),
        vrhaddq_u16(
          depack_u16x8::<false>(pm.0, swap, mask, neg_shift),
          depack_u16x8::<false>(cm.0, swap, mask, neg_shift),
        ),
        vrhaddq_u16(
          depack_u16x8::<false>(pr.0, swap, mask, neg_shift),
          depack_u16x8::<false>(cr.0, swap, mask, neg_shift),
        ),
      );
      let (v_even, v_odd) = blend_u16x8(
        vrhaddq_u16(
          depack_u16x8::<false>(pl.1, swap, mask, neg_shift),
          depack_u16x8::<false>(cl.1, swap, mask, neg_shift),
        ),
        vrhaddq_u16(
          depack_u16x8::<false>(pm.1, swap, mask, neg_shift),
          depack_u16x8::<false>(cm.1, swap, mask, neg_shift),
        ),
        vrhaddq_u16(
          depack_u16x8::<false>(pr.1, swap, mask, neg_shift),
          depack_u16x8::<false>(cr.1, swap, mask, neg_shift),
        ),
      );
      vst4q_u16(
        uv_full.as_mut_ptr().add(4 * j),
        uint16x8x4_t(
          repack_u16x8::<false>(u_even, swap, pos_shift),
          repack_u16x8::<false>(v_even, swap, pos_shift),
          repack_u16x8::<false>(u_odd, swap, pos_shift),
          repack_u16x8::<false>(v_odd, swap, pos_shift),
        ),
      );
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottom_even_h_p0xx_pair::<BITS>(
      prev_uv_half,
      cur_uv_half,
      uv_full,
      j,
      half,
      big_endian,
    );
    j += 1;
  }
}

/// NEON semi-planar P-format bottom-left-sited even-row 4:2:0 chroma upsample —
/// the SIMD twin of
/// [`chroma_upsample_420_bottomleft_even_h_p0xx`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h_p0xx).
/// High-bit-packed only; per-column `e = vrhaddq_u16` of the de-packed prev /
/// cur `U` and `V`, re-packed and replicated across the column pair.
///
/// Block size: 8 chroma samples / iter.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `width` even;
/// `prev_uv_half.len() >= width`; `cur_uv_half.len() >= width`;
/// `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_420_bottomleft_even_h_p0xx_row<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(prev_uv_half.len() >= width, "prev_uv_half row too short");
  debug_assert!(cur_uv_half.len() >= width, "cur_uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let half = width / 2;
  let swap = big_endian != cfg!(target_endian = "big");
  let shift = (16 - BITS) as i16;
  let mut j = 0usize;
  // SAFETY: `j + 8 <= half` keeps every `vld2q_u16` inside the half rows and
  // every `vst4q_u16` inside `uv_full[0..2*width]`.
  unsafe {
    let mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let neg_shift = vdupq_n_s16(-shift);
    let pos_shift = vdupq_n_s16(shift);
    while j + 8 <= half {
      let p = vld2q_u16(prev_uv_half.as_ptr().add(2 * j));
      let c = vld2q_u16(cur_uv_half.as_ptr().add(2 * j));
      let e_u = repack_u16x8::<false>(
        vrhaddq_u16(
          depack_u16x8::<false>(p.0, swap, mask, neg_shift),
          depack_u16x8::<false>(c.0, swap, mask, neg_shift),
        ),
        swap,
        pos_shift,
      );
      let e_v = repack_u16x8::<false>(
        vrhaddq_u16(
          depack_u16x8::<false>(p.1, swap, mask, neg_shift),
          depack_u16x8::<false>(c.1, swap, mask, neg_shift),
        ),
        swap,
        pos_shift,
      );
      vst4q_u16(
        uv_full.as_mut_ptr().add(4 * j),
        uint16x8x4_t(e_u, e_v, e_u, e_v),
      );
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottomleft_even_h_p0xx_pair::<BITS>(
      prev_uv_half,
      cur_uv_half,
      uv_full,
      j,
      big_endian,
    );
    j += 1;
  }
}

/// NEON full-width vertical chroma rounding-average for the **bottom-sited**
/// even output luma row of a 4:4:0 source. Byte-identical to
/// [`chroma_upsample_440_bottom_v`](crate::row::scalar::chroma_upsample_440_bottom_v):
/// `out[j] = (prev[j] + cur[j] + 1) >> 1` for every column, via `vrhaddq_u8`
/// (16 lanes per iteration) with a scalar tail. 4:4:0 keeps full-width chroma,
/// so there is no horizontal reconstruction — just the vertical average.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `prev.len() >= width`;
/// `cur.len() >= width`; `out.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_440_bottom_v_row(
  prev: &[u8],
  cur: &[u8],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(prev.len() >= width, "prev row too short");
  debug_assert!(cur.len() >= width, "cur row too short");
  debug_assert!(out.len() >= width, "out row too short");

  let mut j = 0;
  // SAFETY: each iteration reads/writes 16 bytes at offset `j` with
  // `j + 16 <= width <= len`, so every access stays in bounds.
  unsafe {
    while j + 16 <= width {
      let p = vld1q_u8(prev.as_ptr().add(j));
      let c = vld1q_u8(cur.as_ptr().add(j));
      vst1q_u8(out.as_mut_ptr().add(j), vrhaddq_u8(p, c));
      j += 16;
    }
  }
  // Scalar tail (`j < width <= len`, so the indexing cannot panic).
  while j < width {
    out[j] = (((prev[j] as u16) + (cur[j] as u16) + 1) >> 1) as u8;
    j += 1;
  }
}

/// NEON `u16` twin of [`chroma_upsample_440_bottom_v_row`] for the high-bit
/// planar 4:4:0 sink, byte-identical to
/// [`chroma_upsample_440_bottom_v_u16_wire`](crate::row::scalar::chroma_upsample_440_bottom_v_u16_wire):
/// each 8-lane block is normalized wire → host, masked to the low `BITS`,
/// averaged with `vrhaddq_u16`, and re-encoded to the same wire order, with a
/// scalar tail. `prev` / `cur` / `out` all stay in the source's wire byte order.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `prev.len() >= width`;
/// `cur.len() >= width`; `out.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_440_bottom_v_u16_row<const BITS: u32>(
  prev: &[u16],
  cur: &[u16],
  out: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert!(prev.len() >= width, "prev row too short");
  debug_assert!(cur.len() >= width, "cur row too short");
  debug_assert!(out.len() >= width, "out row too short");

  let swap = big_endian != cfg!(target_endian = "big");
  let mut j = 0;
  // SAFETY: each iteration reads/writes 8 u16 lanes at offset `j` with
  // `j + 8 <= width <= len`, so every access stays in bounds.
  unsafe {
    let mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    while j + 8 <= width {
      let p = vandq_u16(
        maybe_bswap_u16x8(vld1q_u16(prev.as_ptr().add(j)), swap),
        mask,
      );
      let c = vandq_u16(
        maybe_bswap_u16x8(vld1q_u16(cur.as_ptr().add(j)), swap),
        mask,
      );
      let avg = maybe_bswap_u16x8(vrhaddq_u16(p, c), swap);
      vst1q_u16(out.as_mut_ptr().add(j), avg);
      j += 8;
    }
  }
  // Scalar tail — byte-identical to the wire-order scalar reference.
  let mask = ((1u32 << BITS) - 1) as u16;
  let load = |raw: u16| -> u32 {
    let logical = if big_endian {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    u32::from(logical & mask)
  };
  while j < width {
    let blended = ((load(prev[j]) + load(cur[j]) + 1) >> 1) as u16;
    out[j] = if big_endian {
      blended.to_be()
    } else {
      blended.to_le()
    };
    j += 1;
  }
}

/// Computes the four fixed-weight phase outputs of one 16-sample interior block
/// of the centered 1→4 upsample, returning `(a, b, c, d)` where, per lane,
/// `a = (3·left + 5·mid + 4) >> 3`, `b = (left + 7·mid + 4) >> 3`,
/// `c = (7·mid + right + 4) >> 3`, `d = (5·mid + 3·right + 4) >> 3`. The u16
/// accumulator cannot overflow (`8·255 + 4 = 2044`) and each result narrows
/// exactly (`≤ 255`).
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
fn blend4_u8x16(
  left: uint8x16_t,
  mid: uint8x16_t,
  right: uint8x16_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
  let four = vdupq_n_u16(4);
  let left_lo = vmovl_u8(vget_low_u8(left));
  let left_hi = vmovl_high_u8(left);
  let mid_lo = vmovl_u8(vget_low_u8(mid));
  let mid_hi = vmovl_high_u8(mid);
  let right_lo = vmovl_u8(vget_low_u8(right));
  let right_hi = vmovl_high_u8(right);
  let phase = |l: uint16x8_t, wl: u16, m: uint16x8_t, wm: u16, r: uint16x8_t, wr: u16| {
    let acc = vaddq_u16(
      vaddq_u16(
        vaddq_u16(vmulq_n_u16(l, wl), vmulq_n_u16(m, wm)),
        vmulq_n_u16(r, wr),
      ),
      four,
    );
    vshrq_n_u16::<3>(acc)
  };
  let a = vcombine_u8(
    vmovn_u16(phase(left_lo, 3, mid_lo, 5, right_lo, 0)),
    vmovn_u16(phase(left_hi, 3, mid_hi, 5, right_hi, 0)),
  );
  let b = vcombine_u8(
    vmovn_u16(phase(left_lo, 1, mid_lo, 7, right_lo, 0)),
    vmovn_u16(phase(left_hi, 1, mid_hi, 7, right_hi, 0)),
  );
  let c = vcombine_u8(
    vmovn_u16(phase(left_lo, 0, mid_lo, 7, right_lo, 1)),
    vmovn_u16(phase(left_hi, 0, mid_hi, 7, right_hi, 1)),
  );
  let d = vcombine_u8(
    vmovn_u16(phase(left_lo, 0, mid_lo, 5, right_lo, 3)),
    vmovn_u16(phase(left_hi, 0, mid_hi, 5, right_hi, 3)),
  );
  (a, b, c, d)
}

/// NEON u8 centered 1→4 horizontal chroma upsample — the SIMD twin of
/// [`chroma_upsample_4to1_center_h`](crate::row::scalar::chroma_upsample_4to1_center_h).
/// Each quarter-width sample expands to four output columns
/// `{(3,5),(1,7),(7,1),(5,3)}/8`-weighted blends of the two nearest samples,
/// stored interleaved via `vst4q_u8`. The two boundary groups (`j = 0`,
/// `j = quarter-1`) and the trailing partial group reuse the shared scalar
/// per-group reference so the edges stay byte-identical; the vector loop covers
/// the strict interior (real neighbours, four full in-width columns).
///
/// Block size: 16 quarter samples / iter (→ 64 output columns).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
/// `c_quarter.len() >= width.div_ceil(4)`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chroma_upsample_4to1_center_h_row(
  c_quarter: &[u8],
  c_full: &mut [u8],
  width: usize,
) {
  debug_assert!(
    c_quarter.len() >= width.div_ceil(4),
    "c_quarter row too short"
  );
  debug_assert!(c_full.len() >= width, "c_full row too short");

  let quarter = width.div_ceil(4);
  if quarter == 0 {
    return;
  }
  scalar::chroma_upsample_4to1_center_h_group(c_quarter, c_full, 0, quarter, width);
  if quarter == 1 {
    return;
  }

  let mut j = 1usize;
  // SAFETY: `j + 16 < quarter` keeps `left = c[j-1]`, `mid = c[j]`,
  // `right = c[j+1]` loads inside `c_quarter[0..quarter]` and the 64-byte `vst4`
  // store inside `c_full[0..width]` — interior groups have real neighbours and
  // four full in-width columns (`4·(quarter-2)+3 < width`).
  unsafe {
    while j + 16 < quarter {
      let left = vld1q_u8(c_quarter.as_ptr().add(j - 1));
      let mid = vld1q_u8(c_quarter.as_ptr().add(j));
      let right = vld1q_u8(c_quarter.as_ptr().add(j + 1));
      let (a, b, c, d) = blend4_u8x16(left, mid, right);
      vst4q_u8(c_full.as_mut_ptr().add(4 * j), uint8x16x4_t(a, b, c, d));
      j += 16;
    }
  }

  while j < quarter {
    scalar::chroma_upsample_4to1_center_h_group(c_quarter, c_full, j, quarter, width);
    j += 1;
  }
}

#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
mod tests {
  use crate::row::scalar;

  fn pseudo_random_u16(out: &mut [u16], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 8) as u16;
    }
  }

  // Even widths: fully-scalar (half < block), block boundaries, and
  // non-multiples of the vector width with a scalar tail.
  const WIDTHS: &[usize] = &[2, 4, 6, 8, 16, 18, 30, 32, 34, 62, 64, 66, 128, 130];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_u8_matches_scalar_widths() {
    for &w in WIDTHS {
      let half = w / 2;
      let mut c_half = std::vec![0u8; half];
      let mut state = 0xC0FFEEu32;
      for v in c_half.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (state >> 16) as u8;
      }
      let mut out_simd = std::vec![0u8; w];
      let mut out_scalar = std::vec![0u8; w];
      unsafe { super::chroma_upsample_2to1_center_h_row(&c_half, &mut out_simd, w) };
      scalar::chroma_upsample_2to1_center_h(&c_half, &mut out_scalar, w);
      assert_eq!(out_simd, out_scalar, "u8 width={w}");
    }
  }

  fn check_u16<const BITS: u32>(big_endian: bool) {
    for &w in WIDTHS {
      let half = w / 2;
      let mut c_half = std::vec![0u16; half];
      pseudo_random_u16(&mut c_half, 0x1234 ^ BITS ^ (big_endian as u32));
      let mut out_simd = std::vec![0u16; w];
      let mut out_scalar = std::vec![0u16; w];
      unsafe {
        super::chroma_upsample_2to1_center_h_u16_row::<BITS>(&c_half, &mut out_simd, w, big_endian)
      };
      scalar::chroma_upsample_2to1_center_h_u16::<BITS>(&c_half, &mut out_scalar, w, big_endian);
      assert_eq!(
        out_simd, out_scalar,
        "u16 BITS={BITS} be={big_endian} width={w}"
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_u16_matches_scalar_widths() {
    check_u16::<10>(false);
    check_u16::<10>(true);
    check_u16::<12>(false);
    check_u16::<12>(true);
    check_u16::<16>(false);
    check_u16::<16>(true);
  }

  fn check_p0xx<const BITS: u32, const LOW_PACKED: bool>(big_endian: bool) {
    for &w in WIDTHS {
      let mut uv_half = std::vec![0u16; w];
      pseudo_random_u16(
        &mut uv_half,
        0x9E37 ^ BITS ^ ((LOW_PACKED as u32) << 8) ^ (big_endian as u32),
      );
      let mut out_simd = std::vec![0u16; 2 * w];
      let mut out_scalar = std::vec![0u16; 2 * w];
      unsafe {
        super::chroma_upsample_2to1_center_h_p0xx_row::<BITS, LOW_PACKED>(
          &uv_half,
          &mut out_simd,
          w,
          big_endian,
        )
      };
      scalar::chroma_upsample_2to1_center_h_p0xx::<BITS, LOW_PACKED>(
        &uv_half,
        &mut out_scalar,
        w,
        big_endian,
      );
      assert_eq!(
        out_simd, out_scalar,
        "p0xx BITS={BITS} low_packed={LOW_PACKED} be={big_endian} width={w}"
      );
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_p0xx_matches_scalar_widths() {
    check_p0xx::<10, false>(false);
    check_p0xx::<10, false>(true);
    check_p0xx::<10, true>(false);
    check_p0xx::<10, true>(true);
    check_p0xx::<12, false>(false);
    check_p0xx::<16, false>(false);
    check_p0xx::<16, false>(true);
  }

  fn pseudo_random_u8(out: &mut [u8], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 16) as u8;
    }
  }

  // Both the top-edge (prev == cur, box-blend clamps to the current row) and the
  // interior (distinct prev / cur) vertical cases are exercised per width.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_u8_vertical_matches_scalar_widths() {
    for &w in WIDTHS {
      let half = w / 2;
      let mut prev = std::vec![0u8; half];
      let mut cur = std::vec![0u8; half];
      pseudo_random_u8(&mut prev, 0xBEEF);
      pseudo_random_u8(&mut cur, 0xFACE);
      for (prev_row, tag) in [(prev.as_slice(), "interior"), (cur.as_slice(), "topedge")] {
        let mut bot_simd = std::vec![0u8; w];
        let mut bot_scalar = std::vec![0u8; w];
        let mut bl_simd = std::vec![0u8; w];
        let mut bl_scalar = std::vec![0u8; w];
        unsafe {
          super::chroma_upsample_420_bottom_even_h_row(prev_row, &cur, &mut bot_simd, w);
          super::chroma_upsample_420_bottomleft_even_h_row(prev_row, &cur, &mut bl_simd, w);
        }
        scalar::chroma_upsample_420_bottom_even_h(prev_row, &cur, &mut bot_scalar, w);
        scalar::chroma_upsample_420_bottomleft_even_h(prev_row, &cur, &mut bl_scalar, w);
        assert_eq!(bot_simd, bot_scalar, "u8 bottom {tag} width={w}");
        assert_eq!(bl_simd, bl_scalar, "u8 bottomleft {tag} width={w}");
      }
    }
  }

  fn check_u16_vertical<const BITS: u32>(big_endian: bool) {
    for &w in WIDTHS {
      let half = w / 2;
      let mut prev = std::vec![0u16; half];
      let mut cur = std::vec![0u16; half];
      pseudo_random_u16(&mut prev, 0x51A5 ^ BITS ^ (big_endian as u32));
      pseudo_random_u16(&mut cur, 0xC0DE ^ BITS ^ (big_endian as u32));
      for (prev_row, tag) in [(prev.as_slice(), "interior"), (cur.as_slice(), "topedge")] {
        let mut bot_simd = std::vec![0u16; w];
        let mut bot_scalar = std::vec![0u16; w];
        let mut bl_simd = std::vec![0u16; w];
        let mut bl_scalar = std::vec![0u16; w];
        unsafe {
          super::chroma_upsample_420_bottom_even_h_u16_row::<BITS>(
            prev_row,
            &cur,
            &mut bot_simd,
            w,
            big_endian,
          );
          super::chroma_upsample_420_bottomleft_even_h_u16_row::<BITS>(
            prev_row,
            &cur,
            &mut bl_simd,
            w,
            big_endian,
          );
        }
        scalar::chroma_upsample_420_bottom_even_h_u16::<BITS>(
          prev_row,
          &cur,
          &mut bot_scalar,
          w,
          big_endian,
        );
        scalar::chroma_upsample_420_bottomleft_even_h_u16::<BITS>(
          prev_row,
          &cur,
          &mut bl_scalar,
          w,
          big_endian,
        );
        assert_eq!(
          bot_simd, bot_scalar,
          "u16 bottom BITS={BITS} be={big_endian} {tag} width={w}"
        );
        assert_eq!(
          bl_simd, bl_scalar,
          "u16 bottomleft BITS={BITS} be={big_endian} {tag} width={w}"
        );
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_u16_vertical_matches_scalar_widths() {
    check_u16_vertical::<10>(false);
    check_u16_vertical::<10>(true);
    check_u16_vertical::<12>(false);
    check_u16_vertical::<12>(true);
    check_u16_vertical::<16>(false);
    check_u16_vertical::<16>(true);
  }

  fn check_p0xx_vertical<const BITS: u32>(big_endian: bool) {
    for &w in WIDTHS {
      let mut prev = std::vec![0u16; w];
      let mut cur = std::vec![0u16; w];
      pseudo_random_u16(&mut prev, 0x7E57 ^ BITS ^ (big_endian as u32));
      pseudo_random_u16(&mut cur, 0xABCD ^ BITS ^ (big_endian as u32));
      for (prev_row, tag) in [(prev.as_slice(), "interior"), (cur.as_slice(), "topedge")] {
        let mut bot_simd = std::vec![0u16; 2 * w];
        let mut bot_scalar = std::vec![0u16; 2 * w];
        let mut bl_simd = std::vec![0u16; 2 * w];
        let mut bl_scalar = std::vec![0u16; 2 * w];
        unsafe {
          super::chroma_upsample_420_bottom_even_h_p0xx_row::<BITS>(
            prev_row,
            &cur,
            &mut bot_simd,
            w,
            big_endian,
          );
          super::chroma_upsample_420_bottomleft_even_h_p0xx_row::<BITS>(
            prev_row,
            &cur,
            &mut bl_simd,
            w,
            big_endian,
          );
        }
        scalar::chroma_upsample_420_bottom_even_h_p0xx::<BITS>(
          prev_row,
          &cur,
          &mut bot_scalar,
          w,
          big_endian,
        );
        scalar::chroma_upsample_420_bottomleft_even_h_p0xx::<BITS>(
          prev_row,
          &cur,
          &mut bl_scalar,
          w,
          big_endian,
        );
        assert_eq!(
          bot_simd, bot_scalar,
          "p0xx bottom BITS={BITS} be={big_endian} {tag} width={w}"
        );
        assert_eq!(
          bl_simd, bl_scalar,
          "p0xx bottomleft BITS={BITS} be={big_endian} {tag} width={w}"
        );
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_p0xx_vertical_matches_scalar_widths() {
    check_p0xx_vertical::<10>(false);
    check_p0xx_vertical::<10>(true);
    check_p0xx_vertical::<12>(false);
    check_p0xx_vertical::<12>(true);
    check_p0xx_vertical::<16>(false);
    check_p0xx_vertical::<16>(true);
  }

  // 4:4:0 full-width vertical average + 1→4 centered horizontal upsample.
  // Any-parity widths (4:4:0 is full-horizontal; 4:1:x permits non-multiple-of-4)
  // straddling every SIMD chunk boundary + scalar tail, plus the minimum, odd,
  // and partial-last-group cases.
  const WIDTHS_440: &[usize] = &[
    1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 129, 1920,
  ];
  const WIDTHS_4TO1: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 16, 20, 63, 64, 65, 66, 67, 68, 69, 72, 73, 128, 129, 130, 131, 260,
  ];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_440_bottom_v_matches_scalar_widths() {
    for &w in WIDTHS_440 {
      let mut prev = std::vec![0u8; w];
      let mut cur = std::vec![0u8; w];
      pseudo_random_u8(&mut prev, 0x440B);
      pseudo_random_u8(&mut cur, 0x440C);
      // Interior (distinct prev / cur) and top-edge (prev == cur → identity).
      for (prev_row, tag) in [(prev.as_slice(), "interior"), (cur.as_slice(), "topedge")] {
        let mut simd = std::vec![0u8; w];
        let mut sc = std::vec![0u8; w];
        unsafe { super::chroma_upsample_440_bottom_v_row(prev_row, &cur, &mut simd, w) };
        scalar::chroma_upsample_440_bottom_v(prev_row, &cur, &mut sc, w);
        assert_eq!(simd, sc, "u8 440 {tag} width={w}");
      }
    }
  }

  fn check_440_u16<const BITS: u32>(big_endian: bool) {
    for &w in WIDTHS_440 {
      let mut prev = std::vec![0u16; w];
      let mut cur = std::vec![0u16; w];
      pseudo_random_u16(&mut prev, 0x4416 ^ BITS ^ (big_endian as u32));
      pseudo_random_u16(&mut cur, 0x4417 ^ BITS ^ (big_endian as u32));
      for (prev_row, tag) in [(prev.as_slice(), "interior"), (cur.as_slice(), "topedge")] {
        let mut simd = std::vec![0u16; w];
        let mut sc = std::vec![0u16; w];
        unsafe {
          super::chroma_upsample_440_bottom_v_u16_row::<BITS>(
            prev_row, &cur, &mut simd, w, big_endian,
          )
        };
        scalar::chroma_upsample_440_bottom_v_u16_wire::<BITS>(
          prev_row, &cur, &mut sc, w, big_endian,
        );
        assert_eq!(
          simd, sc,
          "u16 440 BITS={BITS} be={big_endian} {tag} width={w}"
        );
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_440_bottom_v_u16_matches_scalar_widths() {
    check_440_u16::<10>(false);
    check_440_u16::<10>(true);
    check_440_u16::<12>(false);
    check_440_u16::<12>(true);
    check_440_u16::<16>(false);
    check_440_u16::<16>(true);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_4to1_center_h_matches_scalar_widths() {
    for &w in WIDTHS_4TO1 {
      let quarter = w.div_ceil(4);
      let mut cq = std::vec![0u8; quarter];
      pseudo_random_u8(&mut cq, 0x4701 ^ w as u32);
      let mut simd = std::vec![0u8; w];
      let mut sc = std::vec![0u8; w];
      unsafe { super::chroma_upsample_4to1_center_h_row(&cq, &mut simd, w) };
      scalar::chroma_upsample_4to1_center_h(&cq, &mut sc, w);
      assert_eq!(simd, sc, "u8 4to1 width={w}");
    }
  }
}
