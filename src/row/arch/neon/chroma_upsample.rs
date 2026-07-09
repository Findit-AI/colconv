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
}
