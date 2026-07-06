//! NEON fused-downscale H-pass: per output span, widening
//! multiply-accumulate over a zero-padded u16 weight arena.
//!
//! The arena (built once per plan by the resample engine) pads every
//! span's weights to a multiple of 8 with zeros, so the hot loop is
//! pure wide loads: 8 source samples widen `u8 -> u16` and meet 8
//! arena weights in `vmull_u16`/`vmull_high_u16` exact u32 lanes.
//! Padding lanes multiply by zero, so samples past a span's last tap
//! contribute nothing — sample loads only stage through a stack copy
//! at the row-end boundary, where a direct 8-byte load would cross
//! the slice end. Bit-identical to the scalar reference by integer
//! associativity.

#![cfg_attr(
  any(
    not(feature = "std"),
    not(any(feature = "yuv-planar", feature = "rgb"))
  ),
  allow(dead_code)
)]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use super::miri_compat::*;

/// # Safety
///
/// NEON must be available (baseline on aarch64). Caller guarantees:
/// `w16_off.len() == starts.len() + 1` with monotonic entries bounded
/// by `w16.len()`, every span's padded length a multiple of 8,
/// `row.len() >= cells`, and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_h_reduce_row_c1(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 8-byte sample load is either fully in-bounds
    // (guarded against row.len()) or staged through a zero-filled
    // stack copy; weight loads come from the 8-multiple arena slice.
    unsafe {
      let mut acc = vdupq_n_u32(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let s8 = if base + 8 <= row.len() {
          vld1_u8(row.as_ptr().add(base))
        } else {
          let mut sbuf = [0u8; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          vld1_u8(sbuf.as_ptr())
        };
        let s16 = vmovl_u8(s8);
        let w = vld1q_u16(chunk.as_ptr());
        acc = vaddq_u32(acc, vmull_u16(vget_low_u16(s16), vget_low_u16(w)));
        acc = vaddq_u32(acc, vmull_high_u16(s16, w));
      }
      h_tmp[j] = vaddvq_u32_compat(acc);
    }
  }
}

/// 3-channel (interleaved RGB) variant: `vld3_u8` deinterleaves eight
/// pixels per iteration into per-channel lanes sharing one weight
/// vector.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_h_reduce_row_c3(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 24-byte vld3_u8 load is either fully in-bounds
    // (guarded against row.len()) or staged through a zero-filled
    // stack copy; weight loads come from the 8-multiple arena slice.
    unsafe {
      let mut acc0 = vdupq_n_u32(0);
      let mut acc1 = vdupq_n_u32(0);
      let mut acc2 = vdupq_n_u32(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let px = if (base + 8) * 3 <= row.len() {
          vld3_u8(row.as_ptr().add(base * 3))
        } else {
          let mut sbuf = [0u8; 24];
          let take = row.len() - base * 3;
          sbuf[..take].copy_from_slice(&row[base * 3..]);
          vld3_u8(sbuf.as_ptr())
        };
        let w = vld1q_u16(chunk.as_ptr());
        let wl = vget_low_u16(w);
        let s0 = vmovl_u8(px.0);
        acc0 = vaddq_u32(acc0, vmull_u16(vget_low_u16(s0), wl));
        acc0 = vaddq_u32(acc0, vmull_high_u16(s0, w));
        let s1 = vmovl_u8(px.1);
        acc1 = vaddq_u32(acc1, vmull_u16(vget_low_u16(s1), wl));
        acc1 = vaddq_u32(acc1, vmull_high_u16(s1, w));
        let s2 = vmovl_u8(px.2);
        acc2 = vaddq_u32(acc2, vmull_u16(vget_low_u16(s2), wl));
        acc2 = vaddq_u32(acc2, vmull_high_u16(s2, w));
      }
      h_tmp[j * 3] = vaddvq_u32_compat(acc0);
      h_tmp[j * 3 + 1] = vaddvq_u32_compat(acc1);
      h_tmp[j * 3 + 2] = vaddvq_u32_compat(acc2);
    }
  }
}

/// V-pass AXPY: `acc[i] += w * h_tmp[i]`, exact u64 lanes via
/// `vmull_u32`/`vmull_high_u32` (4 elements per iteration).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
/// `h_tmp.len() >= acc.len()`; every product-sum stays within u64
/// (the engine's denominator bound).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wq = vdupq_n_u32(w);
  let mut i = 0usize;
  // SAFETY: loop guard `i + 4 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 4 <= n {
      let t = vld1q_u32(h_tmp.as_ptr().add(i));
      let a0 = vld1q_u64(acc.as_ptr().add(i));
      let a1 = vld1q_u64(acc.as_ptr().add(i + 2));
      let p0 = vmull_u32(vget_low_u32(t), vget_low_u32(wq));
      let p1 = vmull_high_u32(t, wq);
      vst1q_u64(acc.as_mut_ptr().add(i), vaddq_u64(a0, p0));
      vst1q_u64(acc.as_mut_ptr().add(i + 2), vaddq_u64(a1, p1));
      i += 4;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * u64::from(h_tmp[k]);
  }
}

/// 16-bit-element H-pass (1 channel): like [`area_h_reduce_row_c1`] but
/// the samples load directly as `u16` and the `u16 * u16 -> u32`
/// products widen to `u64` before accumulating — a single span sum can
/// exceed `u32`, so the running total lives in `u64` lanes.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `u16`
/// elements and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_h_reduce_row_u16_c1(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 8-element u16 load is fully in-bounds (guarded) or
    // staged through a zero-filled stack copy; weights come from the
    // 8-multiple arena slice.
    unsafe {
      let mut acc = vdupq_n_u64(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let s16 = if base + 8 <= row.len() {
          vld1q_u16(row.as_ptr().add(base))
        } else {
          let mut sbuf = [0u16; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          vld1q_u16(sbuf.as_ptr())
        };
        let w = vld1q_u16(chunk.as_ptr());
        let p_lo = vmull_u16(vget_low_u16(s16), vget_low_u16(w));
        let p_hi = vmull_high_u16(s16, w);
        acc = vaddq_u64(acc, vmovl_u32(vget_low_u32(p_lo)));
        acc = vaddq_u64(acc, vmovl_high_u32(p_lo));
        acc = vaddq_u64(acc, vmovl_u32(vget_low_u32(p_hi)));
        acc = vaddq_u64(acc, vmovl_high_u32(p_hi));
      }
      h_tmp[j] = vaddvq_u64_compat(acc);
    }
  }
}

/// 16-bit-element H-pass (3-channel interleaved RGB): `vld3q_u16`
/// deinterleaves eight `u16` pixels per iteration into per-channel
/// lanes sharing one weight vector.
///
/// # Safety
///
/// As [`area_h_reduce_row_u16_c1`], with `row.len() >= cells * 3`
/// `u16` elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_h_reduce_row_u16_c3(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 24-element vld3q_u16 load is fully in-bounds
    // (guarded) or staged through a zero-filled stack copy; weights
    // come from the 8-multiple arena slice.
    unsafe {
      let mut acc0 = vdupq_n_u64(0);
      let mut acc1 = vdupq_n_u64(0);
      let mut acc2 = vdupq_n_u64(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let px = if (base + 8) * 3 <= row.len() {
          vld3q_u16(row.as_ptr().add(base * 3))
        } else {
          let mut sbuf = [0u16; 24];
          let take = row.len() - base * 3;
          sbuf[..take].copy_from_slice(&row[base * 3..]);
          vld3q_u16(sbuf.as_ptr())
        };
        let w = vld1q_u16(chunk.as_ptr());
        let wl = vget_low_u16(w);
        let p0l = vmull_u16(vget_low_u16(px.0), wl);
        let p0h = vmull_high_u16(px.0, w);
        acc0 = vaddq_u64(acc0, vmovl_u32(vget_low_u32(p0l)));
        acc0 = vaddq_u64(acc0, vmovl_high_u32(p0l));
        acc0 = vaddq_u64(acc0, vmovl_u32(vget_low_u32(p0h)));
        acc0 = vaddq_u64(acc0, vmovl_high_u32(p0h));
        let p1l = vmull_u16(vget_low_u16(px.1), wl);
        let p1h = vmull_high_u16(px.1, w);
        acc1 = vaddq_u64(acc1, vmovl_u32(vget_low_u32(p1l)));
        acc1 = vaddq_u64(acc1, vmovl_high_u32(p1l));
        acc1 = vaddq_u64(acc1, vmovl_u32(vget_low_u32(p1h)));
        acc1 = vaddq_u64(acc1, vmovl_high_u32(p1h));
        let p2l = vmull_u16(vget_low_u16(px.2), wl);
        let p2h = vmull_high_u16(px.2, w);
        acc2 = vaddq_u64(acc2, vmovl_u32(vget_low_u32(p2l)));
        acc2 = vaddq_u64(acc2, vmovl_high_u32(p2l));
        acc2 = vaddq_u64(acc2, vmovl_u32(vget_low_u32(p2h)));
        acc2 = vaddq_u64(acc2, vmovl_high_u32(p2h));
      }
      h_tmp[j * 3] = vaddvq_u64_compat(acc0);
      h_tmp[j * 3 + 1] = vaddvq_u64_compat(acc1);
      h_tmp[j * 3 + 2] = vaddvq_u64_compat(acc2);
    }
  }
}

/// 16-bit-element V-pass AXPY: `acc[i] += w * h_tmp[i]` with `h_tmp`
/// already `u64`. The `u32 * u64 -> u64` product has no single NEON
/// instruction, so it splits `h_tmp` into 32-bit halves: the low half
/// gives `w * lo` and the high half `(w * hi) << 32`, summed mod 2^64
/// (exact — the engine's denominator bound keeps every total in u64).
/// Two elements per iteration.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `h_tmp.len() >=
/// acc.len()`; every product-sum stays within u64.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_v_accumulate_u16(acc: &mut [u64], h_tmp: &[u64], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wd = vdup_n_u32(w);
  let mut i = 0usize;
  // SAFETY: loop guard `i + 2 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 2 <= n {
      let t = vld1q_u64(h_tmp.as_ptr().add(i));
      let a = vld1q_u64(acc.as_ptr().add(i));
      let t_lo = vmovn_u64(t);
      let t_hi = vshrn_n_u64(t, 32);
      let prod_lo = vmull_u32(t_lo, wd);
      let prod_hi = vmull_u32(t_hi, wd);
      let prod = vaddq_u64(prod_lo, vshlq_n_u64(prod_hi, 32));
      vst1q_u64(acc.as_mut_ptr().add(i), vaddq_u64(a, prod));
      i += 2;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * h_tmp[k];
  }
}

/// Float-element H-pass (1 channel): the samples are `f32` and the
/// per-span sums live in `f64` (see the scalar reference for why the
/// float accumulators are `f64`). Each `u16` arena weight widens to
/// `f64` and each `f32` sample widens to `f64`; the product is exact (a
/// `u16` weight times a 24-bit `f32` mantissa fits 53 bits), so the only
/// departure from the scalar reference is the order the taps are summed
/// — float addition does not associate, so parity is a small tolerance,
/// not the integer kernels' bit-exactness.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `f32`
/// elements and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_h_reduce_row_f32_c1(
  row: &[f32],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 8-element f32 load is fully in-bounds (guarded) or
    // staged through a zero-filled stack copy; weights come from the
    // 8-multiple arena slice.
    unsafe {
      let mut acc = vdupq_n_f64(0.0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let (s_lo, s_hi) = if base + 8 <= row.len() {
          (
            vld1q_f32(row.as_ptr().add(base)),
            vld1q_f32(row.as_ptr().add(base + 4)),
          )
        } else {
          let mut sbuf = [0f32; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          (vld1q_f32(sbuf.as_ptr()), vld1q_f32(sbuf.as_ptr().add(4)))
        };
        let w = vld1q_u16(chunk.as_ptr());
        let (wf0, wf1, wf2, wf3) = widen_w16_f64(w);
        acc = fma_8(acc, s_lo, s_hi, wf0, wf1, wf2, wf3);
      }
      h_tmp[j] = vaddvq_f64_compat(acc);
    }
  }
}

/// Float-element H-pass (3-channel interleaved RGB): `vld3q_f32`
/// deinterleaves four `f32` pixels per load into per-channel lanes; two
/// loads cover the eight-pixel chunk, sharing one widened weight set.
///
/// # Safety
///
/// As [`area_h_reduce_row_f32_c1`], with `row.len() >= cells * 3` `f32`
/// elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_h_reduce_row_f32_c3(
  row: &[f32],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each pair of 12-element vld3q_f32 loads is fully in-bounds
    // (guarded) or staged through a zero-filled stack copy; weights come
    // from the 8-multiple arena slice.
    unsafe {
      let mut acc0 = vdupq_n_f64(0.0);
      let mut acc1 = vdupq_n_f64(0.0);
      let mut acc2 = vdupq_n_f64(0.0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let (px_lo, px_hi) = if (base + 8) * 3 <= row.len() {
          (
            vld3q_f32(row.as_ptr().add(base * 3)),
            vld3q_f32(row.as_ptr().add((base + 4) * 3)),
          )
        } else {
          let mut sbuf = [0f32; 24];
          let take = row.len() - base * 3;
          sbuf[..take].copy_from_slice(&row[base * 3..]);
          (vld3q_f32(sbuf.as_ptr()), vld3q_f32(sbuf.as_ptr().add(12)))
        };
        let w = vld1q_u16(chunk.as_ptr());
        let (wf0, wf1, wf2, wf3) = widen_w16_f64(w);
        acc0 = fma_8(acc0, px_lo.0, px_hi.0, wf0, wf1, wf2, wf3);
        acc1 = fma_8(acc1, px_lo.1, px_hi.1, wf0, wf1, wf2, wf3);
        acc2 = fma_8(acc2, px_lo.2, px_hi.2, wf0, wf1, wf2, wf3);
      }
      h_tmp[j * 3] = vaddvq_f64_compat(acc0);
      h_tmp[j * 3 + 1] = vaddvq_f64_compat(acc1);
      h_tmp[j * 3 + 2] = vaddvq_f64_compat(acc2);
    }
  }
}

/// Widens eight `u16` arena weights to four `f64` lane-pairs
/// `(w0w1, w2w3, w4w5, w6w7)`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn widen_w16_f64(w: uint16x8_t) -> (float64x2_t, float64x2_t, float64x2_t, float64x2_t) {
  let w_lo = vmovl_u16(vget_low_u16(w));
  let w_hi = vmovl_high_u16(w);
  (
    vcvtq_f64_u64(vmovl_u32(vget_low_u32(w_lo))),
    vcvtq_f64_u64(vmovl_high_u32(w_lo)),
    vcvtq_f64_u64(vmovl_u32(vget_low_u32(w_hi))),
    vcvtq_f64_u64(vmovl_high_u32(w_hi)),
  )
}

/// Accumulates eight `f32` samples (`s_lo` lanes 0-3, `s_hi` lanes 4-7)
/// against four widened weight pairs into a running `f64` lane-pair.
/// Each widened sample is first zeroed in the lanes whose weight is zero
/// (arena padding), so a non-finite sample in a direct-loaded padding
/// lane cannot poison the span through `0.0 * NaN`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fma_8(
  acc: float64x2_t,
  s_lo: float32x4_t,
  s_hi: float32x4_t,
  wf0: float64x2_t,
  wf1: float64x2_t,
  wf2: float64x2_t,
  wf3: float64x2_t,
) -> float64x2_t {
  // The product of an integer weight and an f32 sample is exact in f64,
  // so a fused multiply-add matches a separate multiply-then-add here.
  let mut a = vfmaq_f64(
    acc,
    mask_pad_f64(vcvt_f64_f32(vget_low_f32(s_lo)), wf0),
    wf0,
  );
  a = vfmaq_f64(a, mask_pad_f64(vcvt_high_f64_f32(s_lo), wf1), wf1);
  a = vfmaq_f64(a, mask_pad_f64(vcvt_f64_f32(vget_low_f32(s_hi)), wf2), wf2);
  vfmaq_f64(a, mask_pad_f64(vcvt_high_f64_f32(s_hi), wf3), wf3)
}

/// Zeroes the `f64` sample lanes whose weight lane is zero — the arena's
/// padding lanes. The integer kernels lean on `0 * sample == 0`, but
/// `0.0 * NaN` and `0.0 * inf` are NaN, so a direct-loaded padding lane
/// holding a non-finite neighbor would otherwise poison the span.
#[inline]
#[target_feature(enable = "neon")]
fn mask_pad_f64(sf: float64x2_t, wf: float64x2_t) -> float64x2_t {
  // sf AND NOT(wf == 0): keep the sample where the weight is nonzero,
  // clear it to +0.0 where the weight is zero.
  vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(sf), vceqzq_f64(wf)))
}

/// Float-element V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64`. A
/// separate multiply-then-add (not a fused multiply-add) so each lane
/// matches the scalar reference bit-for-bit — the V-pass is
/// element-wise, so unlike the H-pass it has no reordering and stays
/// exactly equal. Two elements per iteration.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `h_tmp.len() >=
/// acc.len()`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_v_accumulate_f32(acc: &mut [f64], h_tmp: &[f64], w: f64) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wq = vdupq_n_f64(w);
  let mut i = 0usize;
  // SAFETY: loop guard `i + 2 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 2 <= n {
      let t = vld1q_f64(h_tmp.as_ptr().add(i));
      let a = vld1q_f64(acc.as_ptr().add(i));
      vst1q_f64(acc.as_mut_ptr().add(i), vaddq_f64(a, vmulq_f64(t, wq)));
      i += 2;
    }
  }
  for k in i..n {
    acc[k] += w * h_tmp[k];
  }
}
