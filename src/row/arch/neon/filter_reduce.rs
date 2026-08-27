//! NEON separable-filter H/V passes: per output span, a dot product of
//! **signed** `f32` coefficients over a zero-padded coefficient arena.
//!
//! The signed twin of [`area_reduce`](super::area_reduce). The arena
//! (built once per plan by the resample engine) pads every span's `f32`
//! coefficients to a multiple of 8 with zeros, so the hot loop is pure
//! wide loads: 8 source samples and 8 coefficients per iteration. A zero
//! coefficient annihilates its sample, so taps past a span's last real
//! coefficient contribute nothing — sample loads only stage through a
//! stack copy at the row-end boundary.
//!
//! Both passes accumulate in `f64` (the coefficients widen to `f64`; an
//! integer sample widens losslessly and an `f32` sample widens exactly,
//! so every per-lane product is exact). The only departure from the
//! scalar reference is the order the H-pass taps are summed — float
//! addition does not associate — so H parity is a small tolerance, not
//! bit-exactness; downstream the integer streams land within `±1` LSB of
//! Pillow. The V-pass is element-wise (mul+add, **not** fma) so it stays
//! bit-equal to the scalar reference.

#![cfg_attr(
  any(not(feature = "std"), not(any(feature = "rgb", feature = "gray"))),
  allow(dead_code)
)]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use super::miri_compat::*;
use crate::row::scalar::filter_reduce::padding_keep_mask8;

/// Eight source samples widened to four `f64` lane-pairs
/// `(lanes 0-1, 2-3, 4-5, 6-7)`.
type F64x8 = (float64x2_t, float64x2_t, float64x2_t, float64x2_t);

/// One element type the filter H-pass widens to `f64` lanes. The c1/c3
/// kernel bodies are generic over how 8 source samples (contiguous for
/// c1, channel-strided for c3) load and widen, so the signed-coefficient
/// accumulation is written once; the public entry points below pin it to
/// `u8` / `u16` / `f32`, mirroring the area engine's per-element kernels.
///
/// # Safety
///
/// Implementors run inside a `#[target_feature(enable = "neon")]` caller.
trait NeonElem: Copy + Default {
  /// Widen the 8 contiguous samples at `row[base..base + 8]`.
  ///
  /// # Safety
  ///
  /// `base + 8 <= row.len()`; NEON available.
  unsafe fn load8(row: &[Self], base: usize) -> F64x8;

  /// Widen the 8 channel-`ch` samples of an 8-pixel interleaved group at
  /// cell `cell` (samples at `(cell + t) * 3 + ch`, `t = 0..8`).
  ///
  /// # Safety
  ///
  /// `(cell + 8) * 3 <= row.len()`; `ch < 3`; NEON available.
  unsafe fn load8_c3(row: &[Self], cell: usize, ch: usize) -> F64x8;
}

/// Widens an `f32x4` to two `f64` pairs `(lanes 0-1, lanes 2-3)`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn widen_f32x4(s: float32x4_t) -> (float64x2_t, float64x2_t) {
  (vcvt_f64_f32(vget_low_f32(s)), vcvt_high_f64_f32(s))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn widen_u16x8(s16: uint16x8_t) -> F64x8 {
  // SAFETY: register-only conversion of 8 u16 lanes to four f64 pairs.
  unsafe {
    let s_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(s16)));
    let s_hi = vcvtq_f32_u32(vmovl_high_u16(s16));
    let (a, b) = widen_f32x4(s_lo);
    let (c, d) = widen_f32x4(s_hi);
    (a, b, c, d)
  }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn widen_f32x8(lo: float32x4_t, hi: float32x4_t) -> F64x8 {
  // SAFETY: register-only.
  unsafe {
    let (a, b) = widen_f32x4(lo);
    let (c, d) = widen_f32x4(hi);
    (a, b, c, d)
  }
}

impl NeonElem for u8 {
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn load8(row: &[u8], base: usize) -> F64x8 {
    // SAFETY: caller guarantees `base + 8 <= row.len()`.
    unsafe { widen_u16x8(vmovl_u8(vld1_u8(row.as_ptr().add(base)))) }
  }
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn load8_c3(row: &[u8], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: caller guarantees `(cell + 8) * 3 <= row.len()`; `vld3_u8`
    // deinterleaves 8 RGB pixels, `.ch` selects the channel's 8 u8.
    unsafe {
      let px = vld3_u8(row.as_ptr().add(cell * 3));
      let lane = match ch {
        0 => px.0,
        1 => px.1,
        _ => px.2,
      };
      widen_u16x8(vmovl_u8(lane))
    }
  }
}

impl NeonElem for u16 {
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn load8(row: &[u16], base: usize) -> F64x8 {
    // SAFETY: caller guarantees `base + 8 <= row.len()`.
    unsafe { widen_u16x8(vld1q_u16(row.as_ptr().add(base))) }
  }
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn load8_c3(row: &[u16], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: caller guarantees `(cell + 8) * 3 <= row.len()`.
    unsafe {
      let px = vld3q_u16(row.as_ptr().add(cell * 3));
      let lane = match ch {
        0 => px.0,
        1 => px.1,
        _ => px.2,
      };
      widen_u16x8(lane)
    }
  }
}

impl NeonElem for f32 {
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn load8(row: &[f32], base: usize) -> F64x8 {
    // SAFETY: caller guarantees `base + 8 <= row.len()`.
    unsafe {
      widen_f32x8(
        vld1q_f32(row.as_ptr().add(base)),
        vld1q_f32(row.as_ptr().add(base + 4)),
      )
    }
  }
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn load8_c3(row: &[f32], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: caller guarantees `(cell + 8) * 3 <= row.len()`; two
    // `vld3q_f32` cover the 8 pixels (4 each), `.ch` selects the channel.
    unsafe {
      let lo = vld3q_f32(row.as_ptr().add(cell * 3));
      let hi = vld3q_f32(row.as_ptr().add((cell + 4) * 3));
      let (l, h) = match ch {
        0 => (lo.0, hi.0),
        1 => (lo.1, hi.1),
        _ => (lo.2, hi.2),
      };
      widen_f32x8(l, h)
    }
  }
}

/// Loads + widens the 8 contiguous samples at cell `base`, staging
/// through a zero-filled `S` buffer when the direct load would cross the
/// row end.
///
/// # Safety
///
/// `base < cells`; NEON available; `row.len() >= cells`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load8_staged_c1<S: NeonElem>(row: &[S], base: usize) -> F64x8 {
  // SAFETY: a full chunk loads directly; a row-end chunk stages through a
  // zero-filled 8-element copy so the load never reads past the slice.
  unsafe {
    if base + 8 <= row.len() {
      S::load8(row, base)
    } else {
      let mut sbuf = [S::default(); 8];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      S::load8(&sbuf, 0)
    }
  }
}

/// Loads + widens channel `ch` of the 8-pixel interleaved group at cell
/// `cell`, staging through a zero-filled `S` buffer at the row end.
///
/// # Safety
///
/// `cell < cells`; `ch < 3`; NEON available; `row.len() >= cells * 3`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load8_staged_c3<S: NeonElem>(row: &[S], cell: usize, ch: usize) -> F64x8 {
  // SAFETY: a full group loads directly; a row-end group stages its 24
  // interleaved samples through a zero-filled copy.
  unsafe {
    if (cell + 8) * 3 <= row.len() {
      S::load8_c3(row, cell, ch)
    } else {
      let mut sbuf = [S::default(); 24];
      let take = row.len() - cell * 3;
      sbuf[..take].copy_from_slice(&row[cell * 3..]);
      S::load8_c3(&sbuf, 0, ch)
    }
  }
}

/// Widens eight signed `f32` arena coefficients to four `f64` lane-pairs
/// `(c0c1, c2c3, c4c5, c6c7)`.
///
/// # Safety
///
/// NEON available; `c.len() >= 8` (the arena pads spans to a multiple of
/// 8).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn widen_coeffs(c: &[f32]) -> F64x8 {
  // SAFETY: caller passes an 8-multiple arena chunk.
  unsafe { widen_f32x8(vld1q_f32(c.as_ptr()), vld1q_f32(c.as_ptr().add(4))) }
}

/// Accumulates eight widened samples against four widened coefficient
/// pairs into a running `f64` lane-pair, every lane real. A separate
/// multiply then add (not fma): the per-lane product is exact in `f64`,
/// so the two forms agree and the H-pass tolerance comes only from the
/// cross-lane sum order. A real **zero** coefficient multiplies its
/// (possibly non-finite) sample as is, matching the scalar reference's
/// `0.0 * NaN == NaN` — only a padding lane (handled by [`mac8_masked`])
/// is forced to `+0.0`.
///
/// # Safety
///
/// NEON available.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn mac8_full(acc: float64x2_t, s: F64x8, c: F64x8) -> float64x2_t {
  let a = vaddq_f64(acc, vmulq_f64(s.0, c.0));
  let a = vaddq_f64(a, vmulq_f64(s.1, c.1));
  let a = vaddq_f64(a, vmulq_f64(s.2, c.2));
  vaddq_f64(a, vmulq_f64(s.3, c.3))
}

/// As [`mac8_full`] for a span's trailing partial chunk: `keep` is the
/// per-lane padding mask ([`padding_keep_mask8`]) — real lanes pass their
/// sample through (so a real zero coefficient still annihilates exactly as
/// scalar), padding lanes are cleared to `+0.0` so a non-finite
/// out-of-window sample cannot poison the span through `0.0 * NaN`.
///
/// # Safety
///
/// NEON available.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn mac8_masked(acc: float64x2_t, s: F64x8, c: F64x8, keep: &[f64; 8]) -> float64x2_t {
  // SAFETY: `keep` is a fixed 8-element array; the four pair loads are in
  // bounds.
  unsafe {
    let k0 = vld1q_f64(keep.as_ptr());
    let k1 = vld1q_f64(keep.as_ptr().add(2));
    let k2 = vld1q_f64(keep.as_ptr().add(4));
    let k3 = vld1q_f64(keep.as_ptr().add(6));
    let m0 = mask_lane(s.0, k0);
    let m1 = mask_lane(s.1, k1);
    let m2 = mask_lane(s.2, k2);
    let m3 = mask_lane(s.3, k3);
    let a = vaddq_f64(acc, vmulq_f64(m0, c.0));
    let a = vaddq_f64(a, vmulq_f64(m1, c.1));
    let a = vaddq_f64(a, vmulq_f64(m2, c.2));
    vaddq_f64(a, vmulq_f64(m3, c.3))
  }
}

/// Bitwise-ANDs a sample lane-pair with a keep-mask lane-pair (all-ones to
/// keep, `+0.0` to clear).
#[inline]
#[target_feature(enable = "neon")]
fn mask_lane(sf: float64x2_t, keep: float64x2_t) -> float64x2_t {
  vreinterpretq_f64_u64(vandq_u64(
    vreinterpretq_u64_f64(sf),
    vreinterpretq_u64_f64(keep),
  ))
}

/// Filter H-pass (1 channel): `h_tmp[j] = Σ coeffs[k] * row[start_j + k]`
/// in `f64` over the signed `f32` coefficient arena.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). Caller guarantees:
/// `coff.len() == starts.len() + 1` with monotonic entries bounded by
/// `coeffs.len()`, every span's padded length a multiple of 8,
/// `row.len() >= cells`, and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn h_reduce_c1<S: NeonElem>(
  row: &[S],
  starts: &[usize],
  ksize: &[usize],
  coeffs: &[f32],
  coff: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &coeffs[coff[j]..coff[j + 1]];
    let real = ksize[j];
    // SAFETY: each 8-sample load is either fully in-bounds or staged
    // through a zero-filled stack copy; coefficient loads come from the
    // 8-multiple arena slice.
    unsafe {
      let mut acc = vdupq_n_f64(0.0);
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let base = start + ci * 8;
        let s = load8_staged_c1(row, base);
        let c = widen_coeffs(chunk);
        let real_in_chunk = real.saturating_sub(ci * 8).min(8);
        acc = if real_in_chunk == 8 {
          mac8_full(acc, s, c)
        } else {
          mac8_masked(acc, s, c, &padding_keep_mask8(real_in_chunk))
        };
      }
      h_tmp[j] = vaddvq_f64_compat(acc);
    }
  }
}

/// Filter H-pass (3-channel interleaved): the signed `f32` coefficients
/// are shared across the three channels; each channel keeps its own `f64`
/// accumulator and deinterleaves through `vld3`.
///
/// # Safety
///
/// As [`filter_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn h_reduce_c3<S: NeonElem>(
  row: &[S],
  starts: &[usize],
  ksize: &[usize],
  coeffs: &[f32],
  coff: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &coeffs[coff[j]..coff[j + 1]];
    let real = ksize[j];
    // SAFETY: each 8-pixel group deinterleaves fully in-bounds or stages
    // through a zero-filled 24-sample copy; coefficients come from the
    // 8-multiple arena slice.
    unsafe {
      let mut acc0 = vdupq_n_f64(0.0);
      let mut acc1 = vdupq_n_f64(0.0);
      let mut acc2 = vdupq_n_f64(0.0);
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let cell = start + ci * 8;
        let c = widen_coeffs(chunk);
        let real_in_chunk = real.saturating_sub(ci * 8).min(8);
        if real_in_chunk == 8 {
          acc0 = mac8_full(acc0, load8_staged_c3(row, cell, 0), c);
          acc1 = mac8_full(acc1, load8_staged_c3(row, cell, 1), c);
          acc2 = mac8_full(acc2, load8_staged_c3(row, cell, 2), c);
        } else {
          let keep = padding_keep_mask8(real_in_chunk);
          acc0 = mac8_masked(acc0, load8_staged_c3(row, cell, 0), c, &keep);
          acc1 = mac8_masked(acc1, load8_staged_c3(row, cell, 1), c, &keep);
          acc2 = mac8_masked(acc2, load8_staged_c3(row, cell, 2), c, &keep);
        }
      }
      h_tmp[j * 3] = vaddvq_f64_compat(acc0);
      h_tmp[j * 3 + 1] = vaddvq_f64_compat(acc1);
      h_tmp[j * 3 + 2] = vaddvq_f64_compat(acc2);
    }
  }
}

// ---- Concrete per-element entry points (the dispatcher's targets) -----
//
// Each pins the generic c1/c3 kernel to one element type, mirroring the
// area engine's `area_h_reduce_row{,_u16,_f32}` split so the dispatcher
// names a concrete function per element / channel count.

macro_rules! neon_h_entry {
  ($c1:ident, $c3:ident, $elem:ty, $doc:literal) => {
    #[doc = $doc]
    ///
    /// # Safety
    ///
    /// See [`h_reduce_c1`]: NEON available; the arena binds to this row.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn $c1(
      row: &[$elem],
      starts: &[usize],
      ksize: &[usize],
      coeffs: &[f32],
      coff: &[usize],
      h_tmp: &mut [f64],
    ) {
      // SAFETY: forwarded to the generic c1 kernel under the caller's
      // arena guarantees.
      unsafe { h_reduce_c1::<$elem>(row, starts, ksize, coeffs, coff, h_tmp) }
    }

    #[doc = $doc]
    ///
    /// # Safety
    ///
    /// See [`h_reduce_c3`]: NEON available; the arena binds to this row.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn $c3(
      row: &[$elem],
      starts: &[usize],
      ksize: &[usize],
      coeffs: &[f32],
      coff: &[usize],
      h_tmp: &mut [f64],
    ) {
      // SAFETY: forwarded to the generic c3 kernel under the caller's
      // arena guarantees.
      unsafe { h_reduce_c3::<$elem>(row, starts, ksize, coeffs, coff, h_tmp) }
    }
  };
}

neon_h_entry!(
  filter_h_reduce_row_u8_c1,
  filter_h_reduce_row_u8_c3,
  u8,
  "Filter H-pass over `u8` samples (1 / 3 channel)."
);
neon_h_entry!(
  filter_h_reduce_row_u16_c1,
  filter_h_reduce_row_u16_c3,
  u16,
  "Filter H-pass over `u16` samples (1 / 3 channel)."
);
neon_h_entry!(
  filter_h_reduce_row_f32_c1,
  filter_h_reduce_row_f32_c3,
  f32,
  "Filter H-pass over `f32` samples (1 / 3 channel)."
);

/// Filter V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64`. A separate
/// multiply then add (not fma) so each lane matches the scalar reference
/// bit-for-bit — the V-pass is element-wise, with no reordering. Two
/// elements per iteration.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64). `h_tmp.len() >=
/// acc.len()`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn filter_v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: f32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wq = vdupq_n_f64(f64::from(w));
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
    acc[k] += f64::from(w) * h_tmp[k];
  }
}
