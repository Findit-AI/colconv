//! SSE4.1 separable-filter H/V passes: per output span, a dot product of
//! **signed** `f32` coefficients over a zero-padded coefficient arena.
//!
//! The signed twin of [`area_reduce`](super::area_reduce). The arena pads
//! every span's `f32` coefficients to a multiple of 8 with zeros, so the
//! hot loop runs pure wide loads (8 samples + 8 coefficients per chunk); a
//! zero coefficient annihilates its sample, so sample loads only stage
//! through a stack copy at the row-end boundary.
//!
//! Both passes accumulate in `f64` (coefficients widen to `f64`; an
//! integer sample widens losslessly, an `f32` sample exactly, so every
//! per-lane product is exact). H parity is a small tolerance — float
//! addition does not associate, so only the tap-sum order differs — while
//! the V-pass is element-wise (mul+add, **not** fma) and stays bit-equal
//! to the scalar reference. Mirrors the SSE4.1 area f32 path
//! (`widen`/`mask_pd`/`hsum_pd`/`deint3_f32`), with the integer c3 gathers
//! reused from the area u8/u16 c3 masks.

#![cfg_attr(not(feature = "std"), allow(dead_code))]
#![cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]

#[cfg(target_arch = "x86_64")]
#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::scalar::filter_reduce::padding_keep_mask8;

/// Eight source samples widened to four `f64` pairs.
type F64x8 = (__m128d, __m128d, __m128d, __m128d);

/// One element type the filter H-pass widens to `f64`. The c1/c3 kernel
/// bodies are generic over the load + widen; the public entry points pin
/// it to `u8` / `u16` / `f32`.
trait Sse41Elem: Copy + Default {
  /// Widen the 8 contiguous samples at `row[base..base + 8]`.
  ///
  /// # Safety
  ///
  /// `base + 8 <= row.len()`; SSE4.1 available.
  unsafe fn load8(row: &[Self], base: usize) -> F64x8;

  /// Widen channel `ch` of the 8-pixel interleaved group at cell `cell`.
  ///
  /// # Safety
  ///
  /// `(cell + 8) * 3 <= row.len()`; `ch < 3`; SSE4.1 available.
  unsafe fn load8_c3(row: &[Self], cell: usize, ch: usize) -> F64x8;
}

/// Widens an `f32x4` (`__m128`) to two `f64` pairs `(lanes 0-1, 2-3)`.
/// Register-only (matches the area path's safe `fn` convention), so a
/// `#[target_feature]`-enabled caller invokes it without an `unsafe` block.
#[inline]
#[target_feature(enable = "sse4.1")]
fn widen_ps(s: __m128) -> (__m128d, __m128d) {
  (_mm_cvtps_pd(s), _mm_cvtps_pd(_mm_movehl_ps(s, s)))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn widen_ps8(lo: __m128, hi: __m128) -> F64x8 {
  let (a, b) = widen_ps(lo);
  let (c, d) = widen_ps(hi);
  (a, b, c, d)
}

/// Widens 8 `u16` lanes (low 64 bits hold 4, etc.) to four `f64` pairs.
#[inline]
#[target_feature(enable = "sse4.1")]
fn widen_u16x8(s16: __m128i) -> F64x8 {
  let lo = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(s16));
  let hi = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(_mm_srli_si128::<8>(s16)));
  widen_ps8(lo, hi)
}

impl Sse41Elem for u8 {
  #[inline]
  #[target_feature(enable = "sse4.1")]
  unsafe fn load8(row: &[u8], base: usize) -> F64x8 {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe {
      widen_u16x8(_mm_cvtepu8_epi16(_mm_loadl_epi64(
        row.as_ptr().add(base).cast(),
      )))
    }
  }
  #[inline]
  #[target_feature(enable = "sse4.1")]
  unsafe fn load8_c3(row: &[u8], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: `(cell + 8) * 3 <= row.len()`; two 16-byte loads cover the
    // 24-byte group, the per-channel mask pair gathers 8 u8.
    unsafe {
      let base = cell * 3;
      let v0 = _mm_loadu_si128(row.as_ptr().add(base).cast());
      let v1 = _mm_loadu_si128(row.as_ptr().add(base + 8).cast());
      let g = gather_u8_c3(v0, v1, ch);
      widen_u16x8(_mm_cvtepu8_epi16(g))
    }
  }
}

impl Sse41Elem for u16 {
  #[inline]
  #[target_feature(enable = "sse4.1")]
  unsafe fn load8(row: &[u16], base: usize) -> F64x8 {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe { widen_u16x8(_mm_loadu_si128(row.as_ptr().add(base).cast())) }
  }
  #[inline]
  #[target_feature(enable = "sse4.1")]
  unsafe fn load8_c3(row: &[u16], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: `(cell + 8) * 3 <= row.len()`; three 16-byte loads cover the
    // 48-byte group, the per-channel mask triple gathers 8 u16.
    unsafe {
      let base = cell * 3;
      let v0 = _mm_loadu_si128(row.as_ptr().add(base).cast());
      let v1 = _mm_loadu_si128(row.as_ptr().add(base + 8).cast());
      let v2 = _mm_loadu_si128(row.as_ptr().add(base + 16).cast());
      widen_u16x8(gather_u16_c3(v0, v1, v2, ch))
    }
  }
}

impl Sse41Elem for f32 {
  #[inline]
  #[target_feature(enable = "sse4.1")]
  unsafe fn load8(row: &[f32], base: usize) -> F64x8 {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe {
      widen_ps8(
        _mm_loadu_ps(row.as_ptr().add(base)),
        _mm_loadu_ps(row.as_ptr().add(base + 4)),
      )
    }
  }
  #[inline]
  #[target_feature(enable = "sse4.1")]
  unsafe fn load8_c3(row: &[f32], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: `(cell + 8) * 3 <= row.len()`; two four-pixel deinterleaves
    // cover the 8-pixel group, `.ch` selects the channel.
    unsafe {
      let p = row.as_ptr().add(cell * 3);
      let (r0, g0, b0) = deint3_f32(
        _mm_loadu_ps(p),
        _mm_loadu_ps(p.add(4)),
        _mm_loadu_ps(p.add(8)),
      );
      let (r1, g1, b1) = deint3_f32(
        _mm_loadu_ps(p.add(12)),
        _mm_loadu_ps(p.add(16)),
        _mm_loadu_ps(p.add(20)),
      );
      let (lo, hi) = match ch {
        0 => (r0, r1),
        1 => (g0, g1),
        _ => (b0, b1),
      };
      widen_ps8(lo, hi)
    }
  }
}

/// Gathers channel `ch`'s 8 `u8` samples of a 24-byte RGB group split
/// across two overlapping 16-byte loads (area u8 c3 masks). Register-only.
#[inline]
#[target_feature(enable = "sse4.1")]
fn gather_u8_c3(v0: __m128i, v1: __m128i, ch: usize) -> __m128i {
  let (m0, m1) = match ch {
    0 => (
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0),
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, -1, -1, -1, -1, -1, -1,
      ),
    ),
    1 => (
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1),
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, -1, -1, -1, -1, -1,
      ),
    ),
    _ => (
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2),
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, -1, -1, -1, -1, -1,
      ),
    ),
  };
  _mm_or_si128(_mm_shuffle_epi8(v0, m0), _mm_shuffle_epi8(v1, m1))
}

/// Gathers channel `ch`'s 8 `u16` samples of a 48-byte RGB group split
/// across three overlapping 16-byte loads (area u16 c3 masks). Register-only.
#[inline]
#[target_feature(enable = "sse4.1")]
fn gather_u16_c3(v0: __m128i, v1: __m128i, v2: __m128i, ch: usize) -> __m128i {
  let (m0, m1, m2) = match ch {
    0 => (
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 7, 6, 1, 0),
      _mm_set_epi8(-1, -1, -1, -1, 15, 14, 9, 8, 3, 2, -1, -1, -1, -1, -1, -1),
      _mm_set_epi8(11, 10, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ),
    1 => (
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 14, 9, 8, 3, 2),
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, 11, 10, 5, 4, -1, -1, -1, -1, -1, -1),
      _mm_set_epi8(13, 12, 7, 6, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ),
    _ => (
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 10, 5, 4),
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 12, 7, 6, 1, 0, -1, -1, -1, -1),
      _mm_set_epi8(15, 14, 9, 8, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ),
  };
  _mm_or_si128(
    _mm_or_si128(_mm_shuffle_epi8(v0, m0), _mm_shuffle_epi8(v1, m1)),
    _mm_shuffle_epi8(v2, m2),
  )
}

/// Deinterleaves four interleaved RGB `f32` pixels into planar
/// `(R0..R3, G0..G3, B0..B3)` (area SSE4.1 `deint3_f32`). Register-only.
#[inline]
#[target_feature(enable = "sse4.1")]
fn deint3_f32(x: __m128, y: __m128, z: __m128) -> (__m128, __m128, __m128) {
  let yz_r = _mm_shuffle_ps::<0b00_01_00_10>(y, z);
  let r = _mm_shuffle_ps::<0b10_00_11_00>(x, yz_r);
  let xy_g = _mm_shuffle_ps::<0b00_00_00_01>(x, y);
  let yz_g = _mm_shuffle_ps::<0b00_10_00_11>(y, z);
  let g = _mm_shuffle_ps::<0b10_00_10_00>(xy_g, yz_g);
  let xy_b = _mm_shuffle_ps::<0b00_01_00_10>(x, y);
  let b = _mm_shuffle_ps::<0b11_00_10_00>(xy_b, z);
  (r, g, b)
}

/// Widens 8 signed `f32` coefficients to four `f64` pairs.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn widen_coeffs(c: &[f32]) -> F64x8 {
  // SAFETY: caller passes an 8-multiple arena chunk.
  unsafe { widen_ps8(_mm_loadu_ps(c.as_ptr()), _mm_loadu_ps(c.as_ptr().add(4))) }
}

/// Bitwise-ANDs a sample lane-pair with a keep-mask lane-pair (all-ones to
/// keep a real lane, `+0.0` to clear a padding lane). Register-only.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mask_lane(sf: __m128d, keep: __m128d) -> __m128d {
  _mm_and_pd(sf, keep)
}

/// Accumulates 8 widened samples against 4 widened coefficient pairs into
/// a running `f64` pair (mul+add; the product is exact in `f64`), every
/// lane real — so a real zero coefficient multiplies its (possibly
/// non-finite) sample as is, matching the scalar reference. Register-only.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mac8_full(acc: __m128d, s: F64x8, c: F64x8) -> __m128d {
  let a = _mm_add_pd(acc, _mm_mul_pd(s.0, c.0));
  let a = _mm_add_pd(a, _mm_mul_pd(s.1, c.1));
  let a = _mm_add_pd(a, _mm_mul_pd(s.2, c.2));
  _mm_add_pd(a, _mm_mul_pd(s.3, c.3))
}

/// As [`mac8_full`] for a span's trailing partial chunk: `keep` is the
/// per-lane padding mask ([`padding_keep_mask8`]) — padding lanes are
/// cleared to `+0.0` so a non-finite out-of-window sample cannot poison
/// the span through `0.0 * NaN`, while real lanes (including a real zero
/// coefficient) pass through.
///
/// # Safety
///
/// SSE4.1 available; `keep` is a fixed 8-element array.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn mac8_masked(acc: __m128d, s: F64x8, c: F64x8, keep: &[f64; 8]) -> __m128d {
  // SAFETY: the four pair loads stay within the 8-element `keep`.
  unsafe {
    let k0 = _mm_loadu_pd(keep.as_ptr());
    let k1 = _mm_loadu_pd(keep.as_ptr().add(2));
    let k2 = _mm_loadu_pd(keep.as_ptr().add(4));
    let k3 = _mm_loadu_pd(keep.as_ptr().add(6));
    let a = _mm_add_pd(acc, _mm_mul_pd(mask_lane(s.0, k0), c.0));
    let a = _mm_add_pd(a, _mm_mul_pd(mask_lane(s.1, k1), c.1));
    let a = _mm_add_pd(a, _mm_mul_pd(mask_lane(s.2, k2), c.2));
    _mm_add_pd(a, _mm_mul_pd(mask_lane(s.3, k3), c.3))
  }
}

/// Sums the two `f64` lanes. Register-only.
#[inline]
#[target_feature(enable = "sse4.1")]
fn hsum_pd(v: __m128d) -> f64 {
  _mm_cvtsd_f64(_mm_add_sd(v, _mm_unpackhi_pd(v, v)))
}

/// Loads + widens 8 contiguous samples at cell `base`, staging through a
/// zero-filled buffer at the row end.
///
/// # Safety
///
/// `base < cells`; SSE4.1 available; `row.len() >= cells`.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn load8_staged_c1<S: Sse41Elem>(row: &[S], base: usize) -> F64x8 {
  // SAFETY: a full chunk loads directly; the row end stages through a
  // zero-filled 8-element copy.
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

/// Loads + widens channel `ch` of the 8-pixel group at cell `cell`,
/// staging through a zero-filled buffer at the row end.
///
/// # Safety
///
/// `cell < cells`; `ch < 3`; SSE4.1 available; `row.len() >= cells * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn load8_staged_c3<S: Sse41Elem>(row: &[S], cell: usize, ch: usize) -> F64x8 {
  // SAFETY: a full group loads directly; the row end stages its 24
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

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn h_reduce_c1<S: Sse41Elem>(
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
    // SAFETY: each chunk loads in-bounds or stages the row end; coeffs
    // come from the 8-multiple arena slice.
    unsafe {
      let mut acc = _mm_setzero_pd();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let s = load8_staged_c1(row, start + ci * 8);
        let c = widen_coeffs(chunk);
        let real_in_chunk = real.saturating_sub(ci * 8).min(8);
        acc = if real_in_chunk == 8 {
          mac8_full(acc, s, c)
        } else {
          mac8_masked(acc, s, c, &padding_keep_mask8(real_in_chunk))
        };
      }
      h_tmp[j] = hsum_pd(acc);
    }
  }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn h_reduce_c3<S: Sse41Elem>(
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
    // SAFETY: each group loads in-bounds or stages the row end; coeffs
    // come from the 8-multiple arena slice.
    unsafe {
      let mut acc0 = _mm_setzero_pd();
      let mut acc1 = _mm_setzero_pd();
      let mut acc2 = _mm_setzero_pd();
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
      h_tmp[j * 3] = hsum_pd(acc0);
      h_tmp[j * 3 + 1] = hsum_pd(acc1);
      h_tmp[j * 3 + 2] = hsum_pd(acc2);
    }
  }
}

// ---- Concrete per-element entry points (the dispatcher's targets) -----

macro_rules! sse_h_entry {
  ($c1:ident, $c3:ident, $elem:ty, $doc:literal) => {
    #[doc = $doc]
    ///
    /// # Safety
    ///
    /// SSE4.1 available; the arena binds to this row (see [`h_reduce_c1`]).
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn $c1(
      row: &[$elem],
      starts: &[usize],
      ksize: &[usize],
      coeffs: &[f32],
      coff: &[usize],
      h_tmp: &mut [f64],
    ) {
      // SAFETY: forwarded under the caller's arena guarantees.
      unsafe { h_reduce_c1::<$elem>(row, starts, ksize, coeffs, coff, h_tmp) }
    }

    #[doc = $doc]
    ///
    /// # Safety
    ///
    /// SSE4.1 available; the arena binds to this row (see [`h_reduce_c3`]).
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn $c3(
      row: &[$elem],
      starts: &[usize],
      ksize: &[usize],
      coeffs: &[f32],
      coff: &[usize],
      h_tmp: &mut [f64],
    ) {
      // SAFETY: forwarded under the caller's arena guarantees.
      unsafe { h_reduce_c3::<$elem>(row, starts, ksize, coeffs, coff, h_tmp) }
    }
  };
}

sse_h_entry!(
  filter_h_reduce_row_u8_c1,
  filter_h_reduce_row_u8_c3,
  u8,
  "Filter H-pass over `u8` samples (1 / 3 channel)."
);
sse_h_entry!(
  filter_h_reduce_row_u16_c1,
  filter_h_reduce_row_u16_c3,
  u16,
  "Filter H-pass over `u16` samples (1 / 3 channel)."
);
sse_h_entry!(
  filter_h_reduce_row_f32_c1,
  filter_h_reduce_row_f32_c3,
  f32,
  "Filter H-pass over `f32` samples (1 / 3 channel)."
);

/// Filter V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64` (mul+add, **not**
/// fma) so each lane matches the scalar reference bit-for-bit. Two
/// elements per iteration.
///
/// # Safety
///
/// SSE4.1 available. `h_tmp.len() >= acc.len()`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn filter_v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: f32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm_set1_pd(f64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 2 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 2 <= n {
      let t = _mm_loadu_pd(h_tmp.as_ptr().add(i));
      let a = _mm_loadu_pd(acc.as_ptr().add(i));
      _mm_storeu_pd(acc.as_mut_ptr().add(i), _mm_add_pd(a, _mm_mul_pd(t, wv)));
      i += 2;
    }
  }
  for k in i..n {
    acc[k] += f64::from(w) * h_tmp[k];
  }
}
