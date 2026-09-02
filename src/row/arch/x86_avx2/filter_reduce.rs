//! AVX2 separable-filter H/V passes: per output span, a dot product of
//! **signed** `f32` coefficients over a zero-padded coefficient arena.
//!
//! The signed twin of [`area_reduce`](super::area_reduce), widening the
//! SSE4.1 filter kernels. The c1 H-pass accumulates four `f64` lanes per
//! `__m256d` (two vectors per 8-tap chunk); the c3 H-pass keeps the
//! 128-bit per-channel deinterleave (the RGB group does not pack into
//! 256-bit lanes, exactly as the area u16/f32 c3 stays 128-bit). Both
//! passes accumulate in `f64` — the per-lane product is exact, so H
//! parity is a small tolerance from the tap-sum order and the V-pass
//! (mul+add, **not** fma, off the `fma` target-feature) stays bit-equal
//! to the scalar reference.

#![cfg_attr(not(feature = "std"), allow(dead_code))]
#![cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::scalar::filter_reduce::padding_keep_mask8;

/// One element type the c1 H-pass widens to `__m256d` quads. The kernel
/// body is generic over the load; the public entry points pin it to
/// `u8` / `u16` / `f32`.
trait Avx2Elem: Copy + Default {
  /// Widen the 8 contiguous samples at `row[base..base + 8]` to two `f64`
  /// quads `(lanes 0-3, lanes 4-7)`.
  ///
  /// # Safety
  ///
  /// `base + 8 <= row.len()`; AVX2 available.
  unsafe fn load8(row: &[Self], base: usize) -> (__m256d, __m256d);

  /// Widen channel `ch` of the 8-pixel interleaved group at cell `cell`
  /// to two `f64` quads.
  ///
  /// # Safety
  ///
  /// `(cell + 8) * 3 <= row.len()`; `ch < 3`; AVX2 available.
  unsafe fn load8_c3(row: &[Self], cell: usize, ch: usize) -> (__m256d, __m256d);
}

impl Avx2Elem for u8 {
  #[inline]
  #[target_feature(enable = "avx2")]
  unsafe fn load8(row: &[u8], base: usize) -> (__m256d, __m256d) {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe {
      let s16 = _mm_cvtepu8_epi16(_mm_loadl_epi64(row.as_ptr().add(base).cast()));
      widen_u16x8(s16)
    }
  }
  #[inline]
  #[target_feature(enable = "avx2")]
  unsafe fn load8_c3(row: &[u8], cell: usize, ch: usize) -> (__m256d, __m256d) {
    // SAFETY: `(cell + 8) * 3 <= row.len()`.
    unsafe {
      let base = cell * 3;
      let v0 = _mm_loadu_si128(row.as_ptr().add(base).cast());
      let v1 = _mm_loadu_si128(row.as_ptr().add(base + 8).cast());
      widen_u16x8(_mm_cvtepu8_epi16(gather_u8_c3(v0, v1, ch)))
    }
  }
}

impl Avx2Elem for u16 {
  #[inline]
  #[target_feature(enable = "avx2")]
  unsafe fn load8(row: &[u16], base: usize) -> (__m256d, __m256d) {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe { widen_u16x8(_mm_loadu_si128(row.as_ptr().add(base).cast())) }
  }
  #[inline]
  #[target_feature(enable = "avx2")]
  unsafe fn load8_c3(row: &[u16], cell: usize, ch: usize) -> (__m256d, __m256d) {
    // SAFETY: `(cell + 8) * 3 <= row.len()`.
    unsafe {
      let base = cell * 3;
      let v0 = _mm_loadu_si128(row.as_ptr().add(base).cast());
      let v1 = _mm_loadu_si128(row.as_ptr().add(base + 8).cast());
      let v2 = _mm_loadu_si128(row.as_ptr().add(base + 16).cast());
      widen_u16x8(gather_u16_c3(v0, v1, v2, ch))
    }
  }
}

impl Avx2Elem for f32 {
  #[inline]
  #[target_feature(enable = "avx2")]
  unsafe fn load8(row: &[f32], base: usize) -> (__m256d, __m256d) {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe {
      (
        _mm256_cvtps_pd(_mm_loadu_ps(row.as_ptr().add(base))),
        _mm256_cvtps_pd(_mm_loadu_ps(row.as_ptr().add(base + 4))),
      )
    }
  }
  #[inline]
  #[target_feature(enable = "avx2")]
  unsafe fn load8_c3(row: &[f32], cell: usize, ch: usize) -> (__m256d, __m256d) {
    // SAFETY: `(cell + 8) * 3 <= row.len()`.
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
      (_mm256_cvtps_pd(lo), _mm256_cvtps_pd(hi))
    }
  }
}

/// Widens 8 `u16` lanes to two `f64` quads `(lanes 0-3, 4-7)`.
/// Register-only (matches the area path's safe `fn` convention).
#[inline]
#[target_feature(enable = "avx2")]
fn widen_u16x8(s16: __m128i) -> (__m256d, __m256d) {
  (
    _mm256_cvtepi32_pd(_mm_cvtepu16_epi32(s16)),
    _mm256_cvtepi32_pd(_mm_cvtepu16_epi32(_mm_srli_si128::<8>(s16))),
  )
}

/// Gathers channel `ch`'s 8 `u8` samples of a 24-byte RGB group (area u8
/// c3 masks). Register-only.
#[inline]
#[target_feature(enable = "avx2")]
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

/// Gathers channel `ch`'s 8 `u16` samples of a 48-byte RGB group (area
/// u16 c3 masks). Register-only.
#[inline]
#[target_feature(enable = "avx2")]
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

/// Deinterleaves four interleaved RGB `f32` pixels into planar lanes (area
/// SSE4.1 `deint3_f32`, reused — the RGB group stays 128-bit). Register-only.
#[inline]
#[target_feature(enable = "avx2")]
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

/// Widens 8 signed `f32` coefficients to two `f64` quads.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn widen_coeffs(c: &[f32]) -> (__m256d, __m256d) {
  // SAFETY: caller passes an 8-multiple arena chunk.
  unsafe {
    (
      _mm256_cvtps_pd(_mm_loadu_ps(c.as_ptr())),
      _mm256_cvtps_pd(_mm_loadu_ps(c.as_ptr().add(4))),
    )
  }
}

/// Bitwise-ANDs a sample quad with a keep-mask quad (all-ones keeps a real
/// lane, `+0.0` clears a padding lane). Register-only.
#[inline]
#[target_feature(enable = "avx2")]
fn mask_lane(sf: __m256d, keep: __m256d) -> __m256d {
  _mm256_and_pd(sf, keep)
}

/// Accumulates two sample quads against two coefficient quads (mul+add),
/// every lane real — so a real zero coefficient multiplies its (possibly
/// non-finite) sample as is, matching scalar. Register-only.
#[inline]
#[target_feature(enable = "avx2")]
fn mac8_full(acc: __m256d, s_lo: __m256d, s_hi: __m256d, c_lo: __m256d, c_hi: __m256d) -> __m256d {
  let a = _mm256_add_pd(acc, _mm256_mul_pd(s_lo, c_lo));
  _mm256_add_pd(a, _mm256_mul_pd(s_hi, c_hi))
}

/// As [`mac8_full`] for a span's trailing partial chunk: `keep` is the
/// per-lane padding mask ([`padding_keep_mask8`]) — padding lanes clear to
/// `+0.0` (no `0.0 * NaN` poison), real lanes (including a real zero
/// coefficient) pass through.
///
/// # Safety
///
/// AVX2 available; `keep` is a fixed 8-element array.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mac8_masked(
  acc: __m256d,
  s_lo: __m256d,
  s_hi: __m256d,
  c_lo: __m256d,
  c_hi: __m256d,
  keep: &[f64; 8],
) -> __m256d {
  // SAFETY: the two quad loads stay within the 8-element `keep`.
  unsafe {
    let k_lo = _mm256_loadu_pd(keep.as_ptr());
    let k_hi = _mm256_loadu_pd(keep.as_ptr().add(4));
    let a = _mm256_add_pd(acc, _mm256_mul_pd(mask_lane(s_lo, k_lo), c_lo));
    _mm256_add_pd(a, _mm256_mul_pd(mask_lane(s_hi, k_hi), c_hi))
  }
}

/// Sums the four `f64` lanes. Register-only.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_pd(v: __m256d) -> f64 {
  let lo = _mm256_castpd256_pd128(v);
  let hi = _mm256_extractf128_pd::<1>(v);
  let s = _mm_add_pd(lo, hi);
  _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)))
}

/// Loads + widens 8 contiguous samples at cell `base`, staging the row
/// end.
///
/// # Safety
///
/// `base < cells`; AVX2 available; `row.len() >= cells`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load8_staged_c1<S: Avx2Elem>(row: &[S], base: usize) -> (__m256d, __m256d) {
  // SAFETY: a full chunk loads directly; the row end stages a zero-filled
  // 8-element copy.
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

/// Loads + widens channel `ch` of the group at cell `cell`, staging the
/// row end.
///
/// # Safety
///
/// `cell < cells`; `ch < 3`; AVX2 available; `row.len() >= cells * 3`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load8_staged_c3<S: Avx2Elem>(row: &[S], cell: usize, ch: usize) -> (__m256d, __m256d) {
  // SAFETY: a full group loads directly; the row end stages its 24
  // interleaved samples.
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
#[target_feature(enable = "avx2")]
unsafe fn h_reduce_c1<S: Avx2Elem>(
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
    // from the 8-multiple arena slice.
    unsafe {
      let mut acc = _mm256_setzero_pd();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let (s_lo, s_hi) = load8_staged_c1(row, start + ci * 8);
        let (c_lo, c_hi) = widen_coeffs(chunk);
        let real_in_chunk = real.saturating_sub(ci * 8).min(8);
        acc = if real_in_chunk == 8 {
          mac8_full(acc, s_lo, s_hi, c_lo, c_hi)
        } else {
          mac8_masked(
            acc,
            s_lo,
            s_hi,
            c_lo,
            c_hi,
            &padding_keep_mask8(real_in_chunk),
          )
        };
      }
      h_tmp[j] = hsum_pd(acc);
    }
  }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn h_reduce_c3<S: Avx2Elem>(
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
    // from the 8-multiple arena slice.
    unsafe {
      let mut acc0 = _mm256_setzero_pd();
      let mut acc1 = _mm256_setzero_pd();
      let mut acc2 = _mm256_setzero_pd();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let cell = start + ci * 8;
        let (c_lo, c_hi) = widen_coeffs(chunk);
        let real_in_chunk = real.saturating_sub(ci * 8).min(8);
        let (r_lo, r_hi) = load8_staged_c3(row, cell, 0);
        let (g_lo, g_hi) = load8_staged_c3(row, cell, 1);
        let (b_lo, b_hi) = load8_staged_c3(row, cell, 2);
        if real_in_chunk == 8 {
          acc0 = mac8_full(acc0, r_lo, r_hi, c_lo, c_hi);
          acc1 = mac8_full(acc1, g_lo, g_hi, c_lo, c_hi);
          acc2 = mac8_full(acc2, b_lo, b_hi, c_lo, c_hi);
        } else {
          let keep = padding_keep_mask8(real_in_chunk);
          acc0 = mac8_masked(acc0, r_lo, r_hi, c_lo, c_hi, &keep);
          acc1 = mac8_masked(acc1, g_lo, g_hi, c_lo, c_hi, &keep);
          acc2 = mac8_masked(acc2, b_lo, b_hi, c_lo, c_hi, &keep);
        }
      }
      h_tmp[j * 3] = hsum_pd(acc0);
      h_tmp[j * 3 + 1] = hsum_pd(acc1);
      h_tmp[j * 3 + 2] = hsum_pd(acc2);
    }
  }
}

// ---- Concrete per-element entry points (the dispatcher's targets) -----

macro_rules! avx2_h_entry {
  ($c1:ident, $c3:ident, $elem:ty, $doc:literal) => {
    #[doc = $doc]
    ///
    /// # Safety
    ///
    /// AVX2 available; the arena binds to this row (see [`h_reduce_c1`]).
    #[inline]
    #[target_feature(enable = "avx2")]
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
    /// AVX2 available; the arena binds to this row (see [`h_reduce_c3`]).
    #[inline]
    #[target_feature(enable = "avx2")]
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

avx2_h_entry!(
  filter_h_reduce_row_u8_c1,
  filter_h_reduce_row_u8_c3,
  u8,
  "Filter H-pass over `u8` samples (1 / 3 channel)."
);
avx2_h_entry!(
  filter_h_reduce_row_u16_c1,
  filter_h_reduce_row_u16_c3,
  u16,
  "Filter H-pass over `u16` samples (1 / 3 channel)."
);
avx2_h_entry!(
  filter_h_reduce_row_f32_c1,
  filter_h_reduce_row_f32_c3,
  f32,
  "Filter H-pass over `f32` samples (1 / 3 channel)."
);

/// Filter V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64` (mul+add, **not**
/// fma) so each lane matches the scalar reference bit-for-bit. Four
/// elements per iteration.
///
/// # Safety
///
/// AVX2 available. `h_tmp.len() >= acc.len()`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn filter_v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: f32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm256_set1_pd(f64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 4 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 4 <= n {
      let t = _mm256_loadu_pd(h_tmp.as_ptr().add(i));
      let a = _mm256_loadu_pd(acc.as_ptr().add(i));
      _mm256_storeu_pd(
        acc.as_mut_ptr().add(i),
        _mm256_add_pd(a, _mm256_mul_pd(t, wv)),
      );
      i += 4;
    }
  }
  for k in i..n {
    acc[k] += f64::from(w) * h_tmp[k];
  }
}
