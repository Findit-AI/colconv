//! AVX-512 separable-filter H/V passes: per output span, a dot product of
//! **signed** `f32` coefficients over a zero-padded coefficient arena.
//!
//! The signed twin of [`area_reduce`](super::area_reduce), widening the
//! SSE4.1 filter kernels. The c1 H-pass fits all eight taps of a chunk in
//! one `__m512d` (eight `f64` lanes); the c3 H-pass keeps the 128-bit
//! per-channel deinterleave (the RGB group does not pack into wide lanes,
//! exactly as the area u16/f32 c3 stays 128-bit). Both passes accumulate
//! in `f64` — the per-lane product is exact, so H parity is a small
//! tolerance from the tap-sum order and the V-pass (mul+add, **not** fma)
//! stays bit-equal to the scalar reference.

#![cfg_attr(not(feature = "std"), allow(dead_code))]
#![cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

/// One element type the c1 H-pass widens to a `__m512d`. The kernel body
/// is generic over the load; the public entry points pin it to
/// `u8` / `u16` / `f32`.
trait Avx512Elem: Copy + Default {
  /// Widen the 8 contiguous samples at `row[base..base + 8]` to one
  /// `__m512d`.
  ///
  /// # Safety
  ///
  /// `base + 8 <= row.len()`; AVX-512F+BW available.
  unsafe fn load8(row: &[Self], base: usize) -> __m512d;

  /// Widen channel `ch` of the 8-pixel interleaved group at cell `cell`
  /// to one `__m512d`.
  ///
  /// # Safety
  ///
  /// `(cell + 8) * 3 <= row.len()`; `ch < 3`; AVX-512F+BW available.
  unsafe fn load8_c3(row: &[Self], cell: usize, ch: usize) -> __m512d;
}

impl Avx512Elem for u8 {
  #[inline]
  #[target_feature(enable = "avx512f,avx512bw")]
  unsafe fn load8(row: &[u8], base: usize) -> __m512d {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe {
      widen_u16x8(_mm_cvtepu8_epi16(_mm_loadl_epi64(
        row.as_ptr().add(base).cast(),
      )))
    }
  }
  #[inline]
  #[target_feature(enable = "avx512f,avx512bw")]
  unsafe fn load8_c3(row: &[u8], cell: usize, ch: usize) -> __m512d {
    // SAFETY: `(cell + 8) * 3 <= row.len()`.
    unsafe {
      let base = cell * 3;
      let v0 = _mm_loadu_si128(row.as_ptr().add(base).cast());
      let v1 = _mm_loadu_si128(row.as_ptr().add(base + 8).cast());
      widen_u16x8(_mm_cvtepu8_epi16(gather_u8_c3(v0, v1, ch)))
    }
  }
}

impl Avx512Elem for u16 {
  #[inline]
  #[target_feature(enable = "avx512f,avx512bw")]
  unsafe fn load8(row: &[u16], base: usize) -> __m512d {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe { widen_u16x8(_mm_loadu_si128(row.as_ptr().add(base).cast())) }
  }
  #[inline]
  #[target_feature(enable = "avx512f,avx512bw")]
  unsafe fn load8_c3(row: &[u16], cell: usize, ch: usize) -> __m512d {
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

impl Avx512Elem for f32 {
  #[inline]
  #[target_feature(enable = "avx512f,avx512bw")]
  unsafe fn load8(row: &[f32], base: usize) -> __m512d {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe { _mm512_cvtps_pd(_mm256_loadu_ps(row.as_ptr().add(base))) }
  }
  #[inline]
  #[target_feature(enable = "avx512f,avx512bw")]
  unsafe fn load8_c3(row: &[f32], cell: usize, ch: usize) -> __m512d {
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
      _mm512_cvtps_pd(_mm256_set_m128(hi, lo))
    }
  }
}

/// Widens 8 `u16` lanes to one `__m512d` of 8 `f64`.
/// Register-only (matches the area path's safe `fn` convention).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn widen_u16x8(s16: __m128i) -> __m512d {
  _mm512_cvtepi32_pd(_mm256_cvtepu16_epi32(s16))
}

/// Gathers channel `ch`'s 8 `u8` samples of a 24-byte RGB group (area u8
/// c3 masks). Register-only.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
#[target_feature(enable = "avx512f,avx512bw")]
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
#[target_feature(enable = "avx512f,avx512bw")]
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

/// Widens 8 signed `f32` coefficients to one `__m512d`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn widen_coeffs(c: &[f32]) -> __m512d {
  // SAFETY: caller passes an 8-multiple arena chunk.
  unsafe { _mm512_cvtps_pd(_mm256_loadu_ps(c.as_ptr())) }
}

/// Accumulates 8 widened samples against 8 widened coefficients (mul+add),
/// every lane real — so a real zero coefficient multiplies its (possibly
/// non-finite) sample as is, matching the scalar reference. Register-only.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac8_full(acc: __m512d, s: __m512d, c: __m512d) -> __m512d {
  _mm512_add_pd(acc, _mm512_mul_pd(s, c))
}

/// As [`mac8_full`] for a span's trailing partial chunk: keeps the low
/// `real_in_chunk` sample lanes (the real taps) and zeroes the rest (arena
/// padding) so a non-finite out-of-window sample cannot poison the span
/// through `0.0 * NaN`. A real zero coefficient sits in a kept lane, so it
/// multiplies its sample exactly as scalar does. `real_in_chunk` is in
/// `1..=7` (full chunks take [`mac8_full`]), so `1 << real_in_chunk` never
/// overflows the `__mmask8`. Register-only.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac8_masked(acc: __m512d, s: __m512d, c: __m512d, real_in_chunk: usize) -> __m512d {
  let keep: __mmask8 = (1u8 << real_in_chunk) - 1;
  let sf = _mm512_maskz_mov_pd(keep, s);
  _mm512_add_pd(acc, _mm512_mul_pd(sf, c))
}

/// Loads + widens 8 contiguous samples at cell `base`, staging the row
/// end.
///
/// # Safety
///
/// `base < cells`; AVX-512F+BW available; `row.len() >= cells`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load8_staged_c1<S: Avx512Elem>(row: &[S], base: usize) -> __m512d {
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
/// `cell < cells`; `ch < 3`; AVX-512F+BW available; `row.len() >= cells *
/// 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load8_staged_c3<S: Avx512Elem>(row: &[S], cell: usize, ch: usize) -> __m512d {
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
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn h_reduce_c1<S: Avx512Elem>(
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
      let mut acc = _mm512_setzero_pd();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let s = load8_staged_c1(row, start + ci * 8);
        let c = widen_coeffs(chunk);
        let real_in_chunk = real.saturating_sub(ci * 8).min(8);
        acc = if real_in_chunk == 8 {
          mac8_full(acc, s, c)
        } else {
          mac8_masked(acc, s, c, real_in_chunk)
        };
      }
      h_tmp[j] = _mm512_reduce_add_pd(acc);
    }
  }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn h_reduce_c3<S: Avx512Elem>(
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
      let mut acc0 = _mm512_setzero_pd();
      let mut acc1 = _mm512_setzero_pd();
      let mut acc2 = _mm512_setzero_pd();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let cell = start + ci * 8;
        let c = widen_coeffs(chunk);
        let real_in_chunk = real.saturating_sub(ci * 8).min(8);
        if real_in_chunk == 8 {
          acc0 = mac8_full(acc0, load8_staged_c3(row, cell, 0), c);
          acc1 = mac8_full(acc1, load8_staged_c3(row, cell, 1), c);
          acc2 = mac8_full(acc2, load8_staged_c3(row, cell, 2), c);
        } else {
          acc0 = mac8_masked(acc0, load8_staged_c3(row, cell, 0), c, real_in_chunk);
          acc1 = mac8_masked(acc1, load8_staged_c3(row, cell, 1), c, real_in_chunk);
          acc2 = mac8_masked(acc2, load8_staged_c3(row, cell, 2), c, real_in_chunk);
        }
      }
      h_tmp[j * 3] = _mm512_reduce_add_pd(acc0);
      h_tmp[j * 3 + 1] = _mm512_reduce_add_pd(acc1);
      h_tmp[j * 3 + 2] = _mm512_reduce_add_pd(acc2);
    }
  }
}

// ---- Concrete per-element entry points (the dispatcher's targets) -----

macro_rules! avx512_h_entry {
  ($c1:ident, $c3:ident, $elem:ty, $doc:literal) => {
    #[doc = $doc]
    ///
    /// # Safety
    ///
    /// AVX-512F+BW available; the arena binds to this row (see
    /// [`h_reduce_c1`]).
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
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
    /// AVX-512F+BW available; the arena binds to this row (see
    /// [`h_reduce_c3`]).
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
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

avx512_h_entry!(
  filter_h_reduce_row_u8_c1,
  filter_h_reduce_row_u8_c3,
  u8,
  "Filter H-pass over `u8` samples (1 / 3 channel)."
);
avx512_h_entry!(
  filter_h_reduce_row_u16_c1,
  filter_h_reduce_row_u16_c3,
  u16,
  "Filter H-pass over `u16` samples (1 / 3 channel)."
);
avx512_h_entry!(
  filter_h_reduce_row_f32_c1,
  filter_h_reduce_row_f32_c3,
  f32,
  "Filter H-pass over `f32` samples (1 / 3 channel)."
);

/// Filter V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64` (mul+add, **not**
/// fma) so each lane matches the scalar reference bit-for-bit. Eight
/// elements per iteration.
///
/// # Safety
///
/// AVX-512F available. `h_tmp.len() >= acc.len()`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn filter_v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: f32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm512_set1_pd(f64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 8 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 8 <= n {
      let t = _mm512_loadu_pd(h_tmp.as_ptr().add(i));
      let a = _mm512_loadu_pd(acc.as_ptr().add(i));
      _mm512_storeu_pd(
        acc.as_mut_ptr().add(i),
        _mm512_add_pd(a, _mm512_mul_pd(t, wv)),
      );
      i += 8;
    }
  }
  for k in i..n {
    acc[k] += f64::from(w) * h_tmp[k];
  }
}
