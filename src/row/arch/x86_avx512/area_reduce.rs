//! AVX-512 fused-downscale H-pass and V-pass, widening the SSE4.1
//! kernels over the plan-time zero-padded u16 weight arena (see the
//! NEON sibling for the arena contract).
//!
//! The H-pass widens within a span to 32 taps per iteration
//! (`_mm512_mullo_epi16` / `_mm512_mulhi_epu16` into exact u32 lanes);
//! the remaining ≤24 padding-aligned taps and any chunk that would
//! read past the row end fall to the proven 128-bit SSE step, which
//! owns the row-end staging. The V-pass reaches 16 `u64` products per
//! iteration. Bit-identical to the scalar reference by integer
//! associativity.

#![cfg_attr(not(feature = "std"), allow(dead_code))]
#![cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

/// Sums the four u32 lanes of a 128-bit accumulator.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn hsum128_u32(acc: __m128i) -> u32 {
  let hi = _mm_shuffle_epi32::<0b01_00_11_10>(acc);
  let s = _mm_add_epi32(acc, hi);
  let s2 = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
  _mm_cvtsi128_si32(s2) as u32
}

/// Accumulates the eight exact u32 products of `s16 * w` (128-bit, 8
/// u16 lanes) into `acc`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac128_u16x8(acc: __m128i, s16: __m128i, w: __m128i) -> __m128i {
  let lo = _mm_mullo_epi16(s16, w);
  let hi = _mm_mulhi_epu16(s16, w);
  let acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(lo, hi));
  _mm_add_epi32(acc, _mm_unpackhi_epi16(lo, hi))
}

/// Accumulates the thirty-two exact u32 products of `s16 * w`
/// (512-bit, 32 u16 lanes) into `acc`. The `unpack` interleaves run
/// per 128-bit lane, so a product lands in some lane of `acc` — which
/// lane is immaterial to the final reduction.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac512_u16x32(acc: __m512i, s16: __m512i, w: __m512i) -> __m512i {
  let lo = _mm512_mullo_epi16(s16, w);
  let hi = _mm512_mulhi_epu16(s16, w);
  let acc = _mm512_add_epi32(acc, _mm512_unpacklo_epi16(lo, hi));
  _mm512_add_epi32(acc, _mm512_unpackhi_epi16(lo, hi))
}

/// One proven 128-bit c1 step for an 8-tap group at `base`, with
/// row-end staging.
///
/// # Safety
///
/// AVX-512 (⊇ SSE4.1) available; `base < row.len()`; `w8.len() >= 8`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn c1_group(acc: __m128i, row: &[u8], base: usize, w8: &[u16]) -> __m128i {
  // SAFETY: the 8-byte load is guarded against row.len() or staged
  // through a zero-filled stack copy; the weight load reads 8 arena
  // u16.
  unsafe {
    let s8 = if base + 8 <= row.len() {
      _mm_loadl_epi64(row.as_ptr().add(base).cast())
    } else {
      let mut sbuf = [0u8; 8];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      _mm_loadl_epi64(sbuf.as_ptr().cast())
    };
    let w = _mm_loadu_si128(w8.as_ptr().cast());
    mac128_u16x8(acc, _mm_cvtepu8_epi16(s8), w)
  }
}

/// # Safety
///
/// AVX-512F+BW must be available. Caller guarantees the padded-arena
/// contract of the NEON sibling: `w16_off.len() == starts.len() + 1`
/// monotonic and bounded by `w16.len()`, span lengths multiples of 8,
/// `row.len() >= cells`, `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_h_reduce_row_c1(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: the wide 32-byte load runs only when fully in-bounds
    // (`start + t + 32 <= row.len()`); the remaining ≤24 taps and the
    // boundary chunk delegate to `c1_group`, which stages the row end.
    unsafe {
      let mut acc = _mm512_setzero_si512();
      let mut acc128 = _mm_setzero_si128();
      let mut t = 0usize;
      while t + 32 <= len && start + t + 32 <= row.len() {
        // 32 contiguous u8 -> 32 u16.
        let s8 = _mm256_loadu_si256(row.as_ptr().add(start + t).cast());
        let s16 = _mm512_cvtepu8_epi16(s8);
        let w = _mm512_loadu_si512(span[t..].as_ptr().cast());
        acc = mac512_u16x32(acc, s16, w);
        t += 32;
      }
      while t + 8 <= len {
        acc128 = c1_group(acc128, row, start + t, &span[t..]);
        t += 8;
      }
      let wide = _mm512_reduce_add_epi32(acc) as u32;
      h_tmp[j] = wide.wrapping_add(hsum128_u32(acc128));
    }
  }
}

/// The SSE4.1 per-128-lane deinterleave masks broadcast to all four
/// 512-bit lanes, so one shuffle deinterleaves four 8-pixel groups.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn c3_masks() -> ([__m512i; 3], [__m512i; 3]) {
  let m0 = [
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2),
  ];
  let m1 = [
    _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, -1, -1, -1, -1, -1, -1,
    ),
    _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, -1, -1, -1, -1, -1,
    ),
    _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, -1, -1, -1, -1, -1,
    ),
  ];
  (
    [
      _mm512_broadcast_i32x4(m0[0]),
      _mm512_broadcast_i32x4(m0[1]),
      _mm512_broadcast_i32x4(m0[2]),
    ],
    [
      _mm512_broadcast_i32x4(m1[0]),
      _mm512_broadcast_i32x4(m1[1]),
      _mm512_broadcast_i32x4(m1[2]),
    ],
  )
}

/// One proven 128-bit c3 step for an 8-pixel group at cell `cell`,
/// with row-end staging.
///
/// # Safety
///
/// AVX-512 (⊇ SSE4.1) available; `cell < cells`; `w8.len() >= 8`; `m0`
/// / `m1` carry the SSE masks in their low 128 bits.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn c3_group(
  acc: &mut [__m128i; 3],
  row: &[u8],
  cell: usize,
  w8: &[u16],
  m0: &[__m512i; 3],
  m1: &[__m512i; 3],
) {
  let base = cell * 3;
  // SAFETY: the two 16-byte loads cover bytes 0..24 of the chunk and
  // are guarded against row.len() or staged through a zero-filled
  // 24-byte copy.
  unsafe {
    let (v0, v1) = if base + 24 <= row.len() {
      (
        _mm_loadu_si128(row.as_ptr().add(base).cast()),
        _mm_loadu_si128(row.as_ptr().add(base + 8).cast()),
      )
    } else {
      let mut sbuf = [0u8; 24];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      (
        _mm_loadu_si128(sbuf.as_ptr().cast()),
        _mm_loadu_si128(sbuf.as_ptr().add(8).cast()),
      )
    };
    let w = _mm_loadu_si128(w8.as_ptr().cast());
    for ch in 0..3 {
      let m0c = _mm512_castsi512_si128(m0[ch]);
      let m1c = _mm512_castsi512_si128(m1[ch]);
      let gathered = _mm_or_si128(_mm_shuffle_epi8(v0, m0c), _mm_shuffle_epi8(v1, m1c));
      acc[ch] = mac128_u16x8(acc[ch], _mm_cvtepu8_epi16(gathered), w);
    }
  }
}

/// 3-channel variant. The wide path packs four 8-pixel groups into the
/// four 128-bit lanes and deinterleaves all with one per-lane
/// `_mm512_shuffle_epi8`; `_mm512_unpacklo_epi8` widens each lane's
/// low 8 samples `u8 -> u16`.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_h_reduce_row_c3(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  let (m0, m1) = c3_masks();
  let zero = _mm512_setzero_si512();
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: the wide path runs only when all four groups' 24-byte
    // spans are fully in-bounds (`(start + t + 32) * 3 <= row.len()`);
    // boundary and trailing groups delegate to `c3_group`, which
    // stages the row end.
    unsafe {
      let mut acc = [_mm512_setzero_si512(); 3];
      let mut acc128 = [_mm_setzero_si128(); 3];
      let mut t = 0usize;
      while t + 32 <= len && (start + t + 32) * 3 <= row.len() {
        // Four 8-pixel groups -> the four 128-bit lanes. `inserti32x4`
        // takes a const lane index, so the four inserts are unrolled.
        let lo = |g: usize| _mm_loadu_si128(row.as_ptr().add((start + t + g * 8) * 3).cast());
        let hi = |g: usize| _mm_loadu_si128(row.as_ptr().add((start + t + g * 8) * 3 + 8).cast());
        let v0 = _mm512_inserti32x4::<3>(
          _mm512_inserti32x4::<2>(
            _mm512_inserti32x4::<1>(_mm512_castsi128_si512(lo(0)), lo(1)),
            lo(2),
          ),
          lo(3),
        );
        let v1 = _mm512_inserti32x4::<3>(
          _mm512_inserti32x4::<2>(
            _mm512_inserti32x4::<1>(_mm512_castsi128_si512(hi(0)), hi(1)),
            hi(2),
          ),
          hi(3),
        );
        // The four 128-bit lanes hold the four groups' 8 u16 weights.
        let w = _mm512_loadu_si512(span[t..].as_ptr().cast());
        for ch in 0..3 {
          let gathered = _mm512_or_si512(
            _mm512_shuffle_epi8(v0, m0[ch]),
            _mm512_shuffle_epi8(v1, m1[ch]),
          );
          let s16 = _mm512_unpacklo_epi8(gathered, zero);
          acc[ch] = mac512_u16x32(acc[ch], s16, w);
        }
        t += 32;
      }
      while t + 8 <= len {
        c3_group(&mut acc128, row, start + t, &span[t..], &m0, &m1);
        t += 8;
      }
      for ch in 0..3 {
        let wide = _mm512_reduce_add_epi32(acc[ch]) as u32;
        h_tmp[j * 3 + ch] = wide.wrapping_add(hsum128_u32(acc128[ch]));
      }
    }
  }
}

/// V-pass AXPY: `acc[i] += w * h_tmp[i]`, exact u64 lanes via
/// `_mm512_mul_epu32` over even/odd u32 lanes, reassembled with
/// `_mm512_permutex2var_epi64` (16 elements per iteration).
///
/// # Safety
///
/// AVX-512F must be available. `h_tmp.len() >= acc.len()`; every
/// product-sum stays within u64 (the engine's denominator bound).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm512_set1_epi64(i64::from(w));
  // even product e -> contiguous lane 2e; odd product o -> lane 2o+1.
  let idx_lo = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  let idx_hi = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  let mut i = 0usize;
  // SAFETY: loop guard `i + 16 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 16 <= n {
      let t = _mm512_loadu_si512(h_tmp.as_ptr().add(i).cast());
      // even = [t0,t2,..,t14]*w, odd = [t1,t3,..,t15]*w (u64 lanes).
      let even = _mm512_mul_epu32(t, wv);
      let odd = _mm512_mul_epu32(_mm512_srli_epi64::<32>(t), wv);
      // Interleave back to contiguous [t0..t7]*w and [t8..t15]*w.
      let lo = _mm512_permutex2var_epi64(even, idx_lo, odd);
      let hi = _mm512_permutex2var_epi64(even, idx_hi, odd);
      let a_lo = _mm512_loadu_si512(acc.as_ptr().add(i).cast());
      let a_hi = _mm512_loadu_si512(acc.as_ptr().add(i + 8).cast());
      _mm512_storeu_si512(acc.as_mut_ptr().add(i).cast(), _mm512_add_epi64(a_lo, lo));
      _mm512_storeu_si512(
        acc.as_mut_ptr().add(i + 8).cast(),
        _mm512_add_epi64(a_hi, hi),
      );
      i += 16;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * u64::from(h_tmp[k]);
  }
}

/// Sums the two u64 lanes of a 128-bit accumulator.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn hsum128_u64(acc: __m128i) -> u64 {
  let hi = _mm_unpackhi_epi64(acc, acc);
  _mm_cvtsi128_si64(_mm_add_epi64(acc, hi)) as u64
}

/// Accumulates the eight exact `u16 * u16 -> u32` products of `s16 * w`
/// (128-bit, 8 u16 lanes) into the two u64 lanes of `acc`. Mirrors the
/// SSE4.1 `mac_u16x8_u64`: unlike the u8 [`mac128_u16x8`], a `u16` span
/// sum overflows `u32`, so the products widen to u64 before
/// accumulating.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac128_u16x8_u64(acc: __m128i, s16: __m128i, w: __m128i) -> __m128i {
  let lo = _mm_mullo_epi16(s16, w);
  let hi = _mm_mulhi_epu16(s16, w);
  let p_lo = _mm_unpacklo_epi16(lo, hi);
  let p_hi = _mm_unpackhi_epi16(lo, hi);
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(p_lo));
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(_mm_srli_si128::<8>(p_lo)));
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(p_hi));
  _mm_add_epi64(acc, _mm_cvtepu32_epi64(_mm_srli_si128::<8>(p_hi)))
}

/// Accumulates the thirty-two exact `u16 * u16 -> u32` products of
/// `s16 * w` (512-bit, 32 u16 lanes) into the eight u64 lanes of `acc`.
/// The `unpack` interleaves run per 128-bit lane, so a product lands in
/// some lane of `acc` — which lane is immaterial to the final
/// reduction. As [`mac128_u16x8_u64`], the products widen `u32 -> u64`
/// (a `u16` span sum overflows `u32`) via `_mm512_cvtepu32_epi64` over
/// the low/high 256-bit halves of each interleaved product vector.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac512_u16x32_u64(acc: __m512i, s16: __m512i, w: __m512i) -> __m512i {
  let lo = _mm512_mullo_epi16(s16, w);
  let hi = _mm512_mulhi_epu16(s16, w);
  let p_lo = _mm512_unpacklo_epi16(lo, hi);
  let p_hi = _mm512_unpackhi_epi16(lo, hi);
  let acc = _mm512_add_epi64(acc, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(p_lo)));
  let acc = _mm512_add_epi64(
    acc,
    _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(p_lo)),
  );
  let acc = _mm512_add_epi64(acc, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(p_hi)));
  _mm512_add_epi64(
    acc,
    _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(p_hi)),
  )
}

/// One proven 128-bit u16 c1 step for an 8-tap group at `base`, with
/// row-end staging. The samples load directly as `u16` and the products
/// accumulate in `u64` lanes.
///
/// # Safety
///
/// AVX-512 (⊇ SSE4.1) available; `base < row.len()`; `w8.len() >= 8`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn c1_group_u16(acc: __m128i, row: &[u16], base: usize, w8: &[u16]) -> __m128i {
  // SAFETY: the 8-element u16 load is guarded against row.len() or
  // staged through a zero-filled stack copy; the weight load reads 8
  // arena u16.
  unsafe {
    let s16 = if base + 8 <= row.len() {
      _mm_loadu_si128(row.as_ptr().add(base).cast())
    } else {
      let mut sbuf = [0u16; 8];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      _mm_loadu_si128(sbuf.as_ptr().cast())
    };
    let w = _mm_loadu_si128(w8.as_ptr().cast());
    mac128_u16x8_u64(acc, s16, w)
  }
}

/// 16-bit-element H-pass (1 channel): like [`area_h_reduce_row_c1`] but
/// the samples load directly as `u16` (no `_mm512_cvtepu8_epi16`) and
/// the products accumulate in `u64` lanes (a single span sum can exceed
/// `u32`). The wide path still widens 32 taps per iteration; the
/// remaining ≤24 padding-aligned taps and any chunk that would read
/// past the row end fall to the 128-bit u16 step.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `u16`
/// elements and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_h_reduce_row_u16_c1(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: the wide 32-element u16 load runs only when fully
    // in-bounds (`start + t + 32 <= row.len()`); the remaining ≤24 taps
    // and the boundary chunk delegate to `c1_group_u16`, which stages
    // the row end.
    unsafe {
      let mut acc = _mm512_setzero_si512();
      let mut acc128 = _mm_setzero_si128();
      let mut t = 0usize;
      while t + 32 <= len && start + t + 32 <= row.len() {
        let s16 = _mm512_loadu_si512(row.as_ptr().add(start + t).cast());
        let w = _mm512_loadu_si512(span[t..].as_ptr().cast());
        acc = mac512_u16x32_u64(acc, s16, w);
        t += 32;
      }
      while t + 8 <= len {
        acc128 = c1_group_u16(acc128, row, start + t, &span[t..]);
        t += 8;
      }
      let wide = _mm512_reduce_add_epi64(acc) as u64;
      h_tmp[j] = wide.wrapping_add(hsum128_u64(acc128));
    }
  }
}

/// The SSE4.1 per-channel u16 deinterleave mask triples (`u16` index
/// `ch + 3t` split across three overlapping 16-byte loads), copied
/// verbatim from the validated SSE4.1 u16 c3 kernel.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn c3_masks_u16() -> ([__m128i; 3], [__m128i; 3], [__m128i; 3]) {
  let m0: [__m128i; 3] = [
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 7, 6, 1, 0),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 14, 9, 8, 3, 2),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 10, 5, 4),
  ];
  let m1: [__m128i; 3] = [
    _mm_set_epi8(-1, -1, -1, -1, 15, 14, 9, 8, 3, 2, -1, -1, -1, -1, -1, -1),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, 11, 10, 5, 4, -1, -1, -1, -1, -1, -1),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 12, 7, 6, 1, 0, -1, -1, -1, -1),
  ];
  let m2: [__m128i; 3] = [
    _mm_set_epi8(11, 10, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    _mm_set_epi8(13, 12, 7, 6, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    _mm_set_epi8(15, 14, 9, 8, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
  ];
  (m0, m1, m2)
}

/// One proven 128-bit u16 c3 step for an 8-pixel group at cell `cell`
/// (`u16` base `cell * 3`), with row-end staging. Three overlapping
/// 16-byte loads cover the chunk's 24 `u16`, and per-channel
/// `_mm_shuffle_epi8` mask triples gather each channel's eight samples
/// (a `u16` lives in exactly one load, the other two masks zeroing that
/// lane); the products accumulate in `u64` lanes. Mirrors the validated
/// SSE4.1 u16 c3 inner step.
///
/// # Safety
///
/// AVX-512 (⊇ SSE4.1) available; `cell < cells`; `w8.len() >= 8`; `m0`
/// / `m1` / `m2` carry the SSE u16 masks.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn c3_group_u16(
  acc: &mut [__m128i; 3],
  row: &[u16],
  cell: usize,
  w8: &[u16],
  m0: &[__m128i; 3],
  m1: &[__m128i; 3],
  m2: &[__m128i; 3],
) {
  let base = cell * 3;
  // SAFETY: the three 16-byte loads cover bytes 0..48 of the chunk and
  // are guarded against row.len() or staged through a zero-filled
  // 24-u16 copy.
  unsafe {
    let (v0, v1, v2) = if base + 24 <= row.len() {
      (
        _mm_loadu_si128(row.as_ptr().add(base).cast()),
        _mm_loadu_si128(row.as_ptr().add(base + 8).cast()),
        _mm_loadu_si128(row.as_ptr().add(base + 16).cast()),
      )
    } else {
      let mut sbuf = [0u16; 24];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      (
        _mm_loadu_si128(sbuf.as_ptr().cast()),
        _mm_loadu_si128(sbuf.as_ptr().add(8).cast()),
        _mm_loadu_si128(sbuf.as_ptr().add(16).cast()),
      )
    };
    let w = _mm_loadu_si128(w8.as_ptr().cast());
    for ch in 0..3 {
      let gathered = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(v0, m0[ch]), _mm_shuffle_epi8(v1, m1[ch])),
        _mm_shuffle_epi8(v2, m2[ch]),
      );
      acc[ch] = mac128_u16x8_u64(acc[ch], gathered, w);
    }
  }
}

/// 16-bit-element H-pass (3-channel interleaved RGB).
///
/// Unlike the u8 c3, the wide-lane "groups-packed-into-128-lanes"
/// deinterleave does not extend to u16: an 8-pixel `u16` group is 48
/// bytes (three 16-byte loads), not 24, so it cannot share the u8
/// single-`_mm512_shuffle_epi8`-per-lane trick. A 512-bit c3 wide path
/// is therefore impractical, and this kernel uses the proven 128-bit
/// u16 step exclusively — looping over 8-tap chunks through
/// [`c3_group_u16`], which mirrors the validated SSE4.1 u16 c3 (three
/// overlapping loads + nine `_mm_shuffle_epi8` masks + a
/// u64-accumulating mac).
///
/// # Safety
///
/// As [`area_h_reduce_row_u16_c1`], with `row.len() >= cells * 3` `u16`
/// elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_h_reduce_row_u16_c3(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  let (m0, m1, m2) = c3_masks_u16();
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: every 8-tap chunk delegates to `c3_group_u16`, which
    // guards its three 16-byte loads against row.len() and stages the
    // row end; weights come from the 8-multiple arena slice.
    unsafe {
      let mut acc = [_mm_setzero_si128(); 3];
      let mut t = 0usize;
      while t + 8 <= len {
        c3_group_u16(&mut acc, row, start + t, &span[t..], &m0, &m1, &m2);
        t += 8;
      }
      for ch in 0..3 {
        h_tmp[j * 3 + ch] = hsum128_u64(acc[ch]);
      }
    }
  }
}

/// 16-bit-element V-pass AXPY: `acc[i] += w * h_tmp[i]` with `h_tmp`
/// already `u64`. The `u32 * u64 -> u64` product splits `h_tmp` into
/// 32-bit halves — `_mm512_mul_epu32` gives `w * lo` and `w * hi`, the
/// latter shifted up 32 — summed mod 2^64 (exact by the engine bound).
/// Eight elements per iteration; the tail falls to scalar. Mirrors the
/// SSE4.1 u16 V-pass widened to 512 bits.
///
/// # Safety
///
/// AVX-512F must be available. `h_tmp.len() >= acc.len()`; every
/// product-sum stays within u64.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_v_accumulate_u16(acc: &mut [u64], h_tmp: &[u64], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm512_set1_epi64(i64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 8 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 8 <= n {
      let t = _mm512_loadu_si512(h_tmp.as_ptr().add(i).cast());
      let prod_lo = _mm512_mul_epu32(t, wv);
      let prod_hi = _mm512_mul_epu32(_mm512_srli_epi64::<32>(t), wv);
      let prod = _mm512_add_epi64(prod_lo, _mm512_slli_epi64::<32>(prod_hi));
      let a = _mm512_loadu_si512(acc.as_ptr().add(i).cast());
      _mm512_storeu_si512(acc.as_mut_ptr().add(i).cast(), _mm512_add_epi64(a, prod));
      i += 8;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * h_tmp[k];
  }
}

/// Widens eight `u16` arena weights to one `__m512d` of eight `f64`
/// lanes `(w0..w7)`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn widen_w16_f64(w: __m128i) -> __m512d {
  _mm512_cvtepi32_pd(_mm256_cvtepu16_epi32(w))
}

/// Accumulates eight `f32` samples (`s` lanes 0-7) against eight widened
/// `f64` weights into a running `__m512d`. A separate multiply then add
/// — since the integer-weight times f32-sample product is exact in `f64`
/// the fused and unfused forms agree, so this matches the H-pass scalar
/// reference up to the tap-sum order.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac8_f32(acc: __m512d, s: __m256, wf: __m512d) -> __m512d {
  // Zero the sample lanes whose weight is zero (arena padding) before
  // the multiply: the integer kernels lean on `0 * sample == 0`, but
  // `0.0 * NaN` and `0.0 * inf` are NaN, so a direct-loaded padding lane
  // holding a non-finite neighbor would otherwise poison the span.
  let keep = _mm512_cmp_pd_mask::<_CMP_NEQ_OQ>(wf, _mm512_setzero_pd());
  let sf = _mm512_maskz_mov_pd(keep, _mm512_cvtps_pd(s));
  _mm512_add_pd(acc, _mm512_mul_pd(sf, wf))
}

/// Deinterleaves four interleaved RGB `f32` pixels (`x = R0 G0 B0 R1`,
/// `y = G1 B1 R2 G2`, `z = B2 R3 G3 B3`) into planar
/// `(R0..R3, G0..G3, B0..B3)` via `_mm_shuffle_ps`. The 128-bit SSE4.1
/// step, run twice per 8-pixel chunk (the RGB group does not pack into
/// wide lanes, just as the u8 c3 runs its 128-bit gather per tier).
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

/// Float-element H-pass (1 channel): `f32` samples widen to `f64` and
/// meet the `u16` weights widened to `f64`; the per-span sums live in
/// `f64` (see the scalar reference for why the float accumulators are
/// `f64`). All eight taps of a chunk fit one `__m512d`. The products are
/// exact (a `u16` weight times a 24-bit `f32` mantissa fits 53 bits), so
/// the only departure from the scalar reference is the tap-sum order —
/// float addition does not associate, so parity is a small tolerance,
/// not the integer kernels' bit-exactness.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `f32` elements
/// and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
      let mut acc = _mm512_setzero_pd();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let base = start + ci * 8;
        let s = if base + 8 <= row.len() {
          _mm256_loadu_ps(row.as_ptr().add(base))
        } else {
          let mut sbuf = [0f32; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          _mm256_loadu_ps(sbuf.as_ptr())
        };
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        acc = mac8_f32(acc, s, widen_w16_f64(w));
      }
      h_tmp[j] = _mm512_reduce_add_pd(acc);
    }
  }
}

/// Float-element H-pass (3-channel interleaved RGB): two overlapping
/// four-pixel `_mm_shuffle_ps` deinterleaves cover the eight-pixel chunk
/// (the RGB group stays 128-bit — like the u8 c3, which runs its 128-bit
/// gather per tier), each channel's eight planar samples joined into a
/// `__m256` and sharing one widened weight set.
///
/// # Safety
///
/// As [`area_h_reduce_row_f32_c1`], with `row.len() >= cells * 3` `f32`
/// elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_h_reduce_row_f32_c3(
  row: &[f32],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: the six 16-byte loads cover the chunk's 24 f32 and are
    // either fully in-bounds (guarded) or staged through a zero-filled
    // 24-element stack copy; weights come from the 8-multiple arena
    // slice.
    unsafe {
      let mut acc0 = _mm512_setzero_pd();
      let mut acc1 = _mm512_setzero_pd();
      let mut acc2 = _mm512_setzero_pd();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let base = (start + ci * 8) * 3;
        let mut sbuf = [0f32; 24];
        let p = if base + 24 <= row.len() {
          row.as_ptr().add(base)
        } else {
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          sbuf.as_ptr()
        };
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
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        let wf = widen_w16_f64(w);
        acc0 = mac8_f32(acc0, _mm256_set_m128(r1, r0), wf);
        acc1 = mac8_f32(acc1, _mm256_set_m128(g1, g0), wf);
        acc2 = mac8_f32(acc2, _mm256_set_m128(b1, b0), wf);
      }
      h_tmp[j * 3] = _mm512_reduce_add_pd(acc0);
      h_tmp[j * 3 + 1] = _mm512_reduce_add_pd(acc1);
      h_tmp[j * 3 + 2] = _mm512_reduce_add_pd(acc2);
    }
  }
}

/// Float-element V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64`. A
/// separate multiply then add (not a fused multiply-add) so each lane
/// matches the scalar reference bit-for-bit — the V-pass is
/// element-wise, with no reordering. Eight elements per iteration.
///
/// # Safety
///
/// AVX-512F must be available. `h_tmp.len() >= acc.len()`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_v_accumulate_f32(acc: &mut [f64], h_tmp: &[f64], w: f64) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm512_set1_pd(w);
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
    acc[k] += w * h_tmp[k];
  }
}
