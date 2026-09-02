//! SSE4.1 fused-downscale H-pass: per output span, widening
//! multiply-accumulate over the plan-time zero-padded u16 weight
//! arena (see the NEON sibling for the arena contract).
//!
//! Per 8-tap chunk: 8 samples zero-extend `u8 -> u16`
//! (`_mm_cvtepu8_epi16`) and meet 8 arena weights through the
//! `_mm_mullo_epi16` / `_mm_mulhi_epu16` pair, whose halves
//! interleave (`_mm_unpacklo/hi_epi16`) into exact u32 products.
//! Padding lanes multiply by zero, so sample loads only stage through
//! a stack copy at the row-end boundary. 3-channel rows deinterleave
//! with paired `_mm_shuffle_epi8` masks over two overlapping 16-byte
//! loads. Bit-identical to the scalar reference by integer
//! associativity.

#![cfg_attr(not(feature = "std"), allow(dead_code))]
#![cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]

#[cfg(target_arch = "x86_64")]
#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

/// Sums the four u32 lanes of `acc`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn hsum_u32(acc: __m128i) -> u32 {
  let hi = _mm_shuffle_epi32::<0b01_00_11_10>(acc);
  let s = _mm_add_epi32(acc, hi);
  let s2 = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
  _mm_cvtsi128_si32(s2) as u32
}

/// Accumulates the eight exact u32 products of `s16 * w` into `acc`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mac_u16x8(acc: __m128i, s16: __m128i, w: __m128i) -> __m128i {
  let lo = _mm_mullo_epi16(s16, w);
  let hi = _mm_mulhi_epu16(s16, w);
  let acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(lo, hi));
  _mm_add_epi32(acc, _mm_unpackhi_epi16(lo, hi))
}

/// # Safety
///
/// SSE4.1 must be available. Caller guarantees the padded-arena
/// contract of the NEON sibling: `w16_off.len() == starts.len() + 1`
/// monotonic and bounded by `w16.len()`, span lengths multiples of 8,
/// `row.len() >= cells`, `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let mut acc = _mm_setzero_si128();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let base = start + ci * 8;
        let s8 = if base + 8 <= row.len() {
          _mm_loadl_epi64(row.as_ptr().add(base).cast())
        } else {
          let mut sbuf = [0u8; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          _mm_loadl_epi64(sbuf.as_ptr().cast())
        };
        let s16 = _mm_cvtepu8_epi16(s8);
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        acc = mac_u16x8(acc, s16, w);
      }
      h_tmp[j] = hsum_u32(acc);
    }
  }
}

/// 3-channel (interleaved RGB) variant: two overlapping 16-byte loads
/// cover each chunk's 24 bytes, and per-channel `_mm_shuffle_epi8`
/// mask pairs gather the eight samples of each channel.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn area_h_reduce_row_c3(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  // Byte g of a 24-byte chunk lives in lane g of the first load
  // (bytes 0..16) or lane g - 8 of the second (bytes 8..24); each
  // channel's eight samples (g = ch + 3t) gather from whichever load
  // holds them, with -1 zeroing the other mask's lane.
  let (m0, m1): ([__m128i; 3], [__m128i; 3]) = (
    [
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0),
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1),
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2),
    ],
    [
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, -1, -1, -1, -1, -1, -1,
      ),
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, -1, -1, -1, -1, -1,
      ),
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, -1, -1, -1, -1, -1,
      ),
    ],
  );
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: the two 16-byte loads cover bytes 0..24 of the chunk and
    // are either fully in-bounds (guarded against row.len()) or staged
    // through a zero-filled 24-byte stack copy; weight loads come from
    // the 8-multiple arena slice.
    unsafe {
      let mut acc = [_mm_setzero_si128(); 3];
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let base = (start + ci * 8) * 3;
        let mut sbuf = [0u8; 24];
        let (v0, v1) = if base + 24 <= row.len() {
          (
            _mm_loadu_si128(row.as_ptr().add(base).cast()),
            _mm_loadu_si128(row.as_ptr().add(base + 8).cast()),
          )
        } else {
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          (
            _mm_loadu_si128(sbuf.as_ptr().cast()),
            _mm_loadu_si128(sbuf.as_ptr().add(8).cast()),
          )
        };
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        for ch in 0..3 {
          let gathered = _mm_or_si128(_mm_shuffle_epi8(v0, m0[ch]), _mm_shuffle_epi8(v1, m1[ch]));
          acc[ch] = mac_u16x8(acc[ch], _mm_cvtepu8_epi16(gathered), w);
        }
      }
      for ch in 0..3 {
        h_tmp[j * 3 + ch] = hsum_u32(acc[ch]);
      }
    }
  }
}

/// V-pass AXPY: `acc[i] += w * h_tmp[i]`, exact u64 lanes via
/// `_mm_mul_epu32` over even/odd u32 lanes re-paired with
/// `_mm_unpacklo/hi_epi64` (4 elements per iteration).
///
/// # Safety
///
/// SSE4.1 must be available. `h_tmp.len() >= acc.len()`; every
/// product-sum stays within u64 (the engine's denominator bound).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  // Each 64-bit lane's low half holds `w` (high half zero), the form
  // `_mm_mul_epu32` consumes.
  let wv = _mm_set1_epi64x(i64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 4 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 4 <= n {
      let t = _mm_loadu_si128(h_tmp.as_ptr().add(i).cast());
      let even = _mm_mul_epu32(t, wv);
      let odd = _mm_mul_epu32(_mm_srli_epi64::<32>(t), wv);
      let a01 = _mm_loadu_si128(acc.as_ptr().add(i).cast());
      let a23 = _mm_loadu_si128(acc.as_ptr().add(i + 2).cast());
      let p01 = _mm_unpacklo_epi64(even, odd);
      let p23 = _mm_unpackhi_epi64(even, odd);
      _mm_storeu_si128(acc.as_mut_ptr().add(i).cast(), _mm_add_epi64(a01, p01));
      _mm_storeu_si128(acc.as_mut_ptr().add(i + 2).cast(), _mm_add_epi64(a23, p23));
      i += 4;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * u64::from(h_tmp[k]);
  }
}

/// Sums the two u64 lanes of `acc`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn hsum_u64(acc: __m128i) -> u64 {
  let hi = _mm_unpackhi_epi64(acc, acc);
  _mm_cvtsi128_si64(_mm_add_epi64(acc, hi)) as u64
}

/// Accumulates the eight exact `u16 * u16 -> u32` products of `s16 * w`
/// into the two u64 lanes of `acc` (lanes 0/1 hold the even/odd product
/// partial sums; a span total is their sum). Unlike the u8
/// [`mac_u16x8`], a `u16` span sum overflows `u32`, so the products
/// widen to u64 before accumulating.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mac_u16x8_u64(acc: __m128i, s16: __m128i, w: __m128i) -> __m128i {
  let lo = _mm_mullo_epi16(s16, w);
  let hi = _mm_mulhi_epu16(s16, w);
  let p_lo = _mm_unpacklo_epi16(lo, hi);
  let p_hi = _mm_unpackhi_epi16(lo, hi);
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(p_lo));
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(_mm_srli_si128::<8>(p_lo)));
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(p_hi));
  _mm_add_epi64(acc, _mm_cvtepu32_epi64(_mm_srli_si128::<8>(p_hi)))
}

/// 16-bit-element H-pass (1 channel): like [`area_h_reduce_row_c1`] but
/// the samples load directly as `u16` and the products accumulate in
/// `u64` lanes (a single span sum can exceed `u32`).
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `u16`
/// elements and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let mut acc = _mm_setzero_si128();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let base = start + ci * 8;
        let s16 = if base + 8 <= row.len() {
          _mm_loadu_si128(row.as_ptr().add(base).cast())
        } else {
          let mut sbuf = [0u16; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          _mm_loadu_si128(sbuf.as_ptr().cast())
        };
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        acc = mac_u16x8_u64(acc, s16, w);
      }
      h_tmp[j] = hsum_u64(acc);
    }
  }
}

/// 16-bit-element H-pass (3-channel interleaved RGB): each 8-tap chunk
/// spans 24 `u16` (48 bytes), so three overlapping 16-byte loads cover
/// it and a per-channel triple of `_mm_shuffle_epi8` masks gathers each
/// channel's eight samples (a `u16` lives in exactly one load, the
/// other two masks zeroing that lane).
///
/// # Safety
///
/// As [`area_h_reduce_row_u16_c1`], with `row.len() >= cells * 3`
/// `u16` elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn area_h_reduce_row_u16_c3(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  // For channel ch the eight samples sit at u16 index ch + 3t (byte
  // 2*(ch + 3t)), split across the three 16-byte loads; each mask pulls
  // its load's contributing u16 pairs into output lanes 0..7 and -1
  // zeroes the rest.
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
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: the three 16-byte loads cover bytes 0..48 of the chunk and
    // are either fully in-bounds (guarded) or staged through a
    // zero-filled 48-byte stack copy; weights come from the 8-multiple
    // arena slice.
    unsafe {
      let mut acc = [_mm_setzero_si128(); 3];
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let base = (start + ci * 8) * 3;
        let mut sbuf = [0u16; 24];
        let (v0, v1, v2) = if base + 24 <= row.len() {
          (
            _mm_loadu_si128(row.as_ptr().add(base).cast()),
            _mm_loadu_si128(row.as_ptr().add(base + 8).cast()),
            _mm_loadu_si128(row.as_ptr().add(base + 16).cast()),
          )
        } else {
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          (
            _mm_loadu_si128(sbuf.as_ptr().cast()),
            _mm_loadu_si128(sbuf.as_ptr().add(8).cast()),
            _mm_loadu_si128(sbuf.as_ptr().add(16).cast()),
          )
        };
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        for ch in 0..3 {
          let gathered = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(v0, m0[ch]), _mm_shuffle_epi8(v1, m1[ch])),
            _mm_shuffle_epi8(v2, m2[ch]),
          );
          acc[ch] = mac_u16x8_u64(acc[ch], gathered, w);
        }
      }
      for ch in 0..3 {
        h_tmp[j * 3 + ch] = hsum_u64(acc[ch]);
      }
    }
  }
}

/// 16-bit-element V-pass AXPY: `acc[i] += w * h_tmp[i]` with `h_tmp`
/// already `u64`. The `u32 * u64 -> u64` product splits `h_tmp` into
/// 32-bit halves — `_mm_mul_epu32` gives `w * lo` and `w * hi`, the
/// latter shifted up 32 — summed mod 2^64 (exact by the engine bound).
/// Two elements per iteration.
///
/// # Safety
///
/// SSE4.1 must be available. `h_tmp.len() >= acc.len()`; every
/// product-sum stays within u64.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn area_v_accumulate_u16(acc: &mut [u64], h_tmp: &[u64], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm_set1_epi64x(i64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 2 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 2 <= n {
      let t = _mm_loadu_si128(h_tmp.as_ptr().add(i).cast());
      let prod_lo = _mm_mul_epu32(t, wv);
      let prod_hi = _mm_mul_epu32(_mm_srli_epi64::<32>(t), wv);
      let prod = _mm_add_epi64(prod_lo, _mm_slli_epi64::<32>(prod_hi));
      let a = _mm_loadu_si128(acc.as_ptr().add(i).cast());
      _mm_storeu_si128(acc.as_mut_ptr().add(i).cast(), _mm_add_epi64(a, prod));
      i += 2;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * h_tmp[k];
  }
}

/// Sums the two f64 lanes of `v`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn hsum_pd(v: __m128d) -> f64 {
  _mm_cvtsd_f64(_mm_add_sd(v, _mm_unpackhi_pd(v, v)))
}

/// Widens eight `u16` arena weights to four `f64` lane-pairs
/// `(w0w1, w2w3, w4w5, w6w7)`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn widen_w16_f64(w: __m128i) -> (__m128d, __m128d, __m128d, __m128d) {
  let w_lo = _mm_cvtepu16_epi32(w);
  let w_hi = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(w));
  (
    _mm_cvtepi32_pd(w_lo),
    _mm_cvtepi32_pd(_mm_unpackhi_epi64(w_lo, w_lo)),
    _mm_cvtepi32_pd(w_hi),
    _mm_cvtepi32_pd(_mm_unpackhi_epi64(w_hi, w_hi)),
  )
}

/// Accumulates eight `f32` samples (`s_lo` lanes 0-3, `s_hi` lanes 4-7)
/// against four widened weight pairs. A separate multiply then add —
/// SSE4.1 has no fused multiply-add, and since the integer-weight times
/// f32-sample product is exact in f64 the two forms agree anyway.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mac8_f32(
  acc: __m128d,
  s_lo: __m128,
  s_hi: __m128,
  wf0: __m128d,
  wf1: __m128d,
  wf2: __m128d,
  wf3: __m128d,
) -> __m128d {
  let a = _mm_add_pd(acc, _mm_mul_pd(mask_pd(_mm_cvtps_pd(s_lo), wf0), wf0));
  let a = _mm_add_pd(
    a,
    _mm_mul_pd(mask_pd(_mm_cvtps_pd(_mm_movehl_ps(s_lo, s_lo)), wf1), wf1),
  );
  let a = _mm_add_pd(a, _mm_mul_pd(mask_pd(_mm_cvtps_pd(s_hi), wf2), wf2));
  _mm_add_pd(
    a,
    _mm_mul_pd(mask_pd(_mm_cvtps_pd(_mm_movehl_ps(s_hi, s_hi)), wf3), wf3),
  )
}

/// Zeroes the `f64` sample lanes whose weight lane is zero — the arena's
/// padding lanes. The integer kernels lean on `0 * sample == 0`, but
/// `0.0 * NaN` and `0.0 * inf` are NaN, so a direct-loaded padding lane
/// holding a non-finite neighbor would otherwise poison the span.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mask_pd(sf: __m128d, wf: __m128d) -> __m128d {
  _mm_and_pd(sf, _mm_cmpneq_pd(wf, _mm_setzero_pd()))
}

/// Deinterleaves four interleaved RGB `f32` pixels (`x = R0 G0 B0 R1`,
/// `y = G1 B1 R2 G2`, `z = B2 R3 G3 B3`) into planar
/// `(R0..R3, G0..G3, B0..B3)` via `_mm_shuffle_ps`.
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

/// Float-element H-pass (1 channel): `f32` samples widen to `f64` and
/// meet the `u16` weights widened to `f64`; the per-span sums live in
/// `f64`. The products are exact, so the only departure from the scalar
/// reference is the tap-sum order — float addition does not associate,
/// so parity is a small tolerance, not bit-exactness.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `f32`
/// elements and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let mut acc = _mm_setzero_pd();
      for (ci, chunk) in span.as_chunks::<8>().0.iter().enumerate() {
        let base = start + ci * 8;
        let (s_lo, s_hi) = if base + 8 <= row.len() {
          (
            _mm_loadu_ps(row.as_ptr().add(base)),
            _mm_loadu_ps(row.as_ptr().add(base + 4)),
          )
        } else {
          let mut sbuf = [0f32; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          (
            _mm_loadu_ps(sbuf.as_ptr()),
            _mm_loadu_ps(sbuf.as_ptr().add(4)),
          )
        };
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        let (wf0, wf1, wf2, wf3) = widen_w16_f64(w);
        acc = mac8_f32(acc, s_lo, s_hi, wf0, wf1, wf2, wf3);
      }
      h_tmp[j] = hsum_pd(acc);
    }
  }
}

/// Float-element H-pass (3-channel interleaved RGB): two overlapping
/// four-pixel `_mm_shuffle_ps` deinterleaves cover the eight-pixel
/// chunk, each channel sharing one widened weight set.
///
/// # Safety
///
/// As [`area_h_reduce_row_f32_c1`], with `row.len() >= cells * 3` `f32`
/// elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let mut acc0 = _mm_setzero_pd();
      let mut acc1 = _mm_setzero_pd();
      let mut acc2 = _mm_setzero_pd();
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
        let (wf0, wf1, wf2, wf3) = widen_w16_f64(w);
        acc0 = mac8_f32(acc0, r0, r1, wf0, wf1, wf2, wf3);
        acc1 = mac8_f32(acc1, g0, g1, wf0, wf1, wf2, wf3);
        acc2 = mac8_f32(acc2, b0, b1, wf0, wf1, wf2, wf3);
      }
      h_tmp[j * 3] = hsum_pd(acc0);
      h_tmp[j * 3 + 1] = hsum_pd(acc1);
      h_tmp[j * 3 + 2] = hsum_pd(acc2);
    }
  }
}

/// Float-element V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64`. A
/// separate multiply then add (not a fused multiply-add) so each lane
/// matches the scalar reference bit-for-bit — the V-pass is
/// element-wise, with no reordering. Two elements per iteration.
///
/// # Safety
///
/// SSE4.1 must be available. `h_tmp.len() >= acc.len()`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn area_v_accumulate_f32(acc: &mut [f64], h_tmp: &[f64], w: f64) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm_set1_pd(w);
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
    acc[k] += w * h_tmp[k];
  }
}
