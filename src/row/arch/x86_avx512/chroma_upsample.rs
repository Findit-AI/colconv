//! AVX-512 (F + BW) centered (#302 phase-0.5) 2:1 horizontal
//! chroma-upsample reconstruct kernels — the SIMD twins of the scalar
//! [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h)
//! family. The u8 and u16 planar kernels run 512-bit arithmetic and narrow
//! back with the single-instruction cross-lane truncating converts
//! (`_mm512_cvtepi16_epi8` / `_mm512_cvtepi32_epi16`, exact because each
//! result fits the destination), then reassemble the interleaved output with
//! the AVX2 256-bit combine idiom. The interleaved-UV P-format kernel reuses
//! the 128-bit SSE4.1 core (its de-interleave / re-interleave does not widen
//! cleanly, and SSE4.1 is always available under AVX-512). Boundary columns
//! (`j = 0`, `j = half-1`) reuse the shared scalar per-sample reference so
//! the edges stay byte-identical.

#![cfg_attr(
  not(all(
    any(feature = "std", feature = "alloc"),
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  )),
  allow(dead_code)
)]

#[cfg(target_arch = "x86_64")]
#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::scalar;

/// Per-128-bit-lane byte-swap mask for eight `u16` pairs (`_mm256_shuffle_epi8`).
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn bswap16_mask256() -> __m256i {
  _mm256_set_epi8(
    14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5,
    2, 3, 0, 1,
  )
}

/// Interleaves two 32-lane u8 vectors (`even`, `odd`) into 64 sequential
/// bytes `[e0,o0,e1,o1,…]`, returned as `(out0, out1)`. Uses the AVX2
/// per-128-lane zip plus the `0x20/0x31` 128-bit combine.
#[cfg(any(
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuva"
))]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn zip_u8x32(even: __m256i, odd: __m256i) -> (__m256i, __m256i) {
  let lo = _mm256_unpacklo_epi8(even, odd);
  let hi = _mm256_unpackhi_epi8(even, odd);
  (
    _mm256_permute2x128_si256::<0x20>(lo, hi),
    _mm256_permute2x128_si256::<0x31>(lo, hi),
  )
}

/// Interleaves two 16-lane u16 vectors (`even`, `odd`) into 32 sequential
/// `u16` `[e0,o0,e1,o1,…]`, returned as `(out0, out1)`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn zip_u16x16(even: __m256i, odd: __m256i) -> (__m256i, __m256i) {
  let lo = _mm256_unpacklo_epi16(even, odd);
  let hi = _mm256_unpackhi_epi16(even, odd);
  (
    _mm256_permute2x128_si256::<0x20>(lo, hi),
    _mm256_permute2x128_si256::<0x31>(lo, hi),
  )
}

/// AVX-512 u8 centered 2:1 horizontal chroma upsample.
///
/// Block size: 32 chroma samples / iter (→ 64 output columns).
///
/// # Safety
///
/// AVX-512F + BW must be available. `width` even; `c_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  scalar::chroma_upsample_2to1_center_h_pair(c_half, c_full, 0, half);
  if half == 1 {
    return;
  }

  let mut j = 1usize;
  // SAFETY: `j + 32 < half` keeps every offset load inside `c_half[0..half]`
  // and every 64-byte store inside `c_full[0..width]`.
  unsafe {
    let two = _mm512_set1_epi16(2);
    let three = _mm512_set1_epi16(3);
    while j + 32 < half {
      let mid = _mm512_cvtepu8_epi16(_mm256_loadu_si256(c_half.as_ptr().add(j).cast()));
      let left = _mm512_cvtepu8_epi16(_mm256_loadu_si256(c_half.as_ptr().add(j - 1).cast()));
      let right = _mm512_cvtepu8_epi16(_mm256_loadu_si256(c_half.as_ptr().add(j + 1).cast()));
      let tm = _mm512_mullo_epi16(mid, three);
      let even = _mm512_srli_epi16::<2>(_mm512_add_epi16(_mm512_add_epi16(left, tm), two));
      let odd = _mm512_srli_epi16::<2>(_mm512_add_epi16(_mm512_add_epi16(tm, right), two));
      let (out0, out1) = zip_u8x32(_mm512_cvtepi16_epi8(even), _mm512_cvtepi16_epi8(odd));
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j).cast(), out0);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j + 32).cast(), out1);
      j += 32;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_pair(c_half, c_full, j, half);
    j += 1;
  }
}

/// AVX-512 u16 centered 2:1 horizontal chroma upsample.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// AVX-512F + BW must be available. `width` even; `c_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // SAFETY: `j + 16 < half` keeps every offset load inside `c_half[0..half]`
  // and every 32-`u16` store inside `c_full[0..width]`.
  unsafe {
    let bmask = bswap16_mask256();
    let mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    let two = _mm512_set1_epi32(2);
    let three = _mm512_set1_epi32(3);
    let norm = |ptr_off: usize| {
      let raw = _mm256_loadu_si256(c_half.as_ptr().add(ptr_off).cast());
      let host = if swap {
        _mm256_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm512_cvtepu16_epi32(_mm256_and_si256(host, mask))
    };
    while j + 16 < half {
      let mid = norm(j);
      let left = norm(j - 1);
      let right = norm(j + 1);
      let tm = _mm512_mullo_epi32(mid, three);
      let even32 = _mm512_srli_epi32::<2>(_mm512_add_epi32(_mm512_add_epi32(left, tm), two));
      let odd32 = _mm512_srli_epi32::<2>(_mm512_add_epi32(_mm512_add_epi32(tm, right), two));
      let mut even = _mm512_cvtepi32_epi16(even32);
      let mut odd = _mm512_cvtepi32_epi16(odd32);
      if swap {
        even = _mm256_shuffle_epi8(even, bmask);
        odd = _mm256_shuffle_epi8(odd, bmask);
      }
      let (out0, out1) = zip_u16x16(even, odd);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j).cast(), out0);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j + 16).cast(), out1);
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_u16_pair::<BITS>(c_half, c_full, j, half, big_endian);
    j += 1;
  }
}

/// AVX-512 semi-planar P-format centered 2:1 horizontal chroma upsample.
///
/// Delegates to the 128-bit SSE4.1 kernel (see the AVX2 twin): the
/// interleaved layout does not widen cleanly, and SSE4.1 is always available
/// under AVX-512. Bit-identical to the scalar reference per tier.
///
/// # Safety
///
/// AVX-512 (hence SSE4.1) must be available. `width` even;
/// `uv_half.len() >= width`; `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn chroma_upsample_2to1_center_h_p0xx_row<
  const BITS: u32,
  const LOW_PACKED: bool,
>(
  uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  // SAFETY: SSE4.1 ⊆ AVX-512; the delegate carries the same slice contract.
  unsafe {
    crate::row::arch::x86_sse41::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<
      BITS,
      LOW_PACKED,
    >(uv_half, uv_full, width, big_endian);
  }
}

/// AVX-512 u8 bottom-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottom_even_h`](crate::row::scalar::chroma_upsample_420_bottom_even_h).
/// Each 32-sample offset's vertical box average `e = (prev + cur + 1) >> 1` is
/// one `_mm256_avg_epu8`, widened to u16x32 and fed into the same centered
/// `1/4`–`3/4` blend as the horizontal sibling; boundary columns reuse the
/// shared scalar pair.
///
/// Block size: 32 chroma samples / iter (→ 64 output columns).
///
/// # Safety
///
/// AVX-512F + BW must be available. `width` even; `prev_half.len() >= width / 2`;
/// `cur_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // SAFETY: `j + 32 < half` keeps every offset load inside the half rows and
  // every 64-byte store inside `c_full[0..width]`.
  unsafe {
    let two = _mm512_set1_epi16(2);
    let three = _mm512_set1_epi16(3);
    let avg = |off: usize| {
      _mm512_cvtepu8_epi16(_mm256_avg_epu8(
        _mm256_loadu_si256(prev_half.as_ptr().add(off).cast()),
        _mm256_loadu_si256(cur_half.as_ptr().add(off).cast()),
      ))
    };
    while j + 32 < half {
      let e_left = avg(j - 1);
      let e_mid = avg(j);
      let e_right = avg(j + 1);
      let tm = _mm512_mullo_epi16(e_mid, three);
      let even = _mm512_srli_epi16::<2>(_mm512_add_epi16(_mm512_add_epi16(e_left, tm), two));
      let odd = _mm512_srli_epi16::<2>(_mm512_add_epi16(_mm512_add_epi16(tm, e_right), two));
      let (out0, out1) = zip_u8x32(_mm512_cvtepi16_epi8(even), _mm512_cvtepi16_epi8(odd));
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j).cast(), out0);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j + 32).cast(), out1);
      j += 32;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottom_even_h_pair(prev_half, cur_half, c_full, j, half);
    j += 1;
  }
}

/// AVX-512 u8 bottom-left-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottomleft_even_h`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h).
/// The co-sited (`h = 0`) horizontal phase is a plain 2× replicate:
/// `e = _mm256_avg_epu8(prev, cur)`, self-interleaved via [`zip_u8x32`].
///
/// Block size: 32 chroma samples / iter (→ 64 output columns).
///
/// # Safety
///
/// AVX-512F + BW must be available. `width` even; `prev_half.len() >= width / 2`;
/// `cur_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // SAFETY: `j + 32 <= half` keeps every load inside the half rows and every
  // 64-byte store inside `c_full[0..width]`.
  unsafe {
    while j + 32 <= half {
      let e = _mm256_avg_epu8(
        _mm256_loadu_si256(prev_half.as_ptr().add(j).cast()),
        _mm256_loadu_si256(cur_half.as_ptr().add(j).cast()),
      );
      let (out0, out1) = zip_u8x32(e, e);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j).cast(), out0);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j + 32).cast(), out1);
      j += 32;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottomleft_even_h_pair(prev_half, cur_half, c_full, j);
    j += 1;
  }
}

/// AVX-512 u16 bottom-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottom_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottom_even_h_u16).
/// Each 16-sample offset's `e` is `_mm256_avg_epu16` of the two masked,
/// host-normalized rows, widened to u32x16 and fed into the same centered blend
/// as the horizontal sibling, then re-encoded to wire order.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// AVX-512F + BW must be available. `width` even; `prev_half.len() >= width / 2`;
/// `cur_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // SAFETY: `j + 16 < half` keeps every offset load inside the half rows and
  // every 32-`u16` store inside `c_full[0..width]`.
  unsafe {
    let bmask = bswap16_mask256();
    let mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    let two = _mm512_set1_epi32(2);
    let three = _mm512_set1_epi32(3);
    let norm16 = |row: &[u16], off: usize| {
      let raw = _mm256_loadu_si256(row.as_ptr().add(off).cast());
      let host = if swap {
        _mm256_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm256_and_si256(host, mask)
    };
    let vavg = |off: usize| {
      _mm512_cvtepu16_epi32(_mm256_avg_epu16(
        norm16(prev_half, off),
        norm16(cur_half, off),
      ))
    };
    while j + 16 < half {
      let e_left = vavg(j - 1);
      let e_mid = vavg(j);
      let e_right = vavg(j + 1);
      let tm = _mm512_mullo_epi32(e_mid, three);
      let even32 = _mm512_srli_epi32::<2>(_mm512_add_epi32(_mm512_add_epi32(e_left, tm), two));
      let odd32 = _mm512_srli_epi32::<2>(_mm512_add_epi32(_mm512_add_epi32(tm, e_right), two));
      let mut even = _mm512_cvtepi32_epi16(even32);
      let mut odd = _mm512_cvtepi32_epi16(odd32);
      if swap {
        even = _mm256_shuffle_epi8(even, bmask);
        odd = _mm256_shuffle_epi8(odd, bmask);
      }
      let (out0, out1) = zip_u16x16(even, odd);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j).cast(), out0);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j + 16).cast(), out1);
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottom_even_h_u16_pair::<BITS>(
      prev_half, cur_half, c_full, j, half, big_endian,
    );
    j += 1;
  }
}

/// AVX-512 u16 bottom-left-sited even-row 4:2:0 chroma upsample — the SIMD twin
/// of
/// [`chroma_upsample_420_bottomleft_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16).
/// Per-column `e = _mm256_avg_epu16` of the masked, host-normalized rows,
/// re-encoded to wire order and replicated across the column pair via
/// [`zip_u16x16`].
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// AVX-512F + BW must be available. `width` even; `prev_half.len() >= width / 2`;
/// `cur_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // SAFETY: `j + 16 <= half` keeps every load inside the half rows and every
  // 32-`u16` store inside `c_full[0..width]`.
  unsafe {
    let bmask = bswap16_mask256();
    let mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    let norm16 = |row: &[u16], off: usize| {
      let raw = _mm256_loadu_si256(row.as_ptr().add(off).cast());
      let host = if swap {
        _mm256_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm256_and_si256(host, mask)
    };
    while j + 16 <= half {
      let e = _mm256_avg_epu16(norm16(prev_half, j), norm16(cur_half, j));
      let e = if swap {
        _mm256_shuffle_epi8(e, bmask)
      } else {
        e
      };
      let (out0, out1) = zip_u16x16(e, e);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j).cast(), out0);
      _mm256_storeu_si256(c_full.as_mut_ptr().add(2 * j + 16).cast(), out1);
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottomleft_even_h_u16_pair::<BITS>(
      prev_half, cur_half, c_full, j, big_endian,
    );
    j += 1;
  }
}

/// AVX-512 semi-planar P-format bottom-sited even-row 4:2:0 chroma upsample.
/// Delegates to the 128-bit SSE4.1 kernel (see the centered twin): the
/// interleaved layout does not widen cleanly, and SSE4.1 is always available
/// under AVX-512.
///
/// # Safety
///
/// AVX-512 (hence SSE4.1) must be available. `width` even;
/// `prev_uv_half.len() >= width`; `cur_uv_half.len() >= width`;
/// `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn chroma_upsample_420_bottom_even_h_p0xx_row<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  // SAFETY: SSE4.1 ⊆ AVX-512; the delegate carries the same slice contract.
  unsafe {
    crate::row::arch::x86_sse41::chroma_upsample::chroma_upsample_420_bottom_even_h_p0xx_row::<BITS>(
      prev_uv_half,
      cur_uv_half,
      uv_full,
      width,
      big_endian,
    );
  }
}

/// AVX-512 semi-planar P-format bottom-left-sited even-row 4:2:0 chroma upsample.
/// Delegates to the 128-bit SSE4.1 kernel (see the centered twin).
///
/// # Safety
///
/// AVX-512 (hence SSE4.1) must be available. `width` even;
/// `prev_uv_half.len() >= width`; `cur_uv_half.len() >= width`;
/// `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn chroma_upsample_420_bottomleft_even_h_p0xx_row<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  // SAFETY: SSE4.1 ⊆ AVX-512; the delegate carries the same slice contract.
  unsafe {
    crate::row::arch::x86_sse41::chroma_upsample::chroma_upsample_420_bottomleft_even_h_p0xx_row::<
      BITS,
    >(prev_uv_half, cur_uv_half, uv_full, width, big_endian);
  }
}

/// AVX-512 full-width vertical chroma rounding-average for the **bottom-sited**
/// even output luma row of a 4:4:0 source. Byte-identical to
/// [`chroma_upsample_440_bottom_v`](crate::row::scalar::chroma_upsample_440_bottom_v):
/// `out[j] = (prev[j] + cur[j] + 1) >> 1` via `_mm512_avg_epu8` (64 lanes/iter)
/// with a scalar tail.
///
/// # Safety
///
/// AVX-512F + BW must be available. `prev.len() >= width`; `cur.len() >= width`;
/// `out.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // SAFETY: each iteration reads/writes 64 bytes at offset `j` with
  // `j + 64 <= width <= len`, so every access stays in bounds.
  unsafe {
    while j + 64 <= width {
      let p = _mm512_loadu_si512(prev.as_ptr().add(j).cast());
      let c = _mm512_loadu_si512(cur.as_ptr().add(j).cast());
      _mm512_storeu_si512(out.as_mut_ptr().add(j).cast(), _mm512_avg_epu8(p, c));
      j += 64;
    }
  }
  while j < width {
    out[j] = (((prev[j] as u16) + (cur[j] as u16) + 1) >> 1) as u8;
    j += 1;
  }
}

/// AVX-512 `u16` twin of [`chroma_upsample_440_bottom_v_row`] for the high-bit
/// planar 4:4:0 sink, byte-identical to
/// [`chroma_upsample_440_bottom_v_u16_wire`](crate::row::scalar::chroma_upsample_440_bottom_v_u16_wire).
/// Uses the proven 256-bit `_mm256_avg_epu16` + `_mm256_shuffle_epi8` normalize
/// path (16 `u16` lanes/iter) — the same width the AVX-512 u16 planar kernels
/// use for their wire ↔ host reorder — with a scalar tail.
///
/// # Safety
///
/// AVX-512F + BW must be available. `prev.len() >= width`; `cur.len() >= width`;
/// `out.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
  // SAFETY: each iteration reads/writes 16 u16 lanes at offset `j` with
  // `j + 16 <= width <= len`, so every access stays in bounds.
  unsafe {
    let bmask = bswap16_mask256();
    let mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    let norm = |raw: __m256i| {
      let host = if swap {
        _mm256_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm256_and_si256(host, mask)
    };
    while j + 16 <= width {
      let p = norm(_mm256_loadu_si256(prev.as_ptr().add(j).cast()));
      let c = norm(_mm256_loadu_si256(cur.as_ptr().add(j).cast()));
      let avg = _mm256_avg_epu16(p, c);
      let enc = if swap {
        _mm256_shuffle_epi8(avg, bmask)
      } else {
        avg
      };
      _mm256_storeu_si256(out.as_mut_ptr().add(j).cast(), enc);
      j += 16;
    }
  }
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

/// AVX-512 u8 centered 1→4 horizontal chroma upsample. Delegates to the 128-bit
/// SSE4.1 kernel (see the AVX2 twin): the 4-way byte interleave does not widen
/// cleanly past 128-bit lanes, and SSE4.1 is always available under AVX-512.
/// Bit-identical to the scalar reference per tier.
///
/// # Safety
///
/// AVX-512 (hence SSE4.1) must be available.
/// `c_quarter.len() >= width.div_ceil(4)`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn chroma_upsample_4to1_center_h_row(
  c_quarter: &[u8],
  c_full: &mut [u8],
  width: usize,
) {
  // SAFETY: SSE4.1 ⊆ AVX-512; the delegate carries the same slice contract.
  unsafe {
    crate::row::arch::x86_sse41::chroma_upsample::chroma_upsample_4to1_center_h_row(
      c_quarter, c_full, width,
    );
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

  const WIDTHS: &[usize] = &[2, 4, 6, 8, 16, 18, 30, 32, 34, 62, 64, 66, 128, 130];

  fn have_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
      && std::arch::is_x86_feature_detected!("avx512bw")
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx512_u8_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
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
  fn avx512_u16_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
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
  fn avx512_p0xx_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
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
  fn avx512_u8_vertical_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
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
  fn avx512_u16_vertical_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
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
  fn avx512_p0xx_vertical_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
    check_p0xx_vertical::<10>(false);
    check_p0xx_vertical::<10>(true);
    check_p0xx_vertical::<12>(false);
    check_p0xx_vertical::<12>(true);
    check_p0xx_vertical::<16>(false);
    check_p0xx_vertical::<16>(true);
  }
}

#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
mod tests_planar {
  use crate::row::scalar;

  fn pseudo_random_u8(out: &mut [u8], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 16) as u8;
    }
  }

  fn pseudo_random_u16(out: &mut [u16], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 8) as u16;
    }
  }

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
  fn avx512_440_bottom_v_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
    for &w in WIDTHS_440 {
      let mut prev = std::vec![0u8; w];
      let mut cur = std::vec![0u8; w];
      pseudo_random_u8(&mut prev, 0x440B);
      pseudo_random_u8(&mut cur, 0x440C);
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
  fn avx512_440_bottom_v_u16_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
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
  fn avx512_4to1_center_h_matches_scalar_widths() {
    if !have_avx512() {
      return;
    }
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
