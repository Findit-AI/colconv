//! AVX2 centered (#302 phase-0.5) 2:1 horizontal chroma-upsample
//! reconstruct kernels — the SIMD twins of the scalar
//! [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h)
//! family. The u8 and u16 planar kernels run genuine 256-bit arithmetic and
//! reassemble the interleaved output with the codebase's proven pack fixup
//! (`_mm256_permute4x64_epi64::<0xD8>`) and 128-bit-lane combine
//! (`_mm256_permute2x128_si256::<0x20/0x31>`) idioms. The interleaved-UV
//! P-format kernel reuses the 128-bit SSE4.1 core: its de-interleave /
//! re-interleave does not widen cleanly past 128 bits, and SSE4.1 is always
//! available under AVX2. The two boundary columns (`j = 0`, `j = half-1`)
//! reuse the shared scalar per-sample reference so the edges stay
//! byte-identical; the vector loop covers the strict interior.

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
#[target_feature(enable = "avx2")]
fn bswap16_mask() -> __m256i {
  _mm256_set_epi8(
    14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5,
    2, 3, 0, 1,
  )
}

/// AVX2 u8 centered 2:1 horizontal chroma upsample.
///
/// Block size: 32 chroma samples / iter (→ 64 output columns).
///
/// # Safety
///
/// AVX2 must be available. `width` even; `c_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx2")]
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
    let two = _mm256_set1_epi16(2);
    let three = _mm256_set1_epi16(3);
    while j + 32 < half {
      let mid = _mm256_loadu_si256(c_half.as_ptr().add(j).cast());
      let left = _mm256_loadu_si256(c_half.as_ptr().add(j - 1).cast());
      let right = _mm256_loadu_si256(c_half.as_ptr().add(j + 1).cast());
      // Widen each 32-u8 load to two 16-u16 halves (samples 0..15, 16..31).
      let widen_lo = |v: __m256i| _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v));
      let widen_hi = |v: __m256i| _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v));
      let blend = |a: __m256i, b3: __m256i| {
        _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(a, b3), two))
      };
      let mid_lo = widen_lo(mid);
      let mid_hi = widen_hi(mid);
      let tm_lo = _mm256_mullo_epi16(mid_lo, three);
      let tm_hi = _mm256_mullo_epi16(mid_hi, three);
      let even_lo = blend(widen_lo(left), tm_lo);
      let even_hi = blend(widen_hi(left), tm_hi);
      let odd_lo = blend(widen_lo(right), tm_lo);
      let odd_hi = blend(widen_hi(right), tm_hi);
      // Narrow (per-128-lane pack + `0xD8` fixup) → 32 sequential u8.
      let even = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(even_lo, even_hi));
      let odd = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(odd_lo, odd_hi));
      // Zip even/odd within lanes, then combine 128-bit halves in order.
      let lo = _mm256_unpacklo_epi8(even, odd);
      let hi = _mm256_unpackhi_epi8(even, odd);
      let out0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
      let out1 = _mm256_permute2x128_si256::<0x31>(lo, hi);
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

/// AVX2 u16 centered 2:1 horizontal chroma upsample.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// AVX2 must be available. `width` even; `c_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "avx2")]
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
    let bmask = bswap16_mask();
    let mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    let two = _mm256_set1_epi32(2);
    let three = _mm256_set1_epi32(3);
    let norm = |ptr_off: usize| {
      let raw = _mm256_loadu_si256(c_half.as_ptr().add(ptr_off).cast());
      let host = if swap {
        _mm256_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm256_and_si256(host, mask)
    };
    let widen_lo = |v: __m256i| _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
    let widen_hi = |v: __m256i| _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(v));
    let blend = |a: __m256i, b3: __m256i| {
      _mm256_srli_epi32::<2>(_mm256_add_epi32(_mm256_add_epi32(a, b3), two))
    };
    while j + 16 < half {
      let mid = norm(j);
      let left = norm(j - 1);
      let right = norm(j + 1);
      let tm_lo = _mm256_mullo_epi32(widen_lo(mid), three);
      let tm_hi = _mm256_mullo_epi32(widen_hi(mid), three);
      let even_lo = blend(widen_lo(left), tm_lo);
      let even_hi = blend(widen_hi(left), tm_hi);
      let odd_lo = blend(widen_lo(right), tm_lo);
      let odd_hi = blend(widen_hi(right), tm_hi);
      let even = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(even_lo, even_hi));
      let odd = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(odd_lo, odd_hi));
      let even = if swap {
        _mm256_shuffle_epi8(even, bmask)
      } else {
        even
      };
      let odd = if swap {
        _mm256_shuffle_epi8(odd, bmask)
      } else {
        odd
      };
      let lo = _mm256_unpacklo_epi16(even, odd);
      let hi = _mm256_unpackhi_epi16(even, odd);
      let out0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
      let out1 = _mm256_permute2x128_si256::<0x31>(lo, hi);
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

/// AVX2 semi-planar P-format centered 2:1 horizontal chroma upsample.
///
/// Delegates to the 128-bit SSE4.1 kernel: the interleaved half-row
/// de-interleave / re-interleave does not widen cleanly past 128 bits, and
/// SSE4.1 is always available under AVX2. Bit-identical to the scalar
/// reference per tier.
///
/// # Safety
///
/// AVX2 (hence SSE4.1) must be available. `width` even;
/// `uv_half.len() >= width`; `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn chroma_upsample_2to1_center_h_p0xx_row<
  const BITS: u32,
  const LOW_PACKED: bool,
>(
  uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  // SAFETY: SSE4.1 ⊆ AVX2; the delegate carries the same slice contract.
  unsafe {
    crate::row::arch::x86_sse41::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<
      BITS,
      LOW_PACKED,
    >(uv_half, uv_full, width, big_endian);
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

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_u8_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
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
  fn avx2_u16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
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
  fn avx2_p0xx_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
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
}
