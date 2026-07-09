//! SSE4.1 centered (#302 phase-0.5) 2:1 horizontal chroma-upsample
//! reconstruct kernels — the SIMD twins of the scalar
//! [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h)
//! family (u8 planar, u16 planar, interleaved-UV P-format). Each output
//! pair is the fixed-weight neighbour blend
//!
//! ```text
//!   even col 2j   → (c[j-1] + 3·c[j] + 2) >> 2   (c[-1]   clamps to c[0])
//!   odd  col 2j+1 → (3·c[j] + c[j+1] + 2) >> 2   (c[half] clamps to c[half-1])
//! ```
//!
//! computed with offset loads and an unpack-based interleaved store. The
//! two boundary columns (`j = 0`, `j = half-1`) — the only samples whose
//! neighbours clamp — reuse the shared scalar per-sample reference, so the
//! edges stay byte-identical; the vector loop covers the strict interior.

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

/// Byte-swap mask for reversing the two bytes of each of eight `u16` lanes
/// (`_mm_shuffle_epi8`). Used to normalize a wire-order load to host-native
/// (and back) when `big_endian != host_is_big_endian`.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "sse4.1")]
fn bswap16_mask() -> __m128i {
  _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1)
}

/// Blends eight interior chroma samples in `u32` lanes and narrows back to
/// `u16`: returns `(even, odd)` with `even = (left + 3·mid + 2) >> 2` and
/// `odd = (3·mid + right + 2) >> 2`. Inputs are logical `u16x8`; the
/// widen-to-`u32` avoids the `BITS = 16` overflow and `_mm_packus_epi32`
/// narrows exactly (each result `≤ 65535`).
///
/// # Safety
///
/// SSE4.1 must be available.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "sse4.1")]
fn blend_u16x8(left: __m128i, mid: __m128i, right: __m128i) -> (__m128i, __m128i) {
  let two = _mm_set1_epi32(2);
  let three = _mm_set1_epi32(3);
  let widen_lo = |v: __m128i| _mm_cvtepu16_epi32(v);
  let widen_hi = |v: __m128i| _mm_cvtepu16_epi32(_mm_srli_si128::<8>(v));
  let mid_lo = widen_lo(mid);
  let mid_hi = widen_hi(mid);
  let left_lo = widen_lo(left);
  let left_hi = widen_hi(left);
  let right_lo = widen_lo(right);
  let right_hi = widen_hi(right);
  let tm_lo = _mm_mullo_epi32(mid_lo, three);
  let tm_hi = _mm_mullo_epi32(mid_hi, three);
  let even_lo = _mm_srli_epi32::<2>(_mm_add_epi32(_mm_add_epi32(left_lo, tm_lo), two));
  let even_hi = _mm_srli_epi32::<2>(_mm_add_epi32(_mm_add_epi32(left_hi, tm_hi), two));
  let odd_lo = _mm_srli_epi32::<2>(_mm_add_epi32(_mm_add_epi32(tm_lo, right_lo), two));
  let odd_hi = _mm_srli_epi32::<2>(_mm_add_epi32(_mm_add_epi32(tm_hi, right_hi), two));
  (
    _mm_packus_epi32(even_lo, even_hi),
    _mm_packus_epi32(odd_lo, odd_hi),
  )
}

/// SSE4.1 u8 centered 2:1 horizontal chroma upsample.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `c_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "sse4.1")]
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
  // SAFETY: `j + 16 < half` keeps every offset load inside `c_half[0..half]`
  // and every 32-byte store inside `c_full[0..width]`.
  unsafe {
    let two = _mm_set1_epi16(2);
    let three = _mm_set1_epi16(3);
    while j + 16 < half {
      let mid = _mm_loadu_si128(c_half.as_ptr().add(j).cast());
      let left = _mm_loadu_si128(c_half.as_ptr().add(j - 1).cast());
      let right = _mm_loadu_si128(c_half.as_ptr().add(j + 1).cast());
      let widen_lo = |v: __m128i| _mm_cvtepu8_epi16(v);
      let widen_hi = |v: __m128i| _mm_cvtepu8_epi16(_mm_srli_si128::<8>(v));
      let mid_lo = widen_lo(mid);
      let mid_hi = widen_hi(mid);
      let left_lo = widen_lo(left);
      let left_hi = widen_hi(left);
      let right_lo = widen_lo(right);
      let right_hi = widen_hi(right);
      let tm_lo = _mm_mullo_epi16(mid_lo, three);
      let tm_hi = _mm_mullo_epi16(mid_hi, three);
      let even_lo = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(left_lo, tm_lo), two));
      let even_hi = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(left_hi, tm_hi), two));
      let odd_lo = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(tm_lo, right_lo), two));
      let odd_hi = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(tm_hi, right_hi), two));
      // `packus_epi16(lo, hi)` packs the eight low then eight high u16 into
      // one u8x16 (values ≤ 255, no saturation); `unpack` zips even/odd.
      let even = _mm_packus_epi16(even_lo, even_hi);
      let odd = _mm_packus_epi16(odd_lo, odd_hi);
      let out_lo = _mm_unpacklo_epi8(even, odd);
      let out_hi = _mm_unpackhi_epi8(even, odd);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j).cast(), out_lo);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j + 16).cast(), out_hi);
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_pair(c_half, c_full, j, half);
    j += 1;
  }
}

/// SSE4.1 u16 centered 2:1 horizontal chroma upsample.
///
/// Block size: 8 chroma samples / iter (→ 16 output columns). Samples are
/// normalized wire → host-native, masked to the low `BITS`, blended, and
/// re-encoded to the same wire order.
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `c_half.len() >= width / 2`;
/// `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let bmask = bswap16_mask();
    let mask = _mm_set1_epi16(((1u32 << BITS) - 1) as i16);
    let norm = |v: __m128i| {
      let host = if swap { _mm_shuffle_epi8(v, bmask) } else { v };
      _mm_and_si128(host, mask)
    };
    while j + 8 < half {
      let mid = norm(_mm_loadu_si128(c_half.as_ptr().add(j).cast()));
      let left = norm(_mm_loadu_si128(c_half.as_ptr().add(j - 1).cast()));
      let right = norm(_mm_loadu_si128(c_half.as_ptr().add(j + 1).cast()));
      let (even, odd) = blend_u16x8(left, mid, right);
      let even = if swap {
        _mm_shuffle_epi8(even, bmask)
      } else {
        even
      };
      let odd = if swap {
        _mm_shuffle_epi8(odd, bmask)
      } else {
        odd
      };
      let out_lo = _mm_unpacklo_epi16(even, odd);
      let out_hi = _mm_unpackhi_epi16(even, odd);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j).cast(), out_lo);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j + 8).cast(), out_hi);
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_u16_pair::<BITS>(c_half, c_full, j, half, big_endian);
    j += 1;
  }
}

/// SSE4.1 semi-planar P-format centered 2:1 horizontal chroma upsample.
///
/// Block size: 4 chroma samples / iter. The interleaved half-row is
/// de-interleaved with `_mm_shuffle_epi8`, de-packed (`>> (16 - BITS)`
/// high-bit-packed, `& mask` low-packed), blended, re-packed, and
/// re-interleaved into the `U,V,U,V` output with `unpack`s.
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `uv_half.len() >= width`;
/// `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "sse4.1")]
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
  let mut j = 1usize;
  // SAFETY: `j + 4 < half` keeps every 8-`u16` load inside `uv_half[0..width]`
  // and every 16-`u16` store inside `uv_full[0..2*width]`.
  unsafe {
    let bmask = bswap16_mask();
    let mask32 = _mm_set1_epi32(((1u32 << BITS) - 1) as i32);
    let mask16 = _mm_set1_epi16(((1u32 << BITS) - 1) as i16);
    let shift_ct = _mm_cvtsi32_si128((16 - BITS) as i32);
    // Gather the even (U) / odd (V) u16 lanes of an 8-`u16` load into the
    // low 64 bits.
    let u_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    let v_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 15, 14, 11, 10, 7, 6, 3, 2,
    );
    while j + 4 < half {
      let load = |off: usize| _mm_loadu_si128(uv_half.as_ptr().add(2 * off).cast());
      let ll = load(j - 1);
      let mm = load(j);
      let rr = load(j + 1);
      // De-interleave + normalize wire order, then widen U/V to u32x4.
      let deint = |v: __m128i, chan: __m128i| {
        let host = if swap { _mm_shuffle_epi8(v, bmask) } else { v };
        _mm_cvtepu16_epi32(_mm_shuffle_epi8(host, chan))
      };
      let depack = |v: __m128i| {
        if LOW_PACKED {
          _mm_and_si128(v, mask32)
        } else {
          _mm_srl_epi32(v, shift_ct)
        }
      };
      let ul = depack(deint(ll, u_mask));
      let um = depack(deint(mm, u_mask));
      let ur = depack(deint(rr, u_mask));
      let vl = depack(deint(ll, v_mask));
      let vm = depack(deint(mm, v_mask));
      let vr = depack(deint(rr, v_mask));
      let two = _mm_set1_epi32(2);
      let three = _mm_set1_epi32(3);
      let blend_even = |a: __m128i, b: __m128i| {
        _mm_srli_epi32::<2>(_mm_add_epi32(
          _mm_add_epi32(a, _mm_mullo_epi32(b, three)),
          two,
        ))
      };
      let blend_odd = |b: __m128i, c: __m128i| {
        _mm_srli_epi32::<2>(_mm_add_epi32(
          _mm_add_epi32(_mm_mullo_epi32(b, three), c),
          two,
        ))
      };
      // Narrow to u16x4 (low 64 bits) and re-pack to wire order.
      let repack = |v: __m128i| {
        let packed16 = _mm_packus_epi32(v, _mm_setzero_si128());
        let aligned = if LOW_PACKED {
          packed16
        } else {
          _mm_sll_epi16(_mm_and_si128(packed16, mask16), shift_ct)
        };
        if swap {
          _mm_shuffle_epi8(aligned, bmask)
        } else {
          aligned
        }
      };
      let u_even = repack(blend_even(ul, um));
      let u_odd = repack(blend_odd(um, ur));
      let v_even = repack(blend_even(vl, vm));
      let v_odd = repack(blend_odd(vm, vr));
      // Re-interleave: uv_even = [Ue0,Ve0,Ue1,Ve1,...]; uv_odd likewise; then
      // zip 32-bit (U,V) groups so each sample's four elements land adjacent.
      let uv_even = _mm_unpacklo_epi16(u_even, v_even);
      let uv_odd = _mm_unpacklo_epi16(u_odd, v_odd);
      let out_lo = _mm_unpacklo_epi32(uv_even, uv_odd);
      let out_hi = _mm_unpackhi_epi32(uv_even, uv_odd);
      _mm_storeu_si128(uv_full.as_mut_ptr().add(4 * j).cast(), out_lo);
      _mm_storeu_si128(uv_full.as_mut_ptr().add(4 * j + 8).cast(), out_hi);
      j += 4;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_p0xx_pair::<BITS, LOW_PACKED>(
      uv_half, uv_full, j, half, big_endian,
    );
    j += 1;
  }
}

/// SSE4.1 u8 bottom-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottom_even_h`](crate::row::scalar::chroma_upsample_420_bottom_even_h).
/// Each interior offset's vertical box average `e = (prev + cur + 1) >> 1` is one
/// `_mm_avg_epu8` (rounding halving add), fed into the same centered `1/4`–`3/4`
/// blend as the horizontal sibling. Boundary columns reuse the shared scalar pair.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `prev_half.len() >= width / 2`;
/// `cur_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "sse4.1")]
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
  // SAFETY: `j + 16 < half` keeps every offset load inside the half rows and
  // every 32-byte store inside `c_full[0..width]`; the interior samples read
  // here have real (non-clamped) neighbours.
  unsafe {
    let two = _mm_set1_epi16(2);
    let three = _mm_set1_epi16(3);
    let avg = |off: usize| {
      _mm_avg_epu8(
        _mm_loadu_si128(prev_half.as_ptr().add(off).cast()),
        _mm_loadu_si128(cur_half.as_ptr().add(off).cast()),
      )
    };
    let widen_lo = |v: __m128i| _mm_cvtepu8_epi16(v);
    let widen_hi = |v: __m128i| _mm_cvtepu8_epi16(_mm_srli_si128::<8>(v));
    while j + 16 < half {
      let e_left = avg(j - 1);
      let e_mid = avg(j);
      let e_right = avg(j + 1);
      let mid_lo = widen_lo(e_mid);
      let mid_hi = widen_hi(e_mid);
      let left_lo = widen_lo(e_left);
      let left_hi = widen_hi(e_left);
      let right_lo = widen_lo(e_right);
      let right_hi = widen_hi(e_right);
      let tm_lo = _mm_mullo_epi16(mid_lo, three);
      let tm_hi = _mm_mullo_epi16(mid_hi, three);
      let even_lo = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(left_lo, tm_lo), two));
      let even_hi = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(left_hi, tm_hi), two));
      let odd_lo = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(tm_lo, right_lo), two));
      let odd_hi = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(tm_hi, right_hi), two));
      let even = _mm_packus_epi16(even_lo, even_hi);
      let odd = _mm_packus_epi16(odd_lo, odd_hi);
      let out_lo = _mm_unpacklo_epi8(even, odd);
      let out_hi = _mm_unpackhi_epi8(even, odd);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j).cast(), out_lo);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j + 16).cast(), out_hi);
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottom_even_h_pair(prev_half, cur_half, c_full, j, half);
    j += 1;
  }
}

/// SSE4.1 u8 bottom-left-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottomleft_even_h`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h).
/// The co-sited (`h = 0`) horizontal phase is a plain 2× replicate with no
/// neighbour, so each column is independent: `e = _mm_avg_epu8(prev, cur)` then a
/// self-interleaved unpack.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `prev_half.len() >= width / 2`;
/// `cur_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "sse4.1")]
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
  // SAFETY: `j + 16 <= half` keeps every load inside the half rows and every
  // 32-byte store inside `c_full[0..width]`.
  unsafe {
    while j + 16 <= half {
      let e = _mm_avg_epu8(
        _mm_loadu_si128(prev_half.as_ptr().add(j).cast()),
        _mm_loadu_si128(cur_half.as_ptr().add(j).cast()),
      );
      let out_lo = _mm_unpacklo_epi8(e, e);
      let out_hi = _mm_unpackhi_epi8(e, e);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j).cast(), out_lo);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j + 16).cast(), out_hi);
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottomleft_even_h_pair(prev_half, cur_half, c_full, j);
    j += 1;
  }
}

/// SSE4.1 u16 bottom-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottom_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottom_even_h_u16).
/// Each offset's `e` is `_mm_avg_epu16` of the two masked, host-normalized rows,
/// fed into the same centered blend as the horizontal sibling; the output is
/// re-encoded to wire order — bit-identical to the scalar reference per tier.
///
/// Block size: 8 chroma samples / iter (→ 16 output columns).
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `prev_half.len() >= width / 2`;
/// `cur_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "sse4.1")]
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
  // SAFETY: `j + 8 < half` keeps every offset load inside the half rows and
  // every 16-`u16` store inside `c_full[0..width]`.
  unsafe {
    let bmask = bswap16_mask();
    let mask = _mm_set1_epi16(((1u32 << BITS) - 1) as i16);
    let norm = |row: &[u16], off: usize| {
      let raw = _mm_loadu_si128(row.as_ptr().add(off).cast());
      let host = if swap {
        _mm_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm_and_si128(host, mask)
    };
    let vavg = |off: usize| _mm_avg_epu16(norm(prev_half, off), norm(cur_half, off));
    while j + 8 < half {
      let e_left = vavg(j - 1);
      let e_mid = vavg(j);
      let e_right = vavg(j + 1);
      let (even, odd) = blend_u16x8(e_left, e_mid, e_right);
      let even = if swap {
        _mm_shuffle_epi8(even, bmask)
      } else {
        even
      };
      let odd = if swap {
        _mm_shuffle_epi8(odd, bmask)
      } else {
        odd
      };
      let out_lo = _mm_unpacklo_epi16(even, odd);
      let out_hi = _mm_unpackhi_epi16(even, odd);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j).cast(), out_lo);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j + 8).cast(), out_hi);
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottom_even_h_u16_pair::<BITS>(
      prev_half, cur_half, c_full, j, half, big_endian,
    );
    j += 1;
  }
}

/// SSE4.1 u16 bottom-left-sited even-row 4:2:0 chroma upsample — the SIMD twin of
/// [`chroma_upsample_420_bottomleft_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16).
/// Per-column `e = _mm_avg_epu16` of the masked, host-normalized rows, re-encoded
/// to wire order and replicated across the column pair.
///
/// Block size: 8 chroma samples / iter (→ 16 output columns).
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `prev_half.len() >= width / 2`;
/// `cur_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "sse4.1")]
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
  // SAFETY: `j + 8 <= half` keeps every load inside the half rows and every
  // 16-`u16` store inside `c_full[0..width]`.
  unsafe {
    let bmask = bswap16_mask();
    let mask = _mm_set1_epi16(((1u32 << BITS) - 1) as i16);
    let norm = |row: &[u16], off: usize| {
      let raw = _mm_loadu_si128(row.as_ptr().add(off).cast());
      let host = if swap {
        _mm_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm_and_si128(host, mask)
    };
    while j + 8 <= half {
      let e = _mm_avg_epu16(norm(prev_half, j), norm(cur_half, j));
      let e = if swap { _mm_shuffle_epi8(e, bmask) } else { e };
      let out_lo = _mm_unpacklo_epi16(e, e);
      let out_hi = _mm_unpackhi_epi16(e, e);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j).cast(), out_lo);
      _mm_storeu_si128(c_full.as_mut_ptr().add(2 * j + 8).cast(), out_hi);
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottomleft_even_h_u16_pair::<BITS>(
      prev_half, cur_half, c_full, j, big_endian,
    );
    j += 1;
  }
}

/// SSE4.1 semi-planar P-format bottom-sited even-row 4:2:0 chroma upsample — the
/// SIMD twin of
/// [`chroma_upsample_420_bottom_even_h_p0xx`](crate::row::scalar::chroma_upsample_420_bottom_even_h_p0xx).
/// High-bit-packed (P010/P012/P016) only; the interleaved `U`/`V` are gathered
/// with `_mm_shuffle_epi8`, de-packed (`>> (16 - BITS)`), vertically averaged
/// `(prev + cur + 1) >> 1` in the `u32` domain, centered-blended, re-packed, and
/// re-interleaved into the `U,V,U,V` output.
///
/// Block size: 4 chroma samples / iter.
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `prev_uv_half.len() >= width`;
/// `cur_uv_half.len() >= width`; `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn chroma_upsample_420_bottom_even_h_p0xx_row<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(prev_uv_half.len() >= width, "prev_uv_half row too short");
  debug_assert!(cur_uv_half.len() >= width, "cur_uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let half = width / 2;
  if half == 0 {
    return;
  }
  scalar::chroma_upsample_420_bottom_even_h_p0xx_pair::<BITS>(
    prev_uv_half,
    cur_uv_half,
    uv_full,
    0,
    half,
    big_endian,
  );
  if half == 1 {
    return;
  }

  let swap = big_endian != cfg!(target_endian = "big");
  let mut j = 1usize;
  // SAFETY: `j + 4 < half` keeps every 8-`u16` load inside the half rows and
  // every 16-`u16` store inside `uv_full[0..2*width]`.
  unsafe {
    let bmask = bswap16_mask();
    let mask16 = _mm_set1_epi16(((1u32 << BITS) - 1) as i16);
    let shift_ct = _mm_cvtsi32_si128((16 - BITS) as i32);
    let one = _mm_set1_epi32(1);
    let two = _mm_set1_epi32(2);
    let three = _mm_set1_epi32(3);
    let u_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    let v_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 15, 14, 11, 10, 7, 6, 3, 2,
    );
    // De-interleave one channel of an 8-`u16` interleaved load, normalize wire
    // order, widen to u32x4, and de-pack (high-bit-packed: `>> (16 - BITS)`).
    let dp = |row: &[u16], off: usize, chan: __m128i| {
      let raw = _mm_loadu_si128(row.as_ptr().add(2 * off).cast());
      let host = if swap {
        _mm_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm_srl_epi32(_mm_cvtepu16_epi32(_mm_shuffle_epi8(host, chan)), shift_ct)
    };
    let vavg =
      |a: __m128i, b: __m128i| _mm_srli_epi32::<1>(_mm_add_epi32(_mm_add_epi32(a, b), one));
    let blend_even = |a: __m128i, b: __m128i| {
      _mm_srli_epi32::<2>(_mm_add_epi32(
        _mm_add_epi32(a, _mm_mullo_epi32(b, three)),
        two,
      ))
    };
    let blend_odd = |b: __m128i, c: __m128i| {
      _mm_srli_epi32::<2>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(b, three), c),
        two,
      ))
    };
    let repack = |v: __m128i| {
      let packed16 = _mm_packus_epi32(v, _mm_setzero_si128());
      let aligned = _mm_sll_epi16(_mm_and_si128(packed16, mask16), shift_ct);
      if swap {
        _mm_shuffle_epi8(aligned, bmask)
      } else {
        aligned
      }
    };
    while j + 4 < half {
      let ul = vavg(
        dp(prev_uv_half, j - 1, u_mask),
        dp(cur_uv_half, j - 1, u_mask),
      );
      let um = vavg(dp(prev_uv_half, j, u_mask), dp(cur_uv_half, j, u_mask));
      let ur = vavg(
        dp(prev_uv_half, j + 1, u_mask),
        dp(cur_uv_half, j + 1, u_mask),
      );
      let vl = vavg(
        dp(prev_uv_half, j - 1, v_mask),
        dp(cur_uv_half, j - 1, v_mask),
      );
      let vm = vavg(dp(prev_uv_half, j, v_mask), dp(cur_uv_half, j, v_mask));
      let vr = vavg(
        dp(prev_uv_half, j + 1, v_mask),
        dp(cur_uv_half, j + 1, v_mask),
      );
      let u_even = repack(blend_even(ul, um));
      let u_odd = repack(blend_odd(um, ur));
      let v_even = repack(blend_even(vl, vm));
      let v_odd = repack(blend_odd(vm, vr));
      let uv_even = _mm_unpacklo_epi16(u_even, v_even);
      let uv_odd = _mm_unpacklo_epi16(u_odd, v_odd);
      let out_lo = _mm_unpacklo_epi32(uv_even, uv_odd);
      let out_hi = _mm_unpackhi_epi32(uv_even, uv_odd);
      _mm_storeu_si128(uv_full.as_mut_ptr().add(4 * j).cast(), out_lo);
      _mm_storeu_si128(uv_full.as_mut_ptr().add(4 * j + 8).cast(), out_hi);
      j += 4;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottom_even_h_p0xx_pair::<BITS>(
      prev_uv_half,
      cur_uv_half,
      uv_full,
      j,
      half,
      big_endian,
    );
    j += 1;
  }
}

/// SSE4.1 semi-planar P-format bottom-left-sited even-row 4:2:0 chroma upsample —
/// the SIMD twin of
/// [`chroma_upsample_420_bottomleft_even_h_p0xx`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h_p0xx).
/// High-bit-packed only; per-column `e = (prev + cur + 1) >> 1` of the de-packed
/// `U` and `V`, re-packed and replicated across the column pair.
///
/// Block size: 4 chroma samples / iter.
///
/// # Safety
///
/// SSE4.1 must be available. `width` even; `prev_uv_half.len() >= width`;
/// `cur_uv_half.len() >= width`; `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn chroma_upsample_420_bottomleft_even_h_p0xx_row<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(prev_uv_half.len() >= width, "prev_uv_half row too short");
  debug_assert!(cur_uv_half.len() >= width, "cur_uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let half = width / 2;
  let swap = big_endian != cfg!(target_endian = "big");
  let mut j = 0usize;
  // SAFETY: `j + 4 <= half` keeps every 8-`u16` load inside the half rows and
  // every 16-`u16` store inside `uv_full[0..2*width]`.
  unsafe {
    let bmask = bswap16_mask();
    let mask16 = _mm_set1_epi16(((1u32 << BITS) - 1) as i16);
    let shift_ct = _mm_cvtsi32_si128((16 - BITS) as i32);
    let one = _mm_set1_epi32(1);
    let u_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    let v_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 15, 14, 11, 10, 7, 6, 3, 2,
    );
    let dp = |row: &[u16], off: usize, chan: __m128i| {
      let raw = _mm_loadu_si128(row.as_ptr().add(2 * off).cast());
      let host = if swap {
        _mm_shuffle_epi8(raw, bmask)
      } else {
        raw
      };
      _mm_srl_epi32(_mm_cvtepu16_epi32(_mm_shuffle_epi8(host, chan)), shift_ct)
    };
    let vavg =
      |a: __m128i, b: __m128i| _mm_srli_epi32::<1>(_mm_add_epi32(_mm_add_epi32(a, b), one));
    let repack = |v: __m128i| {
      let packed16 = _mm_packus_epi32(v, _mm_setzero_si128());
      let aligned = _mm_sll_epi16(_mm_and_si128(packed16, mask16), shift_ct);
      if swap {
        _mm_shuffle_epi8(aligned, bmask)
      } else {
        aligned
      }
    };
    while j + 4 <= half {
      let u = repack(vavg(
        dp(prev_uv_half, j, u_mask),
        dp(cur_uv_half, j, u_mask),
      ));
      let v = repack(vavg(
        dp(prev_uv_half, j, v_mask),
        dp(cur_uv_half, j, v_mask),
      ));
      let uv = _mm_unpacklo_epi16(u, v);
      let out_lo = _mm_unpacklo_epi32(uv, uv);
      let out_hi = _mm_unpackhi_epi32(uv, uv);
      _mm_storeu_si128(uv_full.as_mut_ptr().add(4 * j).cast(), out_lo);
      _mm_storeu_si128(uv_full.as_mut_ptr().add(4 * j + 8).cast(), out_hi);
      j += 4;
    }
  }

  while j < half {
    scalar::chroma_upsample_420_bottomleft_even_h_p0xx_pair::<BITS>(
      prev_uv_half,
      cur_uv_half,
      uv_full,
      j,
      big_endian,
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

  const WIDTHS: &[usize] = &[2, 4, 6, 8, 16, 18, 30, 32, 34, 62, 64, 66, 128, 130];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn sse41_u8_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_u16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_p0xx_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_u8_vertical_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_u16_vertical_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_p0xx_vertical_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
