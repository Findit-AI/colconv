//! wasm-simd128 centered (#302 phase-0.5) 2:1 horizontal chroma-upsample
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
//! computed with offset loads and `i8x16_shuffle`-based interleaved stores.
//! The two boundary columns (`j = 0`, `j = half-1`) reuse the shared scalar
//! per-sample reference so the edges stay byte-identical; the vector loop
//! covers the strict interior.

#![cfg_attr(
  not(all(
    any(feature = "std", feature = "alloc"),
    feature = "yuv-planar",
    feature = "yuv-semi-planar"
  )),
  allow(dead_code)
)]

use core::arch::wasm32::*;

use crate::row::scalar;

/// Byte-swaps the two bytes of each `u16` lane (wire ↔ host-native).
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "simd128")]
fn bswap_u16(v: v128) -> v128 {
  i8x16_shuffle::<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(v, v)
}

/// Blends eight interior chroma samples in `u32` lanes and narrows back to
/// `u16`: returns `(even, odd)` with `even = (left + 3·mid + 2) >> 2` and
/// `odd = (3·mid + right + 2) >> 2`. Inputs are logical `u16` lanes; the
/// widen-to-`u32` avoids the `BITS = 16` overflow and the narrow is exact.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "simd128")]
fn blend_u16(left: v128, mid: v128, right: v128) -> (v128, v128) {
  let two = i32x4_splat(2);
  let three = i32x4_splat(3);
  let mid_lo = u32x4_extend_low_u16x8(mid);
  let mid_hi = u32x4_extend_high_u16x8(mid);
  let left_lo = u32x4_extend_low_u16x8(left);
  let left_hi = u32x4_extend_high_u16x8(left);
  let right_lo = u32x4_extend_low_u16x8(right);
  let right_hi = u32x4_extend_high_u16x8(right);
  let tm_lo = i32x4_mul(mid_lo, three);
  let tm_hi = i32x4_mul(mid_hi, three);
  let even_lo = u32x4_shr(i32x4_add(i32x4_add(left_lo, tm_lo), two), 2);
  let even_hi = u32x4_shr(i32x4_add(i32x4_add(left_hi, tm_hi), two), 2);
  let odd_lo = u32x4_shr(i32x4_add(i32x4_add(tm_lo, right_lo), two), 2);
  let odd_hi = u32x4_shr(i32x4_add(i32x4_add(tm_hi, right_hi), two), 2);
  (
    u16x8_narrow_i32x4(even_lo, even_hi),
    u16x8_narrow_i32x4(odd_lo, odd_hi),
  )
}

/// wasm-simd128 u8 centered 2:1 horizontal chroma upsample.
///
/// Block size: 16 chroma samples / iter (→ 32 output columns).
///
/// # Safety
///
/// simd128 must be enabled at compile time. `width` even;
/// `c_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "simd128")]
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
  let two = i16x8_splat(2);
  let three = i16x8_splat(3);
  // SAFETY: `j + 16 < half` keeps every offset load inside `c_half[0..half]`
  // and every 32-byte store inside `c_full[0..width]`.
  unsafe {
    while j + 16 < half {
      let mid = v128_load(c_half.as_ptr().add(j).cast());
      let left = v128_load(c_half.as_ptr().add(j - 1).cast());
      let right = v128_load(c_half.as_ptr().add(j + 1).cast());
      let mid_lo = u16x8_extend_low_u8x16(mid);
      let mid_hi = u16x8_extend_high_u8x16(mid);
      let left_lo = u16x8_extend_low_u8x16(left);
      let left_hi = u16x8_extend_high_u8x16(left);
      let right_lo = u16x8_extend_low_u8x16(right);
      let right_hi = u16x8_extend_high_u8x16(right);
      let tm_lo = i16x8_mul(mid_lo, three);
      let tm_hi = i16x8_mul(mid_hi, three);
      let even_lo = u16x8_shr(i16x8_add(i16x8_add(left_lo, tm_lo), two), 2);
      let even_hi = u16x8_shr(i16x8_add(i16x8_add(left_hi, tm_hi), two), 2);
      let odd_lo = u16x8_shr(i16x8_add(i16x8_add(tm_lo, right_lo), two), 2);
      let odd_hi = u16x8_shr(i16x8_add(i16x8_add(tm_hi, right_hi), two), 2);
      let even = u8x16_narrow_i16x8(even_lo, even_hi);
      let odd = u8x16_narrow_i16x8(odd_lo, odd_hi);
      let out_lo =
        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(even, odd);
      let out_hi =
        i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(even, odd);
      v128_store(c_full.as_mut_ptr().add(2 * j).cast(), out_lo);
      v128_store(c_full.as_mut_ptr().add(2 * j + 16).cast(), out_hi);
      j += 16;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_pair(c_half, c_full, j, half);
    j += 1;
  }
}

/// wasm-simd128 u16 centered 2:1 horizontal chroma upsample.
///
/// Block size: 8 chroma samples / iter (→ 16 output columns).
///
/// # Safety
///
/// simd128 must be enabled at compile time. `width` even;
/// `c_half.len() >= width / 2`; `c_full.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "simd128")]
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
  let mask = u16x8_splat(((1u32 << BITS) - 1) as u16);
  let mut j = 1usize;
  // SAFETY: `j + 8 < half` keeps every offset load inside `c_half[0..half]`
  // and every 16-`u16` store inside `c_full[0..width]`.
  unsafe {
    while j + 8 < half {
      let norm = |ptr_off: usize| {
        let raw = v128_load(c_half.as_ptr().add(ptr_off).cast());
        let host = if swap { bswap_u16(raw) } else { raw };
        v128_and(host, mask)
      };
      let mid = norm(j);
      let left = norm(j - 1);
      let right = norm(j + 1);
      let (even, odd) = blend_u16(left, mid, right);
      let even = if swap { bswap_u16(even) } else { even };
      let odd = if swap { bswap_u16(odd) } else { odd };
      let out_lo =
        i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(even, odd);
      let out_hi =
        i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(even, odd);
      v128_store(c_full.as_mut_ptr().add(2 * j).cast(), out_lo);
      v128_store(c_full.as_mut_ptr().add(2 * j + 8).cast(), out_hi);
      j += 8;
    }
  }

  while j < half {
    scalar::chroma_upsample_2to1_center_h_u16_pair::<BITS>(c_half, c_full, j, half, big_endian);
    j += 1;
  }
}

/// wasm-simd128 semi-planar P-format centered 2:1 horizontal chroma upsample.
///
/// Block size: 4 chroma samples / iter. `i8x16_shuffle` de-interleaves the
/// half-row `U`/`V`; each channel is de-packed, blended, re-packed, and
/// re-interleaved into the `U,V,U,V` output.
///
/// # Safety
///
/// simd128 must be enabled at compile time. `width` even;
/// `uv_half.len() >= width`; `uv_full.len() >= 2 * width`.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[inline]
#[target_feature(enable = "simd128")]
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
  let shift = 16 - BITS;
  let mask = u32x4_splat((1u32 << BITS) - 1);
  let mut j = 1usize;
  // SAFETY: `j + 4 < half` keeps every 8-`u16` load inside `uv_half[0..width]`
  // and every 16-`u16` store inside `uv_full[0..2*width]`.
  unsafe {
    while j + 4 < half {
      // De-interleave U (even u16 lanes) / V (odd lanes) into u32x4, wire
      // → host normalized.
      let load_uv = |off: usize| -> (v128, v128) {
        let raw = v128_load(uv_half.as_ptr().add(2 * off).cast());
        let host = if swap { bswap_u16(raw) } else { raw };
        let u = i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13>(host, host);
        let v = i8x16_shuffle::<2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15>(host, host);
        (u32x4_extend_low_u16x8(u), u32x4_extend_low_u16x8(v))
      };
      let depack = |v: v128| {
        if LOW_PACKED {
          v128_and(v, mask)
        } else {
          u32x4_shr(v, shift)
        }
      };
      let (ul_r, vl_r) = load_uv(j - 1);
      let (um_r, vm_r) = load_uv(j);
      let (ur_r, vr_r) = load_uv(j + 1);
      let ul = depack(ul_r);
      let um = depack(um_r);
      let ur = depack(ur_r);
      let vl = depack(vl_r);
      let vm = depack(vm_r);
      let vr = depack(vr_r);
      let two = i32x4_splat(2);
      let three = i32x4_splat(3);
      let blend_even =
        |a: v128, b: v128| u32x4_shr(i32x4_add(i32x4_add(a, i32x4_mul(b, three)), two), 2);
      let blend_odd =
        |b: v128, c: v128| u32x4_shr(i32x4_add(i32x4_add(i32x4_mul(b, three), c), two), 2);
      // Narrow u32x4 → u16 (low 4 lanes) and re-pack to wire order.
      let repack = |v: v128| {
        let n = u16x8_narrow_i32x4(v, v);
        let aligned = if LOW_PACKED { n } else { u16x8_shl(n, shift) };
        if swap { bswap_u16(aligned) } else { aligned }
      };
      let u_even = repack(blend_even(ul, um));
      let u_odd = repack(blend_odd(um, ur));
      let v_even = repack(blend_even(vl, vm));
      let v_odd = repack(blend_odd(vm, vr));
      // uv_even = [Ue0,Ve0,Ue1,Ve1,...]; uv_odd likewise; then zip 32-bit
      // (U,V) groups so each sample's four elements land adjacent.
      let uv_even =
        i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(u_even, v_even);
      let uv_odd =
        i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(u_odd, v_odd);
      let out_lo =
        i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(uv_even, uv_odd);
      let out_hi = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
        uv_even, uv_odd,
      );
      v128_store(uv_full.as_mut_ptr().add(4 * j).cast(), out_lo);
      v128_store(uv_full.as_mut_ptr().add(4 * j + 8).cast(), out_hi);
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
  fn wasm_u8_matches_scalar_widths() {
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
  fn wasm_u16_matches_scalar_widths() {
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
  fn wasm_p0xx_matches_scalar_widths() {
    check_p0xx::<10, false>(false);
    check_p0xx::<10, false>(true);
    check_p0xx::<10, true>(false);
    check_p0xx::<10, true>(true);
    check_p0xx::<12, false>(false);
    check_p0xx::<16, false>(false);
    check_p0xx::<16, false>(true);
  }
}
