//! V30X (Tier 5 packed YUV 4:4:4, 10-bit, one u32 word per pixel)
//! dispatchers.
//!
//! Six entries: `v30x_to_{rgb,rgba}_row` (u8) and the matching
//! `_u16` variants for native-depth output, plus
//! `v30x_to_luma_row` / `v30x_to_luma_u16_row` for direct luma
//! extraction. Routes through the standard `cfg_select!` per-arch
//! block; `use_simd = false` forces scalar.
//!
//! V30X is 4:4:4 (no chroma subsampling): each u32 word encodes one
//! complete pixel as 10-bit U / Y / V packed into bits [11:2] / [21:12]
//! / [31:22] with 2-bit padding at the bottom. Buffer length is `width`
//! u32 elements — no even-width restriction, no width×2 scaling.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::{
  ColorMatrix,
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar},
};

/// Converts one row of V30X to packed RGB (u8). See
/// [`scalar::v30x_to_rgb_or_rgba_row`] for word layout / numerical
/// contract. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v30x_to_rgb_row(
  packed: &[u32],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(packed.len() >= width, "packed row too short");
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe { arch::neon::v30x_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v30x_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v30x_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v30x_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v30x_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of V30X to packed RGBA (u8) with `α = 0xFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v30x_to_rgba_row(
  packed: &[u32],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(packed.len() >= width, "packed row too short");
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::v30x_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v30x_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v30x_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v30x_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v30x_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Converts one row of V30X to packed `u16` RGB at native 10-bit
/// depth (low-bit-packed, `[0, 1023]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v30x_to_rgb_u16_row(
  packed: &[u32],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(packed.len() >= width, "packed row too short");
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::v30x_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v30x_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v30x_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v30x_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v30x_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of V30X to packed `u16` RGBA at native 10-bit
/// depth with `α = 1023` (10-bit opaque maximum).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v30x_to_rgba_u16_row(
  packed: &[u32],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(packed.len() >= width, "packed row too short");
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::v30x_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v30x_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v30x_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v30x_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v30x_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed V30X buffer.
/// Y values are downshifted from 10-bit to 8-bit via `>> 2`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v30x_to_luma_row(packed: &[u32], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert!(packed.len() >= width, "packed row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::v30x_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v30x_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v30x_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v30x_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v30x_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of native-depth `u16` luma from a packed V30X
/// buffer (low-bit-packed: each `u16` carries the 10-bit Y value in
/// its low 10 bits).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v30x_to_luma_u16_row(packed: &[u32], luma_out: &mut [u16], width: usize, use_simd: bool) {
  assert!(packed.len() >= width, "packed row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::v30x_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v30x_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v30x_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v30x_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v30x_to_luma_u16_row(packed, luma_out, width);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the public V30X dispatchers. Walker / kernel
  //! correctness lives in the per-arch tests and the scalar reference's
  //! own inline tests; this block verifies the dispatcher correctly
  //! reaches its scalar fallback when SIMD is disabled and panics on
  //! invalid inputs.
  use super::*;

  /// Pack one V30X word from explicit U / Y / V samples (10-bit each).
  fn pack_v30x(u: u32, y: u32, v: u32) -> u32 {
    debug_assert!(u < 1024 && y < 1024 && v < 1024);
    (v << 22) | (y << 12) | (u << 2)
  }

  /// Build a `Vec<u32>` V30X row of `width` pixels with `(U, Y, V)`
  /// repeated. Any positive width is valid (4:4:4, no chroma subsampling).
  fn solid_v30x(width: usize, u: u32, y: u32, v: u32) -> std::vec::Vec<u32> {
    (0..width).map(|_| pack_v30x(u, y, v)).collect()
  }

  #[test]
  #[should_panic(expected = "packed row too short")]
  fn v30x_dispatcher_rejects_short_packed() {
    // packed buffer has only 2 elements for width=4 (needs 4).
    let packed = [0u32; 2];
    let mut rgb = [0u8; 4 * 3];
    v30x_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgb_out row too short")]
  fn v30x_dispatcher_rejects_short_output() {
    // output buffer has only 2 bytes for width=4 (needs 12).
    let packed = [0u32; 4];
    let mut rgb = [0u8; 2];
    v30x_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  fn v30x_dispatchers_route_with_simd_false() {
    // Full-range gray (Y=512, U=V=512 at 10-bit). Every dispatcher
    // should reach its scalar fallback when `use_simd = false`,
    // produce the documented gray output, and not panic.
    let buf = solid_v30x(8, 512, 512, 512);

    // u8 RGB
    let mut rgb = [0u8; 8 * 3];
    v30x_to_rgb_row(&buf, &mut rgb, 8, ColorMatrix::Bt709, true, false);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u8 RGBA — alpha = 0xFF
    let mut rgba = [0u8; 8 * 4];
    v30x_to_rgba_row(&buf, &mut rgba, 8, ColorMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[3], 0xFF);
    }

    // u16 RGB at native 10-bit depth.
    let mut rgb_u16 = [0u16; 8 * 3];
    v30x_to_rgb_u16_row(&buf, &mut rgb_u16, 8, ColorMatrix::Bt709, true, false);
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(512) <= 2);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u16 RGBA — alpha = 1023 (10-bit opaque maximum).
    let mut rgba_u16 = [0u16; 8 * 4];
    v30x_to_rgba_u16_row(&buf, &mut rgba_u16, 8, ColorMatrix::Bt709, true, false);
    for px in rgba_u16.chunks(4) {
      assert_eq!(px[3], 1023);
    }

    // u8 luma — Y=512 → 128 after `>> 2`.
    let mut luma = [0u8; 8];
    v30x_to_luma_row(&buf, &mut luma, 8, false);
    for &y in &luma {
      assert_eq!(y, (512u32 >> 2) as u8);
    }

    // u16 luma — low-packed 10-bit Y value.
    let mut luma_u16 = [0u16; 8];
    v30x_to_luma_u16_row(&buf, &mut luma_u16, 8, false);
    for &y in &luma_u16 {
      assert_eq!(y, 512);
    }
  }
}
