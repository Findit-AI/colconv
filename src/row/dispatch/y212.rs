//! Y212 (Tier 4 packed YUV 4:2:2 12-bit, MSB-aligned u16 quadruples)
//! dispatchers (Ship 11c).
//!
//! Six entries: `y212_to_{rgb,rgba}_row` (u8) and the matching
//! `_u16` variants for native-depth output, plus
//! `y212_to_luma_row` / `y212_to_luma_u16_row` for direct luma
//! extraction. Routes through the standard `cfg_select!` per-arch
//! block; `use_simd = false` forces scalar.
//!
//! Y212 shares its kernel family with Y210 (Ship 11b) — both are
//! BITS-template specializations of the `y2xx_n_*` kernels. The
//! public dispatchers hard-wire `BITS = 12` and split the
//! const-generic `ALPHA` parameter into the RGB / RGBA variants.

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::{
  ColorMatrix,
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar, y2xx_row_elems},
};

/// Converts one row of Y212 to packed RGB (u8). See
/// [`scalar::y2xx_n_to_rgb_or_rgba_row`] for sample layout / numerical
/// contract. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y212_to_rgb_row(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "Y212 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe { arch::neon::y2xx_n_to_rgb_or_rgba_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y2xx_n_to_rgb_or_rgba_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y2xx_n_to_rgb_or_rgba_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y2xx_n_to_rgb_or_rgba_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y2xx_n_to_rgb_or_rgba_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y212_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of Y212 to packed RGBA (u8) with `α = 0xFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y212_to_rgba_row(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "Y212 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y2xx_n_to_rgb_or_rgba_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y2xx_n_to_rgb_or_rgba_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y2xx_n_to_rgb_or_rgba_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y2xx_n_to_rgb_or_rgba_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y2xx_n_to_rgb_or_rgba_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y212_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Converts one row of Y212 to packed `u16` RGB at native 12-bit
/// depth (low-bit-packed, `[0, 4095]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y212_to_rgb_u16_row(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "Y212 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y212_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of Y212 to packed `u16` RGBA at native 12-bit
/// depth with `α = 4095` (12-bit opaque maximum).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y212_to_rgba_u16_row(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "Y212 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y212_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed Y212 buffer.
/// Y values are downshifted from 12-bit to 8-bit via `>> 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y212_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert!(
    width.is_multiple_of(2),
    "Y212 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y2xx_n_to_luma_row::<12>(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y2xx_n_to_luma_row::<12>(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y2xx_n_to_luma_row::<12>(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y2xx_n_to_luma_row::<12>(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y2xx_n_to_luma_row::<12>(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y212_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of native-depth `u16` luma from a packed Y212
/// buffer (low-bit-packed: each `u16` carries the 12-bit Y value in
/// its low 12 bits).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y212_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize, use_simd: bool) {
  assert!(
    width.is_multiple_of(2),
    "Y212 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y2xx_n_to_luma_u16_row::<12>(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y2xx_n_to_luma_u16_row::<12>(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y2xx_n_to_luma_u16_row::<12>(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y2xx_n_to_luma_u16_row::<12>(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y2xx_n_to_luma_u16_row::<12>(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y212_to_luma_u16_row(packed, luma_out, width);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the public Y212 dispatchers. Walker / kernel
  //! correctness lives in the per-arch tests
  //! (`src/row/arch/*/tests/y2xx.rs`) and the scalar reference's
  //! own inline tests; this block only verifies the dispatcher
  //! correctly reaches its scalar fallback when SIMD is disabled.
  use super::*;

  /// Build one Y212-shaped u16 quadruple `[Y0, U, Y1, V]` with each
  /// sample shifted to MSB-aligned 12-bit form (low 4 bits zero).
  fn y212_quad(y0: u16, u: u16, y1: u16, v: u16) -> [u16; 4] {
    [
      (y0 & 0xFFF) << 4,
      (u & 0xFFF) << 4,
      (y1 & 0xFFF) << 4,
      (v & 0xFFF) << 4,
    ]
  }

  /// Build a `Vec<u16>` Y212 row of `width` pixels with `(Y, U, V)`
  /// repeated. Width must be even.
  fn solid_y212(width: usize, y: u16, u: u16, v: u16) -> std::vec::Vec<u16> {
    let mut buf = std::vec::Vec::with_capacity(width * 2);
    for _ in 0..(width / 2) {
      buf.extend_from_slice(&y212_quad(y, u, y, v));
    }
    buf
  }

  #[test]
  fn y212_dispatchers_route_with_simd_false() {
    // Full-range gray (Y=2048, U=V=2048 at 12-bit). Every dispatcher
    // should reach its scalar fallback when `use_simd = false`,
    // produce the documented gray output, and not panic.
    let buf = solid_y212(8, 2048, 2048, 2048);

    // u8 RGB
    let mut rgb = [0u8; 8 * 3];
    y212_to_rgb_row(&buf, &mut rgb, 8, ColorMatrix::Bt709, true, false);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u8 RGBA — alpha = 0xFF
    let mut rgba = [0u8; 8 * 4];
    y212_to_rgba_row(&buf, &mut rgba, 8, ColorMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[3], 0xFF);
    }

    // u16 RGB at native 12-bit depth.
    let mut rgb_u16 = [0u16; 8 * 3];
    y212_to_rgb_u16_row(&buf, &mut rgb_u16, 8, ColorMatrix::Bt709, true, false);
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(2048) <= 2);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u16 RGBA — alpha = 4095.
    let mut rgba_u16 = [0u16; 8 * 4];
    y212_to_rgba_u16_row(&buf, &mut rgba_u16, 8, ColorMatrix::Bt709, true, false);
    for px in rgba_u16.chunks(4) {
      assert_eq!(px[3], 4095);
    }

    // u8 luma — Y=2048 → 128 after `>> 4`.
    let mut luma = [0u8; 8];
    y212_to_luma_row(&buf, &mut luma, 8, false);
    for &y in &luma {
      assert_eq!(y, (2048u16 >> 4) as u8);
    }

    // u16 luma — low-packed 12-bit Y.
    let mut luma_u16 = [0u16; 8];
    y212_to_luma_u16_row(&buf, &mut luma_u16, 8, false);
    for &y in &luma_u16 {
      assert_eq!(y, 2048);
    }
  }
}
