//! 8-bit YUV 4:4:4 → RGB / RGBA dispatchers (`yuv_444_to_rgb_row`,
//! `yuv_444_to_rgba_row`). Extracted from the parent
//! `dispatch::yuv444` module per source format for organization.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "wasm32"))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::{
  ColorMatrix,
  row::{rgb_row_bytes, rgba_row_bytes, scalar},
};

/// Converts one row of YUV 4:4:4 planar to packed RGB. Dispatches
/// to the best available SIMD backend for the current target.
///
/// Same numerical contract as [`yuv_420_to_rgb_row`]; the difference
/// is 4:4:4 chroma — one U / V pair per Y pixel, full-width chroma
/// planes, no chroma upsampling, no width parity constraint. See
/// `scalar::yuv_444_to_rgb_row` for the reference implementation.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
}

/// Converts one row of YUV 4:4:4 planar to packed **RGBA** (8-bit).
/// Same numerical contract as [`yuv_444_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque, for every pixel). `rgba_out.len() >= 4 * width`.
/// `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
}
