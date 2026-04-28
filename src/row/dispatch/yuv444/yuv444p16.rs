//! 16-bit planar YUV 4:4:4 dispatchers — 4 variants. The BITS-generic
//! helpers in `super::*` are pinned to {9,10,12,14}, so 16-bit gets
//! its own dedicated dispatchers (i64 chroma at native u16 output).

use crate::row::scalar;
use crate::row::{arch, rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems};
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
use crate::ColorMatrix;


/// YUV 4:4:4 planar **16-bit** → packed **u8** RGB. Uses the
/// parallel 16-bit kernel family (same Q15 i32 output-range pipeline
/// as [`yuv_420p16_to_rgb_row`] but with 1:1 chroma per pixel).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p16_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
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
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar **16-bit** → packed **u16** RGB (full-range
/// output in `[0, 65535]`). Widens chroma multiply-add + Y scale to
/// i64 to avoid i32 overflow at 16-bit limited range.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p16_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified. Native 512-bit i64-chroma kernel.
          unsafe {
            arch::x86_avx512::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
}


/// Converts one row of **16-bit** YUV 4:4:4 to packed **8-bit**
/// **RGBA** (`R, G, B, 0xFF`). Routes through the dedicated 16-bit
/// scalar kernel (`scalar::yuv_444p16_to_rgba_row`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p16_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
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
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p16_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p16_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p16_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p16_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p16_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p16_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
}

/// Converts one row of **16-bit** YUV 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — full-range output `[0, 65535]`; alpha element is
/// `0xFFFF`. Routes through the dedicated 16-bit u16-output scalar
/// kernel (`scalar::yuv_444p16_to_rgba_u16_row`) — i64 chroma multiply.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p16_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p16_to_rgba_u16_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p16_to_rgba_u16_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p16_to_rgba_u16_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p16_to_rgba_u16_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p16_to_rgba_u16_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p16_to_rgba_u16_row(y, u, v, rgba_out, width, matrix, full_range);
}
