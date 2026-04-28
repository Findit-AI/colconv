//! 14-bit planar YUV 4:2:0 dispatchers — 4 variants.

use crate::row::scalar;
use crate::row::{arch, rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems};
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
use crate::ColorMatrix;


/// Converts one row of **14‑bit** YUV 4:2:0 to packed **8‑bit** RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p14_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_row::<14>(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_row::<14>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **14‑bit** YUV 4:2:0 to **native‑depth** packed
/// `u16` RGB (14‑bit values in the low 14 of each `u16`).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p14_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_u16_row::<14>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}


/// Converts one row of **14-bit** YUV 4:2:0 to packed **8-bit**
/// **RGBA** (`R, G, B, 0xFF`; alpha defaults to opaque since the
/// source has no alpha plane).
///
/// Same numerical contract as [`yuv420p14_to_rgb_row`] except
/// for the per-pixel stride (4 vs 3) and the constant alpha byte. See
/// `scalar::yuv_420p_n_to_rgba_row` for the reference.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p14_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_420p_n_to_rgba_row::<14>(y, u_half, v_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgba_row::<14>(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgba_row::<14>(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgba_row::<14>(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgba_row::<14>(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgba_row::<14>(y, u_half, v_half, rgba_out, width, matrix, full_range);
}

/// Converts one row of **14-bit** YUV 4:2:0 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, (1 << 14) - 1]`
/// in the low bits of each `u16`); alpha element is `(1 << 14) - 1`
/// (opaque maximum at the input bit depth).
///
/// See `scalar::yuv_420p_n_to_rgba_u16_row` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p14_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_420p_n_to_rgba_u16_row::<14>(y, u_half, v_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgba_u16_row::<14>(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgba_u16_row::<14>(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgba_u16_row::<14>(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgba_u16_row::<14>(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgba_u16_row::<14>(y, u_half, v_half, rgba_out, width, matrix, full_range);
}
