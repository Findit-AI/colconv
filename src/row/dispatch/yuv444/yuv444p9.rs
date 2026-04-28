//! 9-bit planar YUV 4:4:4 dispatchers — 4 variants. The RGB / RGB-u16
//! paths are thin wrappers over the BITS-generic helpers in
//! `super::{yuv_444p_n_to_rgb_row, yuv_444p_n_to_rgb_u16_row}`; the
//! RGBA / RGBA-u16 paths are full dispatchers (the BITS-generic
//! template doesn't apply for the alpha-fill case).

#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::{
  ColorMatrix,
  row::{arch, rgba_row_bytes, rgba_row_elems, scalar},
};

use super::{yuv_444p_n_to_rgb_row, yuv_444p_n_to_rgb_u16_row};

/// YUV 4:4:4 planar 9-bit → u8 RGB. Thin wrapper over the
/// crate-internal `yuv_444p_n_to_rgb_row::<9>`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p9_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_row::<9>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar 9-bit → native-depth u16 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p9_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_u16_row::<9>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

// ---- High-bit 4:4:4 RGBA dispatchers (Ship 8 Tranche 7) ---------------
//
// Both u8 and native-depth `u16` RGBA dispatchers route to per-arch
// SIMD kernels (Ship 8 Tranches 7b + 7c). `use_simd = false` forces
// the scalar reference path on every dispatcher.

/// Converts one row of **9-bit** YUV 4:4:4 to packed **8-bit**
/// **RGBA** (`R, G, B, 0xFF`; alpha defaults to opaque since the
/// source has no alpha plane).
///
/// Same numerical contract as [`yuv444p9_to_rgb_row`] except for the
/// per-pixel stride (4 vs 3) and the constant alpha byte. See
/// `scalar::yuv_444p_n_to_rgba_row` for the reference.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p9_to_rgba_row(
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
            arch::neon::yuv_444p_n_to_rgba_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
}

/// Converts one row of **9-bit** YUV 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, (1 << 9) - 1]`
/// in the low bits of each `u16`); alpha element is `(1 << 9) - 1`
/// (opaque maximum at the input bit depth).
///
/// See `scalar::yuv_444p_n_to_rgba_u16_row` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p9_to_rgba_u16_row(
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
            arch::neon::yuv_444p_n_to_rgba_u16_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_u16_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_u16_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_u16_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_u16_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_u16_row::<9>(y, u, v, rgba_out, width, matrix, full_range);
}
