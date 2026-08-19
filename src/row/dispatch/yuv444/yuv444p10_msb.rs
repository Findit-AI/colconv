//! MSB-aligned 10-bit planar YUV 4:4:4 dispatchers (`Yuv444p10Msb`).
//!
//! The recovery-shift twin of [`yuv444p10`](super::yuv444p10): samples live in
//! the **high** 10 bits of each `u16` (FFmpeg `shift = 6`), recovered via
//! `>> 6` instead of a low-bit mask. RGB / RGB-u16 / HSV are thin wrappers over
//! the BITS-generic `super::yuv_444p_n_msb_*` helpers; the RGBA / RGBA-u16
//! paths are full per-backend dispatchers.

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
  KernelMatrix,
  row::{rgba_row_bytes, rgba_row_elems, scalar},
};

use super::{yuv_444p_n_msb_to_rgb_row, yuv_444p_n_msb_to_rgb_u16_row};

/// MSB-aligned YUV 4:4:4 planar 10-bit → u8 RGB. Endian-aware variant
/// (`big_endian = true` selects the BE-encoded `u16` plane contract).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_msb_to_rgb_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    yuv_444p_n_msb_to_rgb_row::<10, true>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  } else {
    yuv_444p_n_msb_to_rgb_row::<10, false>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// MSB-aligned YUV 4:4:4 planar 10-bit → native-depth u16 RGB. Endian-aware.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_msb_to_rgb_u16_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    yuv_444p_n_msb_to_rgb_u16_row::<10, true>(
      y, u, v, rgb_out, width, matrix, full_range, use_simd,
    );
  } else {
    yuv_444p_n_msb_to_rgb_u16_row::<10, false>(
      y, u, v, rgb_out, width, matrix, full_range, use_simd,
    );
  }
}

/// MSB-aligned YUV 4:4:4 planar 10-bit → packed **8-bit RGBA** (`R, G, B,
/// 0xFF`). Endian-aware full dispatcher. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_msb_to_rgba_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe {
              arch::neon::yuv_444p_n_msb_to_rgba_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_444p_n_msb_to_rgba_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::yuv_444p_n_msb_to_rgba_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_444p_n_msb_to_rgba_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_444p_n_msb_to_rgba_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_444p_n_msb_to_rgba_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_444p_n_msb_to_rgba_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_444p_n_msb_to_rgba_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::yuv_444p_n_msb_to_rgba_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_444p_n_msb_to_rgba_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      _ => {}
    }
  }

  dispatch_be!(
    scalar::yuv_444p_n_msb_to_rgba_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range),
    scalar::yuv_444p_n_msb_to_rgba_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range)
  );
}

/// MSB-aligned YUV 4:4:4 planar 10-bit → **native-depth `u16`** packed RGBA
/// (low-bit-packed; alpha element `1023`). Endian-aware full dispatcher.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_msb_to_rgba_u16_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe {
              arch::neon::yuv_444p_n_msb_to_rgba_u16_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_444p_n_msb_to_rgba_u16_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::yuv_444p_n_msb_to_rgba_u16_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_444p_n_msb_to_rgba_u16_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_444p_n_msb_to_rgba_u16_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_444p_n_msb_to_rgba_u16_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_444p_n_msb_to_rgba_u16_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_444p_n_msb_to_rgba_u16_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::yuv_444p_n_msb_to_rgba_u16_row::<10, false>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_444p_n_msb_to_rgba_u16_row::<10, true>(
                y, u, v, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      _ => {}
    }
  }

  dispatch_be!(
    scalar::yuv_444p_n_msb_to_rgba_u16_row::<10, false>(
      y, u, v, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_444p_n_msb_to_rgba_u16_row::<10, true>(
      y, u, v, rgba_out, width, matrix, full_range
    )
  );
}

/// MSB-aligned YUV 4:4:4 planar 10-bit → planar **HSV** bytes. Endian-aware
/// thin wrapper over the BITS-generic `super::yuv_444p_n_msb_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_msb_to_hsv_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    super::yuv_444p_n_msb_to_hsv_row::<10, true>(
      y, u, v, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  } else {
    super::yuv_444p_n_msb_to_hsv_row::<10, false>(
      y, u, v, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  }
}
