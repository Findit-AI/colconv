//! 8-bit YUV 4:1:0 → RGB / RGBA dispatchers (`yuv_410_to_rgb_row`,
//! `yuv_410_to_rgba_row`). Tier 1 P3 legacy (Cinepak / Sorenson).
//!
//! Backends: scalar (always) + NEON (aarch64) + SSE4.1 / AVX2 /
//! AVX-512 (x86_64) + simd128 (wasm32). Each SIMD backend follows
//! the same `block_size_y` / 4× horizontal chroma fan-out shape:
//! NEON / SSE4.1 / wasm 16 Y per iter, AVX2 32 Y, AVX-512 64 Y.
//! Math is byte-identical to scalar by construction.

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
  row::{rgb_row_bytes, rgba_row_bytes, scalar},
};

/// Converts one row of 4:1:0 YUV to packed RGB.
///
/// Dispatches to the best available backend for the current target.
/// See `scalar::yuv_410_to_rgb_row` for the full semantic
/// specification (range handling, matrix definitions, output layout).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_410_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary — same rationale as the
  // 4:2:0 sibling. Unsafe SIMD kernels rely on these in release builds.
  assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          // Bounds / parity invariants are the caller's obligation.
          unsafe {
            arch::neon::yuv_410_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX-512BW is present.
          unsafe {
            arch::x86_avx512::yuv_410_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` verified AVX2 is present.
          unsafe {
            arch::x86_avx2::yuv_410_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: `sse41_available()` verified SSE4.1 is present.
          unsafe {
            arch::x86_sse41::yuv_410_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time availability verified.
          unsafe {
            arch::wasm_simd128::yuv_410_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      _ => {
        // Targets without a SIMD backend (riscv64, powerpc, …) fall
        // through to scalar.
      }
    }
  }

  scalar::yuv_410_to_rgb_row(y, u_quarter, v_quarter, rgb_out, width, matrix, full_range);
}

/// Converts one row of 4:1:0 YUV to packed **RGBA** (8-bit).
///
/// Same numerical contract as [`yuv_410_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque, for every pixel — sources without an alpha plane
/// produce opaque output). The first three bytes per pixel are
/// byte-identical to what [`yuv_410_to_rgb_row`] would write.
///
/// `rgba_out.len() >= 4 * width`. `use_simd = false` forces the
/// scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_410_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::yuv_410_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX-512BW is present.
          unsafe {
            arch::x86_avx512::yuv_410_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` verified AVX2 is present.
          unsafe {
            arch::x86_avx2::yuv_410_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: `sse41_available()` verified SSE4.1 is present.
          unsafe {
            arch::x86_sse41::yuv_410_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time availability verified.
          unsafe {
            arch::wasm_simd128::yuv_410_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::yuv_410_to_rgba_row(y, u_quarter, v_quarter, rgba_out, width, matrix, full_range);
}

/// Converts one row of 4:1:0 YUV **directly** to planar HSV bytes
/// (OpenCV encoding), without materializing a source-width RGB row.
/// Output is byte-identical to `rgb_to_hsv_row(yuv_410_to_rgb_row(...))`
/// within the selected tier.
///
/// `use_simd = false` forces the scalar reference path. See
/// `scalar::yuv_410_to_hsv_row` for the semantic specification.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_410_to_hsv_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified; bounds asserted above.
          unsafe {
            arch::neon::yuv_410_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_410_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_410_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_410_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_410_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::yuv_410_to_hsv_row(
    y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
  );
}
