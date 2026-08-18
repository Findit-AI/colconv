//! 8-bit YUV 4:1:1 → RGB / RGBA dispatchers (`yuv_411_to_rgb_row`,
//! `yuv_411_to_rgba_row`).
//!
//! 4:1:1 is **legacy DV-NTSC** subsampling: chroma is quarter-width
//! and full-height. SIMD coverage: NEON (16 Y / 4 chroma per iter),
//! SSE4.1 (16 Y / 4 chroma per iter), AVX2 (32 Y / 8 chroma per iter),
//! AVX-512BW (64 Y / 16 chroma per iter), wasm32 simd128 (16 Y / 4
//! chroma per iter). Each backend implements the 1→4 nearest-neighbor
//! chroma fan-out in registers and falls back to the scalar reference
//! for the multiple-of-4 tail.

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

/// Converts one row of 4:1:1 YUV to packed RGB.
///
/// Dispatches to the best available backend for the current target.
/// See `scalar::yuv_411_to_rgb_row` for the full semantic
/// specification (range handling, matrix definitions, output layout).
///
/// `use_simd = false` forces the scalar reference path, bypassing any
/// SIMD backend.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_411_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary. The unsafe SIMD
  // kernels below rely on these invariants for pointer arithmetic,
  // so we validate in *release* builds too — not just under
  // `debug_assert!`. Kernels keep their own `debug_assert!`s as
  // internal sanity checks. `rgb_min` uses `checked_mul` because
  // `3 * width` can wrap `usize` on 32-bit targets (wasm32, i686)
  // for extreme widths.
  //
  // FFmpeg `AV_PIX_FMT_YUV411P`: arbitrary widths accepted; chroma
  // row width is `width.div_ceil(4)` samples. SIMD bodies stride
  // 16/32/64 Y pixels (multiples of 4) and the scalar kernel handles
  // the trailing 1..N-1 Y pixels — including a final partial 1..3-
  // pixel chroma group when `width % 4 != 0`.
  let chroma_min = width.div_ceil(4);
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= chroma_min, "u_quarter row too short");
  assert!(v_quarter.len() >= chroma_min, "v_quarter row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present on this
          // CPU. Bounds / parity invariants are the caller's obligation
          // (asserted above); the kernel re-checks them with
          // `debug_assert` in debug builds.
          unsafe {
            arch::neon::yuv_411_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX-512BW is present.
          unsafe {
            arch::x86_avx512::yuv_411_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` verified AVX2 is present.
          unsafe {
            arch::x86_avx2::yuv_411_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: `sse41_available()` verified SSE4.1 is present.
          unsafe {
            arch::x86_sse41::yuv_411_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: `simd128_available()` (compile-time
          // `cfg!(target_feature = "simd128")`) verified that simd128
          // is on. WASM has no runtime detection — the module's SIMD
          // support is fixed at produce-time.
          unsafe {
            arch::wasm_simd128::yuv_411_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend (riscv64, powerpc, …) fall
        // through to the scalar path below.
      }
    }
  }

  scalar::yuv_411_to_rgb_row(y, u_quarter, v_quarter, rgb_out, width, matrix, full_range);
}

/// Converts one row of 4:1:1 YUV to packed **RGBA** (8-bit).
///
/// Same numerical contract as [`yuv_411_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque, for every pixel — sources without an alpha plane
/// produce opaque output).
///
/// `rgba_out.len() >= 4 * width`. `use_simd = false` forces the
/// scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_411_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary — see
  // [`yuv_411_to_rgb_row`] for rationale (FFmpeg div_ceil(4) chroma
  // semantics), including the checked `width × 4` multiplication via
  // [`rgba_row_bytes`].
  let chroma_min = width.div_ceil(4);
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= chroma_min, "u_quarter row too short");
  assert!(v_quarter.len() >= chroma_min, "v_quarter row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified present; bounds / parity asserted above.
          unsafe {
            arch::neon::yuv_411_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified present.
          unsafe {
            arch::x86_avx512::yuv_411_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified present.
          unsafe {
            arch::x86_avx2::yuv_411_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified present.
          unsafe {
            arch::x86_sse41::yuv_411_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::yuv_411_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_411_to_rgba_row(y, u_quarter, v_quarter, rgba_out, width, matrix, full_range);
}

/// Converts one row of 4:1:1 YUV **directly** to planar HSV bytes
/// (OpenCV encoding), without materializing a source-width RGB row.
/// Output is byte-identical to `rgb_to_hsv_row(yuv_411_to_rgb_row(...))`
/// within the selected tier. FFmpeg-compatible arbitrary widths.
///
/// `use_simd = false` forces the scalar reference path. See
/// `scalar::yuv_411_to_hsv_row` for the semantic specification.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_411_to_hsv_row(
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
  assert!(y.len() >= width, "y row too short");
  assert!(
    u_quarter.len() >= width.div_ceil(4),
    "u_quarter row too short"
  );
  assert!(
    v_quarter.len() >= width.div_ceil(4),
    "v_quarter row too short"
  );
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified; bounds asserted above.
          unsafe {
            arch::neon::yuv_411_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_411_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_411_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_411_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_411_to_hsv_row(
              y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_411_to_hsv_row(
    y, u_quarter, v_quarter, h_out, s_out, v_out, width, matrix, full_range,
  );
}
