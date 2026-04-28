//! P010 (semi-planar 4:2:0, 10-bit high-packed) dispatchers — 4
//! variants.

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
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar},
};

/// Converts one row of **P010** (semi‑planar 4:2:0, 10‑bit, high‑bit‑
/// packed — 10 active bits in the high 10 of each `u16`) to packed
/// **8‑bit** RGB.
///
/// This is the HDR hardware‑decode keystone format: VideoToolbox,
/// VA‑API, NVDEC, D3D11VA, and Intel QSV all emit P010 for 10‑bit
/// output. See `scalar::p_n_to_rgb_row::<10>` for the full semantic
/// specification. `use_simd = false` forces the scalar reference.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P010 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P010** to **native‑depth `u16`** packed RGB
/// (10 active bits in the **low** 10 of each output `u16`, matching
/// `yuv420p10le` convention — **not** the P010 high‑bit packing).
/// Callers feeding this output into a P010 consumer must shift left
/// by 6.
///
/// See `scalar::p_n_to_rgb_u16_row::<10>` for the full spec.
/// `use_simd = false` forces the scalar reference.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P010 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::p_n_to_rgb_u16_row::<10>(
              y, uv_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P010** (semi-planar 4:2:0, 10-bit,
/// high-bit-packed) to packed **8-bit** **RGBA**. Alpha defaults to
/// `0xFF` (opaque).
///
/// See `scalar::p_n_to_rgba_row::<10>` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgba_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::p_n_to_rgba_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::p_n_to_rgba_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_to_rgba_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_to_rgba_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::p_n_to_rgba_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgba_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Converts one row of **P010** (semi-planar 4:2:0, 10-bit,
/// high-bit-packed) to **native-depth `u16`** packed **RGBA** — output
/// is low-bit-packed; alpha element is `(1 << 10) - 1`.
///
/// See `scalar::p_n_to_rgba_u16_row::<10>` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgba_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::p_n_to_rgba_u16_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::p_n_to_rgba_u16_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_to_rgba_u16_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_to_rgba_u16_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::p_n_to_rgba_u16_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgba_u16_row::<10>(y, uv_half, rgba_out, width, matrix, full_range);
}
