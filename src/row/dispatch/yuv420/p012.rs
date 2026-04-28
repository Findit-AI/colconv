//! P012 (semi-planar 4:2:0, 12-bit high-packed) dispatchers — 4
//! variants.

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
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar},
};

/// Converts one row of **P012** (semi‑planar 4:2:0, 12‑bit, high‑bit‑
/// packed — 12 active bits in the high 12 of each `u16`) to packed
/// **8‑bit** RGB.
///
/// P012 is the 12‑bit sibling of P010, emitted by HEVC Main 12 and
/// VP9 Profile 3 hardware decoders. Same shift semantics as P010 but
/// `>> 4` instead of `>> 6` at each `u16` load.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p012_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P012 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P012** to **native‑depth `u16`** packed RGB
/// (12 active bits in the low 12 of each output `u16` — low‑bit‑packed
/// `yuv420p12le` convention, **not** P012's high‑bit packing).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p012_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P012 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::p_n_to_rgb_u16_row::<12>(
              y, uv_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P012** (semi-planar 4:2:0, 12-bit,
/// high-bit-packed) to packed **8-bit** **RGBA**. Alpha defaults to
/// `0xFF` (opaque).
///
/// See `scalar::p_n_to_rgba_row::<12>` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p012_to_rgba_row(
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
            arch::neon::p_n_to_rgba_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::p_n_to_rgba_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_to_rgba_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_to_rgba_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::p_n_to_rgba_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgba_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Converts one row of **P012** (semi-planar 4:2:0, 12-bit,
/// high-bit-packed) to **native-depth `u16`** packed **RGBA** — output
/// is low-bit-packed; alpha element is `(1 << 12) - 1`.
///
/// See `scalar::p_n_to_rgba_u16_row::<12>` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p012_to_rgba_u16_row(
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
            arch::neon::p_n_to_rgba_u16_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::p_n_to_rgba_u16_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_to_rgba_u16_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_to_rgba_u16_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::p_n_to_rgba_u16_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgba_u16_row::<12>(y, uv_half, rgba_out, width, matrix, full_range);
}
