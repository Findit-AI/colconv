//! P010 (semi-planar 4:2:0, 10-bit high-packed) dispatchers — 4
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
  KernelMatrix,
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar},
};

/// Converts one row of **P010** (semi‑planar 4:2:0, 10‑bit, high‑bit‑
/// packed — 10 active bits in the high 10 of each `u16`) to packed
/// **8‑bit** RGB.
///
/// This is the HDR hardware‑decode keystone format: VideoToolbox,
/// VA‑API, NVDEC, D3D11VA, and Intel QSV all emit P010 for 10‑bit
/// output. See `scalar::p_n_to_rgb_row::<10, false>` for the full semantic
/// specification. `use_simd = false` forces the scalar reference.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgb_row_endian(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "P010 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

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
              arch::neon::p_n_to_rgb_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::p_n_to_rgb_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
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
              arch::x86_avx512::p_n_to_rgb_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::p_n_to_rgb_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::p_n_to_rgb_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::p_n_to_rgb_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::p_n_to_rgb_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::p_n_to_rgb_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
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
              arch::wasm_simd128::p_n_to_rgb_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::p_n_to_rgb_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
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
    scalar::p_n_to_rgb_row::<10, false>(y, uv_half, rgb_out, width, matrix, full_range),
    scalar::p_n_to_rgb_row::<10, true>(y, uv_half, rgb_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p010_to_rgb_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p010_to_rgb_row_endian(
    y, uv_half, rgb_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **P010** to **native‑depth `u16`** packed RGB
/// (10 active bits in the **low** 10 of each output `u16`, matching
/// `yuv420p10le` convention — **not** the P010 high‑bit packing).
/// Callers feeding this output into a P010 consumer must shift left
/// by 6.
///
/// See `scalar::p_n_to_rgb_u16_row::<10, false>` for the full spec.
/// `use_simd = false` forces the scalar reference.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgb_u16_row_endian(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "P010 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

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
              arch::neon::p_n_to_rgb_u16_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::p_n_to_rgb_u16_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
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
              arch::x86_avx512::p_n_to_rgb_u16_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::p_n_to_rgb_u16_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::p_n_to_rgb_u16_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::p_n_to_rgb_u16_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::p_n_to_rgb_u16_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::p_n_to_rgb_u16_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
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
              arch::wasm_simd128::p_n_to_rgb_u16_row::<10, false>(
                y, uv_half, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::p_n_to_rgb_u16_row::<10, true>(
                y, uv_half, rgb_out, width, matrix, full_range,
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
    scalar::p_n_to_rgb_u16_row::<10, false>(y, uv_half, rgb_out, width, matrix, full_range),
    scalar::p_n_to_rgb_u16_row::<10, true>(y, uv_half, rgb_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p010_to_rgb_u16_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p010_to_rgb_u16_row_endian(
    y, uv_half, rgb_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **P010** (semi-planar 4:2:0, 10-bit,
/// high-bit-packed) to packed **8-bit** **RGBA**. Alpha defaults to
/// `0xFF` (opaque).
///
/// See `scalar::p_n_to_rgba_row::<10, false>` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgba_row_endian(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
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
              arch::neon::p_n_to_rgba_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::p_n_to_rgba_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::p_n_to_rgba_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::p_n_to_rgba_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::p_n_to_rgba_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::p_n_to_rgba_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::p_n_to_rgba_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::p_n_to_rgba_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::p_n_to_rgba_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::p_n_to_rgba_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
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
    scalar::p_n_to_rgba_row::<10, false>(y, uv_half, rgba_out, width, matrix, full_range),
    scalar::p_n_to_rgba_row::<10, true>(y, uv_half, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p010_to_rgba_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgba_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p010_to_rgba_row_endian(
    y, uv_half, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **P010** (semi-planar 4:2:0, 10-bit,
/// high-bit-packed) to **native-depth `u16`** packed **RGBA** — output
/// is low-bit-packed; alpha element is `(1 << 10) - 1`.
///
/// See `scalar::p_n_to_rgba_u16_row::<10, false>` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgba_u16_row_endian(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
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
              arch::neon::p_n_to_rgba_u16_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::p_n_to_rgba_u16_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::p_n_to_rgba_u16_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::p_n_to_rgba_u16_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::p_n_to_rgba_u16_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::p_n_to_rgba_u16_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::p_n_to_rgba_u16_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::p_n_to_rgba_u16_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::p_n_to_rgba_u16_row::<10, false>(
                y, uv_half, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::p_n_to_rgba_u16_row::<10, true>(
                y, uv_half, rgba_out, width, matrix, full_range,
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
    scalar::p_n_to_rgba_u16_row::<10, false>(y, uv_half, rgba_out, width, matrix, full_range),
    scalar::p_n_to_rgba_u16_row::<10, true>(y, uv_half, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p010_to_rgba_u16_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgba_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p010_to_rgba_u16_row_endian(
    y, uv_half, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **P010** (semi-planar 4:2:0, 10-bit,
/// high-bit-packed) **directly** to planar HSV bytes (OpenCV
/// `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`),
/// without materializing a source-width RGB row. Byte-identical to
/// `rgb_to_hsv_row(p010_to_rgb_row_endian(...))` within the selected
/// tier — the SIMD path stages a fixed 64-pixel 8-bit RGB chunk
/// internally. See `scalar::p_n_to_hsv_row::<10, false, false>` for the
/// reference. `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_hsv_row_endian(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "P010 requires even width");
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

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
              arch::neon::p_n_to_hsv_row::<10, false, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::p_n_to_hsv_row::<10, true, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
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
              arch::x86_avx512::p_n_to_hsv_row::<10, false, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::p_n_to_hsv_row::<10, true, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::p_n_to_hsv_row::<10, false, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::p_n_to_hsv_row::<10, true, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::p_n_to_hsv_row::<10, false, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::p_n_to_hsv_row::<10, true, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
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
              arch::wasm_simd128::p_n_to_hsv_row::<10, false, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::p_n_to_hsv_row::<10, true, false>(
                y, uv_half, h_out, s_out, v_out, width, matrix, full_range,
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
    scalar::p_n_to_hsv_row::<10, false, false>(
      y, uv_half, h_out, s_out, v_out, width, matrix, full_range
    ),
    scalar::p_n_to_hsv_row::<10, true, false>(
      y, uv_half, h_out, s_out, v_out, width, matrix, full_range
    )
  );
}

/// Extracts one row of **P010** native luma: the Y plane's high byte
/// (`>> 8` after host-native normalization — the range-reduced 8-bit
/// luma; for P010's `value << 6` packing this equals the de-packed
/// `value >> 2`). Bit-identical to the P010 sink's former inline
/// native-Y loop. A trivial per-element shift over a contiguous Y plane,
/// so there is no SIMD variant and no `use_simd` knob (the
/// auto-vectorizer handles it; only the packed-Y families need a SIMD
/// deinterleave). See `scalar::p_n_to_luma_row::<10, false>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn p010_to_luma_row_endian(y: &[u16], luma_out: &mut [u8], width: usize, big_endian: bool) {
  assert!(y.len() >= width, "y row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  if big_endian {
    scalar::p_n_to_luma_row::<10, true>(y, luma_out, width);
  } else {
    scalar::p_n_to_luma_row::<10, false>(y, luma_out, width);
  }
}
