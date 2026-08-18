//! 10-bit planar YUV 4:4:4 dispatchers — 4 variants.

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

use super::{yuv_444p_n_to_rgb_row, yuv_444p_n_to_rgb_u16_row};

/// YUV 4:4:4 planar 10-bit → u8 RGB. Endian-aware variant. `big_endian
/// = true` selects the BE-encoded `u16` plane contract (samples stored
/// MSB-first); `false` is the standard LE contract.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgb_row_endian(
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
    yuv_444p_n_to_rgb_row::<10, true>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  } else {
    yuv_444p_n_to_rgb_row::<10, false>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// LE-only wrapper around [`yuv444p10_to_rgb_row_endian`]; preserves the
/// pre-endian-aware public signature so existing little-endian callers
/// compile unchanged. Equivalent to `yuv444p10_to_rgb_row_endian(...,
/// big_endian = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv444p10_to_rgb_row_endian(y, u, v, rgb_out, width, matrix, full_range, use_simd, false);
}

/// YUV 4:4:4 planar 10-bit → native-depth u16 RGB. Endian-aware
/// variant.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgb_u16_row_endian(
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
    yuv_444p_n_to_rgb_u16_row::<10, true>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  } else {
    yuv_444p_n_to_rgb_u16_row::<10, false>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// LE-only wrapper around [`yuv444p10_to_rgb_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv444p10_to_rgb_u16_row_endian(y, u, v, rgb_out, width, matrix, full_range, use_simd, false);
}

/// Converts one row of **10-bit** YUV 4:4:4 to packed **8-bit**
/// **RGBA** (`R, G, B, 0xFF`). Endian-aware variant: `big_endian =
/// true` selects the BE-encoded `u16` plane contract.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgba_row_endian(
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
            unsafe { arch::neon::yuv_444p_n_to_rgba_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::yuv_444p_n_to_rgba_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::yuv_444p_n_to_rgba_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::yuv_444p_n_to_rgba_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::yuv_444p_n_to_rgba_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::yuv_444p_n_to_rgba_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::yuv_444p_n_to_rgba_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::yuv_444p_n_to_rgba_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::yuv_444p_n_to_rgba_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::yuv_444p_n_to_rgba_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::yuv_444p_n_to_rgba_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range),
    scalar::yuv_444p_n_to_rgba_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`yuv444p10_to_rgba_row_endian`]; preserves
/// the pre-endian-aware public signature so existing little-endian
/// callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv444p10_to_rgba_row_endian(
    y, u, v, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **10-bit** YUV 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, 1023]`); alpha
/// element is `1023`. Endian-aware variant.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgba_u16_row_endian(
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
            unsafe { arch::neon::yuv_444p_n_to_rgba_u16_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::yuv_444p_n_to_rgba_u16_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::yuv_444p_n_to_rgba_u16_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::yuv_444p_n_to_rgba_u16_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::yuv_444p_n_to_rgba_u16_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::yuv_444p_n_to_rgba_u16_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::yuv_444p_n_to_rgba_u16_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::yuv_444p_n_to_rgba_u16_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::yuv_444p_n_to_rgba_u16_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::yuv_444p_n_to_rgba_u16_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::yuv_444p_n_to_rgba_u16_row::<10, false>(y, u, v, rgba_out, width, matrix, full_range),
    scalar::yuv_444p_n_to_rgba_u16_row::<10, true>(y, u, v, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`yuv444p10_to_rgba_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv444p10_to_rgba_u16_row_endian(
    y, u, v, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **10-bit** YUV 4:4:4 **directly** to planar
/// HSV bytes (OpenCV `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`,
/// `S, V ∈ [0, 255]`), without materializing a source-width RGB row.
/// Byte-identical to `rgb_to_hsv_row(yuv444p10_to_rgb_row_endian
/// (...))` within the selected tier — the SIMD path stages a fixed
/// 64-pixel 8-bit RGB chunk internally. Also serves 4:4:0.
///
/// Thin endian-dispatching wrapper over the BITS-generic
/// [`super::yuv_444p_n_to_hsv_row`]. `use_simd = false` forces the
/// scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_hsv_row_endian(
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
    super::yuv_444p_n_to_hsv_row::<10, true>(
      y, u, v, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  } else {
    super::yuv_444p_n_to_hsv_row::<10, false>(
      y, u, v, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  }
}
