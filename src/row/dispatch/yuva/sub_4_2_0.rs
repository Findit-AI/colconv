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

// ---- YUVA 4:2:0 RGBA dispatchers --------------------------------------
//
// Per-row dispatchers for the YUVA 4:2:0 source family — Yuva420p
// (8-bit) plus Yuva420p9 / Yuva420p10 / Yuva420p12 / Yuva420p16. The u8
// RGBA dispatchers route through per-arch
// `yuv_420*_to_rgba*_with_alpha_src_row` SIMD wrappers (Ship 8b-2b),
// mirroring the non-alpha sibling dispatchers' `cfg_select!` blocks.
//
// The high-bit dispatchers (`yuva420p9/10/12/16`) each expose an
// `_endian` entry point that threads a runtime `big_endian: bool`
// through every backend (scalar / NEON / SSE4.1 / AVX2 / AVX-512BW /
// wasm-simd128) via the kernels' `<BITS, BE>` (or `<BE>` for 16-bit)
// const-generic pair, including the alpha-source u16 load. The
// pre-existing LE-only public function is preserved as a one-line
// wrapper that forwards `big_endian = false`, mirroring the pattern
// established for non-alpha YUV high-bit dispatchers in commit
// `1c2df3d`.

/// Converts one row of 8‑bit YUVA 4:2:0 to packed **8‑bit** **RGBA**.
/// R / G / B are produced by the same Q15 i32 8‑bit kernel that backs
/// [`yuv_420_to_rgba_row`]; the per-pixel alpha byte is **sourced
/// from `a`** (one byte per pixel, full-width — alpha is at luma
/// resolution in 4:2:0, only chroma is subsampled).
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv_420_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_420_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::yuv_420_to_rgba_with_alpha_src_row(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **9‑bit** YUVA 4:2:0 to packed **8‑bit**
/// **RGBA**. Endian-aware variant: `big_endian = true` selects the
/// BE-encoded `u16` plane contract (samples stored MSB-first across
/// Y / U / V **and** the alpha source plane); `false` is the
/// standard LE contract. R / G / B are produced by the same Q15 i32
/// kernel family that backs [`yuv420p9_to_rgba_row_endian`]; the
/// per-pixel alpha byte is **sourced from `a`** (depth-converted via
/// `a >> 1` to fit `u8`) instead of being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv420p9_to_rgba_row_endian`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p9_to_rgba_row_endian(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
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
              arch::neon::yuv_420p_n_to_rgba_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_420p_n_to_rgba_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::yuv_420p_n_to_rgba_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_420p_n_to_rgba_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::yuv_420p_n_to_rgba_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_420p_n_to_rgba_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
    scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<9, false>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<9, true>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva420p9_to_rgba_row_endian`]; preserves
/// the pre-endian-aware public signature so existing little-endian
/// callers compile unchanged. Equivalent to
/// `yuva420p9_to_rgba_row_endian(.., big_endian = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p9_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva420p9_to_rgba_row_endian(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **9‑bit** YUVA 4:2:0 to **native-depth `u16`**
/// packed **RGBA**. Endian-aware variant. Output is low-bit-packed
/// (`[0, 511]`); the per-pixel alpha element is **sourced from `a`**
/// (already at the source's native bit depth).
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv420p9_to_rgba_u16_row_endian`]'s
/// pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p9_to_rgba_u16_row_endian(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
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
              arch::neon::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
    scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, false>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, true>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva420p9_to_rgba_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p9_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva420p9_to_rgba_u16_row_endian(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **10‑bit** YUVA 4:2:0 to packed **8‑bit**
/// **RGBA**. Endian-aware variant: `big_endian = true` selects the
/// BE-encoded `u16` plane contract for Y / U / V **and** the alpha
/// source plane.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p10_to_rgba_row_endian(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
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
              arch::neon::yuv_420p_n_to_rgba_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_420p_n_to_rgba_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::yuv_420p_n_to_rgba_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_420p_n_to_rgba_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::yuv_420p_n_to_rgba_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_420p_n_to_rgba_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
    scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<10, false>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<10, true>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva420p10_to_rgba_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p10_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva420p10_to_rgba_row_endian(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **10‑bit** YUVA 4:2:0 to **native-depth `u16`**
/// packed **RGBA**. Endian-aware variant. Output is low-bit-packed
/// (`[0, 1023]`); the per-pixel alpha element is sourced from `a` at
/// native depth.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p10_to_rgba_u16_row_endian(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
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
              arch::neon::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
    scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, false>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, true>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva420p10_to_rgba_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p10_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva420p10_to_rgba_u16_row_endian(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **12‑bit** YUVA 4:2:0 to packed **8‑bit**
/// **RGBA**. Endian-aware variant.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p12_to_rgba_row_endian(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
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
              arch::neon::yuv_420p_n_to_rgba_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_420p_n_to_rgba_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::yuv_420p_n_to_rgba_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_420p_n_to_rgba_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::yuv_420p_n_to_rgba_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_420p_n_to_rgba_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
    scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<12, false>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<12, true>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva420p12_to_rgba_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p12_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva420p12_to_rgba_row_endian(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **12‑bit** YUVA 4:2:0 to **native-depth `u16`**
/// packed **RGBA**. Endian-aware variant.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p12_to_rgba_u16_row_endian(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
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
              arch::neon::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
    scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, false>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, true>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva420p12_to_rgba_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p12_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva420p12_to_rgba_u16_row_endian(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **16‑bit** YUVA 4:2:0 to packed **8‑bit**
/// **RGBA**. Endian-aware variant. Uses the dedicated 16-bit i32
/// chroma kernel family (i64 widening only on the u16 RGBA path).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p16_to_rgba_row_endian(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
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
              arch::neon::yuv_420p16_to_rgba_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_420p16_to_rgba_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::yuv_420p16_to_rgba_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_420p16_to_rgba_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_420p16_to_rgba_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_420p16_to_rgba_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_420p16_to_rgba_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_420p16_to_rgba_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::yuv_420p16_to_rgba_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_420p16_to_rgba_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
    scalar::yuv_420p16_to_rgba_with_alpha_src_row::<false>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_420p16_to_rgba_with_alpha_src_row::<true>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva420p16_to_rgba_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p16_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva420p16_to_rgba_row_endian(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **16‑bit** YUVA 4:2:0 to **native-depth `u16`**
/// packed **RGBA**. Endian-aware variant. Full-range output in
/// `[0, 65535]`; the per-pixel alpha element is sourced from `a` at
/// native depth (no shift).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p16_to_rgba_u16_row_endian(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(a.len() >= width, "a row too short");
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
              arch::neon::yuv_420p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_420p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::x86_avx512::yuv_420p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_420p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_420p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_420p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_420p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_420p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
              arch::wasm_simd128::yuv_420p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_420p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u_half, v_half, a, rgba_out, width, matrix, full_range,
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
    scalar::yuv_420p16_to_rgba_u16_with_alpha_src_row::<false>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_420p16_to_rgba_u16_with_alpha_src_row::<true>(
      y, u_half, v_half, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva420p16_to_rgba_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p16_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva420p16_to_rgba_u16_row_endian(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}
