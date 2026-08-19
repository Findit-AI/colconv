//! v210 (Tier 4 packed YUV 4:2:2 10-bit) dispatchers (Ship 11a).
//!
//! Six entries: `v210_to_{rgb,rgba}_row` (u8) and the matching
//! `_u16` variants for native-depth output, plus
//! `v210_to_luma_row` / `v210_to_luma_u16_row` for direct luma
//! extraction. Routes through the standard `cfg_select!` per-arch
//! block; `use_simd = false` forces scalar.
//!
//! The per-format SIMD kernels are const-generic on `ALPHA`
//! (`v210_to_rgb_or_rgba_row::<ALPHA, BE>` /
//! `v210_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>`) — the public
//! dispatchers split them into RGB vs. RGBA entries by hard-wiring
//! `ALPHA = false` / `true`.

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
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar, v210_row_bytes},
};

/// Converts one row of v210 to packed RGB (u8). See
/// [`scalar::v210_to_rgb_or_rgba_row`] for byte layout / numerical
/// contract. `use_simd = false` forces scalar. `big_endian = true` selects
/// the big-endian wire encoding (32-bit words stored MSB-first).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgb_row_endian(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          dispatch_be!(
            unsafe {
              arch::neon::v210_to_rgb_or_rgba_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::v210_to_rgb_or_rgba_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::v210_to_rgb_or_rgba_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::v210_to_rgb_or_rgba_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::v210_to_rgb_or_rgba_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::v210_to_rgb_or_rgba_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::v210_to_rgb_or_rgba_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::v210_to_rgb_or_rgba_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::v210_to_rgb_or_rgba_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::v210_to_rgb_or_rgba_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
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
    scalar::v210_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range),
    scalar::v210_to_rgb_or_rgba_row::<false, true>(packed, rgb_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`v210_to_rgb_row_endian`]; preserves the
/// pre-endian-aware public signature so existing little-endian
/// callers compile unchanged. Equivalent to `v210_to_rgb_row_endian(
/// ..., big_endian = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  v210_to_rgb_row_endian(packed, rgb_out, width, matrix, full_range, use_simd, false)
}

/// Converts one row of v210 to packed RGBA (u8) with `α = 0xFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgba_row_endian(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

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
              arch::neon::v210_to_rgb_or_rgba_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::v210_to_rgb_or_rgba_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::v210_to_rgb_or_rgba_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::v210_to_rgb_or_rgba_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::v210_to_rgb_or_rgba_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::v210_to_rgb_or_rgba_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::v210_to_rgb_or_rgba_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::v210_to_rgb_or_rgba_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::v210_to_rgb_or_rgba_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::v210_to_rgb_or_rgba_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
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
    scalar::v210_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range),
    scalar::v210_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`v210_to_rgba_row_endian`]; preserves the
/// pre-endian-aware public signature so existing little-endian
/// callers compile unchanged. Equivalent to `v210_to_rgba_row_endian(
/// ..., big_endian = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  v210_to_rgba_row_endian(packed, rgba_out, width, matrix, full_range, use_simd, false)
}

/// Converts one row of v210 to packed `u16` RGB at native 10-bit
/// depth (low-bit-packed, `[0, 1023]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgb_u16_row_endian(
  packed: &[u8],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_out row too short"
  );

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
              arch::neon::v210_to_rgb_u16_or_rgba_u16_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::v210_to_rgb_u16_or_rgba_u16_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::v210_to_rgb_u16_or_rgba_u16_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::v210_to_rgb_u16_or_rgba_u16_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::v210_to_rgb_u16_or_rgba_u16_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::v210_to_rgb_u16_or_rgba_u16_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::v210_to_rgb_u16_or_rgba_u16_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::v210_to_rgb_u16_or_rgba_u16_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::v210_to_rgb_u16_or_rgba_u16_row::<false, false>(
                packed, rgb_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::v210_to_rgb_u16_or_rgba_u16_row::<false, true>(
                packed, rgb_out, width, matrix, full_range,
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
    scalar::v210_to_rgb_u16_or_rgba_u16_row::<false, false>(
      packed, rgb_out, width, matrix, full_range
    ),
    scalar::v210_to_rgb_u16_or_rgba_u16_row::<false, true>(
      packed, rgb_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`v210_to_rgb_u16_row_endian`]; preserves the
/// pre-endian-aware public signature so existing little-endian
/// callers compile unchanged. Equivalent to `v210_to_rgb_u16_row_endian(
/// ..., big_endian = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgb_u16_row(
  packed: &[u8],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  v210_to_rgb_u16_row_endian(packed, rgb_out, width, matrix, full_range, use_simd, false)
}

/// Converts one row of v210 to packed `u16` RGBA at native 10-bit
/// depth with `α = 1023` (10-bit opaque maximum).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgba_u16_row_endian(
  packed: &[u8],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_out row too short"
  );

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
              arch::neon::v210_to_rgb_u16_or_rgba_u16_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::v210_to_rgb_u16_or_rgba_u16_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::v210_to_rgb_u16_or_rgba_u16_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::v210_to_rgb_u16_or_rgba_u16_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::v210_to_rgb_u16_or_rgba_u16_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::v210_to_rgb_u16_or_rgba_u16_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::v210_to_rgb_u16_or_rgba_u16_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::v210_to_rgb_u16_or_rgba_u16_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::v210_to_rgb_u16_or_rgba_u16_row::<true, false>(
                packed, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::v210_to_rgb_u16_or_rgba_u16_row::<true, true>(
                packed, rgba_out, width, matrix, full_range,
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
    scalar::v210_to_rgb_u16_or_rgba_u16_row::<true, false>(
      packed, rgba_out, width, matrix, full_range
    ),
    scalar::v210_to_rgb_u16_or_rgba_u16_row::<true, true>(
      packed, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`v210_to_rgba_u16_row_endian`]; preserves the
/// pre-endian-aware public signature so existing little-endian
/// callers compile unchanged. Equivalent to `v210_to_rgba_u16_row_endian(
/// ..., big_endian = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgba_u16_row(
  packed: &[u8],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  v210_to_rgba_u16_row_endian(packed, rgba_out, width, matrix, full_range, use_simd, false)
}

/// Extracts one row of 8-bit luma from a packed v210 buffer.
/// Y values are downshifted from 10-bit to 8-bit via `>> 2`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_luma_row_endian(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
  use_simd: bool,
  big_endian: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

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
              arch::neon::v210_to_luma_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::neon::v210_to_luma_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::v210_to_luma_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::x86_avx512::v210_to_luma_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::v210_to_luma_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::x86_avx2::v210_to_luma_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::v210_to_luma_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::x86_sse41::v210_to_luma_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::v210_to_luma_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::wasm_simd128::v210_to_luma_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
      }
      _ => {}
    }
  }

  dispatch_be!(
    scalar::v210_to_luma_row::<false>(packed, luma_out, width),
    scalar::v210_to_luma_row::<true>(packed, luma_out, width)
  );
}

/// LE-only wrapper around [`v210_to_luma_row_endian`]; preserves the
/// pre-endian-aware public signature so existing little-endian
/// callers compile unchanged. Equivalent to `v210_to_luma_row_endian(
/// ..., big_endian = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize, use_simd: bool) {
  v210_to_luma_row_endian(packed, luma_out, width, use_simd, false)
}

/// Extracts one row of native-depth `u16` luma from a packed v210
/// buffer (low-bit-packed: each `u16` carries the 10-bit Y value in
/// its low 10 bits).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_luma_u16_row_endian(
  packed: &[u8],
  luma_out: &mut [u16],
  width: usize,
  use_simd: bool,
  big_endian: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

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
              arch::neon::v210_to_luma_u16_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::neon::v210_to_luma_u16_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::v210_to_luma_u16_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::x86_avx512::v210_to_luma_u16_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::v210_to_luma_u16_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::x86_avx2::v210_to_luma_u16_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::v210_to_luma_u16_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::x86_sse41::v210_to_luma_u16_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::v210_to_luma_u16_row::<false>(packed, luma_out, width);
            },
            unsafe {
              arch::wasm_simd128::v210_to_luma_u16_row::<true>(packed, luma_out, width);
            }
          );
          return;
        }
      }
      _ => {}
    }
  }

  dispatch_be!(
    scalar::v210_to_luma_u16_row::<false>(packed, luma_out, width),
    scalar::v210_to_luma_u16_row::<true>(packed, luma_out, width)
  );
}

/// LE-only wrapper around [`v210_to_luma_u16_row_endian`]; preserves the
/// pre-endian-aware public signature so existing little-endian
/// callers compile unchanged. Equivalent to `v210_to_luma_u16_row_endian(
/// ..., big_endian = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_luma_u16_row(packed: &[u8], luma_out: &mut [u16], width: usize, use_simd: bool) {
  v210_to_luma_u16_row_endian(packed, luma_out, width, use_simd, false)
}

/// Converts one row of v210 (packed YUV 4:2:2, 10-bit) **directly** to
/// planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`,
/// `S, V ∈ [0, 255]`), without materializing a source-width RGB row.
/// Byte-identical to `rgb_to_hsv_row(v210_to_rgb_row_endian(...))` within the
/// selected tier — the SIMD path stages a fixed 8-bit RGB chunk internally.
/// See `scalar::v210_to_hsv_row` for the reference. `use_simd = false`
/// forces the scalar path; `big_endian = true` selects big-endian wire
/// decoding.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn v210_to_hsv_row_endian(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
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
          // SAFETY: NEON verified at runtime.
          dispatch_be!(
            unsafe {
              arch::neon::v210_to_hsv_row::<false>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::v210_to_hsv_row::<true>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::v210_to_hsv_row::<false>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::v210_to_hsv_row::<true>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::v210_to_hsv_row::<false>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::v210_to_hsv_row::<true>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::v210_to_hsv_row::<false>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::v210_to_hsv_row::<true>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::v210_to_hsv_row::<false>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::v210_to_hsv_row::<true>(
                packed, h_out, s_out, v_out, width, matrix, full_range,
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
    scalar::v210_to_hsv_row::<false>(packed, h_out, s_out, v_out, width, matrix, full_range),
    scalar::v210_to_hsv_row::<true>(packed, h_out, s_out, v_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`v210_to_hsv_row_endian`]; preserves the
/// pre-endian-aware public signature so existing little-endian callers
/// compile unchanged. Equivalent to `v210_to_hsv_row_endian(..., big_endian
/// = false)`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn v210_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  v210_to_hsv_row_endian(
    packed, h_out, s_out, v_out, width, matrix, full_range, use_simd, false,
  )
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the public v210 dispatchers. Walker / kernel
  //! correctness lives in the per-arch tests
  //! (`src/row/arch/*/tests/v210.rs`) and the scalar reference's
  //! own inline tests; this block only verifies the dispatcher
  //! correctly reaches its scalar fallback when SIMD is disabled.
  use super::*;

  /// Build a v210 word from 12 logical samples in v210 standard
  /// order: `[Cb0, Y0, Cr0, Y1, Cb1, Y2, Cr1, Y3, Cb2, Y4, Cr2, Y5]`.
  fn pack_v210_word(samples: [u16; 12]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let w0 = (samples[0] as u32 & 0x3FF)
      | ((samples[1] as u32 & 0x3FF) << 10)
      | ((samples[2] as u32 & 0x3FF) << 20);
    let w1 = (samples[3] as u32 & 0x3FF)
      | ((samples[4] as u32 & 0x3FF) << 10)
      | ((samples[5] as u32 & 0x3FF) << 20);
    let w2 = (samples[6] as u32 & 0x3FF)
      | ((samples[7] as u32 & 0x3FF) << 10)
      | ((samples[8] as u32 & 0x3FF) << 20);
    let w3 = (samples[9] as u32 & 0x3FF)
      | ((samples[10] as u32 & 0x3FF) << 10)
      | ((samples[11] as u32 & 0x3FF) << 20);
    out[0..4].copy_from_slice(&w0.to_le_bytes());
    out[4..8].copy_from_slice(&w1.to_le_bytes());
    out[8..12].copy_from_slice(&w2.to_le_bytes());
    out[12..16].copy_from_slice(&w3.to_le_bytes());
    out
  }

  #[test]
  fn v210_dispatchers_route_with_simd_false() {
    // Full-range gray (Y=512, U=V=512 at 10-bit). Every dispatcher
    // should reach its scalar fallback when `use_simd = false`,
    // produce the documented gray output, and not panic.
    let word = pack_v210_word([512; 12]);

    // u8 RGB
    let mut rgb = [0u8; 6 * 3];
    v210_to_rgb_row(&word, &mut rgb, 6, KernelMatrix::Bt709, true, false);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u8 RGBA — alpha = 0xFF
    let mut rgba = [0u8; 6 * 4];
    v210_to_rgba_row(&word, &mut rgba, 6, KernelMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[3], 0xFF);
    }

    // u16 RGB at native 10-bit depth.
    let mut rgb_u16 = [0u16; 6 * 3];
    v210_to_rgb_u16_row(&word, &mut rgb_u16, 6, KernelMatrix::Bt709, true, false);
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(512) <= 2);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u16 RGBA — alpha = 1023.
    let mut rgba_u16 = [0u16; 6 * 4];
    v210_to_rgba_u16_row(&word, &mut rgba_u16, 6, KernelMatrix::Bt709, true, false);
    for px in rgba_u16.chunks(4) {
      assert_eq!(px[3], 1023);
    }

    // u8 luma — Y=512 → 128 after `>> 2`.
    let mut luma = [0u8; 6];
    v210_to_luma_row(&word, &mut luma, 6, false);
    for &y in &luma {
      assert_eq!(y, (512u16 >> 2) as u8);
    }

    // u16 luma — low-packed 10-bit Y.
    let mut luma_u16 = [0u16; 6];
    v210_to_luma_u16_row(&word, &mut luma_u16, 6, false);
    for &y in &luma_u16 {
      assert_eq!(y, 512);
    }
  }
}
