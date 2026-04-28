//! NV-family dispatchers (NV12 / NV21 / NV24 / NV42, both RGB and
//! RGBA outputs) extracted from `row::mod` for organization.

use crate::row::scalar;
use crate::row::{arch, rgb_row_bytes, rgba_row_bytes};
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
use crate::ColorMatrix;

/// Converts one row of NV12 (semi‑planar 4:2:0) to packed RGB.
///
/// Same numerical contract as [`yuv_420_to_rgb_row`]; the only
/// difference is UV source — NV12 delivers U and V interleaved in a
/// single `width`‑byte row (`U0, V0, U1, V1, …`). See
/// `scalar::nv12_to_rgb_row` for the reference implementation.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary (see
  // [`yuv_420_to_rgb_row`] for rationale, including the checked
  // `width × 3` multiplication).
  assert_eq!(width & 1, 0, "NV12 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present on this
          // CPU. Bounds / parity invariants are the caller's obligation
          // (checked above).
          unsafe {
            arch::neon::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          unsafe {
            arch::x86_avx512::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` verified AVX2 is present.
          unsafe {
            arch::x86_avx2::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: `sse41_available()` verified SSE4.1 is present.
          unsafe {
            arch::x86_sse41::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: `simd128_available()` verified simd128 is on at
          // compile time (WASM has no runtime CPU detection).
          unsafe {
            arch::wasm_simd128::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of NV21 (semi‑planar 4:2:0, VU-ordered) to
/// packed RGB.
///
/// Same numerical contract as [`nv12_to_rgb_row`]; the only
/// difference is chroma byte order — NV21 stores `V0, U0, V1, U1, …`
/// instead of NV12's `U0, V0, U1, V1, …`. See `scalar::nv21_to_rgb_row`
/// for the reference implementation.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary.
  assert_eq!(width & 1, 0, "NV21 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(vu_half.len() >= width, "vu_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          unsafe {
            arch::x86_avx512::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of NV12 (semi‑planar 4:2:0) to packed **RGBA**
/// (8-bit). Same numerical contract as [`nv12_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque, for every pixel — sources without an alpha plane
/// produce opaque output).
///
/// `rgba_out.len() >= 4 * width`. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary — see
  // [`yuv_420_to_rgba_row`] for rationale, including the checked
  // `width × 4` multiplication via [`rgba_row_bytes`].
  assert_eq!(width & 1, 0, "NV12 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::nv12_to_rgba_row(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::nv12_to_rgba_row(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv12_to_rgba_row(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv12_to_rgba_row(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv12_to_rgba_row(y, uv_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::nv12_to_rgba_row(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Converts one row of NV21 (semi‑planar 4:2:0, VU-ordered) to
/// packed **RGBA** (8-bit). Same numerical contract as
/// [`nv21_to_rgb_row`]; alpha defaults to `0xFF` (opaque).
///
/// `rgba_out.len() >= 4 * width`. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "NV21 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(vu_half.len() >= width, "vu_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::nv21_to_rgba_row(y, vu_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::nv21_to_rgba_row(y, vu_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::nv21_to_rgba_row(y, vu_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::nv21_to_rgba_row(y, vu_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::nv21_to_rgba_row(y, vu_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::nv21_to_rgba_row(y, vu_half, rgba_out, width, matrix, full_range);
}

/// Converts one row of NV24 (semi‑planar 4:4:4, UV‑ordered) to packed
/// RGB. Dispatches to the best available SIMD backend for the current
/// target (NEON / SSE4.1 / AVX2 / AVX-512 / wasm simd128), falling
/// back to scalar when no backend is available.
///
/// Same numerical contract as [`yuv_420_to_rgb_row`]; the difference
/// from NV12 is 4:4:4 chroma — one UV pair per Y pixel, no chroma
/// upsampling, and no width parity constraint. See
/// `scalar::nv24_to_rgb_row` for the reference implementation.
///
/// `use_simd = false` forces the scalar reference path, bypassing any
/// SIMD backend. Benchmarks can flip this to compare scalar vs SIMD
/// directly on the same input; production code should pass `true`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  // NV24 chroma carries one UV pair per pixel = `2 * width` bytes.
  // Use `checked_mul` — on 32-bit targets, `2 * width` can overflow
  // `usize` at extreme widths and silently short-circuit the length
  // check before entering unsafe SIMD paths.
  let uv_min = match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) × 2 overflows usize"),
  };
  assert!(y.len() >= width, "y row too short");
  assert!(uv.len() >= uv_min, "uv row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
}

/// Converts one row of NV42 (semi‑planar 4:4:4, VU‑ordered) to packed
/// RGB. Same as [`nv24_to_rgb_row`] but with swapped chroma byte order.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  let vu_min = match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) × 2 overflows usize"),
  };
  assert!(y.len() >= width, "y row too short");
  assert!(vu.len() >= vu_min, "vu row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
}

/// Converts one row of NV24 (semi‑planar 4:4:4, UV-ordered) to packed
/// **RGBA** (8-bit). Same numerical contract as [`nv24_to_rgb_row`];
/// alpha defaults to `0xFF` (opaque).
///
/// `rgba_out.len() >= 4 * width`. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv24_to_rgba_row(
  y: &[u8],
  uv: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  let uv_min = match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) × 2 overflows usize"),
  };
  assert!(y.len() >= width, "y row too short");
  assert!(uv.len() >= uv_min, "uv row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::nv24_to_rgba_row(y, uv, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::nv24_to_rgba_row(y, uv, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv24_to_rgba_row(y, uv, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv24_to_rgba_row(y, uv, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv24_to_rgba_row(y, uv, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::nv24_to_rgba_row(y, uv, rgba_out, width, matrix, full_range);
}

/// Converts one row of NV42 (semi‑planar 4:4:4, VU-ordered) to packed
/// **RGBA** (8-bit). Same as [`nv24_to_rgba_row`] but with swapped
/// chroma byte order.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv42_to_rgba_row(
  y: &[u8],
  vu: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  let vu_min = match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) × 2 overflows usize"),
  };
  assert!(y.len() >= width, "y row too short");
  assert!(vu.len() >= vu_min, "vu row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::nv42_to_rgba_row(y, vu, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::nv42_to_rgba_row(y, vu, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv42_to_rgba_row(y, vu, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv42_to_rgba_row(y, vu, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv42_to_rgba_row(y, vu, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::nv42_to_rgba_row(y, vu, rgba_out, width, matrix, full_range);
}
