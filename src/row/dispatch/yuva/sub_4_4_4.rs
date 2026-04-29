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
  row::{rgba_row_bytes, rgba_row_elems, scalar},
};

// ---- YUVA 4:4:4 RGBA dispatchers --------------------------------------
//
// Per-row dispatchers for Yuva444p (8-bit, Ship 8b‑6), Yuva444p9 /
// Yuva444p10 / Yuva444p12 / Yuva444p14 (BITS-generic Q15 i32 family),
// and Yuva444p16 (dedicated i64 16-bit family). Both the u8 and
// native-depth `u16` RGBA dispatchers route through per-arch SIMD
// wrappers, mirroring the non-alpha siblings.

/// Converts one row of **8-bit** YUVA 4:4:4 to packed **8-bit**
/// **RGBA**. R / G / B are produced by the same Q15 i32 8-bit kernel
/// that backs [`yuv444p_to_rgba_row`]; the per-pixel alpha byte is
/// **sourced from `a`** (one byte per pixel, full-width) instead of
/// being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444_to_rgba_with_alpha_src_row(y, u, v, a, rgba_out, width, matrix, full_range);
}

/// Converts one row of **9-bit** YUVA 4:4:4 to packed **8-bit**
/// **RGBA**. R / G / B are produced by the same Q15 i32 kernel family
/// that backs [`yuva444p10_to_rgba_row`]; the per-pixel alpha byte is
/// **sourced from `a`** (depth-converted via `a >> 1` to fit `u8`)
/// instead of being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuva444p10_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p9_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgba_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<9>(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **9-bit** YUVA 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, 511]`); the
/// per-pixel alpha element is **sourced from `a`** (already at the
/// source's native bit depth) instead of being the opaque maximum
/// `511`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuva444p10_to_rgba_u16_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p9_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<9>(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **10-bit** YUVA 4:4:4 to packed **8-bit**
/// **RGBA**. R / G / B are produced by the same Q15 i32 kernel family
/// that backs [`yuv444p10_to_rgba_row`]; the per-pixel alpha byte is
/// **sourced from `a`** (depth-converted via `a >> 2` to fit `u8`)
/// instead of being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p10_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p10_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgba_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<10>(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **10-bit** YUVA 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, 1023]`); the
/// per-pixel alpha element is **sourced from `a`** (already at the
/// source's native bit depth) instead of being the opaque maximum
/// `1023`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p10_to_rgba_u16_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p10_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<10>(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **12-bit** YUVA 4:4:4 to packed **8-bit**
/// **RGBA**. R / G / B are produced by the same Q15 i32 kernel family
/// that backs [`yuv444p12_to_rgba_row`]; the per-pixel alpha byte is
/// **sourced from `a`** (depth-converted via `a >> 4` to fit `u8`)
/// instead of being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p12_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p12_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgba_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<12>(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **12-bit** YUVA 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, 4095]`); the
/// per-pixel alpha element is **sourced from `a`** (already at the
/// source's native bit depth) instead of being the opaque maximum
/// `4095`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p12_to_rgba_u16_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p12_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<12>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<12>(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}
/// Converts one row of **14-bit** YUVA 4:4:4 to packed **8-bit**
/// **RGBA**. R / G / B are produced by the same Q15 i32 kernel family
/// that backs [`yuv444p14_to_rgba_row`]; the per-pixel alpha byte is
/// **sourced from `a`** (depth-converted via `a >> 6` to fit `u8`)
/// instead of being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p14_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p14_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgba_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<14>(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **14-bit** YUVA 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, 16383]`); the
/// per-pixel alpha element is **sourced from `a`** (already at the
/// source's native bit depth) instead of being the opaque maximum
/// `16383`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p14_to_rgba_u16_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p14_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<14>(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<14>(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}

// ---- YUVA 4:4:4 16-bit RGBA dispatchers (Ship 8b-5a/b/c) -------------
//
// Yuva444p16 uses dedicated 16-bit kernels rather than the
// BITS-generic Q15 i32 template (which only covers {9,10,12,14}). The
// 8-bit RGBA path uses the i32 chroma pipeline (output-target scaling
// keeps `coeff × u_d` inside i32); the native-depth `u16` RGBA path
// uses the widened i64 chroma kernel family. Ship 8b-5b wired the u8
// path; 8b-5c wires the u16 path. Both dispatchers now run cfg_select!
// per-arch with scalar fallback.

/// Converts one row of **16-bit** YUVA 4:4:4 to packed **8-bit**
/// **RGBA**. R / G / B are produced by the same i32 kernel that backs
/// [`yuv444p16_to_rgba_row`]; the per-pixel alpha byte is **sourced
/// from `a`** (depth-converted via `a >> 8` to fit `u8`).
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p16_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p16_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p16_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p16_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p16_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p16_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p16_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p16_to_rgba_with_alpha_src_row(y, u, v, a, rgba_out, width, matrix, full_range);
}

/// Converts one row of **16-bit** YUVA 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — full-range output in `[0, 65535]`; the per-pixel
/// alpha element is **sourced from `a`** at native depth (no shift).
/// Uses the i64 chroma kernel family.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p16_to_rgba_u16_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p16_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p16_to_rgba_u16_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p16_to_rgba_u16_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p16_to_rgba_u16_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p16_to_rgba_u16_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p16_to_rgba_u16_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p16_to_rgba_u16_with_alpha_src_row(
    y, u, v, a, rgba_out, width, matrix, full_range,
  );
}
