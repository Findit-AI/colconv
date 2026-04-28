//! YUVA dispatchers — Yuva444p9/p10 + the Yuva420p family
//! (Yuva420p / Yuva420p9 / Yuva420p10 / Yuva420p16) for both 8-bit
//! RGBA and native-depth `u16` RGBA outputs. Extracted from
//! `row::mod` for organization.
//!
//! The Yuva422p family does not have its own row dispatcher: per-row
//! the chroma layout is identical to 4:2:0 (half-width U / V), so
//! `MixedSinker<Yuva422p*>` delegates row-level work to the
//! `yuva420p*_to_rgba*_with_alpha_src_row` dispatchers. The 4:2:0 vs
//! 4:2:2 difference is purely in the vertical walker (chroma row
//! index `r / 2` vs `r`) and is handled in the walker / sinker layer.

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
// Per-row dispatchers for Yuva444p9 and Yuva444p10 (Yuva444p9 added in
// Ship 8b‑3 — its SIMD comes free off the existing
// `yuv_444p_n_to_rgba*_with_alpha_src_row<BITS>` template). Both the
// u8 and native-depth `u16` RGBA dispatchers route through per-arch
// SIMD wrappers, mirroring the `yuv444p9_to_rgba_row` /
// `yuv444p10_to_rgba_row` dispatchers' patterns.

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

// ---- YUVA 4:2:0 RGBA dispatchers --------------------------------------
//
// Per-row dispatchers for the YUVA 4:2:0 source family — Yuva420p
// (8-bit) plus Yuva420p9 / Yuva420p10 / Yuva420p16. The u8 RGBA
// dispatchers route through per-arch
// `yuv_420*_to_rgba*_with_alpha_src_row` SIMD wrappers (Ship 8b-2b),
// mirroring the non-alpha sibling dispatchers' `cfg_select!` blocks.
// The native-depth `u16` RGBA dispatchers below remain scalar pending
// Ship 8b-2c.

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
  matrix: ColorMatrix,
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
      },
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
      },
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
      },
      _ => {}
    }
  }

  scalar::yuv_420_to_rgba_with_alpha_src_row(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **9‑bit** YUVA 4:2:0 to packed **8‑bit**
/// **RGBA**. R / G / B are produced by the same Q15 i32 kernel family
/// that backs [`yuv420p9_to_rgba_row`]; the per-pixel alpha byte is
/// **sourced from `a`** (depth-converted via `a >> 1` to fit `u8`)
/// instead of being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv420p9_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p9_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
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
            arch::neon::yuv_420p_n_to_rgba_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgba_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgba_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgba_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgba_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<9>(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **9‑bit** YUVA 4:2:0 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, 511]`); the
/// per-pixel alpha element is **sourced from `a`** (already at the
/// source's native bit depth) instead of being the opaque maximum
/// `511`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv420p9_to_rgba_u16_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p9_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
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
            arch::neon::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9>(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **10‑bit** YUVA 4:2:0 to packed **8‑bit**
/// **RGBA**. R / G / B are produced by the same Q15 i32 kernel family
/// that backs [`yuv420p10_to_rgba_row`]; the per-pixel alpha byte is
/// **sourced from `a`** (depth-converted via `a >> 2` to fit `u8`)
/// instead of being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv420p10_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p10_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
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
            arch::neon::yuv_420p_n_to_rgba_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgba_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgba_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgba_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgba_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<10>(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **10‑bit** YUVA 4:2:0 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, 1023]`); the
/// per-pixel alpha element is **sourced from `a`** at native depth.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv420p10_to_rgba_u16_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p10_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
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
            arch::neon::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10>(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10>(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **16‑bit** YUVA 4:2:0 to packed **8‑bit**
/// **RGBA**. R / G / B are produced by the same i32 kernel that backs
/// [`yuv420p16_to_rgba_row`]; the per-pixel alpha byte is **sourced
/// from `a`** (depth-converted via `a >> 8` to fit `u8`).
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv420p16_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p16_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
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
            arch::neon::yuv_420p16_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p16_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p16_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p16_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p16_to_rgba_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p16_to_rgba_with_alpha_src_row(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range,
  );
}

/// Converts one row of **16‑bit** YUVA 4:2:0 to **native-depth `u16`**
/// packed **RGBA** — full-range output in `[0, 65535]`; the per-pixel
/// alpha element is **sourced from `a`** at native depth (no shift).
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv420p16_to_rgba_u16_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva420p16_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_elems(width);
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
            arch::neon::yuv_420p16_to_rgba_u16_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p16_to_rgba_u16_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p16_to_rgba_u16_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p16_to_rgba_u16_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p16_to_rgba_u16_with_alpha_src_row(
              y, u_half, v_half, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p16_to_rgba_u16_with_alpha_src_row(
    y, u_half, v_half, a, rgba_out, width, matrix, full_range,
  );
}
